"""Export the SAM3 LiteText video model (detector + tracker) to ONNX for the onnx backend.

The video pipeline needs more than the single image graph of the export scripts in
sam3/efficientsam3_examples: the detector only runs on some frames and the tracker
runs on every frame, so the model is split into graphs that the pipeline calls
separately. Memory-bank bookkeeping (which past frames and object pointers each
object attends to) stays in Python.

Graphs written to --output-dir:
  image_encoder.onnx             pixel_values [1,3,1008,1008]
                                 -> det_fpn_0/1/2 (detector), trk_fpn_0/1/2 (tracker)
  text_encoder.onnx              tokens [1,ctx] int64 -> language_mask/features/embeds
  detector_decoder.onnx          det_fpn_*, language_* -> scores [Q], masks [Q,288,288]
  tracker_memory_attention.onnx  trk_fpn_2 + memory bank tokens -> pix_feat_with_mem [B,256,72,72]
  tracker_mask_decoder.onnx      pix_feat_with_mem, trk_fpn_0/1
                                 -> low_res_masks [B,1,288,288], iou_scores [B], obj_ptr [B,256],
                                    object_score_logits [B,1]
  tracker_mask_prompt.onnx       masks [B,1,1152,1152], trk_fpn_* -> obj_ptr [B,256]
                                 (object pointer for a track anchored on a detection mask)
  tracker_memory_encoder.onnx    trk_fpn_2, low_res_masks, object_score_logits, binarize
                                 -> maskmem_features [B,64,72,72]
  tracker_constants.npz          positional encodings and small weights used by the memory bank
  config.json                    model/tracker settings

B (objects) and the number of memory tokens are dynamic; the image size is fixed at
1008 (see export_onnx_open_vocab.py for why).

Usage:
    python -m bouldering export-onnx --checkpoint checkpoints/sam3_litetext_mobileclip_s0_ctx16.pt \\
        --text-encoder MobileCLIP-S0 --context-length 16 --output-dir checkpoints/onnx_sam3_litetext_s0
"""

import ctypes
import gc
import json
import os
import time
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2

from sam3.model.data_misc import FindStage
from sam3.model.sam3_tracker_base import NO_OBJ_SCORE

from .onnx_models import GRAPHS
from .preprocessing import image_transform
from .torch_model import build_model

IMGSZ = 1008
DET_INPUTS = ["det_fpn_0", "det_fpn_1", "det_fpn_2", "language_mask", "language_features", "language_embeds"]
MP_INPUTS = ["masks", "trk_fpn_0", "trk_fpn_1", "trk_fpn_2"]
ME_INPUTS = ["trk_fpn_2", "low_res_masks", "object_score_logits", "binarize"]
MA_INPUTS = ["trk_fpn_2", "memory", "memory_pos", "obj_ptrs", "obj_ptr_pos"]
MD_INPUTS = ["pix_feat_with_mem", "trk_fpn_0", "trk_fpn_1"]


# ------------------------------------------------------------------- ONNX-friendly RoPE
def rope_rotate(x, cos, sin):
    """Rotary encoding with real tensors. x [..., N, D]; cos/sin [N, D/2]."""
    x2 = x.float().reshape(*x.shape[:-1], -1, 2)
    xr, xi = x2[..., 0], x2[..., 1]
    return torch.stack([xr * cos - xi * sin, xr * sin + xi * cos], dim=-1).flatten(-2).type_as(x)


def patch_vit_rope(module):
    """Replace the ViT's complex-valued RoPE (unsupported in ONNX) with rope_rotate."""
    from sam3.model import vitdet

    def _apply_rope_real(self, q, k):
        return rope_rotate(q, self.rope_cos, self.rope_sin), rope_rotate(k, self.rope_cos, self.rope_sin)

    # The absolute position embedding only depends on the weights and the (fixed) input
    # size; tracing its tiling produces If nodes, so export it as a precomputed constant.
    get_abs_pos, cache = vitdet.get_abs_pos, {}

    def get_abs_pos_const(abs_pos, has_cls_token, hw, *args, **kwargs):
        hw = tuple(int(v) for v in hw)  # sizes are traced values while exporting
        key = (id(abs_pos), has_cls_token, hw, repr(args), repr(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = get_abs_pos(abs_pos, has_cls_token, hw, *args, **kwargs).detach().clone()
        return cache[key]

    vitdet.get_abs_pos = get_abs_pos_const

    n = 0
    for m in module.modules():
        if isinstance(m, vitdet.Attention) and getattr(m, "use_rope", False) and m.freqs_cis is not None:
            m.register_buffer("rope_cos", m.freqs_cis.real.float().clone())
            m.register_buffer("rope_sin", m.freqs_cis.imag.float().clone())
            del m._buffers["freqs_cis"]
            m.freqs_cis = None
            m._apply_rope = types.MethodType(_apply_rope_real, m)
            n += 1
    return n


def patch_mask_downsampler(tracker):
    """The memory encoder upsamples masks 1008 -> 1152 with antialias=True, which ONNX lacks.
    Antialiasing only changes downsampling, so plain bilinear gives the same result here."""
    ds = tracker.maskmem_backbone.mask_downsampler

    def forward(self, x):
        if self.interpol_size is not None and self.interpol_size != list(x.shape[-2:]):
            assert all(o >= i for o, i in zip(self.interpol_size, x.shape[-2:])), "only upsampling is exact"
            x = F.interpolate(x.float(), size=self.interpol_size, align_corners=False, mode="bilinear")
        return self.encoder(x)

    original = ds.forward
    ds.forward = types.MethodType(forward, ds)
    return original


# ---------------------------------------------------------------------------- wrappers
class ImageEncoder(nn.Module):
    """pixel_values -> detector FPN (3 levels) + tracker FPN (3 levels, conv_s0/s1 applied)."""

    def __init__(self, model):
        super().__init__()
        self.backbone = model.detector.backbone
        self.conv_s0 = model.tracker.sam_mask_decoder.conv_s0
        self.conv_s1 = model.tracker.sam_mask_decoder.conv_s1

    def forward(self, pixel_values):
        out = self.backbone.forward_image(pixel_values)
        det, trk = out["backbone_fpn"], out["sam2_backbone_out"]["backbone_fpn"]
        return det[0], det[1], det[2], self.conv_s0(trk[0]), self.conv_s1(trk[1]), trk[2]


class TextEncoder(nn.Module):
    """tokens -> language_mask [1,ctx], language_features [ctx,1,256], language_embeds [ctx,1,D]."""

    def __init__(self, model):
        super().__init__()
        text = model.detector.backbone.language_backbone
        assert hasattr(text, "projector"), "only LiteText / EfficientSAM3 student text encoders are supported"
        self.encoder, self.projector = text.encoder, text.projector

    def forward(self, tokens):
        embeds = self.encoder.forward_embedding(tokens)
        memory = self.encoder(embeds, return_all_tokens=True, input_is_embeddings=True)
        memory = self.projector(memory)
        return tokens == 0, memory.transpose(0, 1), embeds.transpose(0, 1)  # mask: True = padding


class DetectorDecoder(nn.Module):
    """Detector heads on precomputed image + text features -> presence-weighted scores and mask logits."""

    def __init__(self, model, vision_pos_enc):
        super().__init__()
        self.detector = model.detector
        for i, pos in enumerate(vision_pos_enc):
            self.register_buffer(f"pos_{i}", pos)
        self.find_stage = FindStage(
            img_ids=torch.tensor([0], dtype=torch.long), text_ids=torch.tensor([0], dtype=torch.long),
            input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
            input_points=None, input_points_mask=None,
        )

    def forward(self, det_fpn_0, det_fpn_1, det_fpn_2, language_mask, language_features, language_embeds):
        backbone_out = {
            "backbone_fpn": [det_fpn_0, det_fpn_1, det_fpn_2],
            "vision_pos_enc": [self.pos_0, self.pos_1, self.pos_2],
            "language_mask": language_mask,
            "language_features": language_features,
            "language_embeds": language_embeds,
        }
        out = self.detector.forward_grounding(
            backbone_out=backbone_out, find_input=self.find_stage, find_target=None,
            geometric_prompt=self.detector._get_dummy_prompt(),
        )
        # pred_logits already include the presence score (supervise_joint_box_scores)
        return out["pred_logits"][0, :, 0].sigmoid(), out["pred_masks"][0]


class TrackerMemoryAttention(nn.Module):
    """Re-implementation of the tracker's memory attention (TransformerEncoderCrossAttention
    with RoPEAttention layers) that exports to ONNX:
      - RoPE uses real cos/sin tensors instead of complex numbers,
      - spatial memories and object pointers come in as separate inputs, so RoPE is applied
        to the spatial keys only without slicing by a runtime token count,
      - attention is computed in fixed query chunks to bound peak memory
        (a full [5184 x memory] attention matrix is ~750 MB per object).
    """

    def __init__(self, tracker, curr_pos, chunk):
        super().__init__()
        enc = tracker.transformer.encoder
        assert enc.pos_enc_at_input and enc.batch_first
        for layer in enc.layers:
            assert layer.pre_norm and not layer.cross_attention_first and not layer.pos_enc_at_attn
            assert layer.pos_enc_at_cross_attn_keys and not layer.pos_enc_at_cross_attn_queries
            assert layer.self_attn.num_heads == 1 and layer.cross_attn_image.num_heads == 1
        self.layers, self.norm = enc.layers, enc.norm
        freqs = enc.layers[0].self_attn.freqs_cis  # complex [HW, C/2]
        self.register_buffer("rope_cos", freqs.real.float().clone())
        self.register_buffer("rope_sin", freqs.imag.float().clone())
        self.register_buffer("curr_pos", curr_pos.flatten(2).transpose(1, 2).clone())  # [1, HW, C]
        self.hw = freqs.shape[0]
        self.chunk = chunk

    def _attend(self, q, k, v):
        scale = q.shape[-1] ** -0.5
        kt = k.transpose(1, 2)
        outs = []
        for s in range(0, self.hw, self.chunk):
            attn = torch.softmax(torch.matmul(q[:, s : s + self.chunk], kt) * scale, dim=-1)
            outs.append(torch.matmul(attn, v))
        return torch.cat(outs, dim=1)

    def _rope_frames(self, k):
        """RoPE on spatial memory keys [B, n_frames*HW, C], repeated over memory frames."""
        B, C = k.shape[0], k.shape[2]
        k = k.reshape(B, -1, self.hw, C)
        return rope_rotate(k, self.rope_cos, self.rope_sin).reshape(B, -1, C)

    def forward(self, trk_fpn_2, memory, memory_pos, obj_ptrs, obj_ptr_pos):
        B = memory.shape[0]
        _, C, H, W = trk_fpn_2.shape
        out = trk_fpn_2.flatten(2).transpose(1, 2) + 0.1 * self.curr_pos
        out = out.expand(B, -1, -1)
        mem_k_in, ptr_k_in = memory + memory_pos, obj_ptrs + obj_ptr_pos
        v_in = torch.cat([memory, obj_ptrs], dim=1)
        for layer in self.layers:
            sa = layer.self_attn
            x = layer.norm1(out)
            q = rope_rotate(sa.q_proj(x), self.rope_cos, self.rope_sin)
            k = rope_rotate(sa.k_proj(x), self.rope_cos, self.rope_sin)
            out = out + sa.out_proj(self._attend(q, k, sa.v_proj(x)))

            ca = layer.cross_attn_image
            x = layer.norm2(out)
            q = rope_rotate(ca.q_proj(x), self.rope_cos, self.rope_sin)
            k = torch.cat([self._rope_frames(ca.k_proj(mem_k_in)), ca.k_proj(ptr_k_in)], dim=1)
            out = out + ca.out_proj(self._attend(q, k, ca.v_proj(v_in)))

            x = layer.norm3(out)
            out = out + layer.linear2(layer.activation(layer.linear1(x)))
        out = self.norm(out)
        return out.transpose(1, 2).reshape(B, C, H, W)


class TrackerMaskDecoder(nn.Module):
    """SAM heads on memory-conditioned features (tracking step, multimask output)."""

    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker

    def forward(self, pix_feat_with_mem, trk_fpn_0, trk_fpn_1):
        B = pix_feat_with_mem.shape[0]
        high_res = [trk_fpn_0.expand(B, -1, -1, -1), trk_fpn_1.expand(B, -1, -1, -1)]
        _, _, ious, low_res_masks, _, obj_ptr, object_score_logits = self.tracker._forward_sam_heads(
            backbone_features=pix_feat_with_mem, high_res_features=high_res, multimask_output=True,
        )
        return low_res_masks, ious.max(-1)[0], obj_ptr, object_score_logits


class TrackerMaskPrompt(nn.Module):
    """Object pointer for a mask input (Sam3TrackerBase._use_mask_as_output)."""

    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker

    def forward(self, masks, trk_fpn_0, trk_fpn_1, trk_fpn_2):
        B = masks.shape[0]
        tr = self.tracker
        high_res = [trk_fpn_0.expand(B, -1, -1, -1), trk_fpn_1.expand(B, -1, -1, -1)]
        _, _, _, _, _, obj_ptr, _ = tr._forward_sam_heads(
            backbone_features=trk_fpn_2.expand(B, -1, -1, -1),
            mask_inputs=tr.mask_downsample(masks),
            high_res_features=high_res,
        )
        appearing = (masks.flatten(1).amax(dim=1, keepdim=True) > 0).float()
        return appearing * obj_ptr + (1 - appearing) * tr.no_obj_ptr


class TrackerMemoryEncoder(nn.Module):
    """Encode a predicted (or input) mask into a spatial memory (Sam3TrackerBase._encode_new_memory)."""

    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker

    def forward(self, trk_fpn_2, low_res_masks, object_score_logits, binarize):
        tr = self.tracker
        B = low_res_masks.shape[0]
        high_res = F.interpolate(low_res_masks, size=(tr.image_size, tr.image_size), mode="bilinear", align_corners=False)
        # binarize=True for mask inputs (detections), sigmoid for tracker predictions
        mask_for_mem = torch.where(binarize, (high_res > 0).float(), torch.sigmoid(high_res))
        mask_for_mem = mask_for_mem * tr.sigmoid_scale_for_mem_enc + tr.sigmoid_bias_for_mem_enc
        out = tr.maskmem_backbone(trk_fpn_2.expand(B, -1, -1, -1), mask_for_mem, skip_mask_sigmoid=True)
        feats = out["vision_features"]
        not_appearing = 1 - (object_score_logits > 0).float()
        return feats + not_appearing[..., None, None] * tr.no_obj_embed_spatial[..., None, None]


# ------------------------------------------------------------------------------ export
def free_memory():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)  # hand freed memory back to the OS


def rss_gb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9


def onnx_export(module, args, path, input_names, output_names, dynamic_axes=None, opset=17):
    t0 = time.perf_counter()
    with torch.no_grad():
        torch.onnx.export(
            module, args, path, input_names=input_names, output_names=output_names,
            dynamic_axes=dynamic_axes or {}, opset_version=opset, dynamo=False,
        )
    size = sum(os.path.getsize(os.path.join(os.path.dirname(path), f))
               for f in os.listdir(os.path.dirname(path)) if f.startswith(os.path.basename(path)))
    print(f"[Export] {os.path.basename(path)}  {size / 1e6:.0f} MB  ({time.perf_counter() - t0:.0f}s)", flush=True)


def onnx_export_external(module, args, path, input_names, output_names, opset=17):
    """Export a large module without holding a second copy of its weights in memory.

    The legacy exporter builds the whole ONNX proto (weights included) in memory, which
    runs out of memory for the ViT on small machines. Here the graph is exported without
    weights (they become graph inputs named after the state dict), and the weights are
    then streamed one tensor at a time into `<path>.data` as external initializers.
    """
    import onnx
    import onnx.numpy_helper

    t0 = time.perf_counter()
    # Capture the exporter's name -> tensor map of graph initializers. It also holds tensors
    # the tracer lifted out of the code (named onnx::...), which have no state-dict name.
    tensors = {}
    dedup = torch._C._jit_pass_onnx_deduplicate_initializers

    def capture(graph, params, training):
        params = dedup(graph, params, training)
        tensors.update(params)
        return params

    torch._C._jit_pass_onnx_deduplicate_initializers = capture
    try:
        with torch.no_grad():
            torch.onnx.export(module, args, path, input_names=input_names, output_names=output_names,
                              export_params=False, opset_version=opset, dynamo=False)
    finally:
        torch._C._jit_pass_onnx_deduplicate_initializers = dedup
    model = onnx.load(path)

    def used_names(graph, in_subgraph=False):
        """Names consumed by nodes: (anywhere, inside If/Loop subgraphs only)."""
        anywhere, nested = set(), set()
        for node in graph.node:
            anywhere.update(node.input)
            if in_subgraph:
                nested.update(node.input)
            for attr in node.attribute:
                for sub in [attr.g] if attr.HasField("g") else list(attr.graphs):
                    a, _ = used_names(sub, in_subgraph=True)
                    anywhere |= a
                    nested |= a
        return anywhere, nested

    # ONNX Runtime cannot resolve external data referenced from subgraphs, so those
    # (small) tensors are stored inline in the .onnx file.
    used, used_in_subgraphs = used_names(model.graph)
    data_name = os.path.basename(path) + ".data"
    graph_inputs, offset = [], 0
    with open(os.path.join(os.path.dirname(path), data_name), "wb") as f:
        for inp in model.graph.input:
            if inp.name in input_names:
                graph_inputs.append(inp)
                continue
            if inp.name not in used:
                continue  # leftover inputs the exporter did not prune
            if inp.name not in tensors:
                raise KeyError(f"graph input {inp.name!r} is not a parameter or buffer of the module")
            arr = tensors[inp.name].detach().contiguous().numpy()
            if inp.name in used_in_subgraphs:
                model.graph.initializer.append(onnx.numpy_helper.from_array(arr, inp.name))
                continue
            raw = arr.tobytes()
            f.write(raw)
            init = onnx.TensorProto(name=inp.name, dims=arr.shape,
                                    data_type=onnx.helper.np_dtype_to_tensor_dtype(arr.dtype),
                                    data_location=onnx.TensorProto.EXTERNAL)
            for key, value in (("location", data_name), ("offset", offset), ("length", len(raw))):
                entry = init.external_data.add()
                entry.key, entry.value = key, str(value)
            model.graph.initializer.append(init)
            offset += len(raw)
    del model.graph.input[:]
    model.graph.input.extend(graph_inputs)
    onnx.save(model, path)
    print(f"[Export] {os.path.basename(path)}  {(os.path.getsize(path) + offset) / 1e6:.0f} MB "
          f"(weights in {data_name})  ({time.perf_counter() - t0:.0f}s)", flush=True)


def export(args):
    os.makedirs(args.output_dir, exist_ok=True)
    out = lambda name: os.path.join(args.output_dir, f"{name}.onnx")  # noqa: E731
    only = set(args.only or GRAPHS)

    t0 = time.perf_counter()
    model = build_model(args.model, args.checkpoint, args.text_encoder, args.context_length,
                        args.backbone_type, args.model_name, args.tracker_checkpoint, device="cpu")
    tracker = model.tracker
    for name in ("cond_frame_spatial_embedding", "cond_frame_obj_ptr_embedding"):
        assert getattr(tracker, name, None) is None, f"tracker.{name} is not supported by the export"
    print(f"[Load] {time.perf_counter() - t0:.0f}s; patched RoPE in {patch_vit_rope(model)} ViT attention layers")

    # Reference inputs: first video frame preprocessed like the tracking pipeline
    import cv2

    cap = cv2.VideoCapture(args.video)
    ok, frame = cap.read()
    cap.release()
    assert ok, f"could not read {args.video}"
    pixel_values = image_transform(IMGSZ)(v2.functional.to_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0)
    tokens = model.detector.backbone.language_backbone.tokenizer([args.prompt], context_length=args.context_length)

    refs = {}  # graph -> (inputs, reference outputs) for --verify
    with torch.no_grad():
        image_encoder = ImageEncoder(model).eval()
        feats = image_encoder(pixel_values)
        det_fpn, trk_fpn = feats[:3], feats[3:]
        full = model.detector.backbone.forward_image(pixel_values)
        vision_pos_enc = [p.clone() for p in full["vision_pos_enc"]]
        trk_pos = full["sam2_backbone_out"]["vision_pos_enc"][-1].clone()
        del full
        refs["image_encoder"] = (["pixel_values"], (pixel_values,), feats)

        text_encoder = TextEncoder(model).eval()
        text = text_encoder(tokens)
        ref_text = model.detector.backbone.forward_text([args.prompt], device="cpu")
        for name, got in zip(("language_mask", "language_features", "language_embeds"), text):
            assert torch.equal(got, ref_text[name]) if got.dtype == torch.bool else torch.allclose(got, ref_text[name], atol=1e-5), name
        refs["text_encoder"] = (["tokens"], (tokens,), text)
        if "text_encoder" in only:
            onnx_export(text_encoder, (tokens,), out("text_encoder"), ["tokens"],
                        ["language_mask", "language_features", "language_embeds"], opset=args.opset)

        detector_decoder = DetectorDecoder(model, vision_pos_enc).eval()
        det_in = (*det_fpn, *text)
        det_out = detector_decoder(*det_in)
        refs["detector_decoder"] = (DET_INPUTS, det_in, det_out)
        print(f"[Check] detector: {(det_out[0] > 0.5).sum().item()} detections > 0.5 for {args.prompt!r}")
        if "detector_decoder" in only:
            onnx_export(detector_decoder, det_in, out("detector_decoder"),
                        DET_INPUTS, ["scores", "masks"], opset=args.opset)

        # Tracker graphs, traced with B=2 objects, 2 memory frames and 3 object pointers.
        # Real masks from the detector make the reference outputs meaningful.
        B = 2
        top = det_out[0].argsort(descending=True)[:B]
        det_masks = det_out[1][top][:, None]  # [B,1,288,288]
        res = tracker.input_mask_size
        masks_in = (F.interpolate(det_masks, size=(res, res), mode="bilinear", align_corners=False) > 0).float()

        mask_prompt = TrackerMaskPrompt(tracker).eval()
        mp_in = (masks_in, *trk_fpn)
        refs["tracker_mask_prompt"] = (MP_INPUTS, mp_in, (mask_prompt(*mp_in),))
        if "tracker_mask_prompt" in only:
            onnx_export(mask_prompt, mp_in, out("tracker_mask_prompt"),
                        MP_INPUTS, ["obj_ptr"],
                        {"masks": {0: "objects"}, "obj_ptr": {0: "objects"}}, opset=args.opset)

        memory_encoder = TrackerMemoryEncoder(tracker).eval()
        obj_logits = torch.full((B, 1), 10.0)
        me_in = (trk_fpn[2], det_masks, obj_logits, torch.tensor(True))
        maskmem_aa = memory_encoder(*me_in)
        patch_mask_downsampler(tracker)
        maskmem = memory_encoder(*me_in)
        diff = (maskmem - maskmem_aa).abs().max().item()
        print(f"[Check] memory encoder without antialias vs original: max |diff| {diff:.2e}")
        assert diff < 1e-3, diff
        maskmem_pos_enc = tracker.maskmem_backbone(trk_fpn[2], torch.zeros(1, 1, IMGSZ, IMGSZ), skip_mask_sigmoid=True)["vision_pos_enc"][0]
        refs["tracker_memory_encoder"] = (ME_INPUTS, me_in, (maskmem,))
        if "tracker_memory_encoder" in only:
            onnx_export(memory_encoder, me_in, out("tracker_memory_encoder"),
                        ME_INPUTS, ["maskmem_features"],
                        {"low_res_masks": {0: "objects"}, "object_score_logits": {0: "objects"},
                         "maskmem_features": {0: "objects"}}, opset=args.opset)

        mem_attn = TrackerMemoryAttention(tracker, trk_pos, args.attn_chunk).eval()
        hw, mem_dim = mem_attn.hw, maskmem.shape[1]
        n_frames, n_ptr = 2, 3 * (tracker.hidden_dim // mem_dim)
        memory = maskmem.flatten(2).transpose(1, 2).repeat(1, n_frames, 1)  # [B, n_frames*HW, 64]
        memory_pos = maskmem_pos_enc.flatten(2).transpose(1, 2).repeat(B, n_frames, 1)
        memory_pos = memory_pos + torch.randn(1, 1, mem_dim) * 0.02  # stands in for the temporal encoding
        obj_ptrs = torch.randn(B, n_ptr, mem_dim)
        obj_ptr_pos = torch.randn(B, n_ptr, mem_dim) * 0.1
        ma_in = (trk_fpn[2], memory, memory_pos, obj_ptrs, obj_ptr_pos)
        pix_feat_with_mem = mem_attn(*ma_in)
        # check the re-implementation against the tracker's own memory attention
        src = trk_fpn[2].flatten(2).permute(2, 0, 1).expand(-1, B, -1)
        ref = tracker.transformer.encoder(
            src=[src], src_key_padding_mask=[None], src_pos=[trk_pos.flatten(2).permute(2, 0, 1).expand(-1, B, -1)],
            prompt=torch.cat([memory, obj_ptrs], 1).transpose(0, 1),
            prompt_pos=torch.cat([memory_pos, obj_ptr_pos], 1).transpose(0, 1),
            feat_sizes=[(72, 72)], num_obj_ptr_tokens=n_ptr,
        )["memory"].permute(1, 2, 0).reshape(pix_feat_with_mem.shape)
        diff = (ref - pix_feat_with_mem).abs().max().item()
        print(f"[Check] memory attention re-implementation vs tracker: max |diff| {diff:.2e}")
        assert diff < 1e-3, diff
        refs["tracker_memory_attention"] = (MA_INPUTS, ma_in, (pix_feat_with_mem,))
        if "tracker_memory_attention" in only:
            onnx_export(mem_attn, ma_in, out("tracker_memory_attention"),
                        MA_INPUTS, ["pix_feat_with_mem"],
                        {"memory": {0: "objects", 1: "memory_tokens"}, "memory_pos": {0: "objects", 1: "memory_tokens"},
                         "obj_ptrs": {0: "objects", 1: "ptr_tokens"}, "obj_ptr_pos": {0: "objects", 1: "ptr_tokens"},
                         "pix_feat_with_mem": {0: "objects"}}, opset=args.opset)

        mask_decoder = TrackerMaskDecoder(tracker).eval()
        md_in = (pix_feat_with_mem, trk_fpn[0], trk_fpn[1])
        refs["tracker_mask_decoder"] = (MD_INPUTS, md_in, mask_decoder(*md_in))
        if "tracker_mask_decoder" in only:
            onnx_export(mask_decoder, md_in, out("tracker_mask_decoder"),
                        MD_INPUTS, ["low_res_masks", "iou_scores", "obj_ptr", "object_score_logits"],
                        {"pix_feat_with_mem": {0: "objects"}, "low_res_masks": {0: "objects"},
                         "iou_scores": {0: "objects"}, "obj_ptr": {0: "objects"},
                         "object_score_logits": {0: "objects"}}, opset=args.opset)

        np.savez(
            os.path.join(args.output_dir, "tracker_constants.npz"),
            maskmem_pos_enc=maskmem_pos_enc[0].numpy(),  # [64,72,72]
            maskmem_tpos_enc=tracker.maskmem_tpos_enc[:, 0, 0].numpy(),  # [num_maskmem, 64]
            obj_ptr_tpos_proj_weight=tracker.obj_ptr_tpos_proj.weight.numpy(),  # [64, 256]
            obj_ptr_tpos_proj_bias=tracker.obj_ptr_tpos_proj.bias.numpy(),
        )
        config = {
            "model": args.model, "checkpoint": os.path.basename(str(args.checkpoint)), "text_encoder": args.text_encoder,
            **({"backbone_type": args.backbone_type, "model_name": args.model_name}
               if args.model == "efficientsam3" else {}),
            "context_length": args.context_length, "image_size": tracker.image_size,
            "low_res_mask_size": tracker.low_res_mask_size, "input_mask_size": tracker.input_mask_size,
            "hidden_dim": tracker.hidden_dim, "mem_dim": mem_dim, "num_maskmem": tracker.num_maskmem,
            "max_cond_frames_in_attn": tracker.max_cond_frames_in_attn,
            "keep_first_cond_frame": tracker.keep_first_cond_frame,
            "max_obj_ptrs_in_encoder": tracker.max_obj_ptrs_in_encoder,
            "use_memory_selection": tracker.use_memory_selection, "mf_threshold": tracker.mf_threshold,
            "no_obj_score": NO_OBJ_SCORE, "attn_chunk": args.attn_chunk,
        }
        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        print(f"[Export] tracker_constants.npz, config.json")

        if "image_encoder" in only:
            # The ViT export needs a few GB on top of the model: keep only what the image
            # encoder uses (vision backbone + the tracker's conv_s0/conv_s1) and export it last.
            del text_encoder, detector_decoder, mask_prompt, memory_encoder, mem_attn, mask_decoder
            model.tracker = None
            tracker = None
            det = model.detector
            det.backbone.language_backbone = None
            det.transformer = det.segmentation_head = det.geometry_encoder = det.dot_prod_scoring = None
            det = None
            free_memory()
            print(f"[Memory] {rss_gb():.1f} GB resident before image encoder export", flush=True)
            onnx_export_external(image_encoder, (pixel_values,), out("image_encoder"), ["pixel_values"],
                        ["det_fpn_0", "det_fpn_1", "det_fpn_2", "trk_fpn_0", "trk_fpn_1", "trk_fpn_2"], opset=args.opset)

    if not args.verify:
        return
    # Free the PyTorch model before loading ONNX sessions (the ViT alone is ~2 GB)
    model = tracker = image_encoder = text_encoder = detector_decoder = None
    mask_prompt = memory_encoder = mem_attn = mask_decoder = None
    free_memory()
    import onnxruntime as ort

    print("[Verify] ONNX Runtime vs PyTorch (max |diff| per output)")
    for name in GRAPHS:
        if name not in only:
            continue
        names, inputs, expected = refs.pop(name)
        opts = ort.SessionOptions()
        opts.enable_cpu_mem_arena = False  # the arena grows by GBs per image encoder run
        sess = ort.InferenceSession(out(name), opts, providers=["CPUExecutionProvider"])
        used = {i.name for i in sess.get_inputs()}  # the exporter drops unused inputs
        feed = {n: x.numpy() for n, x in zip(names, inputs) if n in used}
        got = sess.run(None, feed)
        diffs = []
        for o, e, g in zip(sess.get_outputs(), expected, got):
            e = e.numpy()
            d = float(np.abs(e.astype(np.float64) - g.astype(np.float64)).max()) if e.size else 0.0
            diffs.append(f"{o.name} {d:.2e}")
        print(f"  {name}: " + ", ".join(diffs), flush=True)
        del sess, got
        free_memory()
