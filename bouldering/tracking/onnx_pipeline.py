"""Detect-and-track climbing holds with the ONNX export of SAM3 LiteText.

Same pipeline as bouldering.tracking.pipeline -- same re-detection triggers, track
health, association and outputs (it reuses that code) -- but every model call runs in
ONNX Runtime on the graphs written by bouldering.models.onnx_export. The tracker
memory bank that Sam3TrackerPredictor keeps in PyTorch is kept here per object, in NumPy.
"""

import os
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

import sam3.model_builder
from sam3.model.sam3_tracker_utils import select_closest_cond_frames
from sam3.model.tokenizer_ve import SimpleTokenizer

from ..common.masks import nms_masks
from ..models.onnx_models import OnnxModels
from ..models.preprocessing import image_transform
from .pipeline import HoldTrackingPipeline


@dataclass
class MemoryEntry:
    maskmem: np.ndarray  # [mem_dim, 72, 72] float16 spatial memory
    obj_ptr: np.ndarray  # [hidden_dim] object pointer
    eff_iou_score: float = 1.0  # memory selection score (SAM2Long), non-conditioning frames only


@dataclass
class ObjectMemory:
    cond: Dict[int, MemoryEntry] = field(default_factory=dict)  # frames anchored on a detection mask
    non_cond: Dict[int, MemoryEntry] = field(default_factory=dict)  # tracked frames


class OnnxHoldTrackingPipeline(HoldTrackingPipeline):
    """HoldTrackingPipeline with the model calls replaced by ONNX Runtime."""

    def __init__(self, onnx_dir, cfg, providers=("CPUExecutionProvider",), max_batch=8):
        self.cfg = cfg
        self.max_batch = max_batch
        self.models = OnnxModels(onnx_dir, list(providers))
        mc = self.models.config
        self.mc = mc
        self.image_size = mc["image_size"]
        self.transform = image_transform(self.image_size)
        const = self.models.constants
        mem_dim = mc["mem_dim"]
        self.maskmem_pos = const["maskmem_pos_enc"].reshape(mem_dim, -1).T.copy()  # [HW, mem_dim]
        self.maskmem_tpos = const["maskmem_tpos_enc"]  # [num_maskmem, mem_dim]
        self.tpos_w, self.tpos_b = const["obj_ptr_tpos_proj_weight"], const["obj_ptr_tpos_proj_bias"]

        bpe = os.path.join(os.path.dirname(sam3.model_builder.__file__), "..", "assets", "bpe_simple_vocab_16e6.txt.gz")
        self.tokenizer = SimpleTokenizer(bpe_path=bpe)
        self.text = self.encode_prompt(cfg.prompt)

    def encode_prompt(self, prompt):
        """Text features for _detect(text=...); the pipeline's own prompt is encoded at init."""
        tokens = self.tokenizer([prompt], context_length=self.mc["context_length"]).numpy()
        return self.models.run("text_encoder", tokens=tokens)

    def reset(self, num_frames, video_h, video_w):
        super().reset(num_frames, video_h, video_w)
        self.memories: Dict[int, ObjectMemory] = {}

    # --------------------------------------------------------------- features
    def _compute_features(self, frame_idx, rgb):
        img = self.transform(v2.functional.to_image(rgb)).unsqueeze(0).numpy()
        self.feats = self.models.run("image_encoder", pixel_values=img)
        return self.feats

    # -------------------------------------------------------------- detection
    def _detect(self, feats, text=None, score_thresh=None):
        out = self.models.run("detector_decoder", **feats, **(self.text if text is None else text))
        probs, masks = torch.from_numpy(out["scores"]), torch.from_numpy(out["masks"])
        thresh = self.cfg.det_score_thresh if score_thresh is None else score_thresh
        keep = nms_masks(probs, masks, thresh, self.cfg.det_nms_iou)
        return masks[keep], probs[keep]

    # --------------------------------------------------------- tracker memory
    def _add_objects(self, frame_idx, obj_ids, det_masks):
        """Anchor objects on detection masks: object pointer + binarized mask memory."""
        res, low = self.mc["input_mask_size"], self.mc["low_res_mask_size"]
        masks = (F.interpolate(det_masks[:, None], size=(res, res), mode="bilinear", align_corners=False) > 0).float()
        # mask memory from the area-downsampled input mask (like the tracker's consolidation step)
        low_res = (F.interpolate(masks, size=(low, low), mode="bilinear", antialias=True) - 0.5) * 2048
        for i in range(0, len(obj_ids), self.max_batch):
            b = slice(i, i + self.max_batch)
            n = len(obj_ids[b])
            ptr = self.models.run("tracker_mask_prompt", masks=masks[b].numpy(), **self._trk_feats())["obj_ptr"]
            mem = self.models.run(
                "tracker_memory_encoder", trk_fpn_2=self.feats["trk_fpn_2"], low_res_masks=low_res[b].numpy(),
                object_score_logits=np.full((n, 1), 10.0, np.float32), binarize=np.array(True),
            )["maskmem_features"]
            for oid, m, p in zip(obj_ids[b], mem, ptr):
                self.memories[oid] = ObjectMemory(cond={frame_idx: MemoryEntry(m.astype(np.float16), p)})

    def _remove_objects(self, obj_ids):
        for oid in obj_ids:
            self.memories.pop(oid, None)

    def _prune_memory(self, frame_idx):
        oldest = frame_idx - self.cfg.mem_window
        for mem in self.memories.values():
            for fi in [fi for fi in mem.non_cond if fi < oldest]:
                del mem.non_cond[fi]

    def _trk_feats(self):
        return {k: self.feats[k] for k in ("trk_fpn_0", "trk_fpn_1", "trk_fpn_2")}

    def _frame_filter(self, mem, frame_idx, max_num):
        """Sam3TrackerBase.frame_filter: recent tracked frames with a good memory score."""
        if frame_idx == 0:
            return []
        valid = []
        for i in range(frame_idx - 1, 0, -1):
            entry = mem.non_cond.get(i)
            if entry is None:
                continue
            if entry.eff_iou_score > self.mc["mf_threshold"]:
                valid.insert(0, i)
            if len(valid) >= max_num - 1:
                break
        if frame_idx - 1 not in valid:
            valid.append(frame_idx - 1)
        return valid

    def _memory_layout(self, mem, frame_idx):
        """Which memories an object attends to on frame_idx
        (Sam3TrackerBase._prepare_memory_conditioned_features, forward tracking).
        Returns [(t_pos, entry)] spatial memories and [(rel_pos, entry)] object pointers."""
        mc = self.mc
        num_maskmem = mc["num_maskmem"]
        max_ptrs = min(self.num_frames, mc["max_obj_ptrs_in_encoder"])
        selected, unselected = select_closest_cond_frames(
            frame_idx, mem.cond, mc["max_cond_frames_in_attn"], keep_first_cond_frame=mc["keep_first_cond_frame"]
        )
        lookup = lambda t: mem.non_cond.get(t, unselected.get(t))  # noqa: E731
        spatial = [(0, e) for e in selected.values()]
        ptrs = [(frame_idx - t, e) for t, e in selected.items() if t <= frame_idx]
        if mc["use_memory_selection"]:
            valid = self._frame_filter(mem, frame_idx, max_ptrs)
            for t_pos in range(1, num_maskmem):
                t_rel = num_maskmem - t_pos
                if t_rel <= len(valid) and lookup(valid[-t_rel]) is not None:
                    spatial.append((t_pos, lookup(valid[-t_rel])))
            for t_diff in range(1, max_ptrs):
                if t_diff >= len(valid):
                    break
                if lookup(valid[-t_diff]) is not None:
                    ptrs.append((t_diff, lookup(valid[-t_diff])))
        else:
            for t_pos in range(1, num_maskmem):
                if lookup(frame_idx - (num_maskmem - t_pos)) is not None:
                    spatial.append((t_pos, lookup(frame_idx - (num_maskmem - t_pos))))
            for t_diff in range(1, max_ptrs):
                if frame_idx - t_diff < 0:
                    break
                if lookup(frame_idx - t_diff) is not None:
                    ptrs.append((t_diff, lookup(frame_idx - t_diff)))
        return spatial, ptrs, max_ptrs

    def _ptr_pos(self, rel_pos, max_ptrs):
        """Temporal encoding of object pointers (Sam3TrackerBase._get_tpos_enc)."""
        dim = self.mc["hidden_dim"]
        pos = np.asarray(rel_pos, np.float32) / (max_ptrs - 1)
        dim_t = 10000 ** (2 * (np.arange(dim // 2, dtype=np.float32) // 2) / (dim // 2))
        pe = pos[:, None] / dim_t
        pe = np.concatenate([np.sin(pe), np.cos(pe)], axis=-1)
        return pe @ self.tpos_w.T + self.tpos_b  # [N, mem_dim]

    # --------------------------------------------------------------- tracking
    def _propagate(self, frame_idx):
        """Track all objects one frame forward. Objects whose memories have the same layout
        (same temporal positions) are batched through the tracker graphs together."""
        mem_dim, hidden = self.mc["mem_dim"], self.mc["hidden_dim"]
        split = hidden // mem_dim  # an object pointer becomes `split` memory tokens
        groups = {}
        for oid, mem in self.memories.items():
            spatial, ptrs, max_ptrs = self._memory_layout(mem, frame_idx)
            key = (tuple(t for t, _ in spatial), tuple(p for p, _ in ptrs))
            groups.setdefault(key, []).append((oid, spatial, ptrs, max_ptrs))

        preds = {}
        for (t_pos, rel_pos), members in groups.items():
            max_ptrs = members[0][3]
            memory_pos = np.concatenate(
                [self.maskmem_pos + self.maskmem_tpos[self.mc["num_maskmem"] - t - 1] for t in t_pos]
            )
            ptr_pos = np.repeat(self._ptr_pos(rel_pos, max_ptrs), split, axis=0)
            for i in range(0, len(members), self.max_batch):
                chunk = members[i : i + self.max_batch]
                n = len(chunk)
                memory = np.stack([
                    np.concatenate([e.maskmem.reshape(mem_dim, -1).T for _, e in spatial]) for _, spatial, _, _ in chunk
                ]).astype(np.float32)
                obj_ptrs = np.stack([
                    np.concatenate([e.obj_ptr.reshape(split, mem_dim) for _, e in ptrs]) for _, _, ptrs, _ in chunk
                ]).astype(np.float32)
                pix = self.models.run(
                    "tracker_memory_attention", trk_fpn_2=self.feats["trk_fpn_2"], memory=memory,
                    memory_pos=np.ascontiguousarray(np.broadcast_to(memory_pos, memory.shape), np.float32),
                    obj_ptrs=obj_ptrs,
                    obj_ptr_pos=np.ascontiguousarray(np.broadcast_to(ptr_pos, obj_ptrs.shape), np.float32),
                )["pix_feat_with_mem"]
                out = self.models.run("tracker_mask_decoder", pix_feat_with_mem=pix, **self._trk_feats())
                maskmem = self.models.run(
                    "tracker_memory_encoder", trk_fpn_2=self.feats["trk_fpn_2"], low_res_masks=out["low_res_masks"],
                    object_score_logits=out["object_score_logits"], binarize=np.array(False),
                )["maskmem_features"]
                for j, (oid, _, _, _) in enumerate(chunk):
                    logit, iou = float(out["object_score_logits"][j, 0]), float(out["iou_scores"][j])
                    # Sam3TrackerBase.cal_mem_score (per object here; per tracker state in PyTorch)
                    obj_norm = 2 / (1 + np.exp(-logit)) - 1 if logit > 0 else 0.0
                    self.memories[oid].non_cond[frame_idx] = MemoryEntry(
                        maskmem[j].astype(np.float16), out["obj_ptr"][j], obj_norm * iou
                    )
                    preds[oid] = (torch.from_numpy(out["low_res_masks"][j, 0]), 1 / (1 + np.exp(-logit)), iou)
        return preds
