"""Arguments shared by the commands and building pipelines from them."""
import json
import os
import re

import torch

from ..paths import DEFAULT_VIDEO, ONNX_DIR, OUTPUT_DIR, SAM3_CHECKPOINT


def add_model_args(parser):
    g = parser.add_argument_group("PyTorch model (--backend torch)")
    g.add_argument("--model", choices=["sam3", "efficientsam3"], default="sam3",
                   help="sam3: SAM3 ViT backbone (LiteText checkpoint); "
                        "efficientsam3: student backbone (--backbone-type/--model-name)")
    g.add_argument("--checkpoint", default=str(SAM3_CHECKPOINT))
    g.add_argument("--text-encoder", default="MobileCLIP-S0")
    g.add_argument("--context-length", type=int, default=16)
    g.add_argument("--backbone-type", choices=["tinyvit", "repvit", "efficientvit"], default="tinyvit",
                   help="EfficientSAM3 student backbone")
    g.add_argument("--model-name", default="11m",
                   help="EfficientSAM3 backbone size: tinyvit 5m/11m/21m, repvit m0.9/m1.1/m2.3, efficientvit b0/b1/b2")
    g.add_argument("--tracker-checkpoint", default=str(SAM3_CHECKPOINT),
                   help="EfficientSAM3: SAM3 video checkpoint providing the tracker weights missing from --checkpoint")


def add_backend_args(parser):
    parser.add_argument("--backend", choices=["torch", "onnx"], default="torch")
    g = parser.add_argument_group("ONNX Runtime (--backend onnx)")
    g.add_argument("--onnx-dir", default=str(ONNX_DIR), help="output of `python -m bouldering export-onnx`")
    g.add_argument("--max-batch", type=int, default=8, help="objects per tracker graph call")
    g.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"],
                   help="ONNX Runtime execution providers, e.g. CUDAExecutionProvider CPUExecutionProvider")
    add_model_args(parser)


def add_run_args(parser):
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument("--prompt", default="climbing hold")
    parser.add_argument("--output", default=None,
                        help="output video (default: data/bouldering/output/<name from method, model, backend and run options>.mp4)")
    parser.add_argument("--render-only", action="store_true",
                        help="skip the model; only re-render --output from the data saved by a previous run")
    parser.add_argument("--stride", type=int, default=1, help="process every n-th frame")
    parser.add_argument("--start", type=int, default=0, help="first frame to process")
    parser.add_argument("--max-frames", type=int, default=None, help="max number of processed frames")
    parser.add_argument("--max-det-interval", type=int, default=None)
    parser.add_argument("--det-score-thresh", type=float, default=None,
                        help="detection score threshold (EfficientSAM3 students score holds lower; try 0.2)")
    parser.add_argument("--new-track-thresh", type=float, default=None,
                        help="min detection score to start a new track (default: 0.6, or --det-score-thresh if given)")
    parser.add_argument("--alpha", type=float, default=0.45, help="mask overlay opacity")
    parser.add_argument("--no-ids", action="store_true", help="don't draw track ids")


def add_dataclass_args(parser, cls, title):
    """One --flag per field of a config dataclass (defaults from the dataclass)."""
    g = parser.add_argument_group(title)
    for f in cls.__dataclass_fields__.values():
        g.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)


def dataclass_from_args(cls, args):
    return cls(**{f: getattr(args, f) for f in cls.__dataclass_fields__})


def make_config(args):
    from ..tracking.config import PipelineConfig

    # commands without the run options (e.g. pose) get the defaults
    cfg = PipelineConfig(prompt=getattr(args, "prompt", PipelineConfig.prompt))
    if getattr(args, "max_det_interval", None) is not None:
        cfg.max_det_interval = args.max_det_interval
    if getattr(args, "det_score_thresh", None) is not None:
        cfg.det_score_thresh = args.det_score_thresh
        cfg.new_track_score_thresh = cfg.det_score_thresh
    if getattr(args, "new_track_thresh", None) is not None:
        cfg.new_track_score_thresh = args.new_track_thresh
    return cfg


def build_pipeline(args):
    """HoldTrackingPipeline (PyTorch) or OnnxHoldTrackingPipeline, depending on --backend."""
    cfg = make_config(args)
    if args.backend == "onnx":
        from ..tracking.onnx_pipeline import OnnxHoldTrackingPipeline

        return OnnxHoldTrackingPipeline(args.onnx_dir, cfg, args.providers, args.max_batch)

    from ..models.torch_model import build_model
    from ..tracking.pipeline import HoldTrackingPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.model, args.checkpoint, args.text_encoder, args.context_length,
                        args.backbone_type, args.model_name, args.tracker_checkpoint, device=device)
    return HoldTrackingPipeline(model, cfg, device=device)


def _slug(text):
    return re.sub(r"[^a-z0-9.]+", "-", str(text).lower()).strip("-")


def model_tag(args):
    """Short model variant name, e.g. sam3-s0 or esam3-tinyvit-11m-s0.

    With --backend onnx the model options are ignored, so the variant is read from the export's config.json.
    """
    model, text_encoder = args.model, args.text_encoder
    backbone_type, model_name = args.backbone_type, args.model_name
    if getattr(args, "backend", "torch") == "onnx":
        with open(os.path.join(args.onnx_dir, "config.json")) as f:
            cfg = json.load(f)
        model, text_encoder = cfg["model"], cfg.get("text_encoder")
        backbone_type, model_name = cfg.get("backbone_type", "unknown"), cfg.get("model_name", "")
    parts = ["sam3"] if model == "sam3" else ["esam3", backbone_type, model_name]
    if text_encoder:
        parts.append(_slug(text_encoder).removeprefix("mobileclip-"))
    return "-".join(_slug(p) for p in parts if p)


def default_output_name(args, method, ext=".mp4"):
    """<method>_<model variant>[_<backend>][_<run options that differ from the defaults>]<ext>

    method says how holds are followed between frames, e.g. "tracker" (SAM3 tracker) or
    "motion" (camera-motion warping), so runs that differ in any of these don't overwrite each other.
    """
    parts = [method, model_tag(args)]
    if hasattr(args, "backend"):
        parts.append(args.backend)
    if getattr(args, "prompt", "climbing hold") != "climbing hold":
        parts.append("p-" + _slug(args.prompt))
    if getattr(args, "start", 0):
        parts.append(f"from{args.start}")
    if getattr(args, "stride", 1) != 1:
        parts.append(f"s{args.stride}")
    if getattr(args, "max_frames", None) is not None:
        parts.append(f"n{args.max_frames}")
    return str(OUTPUT_DIR / ("_".join(parts) + ext))


def resolve_output(args, method, ext=".mp4"):
    """Fill in --output from the parameters when it wasn't given; returns the output path without extension."""
    args.output = args.output or default_output_name(args, method, ext)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    return os.path.splitext(args.output)[0]
