"""Arguments shared by the commands and building pipelines from them."""
import json
import os
import re

import torch

from ..paths import DEFAULT_VIDEO, ONNX_DIR, OUTPUT_DIR, SAM3_CHECKPOINT
from ..routes.config import RouteConfig


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


# hold detection threshold of routes / climb: low enough for small and dark holds (track keeps the pipeline's 0.5)
ROUTES_DET_SCORE_THRESH = 0.3


def add_run_args(parser, det_score_thresh=None):
    """det_score_thresh: the command's default hold detection threshold (None = the pipeline's 0.5)."""
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument("--prompt", default="climbing hold")
    parser.add_argument("--output", default=None,
                        help="output video, or a folder for the default name (default: data/bouldering/output/<video name>/"
                             "<name from method, model, backend and run options>.mp4)")
    parser.add_argument("--render-only", action="store_true",
                        help="skip the model; only re-render --output from the data saved by a previous run")
    parser.add_argument("--stride", type=int, default=1, help="process every n-th frame")
    parser.add_argument("--start", type=int, default=0, help="first frame to process")
    parser.add_argument("--max-frames", type=int, default=None, help="max number of processed frames")
    parser.add_argument("--max-det-interval", type=int, default=None)
    parser.add_argument("--det-score-thresh", type=float, default=det_score_thresh,
                        help=f"hold detection score threshold (default: {det_score_thresh or 0.5}; "
                             "EfficientSAM3 students score holds lower, try 0.2)")
    parser.set_defaults(default_det_score_thresh=det_score_thresh)  # output names only list non-defaults
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


def build_pose_estimator(args):
    """ViTPose+ on ONNX Runtime (--backend onnx: <onnx-dir>/vitpose.onnx) or PyTorch via transformers."""
    if getattr(args, "backend", "torch") == "onnx":
        from ..pose.onnx_pose import OnnxPoseEstimator

        return OnnxPoseEstimator(args.onnx_dir, args.providers)
    from ..pose.vitpose import PoseEstimator

    return PoseEstimator()


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


def _is_dir_arg(path):
    return path.endswith(("/", os.sep)) or os.path.isdir(path)


def output_dir(args):
    """Folder the outputs of a run go to: --output's folder if given, else data/bouldering/output/<input name>/."""
    if args.output:
        return args.output if _is_dir_arg(args.output) else os.path.dirname(os.path.abspath(args.output))
    source = getattr(args, "video", None) or getattr(args, "image", None)
    return str(OUTPUT_DIR / os.path.splitext(os.path.basename(str(source)))[0])


def default_output_name(args, method, ext=".mp4"):
    """<output dir>/<method>_<model variant>[_<backend>][_<run options that differ from the defaults>]<ext>

    method says how holds are followed between frames, e.g. "tracker" (SAM3 tracker) or
    "motion" (camera-motion warping), so runs that differ in any of these don't overwrite each other.
    The folder is output_dir(args), so results of other commands for the same --output are found next to it.
    """
    parts = [method, model_tag(args)]
    if hasattr(args, "backend"):
        parts.append(args.backend)
    if getattr(args, "prompt", "climbing hold") != "climbing hold":
        parts.append("p-" + _slug(args.prompt))
    det = getattr(args, "det_score_thresh", None)
    if det is not None and det != getattr(args, "default_det_score_thresh", None):
        parts.append(f"det{det:g}")
    kf = getattr(args, "max_keyframe_interval", None)
    if kf is not None and kf != RouteConfig.max_keyframe_interval:
        parts.append(f"kf{kf}")
    if getattr(args, "start", 0):
        parts.append(f"from{args.start}")
    if getattr(args, "stride", 1) != 1:
        parts.append(f"s{args.stride}")
    if getattr(args, "max_frames", None) is not None:
        parts.append(f"n{args.max_frames}")
    return os.path.join(output_dir(args), "_".join(parts) + ext)


def resolve_output(args, method, ext=".mp4"):
    """Fill in --output from the parameters when it wasn't given or is a folder; returns the path without extension."""
    if not args.output or _is_dir_arg(args.output):
        args.output = default_output_name(args, method, ext)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    return os.path.splitext(args.output)[0]
