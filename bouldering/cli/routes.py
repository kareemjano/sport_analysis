"""Detect holds on keyframes, group them into routes and follow them with the camera motion.

Outputs <output>.mp4 (route overlay), <output>_routes.json and <output>_routes/keyframe_*.npz.
The default <output> is named after the parameters, e.g. motion_sam3-s0_onnx.
Load the result with bouldering.routes.load_routes(<output>_routes.json).frame(frame_no).

Examples:
    python -m bouldering routes                     # PyTorch detector
    python -m bouldering routes --backend onnx      # ONNX detector
    python -m bouldering routes --render-only       # re-render from saved data
"""
import time

from ..routes.config import RouteConfig
from .common import (add_backend_args, add_dataclass_args, add_run_args, build_pipeline, dataclass_from_args,
                     resolve_output)


def add_arguments(parser):
    add_run_args(parser)
    add_backend_args(parser)
    add_dataclass_args(parser, RouteConfig, "route pipeline (RouteConfig)")


def run(args):
    from ..routes import HoldDetector, render_routes, run_routes

    cfg = dataclass_from_args(RouteConfig, args)
    stem = resolve_output(args, "motion")
    json_path = stem + "_routes.json"
    if not args.render_only:
        run_routes(HoldDetector(build_pipeline(args)), cfg, args.video, stem + "_routes", json_path,
                   start=args.start, stride=args.stride, max_frames=args.max_frames, prompt=args.prompt)
    t0 = time.perf_counter()
    n = render_routes(json_path, args.output, video_path=args.video, alpha=args.alpha, show_labels=not args.no_ids)
    print(f"Wrote {args.output} ({n} frames, {time.perf_counter() - t0:.0f}s)")
