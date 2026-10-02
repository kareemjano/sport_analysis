"""Judge climbing attempts on the detected routes: start, success / fail, distance, time and holds used.

Uses the `routes` output for the same video and options (runs `routes` first if it is missing),
finds the climber with SAM3 ("person" prompt) and ViTPose+, and writes <output>.mp4 and
<output>_climb.json (per-frame pose / contacts / state and a summary per attempt).

Examples:
    python -m bouldering climb --backend onnx --stride 2
    python -m bouldering climb --backend onnx --stride 2 --top-hands 1 --climber-height 1.80
    python -m bouldering climb --backend onnx --stride 2 --rejudge --top-seconds 0.5   # new thresholds, no models
    python -m bouldering climb --backend onnx --stride 2 --render-only
"""
import os
import time

from ..climb.config import ClimbConfig
from ..routes.config import RouteConfig
from .common import (add_backend_args, add_dataclass_args, add_run_args, build_pipeline, dataclass_from_args,
                     default_output_name, resolve_output)


def add_arguments(parser):
    add_run_args(parser)
    parser.add_argument("--rejudge", action="store_true",
                        help="re-judge the poses saved by a previous run with the current ClimbConfig (no models), "
                             "then render")
    parser.add_argument("--routes-json", default=None,
                        help="routes output to use (default: the `routes` output for the same options)")
    add_backend_args(parser)
    add_dataclass_args(parser, ClimbConfig, "attempt judging (ClimbConfig)")
    add_dataclass_args(parser, RouteConfig, "route pipeline (RouteConfig), if routes have to be computed")


def _print_summary(attempts):
    if not attempts:
        print("No attempt detected")
    for i, a in enumerate(attempts, 1):
        dist = (f"vertical {a['vertical_m']:.2f} m, horizontal {a['horizontal_m']:.2f} m, net height {a['net_height_m']:.2f} m"
                if "vertical_m" in a else f"vertical {a['vertical_px']:.0f} px, horizontal {a['horizontal_px']:.0f} px")
        print(f"Attempt {i}: route R{a['route']} {a['result'].upper()} ({a['reason']}), frames "
              f"{a['start_frame']}-{a['end_frame']}, {a['time_s']:.1f} s, {dist}, "
              f"holds {a['holds_used']}/{a['holds_total']} ({a['holds_used_pct']:.0f}%)")


def run(args):
    from ..climb import ClimberTracker, render_climb, run_climb
    from ..pose import PoseEstimator
    from ..routes import HoldDetector, RouteData, run_routes

    t_start = time.perf_counter()
    routes_json = args.routes_json or os.path.splitext(default_output_name(args, "motion"))[0] + "_routes.json"
    stem = resolve_output(args, "climb")
    climb_json = stem + "_climb.json"
    if args.rejudge:
        from ..climb import rejudge_climb

        attempts = rejudge_climb(climb_json, RouteData(routes_json), dataclass_from_args(ClimbConfig, args))
    elif not args.render_only:
        cfg = dataclass_from_args(ClimbConfig, args)
        pipeline = build_pipeline(args)  # one SAM3 model for holds (if needed) and the climber
        if not os.path.exists(routes_json):
            print(f"{routes_json} not found; running the route pipeline first")
            run_routes(HoldDetector(pipeline), dataclass_from_args(RouteConfig, args), args.video,
                       os.path.splitext(routes_json)[0], routes_json,
                       start=args.start, stride=args.stride, max_frames=args.max_frames, prompt=args.prompt)
        routes = RouteData(routes_json)
        fps = _video_fps(args.video)
        tracker = ClimberTracker(pipeline, PoseEstimator(), cfg, fps)
        attempts = run_climb(tracker, routes, args.video, cfg, climb_json, start=args.start, max_frames=args.max_frames)
    else:
        from ..climb import load_climb

        attempts = load_climb(climb_json)["attempts"]
    t0 = time.perf_counter()
    n = render_climb(climb_json, routes_json, args.output, video_path=args.video, alpha=args.alpha)
    print(f"Wrote {args.output} ({n} frames, {time.perf_counter() - t0:.0f}s render, "
          f"{time.perf_counter() - t_start:.0f}s total)")
    _print_summary(attempts)


def _video_fps(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps
