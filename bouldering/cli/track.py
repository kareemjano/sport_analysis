"""Detect and track every climbing hold through a video (SAM3 detector + tracker).

Outputs <output>.mp4, <output>.json (per-frame boxes and scores) and <output>_masks/ (per-frame masks).
The default <output> is named after the parameters, e.g. tracker_sam3-s0_onnx_s3_n60.

Examples:
    python -m bouldering track --stride 3 --max-frames 60
    python -m bouldering track --backend onnx --stride 3 --max-frames 60

    # EfficientSAM3 (TinyViT backbone, tracker weights from the LiteText checkpoint)
    python -m bouldering track --model efficientsam3 --checkpoint checkpoints/efficientsam3_tinyvit.pt \\
        --backbone-type tinyvit --model-name 11m --prompt hold \\
        --det-score-thresh 0.2 --new-track-thresh 0.25 --stride 3 --max-frames 60

    # re-render the .mp4 from saved masks without running the model
    python -m bouldering track --render-only
"""
import time

from .common import add_backend_args, add_run_args, build_pipeline, resolve_output


def add_arguments(parser):
    add_run_args(parser)
    parser.add_argument("--masks-dir", default=None, help="where per-frame masks are saved (default: <output>_masks/)")
    add_backend_args(parser)


def run(args):
    from ..tracking.runner import render_tracks, run_tracking

    t0 = time.perf_counter()
    stem = resolve_output(args, "tracker")
    masks_dir = args.masks_dir or stem + "_masks"
    if not args.render_only:
        run_tracking(build_pipeline(args), args.video, masks_dir, stem + ".json",
                     start=args.start, stride=args.stride, max_frames=args.max_frames)
    n = render_tracks(args.video, masks_dir, args.output, alpha=args.alpha, show_ids=not args.no_ids)
    print(f"Wrote {args.output} ({n} frames, {time.perf_counter() - t0:.0f}s total)")
