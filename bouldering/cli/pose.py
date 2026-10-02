"""Estimate the climber's pose in an image (SAM3 "person" detection + ViTPose+ from transformers).

Example:
    python -m bouldering pose --backend onnx --image data/bouldering/bouldering.png
"""
import time

from ..paths import DEFAULT_IMAGE
from .common import add_backend_args, build_pipeline, resolve_output


def add_arguments(parser):
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--output", default=None, help="default: data/bouldering/output/pose_<model>_<backend>.png")
    parser.add_argument("--person-prompt", default="person")
    parser.add_argument("--person-score", type=float, default=0.5)
    parser.add_argument("--kpt-thresh", type=float, default=0.3, help="min keypoint score to draw")
    add_backend_args(parser)


def run(args):
    import cv2
    import numpy as np
    from PIL import Image

    from ..climb.climber import detect_people
    from ..pose import PoseEstimator, draw_poses

    resolve_output(args, "pose", ext=".png")
    t0 = time.time()
    pipeline = build_pipeline(args)
    estimator = PoseEstimator()
    print(f"Loaded models in {time.time() - t0:.1f}s on {estimator.device}")

    image = Image.open(args.image).convert("RGB")
    rgb = np.array(image)
    t0 = time.time()
    boxes, scores = detect_people(pipeline, rgb, pipeline.encode_prompt(args.person_prompt), args.person_score)
    print(f"SAM3 found {len(boxes)} person(s) in {time.time() - t0:.1f}s (scores {np.round(scores, 2).tolist()})")
    t0 = time.time()
    poses = estimator.estimate(image, boxes)
    print(f"Pose in {time.time() - t0:.2f}s")
    for i, pose in enumerate(poses):
        print(f"Person {i}:")
        for name, (x, y), score in zip(estimator.keypoint_names, pose["keypoints"], pose["scores"]):
            print(f"  {name:<16} x={x:7.1f} y={y:7.1f} score={score:.2f}")

    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, draw_poses(frame, boxes, poses, args.kpt_thresh))
    print(f"Wrote {args.output}")
