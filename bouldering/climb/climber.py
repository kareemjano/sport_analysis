"""Find and follow the climber: SAM3 "person" detections + ViTPose+ keypoints."""
import numpy as np
import torch
from PIL import Image

from ..common.masks import mask_boxes
from .limbs import keypoint_box


@torch.inference_mode()
def detect_people(pipeline, rgb, text, score_thresh=0.5):
    """COCO (x, y, w, h) person boxes in frame pixels and their scores, from a SAM3 pipeline.

    pipeline: HoldTrackingPipeline or OnnxHoldTrackingPipeline; text: pipeline.encode_prompt("person").
    """
    masks, scores = pipeline._detect(pipeline._compute_features(0, rgb), text=text, score_thresh=score_thresh)
    xyxy = mask_boxes((masks > 0).cpu().numpy(), rgb.shape[0], rgb.shape[1])
    boxes = [(b[0], b[1], b[2] - b[0], b[3] - b[1]) for b in xyxy if b is not None]
    kept = [s for b, s in zip(xyxy, scores.cpu().numpy()) if b is not None]
    return np.array(boxes, np.float32).reshape(-1, 4), np.array(kept, np.float32)


def box_iou(a, b):
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0.0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    return inter / max(a[2] * a[3] + b[2] * b[3] - inter, 1e-6)


class ClimberTracker:
    """Per frame: keypoints of the climber, or None if not found.

    SAM3 person detection (a full backbone pass) only runs when there is no track, when the
    pose looks lost, or every redetect_seconds; otherwise ViTPose runs on the box around the
    previous frame's keypoints.
    """

    def __init__(self, pipeline, pose_estimator, cfg, fps):
        self.pipeline, self.pose, self.cfg = pipeline, pose_estimator, cfg
        self.person_text = pipeline.encode_prompt(cfg.person_prompt)
        self.redetect_frames = cfg.redetect_seconds * fps
        self.box = None
        self.last_detect = -np.inf

    def _pick(self, boxes, wall_box):
        if len(boxes) == 0:
            return None
        if self.box is not None:
            ious = [box_iou(b, self.box) for b in boxes]
            if max(ious) > 0.1:
                return boxes[int(np.argmax(ious))]
        # new climber: the largest person on the wall (overlap with the area covered by holds)
        scores = [b[2] * b[3] * (box_iou(b, wall_box) > 0 if wall_box is not None else 1.0) for b in boxes]
        return boxes[int(np.argmax(scores))] if max(scores) > 0 else None

    def _pose(self, image, box):
        pose = self.pose.estimate(image, [box])[0]
        good = pose["scores"] >= self.cfg.kpt_thresh
        if good.sum() < 6 or pose["scores"][good].mean() < 0.4:
            return None
        return pose

    def __call__(self, frame_no, rgb, wall_box=None):
        """Returns {"box", "keypoints" (17, 2), "scores" (17,), "detected"} or None."""
        image = Image.fromarray(rgb)
        pose, detected = None, False
        if self.box is not None and frame_no - self.last_detect < self.redetect_frames:
            pose = self._pose(image, self.box)
        if pose is None:  # no track, pose lost, or periodic re-detection
            boxes, _ = detect_people(self.pipeline, rgb, self.person_text, self.cfg.person_score)
            self.last_detect, detected = frame_no, True
            box = self._pick(boxes, wall_box)
            self.box = box
            pose = self._pose(image, box) if box is not None else None
        if pose is None:
            self.box = None
            return None
        used = self.box
        self.box = keypoint_box(pose["keypoints"], pose["scores"], self.cfg, rgb.shape[:2])
        return {"box": used, "keypoints": pose["keypoints"], "scores": pose["scores"], "detected": detected}
