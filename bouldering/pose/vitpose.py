"""Pose estimation (ViTPose+ via Hugging Face transformers) on person boxes.

Person boxes come from SAM3 with a "person" prompt (bouldering.climb.climber.detect_people),
so no separate person detector is needed.
"""
import cv2
import numpy as np
import torch
from transformers import AutoProcessor, VitPoseForPoseEstimation

POSE_MODEL = "usyd-community/vitpose-plus-small"

# COCO-17 skeleton (keypoint index pairs)
SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
            (1, 3), (2, 4), (3, 5), (4, 6)]


class PoseEstimator:
    """ViTPose+ top-down: PIL RGB image + COCO (x, y, w, h) person boxes -> keypoints."""

    def __init__(self, device=None, pose_model=POSE_MODEL):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(pose_model)
        self.model = VitPoseForPoseEstimation.from_pretrained(pose_model).to(self.device)
        self.keypoint_names = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]

    @torch.no_grad()
    def estimate(self, image, boxes):
        """One dict per box with "keypoints" (17, 2) and "scores" (17,) numpy arrays in image pixels."""
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        if len(boxes) == 0:
            return []
        inputs = self.processor(image, boxes=[boxes], return_tensors="pt").to(self.device)
        # ViTPose+ is a mixture-of-experts model; dataset_index 0 selects the COCO head
        dataset_index = torch.zeros(len(boxes), dtype=torch.long, device=self.device)
        outputs = self.model(**inputs, dataset_index=dataset_index)
        poses = self.processor.post_process_pose_estimation(outputs, boxes=[boxes])[0]
        return [{"keypoints": p["keypoints"].cpu().numpy(), "scores": p["scores"].cpu().numpy()} for p in poses]


def draw_poses(frame_bgr, boxes, poses, kpt_threshold=0.3, highlight=None, draw_boxes=True):
    """Skeletons on a BGR frame. highlight: optional {(x, y): BGR color} extra points (e.g. limbs on holds)."""
    canvas = frame_bgr.copy()
    for box, pose in zip(boxes, poses):
        if draw_boxes and box is not None:
            x, y, w, h = np.asarray(box).astype(int)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 200, 255), 2)
        kpts = np.asarray(pose["keypoints"])
        scores = np.asarray(pose["scores"])
        for a, b in SKELETON:
            if scores[a] > kpt_threshold and scores[b] > kpt_threshold:
                cv2.line(canvas, tuple(kpts[a].astype(int)), tuple(kpts[b].astype(int)), (0, 255, 0), 2)
        for (px, py), s in zip(kpts, scores):
            if s > kpt_threshold:
                cv2.circle(canvas, (int(px), int(py)), 4, (0, 0, 255), -1)
    for (px, py), color in (highlight or {}).items():
        cv2.circle(canvas, (int(px), int(py)), 9, color, -1)
        cv2.circle(canvas, (int(px), int(py)), 9, (0, 0, 0), 2)
    return canvas
