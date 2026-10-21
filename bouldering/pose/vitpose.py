"""Pose estimation (ViTPose+ via Hugging Face transformers, PyTorch) on person boxes.

Person boxes come from SAM3 with a "person" prompt (bouldering.climb.climber.detect_people),
so no separate person detector is needed.
"""
import numpy as np
import torch
from transformers import AutoProcessor, VitPoseForPoseEstimation

from ..paths import POSE_MODEL


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
