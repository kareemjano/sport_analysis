import numpy as np
import torch
import torch.nn.functional as F


class HoldDetector:
    """Detector part of a tracking pipeline (PyTorch or ONNX): frame -> hold masks (no tracker)."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.pipeline.reset(1, 1, 1)

    @torch.inference_mode()
    def __call__(self, rgb, work_hw):
        """Returns bool masks [N, h, w] at work resolution and scores [N]."""
        masks, scores = self.pipeline._detect(self.pipeline._compute_features(0, rgb))
        if len(masks) == 0:
            return np.zeros((0, *work_hw), bool), np.zeros(0)
        masks = F.interpolate(masks[:, None].float(), size=work_hw, mode="bilinear", align_corners=False)[:, 0] > 0
        keep = masks.flatten(1).any(1)  # tiny masks can vanish at work resolution
        return masks[keep].cpu().numpy(), scores[keep].cpu().numpy()
