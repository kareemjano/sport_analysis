import os

import numpy as np
import torch
import torch.nn.functional as F

from ..common.masks import pack_masks, unpack_masks


class HoldDetector:
    """Detector part of a tracking pipeline (PyTorch or ONNX): frame -> hold masks + where people are.

    The same image features are decoded twice: once with the hold prompt and once with
    `person_prompt` (only the decoder runs again, not the image encoder). Hold detections that
    lie mostly on a person (the climber's shoes, shorts, ...) are dropped, and the person mask is
    returned so hold colors can be measured without the climber in front of them.

    cache_dir: optional folder for the detections per frame. A later run with the same model and
    detection options finds them there and skips SAM3, so route options (hold matching,
    clustering) can be tuned quickly; delete the folder to detect again.
    """

    def __init__(self, pipeline, cache_dir=None, video=None, person_prompt="person", person_score=0.5,
                 person_overlap=0.5):
        self.pipeline = pipeline
        self.cache_dir = cache_dir
        self.video = os.path.abspath(str(video)) if video else ""  # cached detections are only reused for this video
        self.person_text = pipeline.encode_prompt(person_prompt) if person_prompt else None
        self.person_score, self.person_overlap = person_score, person_overlap
        self.pipeline.reset(1, 1, 1)

    def __call__(self, rgb, work_hw, frame_no=None):
        """Returns hold masks [N, h, w] and scores [N] at work resolution, and the person mask [h, w]."""
        path = None
        if self.cache_dir is not None and frame_no is not None:
            path = os.path.join(self.cache_dir, f"frame_{frame_no:06d}.npz")
            if os.path.exists(path):
                z = np.load(path)
                if tuple(z["mask_hw"]) == tuple(work_hw) and str(z["video"]) == self.video and "person" in z:
                    person = unpack_masks(z["person"], z["mask_hw"])[0]
                    return unpack_masks(z["masks"], z["mask_hw"]), z["scores"], person
        masks, scores, person = self._detect(rgb, work_hw)
        if path is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.savez_compressed(path, masks=pack_masks(masks), mask_hw=np.array(work_hw), scores=scores,
                                person=pack_masks(person[None]), video=self.video)
        return masks, scores, person

    def _resize(self, masks, work_hw):
        if len(masks) == 0:
            return np.zeros((0, *work_hw), bool)
        return (F.interpolate(masks[:, None].float(), size=work_hw, mode="bilinear", align_corners=False)[:, 0] > 0
                ).cpu().numpy()

    @torch.inference_mode()
    def _detect(self, rgb, work_hw):
        feats = self.pipeline._compute_features(0, rgb)
        masks, scores = self.pipeline._detect(feats)
        masks, scores = self._resize(masks, work_hw), scores.cpu().numpy()
        person = np.zeros(work_hw, bool)
        if self.person_text is not None:
            people, _ = self.pipeline._detect(feats, text=self.person_text, score_thresh=self.person_score)
            person = self._resize(people, work_hw).any(axis=0) if len(people) else person
        area = masks.reshape(len(masks), -1).sum(1)
        on_person = (masks & person).reshape(len(masks), -1).sum(1) / np.maximum(area, 1)
        keep = (area > 0) & (on_person < self.person_overlap)  # tiny masks can vanish at work resolution
        return masks[keep], scores[keep], person
