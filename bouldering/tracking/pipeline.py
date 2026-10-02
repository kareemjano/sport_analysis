"""Detect-and-track climbing holds in a video with SAM3 LiteText (PyTorch).

The detector (text prompt -> masks) only runs when it is needed. Every frame
runs the vision backbone once; its tracker branch drives the SAM3 tracker, and
the same features are reused by the detector on frames where re-detection is
triggered.

Re-detection is triggered when:
  - there are no tracks yet,
  - enough tracks became unhealthy since the last detection (tracker lost the
    object, predicted mask IoU dropped, or mask area drifted from the area at
    the last anchor),
  - the scene changed a lot since the last detection (camera pan / new holds),
  - or a maximum interval without detection has passed.

On a detection frame, detections are matched to tracks by mask IoU:
  - matched unhealthy / drifted tracks are re-anchored on the detection mask,
  - unmatched confident detections start new tracks,
  - tracks the detector keeps disagreeing with are dropped.
"""

import math
import time
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.transforms import v2

from sam3.model.data_misc import FindStage

from ..common.masks import mask_iou, nms_masks
from ..models.preprocessing import image_transform
from .config import PipelineConfig, Track


class HoldTrackingPipeline:
    def __init__(self, model, cfg: PipelineConfig, device="cpu"):
        self.model = model
        self.detector = model.detector
        self.tracker = model.tracker
        self.cfg = cfg
        self.device = torch.device(device)
        self.image_size = self.tracker.image_size
        self.transform = image_transform(self.image_size)
        self.find_stage = FindStage(
            img_ids=torch.tensor([0], device=self.device, dtype=torch.long),
            text_ids=torch.tensor([0], device=self.device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        self.text_out = self.encode_prompt(cfg.prompt)

    @torch.inference_mode()
    def encode_prompt(self, prompt):
        """Text features for _detect(text=...); the pipeline's own prompt is encoded at init."""
        return self.detector.backbone.forward_text([prompt], device=self.device)

    # ------------------------------------------------------------------ state
    def reset(self, num_frames, video_h, video_w):
        self.num_frames = num_frames
        self.video_h, self.video_w = video_h, video_w
        self.feature_cache: Dict = {}
        self.tracker_states: List[dict] = []
        self.tracks: Dict[int, Track] = {}
        self.next_obj_id = 1
        self.last_det_frame = -(10**9)
        self.last_det_thumb = None

    # --------------------------------------------------------------- features
    def _compute_features(self, frame_idx, rgb):
        img = self.transform(v2.functional.to_image(rgb).to(self.device)).unsqueeze(0)
        feats = self.detector.backbone.forward_image(img)
        sam2 = feats["sam2_backbone_out"]
        dec = self.tracker.sam_mask_decoder
        fpn = [dec.conv_s0(sam2["backbone_fpn"][0]), dec.conv_s1(sam2["backbone_fpn"][1]), sam2["backbone_fpn"][2]]
        tracker_out = {"vision_features": fpn[-1], "vision_pos_enc": sam2["vision_pos_enc"], "backbone_fpn": fpn}
        # only the current frame is needed by the tracker
        self.feature_cache.clear()
        self.feature_cache[frame_idx] = (img[0], {"tracker_backbone_out": tracker_out})
        return feats

    # -------------------------------------------------------------- detection
    def _detect(self, feats, text=None, score_thresh=None):
        """Masks (288x288 logits) and scores of the prompt's detections after NMS.

        text: encode_prompt() output for another prompt (default: the pipeline's prompt);
        score_thresh: default cfg.det_score_thresh.
        """
        out = self.detector.forward_grounding(
            backbone_out={**feats, **(self.text_out if text is None else text)},
            find_input=self.find_stage,
            find_target=None,
            geometric_prompt=self.detector._get_dummy_prompt(),
        )
        probs = out["pred_logits"][0, :, 0].sigmoid()  # presence already folded in
        masks = out["pred_masks"][0]  # (Q, 288, 288) logits
        thresh = self.cfg.det_score_thresh if score_thresh is None else score_thresh
        keep = nms_masks(probs, masks, thresh, self.cfg.det_nms_iou)
        return masks[keep], probs[keep]

    # --------------------------------------------------------------- tracking
    def _propagate(self, frame_idx):
        """Run the tracker one frame forward on all states."""
        obj_ids, masks, obj_scores, iou_scores = [], [], [], []
        for st in self.tracker_states:
            for _, ids, low_res, _, scores in self.tracker.propagate_in_video(
                st, start_frame_idx=frame_idx, max_frame_num_to_track=0,
                reverse=False, tqdm_disable=True, run_mem_encoder=True,
            ):
                obj_ids.extend(ids)
                masks.append(low_res[:, 0])
                obj_scores.append(scores[:, 0].sigmoid())
                out = st["output_dict"]["non_cond_frame_outputs"].get(frame_idx)
                if out is not None and "iou_score" in out:
                    iou_scores.append(out["iou_score"].reshape(-1).float())
                else:  # conditioning frame (mask input)
                    iou_scores.append(torch.ones(len(ids)))
        if not obj_ids:
            return {}
        masks, obj_scores, iou_scores = torch.cat(masks), torch.cat(obj_scores), torch.cat(iou_scores)
        return {oid: (masks[i], obj_scores[i].item(), iou_scores[i].item()) for i, oid in enumerate(obj_ids)}

    def _add_objects(self, frame_idx, obj_ids, det_masks):
        """Start new tracker states conditioned on detection masks at frame_idx."""
        res = self.tracker.input_mask_size
        masks = F.interpolate(det_masks[:, None], size=(res, res), mode="bilinear", align_corners=False)[:, 0] > 0
        n = self.cfg.max_objects_per_state
        for i in range(0, len(obj_ids), n):
            st = self.tracker.init_state(
                cached_features=self.feature_cache, video_height=self.video_h,
                video_width=self.video_w, num_frames=self.num_frames,
            )
            for oid, m in zip(obj_ids[i : i + n], masks[i : i + n]):
                self.tracker.add_new_mask(st, frame_idx, oid, m, add_mask_to_memory=True)
            self.tracker.propagate_in_video_preflight(st, run_mem_encoder=True)
            self.tracker_states.append(st)

    def _remove_objects(self, obj_ids):
        for oid in obj_ids:
            for st in self.tracker_states:
                self.tracker.remove_object(st, oid, strict=False, need_output=False)
        self.tracker_states = [st for st in self.tracker_states if len(st["obj_ids"]) > 0]

    def _drop_tracks(self, obj_ids, preds):
        self._remove_objects(obj_ids)
        for oid in obj_ids:
            self.tracks.pop(oid, None)
            preds.pop(oid, None)

    def _prune_memory(self, frame_idx):
        oldest = frame_idx - self.cfg.mem_window
        for st in self.tracker_states:
            dicts = [st["output_dict"]] + list(st["output_dict_per_obj"].values())
            for d in dicts:
                for fi in [fi for fi in d["non_cond_frame_outputs"] if fi < oldest]:
                    del d["non_cond_frame_outputs"][fi]

    # ----------------------------------------------------------------- health
    def _update_health(self, frame_idx, preds):
        cfg = self.cfg
        for oid, (mask, obj_score, iou_score) in preds.items():
            trk = self.tracks[oid]
            trk.obj_score, trk.iou_score = obj_score, iou_score
            binary = mask > 0
            trk.area = binary.float().sum().item()
            lost = obj_score < cfg.min_obj_score or trk.area == 0
            ratio = trk.area / max(trk.anchor_area, 1.0)
            diverging = not lost and (
                iou_score < cfg.min_iou_score
                or ratio > cfg.max_area_ratio
                or ratio < 1.0 / cfg.max_area_ratio
            )
            trk.lost_frames = trk.lost_frames + 1 if lost else 0
            if lost or diverging:
                if trk.bad_streak == 0:
                    trk.bad_since = frame_idx
                trk.bad_streak += 1
            else:
                trk.bad_streak = 0
                trk.last_good_mask = mask.detach().clone()

    def _thumb(self, rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    def _detection_reason(self, frame_idx, thumb):
        cfg = self.cfg
        since = frame_idx - self.last_det_frame
        if not self.tracks:
            return "no-tracks" if since >= cfg.min_det_interval or frame_idx == 0 else None
        if since >= cfg.max_det_interval:
            return "periodic"
        if since < cfg.min_det_interval:
            return None
        # only count tracks that went bad after the last detection (and passed the patience)
        newly_bad = [
            t for t in self.tracks.values()
            if t.bad_streak >= cfg.bad_patience and t.bad_since > self.last_det_frame
        ]
        needed = max(1, math.ceil(cfg.unhealthy_fraction * len(self.tracks)))
        if len(newly_bad) >= needed:
            return f"unhealthy {len(newly_bad)}/{len(self.tracks)}"
        if self.last_det_thumb is not None:
            change = float(np.abs(thumb - self.last_det_thumb).mean())
            if change >= cfg.scene_change_thresh:
                return f"scene-change {change:.2f}"
        return None

    # ----------------------------------------------------------- association
    def _associate(self, preds, det_masks, det_scores):
        cfg = self.cfg
        trk_ids = list(self.tracks.keys())
        # lost tracks are matched on their last good mask (holds are static, occluded by the climber)
        trk_masks = [
            self.tracks[oid].last_good_mask if self.tracks[oid].is_lost else preds[oid][0]
            for oid in trk_ids
        ]

        n_det, n_trk = len(det_masks), len(trk_ids)
        ious = np.zeros((n_det, n_trk), dtype=np.float32)
        if n_det and n_trk:
            ious = mask_iou(det_masks > 0, torch.stack(trk_masks) > 0).cpu().numpy()

        matched_det, matched_trk = set(), set()
        reanchor = []  # (obj_id, det_idx)
        if n_det and n_trk:
            rows, cols = linear_sum_assignment(-ious)
            for d, t in zip(rows, cols):
                if ious[d, t] < cfg.assoc_iou_thresh:
                    continue
                matched_det.add(d)
                matched_trk.add(t)
                trk = self.tracks[trk_ids[t]]
                trk.missed_dets = 0
                if trk.is_unhealthy or ious[d, t] < cfg.reanchor_iou_thresh:
                    reanchor.append((trk.obj_id, d))

        removed = []
        for t, oid in enumerate(trk_ids):
            if t in matched_trk:
                continue
            trk = self.tracks[oid]
            if trk.is_lost:
                if trk.lost_frames > cfg.max_lost_frames:
                    removed.append(oid)
            else:
                trk.missed_dets += 1
                if trk.missed_dets >= cfg.max_missed_dets or trk.is_unhealthy:
                    removed.append(oid)

        new_dets = [
            d for d in range(n_det)
            if d not in matched_det
            and det_scores[d] >= cfg.new_track_score_thresh
            and (n_trk == 0 or ious[d].max() < cfg.assoc_iou_thresh)
        ]
        return reanchor, removed, new_dets

    # ------------------------------------------------------------------- step
    @torch.inference_mode()
    def step(self, frame_idx, rgb):
        cfg = self.cfg
        t0 = time.time()
        feats = self._compute_features(frame_idx, rgb)
        t_backbone = time.time() - t0

        t1 = time.time()
        preds = self._propagate(frame_idx)
        t_track = time.time() - t1
        self._update_health(frame_idx, preds)
        # tracks lost for too long are dropped even without a detection
        removed = {oid for oid, t in self.tracks.items() if t.lost_frames > cfg.max_lost_frames}

        thumb = self._thumb(rgb)
        reason = self._detection_reason(frame_idx, thumb)
        event = None
        if reason is None:
            self._drop_tracks(removed, preds)
        else:
            t1 = time.time()
            det_masks, det_scores = self._detect(feats)
            t_detect = time.time() - t1
            det_scores_np = det_scores.cpu().numpy()
            reanchor, det_removed, new_dets = self._associate(preds, det_masks, det_scores_np)
            # a long-lost track that the detector finds again is revived, not removed
            removed = (removed | set(det_removed)) - {oid for oid, _ in reanchor}
            self._drop_tracks(removed, preds)
            # re-anchored tracks keep their id but restart from the detection mask
            self._remove_objects([oid for oid, _ in reanchor])

            new_ids, new_masks = [], []
            for oid, d in reanchor:
                trk = self.tracks[oid]
                trk.anchor_frame, trk.score = frame_idx, float(det_scores_np[d])
                trk.anchor_area = float((det_masks[d] > 0).sum())
                trk.bad_streak, trk.lost_frames, trk.missed_dets = 0, 0, 0
                trk.last_good_mask = det_masks[d].clone()
                new_ids.append(oid)
                new_masks.append(det_masks[d])
                preds[oid] = (det_masks[d], 1.0, 1.0)
            for d in new_dets:
                oid = self.next_obj_id
                self.next_obj_id += 1
                self.tracks[oid] = Track(
                    obj_id=oid, anchor_frame=frame_idx,
                    anchor_area=float((det_masks[d] > 0).sum()), score=float(det_scores_np[d]),
                    last_good_mask=det_masks[d].clone(),
                )
                new_ids.append(oid)
                new_masks.append(det_masks[d])
                preds[oid] = (det_masks[d], 1.0, 1.0)
            if new_ids:
                self._add_objects(frame_idx, new_ids, torch.stack(new_masks))

            self.last_det_frame = frame_idx
            self.last_det_thumb = thumb
            event = {
                "reason": reason, "num_dets": len(det_masks), "new": len(new_dets),
                "reanchored": len(reanchor), "removed": len(removed),
                "time_detect": round(t_detect, 2),
            }

        self._prune_memory(frame_idx)
        times = {"backbone": t_backbone, "track": t_track, "total": time.time() - t0}
        return self._format_output(frame_idx, preds, event, times)

    def _format_output(self, frame_idx, preds, event, times):
        objects = {}
        for oid, (mask, obj_score, iou_score) in preds.items():
            trk = self.tracks.get(oid)
            if trk is None or trk.is_lost:
                continue
            objects[oid] = {
                "mask_logits": mask,  # (288, 288) in the resized square frame
                "obj_score": obj_score,
                "iou_score": iou_score,
                "unhealthy": trk.is_unhealthy,
            }
        return {
            "frame_idx": frame_idx,
            "objects": objects,
            "num_tracks": len(self.tracks),
            "num_lost": sum(t.is_lost for t in self.tracks.values()),
            "detection": event,
            "times": times,
        }
