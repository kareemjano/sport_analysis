"""Hold identities across keyframes."""
import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from ..common.masks import mask_iou
from .config import RouteConfig
from .motion import warp_mask


def _overlap_of_smaller(a, b):
    """Intersection over the smaller mask's area, (N, h, w) x (M, h, w) -> (N, M)."""
    a, b = torch.from_numpy(a).flatten(1).float(), torch.from_numpy(b).flatten(1).float()
    inter = a @ b.T
    return (inter / torch.minimum(a.sum(1)[:, None], b.sum(1)[None, :]).clamp(min=1.0)).numpy()


class HoldRegistry:
    """Holds (id -> mask in the current keyframe) kept up to date with the detections of each keyframe.

    - A detection is matched to a known hold (warped into the new keyframe) by IoU (Hungarian),
      or when it mostly lies in / covers one (a different segmentation of the same hold).
    - A matched hold only takes the new mask if it is about the same size, or a larger one that
      contains it without running into other holds: a fragment (part of the hold covered by the
      climber) or two holds merged into one mask never replace a hold's mask.
    - Holds not detected stay (the climber may cover them) while mostly in view, at most
      cfg.max_missed_keyframes keyframes in a row; retired holds get their id back when detected again.
    - After a cut (H is None) every detection is a new hold.
    """

    def __init__(self, cfg: RouteConfig):
        self.cfg = cfg
        self.holds = {}
        self.missed = {}  # hold id -> keyframes in a row without a detection
        self.graveyard = {}  # retired hold id -> mask, for re-identification
        self.next_id = 1

    def _refresh(self, old, det, others):
        """The mask a matched hold keeps: the detection, unless it is a fragment or runs into other holds.

        others: (M, h, w) masks of the other holds.
        """
        a_old, a_det = old.sum(), det.sum()
        ratio = a_det / max(a_old, 1)
        takes_others = len(others) and ((others & det).reshape(len(others), -1).sum(1)
                                        >= 0.3 * others.reshape(len(others), -1).sum(1).clip(1)).any()
        if takes_others:
            return old  # several holds segmented as one
        if 0.7 <= ratio <= 1.5:
            return det
        if ratio > 1.5 and (old & det).sum() >= 0.8 * a_old:
            return det  # the hold was only partly visible before
        return old

    def update(self, H, det_masks, work_hw, person=None):
        """New keyframe. H: previous keyframe -> this keyframe (None after a cut).

        person: optional bool mask of people in this keyframe; no new hold starts on (or right next to)
        a person, e.g. on the climber's dark shorts. Known holds can still be matched there.

        Returns (stats, ids of holds kept without a detection).
        """
        cfg = self.cfg
        stats = {"matched": 0, "new": 0, "kept": 0, "retired": 0, "revived": 0, "duplicates": 0, "on_person": 0}
        near_person = None
        if person is not None and person.any():
            k = 2 * cfg.person_margin + 1
            near_person = cv2.dilate(person.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        if H is None:
            stats["retired"] = len(self.holds)
            prev, self.graveyard, self.missed = {}, {}, {}
        else:
            prev = self.holds
            self.graveyard = {h: w for h, m in self.graveyard.items() if (w := warp_mask(m, H, work_hw)).any()}
        holds, kept, matched_det = {}, set(), set()
        ids = list(prev)
        if ids:
            warped = np.stack([warp_mask(prev[i], H, work_hw) for i in ids])
            matched_prev = set()
            if len(det_masks):
                ious = mask_iou(torch.from_numpy(det_masks), torch.from_numpy(warped)).numpy()
                pairs = [(d, p) for d, p in zip(*linear_sum_assignment(-ious)) if ious[d, p] >= cfg.assoc_iou]
                overlap = _overlap_of_smaller(det_masks, warped)
                for d in range(len(det_masks)):  # leftovers: another segmentation of a known hold
                    if d not in {d for d, _ in pairs} and overlap[d].max() >= cfg.dup_overlap:
                        pairs.append((d, int(overlap[d].argmax())))
                for d, p in pairs:
                    matched_det.add(d)
                    if p in matched_prev:
                        stats["duplicates"] += 1
                        continue
                    holds[ids[p]] = self._refresh(warped[p], det_masks[d], np.delete(warped, p, axis=0))
                    matched_prev.add(p)
            stats["matched"] = len(matched_prev)
            for p, hid in enumerate(ids):
                if p in matched_prev:
                    self.missed[hid] = 0
                    continue
                self.missed[hid] = self.missed.get(hid, 0) + 1
                if (warped[p].sum() >= cfg.keep_visible * prev[hid].sum()
                        and not (cfg.max_missed_keyframes and self.missed[hid] > cfg.max_missed_keyframes)):
                    holds[hid] = warped[p]
                    kept.add(hid)
                    stats["kept"] += 1
                else:
                    stats["retired"] += 1
                    if warped[p].any():
                        self.graveyard[hid] = warped[p]

        new_masks = []
        grave_ids = list(self.graveyard)
        for d in range(len(det_masks)):
            if d in matched_det:
                continue
            det = det_masks[d:d + 1]
            if grave_ids:  # a retired hold seen again keeps its id
                masks = np.stack([self.graveyard[h] for h in grave_ids])
                score = np.maximum(mask_iou(torch.from_numpy(det), torch.from_numpy(masks)).numpy()[0] / cfg.assoc_iou,
                                   _overlap_of_smaller(det, masks)[0] / cfg.dup_overlap)
                if score.max() >= 1.0:
                    hid = grave_ids.pop(int(score.argmax()))
                    holds[hid] = self._refresh(self.graveyard.pop(hid), det_masks[d], np.zeros((0, *work_hw), bool))
                    self.missed[hid] = 0
                    stats["revived"] += 1
                    continue
            if near_person is not None and (det_masks[d] & near_person).sum() >= cfg.person_overlap * det_masks[d].sum():
                stats["on_person"] += 1
                continue
            if new_masks and _overlap_of_smaller(det, np.stack(new_masks)).max() >= cfg.dup_overlap:
                stats["duplicates"] += 1
                continue
            holds[self.next_id] = det_masks[d]
            self.missed[self.next_id] = 0
            new_masks.append(det_masks[d])
            self.next_id += 1
            stats["new"] += 1
        self.missed = {h: n for h, n in self.missed.items() if h in holds}
        self.holds = holds
        return stats, kept
