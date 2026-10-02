"""Hold identities across keyframes."""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from ..common.masks import mask_iou
from .config import RouteConfig
from .motion import warp_mask


def update_holds(prev_holds, H, det_masks, cfg: RouteConfig, next_hold_id):
    """New keyframe: match detections to the previous holds warped into this frame.

    prev_holds: {hold_id: mask in the previous keyframe}; H: previous keyframe -> this
    frame (None after a cut, then every detection is a new hold).
    Returns ({hold_id: mask in this frame}, next_hold_id, stats).
    """
    hw = det_masks.shape[1:] if len(det_masks) else None
    holds, matched_det = {}, set()
    stats = {"matched": 0, "new": 0, "kept": 0, "retired": 0}
    if prev_holds and H is not None and len(det_masks):
        ids = list(prev_holds)
        warped = np.stack([warp_mask(prev_holds[i], H, hw) for i in ids])
        ious = mask_iou(torch.from_numpy(det_masks), torch.from_numpy(warped)).numpy()
        matched_prev = set()
        for d, p in zip(*linear_sum_assignment(-ious)):
            if ious[d, p] >= cfg.assoc_iou:
                holds[ids[p]] = det_masks[d]
                matched_det.add(d)
                matched_prev.add(p)
        stats["matched"] = len(matched_prev)
        for p, hid in enumerate(ids):
            if p in matched_prev:
                continue
            # not detected this time: keep it (holds don't move) while it is mostly in view
            if warped[p].sum() >= cfg.keep_visible * prev_holds[hid].sum():
                holds[hid] = warped[p]
                stats["kept"] += 1
            else:
                stats["retired"] += 1
    else:
        stats["retired"] = len(prev_holds)
    for d in range(len(det_masks)):
        if d not in matched_det:
            holds[next_hold_id] = det_masks[d]
            next_hold_id += 1
            stats["new"] += 1
    return holds, next_hold_id, stats
