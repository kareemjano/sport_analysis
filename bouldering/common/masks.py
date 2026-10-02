"""Mask utilities shared by the tracking and route pipelines."""
import numpy as np
import torch


def mask_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU between binary masks (N, H, W) x (M, H, W) -> (N, M).

    Uses a matmul instead of sam3.perflib.masks_ops.mask_iou, which materializes
    an (N, M, H*W) tensor and runs out of memory for ~100 holds on CPU.
    """
    a = a.flatten(1).float()
    b = b.flatten(1).float()
    inter = a @ b.T
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - inter
    return inter / union.clamp(min=1.0)


def nms_masks(probs, masks, prob_thresh, iou_thresh):
    """Greedy mask NMS; returns a bool keep mask over detections."""
    keep = torch.zeros_like(probs, dtype=torch.bool)
    valid = torch.nonzero(probs > prob_thresh).flatten()
    if len(valid) == 0:
        return keep
    valid = valid[probs[valid].argsort(descending=True)]
    ious = mask_iou(masks[valid] > 0, masks[valid] > 0)
    suppressed = torch.zeros(len(valid), dtype=torch.bool)
    for i in range(len(valid)):
        if suppressed[i]:
            continue
        keep[valid[i]] = True
        suppressed |= ious[i] > iou_thresh
    return keep


def mask_boxes(masks, frame_h, frame_w):
    """xyxy boxes in frame pixels for low-res bool masks (N, h, w); None for empty masks."""
    boxes = []
    for m in masks:
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            boxes.append(None)
            continue
        sx, sy = frame_w / m.shape[1], frame_h / m.shape[0]
        boxes.append([int(xs.min() * sx), int(ys.min() * sy), int((xs.max() + 1) * sx), int((ys.max() + 1) * sy)])
    return boxes


def pack_masks(masks):
    """Bit-pack bool masks (N, h, w) for compact .npz storage."""
    return np.packbits(masks, axis=-1)


def unpack_masks(packed, mask_hw):
    """Inverse of pack_masks."""
    h, w = mask_hw
    return np.unpackbits(packed, axis=-1, count=w).astype(bool).reshape(-1, h, w)
