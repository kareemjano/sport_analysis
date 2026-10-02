"""Per-frame mask files written by the tracking pipeline."""
import json

import numpy as np
import torch

from ..common.masks import pack_masks, unpack_masks


def save_frame_masks(masks_dir, frame_no, result):
    """Save one processed frame's masks (bit-packed, low-res) and track info to an .npz.

    Masks are stored at the tracker's low resolution (288x288 over the whole frame),
    so a full video stays small and can be re-rendered without re-running the model.
    """
    ids = list(result["objects"])
    if ids:
        masks = (torch.stack([result["objects"][i]["mask_logits"] for i in ids]) > 0).cpu().numpy()
    else:
        masks = np.zeros((0, 1, 1), dtype=bool)
    info = {k: result[k] for k in ("frame_idx", "num_tracks", "num_lost", "detection")}
    np.savez_compressed(
        f"{masks_dir}/frame_{frame_no:06d}.npz",
        obj_ids=np.array(ids, dtype=np.int64),
        masks=pack_masks(masks),
        mask_hw=np.array(masks.shape[-2:]),
        unhealthy=np.array([result["objects"][i]["unhealthy"] for i in ids], dtype=bool),
        info=json.dumps(info),
    )


def load_frame_masks(path):
    """Inverse of save_frame_masks: returns obj_ids, bool masks (N, h, w), unhealthy, info."""
    data = np.load(path)
    masks = unpack_masks(data["masks"], data["mask_hw"])
    return data["obj_ids"], masks, data["unhealthy"], json.loads(str(data["info"]))
