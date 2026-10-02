from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class PipelineConfig:
    prompt: str = "climbing hold"
    # detection
    det_score_thresh: float = 0.5  # keep detections above this (presence-weighted) score
    det_nms_iou: float = 0.5
    new_track_score_thresh: float = 0.6  # unmatched detections above this start a new track
    assoc_iou_thresh: float = 0.3  # det <-> track match
    reanchor_iou_thresh: float = 0.6  # matched but IoU below this -> re-anchor on detection
    # track health
    min_obj_score: float = 0.5  # tracker object presence prob; below -> lost
    min_iou_score: float = 0.6  # tracker predicted mask IoU; below -> diverging
    max_area_ratio: float = 2.0  # area vs anchor area outside [1/r, r] -> diverging
    bad_patience: int = 2  # consecutive bad frames before a track counts as unhealthy
    # re-detection triggers
    unhealthy_fraction: float = 0.1  # fraction of tracks newly unhealthy to trigger
    min_det_interval: int = 3  # frames; never re-detect more often (except no tracks)
    max_det_interval: int = 30  # frames; always re-detect after this many frames
    scene_change_thresh: float = 0.08  # mean abs gray diff vs last detection frame
    # track removal
    max_lost_frames: int = 45  # remove tracks lost (e.g. occluded) for this long
    max_missed_dets: int = 2  # remove present tracks unmatched by this many detections
    # tracker memory kept per state (frames); older non-conditioning memory is pruned
    mem_window: int = 16
    # objects per tracker state; bounds peak memory of one propagation batch
    max_objects_per_state: int = 8


@dataclass
class Track:
    obj_id: int
    anchor_frame: int
    anchor_area: float
    score: float = 1.0  # detection score at last anchor
    obj_score: float = 1.0
    iou_score: float = 1.0
    area: float = 0.0
    bad_streak: int = 0
    bad_since: int = -1
    lost_frames: int = 0
    missed_dets: int = 0
    last_good_mask: Optional[torch.Tensor] = field(default=None, repr=False)

    @property
    def is_lost(self):
        return self.lost_frames > 0

    @property
    def is_unhealthy(self):
        return self.bad_streak >= 1
