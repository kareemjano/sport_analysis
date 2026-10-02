from dataclasses import dataclass


@dataclass
class ClimbConfig:
    # climber detection (SAM3 text prompt) and pose
    person_prompt: str = "person"
    person_score: float = 0.5
    redetect_seconds: float = 5.0  # re-run SAM3 person detection at least this often
    kpt_thresh: float = 0.3  # keypoints below this score are ignored
    hand_ext: float = 0.35  # hand = wrist + this fraction of the elbow -> wrist vector
    foot_ext: float = 0.15  # foot = ankle + this fraction of the knee -> ankle vector
    # contacts
    contact_radius: float = 0.06  # a limb touches a hold within this distance (x body length)
    contact_gap_seconds: float = 0.4  # start / top streaks survive contacts missed for this long (pose jitter)
    # attempt start / end
    start_limbs: int = 3  # limbs on one route's holds to start an attempt
    start_seconds: float = 0.5  # ... for this long
    top_hands: int = 2  # hands on the top hold for a success
    top_seconds: float = 1.0  # ... for this long
    fall_seconds: float = 0.7  # no limb on the route for this long ...
    fall_drop: float = 0.25  # ... and the hips this far below their highest point (x body length) -> fell
    lost_seconds: float = 2.0  # climber or route not visible for this long -> fail
    rearm_seconds: float = 1.0  # after an attempt, no limb on any hold for this long -> look for the next one
    # metrics
    climber_height: float = 1.75  # meters; sets the pixel -> meter scale via body proportions
    smooth_seconds: float = 0.6  # moving-average window of the hip trajectory
    move_step: float = 0.1  # distance travelled only counts moves of at least this much (x body length; pose jitter)
