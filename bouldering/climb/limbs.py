"""Hands, feet, hips and body size from COCO-17 keypoints."""
import numpy as np

# COCO-17 keypoint indices
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16

LIMBS = ("left_hand", "right_hand", "left_foot", "right_foot")
HANDS = ("left_hand", "right_hand")
# limb -> (joint before the end, end joint)
_CHAINS = {
    "left_hand": (L_ELBOW, L_WRIST), "right_hand": (R_ELBOW, R_WRIST),
    "left_foot": (L_KNEE, L_ANKLE), "right_foot": (R_KNEE, R_ANKLE),
}
# shoulder -> hip -> knee -> ankle is ~0.78 of the body height and doesn't change with posture
BODY_CHAIN_FRACTION = 0.78
_BODY_CHAINS = [(L_SHOULDER, L_HIP, L_KNEE, L_ANKLE), (R_SHOULDER, R_HIP, R_KNEE, R_ANKLE)]


def limb_points(kpts, scores, cfg):
    """{limb: (x, y) or None}: hands/feet extrapolated past the wrist/ankle (keypoints sit at the joint)."""
    out = {}
    for limb, (a, b) in _CHAINS.items():
        if scores[b] < cfg.kpt_thresh:
            out[limb] = None
            continue
        ext = cfg.hand_ext if "hand" in limb else cfg.foot_ext
        end = kpts[b]
        out[limb] = tuple(end + ext * (end - kpts[a])) if scores[a] >= cfg.kpt_thresh else tuple(end)
    return out


def hip_center(kpts, scores, cfg):
    ok = [i for i in (L_HIP, R_HIP) if scores[i] >= cfg.kpt_thresh]
    return tuple(kpts[ok].mean(axis=0)) if ok else None


def body_length(kpts, scores, cfg):
    """Shoulder -> hip -> knee -> ankle length in pixels (mean of the visible sides), or None."""
    lengths = []
    for chain in _BODY_CHAINS:
        if all(scores[i] >= cfg.kpt_thresh for i in chain):
            lengths.append(sum(np.linalg.norm(kpts[a] - kpts[b]) for a, b in zip(chain, chain[1:])))
    return float(np.mean(lengths)) if lengths else None


def keypoint_box(kpts, scores, cfg, frame_hw, pad=0.25):
    """COCO (x, y, w, h) box around the confident keypoints, padded; None if too few keypoints."""
    ok = scores >= cfg.kpt_thresh
    if ok.sum() < 6:
        return None
    (x0, y0), (x1, y1) = kpts[ok].min(axis=0), kpts[ok].max(axis=0)
    w, h = x1 - x0, y1 - y0
    x0, y0 = max(0.0, x0 - pad * w), max(0.0, y0 - pad * h)
    x1, y1 = min(frame_hw[1] - 1.0, x1 + pad * w), min(frame_hw[0] - 1.0, y1 + pad * h)
    return np.array([x0, y0, x1 - x0, y1 - y0], np.float32)


def transform_point(H, p):
    """Apply a 3x3 homography to an (x, y) point."""
    q = H @ np.array([p[0], p[1], 1.0])
    return (q[0] / q[2], q[1] / q[2])
