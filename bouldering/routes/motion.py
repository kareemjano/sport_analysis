"""Camera motion between frames: ORB homographies and checks on them."""
import cv2
import numpy as np

from .config import RouteConfig


class CameraMotion:
    """Homographies between frames from ORB features (RANSAC)."""

    def __init__(self, cfg: RouteConfig):
        self.cfg = cfg
        self.orb = cv2.ORB_create(cfg.orb_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def features(self, gray):
        return self.orb.detectAndCompute(gray, None)

    def homography(self, src, dst):
        """H mapping src -> dst pixels, the number of inliers, and whether it is reliable."""
        (ks, ds), (kd, dd) = src, dst
        if ds is None or dd is None or len(ks) < 8 or len(kd) < 8:
            return None, 0, False
        matches = self.matcher.match(ds, dd)
        if len(matches) < 8:
            return None, 0, False
        p_src = np.float32([ks[m.queryIdx].pt for m in matches])
        p_dst = np.float32([kd[m.trainIdx].pt for m in matches])
        H, inliers = cv2.findHomography(p_src, p_dst, cv2.RANSAC, 3.0)
        n = int(inliers.sum()) if inliers is not None else 0
        ok = H is not None and n >= self.cfg.min_inliers and n / len(matches) >= self.cfg.min_inlier_ratio
        return H, n, ok


def warp_mask(mask, H, hw):
    return cv2.warpPerspective(mask.astype(np.uint8), H, (hw[1], hw[0]), flags=cv2.INTER_NEAREST) > 0


def plausible(H, H_prev, hw, cfg: RouteConfig):
    """Consecutive frames move little: reject estimates that jump relative to the previous frame."""
    h, w = hw
    corners = np.float32([[[0, 0], [w, 0], [w, h], [0, h]]])
    jump = np.abs(cv2.perspectiveTransform(corners, H) - cv2.perspectiveTransform(corners, H_prev)).max()
    scale = np.linalg.det(H[:2, :2]) / np.linalg.det(H_prev[:2, :2])
    return jump <= cfg.max_jump * w and 1 / cfg.max_scale_step <= scale <= cfg.max_scale_step


def uncovered_fraction(H, hw):
    """Fraction of the current frame that lies outside the warped keyframe."""
    h, w = hw
    corners = cv2.perspectiveTransform(np.float32([[[0, 0], [w, 0], [w, h], [0, h]]]), H)[0]
    cover = np.zeros((h // 4, w // 4), np.uint8)
    cv2.fillPoly(cover, [np.int32(corners / 4)], 1)
    return 1.0 - cover.mean()
