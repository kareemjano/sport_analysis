"""Group holds into routes by color and position."""
import cv2
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform

from .config import UNASSIGNED, RouteConfig


def hold_features(masks, bgr):
    """Median CIELAB color (L in 0..100, a/b centered) and centroid (x, y) / width per hold."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100 / 255
    lab[..., 1:] -= 128
    kernel = np.ones((3, 3), np.uint8)
    colors, centers = [], []
    w = bgr.shape[1]
    for m in masks:
        core = cv2.erode(m.astype(np.uint8), kernel) > 0  # avoid edge pixels mixed with the wall
        pix = lab[core if core.sum() >= 5 else m]
        colors.append(np.median(pix, axis=0))
        ys, xs = np.nonzero(m)
        centers.append([xs.mean() / w, ys.mean() / w])
    return np.array(colors).reshape(-1, 3), np.array(centers).reshape(-1, 2)


def cluster_routes(colors, centers, cfg: RouteConfig):
    """Cluster holds into routes. Returns a cluster label per hold (0..k-1) or UNASSIGNED."""
    n = len(colors)
    if n == 0:
        return np.zeros(0, int)
    if n == 1:
        return np.full(1, UNASSIGNED)
    L, a, b = colors[:, 0], colors[:, 1], colors[:, 2]
    chroma, hue = np.hypot(a, b), np.degrees(np.arctan2(b, a))
    colored = chroma >= cfg.min_chroma
    d_hue = np.abs((hue[:, None] - hue[None, :] + 180) % 360 - 180)
    d_l = np.abs(L[:, None] - L[None, :])
    both_colored = colored[:, None] & colored[None, :]
    both_gray = ~colored[:, None] & ~colored[None, :]
    d_color = np.where(
        both_colored,
        np.sqrt((d_hue / cfg.hue_scale) ** 2 + (np.abs(chroma[:, None] - chroma[None, :]) / cfg.chroma_scale) ** 2
                + (d_l / cfg.l_scale) ** 2),
        d_l / cfg.gray_l_scale,
    )
    d_x = (centers[:, None, 0] - centers[None, :, 0]) / cfg.x_scale
    d_y = (centers[:, None, 1] - centers[None, :, 1]) / cfg.y_scale
    dist = np.sqrt(d_color**2 + d_x**2 + d_y**2)
    # colored vs gray holds, or clearly different hues, never join
    dist[~(both_colored | both_gray) | (both_colored & (d_hue > cfg.max_hue_diff))] = 1e6
    np.fill_diagonal(dist, 0)
    labels = fcluster(linkage(squareform(dist, checks=False), method="single"), t=1.0, criterion="distance") - 1
    sizes = np.bincount(labels)
    labels = np.where(sizes[labels] >= cfg.min_route_holds, labels, UNASSIGNED)
    return labels


def stable_route_ids(labels, hold_ids, prev_routes, next_route_id):
    """Map cluster labels to route ids, reusing the ids of previous routes that share the most holds."""
    clusters = sorted(set(labels.tolist()) - {UNASSIGNED})
    members = [{hold_ids[i] for i in np.nonzero(labels == c)[0]} for c in clusters]
    prev = list(prev_routes.items())  # [(route_id, set of hold ids)]
    mapping = {}
    if clusters and prev:
        overlap = np.array([[len(m & holds) for _, holds in prev] for m in members])
        for ci, pi in zip(*linear_sum_assignment(-overlap)):
            if overlap[ci, pi] > 0:
                mapping[clusters[ci]] = prev[pi][0]
    for c in clusters:
        if c not in mapping:
            mapping[c] = next_route_id
            next_route_id += 1
    route_ids = np.array([mapping.get(int(c), UNASSIGNED) for c in labels], int)
    return route_ids, next_route_id


def route_summary(route_ids, hold_ids, masks, colors, to_frame):
    """Per route: hold ids, median color (Lab and RGB) and bounding box in frame pixels."""
    out = []
    for r in sorted(set(route_ids.tolist()) - {UNASSIGNED}):
        idx = np.nonzero(route_ids == r)[0]
        lab = np.median(colors[idx], axis=0)
        lab8 = np.uint8([[[lab[0] * 255 / 100, lab[1] + 128, lab[2] + 128]]])
        rgb = cv2.cvtColor(lab8, cv2.COLOR_LAB2RGB)[0, 0].tolist()
        ys, xs = np.nonzero(masks[idx].any(axis=0))
        out.append({
            "id": r, "holds": [int(hold_ids[i]) for i in idx],
            "color_lab": [round(float(v), 1) for v in lab], "color_rgb": rgb,
            "bbox_xyxy": [int(xs.min() * to_frame), int(ys.min() * to_frame),
                          int((xs.max() + 1) * to_frame), int((ys.max() + 1) * to_frame)],
        })
    return out
