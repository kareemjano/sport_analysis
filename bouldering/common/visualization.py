"""Drawing masks on frames and writing videos."""
import colorsys

import cv2
import numpy as np


def color_for(obj_id):
    h = (obj_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)  # BGR


def draw_masks(frame_bgr, obj_ids, masks, unhealthy=None, alpha=0.45, show_ids=True, header=None, subheader=None,
               colors=None, labels=None):
    """Overlay segmentation masks on a BGR frame.

    masks: (N, h, w) bool masks covering the whole frame at any resolution (resized here).
    unhealthy: optional (N,) bools; those objects get a thick red contour.
    colors: optional (N,) BGR colors (default: color_for(obj_id)).
    labels: optional (N,) label strings, None = no label (default: the object id).
    """
    colors = colors if colors is not None else [color_for(int(oid)) for oid in obj_ids]
    labels = labels if labels is not None else [str(oid) for oid in obj_ids]
    fh, fw = frame_bgr.shape[:2]
    vis = frame_bgr.copy()
    overlay = vis.copy()
    full = []
    for m, color in zip(masks, colors):
        if m.shape != (fh, fw):
            m = cv2.resize(m.astype(np.float32), (fw, fh), interpolation=cv2.INTER_LINEAR) > 0.5
        full.append(m)
        overlay[m] = color
    vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)

    for i, (m, color, label) in enumerate(zip(full, colors, labels)):
        if not m.any():
            continue
        bad = unhealthy is not None and bool(unhealthy[i])
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 0, 255) if bad else color, 3 if bad else 1)
        if show_ids and label is not None:
            ys, xs = np.nonzero(m)
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(vis, label, (cx - 8, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, label, (cx - 8, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    if header:
        cv2.rectangle(vis, (0, 0), (fw, 56 if subheader else 30), (0, 0, 0), -1)
        cv2.putText(vis, header, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        if subheader:
            cv2.putText(vis, subheader, (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return vis


def open_video_writer(path, fps, size):
    """mp4 writer; prefers H.264 (plays in browsers/VS Code) and falls back to mp4v."""
    log_level = cv2.utils.logging.getLogLevel()
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)  # failed codec probes are noisy
    try:
        for codec in ("avc1", "mp4v"):
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
            if writer.isOpened():
                return writer
    finally:
        cv2.utils.logging.setLogLevel(log_level)
    raise RuntimeError(f"Could not open a video writer for {path}")
