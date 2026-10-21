"""Drawing COCO-17 skeletons."""
import cv2
import numpy as np

# COCO-17 skeleton (keypoint index pairs)
SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
            (1, 3), (2, 4), (3, 5), (4, 6)]


def draw_poses(frame_bgr, boxes, poses, kpt_threshold=0.3, highlight=None, draw_boxes=True):
    """Skeletons on a BGR frame. highlight: optional {(x, y): BGR color} extra points (e.g. limbs on holds)."""
    canvas = frame_bgr.copy()
    for box, pose in zip(boxes, poses):
        if draw_boxes and box is not None:
            x, y, w, h = np.asarray(box).astype(int)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 200, 255), 2)
        kpts = np.asarray(pose["keypoints"])
        scores = np.asarray(pose["scores"])
        for a, b in SKELETON:
            if scores[a] > kpt_threshold and scores[b] > kpt_threshold:
                cv2.line(canvas, tuple(kpts[a].astype(int)), tuple(kpts[b].astype(int)), (0, 255, 0), 2)
        for (px, py), s in zip(kpts, scores):
            if s > kpt_threshold:
                cv2.circle(canvas, (int(px), int(py)), 4, (0, 0, 255), -1)
    for (px, py), color in (highlight or {}).items():
        cv2.circle(canvas, (int(px), int(py)), 9, color, -1)
        cv2.circle(canvas, (int(px), int(py)), 9, (0, 0, 0), 2)
    return canvas
