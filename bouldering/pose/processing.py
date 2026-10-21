"""ViTPose top-down pre- and post-processing in NumPy (no transformers), as in VitPoseImageProcessor.

Pre-processing crops each person box (padded by 1.25, at the model's aspect ratio) with an
affine warp and normalizes it; post-processing decodes the heatmaps (argmax + DARK refinement)
and maps the keypoints back into the image.
"""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

NORMALIZE_FACTOR = 200.0
PADDING = 1.25


def box_to_center_scale(box, input_w, input_h):
    """COCO (x, y, w, h) box -> center (x, y) and scale (w, h) / 200, padded to the input aspect ratio."""
    x, y, w, h = box[:4]
    aspect = input_w / input_h
    center = np.array([x + w * 0.5, y + h * 0.5], np.float32)
    if w > aspect * h:
        h = w / aspect
    elif w < aspect * h:
        w = h * aspect
    return center, np.array([w, h], np.float32) / NORMALIZE_FACTOR * PADDING


def crop_inputs(rgb, boxes, input_w, input_h, mean, std):
    """uint8 RGB image (H, W, 3) + boxes -> pixel_values (N, 3, input_h, input_w) float32, centers, scales."""
    pixels, centers, scales = [], [], []
    for box in boxes:
        center, scale = box_to_center_scale(box, input_w, input_h)
        roi = scale * NORMALIZE_FACTOR
        # unbiased warp (get_warp_matrix with no rotation): box -> (input_w - 1) x (input_h - 1)
        sx, sy = (input_w - 1.0) / roi[0], (input_h - 1.0) / roi[1]
        M = np.array([[sx, 0.0, sx * (0.5 * roi[0] - center[0])],
                      [0.0, sy, sy * (0.5 * roi[1] - center[1])]], np.float32)
        crop = cv2.warpAffine(rgb, M, (input_w, input_h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        pixels.append(((crop.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1))
        centers.append(center)
        scales.append(scale)
    return np.stack(pixels).astype(np.float32), np.stack(centers), np.stack(scales)


def _argmax_keypoints(heatmaps):
    n, k, _, w = heatmaps.shape
    flat = heatmaps.reshape(n, k, -1)
    idx, scores = flat.argmax(2), flat.max(2)
    coords = np.stack([idx % w, idx // w], axis=-1).astype(np.float32)
    coords[scores <= 0] = -1
    return coords, scores


def _dark_refine(coords, heatmaps, kernel=11):
    """DARK: second-order Taylor refinement of the argmax on the log of the smoothed heatmap."""
    n, k, h, w = heatmaps.shape
    radius = (kernel - 1) // 2
    hm = np.array([[gaussian_filter(m, sigma=0.8, radius=(radius, radius), axes=(0, 1)) for m in maps]
                   for maps in heatmaps])
    hm = np.log(np.clip(hm, 0.001, 50))
    pad = np.pad(hm, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge").flatten()
    index = coords[..., 0] + 1 + (coords[..., 1] + 1) * (w + 2)
    index += (w + 2) * (h + 2) * np.arange(n * k).reshape(-1, k)
    index = index.astype(int).reshape(-1, 1)
    i_, ix1, iy1 = pad[index], pad[index + 1], pad[index + w + 2]
    ix1y1, ix1_y1_ = pad[index + w + 3], pad[index - w - 3]
    ix1_, iy1_ = pad[index - 1], pad[index - 2 - w]
    derivative = np.concatenate([0.5 * (ix1 - ix1_), 0.5 * (iy1 - iy1_)], axis=1).reshape(n, k, 2, 1)
    dxx, dyy = ix1 - 2 * i_ + ix1_, iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    hessian = np.linalg.inv(np.concatenate([dxx, dxy, dxy, dyy], axis=1).reshape(n, k, 2, 2)
                            + np.finfo(np.float32).eps * np.eye(2))
    return coords - np.einsum("ijmn,ijnk->ijmk", hessian, derivative).reshape(n, k, 2)


def decode_heatmaps(heatmaps, centers, scales, kernel=11):
    """Heatmaps (N, K, h, w) -> keypoints (N, K, 2) in image pixels and scores (N, K)."""
    _, _, h, w = heatmaps.shape
    coords, scores = _argmax_keypoints(heatmaps)
    coords = _dark_refine(coords, heatmaps, kernel)
    roi = scales * NORMALIZE_FACTOR
    out = np.empty_like(coords)
    out[..., 0] = coords[..., 0] * (roi[:, None, 0] / (w - 1.0)) + centers[:, None, 0] - roi[:, None, 0] * 0.5
    out[..., 1] = coords[..., 1] * (roi[:, None, 1] / (h - 1.0)) + centers[:, None, 1] - roi[:, None, 1] * 0.5
    return out, scores
