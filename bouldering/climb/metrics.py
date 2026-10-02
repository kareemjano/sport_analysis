"""Attempt metrics from the hip trajectory in wall (reference keyframe) coordinates."""
import numpy as np

from .limbs import BODY_CHAIN_FRACTION


def smooth(values, window):
    """Moving average; window in samples. Odd reflection at the ends keeps linear trends (no flattening)."""
    values = np.asarray(values, np.float64)
    if window <= 1 or len(values) < 3:
        return values
    window = min(int(window), len(values))
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="reflect", reflect_type="odd")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def deadband_path(values, step):
    """Path length along one axis, counting a move only once it reaches `step` from the last counted position."""
    anchor, total = values[0], 0.0
    for v in values[1:]:
        if abs(v - anchor) >= step:
            total += abs(v - anchor)
            anchor = v
    return total + abs(values[-1] - anchor)  # the last partial move


def attempt_metrics(xs, ys, body_lengths, touched, route_holds, start_frame, end_frame, fps, px_scale,
                    cfg, stride):
    """Distances travelled, time and holds used.

    xs/ys: hip positions in reference work coords; body_lengths: reference work px. Distances travelled
    are path lengths of the smoothed hips per axis, ignoring back-and-forth below cfg.move_step (jitter);
    px_scale: work px -> frame px; stride: frames between processed frames (for the smoothing window).
    """
    out = {"time_s": round((end_frame - start_frame) / fps, 2)}
    used = sorted(set(touched) & set(route_holds))
    out.update(holds_used=len(used), holds_total=len(route_holds),
               holds_used_pct=round(100.0 * len(used) / max(len(route_holds), 1), 1), holds_used_ids=used)
    if len(xs) >= 2:
        window = cfg.smooth_seconds * fps / stride
        x, y = smooth(xs, window) * px_scale, smooth(ys, window) * px_scale
        step = cfg.move_step * (float(np.median(body_lengths)) * px_scale if body_lengths else 0.0)
        vertical, horizontal = deadband_path(y, step), deadband_path(x, step)
        net = float(y[0] - y.min())  # image y grows downwards
    else:
        vertical = horizontal = net = 0.0
    out.update(vertical_px=round(vertical, 1), horizontal_px=round(horizontal, 1), net_height_px=round(net, 1))
    if body_lengths:
        px_per_m = float(np.median(body_lengths)) * px_scale / (BODY_CHAIN_FRACTION * cfg.climber_height)
        out.update(px_per_m=round(px_per_m, 1), vertical_m=round(vertical / px_per_m, 2),
                   horizontal_m=round(horizontal / px_per_m, 2), net_height_m=round(net / px_per_m, 2))
    return out
