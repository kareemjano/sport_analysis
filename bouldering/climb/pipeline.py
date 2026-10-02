"""Follow the climber on the routes found by the route pipeline and judge each attempt.

Per processed frame of the route output: the climber's pose (SAM3 "person" box + ViTPose+),
which holds the hands and feet are on, and the attempt state:
  - an attempt starts when `start_limbs` limbs stay on one route's holds for `start_seconds`;
    from then on only that route counts (the others are disabled),
  - success: `top_hands` hands on the route's top hold for `top_seconds`,
  - fail: the climber comes off the route and drops (fell), the climber or the route is out of
    view for `lost_seconds`, or the video ends first (did not reach top).
After an attempt the tracker re-arms once the climber is off the wall, so a video can hold several.

Output <stem>_climb.json: per-frame pose, contacts and state, plus a summary per attempt
(time, vertical / horizontal distance travelled, net height, holds used).
"""
import json
import time
from dataclasses import asdict

import cv2
import numpy as np

from .attempt import AttemptTracker
from .contacts import find_contacts
from .limbs import body_length, limb_points


def _wall_box(routes, frame_no):
    """COCO box (frame px) around the hold centroids visible in a frame, or None."""
    geo = routes.holds_geometry(routes.frames[frame_no]["keyframe"])
    if len(geo.centroids) == 0:
        return None
    H = routes.to_frame @ np.array(routes.frames[frame_no]["H"])
    pts = cv2.perspectiveTransform(np.float32(geo.centroids)[None], H)[0]
    (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
    return np.array([x0, y0, x1 - x0, y1 - y0])


def _round(points, nd=1):
    return None if points is None else np.round(np.asarray(points, np.float64), nd).tolist()


def run_climb(climber_tracker, routes, video_path, cfg, json_path, start=0, max_frames=None):
    """climber_tracker: ClimberTracker; routes: RouteData of the same video. Returns the attempt summary."""
    frame_numbers = [f for f in sorted(routes.frames) if f >= start]
    if max_frames is not None:
        frame_numbers = frame_numbers[:max_frames]
    stride = routes.data["stride"]
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    attempts = AttemptTracker(routes, cfg, fps, stride)
    data = {"video": str(video_path), "fps": fps, "stride": stride, "config": asdict(cfg), "frames": [],
            "attempts": []}
    wanted, t_start = set(frame_numbers), time.perf_counter()

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_numbers[0])
    frame_no, k = frame_numbers[0], 0
    try:
        while frame_numbers and frame_no <= frame_numbers[-1]:
            ok, bgr = cap.read()
            if not ok:
                break
            if frame_no in wanted:
                t0 = time.perf_counter()
                climber = climber_tracker(frame_no, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), _wall_box(routes, frame_no))
                record = _judge_frame(attempts, routes, cfg, frame_no, climber)
                data["frames"].append(record)
                status, contacts = record, record["contacts"]
                k += 1
                n_on = sum(c is not None for c in (contacts or {}).values())
                print(f"[{k}/{len(frame_numbers)}] frame {frame_no}: {status['state']}"
                      + (f" route R{status['route']}" if status["route"] is not None else "")
                      + (f", {n_on} limbs on holds" if climber else ", no climber")
                      + (" (SAM3 detect)" if climber and climber["detected"] else "")
                      + f" | {time.perf_counter() - t0:.1f}s", flush=True)
            frame_no += 1
    except KeyboardInterrupt:
        print(f"Interrupted at frame {frame_no}; saving what was processed")
    finally:
        cap.release()
        _save(data, attempts, json_path)
    print(f"Wrote {json_path}: {len(data['frames'])} frames, {len(data['attempts'])} attempts "
          f"({time.perf_counter() - t_start:.0f}s)")
    return data["attempts"]


def _judge_frame(attempts, routes, cfg, frame_no, climber):
    """Limbs, contacts and attempt state for one frame -> the frame's JSON record."""
    limbs = contacts = None
    if climber is not None:
        limbs = limb_points(climber["keypoints"], climber["scores"], cfg)
        length = body_length(climber["keypoints"], climber["scores"], cfg)
        radius = cfg.contact_radius * (length or 0.25 * routes.fh)
        contacts = find_contacts(limbs, routes, frame_no, radius)
    status = attempts.update(frame_no, climber, contacts)
    return {
        "frame": frame_no,
        "box": _round(climber["box"]) if climber else None,
        "keypoints": _round(climber["keypoints"]) if climber else None,
        "scores": _round(climber["scores"], 3) if climber else None,
        "detected": bool(climber["detected"]) if climber else False,
        "limbs": {name: _round(p) for name, p in limbs.items()} if limbs else None,
        "contacts": contacts, **status,
    }


def _save(data, attempts, json_path):
    attempts.finish(data["frames"][-1]["frame"] if data["frames"] else 0)
    if data["frames"] and attempts.current is not None:  # show the final result on the last frame
        data["frames"][-1].update(attempts.status(data["frames"][-1]["frame"], {}))
    data["attempts"] = attempts.summary()
    with open(json_path, "w") as f:
        json.dump(data, f)


def rejudge_climb(json_path, routes, cfg):
    """Re-run contacts and attempt judging on the poses saved in <stem>_climb.json (no models).

    For tuning ClimbConfig thresholds; the detection options (person_*, redetect_seconds) don't apply.
    """
    data = load_climb(json_path)
    attempts = AttemptTracker(routes, cfg, data["fps"], data["stride"])
    frames, data["frames"], data["config"] = data["frames"], [], asdict(cfg)
    for fr in frames:
        climber = None if fr["keypoints"] is None else {
            "box": fr["box"], "keypoints": np.array(fr["keypoints"]), "scores": np.array(fr["scores"]),
            "detected": fr["detected"]}
        data["frames"].append(_judge_frame(attempts, routes, cfg, fr["frame"], climber))
    _save(data, attempts, json_path)
    print(f"Re-judged {json_path}: {len(data['frames'])} frames, {len(data['attempts'])} attempts")
    return data["attempts"]


def load_climb(json_path):
    with open(json_path) as f:
        return json.load(f)
