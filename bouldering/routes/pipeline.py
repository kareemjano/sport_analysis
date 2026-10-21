"""Detect climbing holds, group them into routes and follow them through a video.

Holds are fixed to the wall, so instead of tracking every hold with the SAM3 tracker
(one memory-attention pass per hold and frame) the holds follow the camera motion:
  - the detector (backbone + detector heads) only runs on keyframes,
  - on every other frame a homography keyframe -> frame is estimated from ORB features
    and the keyframe masks are warped with it (milliseconds per frame).

A new keyframe is taken when the camera motion can't be estimated (cut, fast pan), when
too much of the frame shows wall that the keyframe didn't see, or optionally after a
maximum interval. Holds keep their ids across keyframes by mask IoU with the warped
previous holds.

Routes are clusters of holds by color and position (single linkage, so holds chain up
the wall); vertical gaps count less than horizontal ones because routes run upwards.

Outputs:
  <stem>_routes.json           keyframes, routes, per-frame homographies
  <stem>_routes/keyframe_*.npz hold masks per keyframe
Use bouldering.routes.load_routes(<stem>_routes.json).frame(frame_no) to get per-frame hold masks.
"""
import json
import os
import time
from dataclasses import asdict

import cv2
import numpy as np

from ..common.masks import pack_masks
from .clustering import cluster_routes, extend_routes, hold_features, route_summary, stable_route_ids
from .config import UNASSIGNED, RouteConfig
from .holds import HoldRegistry
from .motion import CameraMotion, plausible, uncovered_fraction


def run_routes(detector, cfg: RouteConfig, video_path, data_dir, json_path, start=0, stride=1, max_frames=None,
               prompt=None):
    """detector: a HoldDetector; prompt is only recorded in the JSON."""
    motion = CameraMotion(cfg)

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    work_hw = (round(fh * cfg.work_scale), round(fw * cfg.work_scale))
    frame_numbers = list(range(start, total, stride))
    if max_frames is not None:
        frame_numbers = frame_numbers[:max_frames]
    wanted = set(frame_numbers)
    os.makedirs(data_dir, exist_ok=True)
    for old in os.listdir(data_dir):
        if old.startswith("keyframe_") and old.endswith(".npz"):
            os.remove(os.path.join(data_dir, old))

    data = {
        "video": str(video_path), "frame_size": [fw, fh], "work_size": [work_hw[1], work_hw[0]],
        "stride": stride, "prompt": prompt, "config": asdict(cfg), "keyframes": [], "frames": [],
    }
    registry, routes, next_route_id = HoldRegistry(cfg), {}, 0
    hold_colors = {}  # hold id -> color when it was last seen without the climber in front
    hold_route = {}  # hold id -> route id (or UNASSIGNED); kept for retired holds, which can come back
    key_feats = prev_feats = None
    H_key_prev = np.eye(3)
    last_key = -(10**9)
    t_start = time.perf_counter()

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frame_no = start
    try:
        while frame_no <= frame_numbers[-1]:
            ok, bgr = cap.read()
            if not ok:
                break
            if frame_no not in wanted:
                frame_no += 1
                continue
            small = cv2.resize(bgr, (work_hw[1], work_hw[0]), interpolation=cv2.INTER_AREA)
            feats = motion.features(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))

            # camera motion keyframe -> this frame: direct, else chained through the previous frame
            H, inliers, method = None, 0, "lost"
            if key_feats is not None:
                H, inliers, ok = motion.homography(key_feats, feats)
                method = "direct"
                if not (ok and plausible(H, H_key_prev, work_hw, cfg)):
                    H_step, inliers, ok = motion.homography(prev_feats, feats)
                    H, method = (H_step @ H_key_prev, "chained") if ok else (None, "lost")
                    if H is not None and not plausible(H, H_key_prev, work_hw, cfg):
                        H, method = None, "lost"

            since = frame_no - last_key
            if H is None and key_feats is not None and since < cfg.min_keyframe_interval:
                # right after a keyframe (e.g. motion blur): hold the last estimate for a few frames
                H, method = H_key_prev, "held"
            reason = None
            if key_feats is None:
                reason = "start"
            elif H is None:
                reason = "motion-lost"
            elif since >= cfg.min_keyframe_interval:
                uncovered = uncovered_fraction(H, work_hw)
                if uncovered > cfg.new_area:
                    reason = f"new-area {uncovered:.0%}"
                elif cfg.max_keyframe_interval and since >= cfg.max_keyframe_interval:
                    reason = "periodic"

            if reason is not None:
                t0 = time.perf_counter()
                det_masks, _, person = detector(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), work_hw, frame_no)
                if H is None:
                    hold_route, hold_colors = {}, {}  # new wall: routes are clustered from scratch
                stats, kept = registry.update(H, det_masks, work_hw, person)
                hold_ids = list(registry.holds)
                masks = (np.stack([registry.holds[h] for h in hold_ids]) if hold_ids
                         else np.zeros((0, *work_hw), bool))
                colors, centers = hold_features(masks, small, exclude=person)
                full_colors, _ = hold_features(masks, small)
                # colors are measured without the climber; a hold that isn't visible now (covered, or kept
                # without a detection) keeps its last seen color
                for i, h in enumerate(hold_ids):
                    if (h in kept or np.isnan(colors[i]).any()) and h in hold_colors:
                        colors[i] = hold_colors[h]
                    elif np.isnan(colors[i]).any():
                        colors[i] = full_colors[i]
                    else:
                        hold_colors[h] = colors[i]
                if any(h in hold_route for h in hold_ids):  # same wall: known holds keep their routes
                    route_ids, next_route_id = extend_routes(hold_ids, hold_route, colors, centers, cfg,
                                                             next_route_id)
                else:  # start or after a cut
                    labels = cluster_routes(colors, centers, cfg)
                    route_ids, next_route_id = stable_route_ids(labels, hold_ids, routes, next_route_id)
                hold_route.update({int(h): int(r) for h, r in zip(hold_ids, route_ids)})
                routes = {int(r): {h for h, rr in zip(hold_ids, route_ids) if rr == r}
                          for r in set(route_ids.tolist()) - {UNASSIGNED}}
                file = f"keyframe_{frame_no:06d}.npz"
                np.savez_compressed(
                    os.path.join(data_dir, file), hold_ids=np.array(hold_ids, np.int64),
                    route_ids=route_ids, masks=pack_masks(masks), mask_hw=np.array(work_hw),
                )
                data["keyframes"].append({
                    "frame": frame_no, "reason": reason, "file": file, "stats": stats,
                    "routes": route_summary(route_ids, hold_ids, masks, colors, fw / work_hw[1]),
                    # previous keyframe -> this keyframe (work coords); links wall coordinates across keyframes
                    "H_from_prev": None if H is None else np.asarray(H).round(6).tolist(),
                })
                key_feats, H_key_prev, last_key = feats, np.eye(3), frame_no
                H, method = np.eye(3), "keyframe"
                n_routes = len(set(route_ids.tolist()) - {UNASSIGNED})
                print(f"frame {frame_no}: KEYFRAME [{reason}] {len(hold_ids)} holds ({stats}), "
                      f"{n_routes} routes, {int((route_ids == UNASSIGNED).sum())} unassigned "
                      f"({time.perf_counter() - t0:.1f}s)", flush=True)
            else:
                H_key_prev = H

            data["frames"].append({
                "frame": frame_no, "keyframe": len(data["keyframes"]) - 1,
                "H": np.asarray(H).round(6).tolist(), "inliers": inliers, "motion": method,
            })
            if frame_no % 100 == 0:
                print(f"frame {frame_no}/{total}: motion {method} ({inliers} inliers), "
                      f"{time.perf_counter() - t_start:.0f}s elapsed", flush=True)
            prev_feats = feats
            frame_no += 1
    except KeyboardInterrupt:
        print(f"Interrupted at frame {frame_no}; saving what was processed")
    finally:
        cap.release()
        with open(json_path, "w") as f:
            json.dump(data, f)
    print(f"Wrote {json_path}: {len(data['frames'])} frames, {len(data['keyframes'])} keyframes "
          f"({time.perf_counter() - t_start:.0f}s)")
