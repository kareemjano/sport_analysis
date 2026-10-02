"""Run a hold tracking pipeline over a video and render its saved masks."""
import glob
import json
import os
from dataclasses import asdict

import cv2
import numpy as np

from ..common.masks import mask_boxes
from ..common.visualization import draw_masks, open_video_writer
from .storage import load_frame_masks, save_frame_masks


def run_tracking(pipeline, video_path, masks_dir, log_path, start=0, stride=1, max_frames=None):
    """Run `pipeline` over the video, saving per-frame masks and a JSON log."""
    cfg = pipeline.cfg
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w, vid_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_numbers = list(range(start, total, stride))
    if max_frames is not None:
        frame_numbers = frame_numbers[:max_frames]
    pipeline.reset(len(frame_numbers), vid_h, vid_w)

    os.makedirs(masks_dir, exist_ok=True)
    # clear masks of a previous run so they don't end up in this video
    for old in glob.glob(os.path.join(masks_dir, "frame_*.npz")):
        os.remove(old)
    log = {"config": asdict(cfg), "video": str(video_path), "stride": stride, "frames": []}
    wanted = set(frame_numbers)
    k, frame_no = 0, start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    try:
        while k < len(frame_numbers):
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_no in wanted:
                result = pipeline.step(k, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                save_frame_masks(masks_dir, frame_no, result)
                det = result["detection"]
                print(
                    f"[{k + 1}/{len(frame_numbers)}] frame {frame_no}: {len(result['objects'])} visible, "
                    f"{result['num_tracks']} tracks, {result['num_lost']} lost"
                    + (f" | DETECT {det}" if det else "")
                    + " | " + ", ".join(f"{name} {t:.1f}s" for name, t in result["times"].items()),
                    flush=True,
                )
                ids = list(result["objects"])
                low_res = [(result["objects"][i]["mask_logits"] > 0).cpu().numpy() for i in ids]
                boxes = mask_boxes(low_res, vid_h, vid_w)
                log["frames"].append({
                    "frame": frame_no,
                    "detection": det,
                    "objects": {
                        str(oid): {"box_xyxy": box, "obj_score": round(o["obj_score"], 3),
                                   "iou_score": round(o["iou_score"], 3), "unhealthy": o["unhealthy"]}
                        for oid, box, o in zip(ids, boxes, (result["objects"][i] for i in ids))
                    },
                })
                k += 1
            frame_no += 1
    except KeyboardInterrupt:
        print(f"Interrupted after {k} frames; rendering what was processed so far")
    finally:
        cap.release()
        with open(log_path, "w") as f:
            json.dump(log, f)
    n_det = sum(1 for fr in log["frames"] if fr["detection"])
    print(f"Wrote {log_path}; detector ran on {n_det}/{len(log['frames'])} frames")


def render_tracks(video_path, masks_dir, output_path, alpha=0.45, show_ids=True, show_info=True, fps=None):
    """Render saved segmentation masks (from save_frame_masks) onto the video as a new .mp4.

    Only frames that have saved masks are written, so a run with stride n gives a
    video at fps / n. Returns the number of frames written.
    """
    paths = sorted(glob.glob(os.path.join(masks_dir, "frame_*.npz")))
    if not paths:
        raise FileNotFoundError(f"No frame_*.npz mask files in {masks_dir}")
    frame_to_path = {int(os.path.basename(p)[6:12]): p for p in paths}
    first, last = min(frame_to_path), max(frame_to_path)

    cap = cv2.VideoCapture(str(video_path))
    vid_w, vid_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps is None:
        step = np.diff(sorted(frame_to_path)).min() if len(frame_to_path) > 1 else 1
        fps = (cap.get(cv2.CAP_PROP_FPS) or 30.0) / step
    writer = open_video_writer(output_path, fps, (vid_w, vid_h))

    written = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    for frame_no in range(first, last + 1):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_no not in frame_to_path:
            continue
        obj_ids, masks, unhealthy, info = load_frame_masks(frame_to_path[frame_no])
        header = subheader = None
        if show_info:
            header = f"frame {frame_no}  visible {len(obj_ids)}  tracks {info['num_tracks']} (lost {info['num_lost']})"
            det = info.get("detection")
            if det:
                subheader = (f"DETECT [{det['reason']}] +{det['new']} new, "
                             f"{det['reanchored']} re-anchored, -{det['removed']}")
        writer.write(draw_masks(frame_bgr, obj_ids, masks, unhealthy, alpha, show_ids, header, subheader))
        written += 1
    cap.release()
    writer.release()
    return written
