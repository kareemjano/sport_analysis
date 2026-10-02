"""Climb overlay video: routes, climber skeleton, attempt state and metrics."""
import cv2

from ..common.visualization import color_for, draw_masks, open_video_writer
from ..pose.vitpose import draw_poses
from ..routes.config import UNASSIGNED, UNASSIGNED_COLOR
from ..routes.data import RouteData
from .attempt import CLIMBING, FAIL, SUCCESS
from .pipeline import load_climb

ON_ROUTE, OFF_ROUTE = (0, 255, 0), (0, 165, 255)  # BGR
RESULT_COLORS = {SUCCESS: (0, 170, 0), FAIL: (0, 0, 220)}


def _metrics_text(m):
    if not m:
        return ""
    dist = (f"vertical {m['vertical_m']:.2f} m  horizontal {m['horizontal_m']:.2f} m" if "vertical_m" in m
            else f"vertical {m['vertical_px']:.0f} px  horizontal {m['horizontal_px']:.0f} px")
    return f"{m['time_s']:.1f} s  {dist}  holds {m['holds_used']}/{m['holds_total']} ({m['holds_used_pct']:.0f}%)"


def _banner(vis, text, sub, color):
    h, w = vis.shape[:2]
    y0 = h - 110
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), color, -1)
    vis[:] = cv2.addWeighted(overlay, 0.75, vis, 0.25, 0)
    cv2.putText(vis, text, (16, y0 + 50), cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, sub, (16, y0 + 88), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_frame(bgr, routes, fr, alpha=0.45, kpt_thresh=0.3):
    hold_ids, route_ids, masks = routes.frame(fr["frame"])
    active = fr["route"] if fr["state"] in (CLIMBING, SUCCESS, FAIL) else None
    touched = set(fr.get("touched") or [])
    if active is None:  # searching: all routes as in the route video, the candidate route labelled
        colors = [color_for(int(r)) if r != UNASSIGNED else UNASSIGNED_COLOR for r in route_ids]
        labels = [None] * len(hold_ids)
    else:  # attempt: only the active route in color, the others disabled
        colors = [color_for(int(r)) if r == active else UNASSIGNED_COLOR for r in route_ids]
        labels = ["TOP" if h == fr.get("top_hold") else None for h in hold_ids]
    vis = draw_masks(bgr, hold_ids, masks, alpha=alpha, colors=colors, labels=labels)
    if touched:  # holds used in this attempt: filled a second time
        idx = [i for i, h in enumerate(hold_ids) if h in touched and route_ids[i] == active]
        if idx:
            vis = draw_masks(vis, hold_ids[idx], masks[idx], alpha=0.4, show_ids=False,
                             colors=[colors[i] for i in idx])

    if fr["keypoints"] is not None:
        highlight = {}
        for limb, c in (fr["contacts"] or {}).items():
            p = (fr["limbs"] or {}).get(limb)
            if c is not None and p is not None:
                on = c["route"] == active if active is not None else c["route"] != UNASSIGNED
                highlight[tuple(p)] = ON_ROUTE if on else OFF_ROUTE
        vis = draw_poses(vis, [fr["box"]], [{"keypoints": fr["keypoints"], "scores": fr["scores"]}], kpt_thresh,
                         highlight=highlight, draw_boxes=False)

    attempt = f"attempt {fr['attempt'] + 1}  " if fr.get("attempt") is not None else ""
    route = f"R{fr['route']}  " if fr["route"] is not None else ""
    state = {CLIMBING: "CLIMBING", SUCCESS: "TOPPED", FAIL: "FAILED"}.get(
        fr["state"], "waiting for start" if fr["keypoints"] is not None else "waiting for climber")
    if fr["state"] not in (CLIMBING, SUCCESS, FAIL) and fr["route"] is not None:
        state = "starting"
    header = f"frame {fr['frame']}  {attempt}{route}{state}"
    subheader = _metrics_text(fr.get("metrics")) if active is not None else None
    vis = draw_masks(vis, [], [], header=header, subheader=subheader)

    if fr.get("result"):
        text = "SUCCESS" if fr["result"] == SUCCESS else f"FAIL - {fr['reason']}"
        _banner(vis, f"R{active} {text}", _metrics_text(fr.get("metrics")), RESULT_COLORS[fr["result"]])
    return vis


def render_climb(climb_json, routes_json, output_path, video_path=None, alpha=0.45):
    data = load_climb(climb_json)
    routes = RouteData(routes_json)
    frames = {fr["frame"]: fr for fr in data["frames"]}
    order = sorted(frames)
    cap = cv2.VideoCapture(str(video_path or data["video"]))
    writer = open_video_writer(output_path, data["fps"] / data["stride"], (routes.fw, routes.fh))
    cap.set(cv2.CAP_PROP_POS_FRAMES, order[0])
    written = 0
    for frame_no in range(order[0], order[-1] + 1):
        ok, bgr = cap.read()
        if not ok:
            break
        if frame_no in frames:
            writer.write(draw_frame(bgr, routes, frames[frame_no], alpha, data["config"]["kpt_thresh"]))
            written += 1
    cap.release()
    writer.release()
    return written

