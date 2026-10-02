"""Route overlay video from the route pipeline's output."""
import cv2
import numpy as np

from ..common.visualization import color_for, draw_masks, open_video_writer
from .config import UNASSIGNED, UNASSIGNED_COLOR
from .data import RouteData


def render_routes(json_path, output_path, video_path=None, alpha=0.45, show_labels=True):
    """Render the route overlay; video_path overrides the video recorded in the JSON."""
    routes = RouteData(json_path)
    data = routes.data
    frame_nos = sorted(routes.frames)
    keyframe_at = {kf["frame"]: kf for kf in data["keyframes"]}
    cap = cv2.VideoCapture(str(video_path or data["video"]))
    fps = (cap.get(cv2.CAP_PROP_FPS) or 30.0) / data["stride"]
    writer = open_video_writer(output_path, fps, (routes.fw, routes.fh))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nos[0])
    written, wanted = 0, set(frame_nos)
    for frame_no in range(frame_nos[0], frame_nos[-1] + 1):
        ok, bgr = cap.read()
        if not ok:
            break
        if frame_no not in wanted:
            continue
        hold_ids, route_ids, masks = routes.frame(frame_no)
        colors = [color_for(int(r)) if r != UNASSIGNED else UNASSIGNED_COLOR for r in route_ids]
        labels = [None] * len(hold_ids)
        for r in set(route_ids.tolist()) - {UNASSIGNED}:  # label each route at its top hold
            idx = [i for i in np.nonzero(route_ids == r)[0] if masks[i].any()]
            if idx:
                top = min(idx, key=lambda i: np.nonzero(masks[i].any(axis=1))[0][0])
                labels[top] = f"R{r}"
        n_routes = len(set(route_ids.tolist()) - {UNASSIGNED})
        header = f"frame {frame_no}  routes {n_routes}  holds {len(hold_ids)}"
        kf = keyframe_at.get(frame_no)
        subheader = f"KEYFRAME [{kf['reason']}] {kf['stats']}" if kf else None
        writer.write(draw_masks(bgr, hold_ids, masks, alpha=alpha, show_ids=show_labels,
                                header=header, subheader=subheader, colors=colors, labels=labels))
        written += 1
    cap.release()
    writer.release()
    return written
