"""Attempt state machine: searching -> climbing(route) -> success / fail -> searching."""
from dataclasses import dataclass, field
from typing import List, Optional, Set

import cv2
import numpy as np

from ..routes.config import UNASSIGNED
from .limbs import HANDS, body_length, hip_center
from .metrics import attempt_metrics

SEARCHING, CLIMBING, SUCCESS, FAIL = "searching", "climbing", "success", "fail"


@dataclass
class Attempt:
    route: int
    start_frame: int
    ref_keyframe: int  # wall coordinates = work coords of this keyframe
    touched: Set[int] = field(default_factory=set)
    route_holds: Set[int] = field(default_factory=set)
    xs: List[float] = field(default_factory=list)
    ys: List[float] = field(default_factory=list)
    body_lengths: List[float] = field(default_factory=list)
    best_y: float = np.inf  # highest hip position (smallest y)
    last_contact: int = 0  # last frame with a limb on the route
    last_seen: int = 0  # last frame the climber was found
    route_seen: int = 0  # last frame the route was in view
    top_since: Optional[int] = None
    top_last: Optional[int] = None  # last frame with the hands on the top hold
    top_hold: Optional[int] = None
    end_frame: Optional[int] = None
    result: Optional[str] = None
    reason: Optional[str] = None
    metrics: dict = field(default_factory=dict)


class AttemptTracker:
    def __init__(self, routes, cfg, fps, stride):
        self.routes, self.cfg, self.fps, self.stride = routes, cfg, fps, stride
        self.px_scale = routes.fw / routes.data["work_size"][0]  # work px -> frame px
        self.attempts: List[Attempt] = []
        self.state = SEARCHING
        self.current: Optional[Attempt] = None
        self.candidate = None  # {route, since, last, keyframe, touched}: a possible attempt start
        self.last_any_contact = -np.inf
        self.contact_since = {}  # (limb, hold) -> first frame of the current contact streak

    def seconds(self, frames):
        return frames / self.fps

    # ------------------------------------------------------------------ update
    def update(self, frame_no, climber, contacts):
        """Advance by one processed frame. climber: ClimberTracker output or None; contacts: find_contacts output."""
        contacts = contacts or {}
        on_hold = {limb: c for limb, c in contacts.items() if c is not None}
        if on_hold:
            self.last_any_contact = frame_no
        self.contact_since = {(limb, c["hold"]): self.contact_since.get((limb, c["hold"]), frame_no)
                              for limb, c in on_hold.items()}
        if self.state == SEARCHING:
            self._search(frame_no, on_hold)
        elif self.state == CLIMBING:
            self._climb(frame_no, climber, on_hold)
        elif self.seconds(frame_no - self.last_any_contact) >= self.cfg.rearm_seconds:
            self.state, self.current = SEARCHING, None  # off the wall: ready for the next attempt
        return self.status(frame_no, on_hold)

    def _search(self, frame_no, on_hold):
        counts = {}
        for c in on_hold.values():
            if c["route"] != UNASSIGNED:
                counts[c["route"]] = counts.get(c["route"], 0) + 1
        best = max(counts, key=counts.get) if counts else None
        cand = self.candidate
        if best is None or counts[best] < self.cfg.start_limbs:
            # a limb missed for a moment (pose jitter) doesn't break the streak
            if cand is not None and self.seconds(frame_no - cand["last"]) > self.cfg.contact_gap_seconds:
                self.candidate = None
            return
        holds = self._held(frame_no, on_hold, best)
        if cand is None or cand["route"] != best:
            cand = self.candidate = {"route": best, "since": frame_no, "last": frame_no, "touched": holds,
                                     "keyframe": self.routes.frames[frame_no]["keyframe"]}
        cand["last"] = frame_no
        cand["touched"] |= holds
        if self.seconds(frame_no - cand["since"]) >= self.cfg.start_seconds:
            self.current = Attempt(route=best, start_frame=cand["since"], ref_keyframe=cand["keyframe"],
                                   touched=set(cand["touched"]), last_contact=frame_no, last_seen=frame_no,
                                   route_seen=frame_no)
            self.attempts.append(self.current)
            self.state, self.candidate = CLIMBING, None

    def _climb(self, frame_no, climber, on_hold):
        a, cfg = self.current, self.cfg
        kf = self.routes.frames[frame_no]["keyframe"]
        geo = self.routes.holds_geometry(kf)
        route_idx = np.nonzero(geo.route_ids == a.route)[0]
        if len(route_idx):
            a.route_seen = frame_no
            a.route_holds.update(int(h) for h in geo.hold_ids[route_idx])
            a.top_hold = int(geo.hold_ids[route_idx[np.argmin(geo.centroids[route_idx, 1])]])
        elif self.seconds(frame_no - a.route_seen) >= cfg.lost_seconds:
            return self._finish(frame_no, FAIL, "route out of view")

        if climber is None:
            if self.seconds(frame_no - a.last_seen) >= cfg.lost_seconds:
                self._finish(frame_no, FAIL, "climber lost")
            return
        a.last_seen = frame_no

        on_route = {limb: c for limb, c in on_hold.items() if c["route"] == a.route}
        a.touched.update(self._held(frame_no, on_route, a.route))
        if on_route:
            a.last_contact = frame_no

        # hips and body size in wall coordinates
        H = self.routes.keyframe_to_keyframe(kf, a.ref_keyframe)
        hip = drop = None
        if H is not None:
            H = H @ self.routes.frame_to_keyframe(frame_no)
            kpts = cv2.perspectiveTransform(np.float32(climber["keypoints"])[None], H)[0]
            hip, length = hip_center(kpts, climber["scores"], cfg), body_length(kpts, climber["scores"], cfg)
            if length:
                a.body_lengths.append(length)
            if hip is not None:
                a.xs.append(float(hip[0]))
                a.ys.append(float(hip[1]))
                a.best_y = min(a.best_y, hip[1])
                if a.body_lengths:
                    drop = (hip[1] - a.best_y) / float(np.median(a.body_lengths))

        hands_on_top = sum(1 for limb in HANDS if limb in on_route and on_route[limb]["hold"] == a.top_hold)
        if hands_on_top >= cfg.top_hands:
            a.top_since = frame_no if a.top_since is None else a.top_since
            a.top_last = frame_no
            if self.seconds(frame_no - a.top_since) >= cfg.top_seconds:
                return self._finish(frame_no, SUCCESS, "topped")
        elif a.top_since is not None and self.seconds(frame_no - a.top_last) > cfg.contact_gap_seconds:
            a.top_since = None

        off_route = self.seconds(frame_no - a.last_contact)
        if off_route >= cfg.fall_seconds and (drop is None and off_route >= cfg.lost_seconds
                                              or drop is not None and drop >= cfg.fall_drop):
            self._finish(frame_no, FAIL, "fell")

    def _held(self, frame_no, on_hold, route):
        """Holds of `route` a limb has stayed on for touch_seconds (single-frame pose glitches don't count)."""
        return {c["hold"] for limb, c in on_hold.items()
                if c["route"] == route
                and self.seconds(frame_no - self.contact_since[(limb, c["hold"])]) >= self.cfg.touch_seconds}

    def _finish(self, frame_no, result, reason):
        a = self.current
        a.end_frame, a.result, a.reason = frame_no, result, reason
        a.metrics = self.metrics(a, frame_no)
        self.state = result

    def finish(self, frame_no):
        """End of the video: an attempt still in progress didn't reach the top."""
        if self.state == CLIMBING:
            self._finish(frame_no, FAIL, "did not reach top")

    # ------------------------------------------------------------------ output
    def metrics(self, a, frame_no):
        return attempt_metrics(a.xs, a.ys, a.body_lengths, a.touched, a.route_holds, a.start_frame, frame_no,
                               self.fps, self.px_scale, self.cfg, self.stride)

    def status(self, frame_no, on_hold):
        """Per-frame record: state, active route and live metrics."""
        a = self.current
        out = {"state": self.state, "attempt": len(self.attempts) - 1 if a else None,
               "route": a.route if a else (self.candidate["route"] if self.candidate else None)}
        if a is not None:
            out.update(top_hold=a.top_hold, touched=sorted(a.touched),
                       metrics=a.metrics if a.result else self.metrics(a, frame_no))
            if a.result:
                out.update(result=a.result, reason=a.reason)
        return out

    def summary(self):
        return [{"route": a.route, "start_frame": a.start_frame, "end_frame": a.end_frame, "result": a.result,
                 "reason": a.reason, **a.metrics} for a in self.attempts]
