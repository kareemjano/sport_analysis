"""Reading the route pipeline's output."""
import json
import os

import cv2
import numpy as np

from ..common.masks import unpack_masks


class RouteData:
    """Reads <output>_routes.json and the keyframe masks; .frame(n) gives per-frame hold masks."""

    def __init__(self, json_path):
        with open(json_path) as f:
            self.data = json.load(f)
        self.dir = os.path.splitext(json_path)[0]
        self.frames = {fr["frame"]: fr for fr in self.data["frames"]}
        self.fw, self.fh = self.data["frame_size"]
        ww, wh = self.data["work_size"]
        self.to_frame = np.diag([self.fw / ww, self.fh / wh, 1.0])
        self._cache = {}
        self._geometry = {}

    def keyframe(self, index):
        if index not in self._cache:
            kf = self.data["keyframes"][index]
            z = np.load(os.path.join(self.dir, kf["file"]))
            masks = unpack_masks(z["masks"], z["mask_hw"])
            self._cache = {index: (z["hold_ids"], z["route_ids"], masks)}  # keep one keyframe in memory
        return self._cache[index]

    def frame(self, frame_no):
        """(hold_ids, route_ids, bool masks [N, frame_h, frame_w]) for a processed frame."""
        fr = self.frames[frame_no]
        hold_ids, route_ids, masks = self.keyframe(fr["keyframe"])
        # keyframe work coords -> this frame's work coords -> frame pixels
        H = self.to_frame @ np.array(fr["H"])
        warped = np.stack([
            cv2.warpPerspective(m.astype(np.uint8), H, (self.fw, self.fh), flags=cv2.INTER_NEAREST) > 0
            for m in masks
        ]) if len(masks) else np.zeros((0, self.fh, self.fw), bool)
        return hold_ids, route_ids, warped

    # ------------------------------------------------------------- wall geometry
    def frame_to_keyframe(self, frame_no):
        """3x3 homography: frame pixels of frame_no -> work coords of its keyframe."""
        return np.linalg.inv(self.to_frame @ np.array(self.frames[frame_no]["H"]))

    def holds_geometry(self, index):
        """HoldGeometry of a keyframe (work coords): ids, route ids, centroids and contours per hold."""
        if index not in self._geometry:
            hold_ids, route_ids, masks = self.keyframe(index)
            centroids, contours = [], []
            for m in masks:
                ys, xs = np.nonzero(m)
                centroids.append([xs.mean(), ys.mean()] if len(xs) else [np.nan, np.nan])
                contours.append(cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
            self._geometry[index] = HoldGeometry(np.asarray(hold_ids), np.asarray(route_ids),
                                                 np.array(centroids).reshape(-1, 2), contours)
        return self._geometry[index]

    def keyframe_link(self, index):
        """3x3 homography keyframe index-1 -> keyframe index (work coords), or None if they aren't linked.

        Uses the homography recorded by the route pipeline; outputs written before it was recorded
        fall back to a similarity transform fitted to the centroids of holds both keyframes share
        (none after a cut, where every hold gets a new id).
        """
        kf = self.data["keyframes"][index]
        if index == 0:
            return None
        if "H_from_prev" in kf:
            return None if kf["H_from_prev"] is None else np.array(kf["H_from_prev"])
        prev, cur = self.holds_geometry(index - 1), self.holds_geometry(index)
        shared = [h for h in prev.hold_ids if h in set(cur.hold_ids.tolist())]
        if len(shared) < 3:
            return None
        src = np.float32([prev.centroids[prev.index[h]] for h in shared])
        dst = np.float32([cur.centroids[cur.index[h]] for h in shared])
        A, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        return None if A is None else np.vstack([A, [0, 0, 1]])

    def keyframe_to_keyframe(self, index, ref):
        """3x3 homography work coords of keyframe index -> keyframe ref (ref <= index), or None if unlinked."""
        H = np.eye(3)
        for i in range(index, ref, -1):
            link = self.keyframe_link(i)
            if link is None:
                return None
            H = H @ np.linalg.inv(link)
        return H


class HoldGeometry:
    def __init__(self, hold_ids, route_ids, centroids, contours):
        self.hold_ids, self.route_ids, self.centroids, self.contours = hold_ids, route_ids, centroids, contours
        self.index = {int(h): i for i, h in enumerate(hold_ids)}

    def distances(self, point):
        """Distance (work px) from a point to every hold; 0 inside a hold."""
        p = (float(point[0]), float(point[1]))
        return np.array([
            max(0.0, -max((cv2.pointPolygonTest(c, p, True) for c in cs), default=-np.inf)) for cs in self.contours
        ])


def load_routes(json_path):
    return RouteData(json_path)
