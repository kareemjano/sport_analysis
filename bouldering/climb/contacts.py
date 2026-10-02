"""Which hold (if any) each hand / foot is on."""
from .limbs import transform_point


def find_contacts(limbs, routes, frame_no, radius):
    """{limb: {"hold", "route", "dist"} or None} for limbs within `radius` (frame px) of a hold.

    limbs: {limb: (x, y) frame pixels or None}; routes: RouteData. Distances are measured in
    the keyframe's work coordinates, where the hold masks live.
    """
    fr = routes.frames[frame_no]
    geo = routes.holds_geometry(fr["keyframe"])
    H = routes.frame_to_keyframe(frame_no)
    r = radius * routes.data["work_size"][0] / routes.fw
    out = {}
    for limb, p in limbs.items():
        out[limb] = None
        if p is None or len(geo.hold_ids) == 0:
            continue
        d = geo.distances(transform_point(H, p))
        i = int(d.argmin())
        if d[i] <= r:
            out[limb] = {"hold": int(geo.hold_ids[i]), "route": int(geo.route_ids[i]), "dist": round(float(d[i]), 1)}
    return out
