"""Which hold (if any) each hand / foot is on."""
from .limbs import transform_point


def find_contacts(limbs, routes, frame_no, radius):
    """{limb: {"hold", "route", "dist", "near"} or None} for limbs within `radius` (frame px) of a hold.

    hold / route / dist are the nearest hold; near lists every hold within reach as [hold, route, dist],
    nearest first: a hand can grip a small hold and the volume next to it at the same time, and should
    count for both routes. limbs: {limb: (x, y) frame pixels or None}; routes: RouteData. Distances
    are measured in the keyframe's work coordinates, where the hold masks live.
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
        near = [[int(geo.hold_ids[i]), int(geo.route_ids[i]), round(float(d[i]), 1)]
                for i in d.argsort() if d[i] <= r]
        if near:
            out[limb] = {"hold": near[0][0], "route": near[0][1], "dist": near[0][2], "near": near}
    return out


def limb_holds(contact, route=None):
    """[(hold, route)] a limb is on (nearest first), optionally only the holds of one route."""
    near = contact.get("near") or [[contact["hold"], contact["route"], contact["dist"]]]
    return [(h, r) for h, r, _ in near if route is None or r == route]
