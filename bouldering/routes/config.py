from dataclasses import dataclass

UNASSIGNED = -1
UNASSIGNED_COLOR = (170, 170, 170)


@dataclass
class RouteConfig:
    work_scale: float = 0.5  # masks, motion and clustering run at this fraction of the frame size
    # camera motion
    orb_features: int = 3000
    min_inliers: int = 150
    min_inlier_ratio: float = 0.3
    max_jump: float = 0.2  # reject motion that moves a frame corner by more than this fraction of the width
    max_scale_step: float = 1.25  # ... or changes the area scale by more than this factor per frame
    # keyframes
    new_area: float = 0.15  # take a keyframe if this fraction of the frame is outside the keyframe's view
    min_keyframe_interval: int = 5
    max_keyframe_interval: int = 0  # 0 = off
    # hold association across keyframes
    assoc_iou: float = 0.3
    keep_visible: float = 0.5  # keep undetected holds whose warped mask keeps this fraction of its area
    # route clustering. Colors are compared by hue for colored holds (shading changes lightness
    # and chroma much more than hue) and by lightness for gray/white/black holds; positions in
    # fractions of the frame width.
    min_chroma: float = 10.0  # below: achromatic hold (gray/white/black)
    hue_scale: float = 25.0  # degrees
    max_hue_diff: float = 25.0  # degrees; colored holds further apart in hue never join a route
    chroma_scale: float = 60.0
    l_scale: float = 80.0  # lightness weight for colored holds (varies a lot with shading)
    gray_l_scale: float = 15.0  # lightness weight for achromatic holds (black vs gray vs white)
    x_scale: float = 0.2
    y_scale: float = 0.4  # vertical gaps are cheaper: routes run up the wall
    min_route_holds: int = 3
