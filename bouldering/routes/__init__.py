"""Group holds into routes and follow them through a video with camera-motion warping."""
from .config import UNASSIGNED, RouteConfig
from .data import RouteData, load_routes
from .detector import HoldDetector
from .pipeline import run_routes
from .render import render_routes
