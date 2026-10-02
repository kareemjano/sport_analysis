"""Judge climbing attempts: climber pose on the detected routes -> start, success / fail and metrics."""
import importlib

from .config import ClimbConfig  # noqa: F401

# heavy modules (torch, transformers) are only imported when used
_LAZY = {"ClimberTracker": "climber", "detect_people": "climber", "run_climb": "pipeline", "load_climb": "pipeline", "rejudge_climb": "pipeline",
         "render_climb": "render"}


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(name)
