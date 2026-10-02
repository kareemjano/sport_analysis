"""Detect-and-track climbing holds with the SAM3 detector + tracker."""
from .config import PipelineConfig, Track
from .runner import render_tracks, run_tracking
from .storage import load_frame_masks, save_frame_masks
