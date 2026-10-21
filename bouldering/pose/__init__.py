"""Climber pose: ViTPose+ keypoints on person boxes (PyTorch via transformers, or ONNX Runtime)."""
import importlib

from .drawing import SKELETON, draw_poses  # noqa: F401

# the backends are only imported when used (transformers / onnxruntime)
_LAZY = {"PoseEstimator": "vitpose", "OnnxPoseEstimator": "onnx_pose"}


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(name)
