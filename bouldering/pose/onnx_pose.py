"""ViTPose+ on ONNX Runtime (graph written by `python -m bouldering export-onnx --only vitpose`)."""
import json
import os

import numpy as np
import onnxruntime as ort

from .processing import crop_inputs, decode_heatmaps

VITPOSE_ONNX = "vitpose.onnx"
VITPOSE_CONFIG = "vitpose.json"


class OnnxPoseEstimator:
    """Same interface as PoseEstimator: estimate(image, boxes) -> [{"keypoints", "scores"}]."""

    def __init__(self, onnx_dir, providers=("CPUExecutionProvider",), dataset_index=0):
        path = os.path.join(onnx_dir, VITPOSE_ONNX)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found; export it with "
                                    f"`python -m bouldering export-onnx --only vitpose --output-dir {onnx_dir}`")
        with open(os.path.join(onnx_dir, VITPOSE_CONFIG)) as f:
            self.config = json.load(f)
        self.session = ort.InferenceSession(path, providers=list(providers))
        self.input_w, self.input_h = self.config["input_size"]
        self.mean = np.array(self.config["image_mean"], np.float32)
        self.std = np.array(self.config["image_std"], np.float32)
        self.keypoint_names = self.config["keypoint_names"]
        self.dataset_index = dataset_index  # ViTPose+ expert head; 0 = COCO
        self.device = f"onnxruntime ({self.session.get_providers()[0]})"

    def estimate(self, image, boxes):
        """One dict per box with "keypoints" (17, 2) and "scores" (17,) numpy arrays in image pixels."""
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        if len(boxes) == 0:
            return []
        rgb = np.ascontiguousarray(np.asarray(image, np.uint8))
        pixels, centers, scales = crop_inputs(rgb, boxes, self.input_w, self.input_h, self.mean, self.std)
        heatmaps = self.session.run(["heatmaps"], {
            "pixel_values": pixels, "dataset_index": np.full(len(boxes), self.dataset_index, np.int64),
        })[0]
        keypoints, scores = decode_heatmaps(heatmaps, centers, scales)
        return [{"keypoints": k, "scores": s} for k, s in zip(keypoints, scores)]
