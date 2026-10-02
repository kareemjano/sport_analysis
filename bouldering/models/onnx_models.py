"""ONNX Runtime sessions for the graphs written by bouldering.models.onnx_export."""
import json
import os

import numpy as np
import onnxruntime as ort

GRAPHS = (
    "image_encoder", "text_encoder", "detector_decoder", "tracker_memory_attention",
    "tracker_mask_decoder", "tracker_mask_prompt", "tracker_memory_encoder",
)


class OnnxModels:
    """ONNX Runtime sessions for the exported graphs, plus the export's config and constants."""

    def __init__(self, onnx_dir, providers):
        with open(os.path.join(onnx_dir, "config.json")) as f:
            self.config = json.load(f)
        self.constants = dict(np.load(os.path.join(onnx_dir, "tracker_constants.npz")))
        opts = ort.SessionOptions()
        # the default CPU arena grows by GBs per image encoder run instead of reusing memory
        opts.enable_cpu_mem_arena = False
        self.sessions = {}
        for name in GRAPHS:
            sess = ort.InferenceSession(os.path.join(onnx_dir, f"{name}.onnx"), opts, providers=providers)
            self.sessions[name] = (sess, {i.name for i in sess.get_inputs()}, [o.name for o in sess.get_outputs()])

    def run(self, name, **feeds):
        """Run a graph; unused inputs are dropped (the exporter prunes them). Returns {output: array}."""
        sess, inputs, outputs = self.sessions[name]
        results = sess.run(None, {k: v for k, v in feeds.items() if k in inputs})
        return dict(zip(outputs, results))
