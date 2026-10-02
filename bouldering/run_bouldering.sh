# python -m bouldering track --stride 3 --max-frames 60                  # track every hold (SAM3 tracker)
# python -m bouldering track --backend onnx --stride 3 --max-frames 60   # same, ONNX Runtime
# python -m bouldering routes                                            # holds -> routes, camera-motion warping
python -m bouldering routes --backend onnx
# python -m bouldering segment                                           # single image, text prompt
# python -m bouldering pose                                              # climber pose (ViTPose+-S)
# python -m bouldering export-onnx                                       # once, for --backend onnx