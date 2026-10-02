# bouldering

Video analysis for bouldering: segment and track climbing holds, group them into routes, and estimate the climber's pose.

## Setup

```bash
pip install -r requirements.txt
pip install -e efficientsam3      # provides the `sam3` package (submodule)
```

Model weights go in `checkpoints/` (gitignored):

| file | used by |
|---|---|
| `sam3_litetext_mobileclip_s0_ctx16.pt` | `track`, `routes` (default), `export-onnx`; tracker weights for EfficientSAM3 |
| `sam3_litetext_mobileclip_s1_ctx32.pt` | `segment` (default) |
| `efficientsam3_{tinyvit,repvit,efficientvit}.pt` | `--model efficientsam3` |
| `onnx_sam3_litetext_s0/` | `--backend onnx` (written by `export-onnx`) |

## Commands

Run from the `sport_analysis/` directory. Inputs default to `data/bouldering/`, outputs to `data/bouldering/output/`.

```bash
python -m bouldering track --stride 3 --max-frames 60                  # track every hold (SAM3 tracker)
python -m bouldering track --backend onnx --stride 3 --max-frames 60   # same, ONNX Runtime
python -m bouldering routes                                            # holds -> routes, camera-motion warping
python -m bouldering routes --backend onnx
python -m bouldering climb --backend onnx --stride 2                    # judge attempts on the routes (pose)
python -m bouldering segment                                           # single image, text prompt
python -m bouldering pose                                              # climber pose (SAM3 person + ViTPose+-S)
python -m bouldering export-onnx                                       # once, for --backend onnx
```

Add `--render-only` to `track` / `routes` to re-render the video from saved results without running a model. `python -m bouldering <command> --help` lists all options.

Unless `--output` is given, outputs are named after the parameters so different runs don't overwrite each other:

```
<method>_<model>_<backend>[_p-<prompt>][_from<start>][_s<stride>][_n<max frames>].mp4

method   tracker (track: SAM3 tracker) | motion (routes: camera-motion warping) | climb | segment | pose
model    sam3-s0 | esam3-tinyvit-11m-s0 | ...   (with --backend onnx: read from the export's config.json)
```

e.g. `tracker_sam3-s0_onnx_s3_n60.mp4`, `motion_esam3-repvit-m1.1-s0_torch.mp4`. Re-running with the same options (e.g. adding `--render-only`) finds the same files.

## Climb: judging attempts

`climb` reads the `routes` output for the same options (and runs `routes` first if it is missing), finds the climber with SAM3 (`"person"` prompt — no extra detector) and ViTPose+-S, and follows the hands and feet (extrapolated past wrists / ankles) on the hold masks.

- **Start**: `--start-limbs` (3) limbs on one route's holds for `--start-seconds` (0.5); contacts missed for up to `--contact-gap-seconds` (0.4) don't break the streak (also for the top). From then on only that route is shown in color; the other routes are disabled (gray).
- **Success**: `--top-hands` (2) hands on the route's top hold (highest hold in view) for `--top-seconds` (1.0).
- **Fail**: off the route for `--fall-seconds` (0.7) with the hips `--fall-drop` (0.25 body lengths) below their highest point (*fell*); climber or route out of view for `--lost-seconds` (*climber lost* / *route out of view*); video ends first (*did not reach top*).
- **Next attempt**: once no limb touches any hold for `--rearm-seconds` (1.0).

Metrics per attempt, from the hip trajectory mapped into the wall coordinates of the keyframe where the attempt started (smoothed over `--smooth-seconds`):

| metric | definition |
|---|---|
| `time_s` | start of the 3-limb streak → success / fail |
| `vertical_*`, `horizontal_*` | distance travelled by the hips per axis: path length of the smoothed trajectory, counting moves ≥ `--move-step` (0.1 body chain ≈ 14 cm) so pose jitter doesn't add up; frame px and m |
| `net_height_*` | start hip height → highest hip height |
| `holds_used_pct` | route holds touched by a hand or foot / route holds seen |

Thresholds can be tuned without re-running the models: `--rejudge` re-evaluates the poses saved in `<output>_climb.json` with the given `ClimbConfig` flags and re-renders.

Meters use the shoulder→hip→knee→ankle length (≈ 0.78 × body height, independent of posture) and `--climber-height` (1.75 m). The video shows the active route, used holds, the `TOP` hold, the skeleton (green dots = limbs on the route, orange = on another hold), a live timer / metrics header and a SUCCESS / FAIL banner. `<output>_climb.json` has the per-frame pose, contacts and state plus the `attempts` summary.

EfficientSAM3 students (the detector runs on a distilled backbone, tracker weights come from the LiteText checkpoint):

```bash
python -m bouldering track --model efficientsam3 --checkpoint checkpoints/efficientsam3_tinyvit.pt \
    --backbone-type tinyvit --model-name 11m --prompt hold \
    --det-score-thresh 0.2 --new-track-thresh 0.25 --stride 3 --max-frames 60

python -m bouldering track --model efficientsam3 --checkpoint checkpoints/efficientsam3_repvit.pt \
    --backbone-type repvit --model-name m1.1 --prompt hold \
    --det-score-thresh 0.2 --new-track-thresh 0.25 --stride 3 --max-frames 60
```

## Layout

```
bouldering/
  __main__.py          command dispatcher (python -m bouldering ...)
  paths.py             default checkpoint / data / output locations
  common/              mask ops (IoU, NMS, packing) and drawing / video writing
  models/              PyTorch model builder, ONNX Runtime sessions, ONNX export
  tracking/            detect-and-track pipeline (PyTorch + ONNX), mask storage, runner / renderer
  routes/              keyframe detection, camera motion, hold registry, route clustering, output reader, renderer
  pose/                ViTPose+ keypoints on person boxes
  climb/               climber tracking (SAM3 person + pose), hold contacts, attempt state machine, metrics, renderer
  cli/                 one module per command (argument parsing only)
```

Reading results from code:

```python
from bouldering.routes import load_routes
from bouldering.tracking import load_frame_masks

routes = load_routes("data/bouldering/output/motion_sam3-s0_torch_routes.json")
hold_ids, route_ids, masks = routes.frame(120)   # masks warped into frame 120, full resolution
```
