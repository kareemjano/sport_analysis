# climb pipeline

How `python -m bouldering climb` works: which passes run over the video, and when holds, routes, the climber and the climber's keypoints are computed. For the judging rules and metrics, see [Climb: judging attempts](../README.md#climb-judging-attempts).

## Overview

`climb` makes two passes over the video, plus a render pass ([cli/climb.py](../cli/climb.py)):

```
pipeline = build_pipeline(args)   # one SAM3 model, shared by both loops
if <stem>_routes.json is missing (or --rerun-routes):
    run_routes(...)               # loop 1: the wall  -> <stem>_routes.json + <stem>_routes/keyframe_*.npz
routes = RouteData(routes_json)
run_climb(...)                    # loop 2: the climber -> <stem>_climb.json
render_climb(...)                 # draw everything -> <output>.mp4
```

- **Loop 1** ([routes/pipeline.py](../routes/pipeline.py), `run_routes`) maps the wall: holds, their routes, and the camera motion.
- **Loop 2** ([climb/pipeline.py](pipeline.py), `run_climb`) follows the climber on that map and judges the attempts.

The routes run first because loop 2 needs the whole wall map. It uses the map to pick the climber over other people, to find which hold each hand or foot is on, and to know each route's top hold.

Loop 1 is skipped when its JSON already exists, so a second `climb` run with the same options only repeats loop 2. Hold detections are also cached per frame in `<stem>_detections/`, so `--rerun-routes` (for example, after changing route options) skips SAM3. `--rejudge` re-runs only the judging on the saved poses (no models), and `--render-only` only renders.

## Loop 1: holds and routes

This loop visits every `--stride`-th frame. Holds don't move, so SAM3 doesn't run on every frame. Most frames only get a camera-motion estimate; SAM3 runs only on keyframes.

**Every processed frame (cheap, no neural net)**

- The frame is shrunk to `work_scale` (0.5) and ORB features are computed ([routes/motion.py](../routes/motion.py)).
- A homography from the current keyframe to this frame is estimated, directly against the keyframe, or chained through the previous frame if that fails.
- Only the homography `H` is stored. This frame's hold masks are the keyframe masks warped with `H`.

**Keyframes (expensive)**

A frame becomes a keyframe when one of these holds:

| reason | condition |
|---|---|
| `start` | first frame |
| `motion-lost` | no plausible homography (cut, fast pan) |
| `new-area` | more than `--new-area` (15%) of the frame shows wall the keyframe didn't see |
| `periodic` | `--max-keyframe-interval` frames since the last keyframe (0 = off) |

On a keyframe:

1. **Hold detection**: SAM3 runs with the `--prompt` text (default `"climbing hold"`). Only the backbone and detector heads run, not the tracker ([routes/detector.py](../routes/detector.py)).
2. **Hold identity**: the previous keyframe's holds are warped into this frame and matched to the new detections by mask IoU with Hungarian matching ([routes/holds.py](../routes/holds.py), `update_holds`). A matched detection keeps its hold id. A hold with no detection is kept while it is mostly in view (the climber may cover it), until it has been missed for `--max-missed-keyframes`. Unmatched detections become new holds.
3. **Route clustering** ([routes/clustering.py](../routes/clustering.py)): each hold gets its median CIELAB color and its centroid (`hold_features`).
   - At the start, or after a cut, `cluster_routes` runs single-linkage clustering over a combined color and position distance. Vertical gaps count less than horizontal ones because routes run up the wall. Clusters smaller than `--min-route-holds` are marked unassigned. `stable_route_ids` then reuses the ids of previous routes that share holds.
   - On the same wall, `extend_routes` runs instead. Known holds keep their route; only new or unassigned holds are attached to the nearest route or clustered among themselves. This keeps a route from changing while someone is climbing it.
4. Masks and route ids are saved to `keyframe_XXXXXX.npz`.

So holds and routes change only on keyframes. Every other frame is stored as a keyframe index plus a homography.

## Loop 2: the climber

This loop visits only the frames in the routes output, so it uses the same stride as loop 1. Each frame goes through three steps.

### Person detection and keypoints

`ClimberTracker` ([climb/climber.py](climber.py)) keeps a box around the climber from one frame to the next:

- **Keypoints (ViTPose+)** are computed on every processed frame. Normally ViTPose+ runs on a box built from the previous frame's keypoints, without running SAM3.
- **Person detection (SAM3, `--person-prompt` `"person"`)** runs only when one of these is true:
  - there is no track yet, or the track was lost;
  - the pose looks unreliable (fewer than 6 keypoints at or above `--kpt-thresh`, or a mean score below 0.4);
  - `--redetect-seconds` (2 s) have passed since the last detection.

  These frames are tagged `(SAM3 detect)` in the log.
- **Picking the climber**: the box with the highest IoU with the current track is preferred. For a new climber, the code takes the largest person overlapping the wall box, which is the area covered by the hold centroids from loop 1 (`_wall_box` in [pipeline.py](pipeline.py)). This keeps it from following spectators.

Person detection reuses the SAM3 pipeline from loop 1 with a different text prompt.

### Contacts

- `limb_points` ([limbs.py](limbs.py)) turns the keypoints into hand and foot points. Each one is extended past the wrist (`--hand-ext`) or ankle (`--foot-ext`).
- `find_contacts` ([contacts.py](contacts.py)) maps each point into the keyframe's work coordinates using the inverse of loop 1's homography. A limb is on the nearest hold mask if it is within `--contact-radius` × body length.

### Attempt judging

`AttemptTracker` ([attempt.py](attempt.py)) is a state machine: `searching → climbing(route) → success / fail → searching`. It starts an attempt when enough limbs stay on one route's holds. It ends the attempt with a success when the hands hold the route's top hold, or with a fail on a fall, a lost climber or route, or the end of the video. It re-arms once the climber is off the wall. The thresholds are listed in [Climb: judging attempts](../README.md#climb-judging-attempts). Metrics are computed in [metrics.py](metrics.py).

## Summary

| step | when | how |
|---|---|---|
| Hold detection | loop 1, keyframes only | SAM3, `"climbing hold"` |
| Hold positions between keyframes | loop 1, every processed frame | ORB + RANSAC homography, keyframe masks warped |
| Route clustering | loop 1, keyframes only (full at the start or after a cut, otherwise extended) | single linkage on color and position |
| Person detection | loop 2, when there's no track, the pose is lost, or every `--redetect-seconds` | SAM3, `"person"` |
| Keypoints | loop 2, every processed frame | ViTPose+ |
| Contacts and attempt state | loop 2, every processed frame | geometry only |
