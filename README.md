# TrafficWatch — traffic events and accident anticipation from a fixed road camera

WIUT Hackathon 2026, Computer Vision track. Given an `.mp4` from the competition camera, TrafficWatch
returns every traffic event as `[start_sec, end_sec, label]` (Part A, 14 classes) and streams a causal
accident-risk score for every frame (Part B).

```
video ─► YOLO11s ─► tracker ─► kinematics on the junction map ─► 14 class rules ─► segments
            ▲                      ▲            ▲
   scene layout (SIFT-registered)  signal heads   pixel cues (obstacle, fire)
Part B: same detector + causal tracker ─► time-to-collision, hard braking, wrong way ─► risk per frame
```

## Install and run

```bash
pip install -r requirements.txt
bash weights/download.sh          # only if weights/yolo11s.pt is missing (it is committed); verifies the checksum
python run_submission.py --videos /data/test --out predictions.json
python evaluate.py --pred predictions.json --validate-only
```

* Python ≥ 3.10. `run_submission.py` and `evaluate.py` are the organisers' files, unchanged.
* No internet is needed at run time (the Ultralytics hub is switched off with `YOLO_OFFLINE=1`).
* GPU is used automatically when available. Without a GPU the settings drop to a lighter profile
  (640 px input, fewer analysed frames) — see [Runtime](#runtime).
* Environment overrides for ablations: `TW_IMGSZ`, `TW_RATE` / `TW_STRIDE`, `TW_RISK_RATE` / `TW_RISK_STRIDE`,
  `TW_RISK_IMGSZ`, `TW_BATCH`, `TW_DEVICE`, `TW_WEIGHTS` (`src/trafficwatch/config.py`).

### Reproduce `predictions_samples.json` and the website data

```bash
python run_submission.py --videos samples --out predictions_samples.json --team trafficwatch
pip install -r requirements-tools.txt
python tools/build_site_data.py --videos samples --pred predictions_samples.json --site website
python tools/frame_eda.py --frames docs/frames --site website
```

### Check the scene layout on a new video

```bash
python tools/draw_scene.py --video samples/<clip>.mp4 --out scene.jpg
```

## Approach

### What is learned and what is rule-based

| Stage | Method | Learned? |
|---|---|---|
| Road-user detection | YOLO11s, COCO-pretrained Ultralytics weights, no fine-tuning (person, bicycle, car, motorcycle, bus, truck, animals) | **learned** (pre-trained, not trained by us) |
| Tracking | ByteTrack-style two-stage IoU association, constant-velocity prediction, category groups, longer memory for parked objects, offline stitching of parked fragments | rule-based |
| Scene map | Carriageways, islands, crosswalks, stop lines, lane lines, lane directions, signal heads drawn once on `configs/scene_ref.jpg`, registered to each video with SIFT + RANSAC (similarity transform) | rule-based |
| Signal state | Lit red / amber / green pixels in each signal-head box, timeline with flicker bridging and a 1 s mode filter | rule-based |
| 14 event classes | Rules on trajectories (below) | rule-based |
| Part B risk | Causal tracker → closest point of approach of every pair (how close and how soon their paths meet), hard braking on a collision course, wrong-way driving → `1 − Π(1 − rᵢ)`, fast attack / slow release | rule-based |

Why rules: there are no labels for this camera, and the hidden test set contains events that are not in
the samples. A classifier trained on the samples could not have seen them; rules written from the class
definitions (including the annotators' start/end conventions) do not need examples.

Speeds are measured in **body lengths per second** (image speed divided by √(box area)), which removes most of
the perspective effect: the same thresholds hold in the foreground and at the far end of the avenue.

### Classes

| Class | Rule | Segment |
|---|---|---|
| `accident` | Contact (ground distance < 0.8 body lengths, boxes overlap) after approaching from ≥ 1.5 at ≥ 1.5 bl/s; an impact stop (speed drops to ≤ 20 % across ~1 s) or a ≥ 45° deflection; then both at rest together ≥ 2 s | first contact → both at rest |
| `near_miss` | TTC < 1 s on a real collision course (closest point of approach ≤ 0.5 lengths, not side-by-side passing), closest gap 0.8–1.8 (no contact), emergency braking or a ≥ 60°/s swerve | evasive action → gap > 2.5 |
| `red_light` | Front of the vehicle crosses a stop line while its approach is red (see signals below) | crossing → leaves the junction |
| `wrong_way` | Moving > 120° against the legal direction of its carriageway for ≥ 1.2 s and ≥ 1.5 body lengths | enters → returns / leaves frame |
| `illegal_u_turn` | Heading (while moving ≥ 0.5 bl/s) reverses ≥ 150° within 20 s over a driven arc of ≥ 2.5 body lengths | starts turning → settles |
| `stopped_vehicle` | Stationary ≥ 10 s on the carriageway while ≥ 3 moving vehicles pass it; buses skipped (bus stop) | stops → moves / leaves |
| `jaywalking` | Pedestrian (not a rider/passenger) with feet on the carriageway outside crosswalks ≥ 1 s | steps on → leaves |
| `failure_to_yield` | Vehicle footprint crosses a crosswalk at speed while a pedestrian is on it within 4 body lengths | enters → leaves crossing |
| `illegal_turn` | Movements listed in `configs/scene.json → forbidden_movements` (empty until confirmed → never predicted) | starts → completes turn |
| `solid_line_crossing` | Ground point changes side of a solid lane divider by ≥ 30 % of the box width within 4 s | wheel on line → fully across |
| `stop_line` | Stands still past the stop line, before the junction, ≥ 2 s during red | stops → moves (green) |
| `congestion` | Per direction: ≥ 5–6 vehicles, ≥ 80 % crawling, for 20 s (away flow) / 75 s (signal queue) | queue stops → clears |
| `road_obstacle` | Animal on the road, or a compact new object that appears, stays ≥ 5 s and is not a detected road user (drift-compensated, shift-tolerant background difference) | appears → removed |
| `fire_smoke` | Flame-coloured blob at a fixed place whose area flickers for ≥ 2 s | first flame → clears |

Post-processing per class: bridge short gaps, drop blips, clip to the video, never overlap within a class
(`src/trafficwatch/segments.py`, parameters in `src/trafficwatch/rules/__init__.py`).

**Signals.** The vehicle head on the median faces the camera and is read directly (traffic leaving up the
avenue). The queue approaching the camera has its head on the gantry facing away, so its red phase is inferred:
the pedestrian head of the crosswalk in front of it shows WALK, or vehicles are held still at the stop line
before and after the moment in question.

### EDA findings that shaped the solution (from `docs/frames`)

* The framing moves between clips — up to (−51, +32) px and ~1° against the reference — so the layout is
  registered per video instead of using fixed pixels.
* Brightness ranges from ~92 (noon) to ~27 (dusk); detection holds up, LED heads flicker and wash out, hence
  loose colour thresholds plus a timeline that bridges dropouts.
* The left-pole head is a pedestrian signal; the queue's own head faces away (see above).
* Pedestrians cut diagonally through the junction; people stand on the islands and the median, which are
  excluded from the carriageway.

### Code map

```
solution.py                  interface for the harness (thin wrapper)
src/trafficwatch/
  config.py                  settings, GPU/CPU profiles, seeds
  video.py, detector.py      frame reading, YOLO wrapper
  tracker.py, tracks.py      tracking and kinematics (body lengths / s)
  scene.py, signals.py       layout + registration, signal heads
  pipeline.py                one pass: detection → tracks → cues  (Analysis)
  rules/                     one module per family of classes; detect() = all rules + clean-up
  risk.py                    Part B (causal)
  render.py, viz.py, eda.py  annotated videos, drawings, EDA statistics
  api.py                     shared entry points for the website builder and the demo
tools/                       draw_scene.py, frame_eda.py, build_site_data.py
tests/                       every rule on synthetic trajectories drawn on the real layout; signals on real frames
demo/                        Gradio live demo + Hugging Face Space bundler
website/                     static team website (GitHub Pages), labeling tool for dev sets
```

## Runtime

Frame sampling follows the video's frame rate: Part A analyses 12.5 frames/s in batches of 8 at 960 px (skipped
frames are grabbed, not decoded); Part B runs the detector at 8 frames/s and holds the score in between. Without a
GPU: 640 px, 10 and 5 frames/s.

Measured on a 4-core CPU without GPU (CPU profile), 1080p / 25 fps / 30 s clip: Part A 26.6 s + Part B 12.0 s =
**38.6 s (1.3 × duration, budget 3 ×)**. Not yet measured on a T4.

### Lessons from the first sample video (dense, jammed traffic)

A first run on a sample clip reported 4 "accidents", 5 U-turns and a risk score above 0.5 half of the time.
Diagnosis and fixes:

* Accidents were queue stops and neighbouring lanes overlapping in perspective → an impact now needs an abrupt
  stop from real speed and both road users at rest together afterwards.
* U-turns were headings of queued cars flipping with box jitter → headings only count while moving, over a driven arc.
* The risk fired on cars passing a bus at the stop: centre-to-centre TTC treats side-by-side passing as a collision
  course → closest point of approach, measured in the smaller road user's lengths.
* Frame sampling now follows the video's frame rate instead of assuming 25 fps.

Result on the same clip: 10 events instead of 19, no accidents, alarm share 49 % → 1.8 %.

## Determinism

Seeds are fixed (`random`, NumPy, OpenCV RANSAC, PyTorch; cuDNN deterministic, benchmark off). Two runs of the
harness on the same machine gave identical events and an identical risk curve. Differences between machines can
come from GPU kernels (FP16 on CUDA) and from OpenCV/FFmpeg decoders.

## Development

```bash
pip install -r requirements-tools.txt
pytest -q && ruff check .
```

Build a dev set with the labeling page (`website/labeler.html`, runs locally in the browser), export
`ground_truth.json`, then `python evaluate.py --pred predictions_samples.json --gt ground_truth.json --per-video`.

## Website and live demo

* Website: `website/` is static; `.github/workflows/pages.yml` publishes it with GitHub Pages. Fill in the team in
  `website/config.js`.
* Live demo: `bash demo/build_space.sh space/` builds a Hugging Face Space (Gradio, CPU); push it and set
  `space` in `website/config.js`. Locally: `python demo/app.py`.

## Datasets, models and licences

| Item | Use | Licence |
|---|---|---|
| COCO 2017 (through the pre-trained weights) | detector pre-training, not re-trained by us | CC BY 4.0 (annotations) |
| YOLO11s weights, Ultralytics | detector | AGPL-3.0 |
| Organisers' sample frames (`docs/frames`, `configs/scene_ref.jpg`) | scene layout, EDA, tests | competition use |
| Public test clips (intel-iot-devkit sample-videos) | smoke tests during development only, not shipped | CC BY 4.0 |

Open-source code: Ultralytics (AGPL-3.0), OpenCV (Apache-2.0), NumPy / SciPy (BSD), Gradio and
`@gradio/client` (Apache-2.0). The tracker follows the ByteTrack idea (Zhang et al., 2022; MIT) but is our own
implementation. Because Ultralytics is AGPL-3.0, this project is distributed under compatible terms.

## Team

<!-- TODO(team): real names, roles and contributions; keep in sync with website/config.js -->
| Member | Role | Contributions |
|---|---|---|
| Member 1 | Detection & tracking | detector, tracker, registration, runtime |
| Member 2 | Event rules & evaluation | rules, dev labels, error analysis |
| Member 3 | Anticipation, website & demo | Part B, website, demo, EDA |

## Limitations

* Thresholds were tuned on synthetic trajectories and public clips, not on labelled footage from this camera —
  label the samples and tune per class at IoU 0.7 before relying on the numbers.
* `illegal_turn` is never predicted until the prohibited movements of the junction are confirmed (a predicted
  class that never occurs costs a full class in the macro F1).
* Smoke without visible flames is not detected. Near-miss vs. ordinary hard braking is the least certain boundary.
