# TrafficWatch — traffic events and accident anticipation from a fixed road camera

WIUT Hackathon 2026, Computer Vision track. Given an `.mp4` from the competition camera, TrafficWatch
returns every traffic event as `[start_sec, end_sec, label]` (Part A, 14 classes) and streams a causal
accident-risk score for every frame (Part B).

**Team IntDev** · **Website (team, approach, EDA, sample results, report):** https://bakhti69.github.io/IntDev/ ·
**Live demo:** [Open in Colab](https://colab.research.google.com/github/bakhti69/IntDev/blob/main/demo/colab.ipynb)
(free T4 GPU; *Runtime → Run all*, then open the printed `gradio.live` link) ·
**Predictions on the samples:** [`predictions_samples.json`](predictions_samples.json)

### Results at a glance

| Sample video | Length | Events | Part B alarms | Run time (laptop CPU, no GPU) |
|---|---|---|---|---|
| C3896 | 5:40 | 45 | 3 | 561 s = 1.65 × length |
| C3897 | 5:18 | 45 | 4 | 475 s = 1.49 × length |
| C3902 | 5:18 | 44 | 11 | 489 s = 1.54 × length |
| C3905 | 2:08 | 27 | 3 | 217 s = 1.70 × length |

All within the 3 × budget, format `VALID`, deterministic. On a 67 s clip labelled by us, Score A = **0.705**
(see [Dev set](#dev-set-first-labelled-sample-clip-67-s-8-labelled-events); one clip, one annotator).

```
video ─► YOLO11s ─► tracker ─► kinematics on the junction map ─► 14 class rules ─► segments
            ▲                      ▲            ▲
   scene layout (SIFT-registered)  signal heads   pixel cues (obstacle, fire)
Part B: same detector + causal tracker ─► closest point of approach, hard braking, wrong way ─► risk per frame
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
python run_submission.py --videos samples --out predictions_samples.json --team IntDev
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
| Part B risk | Causal tracker → closest point of approach of every pair (how close and how soon their paths meet), hard braking on a collision course, wrong-way driving → strongest single risk, fast attack / slow release | rule-based |

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
| `red_light` | Front of a moving vehicle crosses a stop line after ≥ 1 s of red on its own vehicle signal head (see signals below) | crossing → leaves the junction |
| `wrong_way` | Moving > 120° against the legal direction of its carriageway for ≥ 1.2 s and ≥ 1.5 body lengths | enters → returns / leaves frame |
| `illegal_u_turn` | Heading (while moving ≥ 0.5 bl/s) reverses ≥ 150° within 20 s over a driven arc of ≥ 2.5 body lengths | starts turning → settles |
| `stopped_vehicle` | Stationary ≥ 15 s in the junction or side road while ≥ 2 vehicles overtake it in its own direction (a signal queue is never overtaken); buses (bus stop) and small far-away vehicles skipped | stops → moves / leaves |
| `jaywalking` | Pedestrian (not a rider/passenger) walking (0.4–1.6 bl/s) with feet on the carriageway outside crosswalks ≥ 1 s | steps on → leaves |
| `failure_to_yield` | Vehicle footprint drives across a crosswalk (≥ 1 bl/s) while a crossing pedestrian is on it within 1.5 vehicle lengths of its path | enters → leaves crossing |
| `illegal_turn` | Movements listed in `configs/scene.json → forbidden_movements` (empty until confirmed → never predicted) | starts → completes turn |
| `solid_line_crossing` | Ground point changes side of a solid lane divider by ≥ 30 % of the box width within 4 s | wheel on line → fully across |
| `stop_line` | Stands still past the stop line, before the junction, ≥ 2 s during red (signal head, or the queue held behind the line where the head faces away) | stops → moves (green) |
| `congestion` | Per direction, the whole flow at a standstill: ≥ 8 vehicles towards the camera / ≥ 5 away, ≥ 75–80 % crawling, for ≥ 30 s | queue stops → clears |
| `road_obstacle` | Animal on the road, or a compact new object that appears, stays ≥ 8 s and is not a detected road user (drift-compensated, shift-tolerant background difference) | appears → removed |
| `fire_smoke` | Flame-coloured blob at a fixed place whose area flickers for ≥ 2 s | first flame → clears |

Post-processing per class: bridge short gaps, drop blips, clip to the video, never overlap within a class
(`src/trafficwatch/segments.py`, parameters in `src/trafficwatch/rules/__init__.py`).

**Signals.** The vehicle head on the median faces the camera and is read directly (traffic leaving up the
avenue); `red_light` is judged only for that approach. The queue approaching the camera has its head on the gantry
facing away, and the head visible on that side is a pedestrian signal (it showed WALK while that queue was
flowing), so no red-light violation is claimed there; `stop_line` for that queue uses vehicles held still behind
the line instead.

### EDA findings that shaped the solution (from `docs/frames`)

* The framing moves between clips — up to (−51, +32) px and ~1° against the reference — so the layout is
  registered per video instead of using fixed pixels.
* Brightness ranges from ~92 (noon) to ~27 (dusk); detection holds up, LED heads flicker and wash out, hence
  loose colour thresholds plus a timeline that bridges dropouts.
* The left-pole head is a pedestrian signal, not the queue's signal; the queue's own head faces away (see above).
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
demo/                        Gradio live demo, Colab notebook, Hugging Face Space bundler
website/                     static team website (GitHub Pages), labeling tool for dev sets
```

## Runtime

Frame sampling follows the video's frame rate: Part A analyses 12.5 frames/s in batches of 8 at 960 px (skipped
frames are grabbed, not decoded); Part B runs the detector at 8 frames/s and holds the score in between. Without a
GPU: 640 px, 10 and 5 frames/s.

Measured with the official harness on the four 4K / 29.97 fps sample videos on a laptop CPU without GPU (CPU
profile): **1.5–1.7 × the video length** (budget 3 ×), see [Results at a glance](#results-at-a-glance). Not yet
timed on a T4 (the GPU profile analyses more frames at 960 px).

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

### Part B on the full sample videos: too many alarms

On the four full videos the alarm (score ≥ 0.5) still started about 5 times a minute. The pairs behind it were
cars whose ground points already coincided in the image (one passing in front of the other) and moving cars
passing a queued car in the next lane, which in this oblique view seem to drive through it; in dense traffic
several weak pairs also added up to an alarm. Fixes: a pair must still have a gap (≥ 0.6 lengths) to count as a
collision course, the closest approach towards a standing road user must be tighter (0.2 instead of 0.35
lengths), and the score is the strongest single pair instead of the combination of all. Alarm starts per
minute: dev clip 5.3 → 0, C3897 5.3 → 1.3; the lead time on simulated crossing, rear-end and head-on
collisions is unchanged (`tests/test_risk.py`).

### Dev set: first labelled sample clip (67 s, 8 labelled events)

Scored with the official `evaluate.py` against our own labels (built with `website/labeler.html`):

| Step | Score A |
|---|---|
| First labelled run | 0.265 |
| red_light only from signal heads that face the camera; pedestrian head dropped (it shows WALK while the queue flows); congestion = the whole direction at a standstill; stopped_vehicle outside the signal queue; jaywalking needs walking | 0.390 |
| No conflicts between tiny far-away road users; obstacles must stay clear of all detected road users | 0.502 |
| stopped_vehicle needs 15 s (10–14 s stops were cars yielding before a turn) | 0.549 |
| Review with the annotator (5 yes/no questions): a U-turn must be one continuous manoeuvre — the "U-turn" of the white truck was an identity switch during a 10 s standstill; the car's U-turn was confirmed and added to the labels | 0.692 |

Per class at the end: congestion, stop_line, stopped_vehicle 1.00; red_light 0.40; jaywalking 0.44;
failure_to_yield 0 (the labelled one is at the bottom frame edge and is missed; at least one of our 3 predictions
is real but was not localised). The confirmed U-turn's boundaries come from our own detection, so its 1.00 is partly
circular: without that class the score is about 0.64. A second video (C3897, 5:18) was reviewed detection by detection (yes/no answers,
`data/dev/review_C3897.md`): border-cut boxes faking U-turns, far-away near-misses, kerbside "stopped vehicles",
pedestrians too far away for failure-to-yield and scooter riders counted as jaywalkers were fixed — 68 → 36 events on
that video, and Score A 0.705 on the labelled clip. One clip and one annotator — a first calibration, not a
validated result. Labels and review notes: `data/dev/`.

## Robustness

* Part A of each video runs in its own child process (`src/trafficwatch/cli.py`). A native crash in the video
  decoder (seen once on a 4K sample: exit code 139) then costs at most that video: the child is retried once with
  single-threaded decoding, and only if that also fails is the video reported with no events.
* Videos are only read front to back (no seeking): the scene is registered on the median of frames from the first
  10 s.

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

* Website: https://bakhti69.github.io/IntDev/ — `website/` is static; `.github/workflows/pages.yml` publishes it
  with GitHub Pages. Team and settings in `website/config.js`.
* Live demo (Gradio, `demo/app.py`):
  * Google Colab, free T4 GPU: open `demo/colab.ipynb` (the website links to it) and *Run all*; it prints a
    public `https://….gradio.live` link. Paste that link into `demo` in `website/config.js` to enable the
    website's upload box while the notebook runs.
  * Locally: `python demo/app.py`, or `TW_SHARE=1 python demo/app.py` for a temporary public link.
  * Hugging Face Space (Gradio Spaces need a paid plan): `bash demo/build_space.sh space/`, push it, and set
    `demo` to `"user/space"`.

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

Team **IntDev**

| Member | Role | Links |
|---|---|---|
| Sadullayeva Mohiraxon | Team captain | [github.com/mokhiraxon555-prog](https://github.com/mokhiraxon555-prog) |
| Parpiyev Baxtiyorjon | | [github.com/bakhti69](https://github.com/bakhti69) |
| Xusenov Shoxrux | | [github.com/antoniobanderes496](https://github.com/antoniobanderes496) |

## Limitations

* Thresholds were calibrated on one labelled 67 s clip and a detection-by-detection review of a second video;
  more labelled footage is needed before the per-class numbers can be trusted.
* `red_light` is not judged for the approach whose signal faces away from the camera.
* `illegal_turn` is never predicted until the prohibited movements of the junction are confirmed (a predicted
  class that never occurs costs a full class in the macro F1).
* Smoke without visible flames is not detected. Near-miss vs. ordinary hard braking is the least certain boundary.
