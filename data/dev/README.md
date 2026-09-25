# Dev set

`ground_truth.json`: our own labels of one sample clip (`C3905.mp4`, 1:00–2:07 of the sample video, re-encoded to
1280×720 H.264 so browsers can play it), made with `website/labeler.html` using the organisers' start/end
conventions. Score with:

```bash
python run_submission.py --videos <folder with the clip> --out dev_pred.json
python evaluate.py --pred dev_pred.json --gt data/dev/ground_truth.json --per-video
```

Review of the detections with the annotator:

* 27–40 s, white truck: **not** a U-turn (tracker identity switch while it stood at the median tip).
* 52–56 s, car turning around into the avenue: U-turn **confirmed**; boundaries taken from the detection.
* 41–42 s, red_light on the away flow: **not** a violation.
* 41–51 s: at least one failure_to_yield happens (not localised, so not in the labels).
* 20–25 s, failure_to_yield on the bottom-left crossing: the car is at the bottom frame edge, where its box is cut
  off, and is missed.

One clip and one annotator: a first calibration, not a benchmark.
