#!/usr/bin/env python3
"""Build the website's data from the sample videos.

For every video: annotated H.264 playback, background / scene / heat-map /
trajectory / direction images, and a JSON with events, risk curve and EDA.
Events and risk are taken from predictions_samples.json (the official
harness output) so the site shows exactly what was submitted.

    python run_submission.py --videos samples --out predictions_samples.json --team <team>
    python tools/build_site_data.py --videos samples --pred predictions_samples.json --site website
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trafficwatch.api import export_media, process, web_payload  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, help="folder with .mp4 files")
    ap.add_argument("--pred", help="predictions.json from run_submission.py (else computed here)")
    ap.add_argument("--site", default="website")
    # GitHub rejects files over 100 MB: 640 px at 10 fps keeps a 17-minute video around 40-80 MB
    ap.add_argument("--video-width", type=int, default=640, help="annotated video width in px")
    ap.add_argument("--video-fps", type=float, default=10.0, help="annotated video frame rate")
    ap.add_argument("--crf", type=int, default=32, help="x264 quality (higher = smaller file)")
    args = ap.parse_args()

    pred = json.loads(Path(args.pred).read_text())["videos"] if args.pred else {}
    site = Path(args.site)
    (site / "data").mkdir(parents=True, exist_ok=True)
    manifest = []
    videos = sorted(v for v in Path(args.videos).iterdir() if v.suffix.lower() == ".mp4")
    for video in videos:
        print(f"[{video.name}] processing")
        p = pred.get(video.name, {})
        report = process(str(video), events=p.get("events"), risk=p.get("risk") or None)
        stride = max(1, round(report.analysis.info.fps / args.video_fps))
        files = export_media(report, site / "media", video.stem, video_width=args.video_width,
                             video_stride=stride, crf=args.crf)
        mb = (site / "media" / files["video"]).stat().st_size / 1e6
        warn = "  WARNING: over GitHub's 100 MB limit, rerun with a higher --crf" if mb > 95 else ""
        print(f"[{video.name}] annotated video {mb:.1f} MB{warn}")
        payload = web_payload(report, video.name, files)
        (site / "data" / f"{video.stem}.json").write_text(json.dumps(payload, separators=(",", ":")))
        manifest.append({"name": video.name, "stem": video.stem, "duration": report.analysis.info.duration,
                         "n_events": len(report.events)})
        print(f"[{video.name}] {len(report.events)} events -> {site / 'data' / (video.stem + '.json')}")
    (site / "data" / "videos.json").write_text(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
