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
    args = ap.parse_args()

    pred = json.loads(Path(args.pred).read_text())["videos"] if args.pred else {}
    site = Path(args.site)
    (site / "data").mkdir(parents=True, exist_ok=True)
    manifest = []
    for video in sorted(Path(args.videos).glob("*.mp4")):
        print(f"[{video.name}] processing")
        p = pred.get(video.name, {})
        report = process(str(video), events=p.get("events"), risk=p.get("risk") or None)
        files = export_media(report, site / "media", video.stem)
        payload = web_payload(report, video.name, files)
        (site / "data" / f"{video.stem}.json").write_text(json.dumps(payload, separators=(",", ":")))
        manifest.append({"name": video.name, "stem": video.stem, "duration": report.analysis.info.duration,
                         "n_events": len(report.events)})
        print(f"[{video.name}] {len(report.events)} events -> {site / 'data' / (video.stem + '.json')}")
    (site / "data" / "videos.json").write_text(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
