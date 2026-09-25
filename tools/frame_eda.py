#!/usr/bin/env python3
"""EDA on still frames of the camera (works before/without the videos).

Per frame: registration offset against the reference layout, brightness /
contrast, detections by class and by road zone, signal-head state, and an
annotated image (layout + detections).

    python tools/frame_eda.py --frames docs/frames --site website
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trafficwatch.config import Settings  # noqa: E402
from trafficwatch.detector import get_detector  # noqa: E402
from trafficwatch.scene import load_scene  # noqa: E402
from trafficwatch.signals import read_heads  # noqa: E402
from trafficwatch.viz import draw_scene, draw_tracks  # noqa: E402


def analyse_frame(path: Path, detector) -> tuple[dict, np.ndarray]:
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    scene = load_scene(img, w, h)
    dets = detector([img])[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    by_class: dict[str, int] = {}
    by_zone: dict[str, dict[str, int]] = {}
    for box, cat in zip(dets.boxes, dets.cats):
        by_class[cat] = by_class.get(cat, 0) + 1
        zone = scene.zone_of((box[0] + box[2]) / 2, box[3]) or "off_road"
        by_zone.setdefault(zone, {}).setdefault(cat, 0)
        by_zone[zone][cat] += 1
    ref_w = 1662.0
    stats = {
        "name": path.name, "width": w, "height": h,
        "registered": scene.registered, "inliers": scene.inliers,
        "stop_line_px": scene.approaches["down"].stop_line.round(1).tolist(),
        "offset_vs_reference": (scene.approaches["down"].stop_line[0] - np.array([250, 446]) * w / ref_w)
        .round(1).tolist(),
        "brightness": round(float(gray.mean()), 1), "contrast": round(float(gray.std()), 1),
        "signals": read_heads(img, scene.signal_lights, scene.signal_kinds),
        "detections": by_class, "by_zone": by_zone,
    }
    vis = draw_scene(img, scene, alpha=0.2)
    draw_tracks(vis, [(i + 1, b, c) for i, (b, c) in enumerate(zip(dets.boxes, dets.cats))])
    return stats, vis


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="docs/frames")
    ap.add_argument("--site", default="website")
    args = ap.parse_args()
    s = Settings()
    detector = get_detector(s.weights, s.imgsz, s.device_or_none)
    out_media = Path(args.site) / "media" / "frames"
    out_media.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(Path(args.frames).glob("*.jpg")):
        stats, vis = analyse_frame(path, detector)
        vis = cv2.resize(vis, None, fx=960 / vis.shape[1], fy=960 / vis.shape[1], interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_media / f"{path.stem}_annotated.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 82])
        raw = cv2.imread(str(path))
        raw = cv2.resize(raw, None, fx=960 / raw.shape[1], fy=960 / raw.shape[1], interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_media / f"{path.stem}.jpg"), raw, [cv2.IMWRITE_JPEG_QUALITY, 82])
        stats["image"] = f"media/frames/{path.stem}_annotated.jpg"
        stats["raw"] = f"media/frames/{path.stem}.jpg"
        rows.append(stats)
        print(json.dumps({k: stats[k] for k in ("name", "registered", "inliers", "offset_vs_reference",
                                                "brightness", "signals", "detections")}))
    (Path(args.site) / "data").mkdir(parents=True, exist_ok=True)
    (Path(args.site) / "data" / "frames.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
