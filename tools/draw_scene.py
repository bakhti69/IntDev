#!/usr/bin/env python3
"""Overlay the registered scene layout on a frame, to check configs/scene.json.

    python tools/draw_scene.py --image docs/frames/frame_1.jpg --out scene.jpg
    python tools/draw_scene.py --video samples/clip.mp4 --out scene.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trafficwatch.scene import load_scene  # noqa: E402
from trafficwatch.video import background_frame  # noqa: E402
from trafficwatch.viz import draw_scene  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image")
    src.add_argument("--video")
    ap.add_argument("--out", default="scene_overlay.jpg")
    args = ap.parse_args()

    frame = cv2.imread(args.image) if args.image else background_frame(args.video)
    if frame is None:
        print("could not read input", file=sys.stderr)
        return 1
    scene = load_scene(frame, frame.shape[1], frame.shape[0])
    print(f"registered={scene.registered} inliers={scene.inliers}")
    cv2.imwrite(args.out, draw_scene(frame, scene))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
