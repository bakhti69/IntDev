"""Part A for one video in a child process: ``python -m trafficwatch.cli VIDEO OUT_JSON``.

A native crash inside the video decoder (seen once on a 4K sample) would kill the
whole harness and empty every video; isolating each video in its own process
confines it to that video, and a retry with single-threaded decoding recovers it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]


def run_isolated(video_path: str) -> list[list]:
    """Events for one video, computed in a child process (retried once, single-threaded)."""
    attempts = [{}, {"OPENCV_FFMPEG_CAPTURE_OPTIONS": "threads;1"}]
    for extra in attempts:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "events.json"
            path = os.pathsep.join(filter(None, [str(SRC), os.environ.get("PYTHONPATH")]))
            env = {**os.environ, **extra, "PYTHONPATH": path}
            proc = subprocess.run([sys.executable, "-m", "trafficwatch.cli", video_path, str(out)], env=env)
            if proc.returncode == 0 and out.exists():
                return json.loads(out.read_text())
            print(f"[trafficwatch] analysis of {Path(video_path).name} exited with {proc.returncode}"
                  f"{', retrying with single-threaded decoding' if not extra else ''}", file=sys.stderr)
    return []


def main() -> int:
    from .config import Settings
    from .pipeline import analyze
    from .rules import detect

    video, out = sys.argv[1], sys.argv[2]
    Path(out).write_text(json.dumps(detect(analyze(video, Settings()))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
