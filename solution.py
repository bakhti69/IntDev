"""
solution.py - entry point imported by the organizers' harness (run_submission.py).

    detect_events(video_path)  -> [[start_sec, end_sec, label], ...]    # Part A
    RiskEstimator().reset(meta); .step(frame, t_sec) -> float           # Part B

The implementation lives in src/trafficwatch: YOLO11 detection -> tracking ->
scene layout registered onto the video -> per-class rules on trajectories.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from trafficwatch.config import Settings  # noqa: E402
from trafficwatch.pipeline import analyze  # noqa: E402
from trafficwatch.risk import CausalRisk  # noqa: E402
from trafficwatch.rules import detect  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# Official class ids (14). All of them have a rule; see README for how each is detected.
CLASSES: list[str] = [
    "accident",
    "near_miss",
    "red_light",
    "wrong_way",
    "illegal_u_turn",
    "stopped_vehicle",
    "jaywalking",
    "failure_to_yield",
    "illegal_turn",
    "solid_line_crossing",
    "stop_line",
    "congestion",
    "road_obstacle",
    "fire_smoke",
]

RISK_HORIZON_SEC = 5.0


def detect_events(video_path: str) -> list[list]:
    """Part A - traffic event detection for one .mp4."""
    return detect(analyze(video_path, Settings()))


class RiskEstimator:
    """Part B - causal accident anticipation. Sees frames in order, nothing else."""

    def reset(self, meta: dict) -> None:
        self.meta = meta
        self.settings = Settings()
        self.stride = max(1, self.settings.risk_stride)
        self.frame_idx = 0
        self.last_score = 0.0
        if not hasattr(self, "_risk"):
            self._risk = CausalRisk(self.settings)
        self._risk.reset()

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        if self.frame_idx % self.stride == 0:
            self.last_score = self._risk.update(frame, t_sec)
        self.frame_idx += 1
        return self.last_score
