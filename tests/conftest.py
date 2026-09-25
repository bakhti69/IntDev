"""Synthetic scenes: hand-made trajectories on the real layout (reference-frame
coordinates), so every rule can be tested without video or a detector."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trafficwatch.pipeline import Analysis, SceneCues  # noqa: E402
from trafficwatch.scene import load_scene  # noqa: E402
from trafficwatch.signals import SignalTimeline  # noqa: E402
from trafficwatch.tracks import TrackHistory, finalize  # noqa: E402
from trafficwatch.video import VideoInfo  # noqa: E402

FPS = 12.5
W, H = 1662, 924
SIZES = {"car": (90, 60), "bus": (200, 110), "person": (22, 55), "motorcycle": (30, 45)}


def make_track(tid: int, category: str, times, points, size_scale: float = 1.0):
    """Track whose ground point follows ``points`` (N, 2) at ``times`` (N,)."""
    w, h = (s * size_scale for s in SIZES[category])
    hist = TrackHistory(tid)
    for t, (x, y) in zip(times, points):
        hist.add(float(t), np.array([x - w / 2, y - h, x + w / 2, y], np.float32), category)
    return finalize(hist)


def path(t0: float, t1: float, p0, p1, ease=None):
    """Straight-line motion between two points, optionally with a speed profile."""
    t = np.round(np.arange(t0, t1 + 1e-9, 1 / FPS) * FPS) / FPS
    u = (t - t0) / max(t1 - t0, 1e-9)
    if ease is not None:
        u = ease(u)
    pts = np.asarray(p0, float)[None] + u[:, None] * (np.asarray(p1, float) - np.asarray(p0, float))[None]
    return t, pts


def still(t0: float, t1: float, p):
    t = np.round(np.arange(t0, t1 + 1e-9, 1 / FPS) * FPS) / FPS
    return t, np.repeat(np.asarray(p, float)[None], len(t), axis=0)


def concat(*parts):
    t = np.concatenate([p[0] for p in parts])
    pts = np.concatenate([p[1] for p in parts])
    _, keep = np.unique(t, return_index=True)
    return t[keep], pts[keep]


def make_analysis(tracks, duration: float = 60.0, signal=None) -> Analysis:
    times = np.round(np.arange(0, duration, 1 / FPS) * FPS) / FPS
    sig = signal or SignalTimeline(times, np.zeros(len(times), int))
    return Analysis(
        video_path="synthetic.mp4",
        info=VideoInfo(fps=25.0, n_frames=int(duration * 25), width=W, height=H),
        scene=load_scene(None, W, H),
        times=times, tracks=tracks, signal=sig,
        cues=SceneCues(scale=384 / W), counts={},
    )


@pytest.fixture
def analysis_factory():
    return make_analysis
