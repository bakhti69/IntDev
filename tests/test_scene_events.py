"""Pixel-cue rules (road_obstacle, fire_smoke) on synthetic low-res frames."""
from __future__ import annotations

import cv2
import numpy as np

from conftest import make_analysis
from trafficwatch.pipeline import CUE_PERIOD, fire_mask
from trafficwatch.rules.scene_events import fire_smoke, road_obstacle

CUE_W, CUE_H = 384, 213


def _fill_cues(an, frames_bgr, boxes=None):
    cues = an.cues
    for k, f in enumerate(frames_bgr):
        cues.t.append(k * CUE_PERIOD)
        cues.gray.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        cues.sat.append(cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[..., 1])
        cues.fire.append(fire_mask(f))
        cues.boxes.append(np.zeros((0, 4), np.float32) if boxes is None else boxes[k])
    return an


def _asphalt(rng):
    base = np.full((CUE_H, CUE_W, 3), 95, np.uint8)
    return np.clip(base + rng.normal(0, 3, base.shape), 0, 255).astype(np.uint8)


def test_obstacle_appears_and_stays():
    rng = np.random.default_rng(0)
    frames = []
    for k in range(160):                       # 80 s of cues
        f = _asphalt(rng)
        if k * CUE_PERIOD >= 50:              # a box falls on the intersection at t=50 s
            cv2.rectangle(f, (250, 150), (262, 160), (200, 200, 205), -1)
        frames.append(f)
    ev = road_obstacle(_fill_cues(make_analysis([], duration=80), frames))
    assert len(ev) >= 1 and 48 <= ev[0][0] <= 53


def test_no_obstacle_on_empty_road():
    rng = np.random.default_rng(1)
    frames = [_asphalt(rng) for _ in range(160)]
    assert road_obstacle(_fill_cues(make_analysis([], duration=80), frames)) == []


def test_flickering_fire_detected_static_orange_ignored():
    rng = np.random.default_rng(2)
    fire, static = [], []
    for k in range(40):
        f = _asphalt(rng)
        r = int(6 + 4 * rng.random())          # flame size flickers
        cv2.circle(f, (250, 150), r, (40, 140, 255), -1)
        fire.append(f)
        g = _asphalt(rng)
        cv2.rectangle(g, (245, 145), (255, 155), (40, 140, 255), -1)   # orange box, steady
        static.append(g)
    assert len(fire_smoke(_fill_cues(make_analysis([], duration=20), fire))) == 1
    assert fire_smoke(_fill_cues(make_analysis([], duration=20), static)) == []
