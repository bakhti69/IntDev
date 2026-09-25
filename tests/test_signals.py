"""Signal heads on real stills of the camera, and signal-based rules."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import FPS, make_analysis, make_track, path
from trafficwatch.rules import detect
from trafficwatch.scene import load_scene
from trafficwatch.signals import STATES, SignalTimeline, read_heads

FRAMES = Path(__file__).resolve().parents[1] / "docs" / "frames"


@pytest.mark.parametrize("name, walk, vehicle", [
    ("frame_1.jpg", "green", "green"),
    ("frame_2.jpg", "red", "red"),
    ("frame_4.jpg", "red", "red"),
])
def test_heads_on_real_frames(name, walk, vehicle):
    img = cv2.imread(str(FRAMES / name))
    scene = load_scene(img, img.shape[1], img.shape[0])
    states = read_heads(img, scene.signal_lights, scene.signal_kinds)
    assert states == {"left_pole": walk, "median_pole": vehicle}


def test_timeline_bridges_flicker():
    t = np.arange(0, 10, 1 / FPS)
    states = ["red" if (i % 7) else "unknown" for i in range(len(t))]
    tl = SignalTimeline.build(t, states)
    assert all(STATES[c] == "red" for c in tl.code[t >= 1.0])  # before the first reading it is unknown


def _timeline(duration, state):
    times = np.round(np.arange(0, duration, 1 / FPS) * FPS) / FPS
    return SignalTimeline(times, np.full(len(times), STATES.index(state)))


def test_red_light_from_walk_signal():
    """Nobody queues, but the crosswalk in front shows WALK: crossing is a violation."""
    car = make_track(1, "car", *path(10, 14, (330, 300), (900, 640)))
    an = make_analysis([car], signals={"left_pole": _timeline(60, "green")})
    assert [e[2] for e in detect(an)].count("red_light") == 1


def test_red_light_up_approach_vehicle_head():
    car = make_track(1, "car", *path(5, 9, (1300, 700), (1000, 330)))
    red = make_analysis([car], signals={"median_pole": _timeline(60, "red")})
    green = make_analysis([car], signals={"median_pole": _timeline(60, "green")})
    assert [e[2] for e in detect(red)].count("red_light") == 1
    assert [e[2] for e in detect(green)].count("red_light") == 0
