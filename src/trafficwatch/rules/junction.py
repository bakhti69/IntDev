"""Signal-related rules and traffic state: red_light, stop_line, congestion.

Signal state per approach:
* ``lights``: read from the signal heads that face the camera (signals.py).
* ``inferred``: the head faces away from the camera, so "red" is inferred
  from behaviour - vehicles standing still at the head of the queue, right
  behind the stop line, before and after the moment in question.
"""
from __future__ import annotations

import numpy as np

from ..pipeline import Analysis
from ..scene import Approach
from ..segments import runs
from ..tracks import TrackData
from .common import Event, extend_to_track_end, front_point, inside, line_progress, vehicle_tracks, within_line_span

JUNCTION_ZONES = {"junction_box", "intersection", "crosswalk_main", "crosswalk_side"}
# queue-head inference
HOLD_MAX_SPEED = 0.15
HOLD_DEPTH = 3.0            # body sizes behind the stop line
HOLD_BEFORE, HOLD_AFTER = 1.5, 2.0
HOLD_MIN_FRACTION = 0.8
# red_light
RL_MIN_SPEED = 0.5
RL_RED_BEFORE = 1.0         # s of continuous red before the crossing (excludes amber)
# stop_line
STL_MAX_SPEED = 0.12
STL_MIN_DURATION = 2.0
# congestion: zone -> (min vehicles, slow speed, min slow fraction, min duration s)
CONGESTION = {
    "approach_down": (6, 0.3, 0.85, 75.0),
    "upper": (5, 0.3, 0.8, 20.0),
}


class QueueHold:
    """Per processed frame: ids of vehicles standing at the head of a queue."""

    def __init__(self, an: Analysis, approach: Approach):
        self.times = an.times
        self.ids: list[set[int]] = [set() for _ in an.times]
        for tr in vehicle_tracks(an, two_wheelers=False):
            prog = line_progress(approach, front_point(tr, approach)) / tr.size
            ok = ((prog > -HOLD_DEPTH) & (prog < 0.3) & (tr.speed < HOLD_MAX_SPEED)
                  & within_line_span(approach, tr.pos))
            for k in np.where(ok)[0]:
                self.ids[int(np.searchsorted(self.times, tr.t[k]))].add(tr.id)

    def holding(self, t0: float, t1: float, exclude: int) -> float:
        """Fraction of frames in [t0, t1] with another vehicle holding at the line."""
        lo, hi = np.searchsorted(self.times, [t0, t1])
        frames = self.ids[lo:hi + 1]
        if not frames:
            return 0.0
        return float(np.mean([bool(ids - {exclude}) for ids in frames]))


def _is_red(an: Analysis, approach: Approach, hold: QueueHold | None, t: float, tr_id: int) -> bool:
    if approach.signal == "lights":
        return an.signal.is_readable and an.signal.red_for(t, RL_RED_BEFORE)
    return hold is not None and hold.holding(t - HOLD_BEFORE, t + HOLD_AFTER, tr_id) >= HOLD_MIN_FRACTION


def _leave_junction(an: Analysis, tr: TrackData, k: int) -> float:
    zones = an.zones(tr)
    for m in range(k + 1, len(tr.t)):
        if zones[m] not in JUNCTION_ZONES:
            return float(tr.t[m])
    return extend_to_track_end(tr, float(tr.t[-1]))


def red_light(an: Analysis) -> list[Event]:
    events = []
    for approach in an.scene.approaches.values():
        hold = QueueHold(an, approach) if approach.signal == "inferred" else None
        for tr in vehicle_tracks(an):
            prog = line_progress(approach, front_point(tr, approach))
            unit = tr.vel / (np.linalg.norm(tr.vel, axis=1, keepdims=True) + 1e-9)
            along = unit @ approach.direction > 0.5
            span = within_line_span(approach, tr.pos)
            for k in range(1, len(tr.t)):
                if not (prog[k - 1] < 0 <= prog[k] and span[k] and along[k] and tr.speed[k] >= RL_MIN_SPEED):
                    continue
                if prog[: k].min() > -0.5 * tr.size[k]:
                    continue  # did not come from behind the line
                if _is_red(an, approach, hold, float(tr.t[k]), tr.id):
                    events.append([float(tr.t[k]), _leave_junction(an, tr, k), "red_light"])
                break
    return events


def stop_line(an: Analysis) -> list[Event]:
    events = []
    for approach in an.scene.approaches.values():
        hold = QueueHold(an, approach) if approach.signal == "inferred" else None
        for tr in vehicle_tracks(an, two_wheelers=False):
            prog = line_progress(approach, front_point(tr, approach)) / tr.size
            past = (prog > 0.1) & inside(approach.box_zone, tr.pos) & (tr.speed < STL_MAX_SPEED)
            for s, e in runs(tr.t, past, max_gap=1.0):
                if e - s < STL_MIN_DURATION:
                    continue
                mid = 0.5 * (s + e)
                if approach.signal == "lights":
                    red = an.signal.is_readable and an.signal.state_at(mid) == "red"
                else:
                    red = hold.holding(s, e, tr.id) >= 0.5
                if red:
                    events.append([s, e, "stop_line"])
    return events


def congestion(an: Analysis) -> list[Event]:
    n = len(an.times)
    total = {z: np.zeros(n, int) for z in CONGESTION}
    slow = {z: np.zeros(n, int) for z in CONGESTION}
    for tr in vehicle_tracks(an, two_wheelers=False):
        zones = an.zones(tr)
        k_idx = np.searchsorted(an.times, tr.t)
        for k, z in enumerate(zones):
            if z in CONGESTION:
                total[z][k_idx[k]] += 1
                slow[z][k_idx[k]] += int(tr.speed[k] < CONGESTION[z][1])
    events = []
    for z, (min_n, _, min_frac, min_dur) in CONGESTION.items():
        jam = (total[z] >= min_n) & (slow[z] >= min_frac * np.maximum(total[z], 1))
        for s, e in runs(an.times, jam, max_gap=5.0):
            if e - s >= min_dur:
                events.append([s, e, "congestion"])
    return events
