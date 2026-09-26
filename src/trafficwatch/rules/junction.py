"""Signal-related rules and traffic state: red_light, stop_line, congestion.

Signal state per approach:
* ``signal_head``: its vehicle head faces the camera and is read (signals.py).
  red_light is only reported for such approaches.
* otherwise the head faces away. For stop_line only, red is inferred from
  vehicles standing still at the head of the queue right behind the stop
  line. (Queue inference was too weak for red_light on the sample video:
  turn lanes wait while through lanes flow.)
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
# red_light
RL_MIN_SPEED = 0.5
RL_RED_BEFORE = 1.0         # s of continuous red before the crossing (excludes amber)
# stop_line
STL_MAX_SPEED = 0.12
STL_MIN_DURATION = 2.0
# congestion: direction -> (zones, min vehicles, slow speed, min slow fraction, min duration s).
# A red phase stops the approach while the junction keeps flowing; a jam stops all of it.
CONGESTION = {
    "towards_camera": (("approach_down", "junction_box", "intersection"), 8, 0.3, 0.75, 30.0),
    "away": (("upper",), 5, 0.3, 0.8, 30.0),
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


def _is_red(an: Analysis, approach: Approach, t: float) -> bool:
    head = an.signals.get(approach.signal_head) if approach.signal_head else None
    return bool(head and head.is_readable and head.held("red", t, RL_RED_BEFORE))


def _red_share(an: Analysis, approach: Approach, hold: QueueHold | None, s: float, e: float, tr_id: int) -> float:
    """Share of [s, e] during which the approach has red."""
    if approach.signal_head:
        head = an.signals.get(approach.signal_head)
        return head.fraction("red", s, e) if head and head.is_readable else 0.0
    return hold.holding(s, e, tr_id) if hold is not None else 0.0


def _leave_junction(an: Analysis, tr: TrackData, k: int) -> float:
    zones = an.zones(tr)
    for m in range(k + 1, len(tr.t)):
        if zones[m] not in JUNCTION_ZONES:
            return float(tr.t[m])
    return extend_to_track_end(tr, float(tr.t[-1]))


def red_light(an: Analysis) -> list[Event]:
    events = []
    for approach in an.scene.approaches.values():
        if not approach.signal_head:
            continue
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
                if _is_red(an, approach, float(tr.t[k])):
                    events.append([float(tr.t[k]), _leave_junction(an, tr, k), "red_light"])
                break
    return events


def stop_line(an: Analysis) -> list[Event]:
    events = []
    for approach in an.scene.approaches.values():
        hold = None if approach.signal_head else QueueHold(an, approach)
        for tr in vehicle_tracks(an, two_wheelers=False):
            prog = line_progress(approach, front_point(tr, approach)) / tr.size
            past = (prog > 0.1) & inside(approach.box_zone, tr.pos) & (tr.speed < STL_MAX_SPEED)
            for s, e in runs(tr.t, past, max_gap=1.0):
                if e - s < STL_MIN_DURATION:
                    continue
                if _red_share(an, approach, hold, s, e, tr.id) >= 0.5:
                    events.append([s, e, "stop_line"])
    return events


def congestion(an: Analysis) -> list[Event]:
    n = len(an.times)
    total = {d: np.zeros(n, int) for d in CONGESTION}
    slow = {d: np.zeros(n, int) for d in CONGESTION}
    for tr in vehicle_tracks(an, two_wheelers=False):
        zones = an.zones(tr)
        k_idx = np.searchsorted(an.times, tr.t)
        for k, z in enumerate(zones):
            for d, (group, _, slow_speed, _, _) in CONGESTION.items():
                if z in group:
                    total[d][k_idx[k]] += 1
                    slow[d][k_idx[k]] += int(tr.speed[k] < slow_speed)
    events = []
    for d, (_, min_n, _, min_frac, min_dur) in CONGESTION.items():
        # smooth over ~2 s: a single missed detection must not break the jam
        w = max(1, int(round(2.0 / max(float(np.median(np.diff(an.times))), 1e-3)))) if n > 1 else 1
        tot = np.convolve(total[d], np.ones(w) / w, mode="same")
        slo = np.convolve(slow[d], np.ones(w) / w, mode="same")
        jam = (tot >= min_n) & (slo >= min_frac * np.maximum(tot, 1e-6))
        for s, e in runs(an.times, jam, max_gap=5.0):
            if e - s >= min_dur:
                events.append([s, e, "congestion"])
    return events
