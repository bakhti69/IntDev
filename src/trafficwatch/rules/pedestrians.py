"""Pedestrian rules: jaywalking, failure_to_yield."""
from __future__ import annotations

import numpy as np

from ..pipeline import Analysis
from ..segments import runs
from ..tracks import TrackData
from .common import Event, inside, vehicle_tracks

JW_MIN_DURATION = 1.0        # s on the carriageway
JW_MIN_SPEED = 0.4           # body sizes / s: walking, not waiting at the kerb
JW_ROAD_MARGIN = 0.25        # fraction of person height the feet must be inside the road
CW_MARGIN = 0.35             # crosswalk tolerance, fraction of person height
FY_MIN_SPEED = 1.0           # vehicle drives through (not creeping in a jam), body sizes / s
FY_PED_SPEED = 0.5           # the pedestrian is crossing, not waiting at the edge
FY_REACH = 2.5               # pedestrian within this many vehicle sizes of the vehicle
RIDER_OVERLAP = 0.3


def _box_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of box a covered by box b."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return w * h / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))


class _Frames:
    """Boxes of vehicles / two-wheelers per timestamp, to recognise riders and
    people seen through windscreens."""

    def __init__(self, an: Analysis):
        self.boxes: dict[float, list[np.ndarray]] = {}
        for tr in an.tracks:
            if tr.is_vehicle or tr.is_two_wheeler:
                for t, b in zip(tr.t, tr.boxes):
                    self.boxes.setdefault(float(t), []).append(b)

    def covered(self, t: float, box: np.ndarray) -> bool:
        return any(_box_overlap(box, b) > RIDER_OVERLAP for b in self.boxes.get(float(t), ()))


def pedestrians(an: Analysis) -> list[TrackData]:
    frames = _Frames(an)
    out = []
    for tr in an.tracks:
        if not tr.is_person:
            continue
        covered = np.mean([frames.covered(t, b) for t, b in zip(tr.t, tr.boxes)])
        if covered < 0.5:  # riders and passengers are not pedestrians
            out.append(tr)
    return out


def on_carriageway(an: Analysis, tr: TrackData) -> np.ndarray:
    height = tr.boxes[:, 3] - tr.boxes[:, 1]
    depth = np.array([an.scene.road_depth(x, y) for x, y in tr.pos])
    on_road = depth >= JW_ROAD_MARGIN * height
    on_crosswalk = np.zeros(len(tr.t), bool)
    for poly in an.scene.crosswalks.values():
        on_crosswalk |= inside(poly, tr.pos, CW_MARGIN * height)
    return on_road & ~on_crosswalk


def jaywalking(an: Analysis) -> list[Event]:
    events = []
    for tr in pedestrians(an):
        for s, e in runs(tr.t, on_carriageway(an, tr), max_gap=0.8):
            sel = (tr.t >= s) & (tr.t <= e)
            # people waiting at the kerb edge stand still; people crossing walk
            if e - s >= JW_MIN_DURATION and np.median(tr.speed[sel]) >= JW_MIN_SPEED:
                events.append([s, e, "jaywalking"])
    return events


def failure_to_yield(an: Analysis) -> list[Event]:
    """A vehicle drives across a crosswalk while a pedestrian is on it near its path."""
    peds = []
    for tr in pedestrians(an):
        height = tr.boxes[:, 3] - tr.boxes[:, 1]
        for name, poly in an.scene.crosswalks.items():
            peds.append((name, tr, inside(poly, tr.pos, CW_MARGIN * height)))
    events = []
    for veh in vehicle_tracks(an):
        # footprint: ground point plus points half a body length ahead and behind
        unit = veh.vel / (np.linalg.norm(veh.vel, axis=1, keepdims=True) + 1e-9)
        reach = 0.5 * veh.size[:, None] * unit
        for name, poly in an.scene.crosswalks.items():
            on = inside(poly, veh.pos) | inside(poly, veh.pos + reach) | inside(poly, veh.pos - reach)
            for s, e in runs(veh.t, on, max_gap=0.5):
                sel = (veh.t >= s) & (veh.t <= e)
                if np.median(veh.speed[sel]) < FY_MIN_SPEED:
                    continue
                if _pedestrian_near(peds, name, veh, sel):
                    events.append([s, e, "failure_to_yield"])
    return events


def _pedestrian_near(peds, crosswalk: str, veh: TrackData, sel: np.ndarray) -> bool:
    reach = FY_REACH * float(np.median(veh.size[sel]))
    for name, ped, on in peds:
        if name != crosswalk:
            continue
        common, iv, ip = np.intersect1d(veh.t[sel], ped.t, return_indices=True)
        if len(common) == 0:
            continue
        vpos = veh.pos[sel][iv]
        near = on[ip] & (np.linalg.norm(ped.pos[ip] - vpos, axis=1) < reach) & (ped.speed[ip] >= FY_PED_SPEED)
        if near.sum() >= 2:
            return True
    return False
