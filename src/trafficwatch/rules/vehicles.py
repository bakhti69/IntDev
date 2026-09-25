"""Single-vehicle manoeuvre rules: wrong_way, illegal_u_turn, stopped_vehicle,
solid_line_crossing, illegal_turn."""
from __future__ import annotations

import numpy as np

from ..pipeline import Analysis
from ..scene import distance_to_polyline
from ..segments import runs
from ..tracks import TrackData, angle_diff
from .common import Event, extend_to_track_end, inside, vehicle_tracks

# wrong_way
WW_MIN_SPEED = 0.5          # body sizes / s
WW_MAX_COS = -0.5           # heading vs legal direction (<= -0.5: more than 120 deg off)
WW_MIN_DURATION = 1.2       # s
WW_MIN_TRAVEL = 1.5         # body sizes travelled against the flow
# illegal_u_turn
UT_MAX_WINDOW = 20.0        # s to complete the turn
UT_MIN_TURN = np.deg2rad(150)
UT_MIN_PATH = 2.5           # body lengths driven during the turn
UT_MIN_EXCURSION = 1.0      # body lengths away from where the turn starts
UT_MIN_SPEED = 0.5          # heading only counts while really moving
# stopped_vehicle
SV_MAX_SPEED = 0.12
SV_MIN_DURATION = 10.0
SV_MIN_PASSING = 3          # distinct moving vehicles passing it while stopped (queue test)
# solid_line_crossing
SL_SIDE_MARGIN = 0.3        # fraction of box width on each side of the line
SL_MAX_DURATION = 4.0


def wrong_way(an: Analysis) -> list[Event]:
    events = []
    for tr in vehicle_tracks(an):
        unit = tr.vel / (np.linalg.norm(tr.vel, axis=1, keepdims=True) + 1e-9)
        wrong = np.zeros(len(tr.t), bool)
        for poly, direction in an.scene.wrong_way_zones.values():
            in_zone = inside(poly, tr.pos)
            wrong |= in_zone & (tr.speed >= WW_MIN_SPEED) & (unit @ direction <= WW_MAX_COS)
        for s, e in runs(tr.t, wrong, max_gap=0.8):
            i, j = tr.index_at(s), tr.index_at(e)
            travel = np.linalg.norm(tr.pos[j] - tr.pos[i]) / np.median(tr.size[i:j + 1])
            if e - s >= WW_MIN_DURATION and travel >= WW_MIN_TRAVEL:
                events.append([s, extend_to_track_end(tr, e), "wrong_way"])
    return events


def _unwrapped_heading(tr: TrackData, min_speed: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Heading over the samples where the vehicle really moves (jitter of a queued car
    flips its heading at random)."""
    ok = ~np.isnan(tr.heading) & (tr.speed >= min_speed)
    return tr.t[ok], np.unwrap(tr.heading[ok]), np.where(ok)[0]


def _u_turn_span(tr: TrackData, t: np.ndarray, th: np.ndarray, idx: np.ndarray) -> tuple[int, int] | None:
    """(start, end) sample indices into t/th of the first U-turn, or None."""
    lo = 0
    for j in range(len(t)):
        while t[j] - t[lo] > UT_MAX_WINDOW:
            lo += 1
        a = lo + int(np.argmax(np.abs(th[lo:j + 1] - th[j])))
        if abs(th[j] - th[a]) < UT_MIN_TURN:
            continue
        # driven arc: moving samples only, and the vehicle really goes out and comes back
        moving = tr.pos[idx[a:j + 1]]
        size = float(np.median(tr.size[idx[a]:idx[j] + 1]))
        path = np.sum(np.linalg.norm(np.diff(moving, axis=0), axis=1)) / size
        excursion = np.max(np.linalg.norm(moving - moving[0], axis=1)) / size
        if path < UT_MIN_PATH or excursion < UT_MIN_EXCURSION:
            continue
        # extend while the vehicle is still turning, then trim both ends to the turn itself
        end = j
        while end + 1 < len(t) and abs(th[end + 1] - th[end]) > np.deg2rad(10) * (t[end + 1] - t[end]):
            end += 1
        start = next(k for k in range(a, j + 1) if abs(th[k] - th[a]) > np.deg2rad(15))
        end = next(k for k in range(start, end + 1) if abs(th[k] - th[end]) < np.deg2rad(20))
        return start, max(end, start + 1)
    return None


def illegal_u_turn(an: Analysis) -> list[Event]:
    """Heading reverses by >= 150 deg within 20 s while travelling >= 2 body sizes."""
    events = []
    for tr in vehicle_tracks(an):
        t, th, idx = _unwrapped_heading(tr, UT_MIN_SPEED)
        if len(t) < 10:
            continue
        span = _u_turn_span(tr, t, th, idx)
        if span:
            events.append([float(t[span[0]]), float(t[min(span[1], len(t) - 1)]), "illegal_u_turn"])
    return events


def _passing_vehicles(an: Analysis, tr: TrackData, s: float, e: float, radius: float) -> int:
    """Distinct moving vehicles that pass within ``radius`` px of the stopped one."""
    i = tr.index_at(s)
    p = tr.pos[i]
    n = 0
    for other in an.tracks:
        if other.id == tr.id or not (other.is_vehicle or other.is_two_wheeler):
            continue
        m = (other.t >= s) & (other.t <= e)
        if not m.any():
            continue
        near = np.linalg.norm(other.pos[m] - p, axis=1) < radius
        if np.any(near & (other.speed[m] > 1.0)):
            n += 1
    return n


def stopped_vehicle(an: Analysis) -> list[Event]:
    """Stationary >= 10 s on the carriageway while other traffic keeps flowing past.

    Traffic flowing past is what separates a stopped vehicle from a queue at the
    signal (the queue moves as a whole). Buses are skipped: bus-stop dwell.
    """
    events = []
    zones_ok = {"approach_down", "upper", "intersection", "side_road", "junction_box"}
    for tr in vehicle_tracks(an, two_wheelers=False):
        if tr.category == "bus" or tr.duration < SV_MIN_DURATION:
            continue
        zones = an.zones(tr)
        for s, e in runs(tr.t, tr.speed < SV_MAX_SPEED, max_gap=1.5):
            if e - s < SV_MIN_DURATION:
                continue
            i, j = tr.index_at(s), tr.index_at(e)
            on_road = [z in zones_ok for z in zones[i:j + 1]]
            if np.mean(on_road) < 0.8:
                continue
            radius = 3.0 * float(np.median(tr.size[i:j + 1]))
            if _passing_vehicles(an, tr, s, e, radius) < SV_MIN_PASSING:
                continue
            events.append([s, extend_to_track_end(tr, e, slack=2.0), "stopped_vehicle"])
    return events


def solid_line_crossing(an: Analysis) -> list[Event]:
    """Ground point moves from one side of a solid lane line to the other."""
    events = []
    for tr in vehicle_tracks(an):
        half_w = 0.5 * (tr.boxes[:, 2] - tr.boxes[:, 0])
        for line in an.scene.solid_lines.values():
            dist, side = zip(*(distance_to_polyline(line, x, y) for x, y in tr.pos))
            dist, side = np.asarray(dist), np.asarray(side)
            signed = np.full(len(dist), np.nan)
            finite = np.isfinite(dist)
            signed[finite] = dist[finite] * side[finite]
            margin = SL_SIDE_MARGIN * 2 * half_w
            left = signed <= -margin
            right = signed >= margin
            state = np.where(left, -1, np.where(right, 1, 0))
            last_side, last_t = 0, None
            for k in range(len(tr.t)):
                if state[k] == 0 or np.isnan(signed[k]):
                    continue
                if last_side and state[k] != last_side and tr.t[k] - last_t <= SL_MAX_DURATION \
                        and tr.speed[k] > 0.3:
                    # wheel crosses the line ~ when the box edge reaches it
                    seg = np.abs(signed) <= half_w
                    between = (tr.t >= last_t) & (tr.t <= tr.t[k])
                    touch = np.where(seg & between)[0]
                    s = tr.t[touch[0]] if len(touch) else last_t
                    e = tr.t[touch[-1]] if len(touch) else tr.t[k]
                    if e > s:
                        events.append([float(s), float(e), "solid_line_crossing"])
                last_side, last_t = state[k], tr.t[k]
    return events


def illegal_turn(an: Analysis) -> list[Event]:
    """Movements listed as forbidden in the scene config (from-zone -> to-zone)."""
    if not an.scene.forbidden_movements:
        return []
    events = []
    forbidden = set(an.scene.forbidden_movements)
    for tr in vehicle_tracks(an):
        zones = [z for z in an.zones(tr)]
        known = [(k, z) for k, z in enumerate(zones) if z]
        if len(known) < 2:
            continue
        first, last = known[0][1], known[-1][1]
        if (first, last) not in forbidden:
            continue
        t, th, idx = _unwrapped_heading(tr)
        if len(t) < 5:
            continue
        turning = np.abs(np.gradient(th, t)) > np.deg2rad(8)
        segs = runs(t, turning, max_gap=1.0)
        if segs:
            s, e = max(segs, key=lambda x: x[1] - x[0])
            if abs(angle_diff(th[-1], th[0])) > np.deg2rad(45):
                events.append([s, e, "illegal_turn"])
    return events
