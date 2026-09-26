"""Two-road-user interactions: accident, near_miss.

Distances are measured between ground points and normalised by the pair's
mean apparent size, so thresholds hold across the depth of the scene.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..pipeline import Analysis
from ..tracker import paired_iou
from ..tracks import TrackData, closest_approach, gaussian_smooth
from .common import Event, vehicle_tracks
from .pedestrians import pedestrians

CANDIDATE_DIST = 2.0        # pairs ever closer than this (normalised) are examined
CONFLICT_MIN_SIZE = 0.045   # of the frame width: far-away road users are too small to judge contact
APPROACH_DIST = 1.5         # ... and must have been at least this far apart just before
# accident
ACC_CONTACT_DIST = 0.8
ACC_CONTACT_IOU = 0.10
ACC_PRE_SPEED = 1.5         # the striking road user was really moving (body lengths / s)
ACC_PRE_CLOSING = 0.8
ACC_KEEP = 0.2              # speed kept 0.5-1.1 s after contact / speed 0.5-1.1 s before:
                            # an impact stops at once, braking into a queue takes 2 s or more
ACC_DEFLECT = np.deg2rad(45)
ACC_STOP_SPEED = 0.2
ACC_REST = 2.0              # s both stay at rest, together, after the contact
ACC_MAX_LEN = 10.0
# near_miss
NM_TTC = 1.0
NM_MIN_CLOSING = 0.8
NM_MIN_GAP, NM_MAX_GAP = ACC_CONTACT_DIST, 1.8
NM_KEEP = 0.4               # emergency braking: speed kept across ~1.6 s around the danger
NM_CPA = 0.5               # closest approach (smaller road user's lengths) of a real conflict
NM_MIN_SPEED = 1.5          # evasive road user's speed before braking / swerving
NM_SWERVE_RATE = np.deg2rad(60)
NM_CLEAR_DIST = 2.5
NM_MAX_LEN = 6.0


@dataclass
class PairSeries:
    a: TrackData
    b: TrackData
    t: np.ndarray
    ia: np.ndarray           # sample indices into a / b
    ib: np.ndarray
    dist: np.ndarray         # normalised ground distance
    closing: np.ndarray      # -d(dist)/dt  (per second)
    iou: np.ndarray
    ok: np.ndarray           # both boxes fully inside the frame


def pair_series(a: TrackData, b: TrackData) -> PairSeries | None:
    t, ia, ib = np.intersect1d(a.t, b.t, return_indices=True)
    if len(t) < 4:
        return None
    dist = np.linalg.norm(a.pos[ia] - b.pos[ib], axis=1) / (0.5 * (a.size[ia] + b.size[ib]))
    dist = gaussian_smooth(t, dist, 0.15)
    closing = -np.gradient(dist, t)
    return PairSeries(a, b, t, ia, ib, dist, closing, paired_iou(a.boxes[ia], b.boxes[ib]),
                      a.in_frame[ia] & b.in_frame[ib])


def candidate_pairs(an: Analysis) -> list[PairSeries]:
    min_size = CONFLICT_MIN_SIZE * an.info.width
    users = [u for u in vehicle_tracks(an) + pedestrians(an) if float(np.median(u.size)) >= min_size]
    by_time: dict[float, list[tuple[int, int]]] = {}
    for u, tr in enumerate(users):
        for k, t in enumerate(tr.t):
            by_time.setdefault(float(t), []).append((u, k))
    close: set[tuple[int, int]] = set()
    for members in by_time.values():
        if len(members) < 2:
            continue
        idx = np.array([u for u, _ in members])
        pos = np.array([users[u].pos[k] for u, k in members])
        size = np.array([users[u].size[k] for u, k in members])
        d = np.linalg.norm(pos[:, None] - pos[None], axis=2) / (0.5 * (size[:, None] + size[None]))
        for i, j in zip(*np.where(np.triu(d < CANDIDATE_DIST, 1))):
            ui, uj = int(idx[i]), int(idx[j])
            if users[ui].is_person and users[uj].is_person:
                continue
            close.add((min(ui, uj), max(ui, uj)))
    pairs = [pair_series(users[i], users[j]) for i, j in sorted(close)]
    return [p for p in pairs if p is not None]


def _on_course(p: PairSeries, k: int) -> bool:
    """Paths meet (closest approach within NM_CPA body lengths of the smaller road user),
    rather than passing side by side in neighbouring lanes."""
    scale = min(p.a.size[p.ia[k]], p.b.size[p.ib[k]])
    rel = (p.b.pos[p.ib[k]] - p.a.pos[p.ia[k]]) / scale
    v_rel = (p.b.vel[p.ib[k]] - p.a.vel[p.ia[k]]) / scale
    t_cpa, d_cpa, _ = closest_approach(rel, v_rel, horizon=2.0)
    return t_cpa is not None and d_cpa <= NM_CPA


def _heading_rate(tr: TrackData, idx: np.ndarray, t: np.ndarray) -> np.ndarray:
    h = tr.heading[idx]
    if np.all(np.isnan(h)):
        return np.zeros(len(idx))
    h = np.unwrap(np.nan_to_num(h, nan=float(np.nanmean(h))))
    return np.abs(np.gradient(h, t))


def _heading_change(tr: TrackData, idx: np.ndarray) -> float:
    """Total heading swing (rad) over the given samples."""
    h = tr.heading[idx]
    h = h[~np.isnan(h)]
    return float(np.ptp(np.unwrap(h))) if len(h) >= 2 else 0.0


def _window(t: np.ndarray, t0: float, t1: float) -> np.ndarray:
    return (t >= t0) & (t <= t1)


def _approached(p: PairSeries, t: float) -> bool:
    """The two were apart shortly before ``t`` (not a rider and their bike,
    nor two boxes of one object that always move together)."""
    before = _window(p.t, t - 3.0, t)
    return bool(np.any(p.dist[before] >= APPROACH_DIST))


def _speed_drop(tr: TrackData, t: float) -> tuple[float, float]:
    """(speed shortly before t, speed shortly after t): windows 0.5-1.1 s away from t,
    outside the ~0.8 s blur that the kinematics smoothing puts around an impact."""
    before = (tr.t >= t - 1.1) & (tr.t <= t - 0.5)
    after = (tr.t >= t + 0.5) & (tr.t <= t + 1.1)
    if not before.any() or not after.any():
        return 0.0, 0.0
    return float(np.median(tr.speed[before])), float(np.median(tr.speed[after]))


def _abrupt_stop(tr: TrackData, t: float, keep: float = ACC_KEEP) -> bool:
    """Road user at speed just before t and (almost) stopped just after it."""
    v0, v1 = _speed_drop(tr, t)
    return v0 >= ACC_PRE_SPEED and v1 <= keep * v0


def _accident_at(p: PairSeries) -> tuple[float, float] | None:
    """Contact after a real approach, an impact-like stop or deflection, then both at rest together.

    Dense queues put vehicles within a body length of each other all the time;
    what separates a crash is the abrupt stop from speed and the aftermath.
    """
    spd_a, spd_b = p.a.speed[p.ia], p.b.speed[p.ib]
    contact = np.where((p.dist < ACC_CONTACT_DIST) & (p.iou > ACC_CONTACT_IOU) & p.ok)[0]
    for c in contact:
        tc = p.t[c]
        if not _approached(p, tc):
            continue
        pre = _window(p.t, tc - 1.5, tc)
        if max(spd_a[pre].max(), spd_b[pre].max()) < ACC_PRE_SPEED or p.closing[pre].max() < ACC_PRE_CLOSING:
            continue
        hit = _window(p.t, tc - 0.5, tc + 1.5)
        abrupt = _abrupt_stop(p.a, tc) or _abrupt_stop(p.b, tc)
        deflect = max(_heading_change(p.a, p.ia[hit]), _heading_change(p.b, p.ib[hit]))
        if not abrupt and deflect < ACC_DEFLECT:
            continue
        after = _window(p.t, tc, tc + ACC_REST + 2.0)
        rest = after & (spd_a < ACC_STOP_SPEED) & (spd_b < ACC_STOP_SPEED) & (p.dist < 1.2)
        if rest.sum() * float(np.median(np.diff(p.t))) < ACC_REST:
            continue
        end = p.t[np.where(rest)[0][0]]
        return float(tc), float(min(max(end, tc + 0.5), tc + ACC_MAX_LEN))
    return None


def accident(an: Analysis, pairs: list[PairSeries] | None = None) -> list[Event]:
    pairs = candidate_pairs(an) if pairs is None else pairs
    events = []
    for p in pairs:
        if p.a.is_person and p.b.is_person:
            continue
        span = _accident_at(p)
        if span:
            events.append([span[0], span[1], "accident"])
    return events


def _evasive(tr: TrackData, idx: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Per common sample: emergency braking around it, or a swerve."""
    drops = np.array([_speed_drop(tr, float(ti)) for ti in t]).reshape(-1, 2)
    braking = (drops[:, 0] >= NM_MIN_SPEED) & (drops[:, 1] <= NM_KEEP * drops[:, 0])
    swerve = (_heading_rate(tr, idx, t) >= NM_SWERVE_RATE) & (tr.speed[idx] >= NM_MIN_SPEED)
    return braking | swerve


def near_miss(an: Analysis, pairs: list[PairSeries] | None = None) -> list[Event]:
    pairs = candidate_pairs(an) if pairs is None else pairs
    events = []
    for p in pairs:
        if _accident_at(p) is not None:
            continue
        gap = np.maximum(p.dist - ACC_CONTACT_DIST, 0.05)
        ttc = np.where(p.closing > NM_MIN_CLOSING, gap / np.maximum(p.closing, 1e-6), np.inf)
        danger = np.where((ttc < NM_TTC) & p.ok)[0]
        if len(danger) == 0:
            continue
        danger = [k for k in danger if _on_course(p, k)]
        if not danger:
            continue
        k = danger[0]
        if not _approached(p, p.t[k]):
            continue
        after = _window(p.t, p.t[k], p.t[k] + 2.0)
        d_min = p.dist[after].min()
        if not (NM_MIN_GAP <= d_min <= NM_MAX_GAP):
            continue
        evasive = _evasive(p.a, p.ia, p.t) | _evasive(p.b, p.ib, p.t)
        k_min = int(np.where(after)[0][np.argmin(p.dist[after])])
        near = _window(p.t, p.t[k] - 1.0, p.t[k_min] + 0.5)
        if not np.any(evasive & near):
            continue
        onset = p.t[np.where(evasive & near)[0][0]]
        clear = np.where((p.t > p.t[k_min]) & (p.dist > NM_CLEAR_DIST))[0]
        end = p.t[clear[0]] if len(clear) else min(p.a.t[-1], p.b.t[-1], p.t[k_min] + 2.0)
        events.append([float(onset), float(min(end, onset + NM_MAX_LEN)), "near_miss"])
    return events
