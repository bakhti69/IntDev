"""Track histories and their kinematics.

Speeds are expressed in *body sizes per second* (image speed divided by the
object's apparent size, sqrt(w*h)), which removes most of the perspective
effect: a car crawling in the foreground and one crawling far away get the
same number. Positions use the ground point (bottom-centre of the box).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .detector import TWO_WHEELERS, VEHICLES


def gaussian_smooth(t: np.ndarray, x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing over (possibly irregular) timestamps. x: (n,) or (n, d)."""
    x = x.astype(np.float64)
    if len(t) < 3 or sigma <= 0:
        return x.copy()
    lo = np.searchsorted(t, t - 3 * sigma, side="left")
    hi = np.searchsorted(t, t + 3 * sigma, side="right")
    out = np.empty_like(x)
    for i in range(len(t)):  # banded: long stationary tracks stay O(n * window)
        w = np.exp(-0.5 * ((t[lo[i]:hi[i]] - t[i]) / sigma) ** 2)
        out[i] = np.tensordot(w / w.sum(), x[lo[i]:hi[i]], axes=1)
    return out


def derivative(t: np.ndarray, x: np.ndarray, half_window: float) -> np.ndarray:
    """Central difference over +/- half_window seconds (clipped at the ends)."""
    n = len(t)
    if n < 2:
        return np.zeros_like(x, dtype=np.float64)
    lo = np.searchsorted(t, t - half_window, side="left")
    hi = np.searchsorted(t, t + half_window, side="right") - 1
    hi = np.maximum(hi, np.minimum(np.arange(n) + 1, n - 1))
    lo = np.minimum(lo, np.maximum(np.arange(n) - 1, 0))
    span = (t[hi] - t[lo])
    span = np.where(span > 1e-6, span, np.inf)
    d = (x[hi] - x[lo])
    return d / (span[:, None] if x.ndim == 2 else span)


@dataclass
class TrackHistory:
    """Raw observations of one track, appended frame by frame."""
    id: int
    t: list[float] = field(default_factory=list)
    boxes: list[np.ndarray] = field(default_factory=list)
    votes: Counter = field(default_factory=Counter)

    def add(self, t: float, box: np.ndarray, category: str) -> None:
        self.t.append(t)
        self.boxes.append(np.asarray(box, np.float32))
        self.votes[category] += 1


@dataclass
class TrackData:
    """Finalised track with smoothed kinematics (arrays indexed by sample)."""
    id: int
    category: str
    t: np.ndarray        # (n,)
    boxes: np.ndarray    # (n, 4) raw boxes
    pos: np.ndarray      # (n, 2) smoothed ground point
    size: np.ndarray     # (n,) smoothed sqrt(w*h)
    vel: np.ndarray      # (n, 2) px/s
    speed: np.ndarray    # (n,) body sizes / s
    accel: np.ndarray    # (n,) d(speed)/dt, body sizes / s^2
    heading: np.ndarray  # (n,) radians, atan2(vy, vx); nan when not moving
    at_edge: np.ndarray | None = None  # (n,) box touches the frame border (cut off: shape unreliable)

    @property
    def in_frame(self) -> np.ndarray:
        """Samples whose box is fully inside the frame."""
        return ~self.at_edge if self.at_edge is not None else np.ones(len(self.t), bool)

    @property
    def is_vehicle(self) -> bool:
        return self.category in VEHICLES

    @property
    def is_two_wheeler(self) -> bool:
        return self.category in TWO_WHEELERS

    @property
    def is_person(self) -> bool:
        return self.category == "person"

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) else 0.0

    def index_at(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))


def edge_mask(boxes: np.ndarray, width: int, height: int, margin_frac: float = 0.01) -> np.ndarray:
    m = margin_frac * width
    return (boxes[:, 0] <= m) | (boxes[:, 1] <= m) | (boxes[:, 2] >= width - m) | (boxes[:, 3] >= height - m)


def finalize(h: TrackHistory, frame_size: tuple[int, int] | None = None,
             pos_sigma: float = 0.25, vel_window: float = 0.4) -> TrackData:
    t = np.asarray(h.t, np.float64)
    boxes = np.stack(h.boxes).astype(np.float64)
    ground = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], axis=1)
    raw_size = np.sqrt(np.clip((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1, None))
    pos = gaussian_smooth(t, ground, pos_sigma)
    size = gaussian_smooth(t, raw_size, 1.0)
    vel = derivative(t, pos, vel_window)
    speed = np.linalg.norm(vel, axis=1) / size
    accel = derivative(t, gaussian_smooth(t, speed, 0.2), 0.3)
    heading = np.arctan2(vel[:, 1], vel[:, 0])
    heading[speed < 0.15] = np.nan
    at_edge = edge_mask(boxes, *frame_size) if frame_size else None
    return TrackData(id=h.id, category=h.votes.most_common(1)[0][0], t=t, boxes=boxes.astype(np.float32),
                     pos=pos, size=size, vel=vel, speed=speed, accel=accel, heading=heading, at_edge=at_edge)


def angle_diff(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    """Smallest signed difference a - b in radians, in (-pi, pi]."""
    return (np.asarray(a) - np.asarray(b) + np.pi) % (2 * np.pi) - np.pi


def closest_approach(rel: np.ndarray, v_rel: np.ndarray, horizon: float = 4.0) -> tuple[float | None, float, float]:
    """(time, distance) of the closest point of approach for relative position and velocity
    (both in body lengths); time is None when the two are not approaching."""
    vv = float(v_rel @ v_rel)
    speed = float(np.sqrt(vv))
    if speed < 0.2:
        return None, float("inf"), speed
    t_cpa = -float(rel @ v_rel) / vv
    if t_cpa <= 0 or t_cpa > horizon:
        return None, float("inf"), speed
    return t_cpa, float(np.linalg.norm(rel + v_rel * t_cpa)), speed
