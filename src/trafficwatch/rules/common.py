"""Geometry helpers shared by the rules."""
from __future__ import annotations

import cv2
import numpy as np

from ..pipeline import Analysis
from ..scene import Approach
from ..tracks import TrackData

Event = list  # [start_sec, end_sec, label]

MIN_SIZE_FRAC = 0.012  # ignore road users smaller than this fraction of the frame width


def vehicle_tracks(an: Analysis, two_wheelers: bool = True) -> list[TrackData]:
    """Vehicle tracks (optionally with bicycles/motorcycles) large enough to trust."""
    min_size = MIN_SIZE_FRAC * an.info.width
    return [tr for tr in an.tracks
            if (tr.is_vehicle or (two_wheelers and tr.is_two_wheeler))
            and float(np.median(tr.size)) >= min_size]


def inside(poly: np.ndarray, pts: np.ndarray, margin: np.ndarray | float = 0.0) -> np.ndarray:
    """Vectorised point-in-polygon with a (per-point) margin in px."""
    d = np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), True) for x, y in pts])
    return d >= -np.broadcast_to(margin, d.shape)


def line_progress(approach: Approach, pts: np.ndarray) -> np.ndarray:
    """Signed distance (px) past the approach's stop line, positive downstream."""
    p0, p1 = approach.stop_line[0].astype(np.float64), approach.stop_line[1].astype(np.float64)
    n = np.array([-(p1 - p0)[1], (p1 - p0)[0]])
    n /= np.linalg.norm(n) + 1e-9
    if np.dot(n, approach.direction) < 0:
        n = -n
    return (np.asarray(pts, np.float64) - p0) @ n


def within_line_span(approach: Approach, pts: np.ndarray, pad: float = 0.1) -> np.ndarray:
    """Whether points project onto the stop line segment (with relative padding)."""
    p0, p1 = approach.stop_line[0].astype(np.float64), approach.stop_line[1].astype(np.float64)
    d = p1 - p0
    u = (np.asarray(pts, np.float64) - p0) @ d / (d @ d + 1e-9)
    return (u >= -pad) & (u <= 1 + pad)


def front_point(tr: TrackData, approach: Approach) -> np.ndarray:
    """Front bumper estimate: the ground point moved along the travel direction
    when the vehicle drives away from the camera (its box bottom is the rear)."""
    away = approach.direction[1] < 0
    if not away:
        return tr.pos
    return tr.pos + approach.direction[None, :] * (0.7 * tr.size[:, None])


def moving_toward(tr: TrackData, direction: np.ndarray, min_speed: float = 0.4) -> np.ndarray:
    """Per sample: moving with a positive component along ``direction``."""
    v = tr.vel / (np.linalg.norm(tr.vel, axis=1, keepdims=True) + 1e-9)
    return (tr.speed >= min_speed) & (v @ direction > 0.3)


def extend_to_track_end(tr: TrackData, end: float, slack: float = 1.0) -> float:
    """If the track disappears shortly after ``end`` (object left the frame), use its end."""
    return float(tr.t[-1]) if tr.t[-1] - end <= slack else end
