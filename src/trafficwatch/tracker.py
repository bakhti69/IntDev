"""Multi-object tracker (ByteTrack-style two-stage association).

Kept deliberately small and deterministic: constant-velocity box prediction,
Hungarian matching on IoU, a second pass that recovers tracks with
low-confidence detections, and category groups so that a car is never
matched to a pedestrian. It is causal, so Part B uses it too.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detector import Detections, category_group


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def paired_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of a[i] with b[i] for aligned (N, 4) box arrays."""
    x1 = np.maximum(a[:, 0], b[:, 0])
    y1 = np.maximum(a[:, 1], b[:, 1])
    x2 = np.minimum(a[:, 2], b[:, 2])
    y2 = np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a + area_b - inter + 1e-9)


@dataclass
class Track:
    id: int
    box: np.ndarray
    t: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))  # centre px / s
    votes: Counter = field(default_factory=Counter)
    hits: int = 1
    last_seen: float = 0.0

    @property
    def category(self) -> str:
        return self.votes.most_common(1)[0][0]

    @property
    def size(self) -> float:
        return float(np.sqrt(max(1.0, (self.box[2] - self.box[0]) * (self.box[3] - self.box[1]))))

    def predict(self, t: float) -> np.ndarray:
        """Last box shifted by the centre velocity; extrapolation is capped so a
        lost track does not fly off along the road and latch onto another car."""
        dt = min(t - self.t, PREDICT_HORIZON)
        dx, dy = self.velocity * dt
        return self.box + np.array([dx, dy, dx, dy], np.float32)


PREDICT_HORIZON = 0.5     # s of constant-velocity extrapolation
MAX_SPEED = 8.0           # body sizes / s, cap on the velocity estimate
GATE_DIST = 0.8           # max centre distance to the prediction, in body sizes
GATE_SIZE = (0.6, 1.6)    # allowed size ratio detection / track


def _centre(b: np.ndarray) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], np.float32)


class Tracker:
    def __init__(self, high: float = 0.35, low: float = 0.15, new_track: float = 0.4,
                 match_iou: float = 0.2, max_age: float = 1.5, max_age_static: float = 6.0,
                 min_hits: int = 2):
        self.high, self.low, self.new_track = high, low, new_track
        self.match_iou, self.max_age, self.min_hits = match_iou, max_age, min_hits
        # parked / queued objects get occluded by passing traffic: keep them longer
        self.max_age_static = max_age_static
        self.tracks: list[Track] = []
        self._next_id = 1

    def _associate(self, tracks: list[Track], boxes: np.ndarray, groups: list[str],
                   t: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks or len(boxes) == 0:
            return [], list(range(len(tracks))), list(range(len(boxes)))
        pred = np.stack([tr.predict(t) for tr in tracks])
        iou = iou_matrix(pred, boxes)
        pc = (pred[:, :2] + pred[:, 2:]) / 2
        bc = (boxes[:, :2] + boxes[:, 2:]) / 2
        tsize = np.array([tr.size for tr in tracks])
        bsize = np.sqrt(np.clip((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1, None))
        dist = np.linalg.norm(pc[:, None] - bc[None], axis=2) / tsize[:, None]
        ratio = bsize[None, :] / tsize[:, None]
        score = iou.copy()
        score[(dist > GATE_DIST) | (ratio < GATE_SIZE[0]) | (ratio > GATE_SIZE[1])] = 0.0
        for i, tr in enumerate(tracks):
            g = category_group(tr.category)
            for j, gj in enumerate(groups):
                if g != gj:
                    score[i, j] = 0.0
        rows, cols = linear_sum_assignment(-score)
        matches = [(r, c) for r, c in zip(rows, cols) if score[r, c] >= self.match_iou]
        mr = {r for r, _ in matches}
        mc = {c for _, c in matches}
        return (matches, [i for i in range(len(tracks)) if i not in mr],
                [j for j in range(len(boxes)) if j not in mc])

    def _update_track(self, tr: Track, box: np.ndarray, cat: str, t: float) -> None:
        dt = t - tr.t
        if dt > 0:
            v = (_centre(box) - _centre(tr.box)) / dt
            v = 0.6 * tr.velocity + 0.4 * v if tr.hits > 1 else v
            cap = MAX_SPEED * tr.size
            speed = float(np.linalg.norm(v))
            tr.velocity = (v * (cap / speed) if speed > cap else v).astype(np.float32)
        tr.box, tr.t, tr.last_seen = box, t, t
        tr.votes[cat] += 1
        tr.hits += 1

    def _max_age(self, tr: Track) -> float:
        moving = float(np.linalg.norm(tr.velocity))
        return self.max_age_static if moving < 0.1 * tr.size and tr.hits >= 5 else self.max_age

    def update(self, dets: Detections, t: float) -> list[Track]:
        """Advance to time ``t``; returns confirmed tracks seen in this frame."""
        groups = [category_group(c) for c in dets.cats]
        hi = np.where(dets.scores >= self.high)[0]
        lo = np.where((dets.scores >= self.low) & (dets.scores < self.high))[0]

        # stage 1: all tracks vs confident detections
        m1, um_tr, um_hi = self._associate(self.tracks, dets.boxes[hi], [groups[i] for i in hi], t)
        for r, c in m1:
            j = hi[c]
            self._update_track(self.tracks[r], dets.boxes[j], dets.cats[j], t)
        # stage 2: leftover tracks vs weak detections
        rest = [self.tracks[i] for i in um_tr]
        m2, _, _ = self._associate(rest, dets.boxes[lo], [groups[i] for i in lo], t)
        for r, c in m2:
            j = lo[c]
            self._update_track(rest[r], dets.boxes[j], dets.cats[j], t)
        # new tracks from unmatched confident detections
        for c in um_hi:
            j = hi[c]
            if dets.scores[j] >= self.new_track:
                tr = Track(id=self._next_id, box=dets.boxes[j].copy(), t=t, last_seen=t)
                tr.votes[dets.cats[j]] += 1
                self.tracks.append(tr)
                self._next_id += 1
        # drop stale tracks
        self.tracks = [tr for tr in self.tracks if t - tr.last_seen <= self._max_age(tr)]
        return [tr for tr in self.tracks if tr.last_seen == t and tr.hits >= self.min_hits]
