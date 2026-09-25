"""Traffic-signal state from the signal heads that face the camera.

Each head is a small box in the scene layout. We count bright, saturated
red / amber / green pixels in it; heads vote and the per-frame state is
median-filtered over ~1 s so that a single blown-out frame does not flip it.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

STATES = ("unknown", "red", "amber", "green")
_CODE = {s: i for i, s in enumerate(STATES)}


def head_state(crop: np.ndarray) -> str:
    if crop.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    lit = (v >= 150) & (s >= 90)
    red = int(np.count_nonzero(lit & ((h <= 8) | (h >= 165))))
    amber = int(np.count_nonzero(lit & (h > 8) & (h <= 30)))
    green = int(np.count_nonzero(lit & (h >= 45) & (h <= 100)))
    counts = {"red": red, "amber": amber, "green": green}
    best = max(counts, key=counts.get)
    min_px = max(3, int(0.01 * crop.shape[0] * crop.shape[1]))
    return best if counts[best] >= min_px else "unknown"


def frame_state(frame: np.ndarray, heads: dict[str, np.ndarray]) -> str:
    votes = []
    H, W = frame.shape[:2]
    for b in heads.values():
        x1, y1 = max(0, int(b[0])), max(0, int(b[1]))
        x2, y2 = min(W, int(np.ceil(b[2]))), min(H, int(np.ceil(b[3])))
        st = head_state(frame[y1:y2, x1:x2])
        if st != "unknown":
            votes.append(st)
    if not votes:
        return "unknown"
    return max(set(votes), key=lambda s: (votes.count(s), s == "red"))


@dataclass
class SignalTimeline:
    t: np.ndarray        # (n,) seconds
    code: np.ndarray     # (n,) index into STATES

    @staticmethod
    def build(t: list[float], states: list[str], window: float = 1.0) -> "SignalTimeline":
        t_arr = np.asarray(t, np.float64)
        codes = np.array([_CODE[s] for s in states], np.int64)
        if len(t_arr) > 2:
            dt = float(np.median(np.diff(t_arr)))
            k = max(1, int(round(window / max(dt, 1e-3)))) | 1
            pad = np.pad(codes, (k // 2, k // 2), mode="edge")
            # mode filter over the window (robust to single-frame glare)
            windows = np.lib.stride_tricks.sliding_window_view(pad, k)
            codes = np.array([np.bincount(w, minlength=len(STATES)).argmax() for w in windows])
        return SignalTimeline(t_arr, codes)

    def state_at(self, t: float) -> str:
        if len(self.t) == 0:
            return "unknown"
        i = int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))
        return STATES[self.code[i]]

    def red_for(self, t: float, min_duration: float) -> bool:
        """True if the signal has been red continuously for >= min_duration at t."""
        if len(self.t) == 0:
            return False
        lo = np.searchsorted(self.t, t - min_duration)
        hi = np.searchsorted(self.t, t, side="right")
        seg = self.code[lo:hi]
        return len(seg) > 0 and bool(np.all(seg == _CODE["red"]))

    @property
    def is_readable(self) -> bool:
        return len(self.code) > 0 and float(np.mean(self.code != _CODE["unknown"])) > 0.5
