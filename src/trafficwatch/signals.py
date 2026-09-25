"""Traffic-signal state from the signal heads that face the camera.

Each head is a small box in the scene layout. We count lit, saturated red /
amber / green pixels in it. LED heads flicker against the camera shutter and
lamps wash out in sunlight, so single frames are often "unknown": the
timeline carries the last known state over short gaps and then applies a
~1 s mode filter.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

STATES = ("unknown", "red", "amber", "green")
_CODE = {s: i for i, s in enumerate(STATES)}
MIN_LIT_PX = 4


def head_state(crop: np.ndarray, kind: str = "vehicle") -> str:
    if crop.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = (hsv[..., i].astype(int) for i in range(3))
    lit = (v >= 90) & (s >= 120)
    counts = {
        "red": int(np.count_nonzero(lit & ((h <= 12) | (h >= 160)))),
        "amber": int(np.count_nonzero(lit & (h > 12) & (h <= 30))),
        "green": int(np.count_nonzero(lit & (h >= 45) & (h <= 100))),
    }
    if kind == "pedestrian":  # standing figure is red/orange, there is no amber
        counts["red"] += counts.pop("amber")
    best = max(counts, key=counts.get)
    return best if counts[best] >= MIN_LIT_PX else "unknown"


def read_heads(frame: np.ndarray, boxes: dict[str, np.ndarray], kinds: dict[str, str]) -> dict[str, str]:
    H, W = frame.shape[:2]
    out = {}
    for name, b in boxes.items():
        x1, y1 = max(0, int(b[0])), max(0, int(b[1]))
        x2, y2 = min(W, int(np.ceil(b[2]))), min(H, int(np.ceil(b[3])))
        out[name] = head_state(frame[y1:y2, x1:x2], kinds.get(name, "vehicle"))
    return out


@dataclass
class SignalTimeline:
    t: np.ndarray        # (n,) seconds
    code: np.ndarray     # (n,) index into STATES

    @staticmethod
    def build(t: list[float], states: list[str], window: float = 1.0, hold: float = 2.0) -> "SignalTimeline":
        t_arr = np.asarray(t, np.float64)
        codes = np.array([_CODE[s] for s in states], np.int64)
        last, last_t = 0, -np.inf
        for i in range(len(codes)):  # carry the last known state over short dropouts
            if codes[i]:
                last, last_t = codes[i], t_arr[i]
            elif t_arr[i] - last_t <= hold:
                codes[i] = last
        if len(t_arr) > 2:
            dt = float(np.median(np.diff(t_arr)))
            k = max(1, int(round(window / max(dt, 1e-3)))) | 1
            pad = np.pad(codes, (k // 2, k // 2), mode="edge")
            windows = np.lib.stride_tricks.sliding_window_view(pad, k)
            codes = np.array([np.bincount(w, minlength=len(STATES)).argmax() for w in windows])
        return SignalTimeline(t_arr, codes)

    def state_at(self, t: float) -> str:
        if len(self.t) == 0:
            return "unknown"
        i = int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))
        return STATES[self.code[i]]

    def held(self, state: str, t: float, min_duration: float) -> bool:
        """True if the head showed ``state`` continuously for >= min_duration up to t."""
        if len(self.t) == 0:
            return False
        lo = np.searchsorted(self.t, t - min_duration)
        hi = np.searchsorted(self.t, t, side="right")
        seg = self.code[lo:hi]
        return len(seg) > 0 and bool(np.all(seg == _CODE[state]))

    def fraction(self, state: str, t0: float, t1: float) -> float:
        lo, hi = np.searchsorted(self.t, [t0, t1])
        seg = self.code[lo:hi + 1]
        return float(np.mean(seg == _CODE[state])) if len(seg) else 0.0

    @property
    def is_readable(self) -> bool:
        return len(self.code) > 0 and float(np.mean(self.code != _CODE["unknown"])) > 0.5
