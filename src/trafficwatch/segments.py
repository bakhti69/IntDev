"""Turning per-sample flags into clean [start, end] segments."""
from __future__ import annotations

import numpy as np

Segment = tuple[float, float]


def runs(t: np.ndarray, mask: np.ndarray, max_gap: float = 0.0) -> list[Segment]:
    """Maximal runs of True in ``mask`` sampled at times ``t``.

    Consecutive True samples further apart than max(max_gap, 1.5 x the
    sampling step) start a new run.
    """
    t = np.asarray(t, np.float64)
    on = t[np.asarray(mask, bool)]
    if len(on) == 0:
        return []
    step = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    breaks = np.where(np.diff(on) > max(max_gap, 1.5 * step))[0]
    starts = np.r_[on[0], on[breaks + 1]]
    ends = np.r_[on[breaks], on[-1]]
    return [(float(s), float(e)) for s, e in zip(starts, ends)]


def merge(segs: list[Segment], gap: float) -> list[Segment]:
    """Merge segments that overlap or are separated by <= gap seconds."""
    out: list[list[float]] = []
    for s, e in sorted(segs):
        if out and s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def finalize_events(events: list[list], duration: float, merge_gap: dict[str, float],
                    min_len: dict[str, float], default_gap: float = 1.0,
                    default_min: float = 0.5) -> list[list]:
    """Per class: merge nearby/overlapping segments, clip to the video, drop blips.

    Same-class segments never overlap in the output (evaluate.py requirement).
    """
    by_label: dict[str, list[Segment]] = {}
    for s, e, label in events:
        by_label.setdefault(label, []).append((float(s), float(e)))
    out = []
    for label, segs in by_label.items():
        for s, e in merge(segs, merge_gap.get(label, default_gap)):
            s, e = max(0.0, s), min(duration, e)
            if e - s >= min_len.get(label, default_min):
                out.append([round(s, 2), round(e, 2), label])
    out.sort(key=lambda x: (x[0], x[2]))
    return out
