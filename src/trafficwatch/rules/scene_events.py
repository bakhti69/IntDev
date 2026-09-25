"""Scene-level rules from low-rate pixel cues: road_obstacle, fire_smoke."""
from __future__ import annotations

import cv2
import numpy as np

from ..pipeline import Analysis
from ..segments import merge, runs
from .common import Event

# road_obstacle
ANIMAL_MIN_DURATION = 1.0
OBS_EVAL_PERIOD = 1.0        # s between evaluations
OBS_PAST = (-40.0, -15.0)    # background window relative to t
OBS_FUTURE = 8.0             # the new object must stay this long
OBS_DIFF = 32                # grey levels
OBS_MAX_COVERAGE = 0.05      # pixels ever under detected road users (people waiting, parked cars) are ignored
OBS_AREA = (0.0006, 0.006)   # blob area as a fraction of the (cue) frame
OBS_MAX_DRIFT = 6.0          # cue px of global view drift we compensate
OBS_MIN_FILL = 0.35          # debris is blob-like; fragments of lane markings are not
OBS_MAX_ASPECT = 4.0
OBS_MIN_PERSIST = 8.0        # s
# fire_smoke
FIRE_WINDOW = 3.0            # s
FIRE_MIN_AREA = 0.0004       # fraction of the cue frame
FIRE_MIN_CV = 0.2            # flicker: coefficient of variation of the flame area
FIRE_MAX_DRIFT = 0.02        # centroid drift, fraction of frame width
FIRE_MIN_DURATION = 2.0


def _animal_events(an: Analysis) -> list[Event]:
    events = []
    for tr in an.tracks:
        if tr.category != "animal":
            continue
        on = np.array([an.scene.road_depth(x, y) >= 0 for x, y in tr.pos])
        for s, e in runs(tr.t, on, max_gap=1.0):
            if e - s >= ANIMAL_MIN_DURATION:
                events.append([s, e, "road_obstacle"])
    return events


def _coverage(boxes: list[np.ndarray], shape: tuple[int, int], pad: float = 0.15) -> np.ndarray:
    cov = np.zeros(shape, np.float32)
    for bs in boxes:
        m = np.zeros(shape, np.uint8)
        for x1, y1, x2, y2 in bs:
            px, py = pad * (x2 - x1), pad * (y2 - y1)
            cv2.rectangle(m, (int(x1 - px), int(y1 - py)), (int(x2 + px), int(y2 + py)), 1, -1)
        cov += m
    return cov / max(1, len(boxes))


def _static_novelty(an: Analysis) -> list[Event]:
    """Something new appears on the carriageway, stays, and is not a detected road user."""
    cues = an.cues
    if len(cues.t) < 10:
        return []
    t = np.asarray(cues.t)
    shape = cues.gray[0].shape
    road = cv2.erode(an.scene.road_mask(cues.scale)[: shape[0], : shape[1]], np.ones((3, 3), np.uint8))
    frame_area = shape[0] * shape[1]
    hits: list[tuple[float, np.ndarray]] = []   # (time, blob centroid)
    for t0 in np.arange(t[0] - OBS_PAST[0], t[-1] - OBS_FUTURE, OBS_EVAL_PERIOD):
        past = np.where((t >= t0 + OBS_PAST[0]) & (t <= t0 + OBS_PAST[1]))[0]
        fut = np.where((t >= t0) & (t <= t0 + OBS_FUTURE))[0]
        if len(past) < 8 or len(fut) < 4:
            continue
        bg = np.median(np.stack([cues.gray[i] for i in past]), axis=0).astype(np.float32)
        cur = np.median(np.stack([cues.gray[i] for i in fut]), axis=0).astype(np.float32)
        # a road user parked in the past window that has since left is not a new object
        cov_past = _coverage([cues.boxes[i] for i in past], shape)
        cov_fut = _coverage([cues.boxes[i] for i in fut], shape)
        free = (cov_past < OBS_MAX_COVERAGE) & (cov_fut < OBS_MAX_COVERAGE)
        # undo any global drift of the view, then a shift-tolerant change test
        (dx, dy), response = cv2.phaseCorrelate(bg, cur)
        if response >= 0.1 and np.hypot(dx, dy) <= OBS_MAX_DRIFT:
            cur = cv2.warpAffine(cur, np.float32([[1, 0, -dx], [0, 1, -dy]]), (shape[1], shape[0]),
                                 borderMode=cv2.BORDER_REPLICATE)
        # changed = outside the local range of the background (pole sway, jitter)
        k = np.ones((5, 5), np.uint8)
        changed = (cur > cv2.dilate(bg, k) + OBS_DIFF) | (cur < cv2.erode(bg, k) - OBS_DIFF)
        mask = (changed & (road > 0) & free).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
        for c in range(1, n):
            area = stats[c, cv2.CC_STAT_AREA] / frame_area
            bw, bh = stats[c, cv2.CC_STAT_WIDTH], stats[c, cv2.CC_STAT_HEIGHT]
            fill = stats[c, cv2.CC_STAT_AREA] / max(1, bw * bh)
            compact = fill >= OBS_MIN_FILL and max(bw, bh) <= OBS_MAX_ASPECT * min(bw, bh)
            if OBS_AREA[0] <= area <= OBS_AREA[1] and compact:
                hits.append((float(t0), cents[c]))
    # group hits at the same place into persistent objects
    events = []
    used = [False] * len(hits)
    radius = 0.02 * shape[1]
    for i, (ti, ci) in enumerate(hits):
        if used[i]:
            continue
        times = [ti]
        used[i] = True
        for j in range(i + 1, len(hits)):
            tj, cj = hits[j]
            if not used[j] and np.linalg.norm(cj - ci) < radius and tj - times[-1] <= 2 * OBS_EVAL_PERIOD:
                times.append(tj)
                used[j] = True
        if times[-1] - times[0] >= OBS_MIN_PERSIST:
            # the window median flips once the object fills half of the future window
            events.append([times[0] + OBS_FUTURE / 2, times[-1] + OBS_FUTURE / 2, "road_obstacle"])
    return events


def road_obstacle(an: Analysis) -> list[Event]:
    return _animal_events(an) + _static_novelty(an)


def fire_smoke(an: Analysis) -> list[Event]:
    """Flame-coloured blob at a fixed place whose area flickers."""
    cues = an.cues
    if not cues.t:
        return []
    t = np.asarray(cues.t)
    shape = cues.fire[0].shape
    near_road = cv2.dilate(an.scene.road_mask(cues.scale)[: shape[0], : shape[1]], np.ones((15, 15), np.uint8))
    area, cent = np.zeros(len(t)), np.full((len(t), 2), np.nan)
    for k, m in enumerate(cues.fire):
        m = (m & (near_road > 0)).astype(np.uint8)
        n, _, stats, cents = cv2.connectedComponentsWithStats(m)
        if n > 1:
            c = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            area[k] = stats[c, cv2.CC_STAT_AREA] / (shape[0] * shape[1])
            cent[k] = cents[c]
    flag = np.zeros(len(t), bool)
    for k in range(len(t)):
        w = (t >= t[k] - FIRE_WINDOW) & (t <= t[k])
        a = area[w]
        if w.sum() < 4 or np.median(a) < FIRE_MIN_AREA:
            continue
        drift = np.nanmax(np.nanstd(cent[w], axis=0)) / shape[1]
        if drift <= FIRE_MAX_DRIFT and np.std(a) / (np.mean(a) + 1e-9) >= FIRE_MIN_CV:
            flag[k] = True
    segs = [(s - FIRE_WINDOW, e) for s, e in runs(t, flag, max_gap=2.0) if e - s >= FIRE_MIN_DURATION]
    return [[max(0.0, s), e, "fire_smoke"] for s, e in merge(segs, 2.0)]
