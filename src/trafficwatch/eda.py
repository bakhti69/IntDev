"""Exploratory statistics of a processed video, for the website.

Everything is derived from one :class:`Analysis`, so EDA shows exactly what
the event rules see (tracks, zones, speeds), not a separate re-run.
"""
from __future__ import annotations

import cv2
import numpy as np

from .pipeline import Analysis

GRID = (32, 18)           # direction-field cells (x, y)
BIN_SEC = 1.0


def _bins(an: Analysis) -> np.ndarray:
    return np.arange(0.0, an.info.duration + BIN_SEC, BIN_SEC)


def counts_over_time(an: Analysis) -> dict:
    """Mean detections per frame by class, in 1 s bins."""
    edges = _bins(an)
    idx = np.clip(np.digitize(an.times, edges) - 1, 0, len(edges) - 2)
    out = {"t": edges[:-1].round(2).tolist()}
    for cat, c in an.counts.items():
        if c.sum() == 0:
            continue
        sums = np.bincount(idx, weights=c, minlength=len(edges) - 1)
        n = np.maximum(np.bincount(idx, minlength=len(edges) - 1), 1)
        out[cat] = (sums / n).round(2).tolist()
    return out


def zone_occupancy(an: Analysis) -> dict:
    """Vehicles per zone and their median speed, in 1 s bins."""
    edges = _bins(an)
    zones = list(an.scene.roads)
    count = {z: np.zeros(len(edges) - 1) for z in zones}
    speed = {z: [[] for _ in range(len(edges) - 1)] for z in zones}
    n_frames = np.maximum(np.bincount(np.clip(np.digitize(an.times, edges) - 1, 0, len(edges) - 2),
                                      minlength=len(edges) - 1), 1)
    for tr in an.tracks:
        if not (tr.is_vehicle or tr.is_two_wheeler):
            continue
        b = np.clip(np.digitize(tr.t, edges) - 1, 0, len(edges) - 2)
        for k, z in enumerate(an.zones(tr)):
            if z in count:
                count[z][b[k]] += 1
                speed[z][b[k]].append(tr.speed[k])
    return {
        "t": edges[:-1].round(2).tolist(),
        "vehicles": {z: (count[z] / n_frames).round(2).tolist() for z in zones},
        "median_speed": {z: [round(float(np.median(s)), 2) if s else None for s in speed[z]] for z in zones},
    }


def speed_histogram(an: Analysis) -> dict:
    speeds = [tr.speed for tr in an.tracks if tr.is_vehicle]
    s = np.concatenate(speeds) if speeds else np.zeros(0)
    hist, edges = np.histogram(s, bins=np.linspace(0, 6, 31))
    return {"edges": edges.round(2).tolist(), "counts": hist.tolist(),
            "stationary_fraction": round(float(np.mean(s < 0.12)), 3) if len(s) else None}


def lighting(an: Analysis) -> dict:
    t = an.cues.t
    return {"t": [round(x, 2) for x in t], "brightness": [round(float(g.mean()), 1) for g in an.cues.gray]}


def summary(an: Analysis) -> dict:
    cats = {}
    for tr in an.tracks:
        cats[tr.category] = cats.get(tr.category, 0) + 1
    return {
        "width": an.info.width, "height": an.info.height, "fps": round(an.info.fps, 2),
        "duration": round(an.info.duration, 2), "n_frames": an.info.n_frames,
        "tracks_by_class": cats,
        "registered": an.scene.registered, "registration_inliers": an.scene.inliers,
        "mean_brightness": round(float(np.mean([g.mean() for g in an.cues.gray])), 1) if an.cues.gray else None,
    }


def trajectories(an: Analysis, min_duration: float = 2.0, step: int = 3) -> list[dict]:
    """Subsampled ground-point polylines (normalised coordinates)."""
    out = []
    for tr in an.tracks:
        if tr.duration < min_duration or tr.category == "animal":
            continue
        p = tr.pos[::step] / [an.info.width, an.info.height]
        out.append({"id": tr.id, "cls": tr.category, "t0": round(float(tr.t[0]), 2),
                    "pts": p.round(4).tolist()})
    return out


def motion_images(an: Analysis, background: np.ndarray) -> dict[str, np.ndarray]:
    """Heat map of moving vehicles, trajectory overlay and the direction field."""
    h, w = background.shape[:2]
    heat = np.zeros((h // 4, w // 4), np.float32)
    traj = background.copy()
    flow = np.zeros((GRID[1], GRID[0], 2))
    cnt = np.zeros((GRID[1], GRID[0]))
    palette = {"person": (255, 200, 0), "bicycle": (0, 255, 0), "motorcycle": (0, 255, 0),
               "car": (0, 0, 255), "bus": (0, 165, 255), "truck": (0, 165, 255)}
    for tr in an.tracks:
        moving = tr.speed > 0.3
        for (x, y), m in zip(tr.pos, moving):
            if m and 0 <= x < w and 0 <= y < h:
                heat[int(y) // 4, int(x) // 4] += 1
        if tr.duration >= 2.0 and tr.category in palette:
            cv2.polylines(traj, [np.int32(tr.pos)], False, palette[tr.category], 1, cv2.LINE_AA)
        if tr.is_vehicle:
            unit = tr.vel / (np.linalg.norm(tr.vel, axis=1, keepdims=True) + 1e-9)
            for (x, y), u, m in zip(tr.pos, unit, moving):
                gx, gy = int(x / w * GRID[0]), int(y / h * GRID[1])
                if m and 0 <= gx < GRID[0] and 0 <= gy < GRID[1]:
                    flow[gy, gx] += u
                    cnt[gy, gx] += 1
    heat = cv2.GaussianBlur(heat, (0, 0), 3)
    heat = (255 * np.sqrt(heat / (heat.max() + 1e-9))).astype(np.uint8)
    heat = cv2.resize(cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO), (w, h))
    heat_img = cv2.addWeighted(background, 0.45, heat, 0.75, 0)

    arrows = (background * 0.55).astype(np.uint8)
    cw, ch = w / GRID[0], h / GRID[1]
    for gy in range(GRID[1]):
        for gx in range(GRID[0]):
            if cnt[gy, gx] < 5:
                continue
            u = flow[gy, gx] / cnt[gy, gx]
            coherence = float(np.linalg.norm(u))
            if coherence < 0.3:
                continue
            c = np.array([(gx + 0.5) * cw, (gy + 0.5) * ch])
            tip = c + u / coherence * 0.45 * cw
            ang = (np.degrees(np.arctan2(u[1], u[0])) + 360) % 360
            color = cv2.cvtColor(np.uint8([[[ang / 2, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
            cv2.arrowedLine(arrows, tuple(np.int32(c)), tuple(np.int32(tip)), color, 2, cv2.LINE_AA, tipLength=0.35)
    return {"heatmap": heat_img, "trajectories": traj, "directions": arrows}


def eda(an: Analysis) -> dict:
    return {
        "summary": summary(an),
        "counts": counts_over_time(an),
        "zones": zone_occupancy(an),
        "speeds": speed_histogram(an),
        "lighting": lighting(an),
        "signal": {"t": an.signal.t[::5].round(2).tolist(), "state": an.signal.code[::5].tolist()},
    }
