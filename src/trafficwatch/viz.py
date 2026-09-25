"""Drawing helpers shared by the renderer, the scene checker and the demo."""
from __future__ import annotations

import colorsys

import cv2
import numpy as np

from .scene import Scene

EVENT_COLORS = {  # BGR
    "accident": (40, 40, 230),
    "near_miss": (0, 140, 255),
    "red_light": (60, 60, 200),
    "wrong_way": (200, 0, 200),
    "illegal_u_turn": (180, 80, 250),
    "stopped_vehicle": (0, 215, 255),
    "jaywalking": (255, 180, 0),
    "failure_to_yield": (255, 120, 60),
    "illegal_turn": (150, 60, 220),
    "solid_line_crossing": (220, 220, 0),
    "stop_line": (80, 80, 255),
    "congestion": (0, 100, 180),
    "road_obstacle": (100, 200, 100),
    "fire_smoke": (0, 69, 255),
}


def track_color(track_id: int) -> tuple[int, int, int]:
    h = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def _text(img, text: str, org, scale: float, color, thickness: int = 1) -> None:
    cv2.putText(img, text, (int(org[0]), int(org[1])), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _polyline(img, pts, color, closed=True, thickness=2):
    cv2.polylines(img, [np.round(pts).astype(np.int32)], closed, color, thickness, cv2.LINE_AA)


def draw_scene(frame: np.ndarray, scene: Scene, alpha: float = 0.35) -> np.ndarray:
    """Overlay carriageways, islands, crosswalks, stop lines, lane lines and signal boxes."""
    out = frame.copy()
    layer = frame.copy()
    for poly in scene.roads.values():
        cv2.fillPoly(layer, [np.round(poly).astype(np.int32)], (255, 120, 0))
    for poly in scene.crosswalks.values():
        cv2.fillPoly(layer, [np.round(poly).astype(np.int32)], (255, 255, 255))
    for poly in scene.islands.values():
        cv2.fillPoly(layer, [np.round(poly).astype(np.int32)], (80, 80, 80))
    out = cv2.addWeighted(layer, alpha, out, 1 - alpha, 0)
    for name, poly in scene.roads.items():
        _polyline(out, poly, (255, 160, 0), thickness=1)
        c = poly.mean(axis=0)
        _text(out, name, (int(c[0]), int(c[1])), 0.5, (255, 255, 255), 1)
    for ap in scene.approaches.values():
        _polyline(out, ap.stop_line, (0, 0, 255), closed=False, thickness=3)
        mid = ap.stop_line.mean(axis=0)
        tip = mid + ap.direction * 60
        cv2.arrowedLine(out, tuple(np.int32(mid)), tuple(np.int32(tip)), (0, 0, 255), 2, tipLength=0.3)
    for zone, direction in scene.wrong_way_zones.values():
        c = zone.mean(axis=0)
        cv2.arrowedLine(out, tuple(np.int32(c)), tuple(np.int32(c + direction * 80)), (0, 255, 0), 3, tipLength=0.3)
    for line in scene.solid_lines.values():
        _polyline(out, line, (0, 255, 255), closed=False, thickness=2)
    for name, b in scene.signal_lights.items():
        cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
    return out


def draw_tracks(frame: np.ndarray, objects, trails: dict[int, list] | None = None) -> None:
    """Draw boxes (in place). ``objects``: iterable of (track_id, box, label)."""
    for tid, box, label in objects:
        color = track_color(tid)
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        _text(frame, f"{label} {tid}", (x1, max(12, y1 - 4)), 0.45, color, 1)
        if trails and tid in trails and len(trails[tid]) > 1:
            cv2.polylines(frame, [np.int32(trails[tid])], False, color, 2, cv2.LINE_AA)


def draw_banner(frame: np.ndarray, t: float, active: list[str], risk: float | None) -> None:
    """Top bar with the timestamp, the active events and the risk meter (in place)."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 34), (20, 20, 20), -1)
    _text(frame, f"t={t:7.2f}s", (8, 23), 0.6, (255, 255, 255), 1)
    x = 130
    for label in active:
        color = EVENT_COLORS.get(label, (255, 255, 255))
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x - 4, 5), (x + tw + 4, 29), color, -1)
        _text(frame, label, (x, 23), 0.55, (0, 0, 0), 2)
        x += tw + 16
    if risk is not None:
        bw = 160
        x0 = w - bw - 70
        _text(frame, "risk", (x0 - 42, 23), 0.55, (255, 255, 255), 1)
        cv2.rectangle(frame, (x0, 10), (x0 + bw, 26), (80, 80, 80), -1)
        color = (0, 0, 255) if risk >= 0.5 else (0, 200, 255) if risk >= 0.25 else (0, 200, 0)
        cv2.rectangle(frame, (x0, 10), (x0 + int(bw * risk), 26), color, -1)
        _text(frame, f"{risk:.2f}", (x0 + bw + 8, 23), 0.55, (255, 255, 255), 1)
