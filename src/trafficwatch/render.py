"""Annotated playback: boxes + trails, active events, risk meter, event timeline.

Encoded as H.264 (yuv420p) through the ffmpeg binary shipped with
imageio-ffmpeg so the files play in every browser.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from .pipeline import Analysis
from .video import iter_frames
from .viz import EVENT_COLORS, draw_banner, draw_tracks

TRAIL_SEC = 2.0
TIMELINE_H = 26


class H264Writer:
    def __init__(self, path: str | Path, width: int, height: int, fps: float, crf: int = 28):
        import imageio_ffmpeg

        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps:.3f}",
               "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed while encoding")


def _draw_timeline(frame: np.ndarray, t: float, duration: float, events: list[list]) -> None:
    h, w = frame.shape[:2]
    y0 = h - TIMELINE_H
    cv2.rectangle(frame, (0, y0), (w, h), (25, 25, 25), -1)
    for s, e, label in events:
        x1, x2 = int(s / duration * w), max(int(s / duration * w) + 2, int(e / duration * w))
        cv2.rectangle(frame, (x1, y0 + 5), (x2, h - 5), EVENT_COLORS.get(label, (200, 200, 200)), -1)
    x = int(t / max(duration, 1e-6) * w)
    cv2.line(frame, (x, y0), (x, h), (255, 255, 255), 2)


def render(an: Analysis, events: list[list], risk: list[list] | None, out_path: str | Path,
           max_width: int = 960, out_stride: int = 2) -> Path:
    """Write the annotated video; ``risk`` is the harness curve [[t, score], ...]."""
    info = an.info
    scale = min(1.0, max_width / info.width)
    w, h = int(info.width * scale) // 2 * 2, int(info.height * scale) // 2 * 2
    writer = H264Writer(out_path, w, h + TIMELINE_H, info.fps / out_stride)

    by_time: dict[float, list] = {}
    for tr in an.tracks:
        for k, t in enumerate(tr.t):
            by_time.setdefault(round(float(t), 3), []).append((tr, k))
    keys = np.array(sorted(by_time))
    risk_t = np.array([r[0] for r in risk]) if risk else None
    risk_v = np.array([r[1] for r in risk]) if risk else None
    try:
        for idx, frame in iter_frames(an.video_path, out_stride):
            t = idx / info.fps
            canvas = np.zeros((h + TIMELINE_H, w, 3), np.uint8)
            canvas[:h] = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            if len(keys):
                near = keys[int(np.clip(np.searchsorted(keys, t), 0, len(keys) - 1))]
                objects, trails = [], {}
                if abs(near - t) <= 0.2:
                    for tr, k in by_time[float(near)]:
                        objects.append((tr.id, tr.boxes[k] * scale, tr.category))
                        sel = (tr.t <= tr.t[k]) & (tr.t >= tr.t[k] - TRAIL_SEC)
                        trails[tr.id] = tr.pos[sel] * scale
                draw_tracks(canvas[:h], objects, trails)
            active = [lab for s, e, lab in events if s <= t <= e]
            score = None
            if risk_t is not None and len(risk_t):
                score = float(risk_v[int(np.clip(np.searchsorted(risk_t, t), 0, len(risk_t) - 1))])
            draw_banner(canvas, t, active, score)
            _draw_timeline(canvas, t, info.duration, events)
            writer.write(canvas)
    finally:
        writer.close()
    return Path(out_path)
