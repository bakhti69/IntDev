"""Video reading helpers (OpenCV)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0


def video_info(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    info = VideoInfo(
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 25.0),
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return info


def iter_frames(path: str, stride: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, BGR frame) for every ``stride``-th frame.

    Skipped frames are grabbed but not decoded to BGR, which is cheaper than
    seeking on long-GOP H.264.
    """
    cap = cv2.VideoCapture(str(path))
    idx = 0
    try:
        while True:
            if idx % stride == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                yield idx, frame
            elif not cap.grab():
                break
            idx += 1
    finally:
        cap.release()


def background_frame(path: str, n_samples: int = 11, span: float = 10.0) -> np.ndarray | None:
    """Temporal median of frames from the first ``span`` seconds (removes moving traffic).

    Reads strictly in order: random seeks in long 4K H.264/H.265 camera files are
    slow and have crashed OpenCV's FFmpeg reader on the sample videos.
    """
    info = video_info(path)
    last = max(1, min(info.n_frames, int(span * info.fps)))
    wanted = set(np.linspace(0, last - 1, n_samples).astype(int).tolist())
    frames = []
    cap = cv2.VideoCapture(str(path))
    try:
        for idx in range(last):
            if idx in wanted:
                ok, f = cap.read()
                if not ok:
                    break
                frames.append(f)
            elif not cap.grab():
                break
    finally:
        cap.release()
    if not frames:
        return None
    return np.median(np.stack(frames), axis=0).astype(np.uint8)
