"""High-level entry points shared by the website builder and the live demo."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2

from .config import Settings
from .eda import eda, motion_images, trajectories
from .pipeline import Analysis, analyze
from .render import render
from .risk import CausalRisk
from .rules import detect
from .video import background_frame, iter_frames, video_info
from .viz import draw_scene

Progress = Callable[[float, str], None]


def risk_curve(video_path: str, settings: Settings | None = None,
               progress: Progress | None = None) -> list[list[float]]:
    """Same stream the harness produces: one [t, score] per frame, causal."""
    settings = settings or Settings()
    info = video_info(video_path)
    est = CausalRisk(settings)
    stride = settings.risk_stride_for(info.fps)
    curve, last = [], 0.0
    for idx, frame in iter_frames(video_path, 1):
        t = idx / info.fps
        if idx % stride == 0:
            last = est.update(frame, t)
            if progress and idx % 50 == 0:
                progress(idx / max(1, info.n_frames), "risk")
        curve.append([round(t, 4), round(last, 4)])
    return curve


@dataclass
class VideoReport:
    analysis: Analysis
    events: list[list]
    risk: list[list[float]]
    stats: dict


def process(video_path: str, settings: Settings | None = None, events: list[list] | None = None,
            risk: list[list[float]] | None = None, progress: Progress | None = None) -> VideoReport:
    """Run Part A (+ Part B unless ``risk`` is given) and the EDA on one video."""
    settings = settings or Settings()
    an = analyze(video_path, settings, progress=(lambda f: progress(f, "detect")) if progress else None)
    if events is None:
        events = detect(an)
    if risk is None:
        risk = risk_curve(video_path, settings, progress)
    return VideoReport(an, events, risk, eda(an))


def export_media(report: VideoReport, out_dir: Path, stem: str, width: int = 960,
                 video_width: int = 960, video_stride: int = 2, crf: int = 28) -> dict[str, str]:
    """Annotated video + EDA images for one video; returns file names by kind."""
    out_dir.mkdir(parents=True, exist_ok=True)
    an = report.analysis
    files = {"video": f"{stem}.mp4"}
    render(an, report.events, report.risk, out_dir / files["video"],
           max_width=video_width, out_stride=video_stride, crf=crf)
    bg = background_frame(an.video_path)
    images = {"background": bg, "scene": draw_scene(bg, an.scene), **motion_images(an, bg)}
    scale = width / bg.shape[1]
    for kind, img in images.items():
        files[kind] = f"{stem}_{kind}.jpg"
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / files[kind]), small, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return files


def web_payload(report: VideoReport, name: str, files: dict[str, str], risk_hz: float = 5.0) -> dict:
    """JSON for the website: events, a thinned risk curve, EDA and trajectories."""
    fps = report.analysis.info.fps
    step = max(1, int(round(fps / risk_hz)))
    return {
        "name": name,
        "media": files,
        "events": report.events,
        "risk": report.risk[::step],
        "eda": report.stats,
        "trajectories": trajectories(report.analysis),
    }
