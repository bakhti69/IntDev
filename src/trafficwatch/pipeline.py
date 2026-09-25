"""One pass over a video: detection -> tracking -> per-frame scene cues.

The result (:class:`Analysis`) is everything the event rules need, so every
rule is a pure function of it and can be unit-tested with synthetic tracks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from .config import Settings, seed_everything
from .detector import Detections, get_detector
from .scene import Scene, load_scene
from .signals import SignalTimeline, read_heads
from .tracker import Tracker
from .tracks import TrackData, TrackHistory, finalize
from .video import VideoInfo, background_frame, iter_frames, video_info

CUE_WIDTH = 384          # width of the low-res frames kept for pixel cues
CUE_PERIOD = 0.5         # seconds between cue frames
MIN_TRACK_SAMPLES = 5


@dataclass
class SceneCues:
    """Low-rate, low-resolution pixel evidence for scene-level classes."""
    scale: float                                   # cue px per frame px
    t: list[float] = field(default_factory=list)
    gray: list[np.ndarray] = field(default_factory=list)      # uint8 (h, w)
    fire: list[np.ndarray] = field(default_factory=list)      # bool (h, w)
    sat: list[np.ndarray] = field(default_factory=list)       # uint8 (h, w) HSV saturation
    boxes: list[np.ndarray] = field(default_factory=list)     # (n, 4) detections, cue px


@dataclass
class Analysis:
    video_path: str
    info: VideoInfo
    scene: Scene
    times: np.ndarray
    tracks: list[TrackData]
    signals: dict[str, SignalTimeline]            # per signal head
    cues: SceneCues
    counts: dict[str, np.ndarray]                  # per processed frame, per category
    _zones: dict[int, list] = field(default_factory=dict, repr=False)

    def zones(self, tr: TrackData) -> list[str | None]:
        """Road zone of the track's ground point at every sample (cached)."""
        if tr.id not in self._zones:
            self._zones[tr.id] = [self.scene.zone_of(x, y) for x, y in tr.pos]
        return self._zones[tr.id]

    def tracks_where(self, pred: Callable[[TrackData], bool]) -> list[TrackData]:
        return [tr for tr in self.tracks if pred(tr)]


def fire_mask(small_bgr: np.ndarray) -> np.ndarray:
    """Flame-coloured pixels: bright, saturated red/orange/yellow with R >> B."""
    b, g, r = (small_bgr[..., i].astype(np.int16) for i in range(3))
    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (r > 190) & (r >= g) & (g > b) & (r - b > 90) & (h <= 30) & (s > 90) & (v > 190)


def _stitch_static(histories: list[TrackHistory], max_gap: float = 8.0) -> list[TrackHistory]:
    """Join fragments of the same parked object split by long occlusions."""
    def ground(box):
        size = float(np.sqrt(max(1.0, (box[2] - box[0]) * (box[3] - box[1]))))
        return np.array([(box[0] + box[2]) / 2, box[3]]), size

    histories = sorted(histories, key=lambda h: h.t[0])
    merged: list[TrackHistory] = []
    for h in histories:
        p0, s0 = ground(h.boxes[0])
        best = None
        for m in merged:
            if m.t[-1] >= h.t[0] or h.t[0] - m.t[-1] > max_gap:
                continue
            p1, s1 = ground(m.boxes[-1])
            tail = np.array([ground(b)[0] for b in m.boxes[-5:]])
            static = np.ptp(tail, axis=0).max() < 0.15 * s1
            if static and np.linalg.norm(p0 - p1) < 0.3 * s1 and 0.7 < s0 / s1 < 1.4 \
                    and m.votes.most_common(1)[0][0] == h.votes.most_common(1)[0][0]:
                best = m
                break
        if best is None:
            merged.append(h)
        else:
            best.t += h.t
            best.boxes += h.boxes
            best.votes += h.votes
    return merged


def analyze(video_path: str, settings: Settings | None = None,
            progress: Callable[[float], None] | None = None) -> Analysis:
    settings = settings or Settings()
    seed_everything()
    info = video_info(video_path)
    scene = load_scene(background_frame(video_path), info.width, info.height)
    detector = get_detector(settings.weights, settings.imgsz, settings.device_or_none)
    tracker = Tracker()

    cue_scale = CUE_WIDTH / max(1, info.width)
    cues = SceneCues(scale=cue_scale)
    histories: dict[int, TrackHistory] = {}
    times: list[float] = []
    sig_s: dict[str, list[str]] = {name: [] for name in scene.signal_lights}
    counts: dict[str, list[int]] = {c: [] for c in ("person", "bicycle", "car", "motorcycle", "bus", "truck", "animal")}
    next_cue = 0.0

    def consume(batch: list[tuple[int, np.ndarray]]) -> None:
        nonlocal next_cue
        dets_list = detector([f for _, f in batch])
        for (idx, frame), dets in zip(batch, dets_list):
            t = idx / info.fps
            times.append(t)
            for c in counts:
                counts[c].append(sum(1 for x in dets.cats if x == c))
            for tr in tracker.update(dets, t):
                histories.setdefault(tr.id, TrackHistory(tr.id)).add(t, tr.box, tr.category)
            for name, state in read_heads(frame, scene.signal_lights, scene.signal_kinds).items():
                sig_s[name].append(state)
            if t + 1e-6 >= next_cue:
                _add_cue(cues, frame, dets, t)
                next_cue = t + CUE_PERIOD
        if progress and info.n_frames:
            progress(min(1.0, (batch[-1][0] + 1) / info.n_frames))

    batch: list[tuple[int, np.ndarray]] = []
    for idx, frame in iter_frames(video_path, settings.stride):
        batch.append((idx, frame))
        if len(batch) == settings.batch:
            consume(batch)
            batch = []
    if batch:
        consume(batch)

    hist = [h for h in _stitch_static(list(histories.values())) if len(h.t) >= MIN_TRACK_SAMPLES]
    for h in hist:  # stitched fragments must be time-ordered
        order = np.argsort(h.t, kind="stable")
        h.t = [h.t[i] for i in order]
        h.boxes = [h.boxes[i] for i in order]
    return Analysis(
        video_path=str(video_path), info=info, scene=scene,
        times=np.asarray(times), tracks=[finalize(h) for h in hist],
        signals={name: SignalTimeline.build(times, states) for name, states in sig_s.items()}, cues=cues,
        counts={k: np.asarray(v) for k, v in counts.items()},
    )


def _add_cue(cues: SceneCues, frame: np.ndarray, dets: Detections, t: float) -> None:
    small = cv2.resize(frame, None, fx=cues.scale, fy=cues.scale, interpolation=cv2.INTER_AREA)
    cues.t.append(t)
    cues.gray.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    cues.sat.append(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 1])
    cues.fire.append(fire_mask(small))
    cues.boxes.append(dets.boxes * cues.scale)
