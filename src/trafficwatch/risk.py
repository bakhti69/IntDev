"""Part B: causal accident-risk score from live tracks.

Only frames already seen are used. About ``risk_rate`` frames per second go
through the detector and the (causal) tracker; each road user keeps a short
history from which we estimate velocity and deceleration. The score combines

* the closest point of approach of every pair of road users: how close
  their paths will get and how soon (side-by-side passing is not a conflict),
* hard braking next to another road user,
* a vehicle driving against the flow,

as 1 - prod(1 - r_i), floored by a small TTC-based baseline so that frames
stay ranked even far below the alarm threshold,, then smoothed with a fast attack / slow release so an
alarm starts early and does not flicker. 0.5 means "paths meet within
about a second".
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import Settings, seed_everything
from .detector import TWO_WHEELERS, VEHICLES, get_detector
from .scene import Scene, load_scene
from .tracker import Tracker
from .tracks import closest_approach

HISTORY = 4.0            # s of history per track
VEL_WINDOW = 0.6         # s used for the velocity fit
RELEASE_TAU = 1.5        # s, decay of the smoothed score
PAIR_MAX_DIST = 3.0      # normalised distance
CPA_HIT = 0.35           # closest approach (body lengths) that counts as a collision course
TCPA_ALARM = 1.0         # s to the closest approach at which the score passes 0.5


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class _Hist:
    category: str
    t: deque = field(default_factory=lambda: deque(maxlen=64))
    pos: deque = field(default_factory=lambda: deque(maxlen=64))
    size: deque = field(default_factory=lambda: deque(maxlen=64))
    speed: deque = field(default_factory=lambda: deque(maxlen=64))

    def velocity(self) -> np.ndarray:
        t = np.asarray(self.t)
        sel = t >= t[-1] - VEL_WINDOW
        if sel.sum() < 3:
            return np.zeros(2)
        p = np.asarray(self.pos)[sel]
        tt = t[sel] - t[sel].mean()
        denom = float(tt @ tt) + 1e-9
        return (tt @ (p - p.mean(axis=0))) / denom

    def decel(self) -> float:
        """Speed change over the last ~0.6 s (body sizes / s^2, negative = braking)."""
        t = np.asarray(self.t)
        s = np.asarray(self.speed)
        sel = t >= t[-1] - VEL_WINDOW
        if sel.sum() < 3:
            return 0.0
        return float((s[sel][-1] - s[sel][0]) / max(t[sel][-1] - t[sel][0], 1e-3))


class CausalRisk:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.detector = get_detector(self.settings.weights, self.settings.risk_imgsz,
                                     self.settings.device_or_none)
        self.reset()

    def reset(self) -> None:
        seed_everything()
        self.tracker = Tracker()
        self.hist: dict[int, _Hist] = {}
        self.scene: Scene | None = None
        self.score = 0.0
        self.raw_prev = 0.0
        self.t_prev: float | None = None

    def update(self, frame: np.ndarray, t: float) -> float:
        if self.scene is None:
            self.scene = load_scene(frame, frame.shape[1], frame.shape[0])
        dets = self.detector([frame])[0]
        live = self.tracker.update(dets, t)
        for tr in live:
            h = self.hist.setdefault(tr.id, _Hist(tr.category))
            b = tr.box
            h.t.append(t)
            h.pos.append(np.array([(b[0] + b[2]) / 2, b[3]], np.float64))
            h.size.append(float(np.sqrt(max(1.0, (b[2] - b[0]) * (b[3] - b[1])))))
            v = h.velocity()
            h.speed.append(float(np.linalg.norm(v)) / h.size[-1])
        self.hist = {k: h for k, h in self.hist.items() if t - h.t[-1] <= HISTORY}
        raw = self._raw_risk([self.hist[tr.id] for tr in live if tr.id in self.hist], t)
        dt = 0.0 if self.t_prev is None else t - self.t_prev
        self.t_prev = t
        attack = 0.5 * (raw + self.raw_prev)          # needs two frames to fire
        self.raw_prev = raw
        self.score = max(attack, self.score * math.exp(-dt / RELEASE_TAU))
        return float(min(1.0, max(0.0, self.score)))

    def _raw_risk(self, users: list[_Hist], t: float) -> float:
        users = [u for u in users if u.t[-1] == t and len(u.t) >= 3]
        if not users:
            return 0.0
        pos = np.array([u.pos[-1] for u in users])
        vel = np.array([u.velocity() for u in users])
        size = np.array([u.size[-1] for u in users])
        speed = np.linalg.norm(vel, axis=1) / size
        is_vehicle = np.array([u.category in VEHICLES or u.category in TWO_WHEELERS for u in users])
        risks, soft = [], 0.0
        n = len(users)
        for i in range(n):
            for j in range(i + 1, n):
                if not (is_vehicle[i] or is_vehicle[j]):
                    continue
                scale = min(size[i], size[j])  # a car passing a bus is measured in car lengths
                rel = (pos[j] - pos[i]) / scale
                if float(np.linalg.norm(rel)) > PAIR_MAX_DIST:
                    continue
                t_cpa, d_cpa, v_rel = closest_approach(rel, (vel[j] - vel[i]) / scale)
                if t_cpa is None:
                    continue
                # on a collision course: paths meet (not side by side), soon, at speed
                course = _sigmoid((CPA_HIT - d_cpa) / 0.12)
                r = course * _sigmoid((TCPA_ALARM - t_cpa) / 0.35) * _sigmoid((v_rel - 1.0) / 0.2)
                braking = min(users[i].decel(), users[j].decel())
                if braking < -1.5 and course > 0.5 and t_cpa < 2.0:
                    r = max(r, 0.4)  # someone brakes hard on a collision course
                risks.append(r)
                # sub-alarm baseline (< 0.2) so that frames stay ranked for AP
                soft = max(soft, 0.2 * course * _sigmoid((2.5 - t_cpa) / 0.8))
        risks += self._wrong_way(pos, vel, speed, is_vehicle)
        combined = 1.0 - float(np.prod([1.0 - r for r in risks])) if risks else 0.0
        return max(combined, soft)

    def _wrong_way(self, pos, vel, speed, is_vehicle) -> list[float]:
        out = []
        for k in np.where(is_vehicle & (speed > 0.5))[0]:
            unit = vel[k] / (np.linalg.norm(vel[k]) + 1e-9)
            for poly, direction in self.scene.wrong_way_zones.values():
                if self.scene.in_polygon(poly, *pos[k]) and float(unit @ direction) < -0.5:
                    out.append(0.3)
        return out
