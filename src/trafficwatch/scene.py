"""Scene layout of the fixed camera and its registration onto a video.

The layout (carriageways, crosswalks, stop lines, signal heads, lane lines)
is drawn once on a reference frame (``configs/scene_ref.jpg``). Test videos
come from the same camera but may differ in resolution or crop, so at
runtime we estimate a similarity transform from the reference frame to the
video (SIFT features + RANSAC) and map every shape through it. If
registration fails we fall back to plain rescaling by the frame size.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
DEFAULT_SCENE = CONFIG_DIR / "scene.json"

MIN_INLIERS = 25


def _poly(points) -> np.ndarray:
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


@dataclass
class Approach:
    name: str
    zone: str
    direction: np.ndarray        # unit vector of legal travel, image coords
    stop_line: np.ndarray        # (2, 2) endpoints
    box_zone: np.ndarray         # polygon between stop line and intersection
    signal_head: str | None      # vehicle signal head read directly, if it faces the camera
    walk_head: str | None        # pedestrian head whose WALK implies red for this approach


@dataclass
class Scene:
    width: int
    height: int
    roads: dict[str, np.ndarray]
    islands: dict[str, np.ndarray]
    crosswalks: dict[str, np.ndarray]
    approaches: dict[str, Approach]
    wrong_way_zones: dict[str, tuple[np.ndarray, np.ndarray]]
    signal_lights: dict[str, np.ndarray]     # name -> [x1, y1, x2, y2]
    signal_kinds: dict[str, str]             # name -> "vehicle" | "pedestrian"
    solid_lines: dict[str, np.ndarray]
    forbidden_movements: list[tuple[str, str]]
    registered: bool = False
    inliers: int = 0
    _masks: dict = field(default_factory=dict, repr=False)

    # ---- point queries -------------------------------------------------
    def zone_of(self, x: float, y: float) -> str | None:
        """Name of the road zone containing the point, or None if off-road."""
        for name, poly in self.islands.items():
            if cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
                return None
        for name, poly in self.roads.items():
            if cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
                return name
        for name, poly in self.crosswalks.items():
            if cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
                return "crosswalk_" + name
        return None

    def road_depth(self, x: float, y: float) -> float:
        """Signed distance (px) inside the carriageway; negative when off-road."""
        best = -1e9
        for poly in list(self.roads.values()) + list(self.crosswalks.values()):
            best = max(best, cv2.pointPolygonTest(poly, (float(x), float(y)), True))
        for poly in self.islands.values():
            d = cv2.pointPolygonTest(poly, (float(x), float(y)), True)
            if d >= 0:
                return -d
        return float(best)

    def crosswalk_of(self, x: float, y: float, margin: float = 0.0) -> str | None:
        for name, poly in self.crosswalks.items():
            if cv2.pointPolygonTest(poly, (float(x), float(y)), True) >= -margin:
                return name
        return None

    def in_polygon(self, poly: np.ndarray, x: float, y: float, margin: float = 0.0) -> bool:
        return cv2.pointPolygonTest(poly, (float(x), float(y)), True) >= -margin

    def road_mask(self, scale: float = 1.0) -> np.ndarray:
        """Binary carriageway mask (uint8) at ``scale`` of the frame size."""
        key = ("road", scale)
        if key not in self._masks:
            h, w = int(round(self.height * scale)), int(round(self.width * scale))
            m = np.zeros((h, w), np.uint8)
            for poly in list(self.roads.values()) + list(self.crosswalks.values()):
                cv2.fillPoly(m, [np.round(poly * scale).astype(np.int32)], 1)
            for poly in self.islands.values():
                cv2.fillPoly(m, [np.round(poly * scale).astype(np.int32)], 0)
            self._masks[key] = m
        return self._masks[key]


def side_of_line(line: np.ndarray, x: float, y: float) -> float:
    """Signed distance of a point from the infinite line through ``line`` (px).

    Positive on the left of the direction p0 -> p1 in image coordinates.
    """
    (x0, y0), (x1, y1) = line[0], line[1]
    dx, dy = x1 - x0, y1 - y0
    n = np.hypot(dx, dy) + 1e-9
    return float((dx * (y - y0) - dy * (x - x0)) / n)


def distance_to_polyline(line: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """(unsigned distance, signed side) of a point relative to a polyline.

    The sign comes from the closest segment; points beyond the polyline's
    ends get +inf distance so they never count as crossing it.
    """
    best, sign = np.inf, 0.0
    p = np.array([x, y], dtype=np.float64)
    for a, b in zip(line[:-1], line[1:]):
        ab = b - a
        t = float(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9))
        if t < 0.0 or t > 1.0:
            continue
        proj = a + t * ab
        d = float(np.linalg.norm(p - proj))
        if d < best:
            best = d
            sign = float(np.sign(ab[0] * (p[1] - a[1]) - ab[1] * (p[0] - a[0])))
    return best, sign


# ---- loading and registration ------------------------------------------------

def _transform_points(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    ones = np.ones((len(pts), 1), np.float32)
    return (np.hstack([pts, ones]) @ M.T).astype(np.float32)


def estimate_transform(ref_gray: np.ndarray, frame_gray: np.ndarray) -> tuple[np.ndarray | None, int]:
    """Similarity transform (2x3) mapping reference pixels to frame pixels.

    SIFT on CLAHE-equalised images: robust to the day/dusk lighting change
    and to the small shifts/rotations seen between clips of this camera.
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    s_ref = 960.0 / ref_gray.shape[1]
    s_frm = 960.0 / frame_gray.shape[1]
    r = clahe.apply(cv2.resize(ref_gray, None, fx=s_ref, fy=s_ref, interpolation=cv2.INTER_AREA))
    f = clahe.apply(cv2.resize(frame_gray, None, fx=s_frm, fy=s_frm, interpolation=cv2.INTER_AREA))
    sift = cv2.SIFT_create(nfeatures=4000)
    kr, dr = sift.detectAndCompute(r, None)
    kf, df = sift.detectAndCompute(f, None)
    if dr is None or df is None or len(kr) < 10 or len(kf) < 10:
        return None, 0
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(dr, df, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < MIN_INLIERS:
        return None, len(good)
    src = np.float32([kr[m.queryIdx].pt for m in good]) / s_ref
    dst = np.float32([kf[m.trainIdx].pt for m in good]) / s_frm
    cv2.setRNGSeed(0)
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                         ransacReprojThreshold=4.0, maxIters=5000,
                                         confidence=0.999, refineIters=20)
    n_in = int(inl.sum()) if inl is not None else 0
    if M is None or n_in < MIN_INLIERS:
        return None, n_in
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    expected = frame_gray.shape[1] / ref_gray.shape[1]
    if not (0.7 * expected < scale < 1.4 * expected):
        return None, n_in
    return M.astype(np.float32), n_in


def load_scene(frame_bgr: np.ndarray | None, width: int, height: int,
               path: Path = DEFAULT_SCENE) -> Scene:
    """Load the layout and map it into the coordinate frame of a video.

    ``frame_bgr`` should be a representative (ideally vehicle-free, e.g. a
    temporal median) frame of the video; pass None to skip registration.
    """
    cfg = json.loads(Path(path).read_text())
    rw, rh = cfg["ref_size"]
    M, n_in = None, 0
    if frame_bgr is not None:
        ref = cv2.imread(str(Path(path).parent / cfg["ref_image"]), cv2.IMREAD_GRAYSCALE)
        if ref is not None:
            M, n_in = estimate_transform(ref, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
    registered = M is not None
    if M is None:
        M = np.array([[width / rw, 0, 0], [0, height / rh, 0]], np.float32)

    def tp(points) -> np.ndarray:
        return _transform_points(M, _poly(points))

    def box(b) -> np.ndarray:
        p = tp([[b[0], b[1]], [b[2], b[3]]])
        return np.array([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()], np.float32)

    def unit(v) -> np.ndarray:
        v = M[:, :2] @ np.asarray(v, np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

    heads = {k: v for k, v in cfg["signal_lights"].items() if not k.startswith("_")}
    approaches = {
        name: Approach(name=name, zone=a["zone"], direction=unit(a["direction"]),
                       stop_line=tp(a["stop_line"]), box_zone=tp(a["box_zone"]),
                       signal_head=a.get("signal_head"), walk_head=a.get("walk_head"))
        for name, a in cfg["approaches"].items()
    }
    return Scene(
        width=width, height=height,
        roads={k: tp(v) for k, v in cfg["road"].items()},
        islands={k: tp(v) for k, v in cfg["islands"].items()},
        crosswalks={k: tp(v) for k, v in cfg["crosswalks"].items()},
        approaches=approaches,
        wrong_way_zones={k: (tp(v["polygon"]), unit(v["direction"]))
                         for k, v in cfg["wrong_way_zones"].items()},
        signal_lights={k: box(v["box"]) for k, v in heads.items()},
        signal_kinds={k: v["kind"] for k, v in heads.items()},
        solid_lines={k: tp(v) for k, v in cfg["solid_lines"].items() if not k.startswith("_")},
        forbidden_movements=[tuple(p) for p in cfg["forbidden_movements"]["pairs"]],
        registered=registered, inliers=n_in,
    )
