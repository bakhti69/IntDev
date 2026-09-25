"""Road-user detector: a COCO-pretrained YOLO (Ultralytics) with a fixed class map."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

# COCO ids -> our coarse categories
COCO_TO_CATEGORY = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    15: "animal", 16: "animal", 17: "animal", 18: "animal", 19: "animal",
}
VEHICLES = {"car", "bus", "truck"}
TWO_WHEELERS = {"bicycle", "motorcycle"}
CATEGORIES = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "animal"]


def category_group(cat: str) -> str:
    """Categories that may be confused for one another share a group (tracking)."""
    if cat in VEHICLES:
        return "vehicle"
    if cat in TWO_WHEELERS:
        return "two_wheeler"
    return cat


@dataclass
class Detections:
    boxes: np.ndarray    # (N, 4) xyxy float32, frame pixels
    scores: np.ndarray   # (N,)
    cats: list[str]      # (N,)

    def __len__(self) -> int:
        return len(self.scores)

    @staticmethod
    def empty() -> "Detections":
        return Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), [])


class Detector:
    def __init__(self, weights: str, imgsz: int = 960, device: str | None = None,
                 conf: float = 0.15, iou: float = 0.6):
        os.environ.setdefault("YOLO_OFFLINE", "1")  # evaluation runs without internet
        import torch
        from ultralytics import YOLO

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.half = self.device.startswith("cuda")
        self.model = YOLO(weights)
        self.imgsz, self.conf, self.iou = imgsz, conf, iou
        self.classes = sorted(COCO_TO_CATEGORY)

    def __call__(self, frames: list[np.ndarray]) -> list[Detections]:
        if not frames:
            return []
        extra = {"half": True} if self.half else {}
        results = self.model.predict(frames, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                     classes=self.classes, device=self.device, verbose=False, **extra)
        out = []
        for r in results:
            b = r.boxes
            if b is None or len(b) == 0:
                out.append(Detections.empty())
                continue
            cls = b.cls.cpu().numpy().astype(int)
            out.append(Detections(b.xyxy.cpu().numpy().astype(np.float32),
                                  b.conf.cpu().numpy().astype(np.float32),
                                  [COCO_TO_CATEGORY[c] for c in cls]))
        return out


@lru_cache(maxsize=4)
def get_detector(weights: str, imgsz: int, device: str | None) -> Detector:
    """Process-wide cache so Part A and Part B share one loaded model."""
    return Detector(weights, imgsz=imgsz, device=device)
