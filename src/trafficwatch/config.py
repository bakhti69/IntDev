"""Runtime settings. Defaults are the official configuration; the environment
variables below exist for the CPU demo and for ablations."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = ROOT / "weights"
SEED = 0


def _env(name: str, default):
    raw = os.environ.get(name)
    return type(default)(raw) if raw not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    weights: str = field(default_factory=lambda: _env("TW_WEIGHTS", str(WEIGHTS_DIR / "yolo11s.pt")))
    imgsz: int = field(default_factory=lambda: _env("TW_IMGSZ", 960))
    device: str = field(default_factory=lambda: _env("TW_DEVICE", ""))
    # Part A: analyse every n-th frame (25 fps / 2 = 12.5 Hz)
    stride: int = field(default_factory=lambda: _env("TW_STRIDE", 2))
    batch: int = field(default_factory=lambda: _env("TW_BATCH", 8))
    # Part B: run the detector on every n-th frame, hold the score in between
    risk_stride: int = field(default_factory=lambda: _env("TW_RISK_STRIDE", 3))
    risk_imgsz: int = field(default_factory=lambda: _env("TW_RISK_IMGSZ", 960))

    @property
    def device_or_none(self) -> str | None:
        return self.device or None


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import cv2
        cv2.setRNGSeed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass
