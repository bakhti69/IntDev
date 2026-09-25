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


def _has_gpu() -> bool:
    if os.environ.get("TW_DEVICE", "").startswith("cpu"):
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _env(name: str, gpu_default, cpu_default=None):
    """Environment override, else the GPU default, else (no GPU) a lighter CPU default
    that keeps a 1080p video inside the 3x-duration time budget."""
    raw = os.environ.get(name)
    if raw not in (None, ""):
        return type(gpu_default)(raw)
    return gpu_default if cpu_default is None or _has_gpu() else cpu_default


@dataclass(frozen=True)
class Settings:
    weights: str = field(default_factory=lambda: _env("TW_WEIGHTS", str(WEIGHTS_DIR / "yolo11s.pt")))
    imgsz: int = field(default_factory=lambda: _env("TW_IMGSZ", 960, 640))
    device: str = field(default_factory=lambda: _env("TW_DEVICE", ""))
    # Part A: analyse every n-th frame (25 fps / 2 = 12.5 Hz)
    stride: int = field(default_factory=lambda: _env("TW_STRIDE", 2, 3))
    batch: int = field(default_factory=lambda: _env("TW_BATCH", 8, 4))
    # Part B: run the detector on every n-th frame, hold the score in between
    risk_stride: int = field(default_factory=lambda: _env("TW_RISK_STRIDE", 3, 6))
    risk_imgsz: int = field(default_factory=lambda: _env("TW_RISK_IMGSZ", 960, 640))

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
