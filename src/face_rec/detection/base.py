"""Face detector interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from face_rec.engines import InferenceEngine
from face_rec.types import Face


class Detector(ABC):
    registry_name: str = ""

    def __init__(self, cfg: dict[str, Any], engine: InferenceEngine):
        self.cfg = cfg
        self.engine = engine
        self.input_size = tuple(cfg.get("input_size", (640, 640)))   # (w, h)
        self.score_threshold = float(cfg.get("score_threshold", 0.5))
        self.nms_threshold = float(cfg.get("nms_threshold", 0.4))

    @abstractmethod
    def detect(self, img_bgr: np.ndarray) -> list[Face]:
        """Return faces with bbox, score and 5-point landmarks in original image coords."""
