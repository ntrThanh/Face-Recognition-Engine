"""Data types passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Face:
    bbox: np.ndarray                     # (4,) x1, y1, x2, y2 in original image pixels
    score: float
    landmarks: np.ndarray | None = None  # (5, 2) eyes, nose, mouth corners
    embedding: np.ndarray | None = None  # (D,) L2-normalized

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])


@dataclass
class Match:
    name: str
    similarity: float


@dataclass
class RecognitionResult:
    face: Face
    matches: list[Match] = field(default_factory=list)
    threshold: float = 0.0

    @property
    def name(self) -> str:
        if self.matches and self.matches[0].similarity >= self.threshold:
            return self.matches[0].name
        return "unknown"

    @property
    def similarity(self) -> float:
        return self.matches[0].similarity if self.matches else 0.0
