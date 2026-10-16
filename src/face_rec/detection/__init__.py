from __future__ import annotations

from typing import Any

from face_rec.detection.base import Detector
from face_rec.engines import build_engine
from face_rec.registry import Registry

DETECTORS: Registry[Detector] = Registry("detector")


def build_detector(cfg: dict[str, Any]) -> Detector:
    cls = DETECTORS.get(cfg["type"])
    return cls(cfg, build_engine(cfg["engines"]))


from face_rec.detection import face_det_lite, scrfd  # noqa: E402,F401

__all__ = ["DETECTORS", "Detector", "build_detector"]
