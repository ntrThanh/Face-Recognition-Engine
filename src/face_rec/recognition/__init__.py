from __future__ import annotations

from typing import Any

from face_rec.engines import build_engine
from face_rec.recognition.base import Recognizer
from face_rec.registry import Registry

RECOGNIZERS: Registry[Recognizer] = Registry("recognizer")


def build_recognizer(cfg: dict[str, Any]) -> Recognizer:
    cls = RECOGNIZERS.get(cfg["type"])
    return cls(cfg, build_engine(cfg["engines"]))


from face_rec.recognition import arcface, cavaface  # noqa: E402,F401

__all__ = ["RECOGNIZERS", "Recognizer", "build_recognizer"]
