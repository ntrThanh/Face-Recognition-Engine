"""Modular face recognition: detection -> alignment -> recognition -> matching."""

from face_rec.pipeline import FaceRecognizer
from face_rec.stream import StreamRecognizer

__all__ = ["FaceRecognizer", "StreamRecognizer"]
__version__ = "0.1.0"
