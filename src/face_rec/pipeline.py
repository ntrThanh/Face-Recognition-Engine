"""End-to-end pipeline: detect -> align -> embed -> match."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from face_rec.alignment import align_face
from face_rec.config import load_config, resolve_path
from face_rec.database import build_database
from face_rec.detection import build_detector
from face_rec.recognition import build_recognizer
from face_rec.types import Face, RecognitionResult
from face_rec.utils.timer import StageTimer, format_stats


class FaceRecognizer:
    def __init__(self, cfg: dict[str, Any], reset_db: bool = False):
        self.cfg = cfg
        # OpenCV's worker threads busy-wait after each call; with sparse calls (a few
        # per frame, frames 30+ ms apart) that burns ~20x the CPU of the work itself.
        cv2.setNumThreads(int(cfg.get("opencv_threads", 1)))
        self.detector = build_detector(cfg["detector"])
        self.recognizer = build_recognizer(cfg["recognizer"])
        db_cfg = {**cfg["database"], "path": str(resolve_path(cfg["database"]["path"]))}
        self.db = build_database(
            db_cfg, self.recognizer.model_id, self.recognizer.embedding_dim, reset=reset_db
        )
        self.min_face_size = int(cfg.get("min_face_size", 0))
        self.timer = StageTimer()

    @classmethod
    def from_config(cls, path: str | Path, reset_db: bool = False) -> "FaceRecognizer":
        return cls(load_config(path), reset_db=reset_db)

    def detect(self, img_bgr: np.ndarray) -> list[Face]:
        with self.timer("detect"):
            faces = self.detector.detect(img_bgr)
        return [f for f in faces if min(f.width, f.height) >= self.min_face_size]

    def embed_faces(self, img_bgr: np.ndarray, faces: list[Face]) -> list[Face]:
        if not faces:
            return faces
        with self.timer("align"):
            crops = [
                align_face(img_bgr, f.landmarks, self.recognizer.align_template) for f in faces
            ]
        with self.timer("embed"):
            embs = self.recognizer.embed(crops)
        for f, e in zip(faces, embs):
            f.embedding = e
        return faces

    def match(self, face: Face) -> RecognitionResult:
        with self.timer("match"):
            return RecognitionResult(face, self.db.search(face.embedding), self.db.threshold)

    def recognize(self, img_bgr: np.ndarray) -> list[RecognitionResult]:
        faces = self.embed_faces(img_bgr, self.detect(img_bgr))
        return [self.match(f) for f in faces]

    def timing_stats(self) -> dict[str, dict[str, float]]:
        """Stage timings plus pure engine time for each model, in pipeline order.

        detect/embed include pre/post-processing; the indented `infer` rows are the
        time spent inside the runtime (NPU compute + transfers) for that model.
        embed is per call (all faces of a frame); its infer row is per face.
        """
        stages = self.timer.stats()
        engines = {
            "detect": self.detector.engine,
            "embed": self.recognizer.engine,
        }
        order = ["frame", "latency", "detect", "track", "align", "embed", "match"]
        rows = {}
        for name in order + [k for k in stages if k not in order]:
            if name in stages:
                rows[name] = stages[name]
            if name in engines:
                infer = engines[name].timer.stats().get("infer")
                if infer:
                    model = Path(engines[name].model_path or "?").stem
                    rows[f"  \u2514 infer {model}"] = infer
        return rows

    def timing_report(self) -> str:
        return format_stats(self.timing_stats())

    def close(self) -> None:
        self.db.close()
