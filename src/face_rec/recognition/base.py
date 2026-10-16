"""Face recognizer interface: aligned crops -> L2-normalized embeddings."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from face_rec.engines import InferenceEngine


class Recognizer:
    """Default implementation covers ArcFace-style models; subclass to override pre/post."""

    registry_name: str = ""

    def __init__(self, cfg: dict[str, Any], engine: InferenceEngine):
        self.cfg = cfg
        self.engine = engine
        self.model_id: str = cfg["model_id"]
        self.input_size = tuple(cfg.get("input_size", (112, 112)))  # (w, h)
        self.embedding_dim = int(cfg.get("embedding_dim", 512))
        self.align_template: str = cfg.get("align_template", "arcface_112")
        self.mean = float(cfg.get("mean", 127.5))
        self.std = float(cfg.get("std", 127.5))
        self.color = cfg.get("color", "RGB")

    def preprocess(self, faces_bgr: list[np.ndarray]) -> np.ndarray:
        batch = []
        for f in faces_bgr:
            if f.shape[1::-1] != self.input_size:
                f = cv2.resize(f, self.input_size)
            if self.color == "RGB":
                f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            batch.append(f)
        x = np.stack(batch).astype(np.float32)
        x = (x - self.mean) / self.std
        return x.transpose(0, 3, 1, 2)  # NHWC -> NCHW

    def postprocess(self, out: np.ndarray) -> np.ndarray:
        out = out.reshape(out.shape[0], -1)
        return out / np.linalg.norm(out, axis=1, keepdims=True).clip(min=1e-12)

    def embed(self, faces_bgr: list[np.ndarray]) -> np.ndarray:
        """(N, D) L2-normalized embeddings. Runs one face at a time (static-batch safe)."""
        if not faces_bgr:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        x = self.preprocess(faces_bgr)
        outs = [self.engine.run(x[i : i + 1])[0] for i in range(len(x))]
        return self.postprocess(np.concatenate(outs, axis=0))
