"""Face embedding database interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from face_rec.types import Match


class ModelMismatchError(RuntimeError):
    """The DB was built with a different recognizer; re-enroll with --rebuild."""


class FaceDatabase(ABC):
    registry_name: str = ""

    def __init__(self, cfg: dict[str, Any], model_id: str, embedding_dim: int):
        self.cfg = cfg
        self.model_id = model_id
        self.embedding_dim = embedding_dim
        self.threshold = float(cfg.get("threshold", 0.4))
        self.top_k = int(cfg.get("top_k", 1))

    def check_model(self, stored_model_id: str, stored_dim: int | None = None) -> None:
        if stored_model_id != self.model_id or (
            stored_dim is not None and stored_dim != self.embedding_dim
        ):
            raise ModelMismatchError(
                f"Database was built with '{stored_model_id}' (dim={stored_dim}) but the "
                f"recognizer is '{self.model_id}' (dim={self.embedding_dim}). "
                f"Run scripts/enroll.py --rebuild."
            )

    def _validate(self, embeddings: np.ndarray) -> np.ndarray:
        embeddings = np.atleast_2d(np.asarray(embeddings, dtype=np.float32))
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected embeddings of dim {self.embedding_dim}, got {embeddings.shape[1]}"
            )
        return embeddings

    @abstractmethod
    def load(self, reset: bool = False) -> None:
        """Open storage and call `check_model`.

        reset=True wipes all stored faces and adopts the current model_id
        (used by `enroll.py --rebuild` after changing the recognizer).
        """

    @abstractmethod
    def add(self, name: str, embeddings: np.ndarray, sources: list[str] | None = None) -> None:
        """Append (N, D) L2-normalized embeddings for `name`; `sources` are optional image paths."""

    @abstractmethod
    def remove(self, name: str) -> None: ...

    @abstractmethod
    def search(self, embedding: np.ndarray, top_k: int | None = None) -> list[Match]:
        """Best cosine similarity per person, highest first."""

    @abstractmethod
    def names(self) -> list[str]: ...

    @abstractmethod
    def count(self, name: str | None = None) -> int:
        """Number of stored embeddings (for one person, or in total)."""

    def save(self) -> None:
        """Flush pending writes. No-op for backends that write through."""

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
