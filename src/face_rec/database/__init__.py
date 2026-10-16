from __future__ import annotations

from typing import Any

from face_rec.database.base import FaceDatabase, ModelMismatchError
from face_rec.registry import Registry

DATABASES: Registry[FaceDatabase] = Registry("database")


def build_database(
    cfg: dict[str, Any], model_id: str, embedding_dim: int, reset: bool = False
) -> FaceDatabase:
    db = DATABASES.get(cfg["type"])(cfg, model_id, embedding_dim)
    db.load(reset=reset)
    return db


from face_rec.database import sqlite_db  # noqa: E402,F401

__all__ = ["DATABASES", "FaceDatabase", "ModelMismatchError", "build_database"]
