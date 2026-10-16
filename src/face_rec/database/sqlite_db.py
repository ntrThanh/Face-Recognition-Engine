"""SQLite face database.

Vectors are stored as float32 BLOBs. On load/after each write the whole gallery is
cached as an (N, D) matrix, so search is a single matmul. Fine up to ~100k embeddings;
beyond that, add an ANN backend (e.g. FAISS) behind the same interface.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from face_rec.database import DATABASES
from face_rec.database.base import FaceDatabase
from face_rec.types import Match

SCHEMA_VERSION = "1"

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS persons (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS embeddings (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id  INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    vector     BLOB NOT NULL,
    source     TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_embeddings_person ON embeddings(person_id);
"""


@DATABASES.register("sqlite")
class SQLiteFaceDatabase(FaceDatabase):
    def __init__(self, cfg, model_id, embedding_dim):
        super().__init__(cfg, model_id, embedding_dim)
        self.path = cfg["path"]
        self.conn: sqlite3.Connection | None = None
        self._names: list[str] = []
        self._matrix = np.empty((0, embedding_dim), dtype=np.float32)
        self._labels = np.empty(0, dtype=np.int64)  # row -> index into self._names

    # ---- lifecycle ----
    def load(self, reset: bool = False) -> None:
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(SCHEMA)

        meta = dict(self.conn.execute("SELECT key, value FROM meta"))
        if reset or "model_id" not in meta:
            with self.conn:
                self.conn.execute("DELETE FROM persons")  # cascades to embeddings
                self.conn.executemany(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                    [
                        ("model_id", self.model_id),
                        ("embedding_dim", str(self.embedding_dim)),
                        ("schema_version", SCHEMA_VERSION),
                    ],
                )
        else:
            self.check_model(meta["model_id"], int(meta["embedding_dim"]))
        self._refresh()

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    # ---- writes ----
    def add(self, name: str, embeddings: np.ndarray, sources: list[str] | None = None) -> None:
        embeddings = self._validate(embeddings)
        sources = sources if sources is not None else [None] * len(embeddings)
        if len(sources) != len(embeddings):
            raise ValueError("sources must match the number of embeddings")
        with self.conn:
            self.conn.execute("INSERT OR IGNORE INTO persons(name) VALUES (?)", (name,))
            (person_id,) = self.conn.execute(
                "SELECT id FROM persons WHERE name = ?", (name,)
            ).fetchone()
            self.conn.executemany(
                "INSERT INTO embeddings(person_id, vector, source) VALUES (?, ?, ?)",
                [(person_id, e.tobytes(), s) for e, s in zip(embeddings, sources)],
            )
        self._refresh()

    def remove(self, name: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM persons WHERE name = ?", (name,))
        self._refresh()

    # ---- reads ----
    def search(self, embedding: np.ndarray, top_k: int | None = None) -> list[Match]:
        if not len(self._matrix):
            return []
        top_k = top_k or self.top_k
        sims = self._matrix @ np.asarray(embedding, dtype=np.float32).ravel()
        best = np.full(len(self._names), -np.inf, dtype=np.float32)
        np.maximum.at(best, self._labels, sims)
        order = np.argsort(-best)[:top_k]
        return [Match(self._names[i], float(best[i])) for i in order]

    def names(self) -> list[str]:
        return list(self._names)

    def count(self, name: str | None = None) -> int:
        if name is None:
            return len(self._matrix)
        if name not in self._names:
            return 0
        return int((self._labels == self._names.index(name)).sum())

    # ---- cache ----
    def _refresh(self) -> None:
        rows = self.conn.execute(
            "SELECT p.name, e.vector FROM embeddings e JOIN persons p ON p.id = e.person_id"
        ).fetchall()
        self._names = sorted({name for name, _ in rows})
        index = {n: i for i, n in enumerate(self._names)}
        if rows:
            self._matrix = np.stack([np.frombuffer(v, dtype=np.float32) for _, v in rows])
            self._labels = np.array([index[n] for n, _ in rows], dtype=np.int64)
        else:
            self._matrix = np.empty((0, self.embedding_dim), dtype=np.float32)
            self._labels = np.empty(0, dtype=np.int64)
