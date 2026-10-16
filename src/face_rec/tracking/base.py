"""Tracker interface and the per-face Track state."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

from face_rec.types import Face


@dataclass
class Track:
    track_id: int
    face: Face
    first_frame: int
    last_frame: int
    hits: int = 1
    missed: int = 0
    last_embed_time: float | None = None
    votes: deque = field(default_factory=lambda: deque(maxlen=5))  # (name, similarity)
    name: str = "unknown"            # identity after voting
    similarity: float = 0.0          # mean similarity of the votes for `name`
    reported_name: str | None = None  # last identity emitted as an event

    def add_vote(self, name: str, similarity: float) -> None:
        self.votes.append((name, similarity))
        counts = Counter(n for n, _ in self.votes)
        best = max(counts, key=lambda n: (counts[n], n != "unknown"))
        self.name = best
        self.similarity = sum(s for n, s in self.votes if n == best) / counts[best]


class Tracker(ABC):
    registry_name: str = ""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.vote_window = int(cfg.get("vote_window", 5))

    @abstractmethod
    def update(self, faces: list[Face], frame_index: int) -> list[Track]:
        """Associate detections with tracks; return the tracks visible in this frame."""

    @abstractmethod
    def reset(self) -> None: ...
