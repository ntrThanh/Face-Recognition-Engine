from __future__ import annotations

from typing import Any

from face_rec.registry import Registry
from face_rec.tracking.base import Track, Tracker

TRACKERS: Registry[Tracker] = Registry("tracker")


def build_tracker(cfg: dict[str, Any]) -> Tracker:
    return TRACKERS.get(cfg.get("type", "iou"))(cfg)


from face_rec.tracking import iou_tracker  # noqa: E402,F401

__all__ = ["TRACKERS", "Track", "Tracker", "build_tracker"]
