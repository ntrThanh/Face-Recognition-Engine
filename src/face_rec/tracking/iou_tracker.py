"""Greedy IoU tracker: cheap and good enough for faces at video frame rates."""

from __future__ import annotations

from collections import deque

import numpy as np

from face_rec.tracking import TRACKERS
from face_rec.tracking.base import Track, Tracker
from face_rec.types import Face


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between (N, 4) and (M, 4) xyxy boxes."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


@TRACKERS.register("iou")
class IoUTracker(Tracker):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.iou_threshold = float(cfg.get("iou_threshold", 0.3))
        self.max_missed = int(cfg.get("max_missed", 15))
        self.tracks: list[Track] = []
        self._next_id = 1

    def reset(self) -> None:
        # Track ids keep increasing so ids in event logs stay unique across resets.
        self.tracks = []

    def update(self, faces: list[Face], frame_index: int) -> list[Track]:
        boxes_t = np.array([t.face.bbox for t in self.tracks]).reshape(-1, 4)
        boxes_d = np.array([f.bbox for f in faces]).reshape(-1, 4)
        ious = iou_matrix(boxes_t, boxes_d)

        matched_t, matched_d = set(), set()
        for flat in np.argsort(-ious, axis=None):
            ti, di = np.unravel_index(flat, ious.shape)
            if ious[ti, di] < self.iou_threshold:
                break
            if ti in matched_t or di in matched_d:
                continue
            matched_t.add(ti)
            matched_d.add(di)
            t = self.tracks[ti]
            t.face, t.last_frame, t.missed = faces[di], frame_index, 0
            t.hits += 1

        for ti, t in enumerate(self.tracks):
            if ti not in matched_t:
                t.missed += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        for di, f in enumerate(faces):
            if di not in matched_d:
                self.tracks.append(Track(self._next_id, f, frame_index, frame_index,
                                         votes=deque(maxlen=self.vote_window)))
                self._next_id += 1

        return [t for t in self.tracks if t.last_frame == frame_index]
