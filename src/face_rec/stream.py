"""Per-frame recognition for video/streams with an inference budget.

To leave the NPU free for other models, inference is rate-limited:
  - detection runs at most `active_fps` times per second while faces are present,
    and drops to `idle_fps` after `idle_after_s` seconds without any face;
  - frames in between run no inference at all and reuse the last tracks;
  - each track is embedded when it appears, then every `reembed_interval_s`
    (`unknown_reembed_interval_s` while unknown); its name is a majority vote.
Timestamps are video time for files and wall-clock time for live streams.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from face_rec.pipeline import FaceRecognizer
from face_rec.tracking import Track, build_tracker


@dataclass
class IdentityEvent:
    frame_index: int
    timestamp: float
    track_id: int
    name: str
    similarity: float
    bbox: list[int]


@dataclass
class FrameResult:
    frame_index: int
    timestamp: float
    tracks: list[Track]
    inferred: bool                 # False: no inference ran, tracks are from an earlier frame
    events: list[IdentityEvent] = field(default_factory=list)
    # ms for inferred frames: total, detect (incl. pre/post), detect_infer (runtime only),
    # embed, embed_infer, plus counts of faces and of faces embedded this frame.
    timing: dict[str, float] = field(default_factory=dict)


class StreamRecognizer:
    def __init__(self, recognizer: FaceRecognizer, cfg: dict[str, Any] | None = None):
        cfg = cfg or {}
        self.fr = recognizer
        self.tracker = build_tracker(cfg)
        self.active_fps = float(cfg.get("active_fps", 5.0))   # 0 = every frame
        self.idle_fps = float(cfg.get("idle_fps", 1.0))
        self.idle_after_s = float(cfg.get("idle_after_s", 2.0))
        self.reembed_interval_s = float(cfg.get("reembed_interval_s", 3.0))
        self.unknown_reembed_interval_s = float(cfg.get("unknown_reembed_interval_s", 0.6))
        self.min_hits = int(cfg.get("min_hits", 2))  # detections before an event is emitted

        self.detect_calls = 0  # NPU calls, for monitoring the inference budget
        self.embed_calls = 0
        self._index = 0
        self._steps = 0        # frames on which inference ran (tracker clock)
        self._last_run: float | None = None
        self._last_face_seen: float | None = None
        self._tracks: list[Track] = []

    # ---- scheduling ----
    @property
    def idle(self) -> bool:
        return self._last_face_seen is None or (
            self._last_run is not None and self._last_run - self._last_face_seen >= self.idle_after_s
        )

    def _should_run(self, ts: float) -> bool:
        if self._last_run is None or ts < self._last_run:  # first frame, or clock went back
            return True
        fps = self.idle_fps if self.idle else self.active_fps
        return fps <= 0 or ts - self._last_run >= 1.0 / fps - 1e-3

    def _embed_due(self, t: Track, ts: float) -> bool:
        if t.last_embed_time is None:
            return True
        interval = (self.unknown_reembed_interval_s if t.name == "unknown"
                    else self.reembed_interval_s)
        return ts - t.last_embed_time >= interval

    # ---- main entry ----
    def process(self, frame_bgr: np.ndarray, timestamp: float) -> FrameResult:
        idx = self._index
        self._index += 1
        if not self._should_run(timestamp):
            return FrameResult(idx, timestamp, self._tracks, inferred=False)

        self._last_run = timestamp
        t0 = time.perf_counter()
        det_eng, emb_eng = self.fr.detector.engine.timer, self.fr.recognizer.engine.timer
        det_before, emb_before = det_eng.totals.get("infer", 0.0), emb_eng.totals.get("infer", 0.0)

        faces = self.fr.detect(frame_bgr)
        t_det = time.perf_counter()
        self.detect_calls += 1
        if faces:
            self._last_face_seen = timestamp
        with self.fr.timer("track"):
            tracks = self.tracker.update(faces, self._steps)
        self._steps += 1

        due = [t for t in tracks if self._embed_due(t, timestamp)]
        t_emb = time.perf_counter()
        self.fr.embed_faces(frame_bgr, [t.face for t in due])
        t_emb_end = time.perf_counter()
        self.embed_calls += len(due)
        for t in due:
            r = self.fr.match(t.face)
            t.add_vote(r.name, r.similarity)
            t.last_embed_time = timestamp

        events = []
        for t in tracks:
            if t.hits >= self.min_hits and t.name != t.reported_name and t.votes:
                t.reported_name = t.name
                events.append(IdentityEvent(idx, timestamp, t.track_id, t.name,
                                            round(t.similarity, 4),
                                            t.face.bbox.astype(int).tolist()))
        self._tracks = tracks
        total = time.perf_counter() - t0
        self.fr.timer.add("frame", total)
        timing = {
            "total": 1000 * total,
            "detect": 1000 * (t_det - t0),
            "detect_infer": 1000 * (det_eng.totals.get("infer", 0.0) - det_before),
            "embed": 1000 * (t_emb_end - t_emb),
            "embed_infer": 1000 * (emb_eng.totals.get("infer", 0.0) - emb_before),
            "faces": len(faces),
            "embedded": len(due),
        }
        return FrameResult(idx, timestamp, tracks, inferred=True, events=events, timing=timing)

    def reset(self) -> None:
        """Call after a stream reconnect: old tracks are no longer valid."""
        self.tracker.reset()
        self._tracks = []
        self._last_run = None
        self._last_face_seen = None
