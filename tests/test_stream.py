from types import SimpleNamespace

import numpy as np

from face_rec.stream import StreamRecognizer
from face_rec.types import Face, Match, RecognitionResult
from face_rec.utils.timer import StageTimer


class FakeRecognizer:
    """Stands in for FaceRecognizer: a face is present when `self.present` is True."""

    def __init__(self):
        self.present = True
        self.timer = StageTimer()
        self.detector = SimpleNamespace(engine=SimpleNamespace(timer=StageTimer()))
        self.recognizer = SimpleNamespace(engine=SimpleNamespace(timer=StageTimer()))

    def detect(self, img):
        return [Face(np.array([10, 10, 60, 60], np.float32), 0.9)] if self.present else []

    def embed_faces(self, img, faces):
        for f in faces:
            f.embedding = np.ones(4, np.float32)
        return faces

    def match(self, face):
        return RecognitionResult(face, [Match("alice", 0.8)], 0.4)


CFG = {"active_fps": 5, "idle_fps": 1, "idle_after_s": 2.0,
       "reembed_interval_s": 3.0, "unknown_reembed_interval_s": 0.6, "min_hits": 2}
FRAME = np.zeros((8, 8, 3), np.uint8)


def run(sr, seconds, fps=25, start=0.0):
    return [sr.process(FRAME, start + i / fps) for i in range(int(seconds * fps))]


def test_active_rate_limits_detection():
    fr = FakeRecognizer()
    sr = StreamRecognizer(fr, CFG)
    results = run(sr, 4.0)                      # 100 frames at 25 fps
    assert 19 <= sr.detect_calls <= 21          # ~5/s, not 25/s
    assert sr.embed_calls == 2                  # at t=0 and after reembed_interval_s
    skipped = [r for r in results if not r.inferred]
    assert skipped and all(r.tracks for r in skipped[5:])  # reuse last tracks
    assert sum(len(r.events) for r in results) == 1


def test_idle_rate_when_no_faces():
    fr = FakeRecognizer()
    fr.present = False
    sr = StreamRecognizer(fr, CFG)
    run(sr, 10.0)
    assert sr.idle and sr.detect_calls <= 12    # ~1/s once idle

    fr.present = True                           # someone walks in
    before = sr.detect_calls
    run(sr, 3.0, start=10.0)
    assert not sr.idle
    assert sr.detect_calls - before >= 12       # back to ~5/s within a second


def test_every_frame_when_active_fps_is_zero():
    sr = StreamRecognizer(FakeRecognizer(), {**CFG, "active_fps": 0})
    run(sr, 1.0)
    assert sr.detect_calls == 25


def test_inferred_frames_carry_timing():
    sr = StreamRecognizer(FakeRecognizer(), CFG)
    results = run(sr, 1.0)
    inferred = [r for r in results if r.inferred]
    assert inferred and all(r.timing["total"] >= r.timing["detect"] >= 0 for r in inferred)
    assert inferred[0].timing["embedded"] == 1 and inferred[1].timing["embedded"] == 0
    assert all(not r.timing for r in results if not r.inferred)
    assert sr.fr.timer.counts["frame"] == len(inferred)
