import numpy as np

from face_rec.tracking import build_tracker
from face_rec.tracking.base import Track
from face_rec.types import Face


def face(x, y, s=50):
    return Face(bbox=np.array([x, y, x + s, y + s], dtype=np.float32), score=0.9)


def test_ids_persist_and_expire():
    tr = build_tracker({"type": "iou", "iou_threshold": 0.3, "max_missed": 2})
    a = tr.update([face(0, 0), face(200, 0)], 0)
    ids = sorted(t.track_id for t in a)
    b = tr.update([face(5, 3), face(204, 2)], 1)  # small motion -> same ids
    assert sorted(t.track_id for t in b) == ids
    for i in range(2, 5):  # disappears longer than max_missed
        assert tr.update([], i) == []
    c = tr.update([face(0, 0)], 5)
    assert c[0].track_id not in ids


def test_reset_keeps_ids_unique():
    tr = build_tracker({"type": "iou"})
    first = tr.update([face(0, 0)], 0)[0].track_id
    tr.reset()
    assert tr.update([face(0, 0)], 1)[0].track_id != first


def test_vote_majority_prefers_known_on_tie():
    t = Track(1, face(0, 0), 0, 0)
    t.add_vote("unknown", 0.2)
    t.add_vote("alice", 0.6)
    assert t.name == "alice" and t.similarity == 0.6
    t.add_vote("alice", 0.8)
    assert abs(t.similarity - 0.7) < 1e-9
