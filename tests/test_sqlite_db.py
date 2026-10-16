import numpy as np
import pytest

from face_rec.database import ModelMismatchError, build_database

DIM = 8


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def make_db(path, model_id="m1", dim=DIM, reset=False):
    cfg = {"type": "sqlite", "path": str(path), "threshold": 0.5, "top_k": 2}
    return build_database(cfg, model_id, dim, reset=reset)


def test_add_search_remove(tmp_path):
    a, b = unit(np.eye(DIM)[0]), unit(np.eye(DIM)[1])
    with make_db(tmp_path / "f.db") as db:
        db.add("alice", np.stack([a, unit(a + 0.1 * b)]), sources=["a1.jpg", "a2.jpg"])
        db.add("bob", b)
        assert db.names() == ["alice", "bob"] and db.count() == 3 and db.count("alice") == 2

        matches = db.search(a)
        assert [m.name for m in matches] == ["alice", "bob"]
        assert matches[0].similarity == pytest.approx(1.0, abs=1e-6)

        db.remove("alice")
        assert db.names() == ["bob"] and db.count() == 1


def test_persists_across_reopen(tmp_path):
    with make_db(tmp_path / "f.db") as db:
        db.add("alice", unit(np.ones(DIM)))
    with make_db(tmp_path / "f.db") as db:
        assert db.names() == ["alice"]
        assert db.search(unit(np.ones(DIM)))[0].similarity == pytest.approx(1.0, abs=1e-6)


def test_model_mismatch_and_reset(tmp_path):
    with make_db(tmp_path / "f.db") as db:
        db.add("alice", unit(np.ones(DIM)))
    with pytest.raises(ModelMismatchError):
        make_db(tmp_path / "f.db", model_id="m2")
    with make_db(tmp_path / "f.db", model_id="m2", reset=True) as db:
        assert db.count() == 0
    with make_db(tmp_path / "f.db", model_id="m2") as db:  # now adopted
        assert db.model_id == "m2"


def test_wrong_dim_rejected(tmp_path):
    with make_db(tmp_path / "f.db") as db:
        with pytest.raises(ValueError):
            db.add("x", np.ones(DIM + 1))
    with pytest.raises(ModelMismatchError):
        make_db(tmp_path / "f.db", dim=DIM + 1)


def test_empty_search(tmp_path):
    with make_db(tmp_path / "f.db") as db:
        assert db.search(unit(np.ones(DIM))) == []
