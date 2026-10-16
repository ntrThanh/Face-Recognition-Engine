import pytest

from face_rec.database import DATABASES
from face_rec.detection import DETECTORS
from face_rec.engines import ENGINES
from face_rec.recognition import RECOGNIZERS
from face_rec.registry import Registry


def test_builtins_registered():
    assert {"onnxruntime", "onnxruntime_qnn", "tflite_qnn", "qnn"} <= set(ENGINES.names())
    assert "scrfd" in DETECTORS
    assert {"arcface", "cavaface"} <= set(RECOGNIZERS.names())
    assert "sqlite" in DATABASES


def test_duplicate_and_unknown():
    r = Registry("thing")

    @r.register("a")
    class A:
        pass

    with pytest.raises(KeyError):
        r.register("a")(A)
    with pytest.raises(KeyError, match="Available"):
        r.get("missing")
