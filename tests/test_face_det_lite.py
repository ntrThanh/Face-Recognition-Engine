from pathlib import Path

import cv2
import pytest

from face_rec.config import resolve_path
from face_rec.detection import build_detector
from face_rec.engines.onnxruntime_engine import _metadata_quant

BASE = resolve_path("models/detection/face_det_lite")
IMAGES = sorted(resolve_path("assets/ntrthanh_face").glob("*.jpg"))
pytestmark = pytest.mark.skipif(not (BASE / "float/face_det_lite.onnx").exists() or not IMAGES,
                                reason="face_det_lite model or sample images missing")


def test_metadata_quant_params():
    q = _metadata_quant(str(BASE / "w8a8/face_det_lite.onnx"))
    assert set(q) == {"input", "heatmap", "bbox", "landmark"}
    assert q["input"][1] == 0 and 0.0039 < q["input"][0] < 0.004
    assert _metadata_quant(str(BASE / "float/face_det_lite.onnx")) == {}


@pytest.mark.parametrize("variant", ["float", "w8a8"])
def test_detects_one_face_with_landmarks(variant):
    det = build_detector({"type": "face_det_lite", "score_threshold": 0.55, "nms_threshold": 0.4,
                          "engines": [{"name": "onnxruntime",
                                       "path": str(BASE / variant / "face_det_lite.onnx")}]})
    assert det.input_size == (640, 480)
    for path in IMAGES:
        img = cv2.imread(str(path))
        faces = det.detect(img)
        assert len(faces) == 1, path.name
        lm = faces[0].landmarks
        x1, y1, x2, y2 = faces[0].bbox
        assert lm.shape == (5, 2)
        assert lm[0, 0] < lm[1, 0]                       # left eye left of right eye
        assert lm[[0, 1], 1].mean() < lm[[3, 4], 1].mean()  # eyes above mouth
        assert (x1 <= lm[:, 0]).all() and (lm[:, 0] <= x2).all()
