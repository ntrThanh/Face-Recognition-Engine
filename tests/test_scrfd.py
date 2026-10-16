from pathlib import Path

import cv2
import numpy as np
import pytest

from face_rec.config import load_config, resolve_path
from face_rec.detection import build_detector
from face_rec.detection.scrfd import nms

CFG = load_config(resolve_path("configs/default.yaml"))
CPU_ENGINES = [c for c in CFG["detector"]["engines"] if c["name"] == "onnxruntime"]
IMAGES = sorted(resolve_path("assets/ntrthanh_face").glob("*.jpg"))
MODEL = Path(resolve_path(CPU_ENGINES[0]["path"])) if CPU_ENGINES else None


def test_nms_suppresses_overlaps():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    keep = nms(boxes, np.array([0.9, 0.8, 0.7]), 0.4)
    assert keep.tolist() == [0, 2]


@pytest.mark.skipif(MODEL is None or not MODEL.exists() or not IMAGES,
                    reason="detector model or sample images missing")
def test_detects_one_face_per_sample():
    det = build_detector({**CFG["detector"], "engines": CPU_ENGINES})
    for path in IMAGES:
        img = cv2.imread(str(path))
        faces = det.detect(img)
        assert len(faces) == 1, path.name
        f = faces[0]
        x1, y1, x2, y2 = f.bbox
        assert 0 <= x1 < x2 <= img.shape[1] and 0 <= y1 < y2 <= img.shape[0]
        assert f.landmarks.shape == (5, 2)
        # eyes above mouth corners
        assert f.landmarks[[0, 1], 1].mean() < f.landmarks[[3, 4], 1].mean()
