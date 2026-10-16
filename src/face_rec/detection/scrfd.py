"""SCRFD detector (InsightFace det_500m / det_2.5g / det_10g).

Model outputs (fmc = 3 strides, 2 anchors per location):
    [score_8, score_16, score_32, bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32]
with shapes (H/s * W/s * 2, 1 | 4 | 10). bbox/kps are distances in units of stride.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from face_rec.detection import DETECTORS
from face_rec.detection.base import Detector
from face_rec.types import Face

log = logging.getLogger(__name__)


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    """Greedy NMS; returns kept indices sorted by score."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= threshold]
    return np.array(keep, dtype=np.int64)


@DETECTORS.register("scrfd")
class SCRFDDetector(Detector):
    strides = (8, 16, 32)
    num_anchors = 2

    def __init__(self, cfg, engine):
        super().__init__(cfg, engine)
        self.mean = float(cfg.get("mean", 127.5))
        self.std = float(cfg.get("std", 128.0))
        self.color = cfg.get("color", "RGB")
        self.max_faces = int(cfg.get("max_faces", 0))  # 0 = unlimited

        # A static engine input shape wins over the config (they must agree for HTP).
        spec = self.engine.inputs()[0]
        dims = spec.shape[2:4] if spec.layout == "NCHW" else spec.shape[1:3]
        if all(isinstance(d, int) and d > 0 for d in dims):
            engine_size = (dims[1], dims[0])  # (w, h)
            if engine_size != self.input_size:
                log.warning("detector input_size %s overridden by engine shape %s",
                            self.input_size, engine_size)
            self.input_size = engine_size

        w, h = self.input_size
        self._anchors = {s: self._make_anchors(h // s, w // s, s) for s in self.strides}

    def _make_anchors(self, h: int, w: int, stride: int) -> np.ndarray:
        centers = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
        centers = (centers * stride).reshape(-1, 2)
        return np.repeat(centers, self.num_anchors, axis=0)

    def preprocess(self, img_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """Letterbox to input_size (top-left aligned) -> (1, 3, H, W) float32, scale."""
        in_w, in_h = self.input_size
        h, w = img_bgr.shape[:2]
        scale = min(in_w / w, in_h / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        canvas = np.zeros((in_h, in_w, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = cv2.resize(img_bgr, (new_w, new_h))
        # One native call for color swap + normalize + HWC->NCHW (~3x faster than numpy).
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / self.std, (in_w, in_h),
                                     (self.mean, self.mean, self.mean),
                                     swapRB=self.color == "RGB")
        return blob, scale

    def detect(self, img_bgr: np.ndarray) -> list[Face]:
        blob, scale = self.preprocess(img_bgr)
        outs = self.engine.run(blob)
        n = len(self.strides)

        all_boxes, all_scores, all_kps = [], [], []
        for i, stride in enumerate(self.strides):
            scores = outs[i].reshape(-1)
            keep = scores >= self.score_threshold
            if not keep.any():
                continue
            centers = self._anchors[stride][keep]
            dist = outs[i + n].reshape(-1, 4)[keep] * stride
            kps = outs[i + 2 * n].reshape(-1, 5, 2)[keep] * stride
            all_boxes.append(np.hstack([centers - dist[:, :2], centers + dist[:, 2:]]))
            all_kps.append(centers[:, None, :] + kps)
            all_scores.append(scores[keep])

        if not all_scores:
            return []
        boxes = np.concatenate(all_boxes) / scale
        kps = np.concatenate(all_kps) / scale
        scores = np.concatenate(all_scores)

        keep = nms(boxes, scores, self.nms_threshold)
        if self.max_faces:
            keep = keep[: self.max_faces]

        h, w = img_bgr.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)
        return [Face(bbox=boxes[i], score=float(scores[i]), landmarks=kps[i]) for i in keep]
