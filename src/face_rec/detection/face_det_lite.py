"""Qualcomm Lightweight-Face-Detection (face_det_lite, BSD-3-Clause, AI Hub v0.64.0).

Input:  1x1x480x640 grayscale in [0, 1] (the model normalizes internally; the w8a8
        export takes uint8, handled by the engine via metadata.json).
Output: CenterNet-style maps at stride 8 (60x80):
        heatmap  (1,1,H,W)  face-center logits
        bbox     (1,4,H,W)  distances left/top/right/bottom from the cell, in cells
        landmark (1,10,H,W) 5 x-offsets then 5 y-offsets from the cell, in cells
Decoding follows qai_hub_models/models/face_det_lite/utils.py: sigmoid, 3x3 max-pool
peaks, score threshold; NMS is optional (peaks rarely overlap).
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from face_rec.detection import DETECTORS
from face_rec.detection.base import Detector
from face_rec.detection.scrfd import nms
from face_rec.types import Face

log = logging.getLogger(__name__)

# skimage.color.rgb2gray weights (used at training time), in BGR order.
_GRAY_BGR = np.array([[0.0721, 0.7154, 0.2125]], dtype=np.float32)


@DETECTORS.register("face_det_lite")
class FaceDetLiteDetector(Detector):
    stride = 8

    def __init__(self, cfg, engine):
        super().__init__(cfg, engine)
        self.max_faces = int(cfg.get("max_faces", 0))
        spec = self.engine.inputs()[0]
        n, c, h, w = spec.shape
        self.input_size = (int(w), int(h))
        names = [o.name for o in self.engine.outputs()]
        self._order = [names.index(k) for k in ("heatmap", "bbox", "landmark")] \
            if {"heatmap", "bbox", "landmark"} <= set(names) else [0, 1, 2]

        # Quantized model: map uint8 pixels straight to the input's integer domain with
        # a 256-entry LUT, skipping the float round trip (the costly part on a slow CPU).
        self._lut = None
        if spec.quant is not None and spec.dtype in ("uint8", "int8"):
            q_scale, q_zero = spec.quant
            info = np.iinfo(spec.dtype)
            levels = np.round(np.arange(256) / 255.0 / q_scale) + q_zero
            self._lut = np.clip(levels, info.min, info.max).astype(spec.dtype)

    def preprocess(self, img_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """Letterbox (top-left) to the model size, grayscale -> (1, 1, H, W).

        float32 in [0, 1] for float models, or already-quantized integers for w8a8.
        """
        in_w, in_h = self.input_size
        h, w = img_bgr.shape[:2]
        scale = min(in_w / w, in_h / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        canvas = np.zeros((in_h, in_w), dtype=np.uint8)
        canvas[:new_h, :new_w] = cv2.transform(cv2.resize(img_bgr, (new_w, new_h)), _GRAY_BGR)
        if self._lut is not None:
            return self._lut[canvas][None, None], scale
        return (canvas.astype(np.float32) * (1.0 / 255.0))[None, None], scale

    def detect(self, img_bgr: np.ndarray) -> list[Face]:
        blob, scale = self.preprocess(img_bgr)
        outs = self.engine.run(blob)
        hm, box, lm = (outs[i][0] for i in self._order)  # (1,H,W) (4,H,W) (10,H,W)

        heat = 1.0 / (1.0 + np.exp(-hm[0]))
        peaks = (heat == cv2.dilate(heat, np.ones((3, 3), np.uint8))) & (heat >= self.score_threshold)
        ys, xs = np.nonzero(peaks)
        if not len(ys):
            return []
        scores = heat[ys, xs]
        d = box[:, ys, xs].T                                  # (n, 4): l, t, r, b
        boxes = np.stack([xs - d[:, 0], ys - d[:, 1], xs + d[:, 2], ys + d[:, 3]], axis=1)
        lx = lm[0:5, ys, xs].T + xs[:, None]
        ly = lm[5:10, ys, xs].T + ys[:, None]
        boxes = boxes * (self.stride / scale)
        kps = np.stack([lx, ly], axis=-1) * (self.stride / scale)  # (n, 5, 2)

        keep = nms(boxes, scores, self.nms_threshold) if self.nms_threshold > 0 \
            else np.argsort(-scores)
        if self.max_faces:
            keep = keep[: self.max_faces]
        h, w = img_bgr.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)
        return [Face(bbox=boxes[i].astype(np.float32), score=float(scores[i]),
                     landmarks=kps[i].astype(np.float32)) for i in keep]
