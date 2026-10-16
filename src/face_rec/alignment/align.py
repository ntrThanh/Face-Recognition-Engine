"""Similarity-transform face alignment from 5 landmarks."""

from __future__ import annotations

import cv2
import numpy as np

from face_rec.alignment.templates import TEMPLATES


def estimate_transform(landmarks: np.ndarray, template: str = "arcface_112") -> np.ndarray:
    dst, _ = TEMPLATES[template]
    M, _ = cv2.estimateAffinePartial2D(
        landmarks.astype(np.float32), dst, method=cv2.LMEDS
    )
    if M is None:
        raise ValueError("Could not estimate alignment transform")
    return M


def align_face(img_bgr: np.ndarray, landmarks: np.ndarray, template: str = "arcface_112") -> np.ndarray:
    """Warp a face to the template's canonical crop (e.g. 112x112 BGR uint8)."""
    _, size = TEMPLATES[template]
    M = estimate_transform(landmarks, template)
    return cv2.warpAffine(img_bgr, M, size, borderValue=0.0)
