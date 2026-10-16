"""Cheap face quality measures from 5-point landmarks and the aligned crop (no extra model)."""

from __future__ import annotations

import cv2
import numpy as np


def yaw_ratio(landmarks: np.ndarray) -> float:
    """Horizontal nose offset from the eye midpoint, in units of eye distance.

    ~0 for a frontal face; |value| grows as the head turns (about 0.5 at ~45 degrees).
    """
    left_eye, right_eye, nose = landmarks[0], landmarks[1], landmarks[2]
    eye_dist = max(float(np.linalg.norm(right_eye - left_eye)), 1.0)
    return float((nose[0] - (left_eye[0] + right_eye[0]) / 2) / eye_dist)


def pitch_ratio(landmarks: np.ndarray) -> float:
    """Nose height between the eye line (0) and the mouth line (1); ~0.4-0.6 when level."""
    eye_y = (landmarks[0, 1] + landmarks[1, 1]) / 2
    mouth_y = (landmarks[3, 1] + landmarks[4, 1]) / 2
    return float((landmarks[2, 1] - eye_y) / max(mouth_y - eye_y, 1.0))


def sharpness(aligned_bgr: np.ndarray) -> float:
    """Variance of the Laplacian on the aligned crop; low = blurry."""
    gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
