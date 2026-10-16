"""Visualization helpers. Note: cv2.putText cannot render Vietnamese diacritics."""

from __future__ import annotations

import cv2
import numpy as np

from face_rec.types import Face, RecognitionResult

KNOWN = (0, 200, 0)
UNKNOWN = (0, 0, 255)


def _draw_face(img: np.ndarray, face: Face, label: str, color) -> None:
    x1, y1, x2, y2 = face.bbox.astype(int)
    thickness = max(1, round(min(img.shape[:2]) / 400))
    scale = max(0.5, min(img.shape[:2]) / 1200)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    y_text = max(y1, th + base + 2)
    cv2.rectangle(img, (x1, y_text - th - base - 2), (x1 + tw + 4, y_text), color, -1)
    cv2.putText(img, label, (x1 + 2, y_text - base), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    if face.landmarks is not None:
        for x, y in face.landmarks.astype(int):
            cv2.circle(img, (x, y), thickness + 1, (255, 255, 0), -1)


def draw_results(img_bgr: np.ndarray, results: list[RecognitionResult]) -> np.ndarray:
    out = img_bgr.copy()
    for r in results:
        _draw_face(out, r.face, f"{r.name} {r.similarity:.2f}",
                   KNOWN if r.name != "unknown" else UNKNOWN)
    return out


def draw_tracks(img_bgr: np.ndarray, tracks, info: str | None = None) -> np.ndarray:
    """Draw tracked faces (face_rec.tracking.Track) plus an optional status line."""
    out = img_bgr.copy()
    for t in tracks:
        _draw_face(out, t.face, f"#{t.track_id} {t.name} {t.similarity:.2f}",
                   KNOWN if t.name != "unknown" else UNKNOWN)
    if info:
        cv2.putText(out, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return out
