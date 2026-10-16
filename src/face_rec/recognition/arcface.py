"""ArcFace-family recognizers (InsightFace w600k_mbf, w600k_r50, ...)."""

from face_rec.recognition import RECOGNIZERS
from face_rec.recognition.base import Recognizer


@RECOGNIZERS.register("arcface")
class ArcFaceRecognizer(Recognizer):
    pass
