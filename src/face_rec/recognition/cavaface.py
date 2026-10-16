"""CavaFace (IR-SE-100, MIT license) from Qualcomm AI Hub. Optional, phase 4."""

from face_rec.recognition import RECOGNIZERS
from face_rec.recognition.base import Recognizer


@RECOGNIZERS.register("cavaface")
class CavaFaceRecognizer(Recognizer):
    # TODO(phase 4): verify input normalization / color order of the AI Hub export.
    pass
