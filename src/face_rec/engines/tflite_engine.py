"""TFLite engine with optional QNN / GPU delegate (Qualcomm NPU). Phase 3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from face_rec.engines import ENGINES
from face_rec.engines.base import InferenceEngine, TensorSpec

QNN_DELEGATE = "/usr/lib/libQnnTFLiteDelegate.so"


@ENGINES.register("tflite_qnn")
class TFLiteQnnEngine(InferenceEngine):
    @classmethod
    def is_available(cls) -> bool:
        try:
            import tflite_runtime  # noqa: F401
        except ImportError:
            return False
        return Path(QNN_DELEGATE).exists()

    def load(self, model_path: str, options: dict[str, Any] | None = None) -> None:
        # TODO(phase 3): load_delegate(QNN_DELEGATE, {"backend_type": options["backend"]}),
        # build Interpreter, read input/output details (NHWC + quant params) into TensorSpec.
        raise NotImplementedError("TFLite + QNN engine is implemented in phase 3")

    def inputs(self) -> list[TensorSpec]:
        raise NotImplementedError

    def outputs(self) -> list[TensorSpec]:
        raise NotImplementedError

    def _run_raw(self, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        raise NotImplementedError
