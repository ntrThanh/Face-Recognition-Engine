"""QNN context-binary engine (HTP NPU, lowest overhead). Phase 3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from face_rec.engines import ENGINES
from face_rec.engines.base import InferenceEngine, TensorSpec

QNN_HTP_LIB = "/usr/lib/libQnnHtp.so"


@ENGINES.register("qnn")
class QnnEngine(InferenceEngine):
    @classmethod
    def is_available(cls) -> bool:
        # TODO(phase 3): also check the Python binding (e.g. qai_appbuilder) is importable.
        return False and Path(QNN_HTP_LIB).exists()

    def load(self, model_path: str, options: dict[str, Any] | None = None) -> None:
        raise NotImplementedError("QNN engine is implemented in phase 3")

    def inputs(self) -> list[TensorSpec]:
        raise NotImplementedError

    def outputs(self) -> list[TensorSpec]:
        raise NotImplementedError

    def _run_raw(self, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        raise NotImplementedError
