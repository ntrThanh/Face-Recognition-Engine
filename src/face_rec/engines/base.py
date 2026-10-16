"""Inference engine interface.

Models always talk to engines in float32 NCHW. Layout conversion (NCHW <-> NHWC)
and INT8 quantize/dequantize are handled here, so a concrete engine only
implements `load`, `inputs`, `outputs` and `_run_raw`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from face_rec.utils.timer import StageTimer


@dataclass
class TensorSpec:
    name: str
    shape: tuple
    dtype: str                                  # "float32", "uint8", "int8", ...
    layout: str = "NCHW"                        # "NCHW" or "NHWC" (4D tensors only)
    quant: tuple[float, int] | None = None      # (scale, zero_point) for quantized tensors


class InferenceEngine(ABC):
    registry_name: str = ""

    def __init__(self) -> None:
        self.model_path: str | None = None
        # "infer": time inside the runtime only (NPU/CPU compute + transfers),
        # excluding layout/quantization conversion and the model's pre/post-processing.
        self.timer = StageTimer()

    # ---- to implement per engine ----
    @classmethod
    def is_available(cls) -> bool:
        """True if the runtime library / hardware is present. Must not raise."""
        return True

    @abstractmethod
    def load(self, model_path: str, options: dict[str, Any] | None = None) -> None: ...

    @abstractmethod
    def inputs(self) -> list[TensorSpec]: ...

    @abstractmethod
    def outputs(self) -> list[TensorSpec]: ...

    @abstractmethod
    def _run_raw(self, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        """Run with tensors already in the engine's native layout/dtype."""

    # ---- shared behaviour ----
    def run(self, *inputs: np.ndarray) -> list[np.ndarray]:
        """Run with float32 NCHW inputs; returns float32 outputs (NCHW for 4D)."""
        specs = self.inputs()
        if len(inputs) != len(specs):
            raise ValueError(f"Expected {len(specs)} inputs, got {len(inputs)}")
        feeds = {s.name: self._to_native(x, s) for x, s in zip(inputs, specs)}
        with self.timer("infer"):
            raw = self._run_raw(feeds)
        return [self._from_native(y, s) for y, s in zip(raw, self.outputs())]

    def warmup(self, n: int = 3) -> None:
        dummies = []
        for s in self.inputs():
            shape = tuple(d if isinstance(d, int) and d > 0 else 1 for d in s.shape)
            if s.layout == "NHWC" and len(shape) == 4:
                shape = (shape[0], shape[3], shape[1], shape[2])
            dummies.append(np.zeros(shape, dtype=np.float32))
        for _ in range(n):
            self.run(*dummies)
        self.timer.reset()

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- helpers ----
    @staticmethod
    def _to_native(x: np.ndarray, spec: TensorSpec) -> np.ndarray:
        if spec.layout == "NHWC" and x.ndim == 4:
            x = x.transpose(0, 2, 3, 1)
        # Fast path: callers may pass an already-quantized tensor of the native dtype
        # (e.g. uint8 pixels mapped with the input's scale/zero_point).
        if spec.quant is not None and x.dtype != np.dtype(spec.dtype):
            scale, zp = spec.quant
            info = np.iinfo(spec.dtype)
            x = np.clip(np.round(x / scale) + zp, info.min, info.max)
        return np.ascontiguousarray(x.astype(spec.dtype, copy=False))

    @staticmethod
    def _from_native(y: np.ndarray, spec: TensorSpec) -> np.ndarray:
        if spec.quant is not None:
            scale, zp = spec.quant
            y = (y.astype(np.float32) - zp) * scale
        else:
            y = y.astype(np.float32, copy=False)
        if spec.layout == "NHWC" and y.ndim == 4:
            y = y.transpose(0, 3, 1, 2)
        return y
