"""ONNX Runtime engines.

- `onnxruntime_qnn`: QNN execution provider on the Hexagon HTP NPU (onnxruntime-qnn plugin).
- `onnxruntime`:     plain CPU, used as fallback and as the accuracy reference.

HTP needs static input shapes; models with dynamic dims (e.g. det_500m's H/W,
w600k_mbf's batch) are fixed in memory via the `input_shapes` option.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from face_rec.engines import ENGINES
from face_rec.engines.base import InferenceEngine, TensorSpec

_ORT_TYPES = {
    "tensor(float)": "float32",
    "tensor(float16)": "float16",
    "tensor(uint8)": "uint8",
    "tensor(int8)": "int8",
}


def _load_model(model_path: str, input_shapes: dict[str, list[int]] | None) -> str | bytes:
    """Return a path, or serialized bytes with the given inputs fixed to static shapes."""
    if not input_shapes:
        return model_path
    import onnx
    from onnxruntime.tools.onnx_model_utils import make_input_shape_fixed

    model = onnx.load(model_path)
    for name, shape in input_shapes.items():
        make_input_shape_fixed(model.graph, name, list(shape))
    return model.SerializeToString()


def _metadata_quant(model_path: str) -> dict[str, tuple[float, int]]:
    """Quantization params of uint8/int8 I/O from a Qualcomm AI Hub `metadata.json`
    next to the model, so quantized models still take/return float values."""
    meta_path = Path(model_path).with_name("metadata.json")
    if not meta_path.exists():
        return {}
    meta = json.loads(meta_path.read_text())
    files = meta.get("model_files", {})
    info = files.get(Path(model_path).name) or next(iter(files.values()), {})
    quant = {}
    for section in ("inputs", "outputs"):
        for name, spec in info.get(section, {}).items():
            q = spec.get("quantization_parameters")
            if q:
                quant[name] = (float(q["scale"]), int(q["zero_point"]))
    return quant


@ENGINES.register("onnxruntime")
class OnnxRuntimeEngine(InferenceEngine):
    """ONNX Runtime on CPU.

    Options:
        threads:      intra-op thread count
        spinning:     let idle ORT threads busy-wait for the next run (lower latency, burns CPU)
        input_shapes: {input_name: [N, C, H, W]} to fix dynamic dims
    """

    default_threads: int | None = None  # None = ORT default (one per core)
    default_spinning = True

    @classmethod
    def is_available(cls) -> bool:
        try:
            import onnxruntime  # noqa: F401
            return True
        except ImportError:
            return False

    def load(self, model_path: str, options: dict[str, Any] | None = None) -> None:
        import onnxruntime as ort

        options = options or {}
        so = ort.SessionOptions()
        threads = options.get("threads", self.default_threads)
        if threads:
            so.intra_op_num_threads = int(threads)
        spin = "1" if options.get("spinning", self.default_spinning) else "0"
        so.add_session_config_entry("session.intra_op.allow_spinning", spin)
        so.add_session_config_entry("session.inter_op.allow_spinning", spin)
        providers = self._configure(so, options)
        model = _load_model(model_path, options.get("input_shapes"))
        self.session = ort.InferenceSession(model, sess_options=so, providers=providers)
        self.model_path = model_path
        quant = _metadata_quant(model_path)
        self._inputs = [self._spec(i, quant) for i in self.session.get_inputs()]
        self._outputs = [self._spec(o, quant) for o in self.session.get_outputs()]

    def _configure(self, so, options: dict[str, Any]) -> list[str] | None:
        """Set up execution providers; return a `providers` list, or None if set on `so`."""
        return ["CPUExecutionProvider"]

    def providers(self) -> list[str]:
        return self.session.get_providers()

    @staticmethod
    def _spec(node, quant: dict[str, tuple[float, int]]) -> TensorSpec:
        dtype = _ORT_TYPES.get(node.type, "float32")
        q = quant.get(node.name) if dtype in ("uint8", "int8") else None
        return TensorSpec(node.name, tuple(node.shape), dtype, quant=q)

    def inputs(self) -> list[TensorSpec]:
        return self._inputs

    def outputs(self) -> list[TensorSpec]:
        return self._outputs

    def _run_raw(self, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        return self.session.run(None, feeds)


_qnn_devices = None


def _get_qnn_devices():
    """Register the onnxruntime-qnn plugin once per process and return its EP devices.

    On Linux, ORT's hardware discovery does not report the Hexagon NPU, so the plugin
    only exposes a device when ORT_QNN_ENABLE_CPU_BACKEND=1 is set before registration.
    The HTP backend is then selected explicitly with `backend_path`.
    """
    global _qnn_devices
    if _qnn_devices is None:
        import onnxruntime as ort
        import onnxruntime_qnn as qnn

        os.environ.setdefault("ORT_QNN_ENABLE_CPU_BACKEND", "1")
        ort.register_execution_provider_library(qnn.get_ep_name(), qnn.get_library_path())
        _qnn_devices = [d for d in ort.get_ep_devices() if d.ep_name == qnn.get_ep_name()]
    if not _qnn_devices:
        raise RuntimeError("onnxruntime-qnn registered but exposed no devices")
    return _qnn_devices


@ENGINES.register("onnxruntime_qnn")
class OnnxRuntimeQnnEngine(OnnxRuntimeEngine):
    """ONNX Runtime with the QNN execution provider (HTP NPU by default).

    Options (in addition to OnnxRuntimeEngine's):
        backend:            htp (default) | gpu | cpu
        htp_arch:           Hexagon arch, 73 for QCS8550 (default)
        performance_mode:   burst (default) | balanced | default | power_saver ...
        allow_cpu_fallback: False (default) -> fail to load if any node is unsupported
                            on QNN, so the engine fallback list moves to the next engine
                            instead of silently running parts on CPU.

    The graph runs on the NPU, so the CPU thread pool only orchestrates: one thread and
    no spinning by default, leaving the CPU to other work between calls.
    """

    default_threads = 1
    default_spinning = False

    @classmethod
    def is_available(cls) -> bool:
        try:
            import onnxruntime_qnn  # noqa: F401
            return True
        except ImportError:
            return False

    def _configure(self, so, options: dict[str, Any]) -> list[str] | None:
        import onnxruntime_qnn as qnn

        backend = options.get("backend", "htp")
        backend_paths = {
            "htp": qnn.get_qnn_htp_path,
            "gpu": qnn.get_qnn_gpu_path,
            "cpu": qnn.get_qnn_cpu_path,
        }
        ep_options = {"backend_path": backend_paths[backend]()}
        if backend == "htp":
            ep_options["htp_arch"] = str(options.get("htp_arch", 73))
            ep_options["htp_performance_mode"] = options.get("performance_mode", "burst")
        if not options.get("allow_cpu_fallback", False):
            so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        so.add_provider_for_devices(_get_qnn_devices(), ep_options)
        return None
