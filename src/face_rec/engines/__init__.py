"""Engine registry and builder with fallback.

Engine modules import their runtime libraries lazily (inside `load`/`is_available`),
so registering every engine here is safe even if the library is not installed.
"""

from __future__ import annotations

import logging
from typing import Any

from face_rec.config import resolve_path
from face_rec.engines.base import InferenceEngine, TensorSpec
from face_rec.registry import Registry

ENGINES: Registry[InferenceEngine] = Registry("engine")

log = logging.getLogger(__name__)


def build_engine(candidates: list[dict[str, Any]]) -> InferenceEngine:
    """Try each `{name, path, options}` in order; return the first that loads."""
    errors = []
    for c in candidates:
        name = c["name"]
        try:
            cls = ENGINES.get(name)
            if not cls.is_available():
                errors.append(f"{name}: not available on this machine")
                continue
            engine = cls()
            engine.load(str(resolve_path(c["path"])), c.get("options") or {})
            log.info("Using engine '%s' for %s", name, c["path"])
            return engine
        except Exception as e:  # fall through to next candidate
            errors.append(f"{name}: {e}")
            log.warning("Engine '%s' failed, trying next: %s", name, e)
    raise RuntimeError("No engine could be loaded:\n  " + "\n  ".join(errors))


# Register built-in engines.
from face_rec.engines import onnxruntime_engine, qnn_engine, tflite_engine  # noqa: E402,F401

__all__ = ["ENGINES", "InferenceEngine", "TensorSpec", "build_engine"]
