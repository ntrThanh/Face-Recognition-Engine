import numpy as np

from face_rec.engines.base import InferenceEngine, TensorSpec


class EchoNHWCInt8(InferenceEngine):
    """Fake NPU-style engine: NHWC int8 in/out, returns its input."""

    spec = TensorSpec("x", (1, 4, 4, 3), "int8", "NHWC", (0.05, 0))

    def load(self, model_path, options=None):
        pass

    def inputs(self):
        return [self.spec]

    def outputs(self):
        return [self.spec]

    def _run_raw(self, feeds):
        assert feeds["x"].dtype == np.int8 and feeds["x"].shape == (1, 4, 4, 3)
        return [feeds["x"]]


def test_layout_and_quant_roundtrip():
    x = np.random.uniform(-1, 1, (1, 3, 4, 4)).astype(np.float32)
    (y,) = EchoNHWCInt8().run(x)
    assert y.shape == x.shape and y.dtype == np.float32
    assert np.abs(y - x).max() <= 0.05 / 2 + 1e-6
