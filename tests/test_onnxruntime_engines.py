import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from face_rec.engines import ENGINES, build_engine  # noqa: E402


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """Conv+Relu with dynamic batch/H/W, like the InsightFace models."""
    w = numpy_helper.from_array(np.random.RandomState(0).randn(8, 3, 3, 3).astype(np.float32), "W")
    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["x", "W"], ["c"], pads=[1, 1, 1, 1]),
            helper.make_node("Relu", ["c"], ["y"]),
        ],
        "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3, "H", "W"])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 8, "H", "W"])],
        [w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tmp_path_factory.mktemp("models") / "tiny.onnx"
    onnx.save(model, path)
    return str(path)


SHAPE = [1, 3, 32, 32]


def test_cpu_engine_fixes_shapes(tiny_model):
    e = build_engine([{"name": "onnxruntime", "path": tiny_model,
                       "options": {"input_shapes": {"x": SHAPE}}}])
    assert e.inputs()[0].shape == tuple(SHAPE)
    (y,) = e.run(np.zeros(SHAPE, np.float32))
    assert y.shape == (1, 8, 32, 32)


@pytest.mark.skipif(not ENGINES.get("onnxruntime_qnn").is_available(),
                    reason="onnxruntime-qnn not installed")
def test_qnn_htp_matches_cpu(tiny_model):
    opts = {"input_shapes": {"x": SHAPE}}
    cpu = build_engine([{"name": "onnxruntime", "path": tiny_model, "options": opts}])
    qnn = build_engine([{"name": "onnxruntime_qnn", "path": tiny_model, "options": opts}])
    assert qnn.providers()[0] == "QNNExecutionProvider"
    x = np.random.RandomState(1).uniform(-1, 1, SHAPE).astype(np.float32)
    (ref,), (out,) = cpu.run(x), qnn.run(x)
    np.testing.assert_allclose(out, ref, atol=0.05)  # HTP computes float models in FP16
