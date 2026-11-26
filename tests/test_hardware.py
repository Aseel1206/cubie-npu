"""Needs a Cubie A7A with the NPU driver and the VIPLite SDK. Point CUBIE_NPU_TEST_NB at any .nb with one uint8 image input."""
import os
import numpy as np
import pytest

NB = os.environ.get("CUBIE_NPU_TEST_NB")
pytestmark = pytest.mark.skipif(not NB, reason="set CUBIE_NPU_TEST_NB=/path/to/model.nb to run on hardware")


def test_describe_and_run():
    from cubie_npu import Model
    with Model(NB) as m:
        assert m.inputs and m.outputs
        x = np.random.randint(0, 255, m.inputs[0].shape).astype(np.uint8) if m.inputs[0].dtype == np.uint8 else np.zeros(m.inputs[0].shape, m.inputs[0].dtype)
        a = m.run(x); b = m.run(x)
        assert all(np.array_equal(p, q) for p, q in zip(a, b))        # deterministic
        assert all(o.dtype == np.float32 for o in a)
        assert m.last_run_ms > 0


def test_wrong_size_is_a_clear_error():
    from cubie_npu import Model
    with Model(NB) as m:
        with pytest.raises(ValueError, match="expects"):
            m.run(np.zeros(3, np.uint8))


def test_benchmark():
    from cubie_npu import Model
    with Model(NB) as m:
        r = m.benchmark(10)
        assert r["mean_ms"] > 0 and r["fps"] > 0
