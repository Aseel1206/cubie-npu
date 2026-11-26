"""Tests that need no NPU: quantisation helpers, SDK discovery errors, importability on any machine."""
import os
import numpy as np
import pytest
import cubie_npu
from cubie_npu import model as M


def T(dtype, quant="none", fl=0, scale=1.0, zp=0, shape=(1, 4)):
    return M.TensorInfo(0, "t", shape, np.dtype(dtype), quant, fl, scale, zp, int(np.prod(shape)) * np.dtype(dtype).itemsize)


def test_import_without_sdk_does_not_fail():
    assert cubie_npu.__version__


def test_fixed_point_roundtrip():
    t = T(np.int16, "dynamic_fixed_point", fl=10)
    x = np.array([0.0, 1.0, -1.5, 0.123456, 40.0, -40.0], np.float32)
    q = M._quantize(x, t)
    assert q.dtype == np.int16
    assert q[1] == 1024 and q[2] == -1536
    assert q[4] == 32767 and q[5] == -32768                      # saturates instead of wrapping
    back = M._dequantize(q, t)
    assert np.abs(back[:4] - x[:4]).max() < 2 ** -10


def test_affine_roundtrip():
    t = T(np.uint8, "affine", scale=0.5, zp=10)
    x = np.array([0.0, 1.0, -5.0, 100.0], np.float32)
    q = M._quantize(x, t)
    assert list(q) == [10, 12, 0, 210]
    assert np.allclose(M._dequantize(q, t), (q.astype(np.float32) - 10) * 0.5)


def test_float16_and_uint8_passthrough():
    assert M._quantize(np.array([0.5, -30.0], np.float32), T(np.float16)).dtype == np.float16
    q = M._quantize(np.array([-3.0, 7.4, 300.0], np.float32), T(np.uint8))
    assert list(q) == [0, 7, 255]                                # unquantised integer input: round + clip


def test_find_sdk_error_message(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    for k in ("CUBIE_NPU_SDK_INCLUDE", "CUBIE_NPU_SDK_LIB", "VIPLITE_INCLUDE", "VIPLITE_LIB"):
        monkeypatch.delenv(k, raising=False)
    from cubie_npu import _backend
    monkeypatch.setattr(os.path, "isfile", lambda p: False)
    with pytest.raises(cubie_npu.NPUError, match="CUBIE_NPU_SDK_INCLUDE"):
        _backend.find_sdk()
