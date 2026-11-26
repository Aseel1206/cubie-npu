"""High-level Model class: load a .nb, describe its inputs/outputs, run it with NumPy arrays."""
import ctypes, os, threading, time
from collections import namedtuple
import numpy as np
from . import _backend
from ._backend import NPUError, CnpuTensor

# vip_buffer_format_e / vip_buffer_quantize_format_e (vip_lite.h)
_FORMAT = {0: np.float32, 1: np.float16, 2: np.uint8, 3: np.int8, 4: np.uint16, 5: np.int16, 6: np.int8, 8: np.int32,
           9: np.uint32, 10: np.int64, 11: np.uint64, 12: np.float64, 16: np.bool_}
_FORMAT_NAME = {0: "fp32", 1: "fp16", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16", 6: "char", 7: "bfp16", 8: "int32",
                9: "uint32", 10: "int64", 11: "uint64", 12: "fp64", 16: "bool"}
_QUANT_NAME = {0: "none", 1: "dynamic_fixed_point", 2: "affine"}

TensorInfo = namedtuple("TensorInfo", "index name shape dtype quant fixed_point_pos scale zero_point nbytes")
TensorInfo.__doc__ = ("Description of one network input or output. `shape` is in NumPy order (batch first), `dtype` is the dtype of the "
                      "buffer the NPU reads/writes, `quant` is 'none', 'dynamic_fixed_point' (value = q * 2**-fixed_point_pos) or "
                      "'affine' (value = (q - zero_point) * scale).")


def _info(i, t, is_input):
    fmt = t.data_format
    if fmt == 7 or fmt not in _FORMAT:
        raise NPUError("unsupported tensor data format %d on %s %d" % (fmt, "input" if is_input else "output", i))
    dims = [int(t.sizes[k]) for k in range(t.ndims)]
    return TensorInfo(i, t.name.decode(errors="replace"), tuple(reversed(dims)), np.dtype(_FORMAT[fmt]), _QUANT_NAME.get(t.quant_format, "?"),
                      int(t.fixed_point_pos), float(t.scale), int(t.zero_point), int(t.bytes))


def _quantize(x, t):
    """float array -> the dtype/format the NPU wants for tensor t."""
    if t.quant == "dynamic_fixed_point":
        info = np.iinfo(t.dtype)
        return np.clip(np.rint(x * float(2 ** t.fixed_point_pos)), info.min, info.max).astype(t.dtype)
    if t.quant == "affine":
        info = np.iinfo(t.dtype)
        return np.clip(np.rint(x / t.scale) + t.zero_point, info.min, info.max).astype(t.dtype)
    if np.issubdtype(t.dtype, np.integer):
        info = np.iinfo(t.dtype)
        return np.clip(np.rint(x), info.min, info.max).astype(t.dtype)
    return x.astype(t.dtype)


def _dequantize(raw, t):
    if t.quant == "dynamic_fixed_point":
        return raw.astype(np.float32) * np.float32(2.0 ** -t.fixed_point_pos)
    if t.quant == "affine":
        return (raw.astype(np.float32) - t.zero_point) * np.float32(t.scale)
    return raw.astype(np.float32)


class Model:
    """A network loaded onto the NPU.

        m = Model("model.nb")
        print(m.inputs, m.outputs)
        out = m.run(x)               # x: NumPy array (see below) -> list of float32 arrays

    Inputs: pass an array whose dtype equals the input's native dtype (e.g. uint8 for networks that were built with the
    normalisation inside, as the models in the Cubie A7A repos are) and it is copied as-is; pass a float array for a
    quantised/fp16 input and it is converted (rounded, clipped) for you. Shapes only need the right number of elements.
    Outputs: converted to float32 (de-quantised) unless raw=True.

    A Model is not thread-safe for concurrent run() calls; calls are serialised with a lock. The NPU has one core, so
    two models used at once simply take turns.
    """

    def __init__(self, path):
        self.path = os.fspath(path)
        if not os.path.isfile(self.path):
            raise FileNotFoundError(self.path)
        L = _backend.lib()
        self._L = L
        self._h = L.cnpu_load(self.path.encode())
        if not self._h:
            raise NPUError("cannot load %s: %s" % (self.path, _backend.last_error()))
        self._lock = threading.Lock()
        nin, nout = ctypes.c_int(), ctypes.c_int()
        L.cnpu_counts(self._h, ctypes.byref(nin), ctypes.byref(nout))
        t = CnpuTensor()
        self.inputs = []
        self.outputs = []
        for i in range(nin.value):
            L.cnpu_tensor_info(self._h, 1, i, ctypes.byref(t)); self.inputs.append(_info(i, t, True))
        for i in range(nout.value):
            L.cnpu_tensor_info(self._h, 0, i, ctypes.byref(t)); self.outputs.append(_info(i, t, False))
        self.last_run_ms = None

    # -- low level -----------------------------------------------------------------------------------------------
    def set_input(self, index, array):
        """Copy one input into the NPU buffer (does not run). Useful when only some inputs change between runs.

        `array` may be: an array of the input's native dtype (copied as-is), a float array (converted to the native
        dtype using the tensor's quantisation), or raw bytes / a uint8 array with exactly `nbytes` bytes."""
        t = self.inputs[index]
        if isinstance(array, (bytes, bytearray, memoryview)):
            a = np.frombuffer(array, dtype=np.uint8)
        else:
            a = np.asarray(array)
        n = int(np.prod(t.shape))
        if a.dtype == t.dtype and a.size == n:
            pass
        elif a.dtype == np.uint8 and a.nbytes == t.nbytes:
            pass                                    # raw bytes of the right length
        elif a.size == n:
            a = _quantize(a.astype(np.float32, copy=False), t)
        else:
            raise ValueError("input %d (%s) expects %d values with shape %s (%s, %d bytes); got %d values of dtype %s with shape %s"
                             % (index, t.name, n, t.shape, t.dtype, t.nbytes, a.size, a.dtype, a.shape))
        a = np.ascontiguousarray(a)
        r = self._L.cnpu_set_input(self._h, index, a.ctypes.data_as(ctypes.c_void_p), a.nbytes)
        if r != 0:
            raise NPUError("set_input failed: " + _backend.last_error())

    def invoke(self):
        """Run on the current input buffers. Returns the NPU time in ms (flush + run + cache invalidate)."""
        ms = ctypes.c_double()
        r = self._L.cnpu_run(self._h, ctypes.byref(ms))
        if r != 0:
            raise NPUError("run failed: " + _backend.last_error())
        self.last_run_ms = ms.value
        return ms.value

    def get_output(self, index, raw=False):
        t = self.outputs[index]
        buf = np.empty(t.shape, t.dtype)
        r = self._L.cnpu_get_output(self._h, index, buf.ctypes.data_as(ctypes.c_void_p), buf.nbytes)
        if r != 0:
            raise NPUError("get_output failed: " + _backend.last_error())
        return buf if raw else _dequantize(buf, t)

    # -- convenience ---------------------------------------------------------------------------------------------
    def run(self, *inputs, raw=False):
        """Set all inputs, run, return the list of outputs (float32, shaped like the network output)."""
        if len(inputs) != len(self.inputs):
            raise ValueError("%s has %d inputs, got %d" % (os.path.basename(self.path), len(self.inputs), len(inputs)))
        with self._lock:
            for i, a in enumerate(inputs):
                self.set_input(i, a)
            self.invoke()
            return [self.get_output(i, raw=raw) for i in range(len(self.outputs))]

    def __call__(self, *inputs, raw=False):
        out = self.run(*inputs, raw=raw)
        return out[0] if len(out) == 1 else out

    def benchmark(self, n=100, warmup=3):
        """Time the NPU alone (inputs left as they are, random bytes if never set). Returns a dict of milliseconds."""
        with self._lock:
            for i, t in enumerate(self.inputs):
                raw = np.random.randint(0, 255, t.nbytes, dtype=np.uint8)
                self._L.cnpu_set_input(self._h, i, raw.ctypes.data_as(ctypes.c_void_p), raw.nbytes)
            for _ in range(warmup):
                self.invoke()
            ms = np.array([self.invoke() for _ in range(n)])
        return {"runs": n, "mean_ms": float(ms.mean()), "p50_ms": float(np.median(ms)), "p95_ms": float(np.percentile(ms, 95)),
                "min_ms": float(ms.min()), "max_ms": float(ms.max()), "fps": float(1000.0 / ms.mean())}

    def close(self):
        if getattr(self, "_h", None):
            self._L.cnpu_free(self._h)
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        def d(ts):
            return ", ".join("%s%s %s/%s" % (t.name or t.index, list(t.shape), t.dtype, t.quant) for t in ts)
        return "Model(%s | in: %s | out: %s)" % (os.path.basename(self.path), d(self.inputs), d(self.outputs))


def driver_version():
    """VIPLite driver version as an int (e.g. 0x00020003 for 2.0.3...)."""
    return int(_backend.lib().cnpu_driver_version())
