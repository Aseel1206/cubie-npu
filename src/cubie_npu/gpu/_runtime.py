"""OpenCL context, kernel cache and the device-resident Image type."""
import math
import numpy as np
from .._backend import NPUError
from . import _setup, _kernels

_state = {}
i32, f32 = np.int32, np.float32


def _cl():
    try:
        import pyopencl as cl
    except ImportError:
        raise NPUError('the GPU helpers need pyopencl: pip install "cubie-npu[gpu]"')
    return cl


def _ctx():
    """(cl, context, queue, program) - created on first use."""
    if "ctx" in _state:
        return _state["cl"], _state["ctx"], _state["q"], _state["prog"]
    _setup.configure()
    cl = _cl()
    dev = None
    try:
        for p in cl.get_platforms():
            for d in p.get_devices(cl.device_type.GPU):
                if dev is None or "PowerVR" in d.name:
                    dev = d
    except cl.Error:
        pass
    if dev is None:
        raise NPUError("no OpenCL GPU found. On the Cubie A7A the PowerVR driver (/usr/lib/libPVROCL.so) must be installed; "
                       "set CUBIE_NPU_OCL_LIB=/path/to/libPVROCL.so if it lives elsewhere, or OCL_ICD_VENDORS to your own ICD folder.")
    ctx = cl.Context([dev]); q = cl.CommandQueue(ctx)
    prog = cl.Program(ctx, _kernels.SRC).build()
    _state.update(cl=cl, ctx=ctx, q=q, prog=prog, dev=dev, kernels={})
    return cl, ctx, q, prog


def _kernel(name):
    cl, ctx, q, prog = _ctx()
    k = _state["kernels"].get(name)
    if k is None:
        k = _state["kernels"][name] = cl.Kernel(prog, name)
    return k


def _launch(name, gsize, *args):
    _, _, q, _ = _ctx()
    return _kernel(name)(q, gsize, None, *args)


_pool = {}                     # rounded size -> [free cl.Buffer]; buffers are reused, never released while kernels may still use them
_POOL_KEEP = 6


def _bucket(nbytes):
    return max(4096, (int(nbytes) + 4095) // 4096 * 4096)


def _alloc(nbytes, flags=None):
    """A device buffer of at least nbytes. Reuse matters: releasing a buffer that a running kernel still uses makes the driver
    busy-wait on the CPU until the kernel is done, which is exactly the CPU time we want to save."""
    cl, ctx, q, _ = _ctx(); b = _bucket(nbytes); free = _pool.get(b)
    if free:
        return free.pop()
    return cl.Buffer(ctx, flags or cl.mem_flags.READ_WRITE, b)


def _recycle(buf, nbytes):
    lst = _pool.setdefault(_bucket(nbytes), [])
    if len(lst) < _POOL_KEEP:
        lst.append(buf)             # in-order queue: any later command that reuses it runs after the earlier ones


def _wait(event, sleep=0.0002):
    """Wait for an event without spinning the CPU."""
    import time
    cl, _, q, _ = _ctx(); q.flush(); done = cl.command_execution_status.COMPLETE
    while event.get_info(cl.event_info.COMMAND_EXECUTION_STATUS) != done:
        time.sleep(sleep)
    event.wait()


def available():
    """True if an OpenCL GPU can be used (pyopencl installed and the PowerVR driver reachable)."""
    try:
        _ctx()
        return True
    except Exception:
        return False


def info():
    """Dict describing the GPU."""
    cl, ctx, q, prog = _ctx(); d = _state["dev"]
    return {"name": d.name.strip(), "vendor": d.vendor, "opencl": d.version.strip(), "driver": d.driver_version,
            "compute_units": d.max_compute_units, "max_clock_mhz": d.max_clock_frequency, "global_mem_mb": d.global_mem_size // 2**20,
            "max_alloc_mb": d.max_mem_alloc_size // 2**20, "local_mem_bytes": d.local_mem_size, "fp16": "cl_khr_fp16" in d.extensions,
            "unified_memory": bool(d.host_unified_memory)}


class Image:
    """An image that lives in GPU memory. Operations return new Images without copying anything back to the CPU.

    Shape is (height, width, channels); 2-D arrays are treated as one channel. `Image(array)` uploads; `.download()` returns NumPy.
    Chain operations on the device and download only what you need (a small result, a keypoint list): host<->GPU copies are slow
    on this board (about 1.4 GB/s), so they are what decides whether the GPU is worth it."""

    def __init__(self, data=None, *, shape=None, dtype=np.uint8, _buf=None):
        cl, ctx, q, _ = _ctx()
        self.dtype = np.dtype(dtype)
        if data is not None:
            a = np.ascontiguousarray(data)
            self.dtype = a.dtype
            h, w = a.shape[:2]; c = a.shape[2] if a.ndim == 3 else 1
            self._squeeze = a.ndim == 2
            self.shape = (h, w, c)
            self._nbytes = max(a.nbytes, 4); self.buf = _alloc(self._nbytes)
            cl.enqueue_copy(q, self.buf, a)
        else:
            h, w, c = shape if len(shape) == 3 else (shape[0], shape[1], 1)
            self._squeeze = len(shape) == 2
            self.shape = (h, w, c)
            self._nbytes = max(h * w * c * self.dtype.itemsize, 4); self.buf = _buf if _buf is not None else _alloc(self._nbytes)

    def __del__(self):
        try:
            _recycle(self.buf, self._nbytes)
        except Exception:
            pass

    @property
    def height(self): return self.shape[0]
    @property
    def width(self): return self.shape[1]
    @property
    def channels(self): return self.shape[2]

    def download(self):
        cl, _, q, _ = _ctx()
        h, w, c = self.shape
        out = np.empty((h, w) if self._squeeze else (h, w, c), self.dtype)
        _wait(cl.enqueue_copy(q, out, self.buf, is_blocking=False))
        return out

    def _new(self, h, w, c, dtype=None):
        return Image(shape=(h, w, c) if c != 1 or not self._squeeze else (h, w), dtype=dtype or self.dtype)

    # -- geometry ---------------------------------------------------------------------------------------------
    def resize(self, size, mode="bilinear"):
        """size = (width, height). mode 'bilinear' (like cv2.INTER_LINEAR) or 'area' (like cv2.INTER_AREA; use it for big downscales)."""
        dw, dh = int(size[0]), int(size[1])
        return self._resize_into(dw, dh, 0, 0, dw, dh, mode, 0)

    def letterbox(self, size, mode="bilinear", pad=0):
        """Scale to fit size x size keeping the aspect ratio, pad the rest. Returns (Image, (scale, left, top)) like the MediaPipe helpers."""
        s = int(size); h, w, _ = self.shape; r = s / max(h, w); nh, nw = round(h * r), round(w * r); t, l = (s - nh) // 2, (s - nw) // 2
        return self._resize_into(s, s, l, t, nw, nh, mode, pad), (r, l, t)

    def _resize_into(self, dw, dh, ox, oy, rw, rh, mode, pad):
        if self.dtype != np.uint8:
            raise ValueError("resize works on uint8 images")
        h, w, c = self.shape; out = self._new(dh, dw, c)
        name = {"bilinear": "resize_bilinear", "area": "resize_area"}[mode]
        _launch(name, (dw, dh), self.buf, out.buf, i32(w), i32(h), i32(c), i32(dw), i32(dh), i32(ox), i32(oy), i32(rw), i32(rh), np.uint8(pad))
        return out

    def warp_affine(self, M, size):
        """dst(x, y) = src(M @ [x, y, 1]) with bilinear sampling and a black border. M is 2x3 and maps destination -> source
        (cv2.warpAffine with WARP_INVERSE_MAP). size = (width, height)."""
        M = np.asarray(M, np.float64).reshape(2, 3); dw, dh = int(size[0]), int(size[1]); h, w, c = self.shape; out = self._new(dh, dw, c)
        _launch("warp_affine", (dw, dh), self.buf, out.buf, i32(w), i32(h), i32(c), i32(dw), i32(dh), *[f32(v) for v in M.ravel()])
        return out

    # -- colour -----------------------------------------------------------------------------------------------
    def swap_rb(self):
        """BGR <-> RGB."""
        if self.channels != 3: raise ValueError("needs 3 channels")
        out = self._new(self.height, self.width, 3); _launch("swap_rb", (self.height * self.width,), self.buf, out.buf, i32(self.height * self.width)); return out

    def to_gray(self, order="bgr"):
        if self.channels != 3: raise ValueError("needs 3 channels")
        n = self.height * self.width; out = Image(shape=(self.height, self.width), dtype=np.uint8)
        _launch("to_gray", (n,), self.buf, out.buf, i32(n), i32(1 if order.lower() == "rgb" else 0)); return out

    def normalize(self, mean=(0, 0, 0), std=(1, 1, 1), *, swap_rb=False, nchw=False, dtype="float32"):
        """(x - mean) / std for model input. Returns a NumPy array (the result leaves the GPU): float32 or float16,
        NHWC (1,H,W,3) or NCHW (1,3,H,W)."""
        cl, ctx, q, _ = _ctx()
        if self.channels != 3 or self.dtype != np.uint8: raise ValueError("needs a uint8 3-channel image")
        h, w = self.height, self.width; half = dtype == "float16"
        out = np.empty((1, 3, h, w) if nchw else (1, h, w, 3), np.float16 if half else np.float32)
        ob = _alloc(out.nbytes)
        _launch("normalize", (w, h), self.buf, ob, i32(w), i32(h), i32(bool(swap_rb)), i32(bool(nchw)), i32(half),
                *[f32(v) for v in mean], *[f32(1.0 / v) for v in std])
        _wait(cl.enqueue_copy(q, out, ob, is_blocking=False)); _recycle(ob, out.nbytes); return out

    # -- filters ----------------------------------------------------------------------------------------------
    def gaussian_blur(self, ksize=5, sigma=0):
        """Like cv2.GaussianBlur((k,k), sigma) with reflect-101 borders. sigma<=0 picks OpenCV's default for the kernel size."""
        cl, ctx, q, _ = _ctx(); k = _gauss(int(ksize), float(sigma)); r = len(k) // 2; h, w, c = self.shape
        kb = _alloc(k.nbytes); cl.enqueue_copy(q, kb, k)
        tmp = _alloc(h * w * c * 4); out = self._new(h, w, c)
        _launch("blur_h", (w, h), self.buf, tmp, kb, i32(r), i32(w), i32(h), i32(c))
        _launch("blur_v", (w, h), tmp, out.buf, kb, i32(r), i32(w), i32(h), i32(c)); _recycle(tmp, h * w * c * 4); _recycle(kb, k.nbytes); return out

    def pyramid(self, levels=4, scale=1.2, mode="bilinear"):
        """[level0 (self), level1, ...], each level `scale` times smaller than the previous one (ORB-style)."""
        out = [self]; h, w = self.height, self.width
        for i in range(1, levels):
            f = scale ** i; out.append(self.resize((max(1, round(w / f)), max(1, round(h / f))), mode))
        return out

    def sobel(self):
        """3x3 Sobel gradients of a gray image -> (gx, gy) as int16 Images."""
        if self.channels != 1: raise ValueError("needs a gray image")
        h, w, _ = self.shape; gx = Image(shape=(h, w), dtype=np.int16); gy = Image(shape=(h, w), dtype=np.int16)
        _launch("sobel", (w, h), self.buf, gx.buf, gy.buf, i32(w), i32(h)); return gx, gy

    # -- features ---------------------------------------------------------------------------------------------
    def fast(self, threshold=20, nms=True, max_keypoints=4000):
        """FAST-9 corners of a gray image. Returns a float32 array (N, 3) of x, y, score, sorted by position; only these few
        numbers are copied back from the GPU. Scores follow OpenCV's definition."""
        cl, ctx, q, _ = _ctx()
        if self.channels != 1: raise ValueError("needs a gray image (use .to_gray())")
        h, w, _ = self.shape; maxn = int(max_keypoints)
        score = _alloc(h * w * 2); out = _alloc(maxn * 16); cnt = _alloc(4); cl.enqueue_copy(q, cnt, np.zeros(1, np.int32))
        _launch("fast_score", (w, h), self.buf, score, i32(w), i32(h), i32(threshold))
        _launch("fast_nms", (w, h), score, out, cnt, i32(w), i32(h), i32(maxn), i32(bool(nms)))
        n = np.empty(1, np.int32); _wait(cl.enqueue_copy(q, n, cnt, is_blocking=False)); n = min(int(n[0]), maxn)
        res = np.empty((n, 4), np.int32)
        if n: _wait(cl.enqueue_copy(q, res, out, is_blocking=False))
        _recycle(score, h * w * 2); _recycle(out, maxn * 16); _recycle(cnt, 4)
        kp = res[:, :3].astype(np.float32); return kp[np.lexsort((kp[:, 0], kp[:, 1]))] if n else kp


def _gauss(ksize, sigma):
    if ksize % 2 == 0 or ksize < 1: raise ValueError("ksize must be odd")
    if sigma <= 0 and ksize in (1, 3, 5, 7):          # OpenCV uses these exact small kernels
        return np.array({1: [1.0], 3: [0.25, 0.5, 0.25], 5: [0.0625, 0.25, 0.375, 0.25, 0.0625],
                         7: [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125]}[ksize], np.float32)
    if sigma <= 0: sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    x = np.arange(ksize) - (ksize - 1) / 2.0; k = np.exp(-(x * x) / (2 * sigma * sigma)); return (k / k.sum()).astype(np.float32)
