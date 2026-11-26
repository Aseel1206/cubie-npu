"""GPU helpers for the Cubie A7A (PowerVR BXM-4-64 through OpenCL): image preprocessing and feature detection that keep the CPU free.

    pip install "cubie-npu[gpu]"

    from cubie_npu import gpu
    print(gpu.info())
    img = gpu.Image(frame_bgr)                         # upload once
    g = img.resize((640, 360)).to_gray()               # these run on the GPU, nothing is copied back
    keypoints = g.gaussian_blur(5).fast(threshold=20)  # (N,3) x, y, score - the only thing that comes back

Be realistic about this GPU: ~1 compute unit, ~19 GFLOPS fp32 and a slow (about 1.4 GB/s) link to host memory. It is not a faster
CPU; it is a second processor that can take image work off the CPU while the CPU (and the NPU) do other things.
"""
from . import _setup
from ._runtime import Image, available, info
from ._setup import configure, find_pvr_library


def resize(img, size, mode="bilinear"):
    """NumPy in, NumPy out convenience (uploads and downloads every call; chain `Image` operations instead when you can)."""
    return Image(img).resize(size, mode).download()


def letterbox(img, size, mode="bilinear", pad=0):
    out, params = Image(img).letterbox(size, mode, pad); return out.download(), params


def gaussian_blur(img, ksize=5, sigma=0):
    return Image(img).gaussian_blur(ksize, sigma).download()


def fast(gray, threshold=20, nms=True, max_keypoints=4000):
    return Image(gray).fast(threshold, nms, max_keypoints)


__all__ = ["Image", "available", "info", "configure", "find_pvr_library", "resize", "letterbox", "gaussian_blur", "fast"]
