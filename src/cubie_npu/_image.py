"""Turn whatever the user hands us (path, URL, bytes, NumPy array, PIL image) into an RGB uint8 array."""
import os, urllib.request
import numpy as np


def _cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        raise ImportError('image pipelines need OpenCV: pip install "cubie-npu[vision]"  (or: sudo apt install python3-opencv)') from None


def load_image(src):
    """-> (H, W, 3) uint8 RGB. NumPy arrays are taken as RGB (use arr[..., ::-1] for OpenCV's BGR)."""
    if isinstance(src, (str, os.PathLike)):
        s = os.fspath(src)
        if s.startswith(("http://", "https://")):
            from . import __version__
            with urllib.request.urlopen(urllib.request.Request(s, headers={"User-Agent": "cubie-npu/" + __version__}), timeout=30) as r:
                src = r.read()
        else:
            if not os.path.isfile(s):
                raise FileNotFoundError(s)
            src = open(s, "rb").read()
    if isinstance(src, (bytes, bytearray, memoryview)):
        cv2 = _cv2(); bgr = cv2.imdecode(np.frombuffer(src, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("could not decode the image data")
        return np.ascontiguousarray(bgr[..., ::-1])
    if hasattr(src, "convert") and hasattr(src, "size"):           # PIL.Image
        return np.asarray(src.convert("RGB"), dtype=np.uint8).copy()
    a = np.asarray(src)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    if a.ndim != 3 or a.shape[2] not in (1, 3, 4):
        raise ValueError("expected an image of shape (H, W, 3), got %s" % (a.shape,))
    if a.shape[2] == 1:
        a = np.repeat(a, 3, 2)
    a = a[..., :3]
    if a.dtype != np.uint8:
        a = np.clip(np.rint(a * 255.0 if a.dtype.kind == "f" and a.max() <= 1.0 else a), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)
