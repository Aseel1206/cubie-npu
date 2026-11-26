"""`cubie-npu gpu-bench`: GPU vs OpenCV-on-CPU for the same image work, with the CPU time each one really burns."""
import time
import numpy as np
from . import Image, info


def _measure(fn, n, warm=3):
    for _ in range(warm):
        fn()
    t0, c0 = time.perf_counter(), time.process_time()
    for _ in range(n):
        fn()
    wall = (time.perf_counter() - t0) / n * 1e3; cpu = (time.process_time() - c0) / n * 1e3
    return wall, cpu


def scene(width, height, cv2):
    """A synthetic but camera-like frame: smooth gradient, filled rectangles and circles (edges + corners), mild noise and blur.
    (Pure random noise would be a worst case for corner detection and nothing like a real image.)"""
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:height, 0:width]
    base = np.stack([60 + 90 * x / width, 70 + 80 * y / height, 110 + 40 * np.sin(x / 90.0)], -1)
    img = np.clip(base + rng.normal(0, 3, base.shape), 0, 255).astype(np.uint8)
    if cv2 is not None:
        for _ in range(60):
            x0, y0 = int(rng.integers(0, width - 80)), int(rng.integers(0, height - 80))
            cv2.rectangle(img, (x0, y0), (x0 + int(rng.integers(20, 160)), y0 + int(rng.integers(20, 120))), tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
        for _ in range(25):
            cv2.circle(img, (int(rng.integers(0, width)), int(rng.integers(0, height))), int(rng.integers(5, 50)), tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def run(n=30, width=1280, height=720, out=print):
    try:
        import cv2
    except ImportError:
        out("OpenCV not installed (pip install opencv-python-headless); showing GPU numbers only")
        cv2 = None
    frame = scene(width, height, cv2); i = info()
    out("GPU: %s | frame %dx%d | %d iterations each\n" % (i["name"], width, height, n))
    out("%-44s %-22s %-22s" % ("task", "OpenCV (1 thread)", "GPU"))
    out("%-44s %-22s %-22s" % ("", "wall ms / CPU ms", "wall ms / CPU ms"))
    if cv2 is not None:
        cv2.setNumThreads(1)      # OpenCV's worker threads spin while idle, which inflates its CPU time; 1 thread is its true cost (the wall time is the same on this board)
    rows = []
    s = 224; r = s / max(height, width); nh = round(height * r); top = (s - nh) // 2
    dev = Image(frame)
    def cpu_letterbox():
        o = np.zeros((s, s, 3), np.uint8); o[top:top + nh] = cv2.resize(frame, (s, nh), interpolation=cv2.INTER_AREA); return o
    rows.append(("letterbox 1280x720 -> 224 (area)", cpu_letterbox if cv2 else None, lambda: Image(frame).letterbox(s, "area")[0].download(), "upload + op + download"))
    rows.append(("  same, frame already on the GPU", None, lambda: dev.letterbox(s, "area")[0].download(), "op + download (224x224x3)"))
    def cpu_feat():
        g = cv2.cvtColor(cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY); g = cv2.GaussianBlur(g, (5, 5), 0)
        return cv2.FastFeatureDetector_create(25, True, cv2.FAST_FEATURE_DETECTOR_TYPE_9_16).detect(g)
    rows.append(("feature front end: resize+gray+blur5+FAST", cpu_feat if cv2 else None,
                 lambda: Image(frame).resize((640, 360), "area").to_gray().gaussian_blur(5).fast(25), "upload + 4 ops + keypoints back"))
    rows.append(("  same, frame already on the GPU", None, lambda: dev.resize((640, 360), "area").to_gray().gaussian_blur(5).fast(25), "4 ops + keypoints back"))
    M = np.array([[1.3, -0.5, 400.0], [0.5, 1.3, 100.0]], np.float32)
    rows.append(("rotated 224x224 crop (warpAffine)", (lambda: cv2.warpAffine(frame, M, (224, 224), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)) if cv2 else None,
                 lambda: dev.warp_affine(M, (224, 224)).download(), "op + download"))
    for name, cpu_fn, gpu_fn, note in rows:
        gw, gc = _measure(gpu_fn, n)
        if cpu_fn:
            cw, cc = _measure(cpu_fn, n); cs = "%7.2f / %6.2f" % (cw, cc)
        else:
            cs = "-"
        out("%-44s %-22s %7.2f / %6.2f   %s" % (name, cs, gw, gc, note))
    out("\nCPU ms = CPU time this process used per iteration (lower = more CPU left for other work). OpenCV is pinned to 1 thread: its worker threads spin and would inflate its CPU time.")
