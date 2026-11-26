"""GPU helper tests: every operation is compared with OpenCV on the CPU. Runs only on a machine with the PowerVR OpenCL driver, pyopencl
and OpenCV (pip install pytest opencv-python-headless pyopencl); skipped elsewhere."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("pyopencl")
from cubie_npu import gpu

pytestmark = pytest.mark.skipif(not gpu.available(), reason="no OpenCL GPU")
rng = np.random.default_rng(1)


def photo(h=480, w=640, c=3):
    """Smooth-ish structured test image (gradients + blobs + noise) so interpolation and FAST have something to chew on."""
    y, x = np.mgrid[0:h, 0:w]
    base = 120 + 60 * np.sin(x / 23.0) * np.cos(y / 31.0)
    for _ in range(40):
        cy, cx, r, a = rng.integers(0, h), rng.integers(0, w), rng.integers(5, 40), rng.integers(-90, 90)
        base += a * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * r * r))
    img = np.stack([base + k * 10 for k in range(c)], -1) + rng.normal(0, 4, (h, w, c))
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img if c > 1 else img[..., 0]


def corners_image(h=480, w=640):
    """Gray image full of corners: random filled rectangles / circles / lines on a noisy background."""
    img = rng.integers(100, 140, (h, w), dtype=np.uint8)
    for _ in range(80):
        x0, y0 = int(rng.integers(0, w - 40)), int(rng.integers(0, h - 40)); x1, y1 = x0 + int(rng.integers(10, 60)), y0 + int(rng.integers(10, 60))
        cv2.rectangle(img, (x0, y0), (min(x1, w - 1), min(y1, h - 1)), int(rng.integers(0, 255)), -1)
    for _ in range(30):
        cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(3, 25)), int(rng.integers(0, 255)), -1)
    return img


def maxdiff(a, b):
    return int(np.abs(a.astype(int) - b.astype(int)).max())


def test_info():
    i = gpu.info(); assert "PowerVR" in i["name"] or i["compute_units"] >= 1


def test_resize_bilinear_down_and_up():
    im = photo()
    for size in [(224, 168), (1000, 700), (641, 479)]:
        g = gpu.Image(im).resize(size).download(); c = cv2.resize(im, size, interpolation=cv2.INTER_LINEAR)
        assert g.shape == c.shape and maxdiff(g, c) <= 2, size


def test_resize_area():
    im = photo(720, 1280)
    g = gpu.Image(im).resize((224, 126), "area").download(); c = cv2.resize(im, (224, 126), interpolation=cv2.INTER_AREA)
    assert maxdiff(g, c) <= 2


def test_letterbox_matches_cpu():
    im = photo(480, 640)
    out, (r, l, t) = gpu.Image(im).letterbox(192, "area"); out = out.download()
    nh, nw = round(480 * r), round(640 * r); ref = np.zeros((192, 192, 3), np.uint8); ref[t:t + nh, l:l + nw] = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
    assert (l, t) == ((192 - nw) // 2, (192 - nh) // 2) and maxdiff(out, ref) <= 2
    assert out[:t].sum() == 0                                           # padding is black


def test_colour():
    im = photo()
    g = gpu.Image(im)
    assert np.array_equal(g.swap_rb().download(), im[..., ::-1])
    # OpenCV's own rounding differs from the textbook fixed-point formula by one level on a fraction of a percent of pixels
    for order, code in (("bgr", cv2.COLOR_BGR2GRAY), ("rgb", cv2.COLOR_RGB2GRAY)):
        d = np.abs(g.to_gray(order).download().astype(int) - cv2.cvtColor(im, code).astype(int))
        assert d.max() <= 1 and (d > 0).mean() < 0.01


def test_normalize():
    im = photo(100, 120); mean, std = (123.675, 116.28, 103.53), (58.395, 57.12, 57.375)
    g = gpu.Image(im)
    ref = ((im[..., ::-1].astype(np.float32) - mean) / std)
    a = g.normalize(mean, std, swap_rb=True); assert a.shape == (1, 100, 120, 3) and np.abs(a[0] - ref).max() < 1e-4
    b = g.normalize(mean, std, swap_rb=True, nchw=True); assert b.shape == (1, 3, 100, 120) and np.abs(b[0] - ref.transpose(2, 0, 1)).max() < 1e-4
    h = g.normalize(mean, std, swap_rb=True, dtype="float16"); assert h.dtype == np.float16 and np.abs(h[0].astype(np.float32) - ref).max() < 5e-3


def test_warp_affine():
    im = photo(); ang = 0.4; c, s = np.cos(ang), np.sin(ang)
    M = np.array([[c * 1.7, -s * 1.7, 90.0], [s * 1.7, c * 1.7, 20.0]], np.float32)
    g = gpu.Image(im).warp_affine(M, (224, 224)).download()
    ref = cv2.warpAffine(im, M, (224, 224), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    d = np.abs(g.astype(int) - ref.astype(int))
    assert d.max() <= 3 and (d > 1).mean() < 0.01


@pytest.mark.parametrize("k", [3, 5, 7, 9, 15])
def test_gaussian_blur(k):
    for im in (photo(), photo(c=1)):
        g = gpu.Image(im).gaussian_blur(k).download(); ref = cv2.GaussianBlur(im, (k, k), 0)
        assert g.shape == ref.shape and maxdiff(g, ref) <= 1


def test_sobel():
    im = photo(c=1); gx, gy = gpu.Image(im).sobel()
    assert np.array_equal(gx.download(), cv2.Sobel(im, cv2.CV_16S, 1, 0, ksize=3)) and np.array_equal(gy.download(), cv2.Sobel(im, cv2.CV_16S, 0, 1, ksize=3))


def test_pyramid_sizes():
    p = gpu.Image(photo(480, 640)).pyramid(4, 1.2)
    assert [(x.width, x.height) for x in p] == [(640, 480), (533, 400), (444, 333), (370, 278)]


def kp_set(arr):
    return {(int(x), int(y)) for x, y, *_ in arr}


def test_fast_without_nms_matches_opencv_exactly():
    im = corners_image(); thr = 20
    ref = cv2.FastFeatureDetector_create(threshold=thr, nonmaxSuppression=False, type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16).detect(im)
    ours = gpu.Image(im).fast(thr, nms=False, max_keypoints=100000)
    assert len(ref) > 100
    assert kp_set(ours) == {(int(k.pt[0]), int(k.pt[1])) for k in ref}


def test_fast_with_nms_close_to_opencv():
    im = corners_image(); thr = 20
    ref = cv2.FastFeatureDetector_create(threshold=thr, nonmaxSuppression=True, type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16).detect(im)
    ours = gpu.Image(im).fast(thr, nms=True, max_keypoints=100000)
    a, b = kp_set(ours), {(int(k.pt[0]), int(k.pt[1])) for k in ref}
    jac = len(a & b) / len(a | b)
    assert jac > 0.9, (len(a), len(b), jac)


def test_chained_pipeline_returns_only_keypoints():
    frame = photo(720, 1280)
    kp = gpu.Image(frame).resize((640, 360), "area").to_gray().gaussian_blur(5).fast(25)
    assert kp.ndim == 2 and kp.shape[1] == 3 and (kp[:, 0] < 640).all() and (kp[:, 1] < 360).all()
