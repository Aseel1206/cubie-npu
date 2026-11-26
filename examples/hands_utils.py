"""CPU-side helpers for the MediaPipe palm detector + hand landmark models (same code as in
github.com/Aseel1206/radxa-cubie-a7a-npu-mediapipe-hands): letterbox, anchor decoding + NMS, rotated hand ROI."""
import math
import cv2
import numpy as np


def letterbox(im, s=192):
    """Resize keeping aspect ratio, pad with black to s x s. Returns (image, (scale, left, top))."""
    h, w = im.shape[:2]; r = s / max(h, w); nh, nw = round(h * r), round(w * r)
    out = np.zeros((s, s, 3), np.uint8); t, l = (s - nh) // 2, (s - nw) // 2
    out[t:t + nh, l:l + nw] = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
    return out, (r, l, t)


def _anchors():
    a = []
    for g, n in ((24, 2), (12, 6)):                 # 24x24x2 + 12x12x6 = 2016 anchors
        for y in range(g):
            for x in range(g):
                a.extend([((x + .5) / g, (y + .5) / g)] * n)
    return np.array(a, np.float32)


_A = _anchors()


def decode(reg, sc, thr=0.5, nms=0.3):
    """reg (2016,18) / sc (2016,) from the palm detector -> list of (score, box[x0,y0,x1,y1], keypoints(7,2)), all normalised to the 192x192 input."""
    p = 1 / (1 + np.exp(-np.clip(sc.reshape(-1), -80, 80))); r = reg.reshape(2016, 18) / 192.0
    cx = r[:, 0] + _A[:, 0]; cy = r[:, 1] + _A[:, 1]; w = r[:, 2]; h = r[:, 3]
    kp = r[:, 4:].reshape(2016, 7, 2) + _A[:, None, :]
    box = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    keep = []
    for i in np.argsort(-p):
        if p[i] < thr:
            break
        ok = True
        for j in keep:
            iw = max(0, min(box[i, 2], box[j, 2]) - max(box[i, 0], box[j, 0])); ih = max(0, min(box[i, 3], box[j, 3]) - max(box[i, 1], box[j, 1]))
            inter = iw * ih
            if inter / (w[i] * h[i] + w[j] * h[j] - inter) > nms:
                ok = False; break
        if ok:
            keep.append(i)
    return [(float(p[i]), box[i], kp[i]) for i in keep]


def roi_from_det(box, kp, r, l, t, scale=2.6, shift_y=-0.5):
    """Rotated square ROI (cx, cy, size, rotation) in original image pixels, MediaPipe style: rotate so wrist->middle-finger is vertical."""
    px = (kp[:, 0] * 192 - l) / r; py = (kp[:, 1] * 192 - t) / r
    rot = math.pi / 2 - math.atan2(-(py[2] - py[0]), px[2] - px[0]); rot -= 2 * math.pi * math.floor((rot + math.pi) / (2 * math.pi))
    cx = ((box[0] + box[2]) / 2 * 192 - l) / r; cy = ((box[1] + box[3]) / 2 * 192 - t) / r
    w = (box[2] - box[0]) * 192 / r; h = (box[3] - box[1]) * 192 / r
    cx += -h * shift_y * math.sin(rot); cy += h * shift_y * math.cos(rot)
    return cx, cy, max(w, h) * scale, rot


def crop(img, cx, cy, size, rot, out=224):
    """Warp the rotated ROI to out x out. Returns (patch, M) where M maps patch pixels -> image pixels."""
    c, s = math.cos(rot), math.sin(rot); k = size / out
    M = np.array([[c * k, -s * k, cx - (c * k * out / 2 - s * k * out / 2)], [s * k, c * k, cy - (s * k * out / 2 + c * k * out / 2)]], np.float32)
    return cv2.warpAffine(img, M, (out, out), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0), M


EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
         (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
