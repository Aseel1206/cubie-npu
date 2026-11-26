#!/usr/bin/env python3
"""Hand tracking with both MediaPipe networks on the NPU, in one Python process.

    python hands.py photo.jpg out.png                      # one image
    python hands.py photo.jpg out.png --bench 100          # + time the whole pipeline (preprocess + NPU + decode + crop) on that image
    python hands.py 0 --show                               # camera 0 (needs a display and a camera)

Models: palm_detector_int8pcq.nb and hand_landmark_int8pcq.nb from github.com/Aseel1206/radxa-cubie-a7a-npu-mediapipe-hands
"""
import argparse, time
import cv2, numpy as np
from cubie_npu import Model
from hands_utils import letterbox, decode, roi_from_det, crop, EDGES


class HandTracker:
    def __init__(self, palm_nb, landmark_nb):
        self.palm, self.lm = Model(palm_nb), Model(landmark_nb)

    def __call__(self, bgr):
        """-> list of dicts: score, landmarks (21,2) in image pixels, presence, roi corners"""
        lb, (r, l, t) = letterbox(bgr)
        reg, sc = self.palm(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB))          # uint8 RGB in; the 1/255 scaling is inside the .nb
        hands = []
        for score, box, kp in decode(reg[0], sc[0]):
            cx, cy, size, rot = roi_from_det(box, kp, r, l, t)
            patch, M = crop(bgr, cx, cy, size, rot)
            out = self.lm(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))          # [landmarks(1,63), presence(1,1), handedness(1,1), world(1,63)]
            pts = (M @ np.c_[out[0].reshape(21, 3)[:, :2], np.ones(21)].T).T
            corners = (M @ np.array([[0, 0, 1], [224, 0, 1], [224, 224, 1], [0, 224, 1]]).T).T.astype(np.int32)
            hands.append({"score": score, "landmarks": pts, "presence": float(out[1][0, 0]), "roi": corners})
        return hands


def draw(im, hands):
    lw = max(1, round(im.shape[1] / 400))
    for k, h in enumerate(hands):
        col = [(0, 0, 255), (255, 128, 0)][k % 2]; p = h["landmarks"]
        for a, b in EDGES: cv2.line(im, tuple(p[a].astype(int)), tuple(p[b].astype(int)), col, lw + 1)
        for q in p: cv2.circle(im, tuple(q.astype(int)), lw + 2, (255, 255, 255), -1)
        cv2.polylines(im, [h["roi"]], True, (0, 255, 255), lw)
    return im


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("source"); ap.add_argument("out", nargs="?")
    ap.add_argument("--palm", default="palm_detector_int8pcq.nb"); ap.add_argument("--lm", default="hand_landmark_int8pcq.nb")
    ap.add_argument("--bench", type=int, default=0); ap.add_argument("--show", action="store_true")
    a = ap.parse_args(); trk = HandTracker(a.palm, a.lm)
    if a.source.isdigit():
        cap = cv2.VideoCapture(int(a.source)); n = 0; t0 = time.time()
        while True:
            ok, fr = cap.read()
            if not ok: break
            hs = trk(fr); n += 1
            if a.show:
                cv2.imshow("hands", draw(fr, hs))
                if cv2.waitKey(1) == 27: break
            if n % 30 == 0: print("%.1f FPS (whole loop incl. camera)" % (n / (time.time() - t0)))
    else:
        im = cv2.imread(a.source); hs = trk(im)
        for k, h in enumerate(hs): print("hand %d: palm score %.2f, landmark presence %.2f" % (k, h["score"], h["presence"]))
        if a.bench:
            t = []
            for _ in range(a.bench): s = time.perf_counter(); trk(im); t.append((time.perf_counter() - s) * 1e3)
            t = np.array(t[3:]); print("full pipeline (python, 1 hand): mean %.1f ms  p50 %.1f  p95 %.1f  -> %.0f FPS" % (t.mean(), np.median(t), np.percentile(t, 95), 1000 / t.mean()))
        if a.out: cv2.imwrite(a.out, draw(im, hs)); print("wrote", a.out)
