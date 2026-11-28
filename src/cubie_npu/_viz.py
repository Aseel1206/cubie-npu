"""Draw pipeline results on an RGB image (used by `cubie-npu run` and handy in notebooks)."""
import numpy as np
from ._image import _cv2
from ._hands import EDGES


def draw(task, im, result):
    cv2 = _cv2(); out = np.ascontiguousarray(im.copy()); lw = max(1, round(im.shape[1] / 400))
    if task == "depth-estimation":
        return result["image"]
    if task == "hand-landmarks":
        for k, h in enumerate(result):
            col = [(255, 0, 0), (0, 128, 255)][k % 2]; p = h["landmarks"]
            for a, b in EDGES: cv2.line(out, tuple(p[a].astype(int)), tuple(p[b].astype(int)), col, lw + 1)
            for q in p: cv2.circle(out, tuple(q.astype(int)), lw + 2, (255, 255, 255), -1)
            cv2.polylines(out, [h["roi"].astype(np.int32)], True, (255, 255, 0), lw)
    elif task == "face-detection":
        for f in result:
            x0, y0, x1, y1 = (int(v) for v in f["box"]); cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), lw + 1)
            for q in f["keypoints"]: cv2.circle(out, tuple(q.astype(int)), lw + 2, (255, 0, 0), -1)
            cv2.putText(out, "%.2f" % f["score"], (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * lw, (0, 255, 0), lw)
    return out
