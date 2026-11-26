#!/usr/bin/env python3
"""ImageNet classification with the DeiT-tiny .nb (github.com/Aseel1206/radxa-cubie-a7a-npu-deit-tiny).

    python classify.py photo.jpg deit_tiny_int8pcq.nb config.json
"""
import sys, json
import numpy as np, cv2
from cubie_npu import Model

img, nb, cfg = sys.argv[1:4]
im = cv2.imread(img); h, w = im.shape[:2]; s = 224 / min(h, w)
im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA); h, w = im.shape[:2]
y, x = (h - 224) // 2, (w - 224) // 2
rgb = cv2.cvtColor(im[y:y + 224, x:x + 224], cv2.COLOR_BGR2RGB)      # the mean/std normalisation is inside the .nb

model = Model(nb)
logits = model(rgb)[0]                                               # uint8 in, float32 logits out
p = np.exp(logits - logits.max()); p /= p.sum()
labels = json.load(open(cfg))["id2label"]
for i in np.argsort(-logits)[:5]:
    print(f"{p[i] * 100:5.1f}%  {labels[str(i)]}")
print(f"NPU time {model.last_run_ms:.1f} ms")
