"""Hugging-Face-style pipelines: name a task or a model, hand it an image (path, URL, NumPy, PIL) or text, get results.

    from cubie_npu import pipeline
    clf = pipeline("image-classification")           # downloads the .nb on first use, loads it onto the NPU
    clf("cat.jpg")                                   # [{'label': 'tabby, tabby cat', 'score': 0.62}, ...]

    pipeline("hand-landmarks")("hand.jpg")           # 21 landmarks per hand, in image pixels
    pipeline("depth-estimation")("room.jpg")["depth"]
    pipeline("face-detection")("people.jpg")
    pipeline("text-generation")("Tell me a joke")

Images are RGB. A NumPy array is taken as RGB (OpenCV gives BGR: pass frame[..., ::-1]).
"""
import json, os, struct, time
import numpy as np
from . import hub
from ._image import load_image, _cv2
from .model import Model

TASKS = {"image-classification": "deit-tiny", "depth-estimation": "depth-anything-v2-small", "hand-landmarks": "mediapipe-hands",
         "face-detection": "blazeface", "text-generation": "smollm2-135m"}


class Pipeline:
    task = None

    def __init__(self, model, variant=None, progress=None):
        self.name, self.entry = hub.resolve(model)
        self.variant = variant or self.entry["default"]
        self.files = hub.pull(self.name, self.variant, progress=progress)

    def close(self):
        for v in vars(self).values():
            if isinstance(v, Model):
                v.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __repr__(self):
        return "%s(%s, %s)" % (type(self).__name__, self.name, self.variant)


class ImageClassification(Pipeline):
    task = "image-classification"

    def __init__(self, model="deit-tiny", variant=None, progress=None):
        super().__init__(model, variant, progress)
        self.model = Model(self.files["nb"])
        self.labels = json.load(open(self.files["config.json"]))["id2label"]

    def __call__(self, image, top_k=5):
        """-> [{'label', 'score'}] best first. Resize short side to 224, centre crop (the normalisation is inside the .nb)."""
        cv2 = _cv2(); im = load_image(image); h, w = im.shape[:2]; s = 224 / min(h, w)
        im = cv2.resize(im, (max(224, round(w * s)), max(224, round(h * s))), interpolation=cv2.INTER_AREA); h, w = im.shape[:2]
        y, x = (h - 224) // 2, (w - 224) // 2
        logits = self.model(np.ascontiguousarray(im[y:y + 224, x:x + 224]))[0]
        p = np.exp(logits - logits.max()); p /= p.sum()
        return [{"label": self.labels[str(i)], "score": float(p[i])} for i in np.argsort(-logits)[:top_k]]


class DepthEstimation(Pipeline):
    task = "depth-estimation"
    SIZE = (336, 252)                                   # width, height of the compiled network

    def __init__(self, model="depth-anything-v2-small", variant=None, progress=None):
        super().__init__(model, variant, progress)
        self.model = Model(self.files["nb"])

    def __call__(self, image):
        """-> {'depth': float32 (H, W) at the input size, bigger = closer (relative inverse depth), 'raw': (252, 336), 'image': RGB map for display}"""
        cv2 = _cv2(); im = load_image(image); h, w = im.shape[:2]
        inp = cv2.resize(im, self.SIZE, interpolation=cv2.INTER_AREA if w > self.SIZE[0] else cv2.INTER_LINEAR)
        raw = self.model(inp)
        raw = np.asarray(raw, np.float32).reshape(self.SIZE[1], self.SIZE[0])
        depth = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR)
        lo, hi = float(depth.min()), float(depth.max())
        vis = cv2.applyColorMap(np.clip((depth - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
        return {"depth": depth, "raw": raw, "image": np.ascontiguousarray(vis)}


class FaceDetection(Pipeline):
    task = "face-detection"

    def __init__(self, model="blazeface", variant=None, progress=None):
        super().__init__(model, variant, progress)
        self.model = Model(self.files["nb"])

    def __call__(self, image, threshold=0.5):
        """-> [{'score', 'box': (x0, y0, x1, y1) px, 'keypoints': (6, 2) px (eyes, nose, mouth, ears)}]"""
        from . import _hands
        im = load_image(image); lb, (r, l, t) = _hands.letterbox(im, 128)
        out = self.model(lb); reg, sc = (out[0], out[1]) if out[0].size > out[1].size else (out[1], out[0])
        res = []
        for score, box, kp in _hands.decode_face(reg[0], sc[0], threshold):
            b = np.array([(box[0] * 128 - l) / r, (box[1] * 128 - t) / r, (box[2] * 128 - l) / r, (box[3] * 128 - t) / r])
            k = np.stack([(kp[:, 0] * 128 - l) / r, (kp[:, 1] * 128 - t) / r], 1)
            res.append({"score": score, "box": tuple(float(v) for v in b), "keypoints": k})
        return res


class HandLandmarks(Pipeline):
    task = "hand-landmarks"

    def __init__(self, model="mediapipe-hands", variant=None, progress=None):
        super().__init__(model, variant, progress)
        self.palm, self.landmark = Model(self.files["palm"]), Model(self.files["landmark"])

    def __call__(self, image, threshold=0.5):
        """-> [{'score' (palm), 'presence' (landmark confidence), 'landmarks': (21, 2) px, 'roi': (4, 2) px corners}]"""
        from . import _hands
        im = load_image(image); lb, (r, l, t) = _hands.letterbox(im)
        reg, sc = self.palm(lb)
        hands = []
        for score, box, kp in _hands.decode_palm(reg[0], sc[0], threshold):
            cx, cy, size, rot = _hands.roi_from_det(box, kp, r, l, t)
            patch, M = _hands.crop(im, cx, cy, size, rot)
            out = self.landmark(patch)                 # [landmarks (1,63), presence (1,1), handedness (1,1), world (1,63)]
            pts = (M @ np.c_[out[0].reshape(21, 3)[:, :2], np.ones(21)].T).T
            corners = (M @ np.array([[0, 0, 1], [224, 0, 1], [224, 224, 1], [0, 224, 1]]).T).T
            hands.append({"score": score, "presence": float(out[1][0, 0]), "landmarks": pts, "roi": corners})
        return hands


class TextGeneration(Pipeline):
    """Chat with SmolLM2-135M. The conversation continues between calls (the NPU keeps a 128-token KV cache); reset() starts over."""
    task = "text-generation"
    EOS = 2

    def __init__(self, model="smollm2-135m", variant=None, progress=None):
        super().__init__(model, variant, progress)
        try:
            from tokenizers import Tokenizer
        except ImportError:
            raise ImportError('text generation needs the tokenizers package: pip install "cubie-npu[llm]"') from None
        self.tok = Tokenizer.from_file(self.files["tokenizer.json"])
        self.model = m = Model(self.files["nb"])
        self.X, self.COS, self.SIN, self.MASK, self.KT, self.VT = m.inputs[:6]
        self.C = self.MASK.shape[-1]; self.HD = 64; self.DM = 576; self.CTX = self.C - 1
        self.E = np.memmap(self.files["embed.f32"], dtype=np.float32, mode="r").reshape(-1, self.DM)
        self.INV = 1.0 / (100000 ** (np.arange(0, self.HD, 2) / self.HD))
        L3 = self.KT.shape[1]
        with open(self.files["prefix.bin"], "rb") as f:        # KV cache of the fixed chat prefix, computed offline in float
            self.P = struct.unpack("<i", f.read(4))[0]
            kf = np.frombuffer(f.read(L3 * self.P * self.HD * 4), np.float32).reshape(L3, self.P, self.HD)
            vf = np.frombuffer(f.read(L3 * self.P * self.HD * 4), np.float32).reshape(L3, self.P, self.HD)
        self._pk0 = np.zeros(self.KT.shape, np.int16); self._pv0 = np.zeros(self.VT.shape, np.int16)
        self._pk0[0, :, :self.P] = self._q(kf, self.KT); self._pv0[0, :, :self.P] = self._q(vf, self.VT)
        self.reset()

    @staticmethod
    def _q(x, t):
        return np.clip(np.rint(x * 2.0 ** t.fixed_point_pos), -32768, 32767).astype(np.int16)

    def reset(self):
        self.pk, self.pv = self._pk0.copy(), self._pv0.copy(); self.pos = self.P; self.turn = 0

    def _step(self, token):
        pos = self.pos; f = pos * self.INV; e = np.concatenate([f, f]); C = self.C
        mask = np.full((1, 1, 1, C), -30.0, np.float32); mask[..., :pos] = 0; mask[..., C - 1] = 0
        m = self.model
        m.set_input(0, self.E[token][None]); m.set_input(1, np.cos(e)[None].astype(np.float32)); m.set_input(2, np.sin(e)[None].astype(np.float32))
        m.set_input(3, mask); m.set_input(4, self.pk); m.set_input(5, self.pv)
        m.invoke()
        nk, nv = m.get_output(1), m.get_output(2)
        self.pk[0, :, pos] = self._q(nk[0, :, 0], self.KT); self.pv[0, :, pos] = self._q(nv[0, :, 0], self.VT); self.pos = pos + 1
        return m.get_output(0)[0]

    @staticmethod
    def _sample(logits, past, temperature, top_k=40):
        l = logits.copy()
        for g in set(past):
            l[g] = l[g] / 1.1 if l[g] > 0 else l[g] * 1.1                  # repetition penalty
        if temperature <= 0:
            return int(l.argmax())
        top = np.argpartition(-l, top_k)[:top_k]; p = np.exp((l[top] - l[top].max()) / temperature); p /= p.sum()
        return int(np.random.choice(top, p=p))

    def __call__(self, prompt, max_new_tokens=120, temperature=0.7, stream=False):
        """Answer `prompt`. Returns the reply as a string, or with stream=True a generator of text pieces."""
        gen = self._generate(prompt, max_new_tokens, temperature)
        return gen if stream else "".join(gen)

    def _generate(self, prompt, max_new, temperature):
        end = "<|im_end|>\n<|im_start|>assistant\n"
        ids = self.tok.encode((prompt if self.turn == 0 else "<|im_end|>\n<|im_start|>user\n" + prompt) + end).ids
        if self.pos + len(ids) + 2 >= self.CTX:                            # context full: start a fresh conversation
            self.reset(); ids = self.tok.encode(prompt + end).ids
        if len(ids) + 2 >= self.CTX - self.P:
            raise ValueError("prompt is too long for the %d-token context (%d tokens free)" % (self.CTX, self.CTX - self.P))
        for t in ids:
            logits = self._step(t)
        out = []; shown = ""; t0 = time.time()
        while len(out) < max_new and self.pos < self.CTX - 1:
            nxt = self._sample(logits, out, temperature)
            if nxt == self.EOS:
                break
            out.append(nxt); txt = self.tok.decode(out)
            if not txt.endswith("�"):
                yield txt[len(shown):]; shown = txt
            logits = self._step(nxt)
        self.turn += 1; self.last_tokens_per_s = len(out) / max(time.time() - t0, 1e-9)


_CLASSES = {c.task: c for c in (ImageClassification, DepthEstimation, FaceDetection, HandLandmarks, TextGeneration)}


def pipeline(task_or_model, model=None, variant=None, progress=None):
    """pipeline('image-classification') or pipeline('mediapipe-hands') or pipeline('depth-estimation', variant='int16')."""
    name, entry = hub.resolve(model or TASKS.get(task_or_model, task_or_model))
    return _CLASSES[entry["task"]](name, variant, progress)
