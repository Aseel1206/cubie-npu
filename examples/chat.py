#!/usr/bin/env python3
"""SmolLM2-135M chat on the NPU, the whole decode loop in Python (github.com/Aseel1206/radxa-cubie-a7a-npu-llm-smollm2-135m).

Needs next to this script (or pass --dir): smollm2_135m_c128_int16.nb, embed.f32, prefix.bin, tokenizer.json   and   pip install tokenizers
The NPU runs the 30 transformer layers for ONE token per call; Python does the embedding lookup, RoPE tables, KV cache and sampling.
"""
import argparse, os, struct, time
import numpy as np
from tokenizers import Tokenizer
from cubie_npu import Model

ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="."); ap.add_argument("--temp", type=float, default=0.7); ap.add_argument("--max-new", type=int, default=120)
a = ap.parse_args(); D = a.dir
tok = Tokenizer.from_file(os.path.join(D, "tokenizer.json"))
m = Model(os.path.join(D, "smollm2_135m_c128_int16.nb"))
X, COS, SIN, MASK, KT, VT = m.inputs[:6]            # quantisation (fixed-point positions etc.) comes from the model itself
C = MASK.shape[-1]; HD = 64; DM = 576; L3 = KT.shape[1]; CTX = C - 1
E = np.memmap(os.path.join(D, "embed.f32"), dtype=np.float32, mode="r").reshape(-1, DM)
INV = 1.0 / (100000 ** (np.arange(0, HD, 2) / HD)); EOS = 2


def q(x, t):                                          # float -> the int16 fixed-point format of tensor t
    return np.clip(np.rint(x * 2.0 ** t.fixed_point_pos), -32768, 32767).astype(np.int16)


pk = np.zeros(KT.shape, np.int16); pv = np.zeros(VT.shape, np.int16)
with open(os.path.join(D, "prefix.bin"), "rb") as f:   # KV cache of the fixed 14-token chat prefix, computed offline in float
    P = struct.unpack("<i", f.read(4))[0]
    kf = np.frombuffer(f.read(L3 * P * HD * 4), np.float32).reshape(L3, P, HD); vf = np.frombuffer(f.read(L3 * P * HD * 4), np.float32).reshape(L3, P, HD)
pk[0, :, :P] = q(kf, KT); pv[0, :, :P] = q(vf, VT)
state = {"pos": P}


def step(token):
    pos = state["pos"]; f = pos * INV; e = np.concatenate([f, f])
    mask = np.full((1, 1, 1, C), -30.0, np.float32); mask[..., :pos] = 0; mask[..., C - 1] = 0
    m.set_input(0, E[token][None]); m.set_input(1, np.cos(e)[None].astype(np.float32)); m.set_input(2, np.sin(e)[None].astype(np.float32))
    m.set_input(3, mask); m.set_input(4, pk); m.set_input(5, pv)     # floats are converted (int16 fixed point / fp16) automatically
    m.invoke()
    nk, nv = m.get_output(1), m.get_output(2)                       # (1, 90, 1, 64) float32
    pk[0, :, pos] = q(nk[0, :, 0], KT); pv[0, :, pos] = q(nv[0, :, 0], VT); state["pos"] = pos + 1
    return m.get_output(0)[0]


def sample(logits, past):
    l = logits.copy()
    for g in set(past): l[g] = l[g] / 1.1 if l[g] > 0 else l[g] * 1.1          # repetition penalty
    if a.temp <= 0: return int(l.argmax())
    top = np.argpartition(-l, 40)[:40]; p = np.exp((l[top] - l[top].max()) / a.temp); p /= p.sum()
    return int(np.random.choice(top, p=p))


print("ready - context %d tokens (%d used by the fixed system prompt). /reset to start over, /quit to leave." % (CTX, P))
turn = 0
while True:
    try: u = input("you> ").strip()
    except (EOFError, KeyboardInterrupt): break
    if u == "/quit": break
    if u == "/reset": state["pos"] = P; pk[0, :, P:] = 0; pv[0, :, P:] = 0; turn = 0; print("[reset]"); continue
    if not u: continue
    ids = tok.encode((u if turn == 0 else "<|im_end|>\n<|im_start|>user\n" + u) + "<|im_end|>\n<|im_start|>assistant\n").ids
    if state["pos"] + len(ids) + 2 >= CTX: print("[context full - resetting]"); state["pos"] = P; pk[0, :, P:] = 0; pv[0, :, P:] = 0; turn = 0; ids = tok.encode(u + "<|im_end|>\n<|im_start|>assistant\n").ids
    for t in ids: logits = step(t)
    out = []; shown = ""; t0 = time.time(); print("bot> ", end="", flush=True)
    while len(out) < a.max_new and state["pos"] < CTX - 1:
        nxt = sample(logits, out)
        if nxt == EOS: break
        out.append(nxt); txt = tok.decode(out)
        if not txt.endswith("�"): print(txt[len(shown):], end="", flush=True); shown = txt
        logits = step(nxt)
    print("\n   [%d tokens, %.1f tok/s, context %d/%d]" % (len(out), len(out) / max(time.time() - t0, 1e-9), state["pos"], CTX)); turn += 1
