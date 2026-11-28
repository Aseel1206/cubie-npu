import argparse, json, os, shutil, sys


def _hub_cmd(a):
    from . import hub, pipeline
    if a.cmd == "models":
        print("%-26s %-20s %-10s %8s  %s" % ("model", "task", "variants", "download", "cached"))
        for m in hub.list_models():
            print("%-26s %-20s %-10s %6.1f MB  %s" % (m["name"], m["task"], "/".join(m["variants"]), m["size_mb"], "yes" if m["cached"] else "-"))
        print("\ncache:", hub.cache_dir())
    elif a.cmd == "pull":
        for k, p in hub.pull(a.model, a.variant, progress=True).items(): print("%-18s %s" % (k, p))
    elif a.cmd == "setup-gpu":
        from .gpu import _doctor
        return 1 if _doctor.report() else 0
    elif a.cmd == "chat":
        bot = pipeline("text-generation")
        print("SmolLM2-135M on the NPU - context %d tokens. /reset starts over, /quit leaves." % bot.CTX)
        while True:
            try: u = input("you> ").strip()
            except (EOFError, KeyboardInterrupt): print(); break
            if u == "/quit": break
            if u == "/reset": bot.reset(); print("[reset]"); continue
            if not u: continue
            print("bot> ", end="", flush=True)
            for piece in bot(u, a.max_new, a.temperature, stream=True): print(piece, end="", flush=True)
            print("\n   [%.1f tok/s, context %d/%d]" % (bot.last_tokens_per_s, bot.pos, bot.CTX))
    else:
        import numpy as np
        from ._image import load_image, _cv2
        cv2 = _cv2(); task = {"classify": "image-classification", "depth": "depth-estimation", "hands": "hand-landmarks", "faces": "face-detection"}[a.task]
        pipe = pipeline(task, variant=a.variant); im = load_image(a.image); r = pipe(im)
        if a.task == "classify":
            for x in r: print("%5.1f%%  %s" % (x["score"] * 100, x["label"]))
        else:
            from . import _viz
            out = _viz.draw(task, im, r)
            if a.task == "depth": print("depth range %.3f .. %.3f" % (r["depth"].min(), r["depth"].max()))
            else: print("%d found" % len(r))
            if a.out: cv2.imwrite(a.out, out[..., ::-1]); print("wrote", a.out)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cubie-npu", description="Tools for running .nb models on the Cubie A7A NPU")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("info", help="print the inputs/outputs of a .nb"); s.add_argument("model")
    b = sub.add_parser("bench", help="time a .nb on the NPU (random input)"); b.add_argument("model"); b.add_argument("-n", type=int, default=100)
    sub.add_parser("models", help="list the models you can download, and which are cached")
    pl = sub.add_parser("pull", help="download a model into the cache"); pl.add_argument("model"); pl.add_argument("--variant")
    ru = sub.add_parser("run", help="run a task on an image: classify | depth | hands | faces"); ru.add_argument("task", choices=["classify", "depth", "hands", "faces"])
    ru.add_argument("image"); ru.add_argument("-o", "--out", help="save the annotated image / depth map here"); ru.add_argument("--variant")
    ch = sub.add_parser("chat", help="chat with SmolLM2-135M on the NPU"); ch.add_argument("--temperature", type=float, default=0.7); ch.add_argument("--max-new", type=int, default=120)
    ins = sub.add_parser("install-sdk", help="fetch the Allwinner VIPLite header + libraries from Radxa's model zoo (only if not already on the board)")
    ins.add_argument("--from", dest="source", help="URL, model-zoo .tar.gz, or unpacked model-zoo folder (default: Radxa's download)"); ins.add_argument("-y", "--yes", action="store_true")
    sub.add_parser("setup-gpu", help="check the GPU/OpenCL/Vulkan setup and print the steps that are missing")
    sub.add_parser("gpu-info", help="show the GPU (OpenCL) the gpu helpers would use")
    gb = sub.add_parser("gpu-bench", help="GPU image ops vs OpenCV on the CPU"); gb.add_argument("-n", type=int, default=30)
    d = sub.add_parser("doctor", help="check the SDK, compiler, driver and (optionally) a model")
    d.add_argument("model", nargs="?")
    a = ap.parse_args(argv)
    if a.cmd == "install-sdk":
        from . import _sdk, _backend
        try:
            _backend.find_sdk(); print("an SDK is already available, nothing to install:", *_backend.find_sdk()); return 0
        except _backend.NPUError:
            pass
        try:
            _sdk.install(a.source, assume_yes=a.yes); print("now run: cubie-npu doctor"); return 0
        except (_backend.NPUError, OSError) as e:
            print("error:", e, file=sys.stderr); return 1
    if a.cmd in ("models", "pull", "run", "chat", "setup-gpu"):
        return _hub_cmd(a)
    from . import Model, NPUError, find_sdk, build_shim, driver_version, __version__
    try:
        if a.cmd == "info":
            m = Model(a.model)
            for kind, ts in (("input", m.inputs), ("output", m.outputs)):
                for t in ts:
                    q = t.quant if t.quant == "none" else "%s (%s)" % (t.quant, "fl=%d" % t.fixed_point_pos if t.quant == "dynamic_fixed_point" else "scale=%g zp=%d" % (t.scale, t.zero_point))
                    print("%-6s %d  %-28s shape %-20s dtype %-8s quant %s  (%d bytes)" % (kind, t.index, t.name, t.shape, t.dtype, q, t.nbytes))
        elif a.cmd == "gpu-info":
            from . import gpu
            for k, v in gpu.info().items(): print("%-16s %s" % (k, v))
        elif a.cmd == "gpu-bench":
            from . import gpu
            from .gpu import _bench; _bench.run(a.n)
        elif a.cmd == "bench":
            m = Model(a.model); r = m.benchmark(a.n)
            print("%s: %d runs  mean %.2f ms  p50 %.2f  p95 %.2f  min %.2f  max %.2f  (%.1f FPS)" % (os.path.basename(a.model), r["runs"], r["mean_ms"], r["p50_ms"], r["p95_ms"], r["min_ms"], r["max_ms"], r["fps"]))
        else:
            print("cubie-npu", __version__)
            print("python      ", sys.version.split()[0])
            print("g++         ", shutil.which("g++") or "NOT FOUND (needed once, to compile the shim)")
            print("/dev/vipcore", "present" if os.path.exists("/dev/vipcore") else "missing (NPU driver not loaded?)")
            inc, lib = find_sdk(); print("SDK include ", inc); print("SDK lib     ", lib)
            print("shim        ", build_shim()); print("driver      ", "0x%08x" % driver_version())
            if a.model:
                m = Model(a.model); print("model       ", m); r = m.benchmark(20); print("20 runs      mean %.2f ms" % r["mean_ms"])
            print("OK")
    except NPUError as e:
        print("error:", e, file=sys.stderr); return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
