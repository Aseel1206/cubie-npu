import argparse, json, os, shutil, sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cubie-npu", description="Tools for running .nb models on the Cubie A7A NPU")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("info", help="print the inputs/outputs of a .nb"); s.add_argument("model")
    b = sub.add_parser("bench", help="time a .nb on the NPU (random input)"); b.add_argument("model"); b.add_argument("-n", type=int, default=100)
    d = sub.add_parser("doctor", help="check the SDK, compiler, driver and (optionally) a model")
    d.add_argument("model", nargs="?")
    a = ap.parse_args(argv)
    from . import Model, NPUError, find_sdk, build_shim, driver_version, __version__
    try:
        if a.cmd == "info":
            m = Model(a.model)
            for kind, ts in (("input", m.inputs), ("output", m.outputs)):
                for t in ts:
                    q = t.quant if t.quant == "none" else "%s (%s)" % (t.quant, "fl=%d" % t.fixed_point_pos if t.quant == "dynamic_fixed_point" else "scale=%g zp=%d" % (t.scale, t.zero_point))
                    print("%-6s %d  %-28s shape %-20s dtype %-8s quant %s  (%d bytes)" % (kind, t.index, t.name, t.shape, t.dtype, q, t.nbytes))
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
