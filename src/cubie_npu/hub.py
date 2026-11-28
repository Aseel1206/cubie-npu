"""Model hub: download converted .nb models (and the small files they need) into a local cache, with checksums.

    from cubie_npu import hub
    files = hub.pull("mediapipe-hands")          # {'palm': '/home/you/.cache/cubie_npu/models/...nb', 'landmark': ...}
    hub.list_models()

Files come from the GitHub repos/releases of the Aseel1206/radxa-cubie-a7a-npu-* projects (and, for tokenizer / label files,
from Hugging Face). The cache is ~/.cache/cubie_npu/models, or $CUBIE_NPU_CACHE/models. Set CUBIE_NPU_OFFLINE=1 to never touch the network.
"""
import hashlib, json, os, sys, time, urllib.request


def cache_dir():
    base = os.environ.get("CUBIE_NPU_CACHE") or os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "cubie_npu")
    return os.path.join(base, "models")


def registry():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "registry.json")
    with open(p) as f:
        return json.load(f)


def resolve(name):
    """Model name or task alias ('hand-landmarks', 'image-classification', ...) -> (canonical name, registry entry)."""
    reg = registry(); name = reg["aliases"].get(name, name)
    if name not in reg["models"]:
        raise KeyError("unknown model %r. Available: %s" % (name, ", ".join(sorted(reg["models"]))))
    return name, reg["models"][name]


def _path(model, fname):
    return os.path.join(cache_dir(), model, fname)


def _have(model, f):
    p = _path(model, f["name"])
    return os.path.isfile(p) and (not f.get("size") or os.path.getsize(p) == f["size"])


def _download(url, dest, sha256, size, show):
    if os.environ.get("CUBIE_NPU_OFFLINE"):
        raise RuntimeError("%s is not in the cache and CUBIE_NPU_OFFLINE is set" % os.path.basename(dest))
    from . import __version__
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"; h = hashlib.sha256(); done = 0; t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "cubie-npu/" + __version__})
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as out:
            total = int(r.headers.get("Content-Length") or size or 0)
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                out.write(b); h.update(b); done += len(b)
                if show and total:
                    sys.stderr.write("\r  %s  %5.1f%%  %.1f / %.1f MB  %.1f MB/s " % (os.path.basename(dest), 100.0 * done / total, done / 1e6, total / 1e6, done / 1e6 / max(time.time() - t0, 1e-6)))
        if show:
            sys.stderr.write("\n")
        if sha256 and h.hexdigest() != sha256:
            raise RuntimeError("checksum mismatch for %s (got %s, expected %s)" % (os.path.basename(dest), h.hexdigest(), sha256))
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def pull(name, variant=None, progress=None, force=False):
    """Make sure every file of `name` (for the chosen variant) is in the cache; return {role or filename: local path}.

    variant: 'int8' (fast, default where it works) or 'int16' (more accurate, slower); see list_models()."""
    name, e = resolve(name)
    variant = variant or e["default"]
    if variant not in e["variants"]:
        raise KeyError("%s has variants %s, not %r" % (name, sorted(e["variants"]), variant))
    roles = e["variants"][variant]; wanted = set(roles.values())
    every_nb = {v for r in e["variants"].values() for v in r.values()}
    show = sys.stderr.isatty() if progress is None else progress
    paths = {}
    for f in e["files"]:
        if f["name"] in every_nb and f["name"] not in wanted:
            continue
        if force or not _have(name, f):
            _download(f["url"], _path(name, f["name"]), f.get("sha256"), f.get("size"), show)
        paths[f["name"]] = _path(name, f["name"])
    out = {role: paths[fn] for role, fn in roles.items()}
    out.update({fn: p for fn, p in paths.items() if fn not in wanted})
    return out


def list_models():
    """[{name, task, description, variants, default, size_mb, cached}]"""
    rows = []
    for name, e in registry()["models"].items():
        rows.append({"name": name, "task": e["task"], "description": e["description"], "variants": sorted(e["variants"]), "default": e["default"],
                     "size_mb": round(sum(f.get("size") or 0 for f in e["files"] if not (f["name"] in {v for r in e["variants"].values() for v in r.values()} and f["name"] not in set(e["variants"][e["default"]].values()))) / 1e6, 1),
                     "cached": all(_have(name, f) for f in e["files"] if f["name"] in set(e["variants"][e["default"]].values()))})
    return rows
