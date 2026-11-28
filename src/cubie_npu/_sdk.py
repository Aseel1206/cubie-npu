"""`cubie-npu install-sdk`: fetch the Allwinner VIPLite header and libraries from Radxa's public model-zoo download.

We do not bundle or redistribute these files (they are Allwinner's). This downloads Radxa's own tarball to your machine, the same
one their docs tell you to download, and keeps only four files from it: vip_lite.h, vip_lite_common.h, libVIPhal.so and libNBGlinker.so
(A733, aarch64). The tarball is streamed, so the 200 MB archive is never stored.
"""
import os, shutil, sys, tarfile, tempfile, urllib.request
from . import _backend

DEFAULT_URL = "https://dl.radxa.com/cubie/allwinner-model-zoo.tar.gz"
_WANT = {"include/vip_lite.h": "include", "include/vip_lite_common.h": "include",
         "lib_linux_aarch64/A733/libVIPhal.so": "lib", "lib_linux_aarch64/A733/libNBGlinker.so": "lib"}
NOTICE = """cubie-npu needs the Allwinner VIPLite header and runtime libraries (vip_lite.h, libVIPhal.so, libNBGlinker.so).
This will download Radxa's model-zoo archive (about 200 MB) from:
    %s
and keep only those four files, in %s.
They are Allwinner's proprietary files; cubie-npu does not redistribute them, your machine fetches them from Radxa.
The libraries must match your board's NPU kernel driver. If the board image already ships them (check `cubie-npu doctor`),
use those instead: this only fills in what is missing."""


def _match(name):
    n = name.replace("\\", "/")
    if "/common/npuruntime/" not in "/" + n:
        return None
    rest = n.split("common/npuruntime/", 1)[1]
    return rest if rest in _WANT else None


def _extract(tf, dest):
    got = set()
    for m in tf:
        rest = _match(m.name) if m.isfile() else None
        if rest:
            sub = os.path.join(dest, _WANT[rest]); os.makedirs(sub, exist_ok=True)
            with tf.extractfile(m) as src, open(os.path.join(sub, os.path.basename(rest)), "wb") as out:
                shutil.copyfileobj(src, out)
            got.add(rest)
            if len(got) == len(_WANT):
                break
    return got


def install(source=None, dest=None, assume_yes=False, out=sys.stderr):
    """Install the SDK files. `source` is a URL, a .tar.gz path, or an unpacked model-zoo folder (default: Radxa's download).
    Returns the (include_dir, lib_dir) that were filled."""
    dest = dest or _backend.sdk_dir(); source = source or DEFAULT_URL
    if not assume_yes:
        print(NOTICE % (source, dest), file=out)
        if not sys.stdin.isatty():
            raise _backend.NPUError("not a terminal: re-run with --yes to accept the notice above")
        if input("Continue? [y/N] ").strip().lower() not in ("y", "yes"):
            raise _backend.NPUError("cancelled")
    os.makedirs(dest, exist_ok=True)
    if os.path.isdir(source):
        got = set()
        for root, _, files in os.walk(source):
            for f in files:
                rest = _match(os.path.join(root, f))
                if rest:
                    sub = os.path.join(dest, _WANT[rest]); os.makedirs(sub, exist_ok=True); shutil.copy(os.path.join(root, f), os.path.join(sub, f)); got.add(rest)
    elif source.startswith(("http://", "https://")):
        from . import __version__
        r = urllib.request.urlopen(urllib.request.Request(source, headers={"User-Agent": "cubie-npu/" + __version__}), timeout=60)
        print("downloading (streamed, only 4 files are kept) ...", file=out)
        with tarfile.open(fileobj=r, mode="r|gz") as tf:
            got = _extract(tf, dest)
    else:
        with tarfile.open(source, "r:*") as tf:
            got = _extract(tf, dest)
    missing = sorted(set(_WANT) - got)
    if missing:
        raise _backend.NPUError("these files were not found in %s: %s (the archive layout may have changed; use --from with the model zoo from the Allwinner portal)" % (source, ", ".join(missing)))
    for f in os.listdir(os.path.join(dest, "lib")):
        os.chmod(os.path.join(dest, "lib", f), 0o755)
    print("installed into", dest, file=out)
    return os.path.join(dest, "include"), os.path.join(dest, "lib")
