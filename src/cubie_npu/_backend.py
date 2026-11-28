"""Finds the user's VIPLite SDK, compiles the small C shim against it (once, cached) and loads it with ctypes.

cubie-npu ships no proprietary code. The Allwinner VIPLite header (vip_lite.h) and runtime libraries (libVIPhal.so,
libNBGlinker.so) must already be on your machine - they come with the Allwinner NPU SDK / AWNPU model zoo, the same files
you need to convert models. See README.md for where to point cubie-npu at them.
"""
import ctypes, glob, hashlib, os, shutil, subprocess, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SHIM_SRC = os.path.join(_HERE, "_shim.cpp")


class NPUError(RuntimeError):
    """Raised for NPU / SDK problems (missing SDK, driver not loaded, bad model, failed run)."""


class CnpuTensor(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char * 256), ("data_format", ctypes.c_int), ("quant_format", ctypes.c_int),
                ("ndims", ctypes.c_int), ("sizes", ctypes.c_uint * 6), ("fixed_point_pos", ctypes.c_int),
                ("scale", ctypes.c_float), ("zero_point", ctypes.c_int), ("bytes", ctypes.c_uint)]


def _cache_dir():
    d = os.environ.get("CUBIE_NPU_CACHE") or os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "cubie_npu")
    os.makedirs(d, exist_ok=True)
    return d


def sdk_dir():
    """Where `cubie-npu install-sdk` puts the files it extracts."""
    return os.path.join(_cache_dir(), "sdk")


def find_sdk():
    """Return (include_dir, lib_dir) of the VIPLite SDK, or raise NPUError with instructions.

    Search order: CUBIE_NPU_SDK_INCLUDE / CUBIE_NPU_SDK_LIB (or VIPLITE_INCLUDE / VIPLITE_LIB), then the usual places
    (an AWNPU model zoo checkout in $HOME, ~/awnn_inference/vendor, /usr/include + /usr/lib)."""
    inc = os.environ.get("CUBIE_NPU_SDK_INCLUDE") or os.environ.get("VIPLITE_INCLUDE")
    lib = os.environ.get("CUBIE_NPU_SDK_LIB") or os.environ.get("VIPLITE_LIB")
    cand = []
    if inc and lib:
        cand.append((inc, lib))
    home = os.path.expanduser("~")
    for zoo in sorted(glob.glob(os.path.join(home, "awnpu_model_zoo*", "common", "npuruntime"))) + sorted(glob.glob(os.path.join(home, "*", "common", "npuruntime"))):
        for l in ("lib_linux_aarch64/A733", "lib_linux_aarch64"):
            cand.append((os.path.join(zoo, "include"), os.path.join(zoo, l)))
    cand.append((os.path.join(home, "awnn_inference", "vendor", "include"), os.path.join(home, "awnn_inference", "vendor", "lib")))
    cand.append((os.path.join(home, "awnn_inference", "include"), os.path.join(home, "awnn_inference", "install", "lib")))
    cand.append(("/usr/include", "/usr/lib"))
    cand.append(("/usr/local/include", "/usr/local/lib"))
    cand.append((os.path.join(sdk_dir(), "include"), os.path.join(sdk_dir(), "lib")))      # filled by `cubie-npu install-sdk`; last, so files already on the board win
    for i, l in cand:
        if os.path.isfile(os.path.join(i, "vip_lite.h")) and os.path.isfile(os.path.join(l, "libNBGlinker.so")) and os.path.isfile(os.path.join(l, "libVIPhal.so")):
            return i, l
    raise NPUError(
        "VIPLite SDK not found. cubie-npu needs the Allwinner header vip_lite.h and the libraries libNBGlinker.so + libVIPhal.so "
        "(from the AWNPU model zoo: common/npuruntime/include and common/npuruntime/lib_linux_aarch64/A733). "
        "Run `cubie-npu install-sdk` to fetch them from Radxa's model zoo, or point to your own copy with "
        "CUBIE_NPU_SDK_INCLUDE=<dir with vip_lite.h> and CUBIE_NPU_SDK_LIB=<dir with the .so files>.")


def build_shim(force=False, verbose=False):
    """Compile _shim.cpp against the SDK. Result is cached in ~/.cache/cubie_npu. Returns the .so path."""
    inc, lib = find_sdk()
    src = open(_SHIM_SRC, "rb").read()
    tag = hashlib.sha1(src + inc.encode() + lib.encode()).hexdigest()[:12]
    out = os.path.join(_cache_dir(), "libcubie_npu_shim-%s.so" % tag)
    if os.path.exists(out) and not force:
        return out
    cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("c++")
    if not cxx:
        raise NPUError("no C++ compiler found (install g++: sudo apt install g++), cubie-npu compiles a small shim on first use")
    cmd = [cxx, "-O2", "-std=c++14", "-shared", "-fPIC", _SHIM_SRC, "-I", inc, "-L", lib, "-lNBGlinker", "-lVIPhal",
           "-Wl,--disable-new-dtags", "-Wl,-rpath," + lib, "-o", out]
    if verbose:
        print(" ".join(cmd), file=sys.stderr)
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if r.returncode != 0:
        raise NPUError("compiling the shim failed:\n" + r.stdout)
    return out


_lib = None


def lib():
    """The loaded shim library (builds it on first call)."""
    global _lib
    if _lib is not None:
        return _lib
    _, sdk_lib = find_sdk()
    for name in ("libVIPhal.so", "libNBGlinker.so"):          # load the vendor libraries explicitly, in dependency order
        try:
            ctypes.CDLL(os.path.join(sdk_lib, name), mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            raise NPUError("cannot load %s from %s: %s" % (name, sdk_lib, e))
    L = ctypes.CDLL(build_shim())
    L.cnpu_last_error.restype = ctypes.c_char_p
    L.cnpu_driver_version.restype = ctypes.c_uint
    L.cnpu_load.restype = ctypes.c_void_p; L.cnpu_load.argtypes = [ctypes.c_char_p]
    L.cnpu_counts.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    L.cnpu_tensor_info.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(CnpuTensor)]
    L.cnpu_set_input.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
    L.cnpu_run.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
    L.cnpu_get_output.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
    L.cnpu_free.argtypes = [ctypes.c_void_p]
    _lib = L
    return L


def last_error():
    return (lib().cnpu_last_error() or b"").decode(errors="replace")
