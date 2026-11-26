"""Makes the PowerVR GPU of the Cubie A7A visible to OpenCL.

On the stock Radxa image the registered OpenCL driver is Mesa's, which does not drive this GPU, so a plain
`pyopencl.get_platforms()` finds nothing useful. The PowerVR driver is installed (libPVROCL.so) but not registered.
We register it by pointing the OpenCL ICD loader at a one-line .icd file in our cache folder, before OpenCL is first used.
If you set OCL_ICD_VENDORS yourself it is left alone.
"""
import os

_CANDIDATES = [
    "/usr/lib/libPVROCL.so", "/usr/lib/aarch64-linux-gnu/libPVROCL.so", "/usr/lib64/libPVROCL.so",
    "/vendor/lib64/libPVROCL.so", "/usr/local/lib/libPVROCL.so",
]
_done = False


def find_pvr_library():
    env = os.environ.get("CUBIE_NPU_OCL_LIB")
    for p in ([env] if env else []) + _CANDIDATES:
        if p and os.path.isfile(p):
            return p
    return None


def configure(cache_dir=None):
    """Register the PowerVR OpenCL driver (once). Returns the path of the library used, or None if not needed / not found."""
    global _done
    if _done:
        return None
    _done = True
    if os.environ.get("OCL_ICD_VENDORS"):          # the user knows better
        return None
    lib = find_pvr_library()
    if lib is None:
        return None
    base = cache_dir or os.environ.get("CUBIE_NPU_CACHE") or os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "cubie_npu")
    d = os.path.join(base, "ocl_icd")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "powervr.icd"), "w") as f:
        f.write(lib + "\n")
    os.environ["OCL_ICD_VENDORS"] = d
    return lib
