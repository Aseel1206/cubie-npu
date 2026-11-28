"""`cubie-npu setup-gpu`: look at what is installed for the Cubie A7A's GPU and print the exact steps that are still missing.

The helpers in cubie_npu.gpu use OpenCL (libPVROCL.so). Vulkan (libVK_IMG.so) is checked and reported too, for your own Vulkan code,
but cubie_npu.gpu does not use it.
"""
import glob, grp, os, shutil, sys

_VK_ICD_DIRS = ["/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d", "/usr/local/share/vulkan/icd.d"]


def _in_groups(names):
    try:
        mine = {grp.getgrgid(g).gr_name for g in os.getgroups()}
    except Exception:
        return {}
    return {n: n in mine for n in names}


def _find(patterns):
    out = []
    for pat in patterns:
        out += glob.glob(pat)
    return sorted(set(out))


def _vulkan():
    """Vulkan is not used by cubie_npu.gpu (OpenCL is), but it is part of 'is my GPU set up', so check it for real with vulkaninfo."""
    import subprocess
    vk_loader = _find(["/usr/lib/aarch64-linux-gnu/libvulkan.so*", "/usr/lib/libvulkan.so*"])
    vk_pvr = _find(["/usr/lib/libVK_IMG*", "/usr/lib/aarch64-linux-gnu/libVK_IMG*"])
    icds = [p for d in _VK_ICD_DIRS for p in glob.glob(d + "/*.json")]
    if not vk_loader or not shutil.which("vulkaninfo"):
        return [("info", "Vulkan tools missing", "", "sudo apt install libvulkan1 vulkan-tools")]
    try:
        out = subprocess.run(["vulkaninfo", "--summary"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True, timeout=30).stdout
    except Exception as e:
        return [("info", "vulkaninfo failed", str(e), "")]
    devs = [l.split("=", 1)[1].strip() for l in out.splitlines() if l.strip().startswith("deviceName")]
    ver = [l.split("(", 1)[-1].rstrip(") ") for l in out.splitlines() if l.strip().startswith("apiVersion")]
    if any("powervr" in d.lower() for d in devs):
        return [("ok", "Vulkan sees the GPU", "; ".join(devs) + (" (Vulkan %s)" % ver[0] if ver else ""), "")]
    fix = ("echo '{\"file_format_version\":\"1.0.0\",\"ICD\":{\"library_path\":\"%s\",\"api_version\":\"1.3.277\"}}' | sudo tee /usr/share/vulkan/icd.d/img_icd.json\n"
           "vulkaninfo --summary | grep deviceName" % vk_pvr[0]) if vk_pvr and not any("img" in os.path.basename(i).lower() for i in icds) else \
          ("PowerVR Vulkan driver (libVK_IMG.so) not found: it comes with the Radxa image's GPU packages; update the image or install them" if not vk_pvr else
           "VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/img_icd.json vulkaninfo --summary   # if this lists the GPU, export that variable")
    return [("warn", "Vulkan does not list the PowerVR GPU", "devices seen: " + (", ".join(devs) or "none"), fix)]


def diagnose():
    """-> list of (status, title, detail, fix) with status 'ok' | 'warn' | 'fail' | 'info'; fix is a shell snippet or ''."""
    from . import _setup
    r = []
    nodes = _find(["/dev/dri/renderD*"]); pvr = _find(["/dev/pvr*", "/dev/dri/card*"])
    if nodes:
        r.append(("ok", "GPU device node", ", ".join(nodes), ""))
        g = _in_groups(["render", "video"])
        if g and not any(g.values()) and not os.access(nodes[0], os.R_OK | os.W_OK):
            r.append(("fail", "no permission on " + nodes[0], "your user is in neither the 'render' nor the 'video' group",
                      "sudo usermod -aG render,video $USER   # then log out and back in"))
    else:
        r.append(("fail", "no /dev/dri/renderD* device", "the GPU kernel driver (pvrsrvkm) is not loaded or not enabled in this image",
                  "lsmod | grep -i pvr    # should list pvrsrvkm; if empty, update the Radxa image/kernel (sudo rsetup -> System -> System Update)"))
    lib = _setup.find_pvr_library()
    if lib:
        r.append(("ok", "PowerVR OpenCL driver", lib, ""))
    else:
        r.append(("fail", "libPVROCL.so not found", "the PowerVR user-space OpenCL driver is missing",
                  "sudo apt update && apt-cache search powervr gpu | head   # install the Radxa GPU/PowerVR package for your image\n"
                  "find / -name 'libPVROCL.so*' 2>/dev/null   # then: export CUBIE_NPU_OCL_LIB=/path/to/libPVROCL.so"))
    loader = _find(["/usr/lib/aarch64-linux-gnu/libOpenCL.so*", "/usr/lib/libOpenCL.so*", "/usr/lib64/libOpenCL.so*"])
    if loader:
        r.append(("ok", "OpenCL ICD loader", loader[0], ""))
    else:
        r.append(("fail", "OpenCL ICD loader (libOpenCL.so.1) missing", "", "sudo apt install ocl-icd-libopencl1 clinfo"))
    if os.environ.get("OCL_ICD_VENDORS"):
        r.append(("info", "OCL_ICD_VENDORS is set", os.environ["OCL_ICD_VENDORS"] + "  (cubie-npu will not touch it; make sure it contains an .icd file naming libPVROCL.so)", ""))
    else:
        r.append(("info", "ICD registration", "automatic: cubie-npu writes ~/.cache/cubie_npu/ocl_icd/powervr.icd and sets OCL_ICD_VENDORS for its own process.\n"
                  "For other programs (clinfo, your own scripts):  mkdir -p ~/ocl_icd && echo %s > ~/ocl_icd/powervr.icd && export OCL_ICD_VENDORS=~/ocl_icd" % (lib or "/usr/lib/libPVROCL.so"), ""))
    try:
        import pyopencl  # noqa: F401
        r.append(("ok", "pyopencl", pyopencl.VERSION_TEXT, ""))
    except ImportError:
        r.append(("fail", "pyopencl not installed", "", 'pip install "cubie-npu[gpu]"    # needs: sudo apt install build-essential python3-dev ocl-icd-opencl-dev'))
    if lib:
        try:
            from . import info
            i = info()
            r.append(("ok", "GPU visible to OpenCL", "%s, %s compute units, %s MB" % (i.get("name"), i.get("compute_units"), i.get("global_mem_mb")), ""))
        except Exception as e:
            r.append(("fail", "OpenCL cannot open the GPU", str(e).splitlines()[0] if str(e) else type(e).__name__,
                      "clinfo | head -20    # with OCL_ICD_VENDORS set as above; if it lists 0 platforms the kernel driver is not running"))
    r += _vulkan()
    return r


def report(file=None):
    f = file or sys.stdout; marks = {"ok": "[ ok ]", "warn": "[warn]", "fail": "[FAIL]", "info": "[ .. ]"}; bad = 0
    for st, title, detail, fix in diagnose():
        print("%s %s%s" % (marks[st], title, ("  -  " + detail.replace("\n", "\n         ")) if detail else ""), file=f)
        if fix and st in ("fail", "info", "warn"):
            print("       how:  " + fix.replace("\n", "\n             "), file=f)
        bad += st == "fail"
    print("\n" + ("Everything needed for cubie_npu.gpu is in place." if not bad else "%d thing(s) to fix above; run this again afterwards." % bad), file=f)
    return bad
