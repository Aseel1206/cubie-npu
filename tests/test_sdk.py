import io, os, tarfile
import pytest
from cubie_npu import _sdk, _backend

FILES = {"common/npuruntime/include/vip_lite.h": b"h", "common/npuruntime/include/vip_lite_common.h": b"hc",
         "common/npuruntime/lib_linux_aarch64/A733/libVIPhal.so": b"A733hal", "common/npuruntime/lib_linux_aarch64/A733/libNBGlinker.so": b"A733link",
         "common/npuruntime/lib_linux_aarch64/T527/libVIPhal.so": b"WRONG", "common/npuruntime/include_85x/vip_lite.h": b"WRONG", "other/readme": b"x"}


def _tar(path, files, prefix="zoo-v1/"):
    with tarfile.open(path, "w:gz") as t:
        for n, d in files.items():
            i = tarfile.TarInfo(prefix + n); i.size = len(d); t.addfile(i, io.BytesIO(d))


def test_install_from_tarball_picks_a733_only(tmp_path):
    _tar(tmp_path / "z.tgz", FILES)
    inc, lib = _sdk.install(str(tmp_path / "z.tgz"), dest=str(tmp_path / "d"), assume_yes=True)
    assert open(os.path.join(lib, "libVIPhal.so"), "rb").read() == b"A733hal" and open(os.path.join(inc, "vip_lite.h"), "rb").read() == b"h"
    assert sorted(os.listdir(lib)) == ["libNBGlinker.so", "libVIPhal.so"] and os.access(os.path.join(lib, "libVIPhal.so"), os.X_OK)


def test_install_from_unpacked_folder(tmp_path):
    for n, d in FILES.items():
        p = tmp_path / "zoo" / n; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(d)
    _, lib = _sdk.install(str(tmp_path / "zoo"), dest=str(tmp_path / "d"), assume_yes=True)
    assert open(os.path.join(lib, "libNBGlinker.so"), "rb").read() == b"A733link"


def test_missing_files_reported(tmp_path):
    _tar(tmp_path / "z.tgz", {k: v for k, v in FILES.items() if "NBGlinker" not in k})
    with pytest.raises(_backend.NPUError, match="NBGlinker"):
        _sdk.install(str(tmp_path / "z.tgz"), dest=str(tmp_path / "d"), assume_yes=True)


def test_needs_consent_without_tty(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    with pytest.raises(_backend.NPUError, match="--yes"):
        _sdk.install(str(tmp_path / "z.tgz"), dest=str(tmp_path / "d"), out=io.StringIO())


def test_find_sdk_uses_installed_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CUBIE_NPU_CACHE", str(tmp_path / "c")); monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for k in ("CUBIE_NPU_SDK_INCLUDE", "CUBIE_NPU_SDK_LIB", "VIPLITE_INCLUDE", "VIPLITE_LIB"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(os.path, "isfile", (lambda f, _r=os.path.isfile: False if f.startswith("/usr/") else _r(f)))
    with pytest.raises(_backend.NPUError, match="install-sdk"):
        _backend.find_sdk()
    _tar(tmp_path / "z.tgz", FILES); _sdk.install(str(tmp_path / "z.tgz"), assume_yes=True)
    assert _backend.find_sdk()[0].startswith(str(tmp_path / "c"))
