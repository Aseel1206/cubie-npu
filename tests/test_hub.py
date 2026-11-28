"""Hub, image loading, pipelines and GPU doctor - everything that does not need the NPU (a fake Model stands in)."""
import hashlib, http.server, json, os, threading, functools
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
import cubie_npu
from cubie_npu import hub, pipelines, _hands, _image


@pytest.fixture()
def server(tmp_path):
    root = tmp_path / "www"; root.mkdir()
    h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    h.log_message = lambda *a, **k: None
    s = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h); threading.Thread(target=s.serve_forever, daemon=True).start()
    yield root, "http://127.0.0.1:%d" % s.server_address[1]
    s.shutdown()


@pytest.fixture()
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CUBIE_NPU_CACHE", str(tmp_path / "cache")); monkeypatch.delenv("CUBIE_NPU_OFFLINE", raising=False)
    return tmp_path / "cache"


def _reg(url, data, sha=None):
    return {"models": {"toy": {"task": "image-classification", "description": "t", "default": "int8",
                               "variants": {"int8": {"nb": "a.nb"}, "int16": {"nb": "b.nb"}},
                               "files": [{"name": "a.nb", "url": url + "/a.nb", "sha256": sha or hashlib.sha256(data).hexdigest(), "size": len(data)},
                                         {"name": "b.nb", "url": url + "/b.nb", "sha256": None, "size": None},
                                         {"name": "config.json", "url": url + "/config.json", "sha256": None, "size": None}]}},
            "aliases": {"image-classification": "toy"}}


def test_registry_is_consistent():
    reg = hub.registry()
    for name, e in reg["models"].items():
        names = {f["name"] for f in e["files"]}
        assert e["default"] in e["variants"]
        for roles in e["variants"].values():
            assert set(roles.values()) <= names, name
        for f in e["files"]:
            assert f["url"].startswith("https://")
            if f["name"].endswith((".nb", ".bin", ".f32")):
                assert len(f["sha256"]) == 64 and f["size"] > 0
    assert set(reg["aliases"].values()) <= set(reg["models"])


def test_pull_downloads_verifies_and_caches(server, cache, monkeypatch):
    root, url = server; data = os.urandom(5000)
    (root / "a.nb").write_bytes(data); (root / "b.nb").write_bytes(b"x"); (root / "config.json").write_text("{}")
    monkeypatch.setattr(hub, "registry", lambda: _reg(url, data))
    p = hub.pull("toy", progress=False)
    assert open(p["nb"], "rb").read() == data and "config.json" in p and not (cache / "models" / "toy" / "b.nb").exists()
    (root / "a.nb").unlink()                                   # second pull must not need the network
    assert hub.pull("image-classification", progress=False)["nb"] == p["nb"]


def test_pull_rejects_bad_checksum_and_offline(server, cache, monkeypatch):
    root, url = server; data = b"hello" * 100
    (root / "a.nb").write_bytes(data); (root / "b.nb").write_bytes(b"x"); (root / "config.json").write_text("{}")
    monkeypatch.setattr(hub, "registry", lambda: _reg(url, data, sha="0" * 64))
    with pytest.raises(RuntimeError, match="checksum"):
        hub.pull("toy", progress=False)
    assert not list((cache / "models").rglob("*.part")) and not (cache / "models" / "toy" / "a.nb").exists()
    monkeypatch.setenv("CUBIE_NPU_OFFLINE", "1")
    with pytest.raises(RuntimeError, match="OFFLINE"):
        hub.pull("toy", progress=False)


def test_unknown_names():
    with pytest.raises(KeyError):
        hub.resolve("nope")
    with pytest.raises(KeyError):
        hub.pull("deit-tiny", "int4")


def test_load_image_sources(tmp_path, server):
    from PIL import Image
    a = np.random.randint(0, 255, (20, 30, 3), np.uint8)
    cv2.imwrite(str(tmp_path / "x.png"), a[..., ::-1])
    assert np.array_equal(_image.load_image(str(tmp_path / "x.png")), a)
    assert np.array_equal(_image.load_image(tmp_path / "x.png"), a)
    assert np.array_equal(_image.load_image((tmp_path / "x.png").read_bytes()), a)
    assert np.array_equal(_image.load_image(Image.fromarray(a)), a)
    assert np.array_equal(_image.load_image(a), a)
    assert _image.load_image(a[..., 0]).shape == (20, 30, 3)
    assert _image.load_image(a.astype(np.float32) / 255).dtype == np.uint8
    root, url = server; (root / "x.png").write_bytes((tmp_path / "x.png").read_bytes())
    assert np.array_equal(_image.load_image(url + "/x.png"), a)
    with pytest.raises(FileNotFoundError):
        _image.load_image("/no/such.png")
    with pytest.raises(ValueError):
        _image.load_image(np.zeros((3, 3, 7)))


class Fake:
    def __init__(self, fn): self.fn = fn; self.seen = []
    def __call__(self, x): self.seen.append(np.asarray(x)); return self.fn(x)
    def close(self): pass


def _bare(cls, **kw):
    o = cls.__new__(cls); o.__dict__.update(kw); return o


def test_classification_pipeline(tmp_path):
    logits = np.zeros(1000, np.float32); logits[3] = 5; logits[7] = 4
    m = Fake(lambda x: [logits])
    p = _bare(pipelines.ImageClassification, model=m, labels={str(i): "c%d" % i for i in range(1000)})
    img = np.random.randint(0, 255, (300, 500, 3), np.uint8)
    r = p(img, top_k=2)
    assert [x["label"] for x in r] == ["c3", "c7"] and r[0]["score"] > r[1]["score"] and abs(sum(x["score"] for x in p(img, 1000)) - 1) < 1e-4
    assert m.seen[0].shape == (224, 224, 3) and m.seen[0].dtype == np.uint8


def test_depth_pipeline_resizes_back():
    m = Fake(lambda x: np.random.rand(1, 252, 336).astype(np.float32))
    p = _bare(pipelines.DepthEstimation, model=m)
    r = p(np.zeros((100, 200, 3), np.uint8))
    assert r["depth"].shape == (100, 200) and r["raw"].shape == (252, 336) and r["image"].shape == (100, 200, 3) and m.seen[0].shape == (252, 336, 3)


def test_face_decode_and_pipeline():
    reg = np.zeros((1, 896, 16), np.float32); sc = np.full((1, 896, 1), -10, np.float32)
    i = 100; ax, ay = _hands.FACE_ANCHORS[i]
    reg[0, i, 2:4] = 32                                         # 32 px wide/high on the 128 input
    reg[0, i, 4:] = 0                                           # keypoints sit on the anchor
    sc[0, i, 0] = 4
    p = _bare(pipelines.FaceDetection, model=Fake(lambda x: [reg, sc]))
    img = np.zeros((256, 512, 3), np.uint8)                     # letterboxed with scale 0.25, top pad 32
    r = p(img)
    assert len(r) == 1 and r[0]["score"] > 0.98
    x0, y0, x1, y1 = r[0]["box"]; assert abs((x1 - x0) - 128) < 1e-3 and abs((y1 - y0) - 128) < 1e-3
    assert abs((x0 + x1) / 2 - ax * 128 / 0.25) < 1e-3 and abs((y0 + y1) / 2 - (ay * 128 - 32) / 0.25) < 1e-3 and r[0]["keypoints"].shape == (6, 2)


def test_hands_pipeline_maps_back_to_image():
    reg = np.zeros((1, 2016, 18), np.float32); sc = np.full((1, 2016, 1), -10, np.float32)
    i = 500; ax, ay = _hands.PALM_ANCHORS[i]
    reg[0, i, 2:4] = 40
    kp = np.zeros((7, 2), np.float32); kp[0] = (0, 30); kp[2] = (0, -30)       # wrist below, middle finger above -> rotation 0
    reg[0, i, 4:] = kp.reshape(-1); sc[0, i, 0] = 3
    lm = np.tile(np.array([112, 112, 0], np.float32), 21)[None]
    lmm = Fake(lambda x: [lm, np.array([[0.9]], np.float32), np.array([[0.5]], np.float32), lm])
    p = _bare(pipelines.HandLandmarks, palm=Fake(lambda x: [reg, sc]), landmark=lmm)
    im = np.zeros((384, 384, 3), np.uint8)
    r = p(im)
    assert len(r) == 1 and r[0]["landmarks"].shape == (21, 2) and r[0]["roi"].shape == (4, 2) and lmm.seen[0].shape == (224, 224, 3)
    cx = ax * 384; assert abs(r[0]["landmarks"][0, 0] - cx) < 1.5          # the patch centre maps to the ROI centre
    assert abs(r[0]["presence"] - 0.9) < 1e-6


def test_gpu_doctor_runs_anywhere():
    from cubie_npu.gpu import _doctor
    rows = _doctor.diagnose()
    assert rows and all(len(x) == 4 and x[0] in ("ok", "warn", "fail", "info") for x in rows)
    assert any("Vulkan" in x[1] for x in rows)


def test_cli_models(capsys):
    from cubie_npu import cli
    assert cli.main(["models"]) == 0
    out = capsys.readouterr().out
    assert "mediapipe-hands" in out and "smollm2-135m" in out
