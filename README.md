# cubie-npu

[![PyPI](https://img.shields.io/pypi/v/cubie-npu)](https://pypi.org/project/cubie-npu/) [![CI](https://github.com/Aseel1206/cubie-npu/actions/workflows/ci.yml/badge.svg)](https://github.com/Aseel1206/cubie-npu/actions/workflows/ci.yml)

Run `.nb` models on the NPU of the Radxa Cubie A7A (Allwinner A733) from Python.

If you have ever looked at the NPU on this board and found that the only way to use it is C or C++, this is for you. You give it a `.nb` file and NumPy arrays, you get NumPy arrays back:

```python
import numpy as np
from cubie_npu import Model

m = Model("deit_tiny_int8pcq.nb")
print(m)                      # Model(deit_tiny_int8pcq.nb | in: input/output[0][1, 224, 224, 3] uint8/none | out: ...[1, 1000] float32/none)

logits = m(rgb_image)         # uint8 RGB 224x224 in, float32 logits out
print(m.last_run_ms)          # 14.6
```

It is a thin layer over the public VIPLite C API (the same one the vendor tools use), so there is no extra runtime and nearly no overhead (table below). It figures out shapes, data types and quantisation from the model itself, so you do not need any side files.

This is an unofficial community project. I tested it on a Cubie A7A only (Debian 11, kernel 5.15.147-a733, VIPLite driver 2.0.3.2). Other Allwinner chips that use the same VIPLite runtime might work. I have not tried.

## Download a model and run it on a picture

You do not have to find the `.nb` files yourself. `pipeline()` downloads the model on first use (checksum-verified, cached in `~/.cache/cubie_npu/models`), loads it onto the NPU, and takes a normal image: a file path, a URL, a NumPy array or a PIL image.

```bash
pip install "cubie-npu[vision]"          # adds OpenCV for reading/resizing images ("[llm]" adds tokenizers for the chat, "[all]" everything)
```

```python
from cubie_npu import pipeline

pipeline("image-classification")("cat.jpg")                  # [{'label': 'tabby, tabby cat', 'score': 0.6}, ...]
pipeline("hand-landmarks")("https://example.com/hand.jpg")   # [{'landmarks': (21, 2) px, 'score', 'presence', 'roi'}]
pipeline("face-detection")(pil_image)                        # [{'box': (x0, y0, x1, y1), 'keypoints': (6, 2), 'score'}]
d = pipeline("depth-estimation")("room.jpg")                 # d["depth"] float32 at the image size, d["image"] a colour map
chat = pipeline("text-generation"); chat("Tell me a joke")   # keeps the conversation; chat.reset() starts over
```

Images are RGB. A NumPy array is taken as RGB, so from OpenCV pass `frame[..., ::-1]`. Pass `variant="int16"` for the slower, more accurate build where there is one. The same from the shell:

```bash
cubie-npu models                         # what can be downloaded, sizes, what is cached
cubie-npu pull mediapipe-hands           # download only
cubie-npu run hands photo.jpg -o out.png # also: classify, depth, faces
cubie-npu chat                           # SmolLM2 chat in the terminal
```

| name | task | download | notes |
|---|---|---|---|
| `deit-tiny` | image-classification | 7 MB | ImageNet-1k, 14.5 ms |
| `mediapipe-hands` | hand-landmarks | 3 MB | palm detector + 21 landmarks |
| `blazeface` | face-detection | 0.2 MB | int16 only (int8 misses the face). Short-range model: the face should fill a good part of the frame (about a fifth of the width or more); small faces in a wide photo are missed |
| `depth-anything-v2-small` | depth-estimation | 35 MB | fixed 252x336 input, 255 ms |
| `smollm2-135m` | text-generation | 337 MB | 128-token context, about 21 tokens/s |

The files live in my per-model repos (conversion details are in each README), so those repos are the source of truth. `CUBIE_NPU_OFFLINE=1` stops it from using the network; `CUBIE_NPU_CACHE` moves the cache. Your own `.nb` files still work with `Model("x.nb")` as before.

## What you need first

cubie-npu does not ship any vendor files, because I can't. It needs the Allwinner VIPLite header `vip_lite.h` and the two libraries `libNBGlinker.so` and `libVIPhal.so`. **Check what you already have first**: it looks in the environment variables below, an `awnpu_model_zoo*` folder in your home directory (the files are in `common/npuruntime`), `~/awnn_inference` (Radxa's wrapper, which was on my board), then `/usr/include` + `/usr/lib`. `cubie-npu doctor` tells you what it found.

If it finds nothing:

```bash
cubie-npu install-sdk          # asks first, then fetches the four files it needs from Radxa's model-zoo download
```

That streams Radxa's public archive (`https://dl.radxa.com/cubie/allwinner-model-zoo.tar.gz`, about 200 MB, never stored) and keeps only `vip_lite.h`, `vip_lite_common.h` and the A733 `libVIPhal.so` / `libNBGlinker.so` in `~/.cache/cubie_npu/sdk`. They are Allwinner's files; your board downloads them from Radxa, cubie-npu does not redistribute them, and the command shows that notice and waits for a yes (`-y` skips it). `--from model-zoo.tar.gz` or `--from unpacked_folder/` installs from a copy you already have, for example one from the Allwinner portal.

One caution: the runtime libraries should match your board's NPU kernel driver. The ones in that mirror are model zoo v0.9.0 (the newer v1.0.0 on the Radxa docs page is only on the Allwinner portal). On my board (kernel driver 2.0.3.2) a clean install with only these downloaded files compiled the shim and ran the palm detector at 3.2 ms. I have not tried them against other board images. If the NPU misbehaves after `install-sdk`, prefer the libraries that came with your image.

On first use, cubie-npu compiles a small C++ file (`_shim.cpp`, about 160 lines) against those files, which needs `g++` on the board (`sudo apt install g++`). The result is cached in `~/.cache/cubie_npu/`, so this happens once. To point at your own copy instead:

```bash
export CUBIE_NPU_SDK_INCLUDE=/path/to/dir/with/vip_lite.h
export CUBIE_NPU_SDK_LIB=/path/to/dir/with/libVIPhal.so
```

You also need the NPU kernel driver loaded (`/dev/vipcore` must exist), which it is on the stock Radxa images.

## Install

```bash
pip install cubie-npu
# or the latest from GitHub:
pip install git+https://github.com/Aseel1206/cubie-npu
```

Needs Python 3.9+ and NumPy. Then check everything with:

```bash
cubie-npu doctor               # SDK found? compiler? driver? builds the shim, prints the driver version
cubie-npu doctor model.nb      # ... and loads + times a model
```

Debian 11's own Python has no `pip` or `venv` module out of the box (`sudo apt install python3-pip python3-venv`, or use conda/miniforge). I tested with Python 3.11 in a venv and with the system Python 3.9.2 with the package on `PYTHONPATH`.

## Using it

```python
from cubie_npu import Model

m = Model("model.nb")           # loads it onto the NPU (milliseconds for small models, about a quarter of a second for the 200 MB language model)
m.inputs                        # list of TensorInfo(index, name, shape, dtype, quant, fixed_point_pos, scale, zero_point, nbytes)
m.outputs

outs = m.run(x)                 # -> list of float32 arrays (de-quantised, shaped like the network outputs)
y = m(x)                        # same, but returns the array itself when there is only one output
m.last_run_ms                   # NPU time of the last run
m.benchmark(100)                # {'mean_ms': ..., 'p50_ms': ..., 'p95_ms': ..., 'fps': ...}
```

**Inputs.** Pass a NumPy array with the right number of elements:

- If its dtype is the input's native dtype it is copied as it is. For the models in my [Cubie A7A repos](#models-that-work-with-it) that is `uint8` RGB, because the normalisation is built into the network.
- If it is a float array and the input is quantised (int16 fixed point, int8/uint8 affine) or fp16, it is converted for you, rounded and clipped.
- Raw bytes work too, if they have exactly `nbytes` bytes.
- A wrong size gives a `ValueError` saying what the model expects.

**Outputs** are always float32 unless you pass `raw=True`.

**Several inputs, or only some changing.** `m.set_input(i, array)`, then `m.invoke()`, then `m.get_output(i)`. A language model with a big KV cache uses this so it does not rewrite everything each token. See `examples/chat.py`.

**Command line.** `cubie-npu info model.nb` prints the inputs and outputs with shapes, dtypes and quantisation; `cubie-npu bench model.nb -n 100` times it.

The NPU has a single core and no preemption. Two models used from different threads or processes just take turns, so their times add up. `Model.run` takes a lock, so one `Model` is safe to share between threads.

The vendor library prints a "VIPLite driver software version" line to stdout. cubie-npu hides it; set `CUBIE_NPU_VERBOSE=1` to see it.

## How much does Python cost?

I ran every model through `cubie-npu bench` and through my C++ benchmark tool on the same board, 100 runs each:

| model | cubie-npu (Python) | C++ tool |
|---|---|---|
| BlazeFace 128x128 int16 | 1.14 ms | 1.02 ms |
| MediaPipe palm detector int8 | 3.32 ms | 3.22 ms |
| MediaPipe hand landmarks int8 | 1.95 ms | 1.77 ms |
| DeiT-tiny int8 | 14.59 ms | 14.59 ms |
| Depth Anything V2 small int8 | 255.9 ms | 255.9 ms |
| SmolLM2-135M int16 (one token) | 41.8 ms | 42.7 ms |

So about 0.1-0.2 ms of call overhead, which only shows on the tiny models. Outputs are bit-identical to the C++ tool (I compared the DeiT logits and the hand landmarks: max difference 0.0).

## Examples

In `examples/`. They need the models from my repos (below), plus `opencv-python-headless` for the image ones and `tokenizers` for the chat.

- **`classify.py`**: ImageNet classification with DeiT-tiny.
- **`hands.py`**: MediaPipe hand tracking with the palm detector and landmark model, in one process. On one still image the whole pipeline (letterbox, palm net, decode + NMS, rotated crop, landmark net) takes about **10 ms on the board** from Python. That does not include a camera or drawing, and it is one hand in one image, so it is not a promise about your camera loop (`hands.py` has a camera mode, `python hands.py 0 --show`, which I have not tested).
- **`chat.py`**: SmolLM2-135M chat with the whole decode loop in Python, about 21-22 tokens/s (the C++ version does 22.8).

```bash
python examples/hands.py photo.jpg out.png --bench 100
```

## GPU helpers (experimental, read this before using)

The board also has a GPU (PowerVR BXM-4-64). `cubie_npu.gpu` makes it usable from Python and has a handful of image functions on it. **My measurements say it does not beat OpenCV on the CPU for these jobs**, so treat it as an experiment, not an optimisation.

```bash
pip install "cubie-npu[gpu]"      # adds pyopencl
cubie-npu gpu-info                # the GPU, with the PowerVR OpenCL driver registered for you
cubie-npu gpu-bench               # GPU vs OpenCV on your board
```

### Setting up the GPU (OpenCL, and what about Vulkan)

```bash
cubie-npu setup-gpu          # checks everything below and prints the commands for whatever is missing
```

What it checks, in the order things usually go wrong:

1. **The kernel driver**: `/dev/dri/renderD128` exists (`lsmod | grep pvr` shows `pvrsrvkm`). If your user is not in the `render`/`video` groups: `sudo usermod -aG render,video $USER`, then log in again.
2. **The PowerVR OpenCL driver**: `/usr/lib/libPVROCL.so` (it was already there on the Radxa image I tested). If it lives elsewhere: `export CUBIE_NPU_OCL_LIB=/path/to/libPVROCL.so`.
3. **The OpenCL loader and Python side**: `sudo apt install ocl-icd-libopencl1 ocl-icd-opencl-dev clinfo build-essential python3-dev`, then `pip install "cubie-npu[gpu]"`.
4. **Registration**: nothing to do for `cubie_npu.gpu`, it registers the driver itself. For other programs (`clinfo`, your own pyopencl code) do it by hand once per shell:
   ```bash
   mkdir -p ~/ocl_icd && echo /usr/lib/libPVROCL.so > ~/ocl_icd/powervr.icd
   export OCL_ICD_VENDORS=~/ocl_icd
   clinfo | head
   ```

**Vulkan:** `cubie_npu.gpu` does not use it, it is OpenCL only, but Vulkan is there too. On the stock image the PowerVR Vulkan driver (`/usr/lib/libVK_IMG.so`, registered through `/usr/share/vulkan/icd.d/img_icd.json`) is already installed, and `vulkaninfo --summary` lists `PowerVR B-Series BXM-4-64 MC1` with Vulkan 1.3 next to a software `llvmpipe` device. If you need it: `sudo apt install libvulkan1 vulkan-tools`, then `vulkaninfo --summary | grep deviceName`. `cubie-npu setup-gpu` runs that check and, if the PowerVR device is missing, prints the ICD file to create or the `VK_ICD_FILENAMES` override to try. I only confirmed that the device shows up; I have not run any Vulkan compute workload on it (ncnn, kompute and the like).

What is useful here regardless of speed: on the stock image the registered OpenCL driver is Mesa's, which does not drive this GPU, so a plain `pyopencl` finds nothing. `cubie_npu.gpu` registers `/usr/lib/libPVROCL.so` for you (it writes a one-line `.icd` file in `~/.cache/cubie_npu/` and points `OCL_ICD_VENDORS` at it; if you set `OCL_ICD_VENDORS` yourself it is left alone).

```python
from cubie_npu import gpu
img = gpu.Image(frame_bgr)                           # upload once; images stay on the GPU between calls
kp = img.resize((640, 360), "area").to_gray().gaussian_blur(5).fast(threshold=25)   # (N,3): x, y, score
```

Available on `gpu.Image`: `resize` (bilinear / area), `letterbox`, `swap_rb`, `to_gray`, `normalize` (to float16/float32, NHWC or NCHW), `warp_affine`, `gaussian_blur`, `pyramid`, `sobel`, `fast` (FAST-9 with non-max suppression). Every one is checked against OpenCV in `tests/test_gpu.py` (resize, warp and blur within 1-3 grey levels, Sobel and swap exact, FAST without suppression returns exactly OpenCV's keypoints).

**The measurements** (Cubie A7A, 1280x720 frame, OpenCV pinned to one thread; `cubie-npu gpu-bench` reproduces them):

| task | OpenCV wall / CPU time | GPU wall / CPU time |
|---|---|---|
| letterbox 1280x720 -> 224, area filter | 5.4 / 5.3 ms | 10.9 / 2.9 ms |
| feature front end: resize + gray + blur + FAST | 2.3 / 2.3 ms | 36 / 10 ms |
| rotated 224x224 crop (warpAffine) | 1.6 / 1.5 ms | 4.5 / 2.8 ms |

- The GPU is small: one compute unit at 600 MHz, about 19 GFLOPS fp32 (36 with fp16 vectors), 4 KiB of local memory, 5 GB/s of memory bandwidth, and slow copies to and from host memory (0.5 to 2.6 GB/s). For scale, my rough estimate is that the CPU manages about as much on a transformer like DeiT-tiny (roughly 36 GFLOPS effective), and the NPU runs that model in 14.5 ms.
- Good news: while a GPU kernel runs, the CPU really is idle (0.04 ms of CPU over a 15 ms wait), so the idea is sound in principle.
- Bad news: every operation costs CPU time to enqueue (about 1 ms) and to read back, and OpenCV's NEON code on one A55 core is simply faster than this GPU for resize, colour conversion, blur, warp and FAST.
- The one place it saved CPU time was the large area-filter downscale (about half the CPU time of OpenCV), at twice the latency.
- My kernels are not fully tuned. On this GPU wide vector loads matter a lot (a 5-tap blur went from 4.1 ms to 0.6 ms in isolation), and I only applied that to some of them. I do not think it changes the verdict, but I can't prove that.
- Where the GPU could make sense: your own compute-heavy OpenCL kernels (big, few launches), where `cubie_npu.gpu.info()` and the driver setup save you the trouble. For image preprocessing, use OpenCV.

## Models that work with it

I converted these for the A733 and wrote up how, including the things that went wrong:

- [radxa-cubie-a7a-npu-mediapipe-hands](https://github.com/Aseel1206/radxa-cubie-a7a-npu-mediapipe-hands): palm detector + hand landmarks
- [radxa-cubie-a7a-npu-blazeface](https://github.com/Aseel1206/radxa-cubie-a7a-npu-blazeface)
- [radxa-cubie-a7a-npu-deit-tiny](https://github.com/Aseel1206/radxa-cubie-a7a-npu-deit-tiny)
- [radxa-cubie-a7a-npu-depth-anything-v2](https://github.com/Aseel1206/radxa-cubie-a7a-npu-depth-anything-v2)
- [radxa-cubie-a7a-npu-llm-smollm2-135m](https://github.com/Aseel1206/radxa-cubie-a7a-npu-llm-smollm2-135m)
- [fast-acvnet-radxa-npu](https://github.com/Aseel1206/fast-acvnet-radxa-npu): stereo depth

I'd expect any `.nb` built with the Allwinner Acuity toolchain for the A733 to load, but I have only tried the ones above.

## Limits

- Fixed shapes only (that is how `.nb` files work), no dynamic batch.
- Supported tensor formats: fp32, fp16, int8, uint8, int16, uint16, int32, uint32, int64, uint64, fp64, bool. bfloat16 raises an error.
- Quantised *outputs* are de-quantised for you, which costs a little CPU time for big outputs. `raw=True` skips it.
- No asynchronous API: `run` blocks until the NPU is done.
- Tested on one board and one driver version. If it does not work on yours, `cubie-npu doctor` output in an issue helps a lot.

## How it works

`src/cubie_npu/_shim.cpp` wraps `vip_create_network`, `vip_create_buffer`, `vip_run_network` and friends in a handful of plain C functions. `_backend.py` finds the SDK, compiles the shim with `g++` and loads it with `ctypes` (no pybind11, no Python headers, so no build problems across Python versions). `model.py` queries each input and output (format, dimensions, fixed-point position or scale/zero-point) through the shim, and does the NumPy conversion.

## Contributing / publishing

Issues and pull requests are welcome. Tests: `pytest` (hardware tests are skipped unless `CUBIE_NPU_TEST_NB=/path/to/model.nb` is set). Releases go to PyPI from GitHub through trusted publishing (`.github/workflows/publish.yml`).

## Licence

MIT, see `LICENSE`. The VIPLite header and libraries are Allwinner's and are not included.
