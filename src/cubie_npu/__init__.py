"""cubie-npu: run Allwinner A733 / VIPLite .nb models on the Radxa Cubie A7A NPU from Python.

    import numpy as np, cubie_npu
    m = cubie_npu.Model("model.nb")
    out = m.run(np.zeros(m.inputs[0].shape, m.inputs[0].dtype))

or, with models downloaded for you and plain images in:

    pipe = cubie_npu.pipeline("image-classification"); pipe("cat.jpg")
"""
from ._backend import NPUError, find_sdk, build_shim
from .model import Model, TensorInfo, driver_version
from . import hub
from .pipelines import pipeline

__version__ = "0.2.0"
__all__ = ["Model", "TensorInfo", "NPUError", "find_sdk", "build_shim", "driver_version", "pipeline", "hub", "__version__"]
