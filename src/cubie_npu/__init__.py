"""cubie-npu: run Allwinner A733 / VIPLite .nb models on the Radxa Cubie A7A NPU from Python.

    import numpy as np, cubie_npu
    m = cubie_npu.Model("model.nb")
    out = m.run(np.zeros(m.inputs[0].shape, m.inputs[0].dtype))
"""
from ._backend import NPUError, find_sdk, build_shim
from .model import Model, TensorInfo, driver_version

__version__ = "0.1.0"
__all__ = ["Model", "TensorInfo", "NPUError", "find_sdk", "build_shim", "driver_version", "__version__"]
