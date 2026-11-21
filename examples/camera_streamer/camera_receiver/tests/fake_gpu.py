# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NumPy stand-ins for CuPy and the NVDEC decoder.

They let the receiver's stream bookkeeping and its IPC control messages run on a
machine without a CUDA Python stack. The exported handles are not usable CUDA
memory, so nothing here checks decoded pixels.
"""

from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import numpy as np


class _Array(np.ndarray):
    @property
    def __cuda_array_interface__(self):
        return {
            "data": (self.ctypes.data, False),
            "shape": self.shape,
            "strides": self.strides,
        }

    def set(self, host):
        self[...] = np.asarray(host).reshape(self.shape)


@contextmanager
def _device(_gpu_id=0):
    yield


def fake_cupy() -> ModuleType:
    module = ModuleType("cupy")
    module.uint8 = np.uint8
    module.empty = lambda shape, dtype=np.uint8: np.zeros(shape, dtype).view(_Array)
    module.RawKernel = lambda _source, _name: lambda *_args: None
    module.cuda = SimpleNamespace(
        Device=_device,
        get_current_stream=lambda: SimpleNamespace(synchronize=lambda: None),
        runtime=SimpleNamespace(
            getDevice=lambda: 0, ipcGetMemHandle=lambda _ptr: bytes(64)
        ),
    )
    return module


class FakeDecoder:
    """Accepts any access unit and reports a decoded frame."""

    def __init__(self, **_kwargs):
        pass

    def decode_nv12(self, _payload, output, rotation=0):
        output[...] = 128
        return True

    decode = decode_nv12

    def reset(self):
        pass
