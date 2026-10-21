# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolated provider doubles for CPU-only mirror lifecycle tests."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def mock_omni_ui(monkeypatch):
    def install(ui):
        omni = ModuleType("omni")
        omni.__path__ = []
        omni.ui = ui
        monkeypatch.setitem(sys.modules, "omni", omni)
        monkeypatch.setitem(sys.modules, "omni.ui", ui)

    return install


@pytest.fixture
def mock_cuda(monkeypatch):
    cuda = ModuleType("cuda")
    cuda.__path__ = []
    bindings = ModuleType("cuda.bindings")
    bindings.__path__ = []
    runtime = ModuleType("cuda.bindings.runtime")
    runtime.cudaStreamLegacy = 1
    runtime.cudaEventDisableTiming = 2
    runtime.cudaMemcpyKind = SimpleNamespace(cudaMemcpyDeviceToDevice=3)
    runtime.cudaGLDeviceList = SimpleNamespace(cudaGLDeviceListAll=1)
    runtime.cudaMemcpy3DPeerParms = lambda: SimpleNamespace(
        dstPtr=SimpleNamespace(), extent=SimpleNamespace()
    )
    for name in (
        "cudaSetDevice",
        "cudaMallocPitch",
        "cudaFree",
        "cudaMemcpy2DAsync",
        "cudaMemcpy2DFromArrayAsync",
        "cudaMemcpy3DPeerAsync",
        "cudaStreamSynchronize",
        "cudaGLGetDevices",
        "cudaEventCreateWithFlags",
        "cudaEventRecord",
        "cudaEventSynchronize",
        "cudaEventDestroy",
    ):
        setattr(runtime, name, Mock(side_effect=AssertionError(f"Unexpected {name}")))
    cuda.bindings = bindings
    bindings.runtime = runtime
    for module in (cuda, bindings, runtime):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return runtime
