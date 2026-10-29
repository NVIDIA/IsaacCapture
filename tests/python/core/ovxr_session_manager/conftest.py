# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolated provider doubles for CPU-only mirror lifecycle tests."""

import asyncio
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def mock_omni_ui(monkeypatch):
    def install(ui):
        async def run_async():
            await asyncio.Future()

        if ui is not None and not hasattr(ui, "run_async"):
            ui.run_async = run_async
        if ui is not None and not hasattr(ui, "has_gpu_byte_image"):
            ui.has_gpu_byte_image = lambda: True
        if ui is not None and not hasattr(ui, "standalone"):
            ui.standalone = SimpleNamespace(
                get_frame_pacing_status=lambda: {"backend": "vulkan"}
            )
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
    for name in (
        "cudaSetDevice",
        "cudaGetDevice",
        "cudaMallocPitch",
        "cudaFree",
        "cudaMemcpy2DAsync",
        "cudaMemcpy2DFromArrayAsync",
        "cudaStreamSynchronize",
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
