# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused setup regressions; keep rendering coverage in the existing GPU suite."""

import sys
from unittest.mock import MagicMock

import pytest

from scripts import check_cuda


@pytest.mark.parametrize("version", ["12.6", "13.0"])
def test_explicit_toolkit_overrides_path(tmp_path, monkeypatch, version):
    toolkit = tmp_path / f"cuda-{version}"
    toolkit.mkdir()
    monkeypatch.setenv("CUDA_PATH", str(toolkit))
    monkeypatch.setattr(check_cuda.shutil, "which", lambda _: "/wrong/bin/nvcc")
    assert check_cuda.detect_toolkit() == (
        toolkit,
        tuple(map(int, version.split("."))),
    )


def test_unknown_toolkit_does_not_guess_cuda12(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_PATH", str(tmp_path))
    with pytest.raises(RuntimeError, match="Cannot determine the CUDA toolkit"):
        check_cuda.detect_toolkit()


@pytest.fixture
def cupy(monkeypatch):
    cp = MagicMock()
    cp.cuda.runtime.getDeviceProperties.return_value = {"name": b"NVIDIA Thor"}
    cp.cuda.nvrtc.getVersion.return_value = (13, 0)
    monkeypatch.setitem(sys.modules, "cupy", cp)
    return cp


def test_mismatched_nvrtc_is_rejected(cupy):
    cupy.cuda.nvrtc.getVersion.return_value = (12, 6)
    with pytest.raises(RuntimeError, match="Selected CUDA 13, but CuPy loaded NVRTC"):
        check_cuda.check_kernel(13)


@pytest.mark.parametrize("at_sync", [False, True])
def test_kernel_failure_fails_setup(cupy, capsys, at_sync):
    operation = cupy.cuda.Stream.null.synchronize if at_sync else cupy.arange
    operation.side_effect = RuntimeError("CUDA_ERROR_NO_BINARY_FOR_GPU")
    assert check_cuda.main(["--cuda-major", "13"]) == 1
    output = capsys.readouterr()
    assert "CUDA_ERROR_NO_BINARY_FOR_GPU" in output.err
    assert "CUPY_COMPILE_WITH_PTX=1" in output.err
    assert "passed" not in output.out
