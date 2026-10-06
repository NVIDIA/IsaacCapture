# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA discovery and failure diagnostics, without requiring a GPU."""

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import check_cuda


@pytest.mark.parametrize("version", ["12.6.10", "13.0.0", "13.2.0"])
def test_explicit_toolkit_takes_precedence(tmp_path, monkeypatch, version):
    toolkit = tmp_path / "selected"
    toolkit.mkdir()
    (toolkit / "version.json").write_text(json.dumps({"cuda": {"version": version}}))
    monkeypatch.setenv("CUDA_PATH", str(toolkit))
    monkeypatch.setattr(
        check_cuda.shutil, "which", lambda _: "/wrong/cuda-12.0/bin/nvcc"
    )
    assert check_cuda.detect_toolkit() == (
        toolkit,
        tuple(map(int, version.split(".")[:2])),
    )


def test_toolkit_on_path(tmp_path, monkeypatch):
    toolkit = tmp_path / "cuda-13.0"
    toolkit.mkdir()
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(check_cuda.shutil, "which", lambda _: str(toolkit / "bin/nvcc"))
    assert check_cuda.detect_toolkit() == (toolkit, (13, 0))


def test_toolkit_symlink(tmp_path, monkeypatch):
    toolkit = tmp_path / "cuda-12.6"
    toolkit.mkdir()
    alias = tmp_path / "cuda"
    alias.symlink_to(toolkit)
    monkeypatch.setenv("CUDA_PATH", str(alias))
    assert check_cuda.detect_toolkit() == (toolkit, (12, 6))


@pytest.mark.parametrize("exists", [False, True])
def test_unknown_toolkit_does_not_guess_cuda12(tmp_path, monkeypatch, exists):
    toolkit = tmp_path / "unknown"
    if exists:
        toolkit.mkdir()
    monkeypatch.setenv("CUDA_PATH", str(toolkit))
    with pytest.raises(RuntimeError, match=r"Cannot determine.*CUDA toolkit"):
        check_cuda.detect_toolkit()


class _Array(np.ndarray):
    def get(self):
        return np.asarray(self)


@pytest.fixture
def cupy(monkeypatch):
    cp = SimpleNamespace(
        __version__="test-version",
        float32=np.float32,
        arange=lambda *a, **kw: np.arange(*a, **kw).view(_Array),
        cuda=SimpleNamespace(
            Device=lambda: SimpleNamespace(id=0),
            runtime=SimpleNamespace(
                getDeviceProperties=lambda _: {"name": b"NVIDIA Thor"}
            ),
            nvrtc=SimpleNamespace(getVersion=lambda: (13, 0)),
            Stream=SimpleNamespace(null=SimpleNamespace(synchronize=lambda: None)),
        ),
        show_config=lambda: print("CUDA diagnostic configuration"),
    )
    monkeypatch.setitem(sys.modules, "cupy", cp)
    return cp


@pytest.mark.parametrize(
    "name, version", [("Orin", (12, 6)), ("Thor", (13, 0)), ("Thor", (13, 2))]
)
def test_matching_runtime_executes_kernel(cupy, name, version):
    cupy.cuda.nvrtc.getVersion = lambda: version
    cupy.cuda.runtime.getDeviceProperties = lambda _: {"name": name.encode()}
    assert name in check_cuda.check_kernel(version[0])


def test_mismatched_nvrtc_is_rejected(cupy):
    cupy.cuda.nvrtc.getVersion = lambda: (12, 6)
    with pytest.raises(
        RuntimeError, match=r"Selected CUDA 13, but CuPy loaded NVRTC 12\.6"
    ):
        check_cuda.check_kernel(13)


@pytest.mark.parametrize("at_sync", [False, True])
def test_kernel_failure_returns_nonzero_and_diagnostics(cupy, capsys, at_sync):
    def fail(*a, **kw):
        raise RuntimeError("CUDA_ERROR_NO_BINARY_FOR_GPU")

    if at_sync:
        cupy.cuda.Stream.null.synchronize = fail
    else:
        cupy.arange = fail
    assert check_cuda.main(["--cuda-major", "13"]) == 1
    output = capsys.readouterr()
    assert "CUDA_ERROR_NO_BINARY_FOR_GPU" in output.err
    assert "CUPY_COMPILE_WITH_PTX=1" in output.err
    assert "CUDA diagnostic configuration" in output.out
    assert "passed" not in output.out


def test_incorrect_kernel_result_is_rejected(cupy):
    cupy.arange = lambda *a, **kw: np.zeros(4).view(_Array)
    with pytest.raises(RuntimeError, match="incorrect results"):
        check_cuda.check_kernel(13)


def test_probe_uses_cold_cache_and_restores_environment(cupy, monkeypatch, tmp_path):
    original = str(tmp_path / "existing-cache")
    monkeypatch.setenv("CUPY_CACHE_DIR", original)
    paths = []
    arange = cupy.arange

    def checked_arange(*a, **kw):
        path = Path(os.environ["CUPY_CACHE_DIR"])
        assert str(path) != original
        assert path.is_dir()
        paths.append(path)
        return arange(*a, **kw)

    cupy.arange = checked_arange
    assert check_cuda.main([]) == 0
    assert os.environ["CUPY_CACHE_DIR"] == original
    assert len(paths) == 1
    assert not paths[0].exists()
