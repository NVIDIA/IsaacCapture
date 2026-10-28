# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused setup regressions; keep rendering coverage in the existing GPU suite."""

import os
import sys
from unittest.mock import MagicMock

import pytest

import cuda_env
from scripts import check_cuda


def _nvcc(root, release, headers=True):
    """A toolkit root whose bin/nvcc reports ``release``, with no version file."""
    (root / "bin").mkdir(parents=True)
    nvcc = root / "bin" / "nvcc"
    nvcc.write_text(
        f"#!/bin/sh\necho 'Cuda compilation tools, release {release}, V{release}.140'\n"
    )
    nvcc.chmod(0o755)
    if headers:
        (root / "include").mkdir()
        (root / "include" / "cuda_runtime.h").touch()
    return root


@pytest.fixture
def host(tmp_path, monkeypatch):
    """No CUDA_PATH, nothing on PATH, and no default toolkit."""
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(check_cuda, "DEFAULT_ROOT", tmp_path / "missing")
    return monkeypatch


def _on_path(monkeypatch, *roots):
    monkeypatch.setenv("PATH", os.pathsep.join(str(r / "bin") for r in roots))


@pytest.mark.parametrize("version", ["12.6", "13.0"])
def test_explicit_toolkit_overrides_path(tmp_path, host, version):
    toolkit = tmp_path / f"cuda-{version}"
    toolkit.mkdir()
    _on_path(host, _nvcc(tmp_path / "usr", "12.0"))
    host.setenv("CUDA_PATH", str(toolkit))
    assert check_cuda.detect_toolkit() == (
        toolkit,
        tuple(map(int, version.split("."))),
        False,
    )


def test_unknown_toolkit_does_not_guess_cuda12(tmp_path, host):
    _on_path(host, _nvcc(tmp_path / "usr", "12.0"))
    default = tmp_path / "cuda-12.6"
    default.mkdir()
    host.setattr(check_cuda, "DEFAULT_ROOT", default)
    unknown = tmp_path / "unknown"
    unknown.mkdir()
    host.setenv("CUDA_PATH", str(unknown))
    with pytest.raises(RuntimeError, match="Cannot determine the CUDA toolkit"):
        check_cuda.detect_toolkit()


def test_first_version_file_on_path_wins(tmp_path, host):
    first = _nvcc(tmp_path / "cuda-12.8", "12.8")
    second = _nvcc(tmp_path / "cuda-12.6", "12.6")
    default = tmp_path / "cuda-13.0"
    default.mkdir()
    host.setattr(check_cuda, "DEFAULT_ROOT", default)
    _on_path(host, first, second)
    assert check_cuda.detect_toolkit() == (first, (12, 8), False)


def test_later_candidates_are_not_read_once_one_wins(tmp_path, host):
    first = _nvcc(tmp_path / "cuda-12.6", "12.6")
    broken = _nvcc(tmp_path / "broken", "13.0")
    (broken / "version.json").write_text("not json")
    _on_path(host, first, broken)
    assert check_cuda.detect_toolkit() == (first, (12, 6), False)


def test_default_root_symlink_loop(tmp_path, host):
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    host.setattr(check_cuda, "DEFAULT_ROOT", loop)
    with pytest.raises(RuntimeError, match="Cannot determine"):
        check_cuda.detect_toolkit()
    toolkit = _nvcc(tmp_path / "cuda-12.6", "12.6")
    _on_path(host, toolkit)
    assert check_cuda.detect_toolkit() == (toolkit, (12, 6), False)


def test_empty_path_entry_is_current_directory(tmp_path, host):
    toolkit = _nvcc(tmp_path / "cuda-13.0", "13.0")
    host.chdir(toolkit / "bin")
    host.setenv("PATH", "")
    with pytest.raises(RuntimeError, match="Cannot determine"):
        check_cuda.detect_toolkit()
    host.setenv("PATH", os.pathsep)
    assert check_cuda.detect_toolkit() == (toolkit, (13, 0), False)


def test_distro_toolkit_is_read_from_nvcc(tmp_path, host, capsys):
    usr = _nvcc(tmp_path / "usr", "12.0")
    _on_path(host, usr)
    assert check_cuda.detect_toolkit() == (usr, (12, 0), True)
    assert check_cuda.main(["--detect"]) == 0
    assert capsys.readouterr().out == f"{usr}\t12\t0\t{usr}\n"


def test_explicit_cuda_path_accepts_distro_root(tmp_path, host):
    usr = _nvcc(tmp_path / "usr", "12.0")
    host.setenv("CUDA_PATH", str(usr))
    assert check_cuda.detect_toolkit() == (usr, (12, 0), True)


def test_version_file_wins_over_earlier_distro_nvcc(tmp_path, host, capsys):
    usr = _nvcc(tmp_path / "usr", "12.0")
    official = _nvcc(tmp_path / "cuda-13.0", "13.0")
    _on_path(host, usr, official)
    assert check_cuda.detect_toolkit() == (official, (13, 0), False)
    assert check_cuda.main(["--detect"]) == 0
    assert capsys.readouterr().out == f"{official}\t13\t0\t\n"


def test_default_root_follows_unreadable_nvcc(tmp_path, host):
    _on_path(host, _nvcc(tmp_path / "opt", "12.4", headers=False))
    default = tmp_path / "cuda-12.6"
    default.mkdir()
    host.setattr(check_cuda, "DEFAULT_ROOT", default)
    assert check_cuda.detect_toolkit() == (default, (12, 6), False)


def test_nvcc_without_headers_is_not_selected(tmp_path, host):
    _on_path(host, _nvcc(tmp_path / "usr", "12.0", headers=False))
    with pytest.raises(RuntimeError, match="Cannot determine the CUDA toolkit"):
        check_cuda.detect_toolkit()


def test_unsupported_toolkit_is_skipped(tmp_path, host):
    _on_path(host, _nvcc(tmp_path / "cuda-11.8", "11.8"))
    default = tmp_path / "cuda-12.6"
    default.mkdir()
    host.setattr(check_cuda, "DEFAULT_ROOT", default)
    assert check_cuda.detect_toolkit() == (default, (12, 6), False)


def test_only_unsupported_toolkits_fail(tmp_path, host):
    _on_path(host, _nvcc(tmp_path / "usr", "11.5"))
    with pytest.raises(RuntimeError, match=r"Unsupported CUDA toolkit 11\.5 at "):
        check_cuda.detect_toolkit()


@pytest.mark.parametrize(
    ("explicit", "expected"),
    [(None, "/usr"), ("CUDA_PATH", "/opt/cuda"), ("CUDA_HOME", None)],
)
def test_recorded_cuda_path_yields_to_explicit_env(
    tmp_path, monkeypatch, explicit, expected
):
    (tmp_path / cuda_env.RECORD).write_text("/usr\n")
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    for name in ("CUDA_PATH", "CUDA_HOME"):
        # setenv first so teardown also undoes what the code under test sets.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    if explicit:
        monkeypatch.setenv(explicit, "/opt/cuda")
    cuda_env.apply_recorded_cuda_path()
    assert os.environ.get("CUDA_PATH") == expected


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


def test_setup_check_applies_recorded_cuda_path(tmp_path, monkeypatch, cupy, capsys):
    (tmp_path / cuda_env.RECORD).write_text("/usr\n")
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    for name in ("CUDA_PATH", "CUDA_HOME"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    seen = []
    cupy.cuda.Device.side_effect = lambda: seen.append(os.environ.get("CUDA_PATH"))
    check_cuda.main(["--cuda-major", "13"])
    assert seen == ["/usr"]


@pytest.mark.parametrize("at_sync", [False, True])
def test_kernel_failure_fails_setup(cupy, capsys, at_sync):
    operation = cupy.cuda.Stream.null.synchronize if at_sync else cupy.arange
    operation.side_effect = RuntimeError("CUDA_ERROR_NO_BINARY_FOR_GPU")
    assert check_cuda.main(["--cuda-major", "13"]) == 1
    output = capsys.readouterr()
    assert "CUDA_ERROR_NO_BINARY_FOR_GPU" in output.err
    assert "CUPY_COMPILE_WITH_PTX=1" in output.err
    assert "passed" not in output.out
