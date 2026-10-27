# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""V4L2 backend failures must surface before capture starts, without a GPU."""

import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from pipeline import SourceSpec
from sources import _v4l2_backend
from sources._helpers import PolledSource
from sources.v4l2 import V4l2Source


@pytest.fixture
def opencv(monkeypatch):
    cv2 = SimpleNamespace(
        __version__="5.0.0",
        CAP_V4L2=200,
        CAP_PROP_FOURCC=6,
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
        CAP_PROP_FPS=5,
        CAP_PROP_BUFFERSIZE=38,
        videoio_registry=SimpleNamespace(hasBackend=Mock(return_value=True)),
        VideoCapture=Mock(),
        VideoWriter_fourcc=Mock(return_value=1196444237),
    )
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    allocations = []

    def init(source, name, width, height, **kwargs):
        allocations.append(name)
        source._cp = np
        source._spec = SourceSpec(name, width, height, "rgba8")

    monkeypatch.setattr(PolledSource, "__init__", init)
    return cv2, allocations


def test_missing_backend_fails_before_gpu_allocation(opencv):
    cv2, allocations = opencv
    cv2.videoio_registry.hasBackend.return_value = False
    with pytest.raises(RuntimeError, match=r"OpenCV 5\.0\.0.*V4L2") as error:
        V4l2Source("usb")
    assert "camera_viz.sh setup" in str(error.value)
    assert not allocations
    cv2.VideoCapture.assert_not_called()


def test_missing_opencv_fails_before_gpu_allocation(opencv, monkeypatch):
    _, allocations = opencv
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(RuntimeError, match="opencv-python"):
        V4l2Source("usb")
    assert not allocations


def test_available_backend_opens_and_configures_camera(opencv):
    cv2, _ = opencv
    cap = cv2.VideoCapture.return_value
    cap.isOpened.return_value = True
    cap.get.side_effect = [640, 480]
    source = V4l2Source("usb", "/dev/video6", 640, 480, 30, "MJPG")
    assert source._open_device()
    cv2.videoio_registry.hasBackend.assert_called_once_with(cv2.CAP_V4L2)
    cv2.VideoCapture.assert_called_once_with("/dev/video6", cv2.CAP_V4L2)
    assert [call.args for call in cap.set.call_args_list] == [
        (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc.return_value),
        (cv2.CAP_PROP_FRAME_WIDTH, 640),
        (cv2.CAP_PROP_FRAME_HEIGHT, 480),
        (cv2.CAP_PROP_FPS, 30.0),
        (cv2.CAP_PROP_BUFFERSIZE, 1),
    ]
    source._close_device()
    cap.release.assert_called_once()


def test_unplugged_camera_can_reconnect(opencv):
    cv2, _ = opencv
    cap = cv2.VideoCapture.return_value
    cap.isOpened.side_effect = [False, True]
    cap.get.side_effect = [640, 480]
    source = V4l2Source("usb", width=640, height=480)
    assert source._open_device() is False
    cap.release.assert_called_once()
    assert source._open_device() is True
    source._close_device()


@pytest.mark.skipif(sys.platform == "win32", reason="setup uses a POSIX shell")
@pytest.mark.parametrize(
    "enabled, available, exit_code",
    [(True, True, 0), (True, False, 1), (False, False, 0)],
)
def test_setup_checks_backend_when_enabled(tmp_path, enabled, available, exit_code):
    camera_dir = Path(_v4l2_backend.__file__).resolve().parent.parent
    toolkit = tmp_path / "cuda-12.6"
    toolkit.mkdir()
    venv = tmp_path / "venv"
    bin_dir = venv / "bin"
    bin_dir.mkdir(parents=True)
    wheel = tmp_path / "isaaccapture.whl"
    wheel.touch()
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "cv2.py").write_text(
        "from types import SimpleNamespace\n"
        "__version__ = 'test-build'\nCAP_V4L2 = 200\n"
        f"videoio_registry = SimpleNamespace(hasBackend=lambda _: {available!r})\n"
    )
    # Stub installation and GPU probes, but execute the real backend check.
    python = bin_dir / "python"
    python.write_text(
        '#!/bin/sh\ncase "$1" in\n'
        f'*/_v4l2_backend.py) exec {shlex.quote(sys.executable)} "$@" ;;\n'
        "esac\nexit 0\n"
    )
    python.chmod(0o755)
    uv = bin_dir / "uv"
    uv.write_text(
        '#!/bin/sh\nif [ "$1" = pip ] && [ "$2" = show ]; then exit 1; fi\nexit 0\n'
    )
    uv.chmod(0o755)
    args = [
        "bash",
        str(camera_dir / "scripts" / "_install_deps.sh"),
        "--venv",
        str(venv),
        "--wheel",
        str(wheel),
        "--no-oakd",
    ]
    if not enabled:
        args.append("--no-v4l2")
    result = subprocess.run(
        args,
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
            "CUDA_PATH": str(toolkit),
            "PYTHONPATH": str(modules),
        },
        check=False,  # A missing backend must make setup exit unsuccessfully.
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == exit_code, result.stdout + result.stderr
    assert ("setup complete" in result.stdout) == (exit_code == 0)
    if enabled:
        assert "opencv-python>=4.5,<5" in result.stdout
        assert ("V4L2 backend available" in result.stdout) == available
        if not available:
            assert "no available V4L2 backend" in result.stderr
    else:
        assert "verifying OpenCV V4L2 backend" not in result.stdout
