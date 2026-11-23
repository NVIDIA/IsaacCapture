# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the Linux dependency preflight against an isolated header/tool root."""

import shutil
import subprocess

import pytest
from repo_paths import repo_root


HEADERS = {
    "X11/Xlib.h": "libx11-dev",
    "X11/extensions/Xrandr.h": "libxrandr-dev",
    "X11/extensions/Xinerama.h": "libxinerama-dev",
    "X11/Xcursor/Xcursor.h": "libxcursor-dev",
    "X11/extensions/XInput2.h": "libxi-dev",
    "X11/extensions/shape.h": "libxext-dev",
    "X11/XKBlib.h": "libx11-dev",
}


def _preflight(tmp_path, *, headers=(), viz=True, x11=True, formatting=False):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required")
    root = tmp_path / "sysroot"
    for header in headers:
        path = root / "usr/include" / header
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    pkg_config = tmp_path / "pkg-config"
    pkg_config.write_text("#!/bin/sh\necho 1.0.0\n")
    pkg_config.chmod(0o755)
    script = tmp_path / "check.cmake"
    script.write_text(
        f"""
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_FIND_ROOT_PATH "{root.as_posix()}")
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM ONLY)
set(CMAKE_INCLUDE_PATH /usr/include)
set(PKG_CONFIG_EXECUTABLE "{pkg_config.as_posix()}")
set(BUILD_VIZ {int(viz)})
set(GLFW_BUILD_X11 {int(x11)})
set(GLFW_BUILD_WAYLAND OFF)
set(ENABLE_CLANG_FORMAT_CHECK {int(formatting)})
set(GLSLANG_VALIDATOR /provided/glslangValidator)
set(PATCHELF_EXECUTABLE /provided/patchelf)
set(EGL_INCLUDE_DIR /provided/include)
include("{(repo_root() / "cmake/CheckBuildDeps.cmake").as_posix()}")
isaac_teleop_check_build_deps()
"""
    )
    return subprocess.run(
        [cmake, "-P", str(script)], capture_output=True, text=True, check=False
    )


def test_missing_x11_headers_report_all_apt_packages(tmp_path):
    result = _preflight(tmp_path)
    assert result.returncode != 0
    command = result.stderr.split("sudo apt-get install -y", 1)[1].splitlines()[0]
    assert set(command.split()) == set(HEADERS.values())
    for header in HEADERS:
        assert header in result.stderr


@pytest.mark.parametrize("missing", HEADERS)
def test_partial_x11_install_reports_the_missing_header(tmp_path, missing):
    result = _preflight(tmp_path, headers=HEADERS.keys() - {missing})
    assert result.returncode != 0
    command = result.stderr.split("sudo apt-get install -y", 1)[1].splitlines()[0]
    assert command.split() == [HEADERS[missing]]


@pytest.mark.parametrize(
    "options",
    [{"headers": HEADERS}, {"x11": False}, {"viz": False}],
)
def test_no_x11_prerequisite_when_satisfied_or_disabled(tmp_path, options):
    result = _preflight(tmp_path, **options)
    assert result.returncode == 0, result.stderr


def test_formatting_gap_is_reported_with_x11_gaps(tmp_path):
    result = _preflight(tmp_path, formatting=True)
    assert result.returncode != 0
    assert "X11/extensions/Xrandr.h" in result.stderr
    assert "clang-format-14" in result.stderr
    command = result.stderr.split("sudo apt-get install -y", 1)[1].splitlines()[0]
    assert "clang-format-14" in command.split()
