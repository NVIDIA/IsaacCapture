# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLFW scancodes map to evdev codes per backend (no window is opened)."""

import glfw
import pytest

from keyboard_printer_example import evdev_code_from_scancode

KEY_W, KEY_UP, KEY_F1 = 17, 103, 59


@pytest.mark.parametrize(
    ("platform", "scancode", "expected"),
    [
        (glfw.PLATFORM_X11, 25, KEY_W),
        (glfw.PLATFORM_X11, 111, KEY_UP),
        (glfw.PLATFORM_X11, 67, KEY_F1),
        (glfw.PLATFORM_WAYLAND, 17, KEY_W),
        (glfw.PLATFORM_WAYLAND, 103, KEY_UP),
        (glfw.PLATFORM_WAYLAND, 59, KEY_F1),
    ],
)
def test_scancode_to_evdev(platform, scancode, expected):
    assert evdev_code_from_scancode(platform, scancode) == expected


@pytest.mark.parametrize(
    "platform", [glfw.PLATFORM_WIN32, glfw.PLATFORM_COCOA, glfw.PLATFORM_NULL]
)
def test_other_platforms_are_rejected(platform):
    with pytest.raises(RuntimeError, match="only X11 and Wayland"):
        evdev_code_from_scancode(platform, 25)


def _load_example_with(glfw_stub, monkeypatch):
    """Import the example afresh against a stand-in glfw module."""
    import importlib.util
    import sys

    import keyboard_printer_example

    monkeypatch.setitem(sys.modules, "glfw", glfw_stub)
    spec = importlib.util.spec_from_file_location(
        "printer_under_test", keyboard_printer_example.__file__
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pyglfw_older_than_2_7_is_rejected_at_import(monkeypatch):
    from types import SimpleNamespace

    with pytest.raises(ImportError, match=r"glfw>=2\.7"):
        _load_example_with(SimpleNamespace(), monkeypatch)


def test_glfw_library_without_get_platform_is_rejected(monkeypatch):
    from types import SimpleNamespace

    terminated = []
    stub = SimpleNamespace(
        PLATFORM_X11=glfw.PLATFORM_X11,
        PLATFORM_WAYLAND=glfw.PLATFORM_WAYLAND,
        init=lambda: True,
        terminate=lambda: terminated.append(True),
    )
    module = _load_example_with(stub, monkeypatch)

    with pytest.raises(RuntimeError, match=r"older than 3\.4"):
        module.GlfwKeyWindow("test")
    assert terminated
