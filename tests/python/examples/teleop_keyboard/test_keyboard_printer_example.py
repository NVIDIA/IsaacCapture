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
