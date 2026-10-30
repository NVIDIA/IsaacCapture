# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Keyboard Printer Example.

Opens a small GLFW window and feeds its key events into a KeyboardSource. Keys only count
while that window has focus; click elsewhere and every held key is released. Prints held
keys each frame plus every press/release, using the "keyboard_held" and
"keyboard_pressed" bitmaps.

The window is a minimal ``KeyEventSource``: any host window (a sim viewer, a browser
viewer, ...) can feed Isaac Teleop the same way. It reports each physical key as its evdev
code, derived from GLFW's scancode (Linux X11 or Wayland), so it needs no key table.
Like every TeleopSession it opens an OpenXR session; the CloudXR launcher arguments start a
runtime, and no headset needs to connect.
"""

import argparse
import contextlib
import sys
import time
from types import SimpleNamespace

import glfw
import numpy as np

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.deviceio_trackers import w3c_code_from_evdev
from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    KeyboardSource,
    KeyListeners,
)
from isaaccapture.teleop_session_manager import TeleopSession, TeleopSessionConfig

# What GLFW's scancode adds to the evdev code, per backend: X11 reports the XKB keycode
# (evdev + 8), Wayland the evdev code itself.
_SCANCODE_OFFSETS = {glfw.PLATFORM_X11: 8, glfw.PLATFORM_WAYLAND: 0}


def _unsupported_platform(platform: int) -> RuntimeError:
    return RuntimeError(
        f"GLFW platform {platform:#x} is unsupported: only X11 and Wayland scancodes "
        "map to evdev key codes"
    )


def evdev_code_from_scancode(platform: int, scancode: int) -> int:
    """Evdev key code for a GLFW ``scancode`` on ``platform`` (``glfw.get_platform()``)."""
    offset = _SCANCODE_OFFSETS.get(platform)
    if offset is None:
        raise _unsupported_platform(platform)
    return scancode - offset


@contextlib.contextmanager
def glfw_library():
    """Initialize GLFW for the duration of the block."""
    if not glfw.init():
        raise RuntimeError("glfw.init() failed (no display?)")
    try:
        yield
    finally:
        glfw.terminate()


class GlfwKeyWindow:
    """A GLFW window implementing the KeyEventSource protocol. Needs GLFW initialized."""

    def __init__(self, title: str):
        self._platform = glfw.get_platform()
        if self._platform not in _SCANCODE_OFFSETS:
            raise _unsupported_platform(self._platform)
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        self._window = glfw.create_window(480, 120, title, None, None)
        if not self._window:
            raise RuntimeError("glfw.create_window() failed")
        self._listeners = KeyListeners()
        glfw.set_key_callback(self._window, self._on_key)
        glfw.set_window_focus_callback(self._window, self._on_focus)

    # KeyEventSource ------------------------------------------------------------
    def add_key_listener(self, on_key, on_focus_lost):
        return self._listeners.add(on_key, on_focus_lost)

    def capture_keyboard(self):
        return SimpleNamespace(
            close=lambda: None
        )  # this window has no key bindings of its own

    # GLFW callbacks --------------------------------------------------------------
    def _on_key(self, _window, _key, scancode, action, _mods):
        # Report the physical key as its evdev code, so no key table is needed.
        code = evdev_code_from_scancode(self._platform, scancode)
        if code > 0 and action != glfw.REPEAT:
            self._listeners.key(code, action == glfw.PRESS)

    def _on_focus(self, _window, focused):
        if not focused:
            self._listeners.focus_lost()

    # Host loop -----------------------------------------------------------------
    def poll(self) -> bool:
        """Dispatch window events; False once the window was closed."""
        glfw.poll_events()
        return not glfw.window_should_close(self._window)

    def close(self) -> None:
        self._listeners.focus_lost()
        glfw.destroy_window(self._window)

    def __enter__(self) -> "GlfwKeyWindow":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _key_name(code: int) -> str:
    return w3c_code_from_evdev(code) or f"code{code}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args()

    print("Click the 'Isaac Teleop keyboard' window and press keys (30 s).")

    keyboard = KeyboardSource(name="keyboard")
    session_config = TeleopSessionConfig(
        app_name="KeyboardPrinterExample", pipeline=keyboard
    )

    with (
        glfw_library(),
        GlfwKeyWindow("Isaac Teleop keyboard") as window,
        keyboard.attach(window),
        CloudXRLauncher.launch_context(args),
        TeleopSession(session_config) as session,
    ):
        start_time = time.time()
        while time.time() - start_time < 30.0 and window.poll():
            result = session.step()
            held_group = result["keyboard_held"]
            pressed_group = result["keyboard_pressed"]
            elapsed = session.get_elapsed_time()

            if held_group.is_none:
                print(f"[{elapsed:5.1f}s] (no keyboard)", end="\r", flush=True)
            else:
                held = np.flatnonzero(np.asarray(held_group[0]))
                pressed = np.flatnonzero(np.asarray(pressed_group[0]))
                for code in pressed:
                    print(f"\n[{elapsed:5.1f}s] {_key_name(int(code))} pressed")
                names = " ".join(_key_name(int(c)) for c in held) or "-"
                print(
                    f"[{elapsed:5.1f}s] Held: {names}" + " " * 20,
                    end="\r",
                    flush=True,
                )

            time.sleep(0.01)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
