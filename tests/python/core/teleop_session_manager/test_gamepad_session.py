# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A gamepad-only TeleopSession runs a real DeviceIOSession end to end, with no plugin.

The in-process GamepadTracker reads a Linux joystick-API device. A FIFO stands in for
/dev/input/jsN: the test writes ``js_event`` records into it and closes it to unplug the pad.
The tracker never touches the OpenXR session handles, so a placeholder OpenXR session stands in
for a runtime.
"""

import os
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from isaaccapture import oxr
from isaaccapture.retargeting_engine.deviceio_source_nodes import GamepadSource
from isaaccapture.teleop_session_manager import (
    SessionMode,
    TeleopSession,
    TeleopSessionConfig,
)

# The live gamepad reads the Linux joystick API; a FIFO stands in for the device.
pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="Linux joystick API"
)

JS_EVENT_BUTTON, JS_EVENT_AXIS = 0x01, 0x02
BUTTON_X, AXIS_LEFT_Y = 2, 1


class _PlaceholderOpenXRSession:
    """Non-null handles that only OpenXR-backed tracker impls would dereference."""

    def __init__(self, *args, **kwargs):
        pass

    def get_handles(self):
        return oxr.OpenXRSessionHandles(1, 1, 1, 1)

    def get_provider_snapshot(self):
        return SimpleNamespace(
            state="AVAILABLE",
            headset_state="CONNECTED",
            reason="NONE",
            result_code=None,
            error="",
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture(autouse=True)
def _placeholder_openxr(monkeypatch):
    monkeypatch.setattr(oxr, "OpenXRSession", _PlaceholderOpenXRSession)


class FakeJoystick:
    """A FIFO the test holds open for writing, standing in for /dev/input/jsN."""

    def __init__(self, path):
        self.path = str(path)
        os.mkfifo(self.path)
        # O_RDWR keeps a writer attached, so the tracker never sees EOF until unplug().
        self._fd = os.open(self.path, os.O_RDWR | os.O_NONBLOCK)

    def send(self, event_type, number, value):
        # struct js_event { __u32 time; __s16 value; __u8 type; __u8 number; }
        os.write(self._fd, struct.pack("=IhBB", 0, value, event_type, number))

    def unplug(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


@pytest.fixture
def joystick(tmp_path):
    pad = FakeJoystick(tmp_path / "js0")
    yield pad
    pad.unplug()


def _buttons(result):
    return np.flatnonzero(np.asarray(result["gamepad_buttons"][0])).tolist()


def _axes(result):
    return np.asarray(result["gamepad_axes"][0]).tolist()


def test_gamepad_session_follows_the_device(joystick):
    gamepad = GamepadSource(name="gamepad", device_path=joystick.path)
    config = TeleopSessionConfig(app_name="GamepadSession", pipeline=gamepad)

    with TeleopSession(config) as session:
        joystick.send(JS_EVENT_BUTTON, BUTTON_X, 1)
        joystick.send(JS_EVENT_AXIS, AXIS_LEFT_Y, -32767)
        result = session.step()
        assert _buttons(result) == [BUTTON_X]
        assert _axes(result)[AXIS_LEFT_Y] == -1.0

        joystick.send(JS_EVENT_BUTTON, BUTTON_X, 0)
        joystick.send(JS_EVENT_AXIS, AXIS_LEFT_Y, 0)
        result = session.step()
        assert _buttons(result) == []
        assert _axes(result)[AXIS_LEFT_Y] == 0.0

        joystick.unplug()
        result = session.step()
        assert result["gamepad_buttons"].is_none
        assert result["gamepad_axes"].is_none


def test_missing_gamepad_yields_none(tmp_path):
    gamepad = GamepadSource(name="gamepad", device_path=str(tmp_path / "absent"))
    config = TeleopSessionConfig(app_name="GamepadMissing", pipeline=gamepad)

    with TeleopSession(config) as session:
        result = session.step()
        assert result["gamepad_buttons"].is_none


def test_gamepad_session_records_and_replays(joystick, tmp_path):
    from isaaccapture.deviceio_session import McapRecordingConfig, McapReplayConfig

    mcap_path = str(tmp_path / "gamepad.mcap")
    config = TeleopSessionConfig(
        app_name="GamepadRecord",
        pipeline=GamepadSource(name="gamepad", device_path=joystick.path),
        mcap_config=McapRecordingConfig(mcap_path),
    )
    live = []
    with TeleopSession(config) as session:
        for button_value in (1, 0):
            joystick.send(JS_EVENT_BUTTON, BUTTON_X, button_value)
            live.append(_buttons(session.step()))

    replay_config = TeleopSessionConfig(
        app_name="GamepadReplay",
        pipeline=GamepadSource(name="gamepad"),
        mode=SessionMode.REPLAY,
        mcap_config=McapReplayConfig(mcap_path),
    )
    with TeleopSession(replay_config) as session:
        replayed = [_buttons(session.step()) for _ in live]

    assert live == [[BUTTON_X], []]
    assert replayed == live
