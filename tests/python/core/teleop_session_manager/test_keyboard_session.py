# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A keyboard-only TeleopSession runs a real DeviceIOSession end to end.

The in-process KeyboardTracker never touches the OpenXR session handles, so a placeholder
OpenXR session stands in for a runtime. Input comes from FakeKeyEventSource, which follows the
KeyEventSource contract (focus loss and host UI keyboard capture release held keys; press-only
surfaces report taps).
"""

from types import SimpleNamespace

import numpy as np
import pytest

from isaaccapture import oxr
from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    FakeKeyEventSource,
    KeyboardSource,
)
from isaaccapture.teleop_session_manager import (
    SessionMode,
    TeleopSession,
    TeleopSessionConfig,
)

KEY_W, KEY_K = 17, 37


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


def _held(result):
    return np.flatnonzero(np.asarray(result["keyboard_held"][0])).tolist()


def _pressed(result):
    return np.flatnonzero(np.asarray(result["keyboard_pressed"][0])).tolist()


def test_keyboard_session_follows_the_surface():
    keyboard = KeyboardSource(name="keyboard")
    surface = FakeKeyEventSource()
    detach = keyboard.attach(surface)
    config = TeleopSessionConfig(app_name="KeyboardSession", pipeline=keyboard)

    with TeleopSession(config) as session:
        surface.press("KeyW")
        result = session.step()
        assert _held(result) == [KEY_W]
        assert _pressed(result) == [KEY_W]

        surface.tap("KeyK")  # press-only surface: pressed this frame, never held
        result = session.step()
        assert _held(result) == [KEY_W]
        assert _pressed(result) == [KEY_K]

        surface.begin_text_input()  # UI takes the keyboard: held keys are released
        surface.press("KeyK")  # typed into the text field: not forwarded
        result = session.step()
        assert _held(result) == []
        assert _pressed(result) == []

        surface.end_text_input()
        surface.press("KeyW")
        surface.blur()  # focus loss releases it again
        result = session.step()
        assert _held(result) == []
        assert _pressed(result) == [KEY_W]

    detach()
    assert surface.listener_count == 0
    assert surface.capture_history == [True, False]


def test_keyboard_session_records_and_replays(tmp_path):
    from isaaccapture.deviceio_session import McapRecordingConfig, McapReplayConfig

    mcap_path = str(tmp_path / "keyboard.mcap")
    keyboard = KeyboardSource(name="keyboard")
    surface = FakeKeyEventSource()
    keyboard.attach(surface)

    live = []
    config = TeleopSessionConfig(
        app_name="KeyboardRecord",
        pipeline=keyboard,
        mcap_config=McapRecordingConfig(mcap_path),
    )
    with TeleopSession(config) as session:
        for action in (
            lambda: surface.press("KeyW"),
            lambda: surface.tap("KeyK"),
            surface.blur,
        ):
            action()
            result = session.step()
            live.append((_held(result), _pressed(result)))

    replay_config = TeleopSessionConfig(
        app_name="KeyboardReplay",
        pipeline=KeyboardSource(name="keyboard"),
        mode=SessionMode.REPLAY,
        mcap_config=McapReplayConfig(mcap_path),
    )
    replayed = []
    with TeleopSession(replay_config) as session:
        for _ in live:
            result = session.step()
            replayed.append((_held(result), _pressed(result)))

    assert live == [([KEY_W], [KEY_W]), ([KEY_W], [KEY_K]), ([], [])]
    assert replayed == live
