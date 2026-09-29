# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the in-process KeyboardSource and its provider API.

Covers the KeyboardOutput -> bitmap conversion (with ``KeyboardOutput`` built through the
real schema bindings), the KeyboardProvider bindings, ``attach`` against a fake
``KeyEventSource``, and ``find_sources``. No OpenXR device is involved; the tracker's
per-frame merge and MCAP round trip are covered by the C++ tests.
"""

import numpy as np
import pytest

from isaaccapture.deviceio_trackers import (
    KEYBOARD_KEY_CODE_COUNT,
    EvdevKeyCode,
    evdev_code_from_w3c,
)
from isaaccapture.retargeters import KeyboardGripperRetargeter
from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    FakeKeyEventSource,
    KeyboardSource,
    KeyEventSource,
    find_sources,
)
from isaaccapture.retargeting_engine.interface import OutputCombiner
from isaaccapture.retargeting_engine.interface.base_retargeter import _make_output_group
from isaaccapture.retargeting_engine.interface.tensor_group import TensorGroup
from isaaccapture.schema import KeyAction, KeyboardOutput, KeyEvent

# Evdev key codes (linux/input-event-codes.h).
KEY_W, KEY_A = 17, 30
KEY_F1 = 59  # exercises "every key" coverage, not just the SE2/SE3 bindings
KEY_FN = 464  # above 255


def _run_source(src, pressed_keys, events=()):
    """Feed a KeyboardOutput (None = no provider attached) through KeyboardSource.compute()."""
    keys = (
        None
        if pressed_keys is None
        else KeyboardOutput(pressed_keys, [KeyEvent(*e) for e in events])
    )

    tg = TensorGroup(src.input_spec()["deviceio_keyboard"])
    tg[0] = keys

    outputs = {name: _make_output_group(gt) for name, gt in src.output_spec().items()}
    src.compute({"deviceio_keyboard": tg}, outputs)
    return outputs


class TestKeyboardSourceConversion:
    def test_source_creates_real_tracker(self):
        tracker = KeyboardSource(name="keyboard").get_tracker()
        assert tracker.get_name() == "KeyboardTracker"

    def test_held_key_marks_held_bitmap(self):
        outputs = _run_source(KeyboardSource(name="keyboard"), [KEY_W])

        bitmap = np.asarray(outputs["keyboard_held"][0])
        assert bitmap[KEY_W] == 1
        assert bitmap.sum() == 1

    def test_held_bitmap_covers_keys_outside_se3_subset(self):
        outputs = _run_source(KeyboardSource(name="keyboard"), [KEY_F1])

        assert np.asarray(outputs["keyboard_held"][0])[KEY_F1] == 1

    def test_no_provider_yields_none(self):
        outputs = _run_source(KeyboardSource(name="keyboard"), None)

        assert outputs["keyboard_held"].is_none
        assert outputs["keyboard_pressed"].is_none

    def test_pressed_bitmap_counts_press_events_only(self):
        """A sub-frame tap (press + release) shows in keyboard_pressed but not as held."""
        events = [
            (1, KEY_W, KeyAction.PRESS),
            (2, KEY_W, KeyAction.RELEASE),
            (3, KEY_A, KeyAction.RELEASE),
        ]
        outputs = _run_source(KeyboardSource(name="keyboard"), [], events)

        pressed = np.asarray(outputs["keyboard_pressed"][0])
        held = np.asarray(outputs["keyboard_held"][0])
        assert pressed[KEY_W] == 1
        assert pressed[KEY_A] == 0
        assert held.sum() == 0

    def test_bitmaps_cover_every_evdev_key_code(self):
        src = KeyboardSource(name="keyboard")
        events = [(1, KEY_FN, KeyAction.PRESS), (2, KEY_FN, KeyAction.RELEASE)]
        outputs = _run_source(src, [KEYBOARD_KEY_CODE_COUNT - 1], events)

        held = np.asarray(outputs["keyboard_held"][0])
        pressed = np.asarray(outputs["keyboard_pressed"][0])
        assert KEYBOARD_KEY_CODE_COUNT == 768  # Linux KEY_CNT
        assert held.shape == pressed.shape == (KEYBOARD_KEY_CODE_COUNT,)
        assert held.nonzero()[0].tolist() == [KEYBOARD_KEY_CODE_COUNT - 1]
        assert pressed.nonzero()[0].tolist() == [KEY_FN]
        assert all(
            0 < int(m) < KEYBOARD_KEY_CODE_COUNT
            for m in EvdevKeyCode.__members__.values()
        )


class TestKeyboardProvider:
    def test_press_release_report_changes(self):
        provider = KeyboardSource(name="keyboard").create_provider("test")

        assert provider.key_down(KEY_W) is True
        assert provider.key_down(KEY_W) is False  # autorepeat: already held
        assert provider.key_up(KEY_W) is True
        assert provider.key_up(KEY_W) is False  # not held

    def test_w3c_codes(self):
        provider = KeyboardSource(name="keyboard").create_provider("test")

        assert provider.key_down("KeyW") is True
        assert provider.key_up(KEY_W) is True  # same key through either spelling
        assert provider.key_down("NotAKey") is False

    def test_codes_outside_the_evdev_range_are_rejected(self):
        provider = KeyboardSource(name="keyboard").create_provider("test")

        assert provider.key_down("Fn") is True
        assert provider.key_down(KEY_FN) is False  # same key, already held
        for code in (
            KEYBOARD_KEY_CODE_COUNT,
            65536,
            -1,
        ):  # beyond uint16 too: False, not TypeError
            assert provider.key_down(code) is False
            assert provider.key_up(code) is False
            assert provider.tap(code) is False

    def test_evdev_code_from_w3c(self):
        assert evdev_code_from_w3c("KeyW") == KEY_W
        assert evdev_code_from_w3c("ArrowUp") == 103
        assert evdev_code_from_w3c("Numpad8") == 72
        assert evdev_code_from_w3c("NotAKey") is None

    def test_evdev_key_code_is_generated_from_the_key_table(self):
        from isaaccapture.deviceio_trackers import w3c_code_from_evdev

        assert EvdevKeyCode.KeyW == KEY_W
        assert int(EvdevKeyCode.ArrowUp) == 103
        assert EvdevKeyCode(KEY_W).name == "KeyW"
        assert w3c_code_from_evdev(KEY_W) == "KeyW"
        bitmap = np.zeros(KEYBOARD_KEY_CODE_COUNT, dtype=np.uint8)
        bitmap[EvdevKeyCode.KeyW] = 1
        assert bitmap[KEY_W] == 1
        # every member is the evdev code of the W3C code it is named after
        for name, member in EvdevKeyCode.__members__.items():
            assert evdev_code_from_w3c(name) == int(member)

    def test_closed_provider_ignores_input(self):
        with KeyboardSource(name="keyboard").create_provider("test") as provider:
            assert provider.name == "test"
            assert not provider.closed
        assert provider.closed
        assert provider.key_down(KEY_W) is False


class _Handle:
    def __init__(self, on_close):
        self.close = on_close


class _FakeSurface:
    """Minimal KeyEventSource: records subscription and capture calls."""

    def __init__(self):
        self.listeners = []
        self.captured = []

    def add_key_listener(self, on_key, on_focus_lost):
        entry = (on_key, on_focus_lost)
        self.listeners.append(entry)
        return _Handle(lambda: self.listeners.remove(entry))

    def capture_keyboard(self):
        self.captured.append(True)
        return _Handle(lambda: self.captured.append(False))


class _RecordingProvider:
    def __init__(self):
        self.calls = []
        self.closed = False

    def key_down(self, code):
        self.calls.append(("down", code))

    def key_up(self, code):
        self.calls.append(("up", code))

    def release_all(self):
        self.calls.append(("release_all",))

    def close(self):
        self.closed = True


def _raise(message):
    """A callable that fails with ``message``."""

    def fail(*args):
        raise RuntimeError(message)

    return fail


def _attach_real(src, surface):
    """Attach with a real provider, returning it too; closing it releases its held keys."""
    providers = []
    create = src.create_provider
    src.create_provider = lambda name: providers.append(create(name)) or providers[-1]
    detach = src.attach(surface)
    return detach, providers[0]


class TestAttach:
    def test_fake_surface_satisfies_protocol(self):
        assert isinstance(_FakeSurface(), KeyEventSource)

    def test_attach_forwards_keys_and_focus_loss(self, monkeypatch):
        src = KeyboardSource(name="keyboard")
        provider = _RecordingProvider()
        monkeypatch.setattr(src, "create_provider", lambda name: provider)
        surface = _FakeSurface()

        detach = src.attach(surface)
        on_key, on_focus_lost = surface.listeners[0]
        on_key("KeyW", True)
        on_key("KeyW", False)
        on_focus_lost()

        assert provider.calls == [("down", "KeyW"), ("up", "KeyW"), ("release_all",)]
        assert surface.captured == [True]

        detach()
        detach()  # idempotent
        assert surface.listeners == []
        assert surface.captured == [True, False]
        assert provider.closed

    def test_attach_without_capture(self):
        surface = _FakeSurface()
        KeyboardSource(name="keyboard").attach(surface, capture=False)()

        assert surface.captured == []

    def test_failed_capture_leaves_nothing_attached(self, monkeypatch):
        src = KeyboardSource(name="keyboard")
        provider = _RecordingProvider()
        monkeypatch.setattr(src, "create_provider", lambda name: provider)
        surface = _FakeSurface()

        def refuse_capture():
            raise RuntimeError("cannot capture")

        surface.capture_keyboard = refuse_capture

        with pytest.raises(RuntimeError):
            src.attach(surface)
        assert surface.listeners == []
        assert provider.closed

    def test_failed_subscribe_closes_the_provider(self, monkeypatch):
        src = KeyboardSource(name="keyboard")
        provider = _RecordingProvider()
        monkeypatch.setattr(src, "create_provider", lambda name: provider)
        surface = _FakeSurface()

        def refuse_listener(on_key, on_focus_lost):
            raise RuntimeError("cannot subscribe")

        surface.add_key_listener = refuse_listener

        with pytest.raises(RuntimeError):
            src.attach(surface)
        assert provider.closed

    def test_detach_finishes_when_unsubscribe_raises(self):
        src = KeyboardSource(name="keyboard")
        surface = _FakeSurface()
        surface.add_key_listener = lambda on_key, on_focus_lost: _Handle(
            _raise("unsubscribe")
        )

        detach, provider = _attach_real(src, surface)
        provider.key_down("KeyW")
        with pytest.raises(RuntimeError, match="unsubscribe"):
            detach()
        detach()  # the remaining steps already ran: nothing left to do or raise

        assert surface.captured == [True, False]
        assert provider.closed
        assert provider.key_down("KeyA") is False

    def test_detach_closes_the_provider_when_capture_restore_raises(self):
        src = KeyboardSource(name="keyboard")
        surface = _FakeSurface()
        surface.capture_keyboard = lambda: _Handle(_raise("restore"))
        detach, provider = _attach_real(src, surface)
        surface.listeners[0][0]("KeyW", True)

        with pytest.raises(RuntimeError, match="restore"):
            detach()

        assert surface.listeners == []
        assert provider.closed

    def test_failed_rollback_still_closes_the_provider(self, monkeypatch):
        src = KeyboardSource(name="keyboard")
        provider = _RecordingProvider()
        monkeypatch.setattr(src, "create_provider", lambda name: provider)
        surface = _FakeSurface()
        surface.add_key_listener = lambda on_key, on_focus_lost: _Handle(
            _raise("unsubscribe")
        )
        surface.capture_keyboard = _raise("capture")

        with pytest.raises(RuntimeError, match="unsubscribe") as excinfo:
            src.attach(surface)

        assert (
            str(excinfo.value.__context__) == "capture"
        )  # the original failure is kept
        assert provider.closed

    def test_captures_from_several_attachments_are_independent(self):
        surface = FakeKeyEventSource()
        keyboard = KeyboardSource(name="keyboard")
        detach_first = keyboard.attach(surface)
        detach_second = KeyboardSource(name="other").attach(surface)

        detach_first()
        assert surface.captured  # the second attachment still holds its capture
        detach_second()
        assert not surface.captured


class TestFindSources:
    def test_finds_keyboard_source_in_pipeline(self):
        keyboard = KeyboardSource(name="keyboard")
        gripper = KeyboardGripperRetargeter(name="gripper").connect(
            {"keyboard_pressed": keyboard.output("keyboard_pressed")}
        )
        pipeline = OutputCombiner({"gripper": gripper.output("gripper_command")})

        assert find_sources(pipeline, KeyboardSource) == [keyboard]
