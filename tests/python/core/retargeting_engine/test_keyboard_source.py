# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the in-process KeyboardSource and its provider API.

Covers the KeyboardOutput -> bitmap conversion (with ``KeyboardOutput`` built through the
real schema bindings), the KeyboardProvider bindings, ``attach`` against a fake
``KeyEventSource``, and ``find_sources``. No OpenXR device is involved; the provider's
behavior, the per-frame merge and the MCAP round trip are covered by the C++ session tests.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from isaaccapture.deviceio_trackers import (
    KEYBOARD_KEY_CODE_COUNT,
    EvdevKeyCode,
    evdev_code_from_w3c,
    w3c_code_from_evdev,
)
from isaaccapture.retargeters import KeyboardGripperRetargeter
from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    FakeKeyEventSource,
    KeyboardSource,
    find_sources,
)
from isaaccapture.retargeting_engine.interface import OutputCombiner
from isaaccapture.retargeting_engine.interface.base_retargeter import _make_output_group
from isaaccapture.retargeting_engine.interface.tensor_group import TensorGroup
from isaaccapture.schema import KeyAction, KeyboardOutput, KeyEvent

KEY_W, KEY_A, KEY_F1 = EvdevKeyCode.KeyW, EvdevKeyCode.KeyA, EvdevKeyCode.F1
KEY_FN = EvdevKeyCode.Fn  # above 255


def _run_source(src, held_keys, events=()):
    """Feed a KeyboardOutput (None = no provider attached) through KeyboardSource.compute()."""
    keys = (
        None
        if held_keys is None
        else KeyboardOutput(held_keys, [KeyEvent(*e) for e in events])
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
        assert held.shape == pressed.shape == (KEYBOARD_KEY_CODE_COUNT,)
        assert held.nonzero()[0].tolist() == [KEYBOARD_KEY_CODE_COUNT - 1]
        assert pressed.nonzero()[0].tolist() == [KEY_FN]
        assert all(
            0 < int(m) < KEYBOARD_KEY_CODE_COUNT
            for m in EvdevKeyCode.__members__.values()
        )


class TestKeyboardProvider:
    """Binding smoke tests; the provider's behavior is covered by the C++ session tests."""

    def test_accepts_evdev_and_w3c_codes(self):
        with KeyboardSource(name="keyboard").create_provider() as provider:
            provider.key_down("KeyW")
            provider.key_up(KEY_W)
            provider.tap("KeyA")
            provider.key_down("NotAKey")  # no evdev equivalent: ignored
            provider.release_all()

    @pytest.mark.parametrize("code", [KEYBOARD_KEY_CODE_COUNT, 65536, -1])
    def test_codes_outside_the_evdev_range_are_ignored(self, code):
        """Beyond uint16 too: ignored, not a TypeError."""
        provider = KeyboardSource(name="keyboard").create_provider()
        provider.key_down(code)
        provider.key_up(code)
        provider.tap(code)

    def test_evdev_code_from_w3c(self):
        assert evdev_code_from_w3c("KeyW") == KEY_W
        assert evdev_code_from_w3c("ArrowUp") == EvdevKeyCode.ArrowUp
        assert evdev_code_from_w3c("NotAKey") is None

    def test_evdev_key_code_is_generated_from_the_key_table(self):
        assert EvdevKeyCode(int(KEY_W)).name == "KeyW"
        assert w3c_code_from_evdev(KEY_W) == "KeyW"
        # every member is the evdev code of the W3C code it is named after
        for name, member in EvdevKeyCode.__members__.items():
            assert evdev_code_from_w3c(name) == int(member)
            assert 0 < int(member) < KEYBOARD_KEY_CODE_COUNT

    def test_calls_after_close_are_ignored(self):
        with KeyboardSource(name="keyboard").create_provider() as provider:
            pass
        provider.key_down(KEY_W)
        provider.close()  # closing again does nothing


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


def _failing_handle(message):
    return SimpleNamespace(close=_raise(message))


@pytest.fixture
def recorded(monkeypatch):
    """A KeyboardSource whose providers are recorded, and the provider it hands out."""
    src = KeyboardSource(name="keyboard")
    provider = _RecordingProvider()
    monkeypatch.setattr(src, "create_provider", lambda: provider)
    return src, provider


class TestAttach:
    def test_attach_forwards_keys_and_focus_loss(self, recorded):
        src, provider = recorded
        surface = FakeKeyEventSource()

        attachment = src.attach(surface)
        surface.press("KeyW")
        surface.release("KeyW")
        surface.blur()

        assert provider.calls == [("down", "KeyW"), ("up", "KeyW"), ("release_all",)]
        assert surface.capture_history == [True]

        attachment.close()
        attachment.close()  # idempotent
        assert surface.listener_count == 0
        assert surface.capture_history == [True, False]
        assert provider.closed

    def test_attachment_detaches_on_leaving_a_with_block(self):
        surface = FakeKeyEventSource()
        with KeyboardSource(name="keyboard").attach(surface):
            assert surface.listener_count == 1
        assert surface.listener_count == 0
        assert surface.capture_history == [True, False]

    def test_attach_without_capture(self):
        surface = FakeKeyEventSource()
        KeyboardSource(name="keyboard").attach(surface, capture=False).close()

        assert surface.capture_history == []

    def test_failed_capture_leaves_nothing_attached(self, recorded):
        src, provider = recorded
        surface = FakeKeyEventSource()
        surface.capture_keyboard = _raise("cannot capture")

        with pytest.raises(RuntimeError):
            src.attach(surface)
        assert surface.listener_count == 0
        assert provider.closed

    def test_failed_subscribe_closes_the_provider(self, recorded):
        src, provider = recorded
        surface = FakeKeyEventSource()
        surface.add_key_listener = _raise("cannot subscribe")

        with pytest.raises(RuntimeError):
            src.attach(surface)
        assert provider.closed

    def test_detach_finishes_when_unsubscribe_raises(self, recorded):
        src, provider = recorded
        surface = FakeKeyEventSource()
        surface.add_key_listener = lambda on_key, on_focus_lost: _failing_handle(
            "unsubscribe"
        )

        attachment = src.attach(surface)
        with pytest.raises(RuntimeError, match="unsubscribe"):
            attachment.close()
        attachment.close()  # the remaining steps already ran: nothing left to do or raise

        assert surface.capture_history == [True, False]
        assert provider.closed

    def test_detach_closes_the_provider_when_capture_restore_raises(self, recorded):
        src, provider = recorded
        surface = FakeKeyEventSource()
        surface.capture_keyboard = lambda: _failing_handle("restore")
        attachment = src.attach(surface)

        with pytest.raises(RuntimeError, match="restore"):
            attachment.close()

        assert surface.listener_count == 0
        assert provider.closed

    def test_failed_rollback_still_closes_the_provider(self, recorded):
        src, provider = recorded
        surface = FakeKeyEventSource()
        surface.add_key_listener = lambda on_key, on_focus_lost: _failing_handle(
            "unsubscribe"
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
        first = KeyboardSource(name="keyboard").attach(surface)
        second = KeyboardSource(name="other").attach(surface)

        first.close()
        assert surface.captured  # the second attachment still holds its capture
        second.close()
        assert not surface.captured


class TestFindSources:
    def test_finds_keyboard_source_in_pipeline(self):
        keyboard = KeyboardSource(name="keyboard")
        gripper = KeyboardGripperRetargeter(name="gripper").connect(
            {"keyboard_pressed": keyboard.output("keyboard_pressed")}
        )
        pipeline = OutputCombiner({"gripper": gripper.output("gripper_command")})

        assert find_sources(pipeline, KeyboardSource) == [keyboard]
