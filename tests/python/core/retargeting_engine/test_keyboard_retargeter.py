# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sim-free unit tests for KeyboardToSe3RelRetargeter, KeyboardGripperRetargeter, and
KeyboardToSe2Retargeter, exercised through KeyboardSource so a regression anywhere in the
schema -> source -> retargeter chain (a field rename, an index drift, a sign flip) fails
here, with no OpenXR device involved.
"""

import numpy as np
import pytest

from isaaccapture.deviceio_trackers import EvdevKeyCode
from isaaccapture.retargeting_engine.deviceio_source_nodes import KeyboardSource
from isaaccapture.retargeting_engine.interface.base_retargeter import _make_output_group
from isaaccapture.retargeting_engine.interface.execution_events import ExecutionEvents
from isaaccapture.retargeting_engine.interface.retargeter_core_types import (
    ComputeContext,
)
from isaaccapture.retargeting_engine.interface.tensor_group import TensorGroup
from isaaccapture.retargeters import (
    KeyboardGripperRetargeter,
    KeyboardToSe2Retargeter,
    KeyboardToSe2RetargeterConfig,
    KeyboardToSe3RelRetargeter,
    KeyboardToSe3RelRetargeterConfig,
)
from isaaccapture.schema import KeyAction, KeyboardOutput, KeyEvent

K = EvdevKeyCode
KEY_W, KEY_A, KEY_S, KEY_D, KEY_Q, KEY_E = (
    K.KeyW,
    K.KeyA,
    K.KeyS,
    K.KeyD,
    K.KeyQ,
    K.KeyE,
)
KEY_K = K.KeyK  # gripper toggle
KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = (
    K.ArrowUp,
    K.ArrowDown,
    K.ArrowLeft,
    K.ArrowRight,
)
KEY_KP8, KEY_KP9 = K.Numpad8, K.Numpad9


def _keyboard_source():
    return KeyboardSource(name="keyboard")


def _run_source(src, held_keys: list[int] | None, pressed_events=()):
    """Feed held keys (None = no provider) plus press events through KeyboardSource.compute()."""
    keys = (
        None
        if held_keys is None
        else KeyboardOutput(
            held_keys,
            [KeyEvent(0, code, KeyAction.PRESS) for code in pressed_events],
        )
    )

    input_spec = src.input_spec()
    tg = TensorGroup(input_spec["deviceio_keyboard"])
    tg[0] = keys

    outputs = {name: _make_output_group(gt) for name, gt in src.output_spec().items()}
    src.compute({"deviceio_keyboard": tg}, outputs)
    return outputs


class _GripperHarness:
    """Steps KeyboardSource -> KeyboardGripperRetargeter frame by frame.

    ``step(held)`` derives this frame's press events from keys newly held since the previous
    frame; ``taps`` adds presses whose release landed in the same frame.
    """

    def __init__(self):
        self.src = _keyboard_source()
        self.retargeter = KeyboardGripperRetargeter(name="gripper")
        self._prev_held: set[int] = set()

    def step(self, held, taps=(), reset=False):
        if held is None:
            src_outputs = _run_source(self.src, None)
        else:
            presses = [k for k in held if k not in self._prev_held] + list(taps)
            src_outputs = _run_source(self.src, held, presses)
            self._prev_held = set(held)
        out = {
            "gripper_command": _make_output_group(
                self.retargeter.output_spec()["gripper_command"]
            )
        }
        context = ComputeContext(execution_events=ExecutionEvents(reset=reset))
        self.retargeter.compute(
            {"keyboard_pressed": src_outputs["keyboard_pressed"]}, out, context
        )
        return float(out["gripper_command"][0])


class TestKeyboardToSe3RelRetargeter:
    def test_movement_key_produces_ee_delta(self):
        """W held -> KeyboardSource -> KeyboardToSe3RelRetargeter -> +X delta."""
        src = _keyboard_source()
        src_outputs = _run_source(src, [KEY_W])

        retargeter = KeyboardToSe3RelRetargeter(
            KeyboardToSe3RelRetargeterConfig(), name="se3"
        )
        out = {"ee_delta": _make_output_group(retargeter.output_spec()["ee_delta"])}
        retargeter.compute({"keyboard_held": src_outputs["keyboard_held"]}, out)

        delta = np.asarray(out["ee_delta"][0])
        assert delta[0] == pytest.approx(0.4)  # default pos_sensitivity
        assert np.allclose(delta[1:], 0.0)

    def test_opposing_keys_combine(self):
        """W (+X) and Q (+Z) held together combine on independent axes."""
        src = _keyboard_source()
        src_outputs = _run_source(src, [KEY_W, KEY_Q])

        retargeter = KeyboardToSe3RelRetargeter(
            KeyboardToSe3RelRetargeterConfig(), name="se3"
        )
        out = {"ee_delta": _make_output_group(retargeter.output_spec()["ee_delta"])}
        retargeter.compute({"keyboard_held": src_outputs["keyboard_held"]}, out)

        delta = np.asarray(out["ee_delta"][0])
        assert delta[0] == pytest.approx(0.4)  # W: +X
        assert delta[2] == pytest.approx(0.4)  # Q: +Z
        assert delta[1] == pytest.approx(0.0)
        assert np.allclose(delta[3:], 0.0)  # no rotation keys held

    def test_inactive_device_yields_zero_delta(self):
        src = _keyboard_source()
        src_outputs = _run_source(src, None)

        se3 = KeyboardToSe3RelRetargeter(KeyboardToSe3RelRetargeterConfig(), name="se3")
        se3_out = {"ee_delta": _make_output_group(se3.output_spec()["ee_delta"])}
        se3.compute({"keyboard_held": src_outputs["keyboard_held"]}, se3_out)
        assert np.allclose(np.asarray(se3_out["ee_delta"][0]), 0.0)


class TestKeyboardGripperRetargeter:
    def test_gripper_toggles_on_press_only(self):
        """K press/hold/release/press toggles exactly on each press."""
        h = _GripperHarness()
        assert h.step([]) == pytest.approx(1.0)  # open (default)
        assert h.step([KEY_K]) == pytest.approx(-1.0)  # press -> close
        assert h.step([KEY_K]) == pytest.approx(-1.0)  # held -> no re-toggle
        assert h.step([]) == pytest.approx(-1.0)  # release -> stays closed
        assert h.step([KEY_K]) == pytest.approx(1.0)  # press again -> open

    def test_sub_frame_tap_toggles(self):
        """A tap pressed and released within one frame still toggles."""
        h = _GripperHarness()
        assert h.step([], taps=[KEY_K]) == pytest.approx(-1.0)

    def test_no_provider_keeps_state(self):
        h = _GripperHarness()
        assert h.step(None) == pytest.approx(1.0)  # default open
        assert h.step([KEY_K]) == pytest.approx(-1.0)
        assert h.step(None) == pytest.approx(-1.0)  # no data -> unchanged

    def test_reset_opens_and_held_key_does_not_retoggle(self):
        """K held across a reset is not a new press and must not re-close the gripper."""
        h = _GripperHarness()
        assert h.step([KEY_K]) == pytest.approx(-1.0)  # press -> close
        assert h.step([KEY_K], reset=True) == pytest.approx(1.0)  # reset -> open
        assert h.step([KEY_K]) == pytest.approx(1.0)  # still held -> stays open
        assert h.step([]) == pytest.approx(1.0)
        assert h.step([KEY_K]) == pytest.approx(-1.0)  # genuine press -> close

    def test_reset_with_no_provider(self):
        h = _GripperHarness()
        assert h.step([KEY_K]) == pytest.approx(-1.0)
        assert h.step(None, reset=True) == pytest.approx(1.0)  # reset still applies
        assert h.step([KEY_K]) == pytest.approx(1.0)  # still held -> no toggle


class TestKeyboardToSe2Retargeter:
    def test_arrow_and_numpad_keys_combine(self):
        """Arrow-Up (+v_x) and Numpad-9 (-omega_z) held together -> combined base_command."""
        src = _keyboard_source()
        src_outputs = _run_source(src, [KEY_UP, KEY_KP9])

        retargeter = KeyboardToSe2Retargeter(
            KeyboardToSe2RetargeterConfig(), name="se2"
        )
        out = {
            "base_command": _make_output_group(retargeter.output_spec()["base_command"])
        }
        retargeter.compute({"keyboard_held": src_outputs["keyboard_held"]}, out)

        velocity = np.asarray(out["base_command"][0])
        assert velocity[0] == pytest.approx(0.8)  # default v_x_sensitivity
        assert velocity[1] == pytest.approx(0.0)
        assert velocity[2] == pytest.approx(-1.0)  # default omega_z_sensitivity

    def test_aliased_keys_do_not_double_count(self):
        """Numpad-8 and Arrow-Up both map to +v_x; holding both is not double the sensitivity."""
        src = _keyboard_source()
        src_outputs = _run_source(src, [KEY_UP, KEY_KP8])

        retargeter = KeyboardToSe2Retargeter(
            KeyboardToSe2RetargeterConfig(), name="se2"
        )
        out = {
            "base_command": _make_output_group(retargeter.output_spec()["base_command"])
        }
        retargeter.compute({"keyboard_held": src_outputs["keyboard_held"]}, out)

        velocity = np.asarray(out["base_command"][0])
        assert velocity[0] == pytest.approx(0.8)
        assert np.allclose(velocity[1:], 0.0)

    def test_inactive_device_yields_zero_velocity(self):
        src = _keyboard_source()
        src_outputs = _run_source(src, None)

        retargeter = KeyboardToSe2Retargeter(
            KeyboardToSe2RetargeterConfig(), name="se2"
        )
        out = {
            "base_command": _make_output_group(retargeter.output_spec()["base_command"])
        }
        retargeter.compute({"keyboard_held": src_outputs["keyboard_held"]}, out)

        assert np.allclose(np.asarray(out["base_command"][0]), 0.0)
