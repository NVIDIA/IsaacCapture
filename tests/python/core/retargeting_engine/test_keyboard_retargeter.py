# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sim-free unit tests for KeyboardToSe3RelRetargeter, KeyboardGripperRetargeter, and
KeyboardToSe2Retargeter, exercised through KeyboardSource so a regression anywhere in the
schema -> source -> retargeter chain (a field rename, an index drift, a sign flip) fails
here, with no OpenXR device involved.
"""

import numpy as np
import pytest

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
from isaaccapture.schema import KeyboardOutput

# Evdev key codes (linux/input-event-codes.h), matching keyboard_plugin.cpp / KeyboardSource.
KEY_W, KEY_A, KEY_S, KEY_D, KEY_Q, KEY_E = 17, 30, 31, 32, 16, 18
KEY_K = 37  # gripper toggle
KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 103, 108, 105, 106
KEY_KP8, KEY_KP9 = 72, 73


def _keyboard_source():
    return KeyboardSource(name="keyboard")


def _run_source(src, pressed_keys: list[int] | None):
    """Feed raw pressed_keys (None = inactive device) through KeyboardSource.compute()."""
    keys = None if pressed_keys is None else KeyboardOutput(pressed_keys, True)

    input_spec = src.input_spec()
    tg = TensorGroup(input_spec["deviceio_keyboard"])
    tg[0] = keys

    outputs = {name: _make_output_group(gt) for name, gt in src.output_spec().items()}
    src.compute({"deviceio_keyboard": tg}, outputs)
    return outputs


class TestKeyboardToSe3RelRetargeter:
    def test_movement_key_produces_ee_delta(self):
        """W held -> KeyboardSource -> KeyboardToSe3RelRetargeter -> +X delta."""
        src = _keyboard_source()
        src_outputs = _run_source(src, [KEY_W])

        retargeter = KeyboardToSe3RelRetargeter(
            KeyboardToSe3RelRetargeterConfig(), name="se3"
        )
        out = {"ee_delta": _make_output_group(retargeter.output_spec()["ee_delta"])}
        retargeter.compute({"keyboard_all_keys": src_outputs["keyboard_all_keys"]}, out)

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
        retargeter.compute({"keyboard_all_keys": src_outputs["keyboard_all_keys"]}, out)

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
        se3.compute({"keyboard_all_keys": src_outputs["keyboard_all_keys"]}, se3_out)
        assert np.allclose(np.asarray(se3_out["ee_delta"][0]), 0.0)


class TestKeyboardGripperRetargeter:
    def test_gripper_toggles_on_rising_edge_only(self):
        """K press/release/press across three frames toggles exactly on each rising edge."""
        src = _keyboard_source()
        retargeter = KeyboardGripperRetargeter(name="gripper")

        def step(pressed_keys):
            src_outputs = _run_source(src, pressed_keys)
            out = {
                "gripper_command": _make_output_group(
                    retargeter.output_spec()["gripper_command"]
                )
            }
            retargeter.compute(
                {"keyboard_all_keys": src_outputs["keyboard_all_keys"]}, out
            )
            return float(out["gripper_command"][0])

        assert step([]) == pytest.approx(1.0)  # open (default)
        assert step([KEY_K]) == pytest.approx(-1.0)  # rising edge -> close
        assert step([KEY_K]) == pytest.approx(
            -1.0
        )  # held -> stays closed, no re-toggle
        assert step([]) == pytest.approx(-1.0)  # release -> stays closed
        assert step([KEY_K]) == pytest.approx(1.0)  # rising edge again -> open

    def test_inactive_device_yields_default_open(self):
        src = _keyboard_source()
        src_outputs = _run_source(src, None)

        gripper = KeyboardGripperRetargeter(name="gripper")
        gripper_out = {
            "gripper_command": _make_output_group(
                gripper.output_spec()["gripper_command"]
            )
        }
        gripper.compute(
            {"keyboard_all_keys": src_outputs["keyboard_all_keys"]}, gripper_out
        )
        assert float(gripper_out["gripper_command"][0]) == pytest.approx(
            1.0
        )  # default open

    def test_reset_does_not_toggle_gripper_while_k_is_held(self):
        """K held across a reset frame is not a rising edge and must not toggle the gripper."""
        src = _keyboard_source()
        retargeter = KeyboardGripperRetargeter(name="gripper")

        def step(pressed_keys, reset=False):
            src_outputs = _run_source(src, pressed_keys)
            out = {
                "gripper_command": _make_output_group(
                    retargeter.output_spec()["gripper_command"]
                )
            }
            context = ComputeContext(execution_events=ExecutionEvents(reset=reset))
            retargeter.compute(
                {"keyboard_all_keys": src_outputs["keyboard_all_keys"]}, out, context
            )
            return float(out["gripper_command"][0])

        assert step([KEY_K]) == pytest.approx(-1.0)  # rising edge -> close
        # Reset resets the gripper to open, but K is still held -- not a new rising
        # edge, so this must not immediately re-close it.
        assert step([KEY_K], reset=True) == pytest.approx(1.0)
        assert step([KEY_K]) == pytest.approx(1.0)  # still held -> stays open
        assert step([]) == pytest.approx(1.0)  # release
        assert step([KEY_K]) == pytest.approx(-1.0)  # genuine rising edge -> close

    def test_reset_with_inactive_device_preserves_prior_edge_state(self):
        """A reset frame with no keyboard data must not clobber _prev_k_pressed."""
        src = _keyboard_source()
        retargeter = KeyboardGripperRetargeter(name="gripper")

        def step(pressed_keys, reset=False):
            src_outputs = _run_source(src, pressed_keys)
            out = {
                "gripper_command": _make_output_group(
                    retargeter.output_spec()["gripper_command"]
                )
            }
            context = ComputeContext(execution_events=ExecutionEvents(reset=reset))
            retargeter.compute(
                {"keyboard_all_keys": src_outputs["keyboard_all_keys"]}, out, context
            )
            return float(out["gripper_command"][0])

        assert step([KEY_K]) == pytest.approx(-1.0)  # rising edge -> close
        # Reset while the device is inactive (keyboard.is_none) -- must not force
        # _prev_k_pressed to False, or the next frame (K still held) would be
        # misread as a fresh rising edge.
        assert step(None, reset=True) == pytest.approx(1.0)  # gripper still resets
        assert step([KEY_K]) == pytest.approx(1.0)  # still held -> no spurious toggle


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
        retargeter.compute({"keyboard_all_keys": src_outputs["keyboard_all_keys"]}, out)

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
        retargeter.compute({"keyboard_all_keys": src_outputs["keyboard_all_keys"]}, out)

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
        retargeter.compute({"keyboard_all_keys": src_outputs["keyboard_all_keys"]}, out)

        assert np.allclose(np.asarray(out["base_command"][0]), 0.0)
