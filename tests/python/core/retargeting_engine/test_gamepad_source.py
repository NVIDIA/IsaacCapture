# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the GamepadSource DeviceIO converter.

Exercises the stateless converter from a raw ``GamepadOutput`` FlatBuffer (constructed via
the real schema Python bindings) into a button-press bitmap and a fixed-size axis array,
with no OpenXR device involved.
"""

import numpy as np

from isaaccapture.retargeting_engine.deviceio_source_nodes import GamepadSource
from isaaccapture.retargeting_engine.interface.base_retargeter import _make_output_group
from isaaccapture.retargeting_engine.interface.tensor_group import TensorGroup
from isaaccapture.schema import GamepadOutput

AXIS_LEFT_Y = 1
BUTTON_X = 2


def _gamepad_source():
    return GamepadSource(name="gamepad")


def _run_source(
    src, pressed_buttons: list[int] | None, axes: list[float] | None = None
):
    """Feed raw button/axis state (None = inactive device) through GamepadSource.compute()."""
    state = (
        None
        if pressed_buttons is None
        else GamepadOutput(pressed_buttons, axes or [], True)
    )

    input_spec = src.input_spec()
    tg = TensorGroup(input_spec["deviceio_gamepad"])
    tg[0] = state

    outputs = {name: _make_output_group(gt) for name, gt in src.output_spec().items()}
    src.compute({"deviceio_gamepad": tg}, outputs)
    return outputs


class TestGamepadSource:
    def test_source_creates_real_tracker(self):
        src = _gamepad_source()
        tracker = src.get_tracker()
        assert tracker is not None
        assert tracker.get_name() == "GamepadTracker"

    def test_button_marks_bitmap(self):
        src = _gamepad_source()
        outputs = _run_source(src, [BUTTON_X])

        assert not outputs["gamepad_buttons"].is_none
        bitmap = np.asarray(outputs["gamepad_buttons"][0])
        assert bitmap[BUTTON_X] == 1
        assert bitmap.sum() == 1

    def test_axes_pad_to_fixed_size(self):
        src = _gamepad_source()
        axes = [0.0] * 8
        axes[AXIS_LEFT_Y] = -1.0
        outputs = _run_source(src, [], axes=axes)

        assert not outputs["gamepad_axes"].is_none
        reported = np.asarray(outputs["gamepad_axes"][0])
        assert reported[AXIS_LEFT_Y] == -1.0

    def test_inactive_device_yields_none(self):
        src = _gamepad_source()
        outputs = _run_source(src, None)
        assert outputs["gamepad_buttons"].is_none
        assert outputs["gamepad_axes"].is_none
