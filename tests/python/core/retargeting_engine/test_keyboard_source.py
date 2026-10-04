# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the KeyboardSource DeviceIO converter.

Exercises the stateless converter from a raw ``KeyboardOutput`` FlatBuffer (constructed via
the real schema Python bindings) into the 256-entry keyboard_all_keys bitmap, with no OpenXR
device involved.
"""

import numpy as np

from isaaccapture.retargeting_engine.deviceio_source_nodes import KeyboardSource
from isaaccapture.retargeting_engine.interface.base_retargeter import _make_output_group
from isaaccapture.retargeting_engine.interface.tensor_group import TensorGroup
from isaaccapture.schema import KeyboardOutput

# Evdev key codes (linux/input-event-codes.h), matching keyboard_plugin.cpp / KeyboardSource.
KEY_W = 17
KEY_F1 = 59  # exercises "every key" coverage, not just the SE2/SE3 bindings


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


class TestKeyboardSource:
    def test_source_creates_real_tracker(self):
        src = _keyboard_source()
        tracker = src.get_tracker()
        assert tracker is not None
        assert tracker.get_name() == "KeyboardTracker"

    def test_movement_key_marks_bitmap(self):
        """W held shows up in the all-keys bitmap."""
        src = _keyboard_source()
        outputs = _run_source(src, [KEY_W])

        assert not outputs["keyboard_all_keys"].is_none
        bitmap = np.asarray(outputs["keyboard_all_keys"][0])
        assert bitmap[KEY_W] == 1

    def test_inactive_device_yields_none(self):
        src = _keyboard_source()
        outputs = _run_source(src, None)
        assert outputs["keyboard_all_keys"].is_none

    def test_all_keys_bitmap_covers_keys_outside_se3_subset(self):
        """F1 (not bound by any SE2/SE3 retargeter) still shows up in keyboard_all_keys."""
        src = _keyboard_source()
        outputs = _run_source(src, [KEY_F1])

        bitmap = np.asarray(outputs["keyboard_all_keys"][0])
        assert bitmap[KEY_F1] == 1
        assert bitmap.sum() == 1  # only F1 is set
