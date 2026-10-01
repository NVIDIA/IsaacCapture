# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SpaceMouseSource DeviceIO converter.

Exercises the stateless converter from a raw ``SpaceMouseOutput`` FlatBuffer (constructed
via the real schema Python bindings) into translation/rotation axis arrays and a button-press
bitmap, with no OpenXR device involved.
"""

import numpy as np
import pytest

from isaaccapture.retargeting_engine.deviceio_source_nodes import SpaceMouseSource
from isaaccapture.retargeting_engine.interface.base_retargeter import _make_output_group
from isaaccapture.retargeting_engine.interface.tensor_group import TensorGroup
from isaaccapture.schema import SpaceMouseOutput

BUTTON_LEFT = 0
BUTTON_RIGHT = 1


def _spacemouse_source():
    return SpaceMouseSource(name="spacemouse")


def _run_source(
    src,
    translation: list[float] | None,
    rotation: list[float] | None = None,
    pressed_buttons: list[int] | None = None,
):
    """Feed raw axis/button state (None translation = inactive device) through SpaceMouseSource.compute()."""
    state = (
        None
        if translation is None
        else SpaceMouseOutput(
            translation, rotation or [0.0, 0.0, 0.0], pressed_buttons or [], True
        )
    )

    input_spec = src.input_spec()
    tg = TensorGroup(input_spec["deviceio_spacemouse"])
    tg[0] = state

    outputs = {name: _make_output_group(gt) for name, gt in src.output_spec().items()}
    src.compute({"deviceio_spacemouse": tg}, outputs)
    return outputs


class TestSpaceMouseSource:
    def test_source_creates_real_tracker(self):
        src = _spacemouse_source()
        tracker = src.get_tracker()
        assert tracker is not None
        assert tracker.get_name() == "SpaceMouseTracker"

    def test_translation_and_rotation_pass_through(self):
        src = _spacemouse_source()
        outputs = _run_source(
            src, translation=[0.1, 0.2, 0.3], rotation=[-0.1, -0.2, -0.3]
        )

        assert not outputs["spacemouse_translation"].is_none
        assert not outputs["spacemouse_rotation"].is_none
        translation = np.asarray(outputs["spacemouse_translation"][0])
        rotation = np.asarray(outputs["spacemouse_rotation"][0])
        assert translation == pytest.approx([0.1, 0.2, 0.3])
        assert rotation == pytest.approx([-0.1, -0.2, -0.3])

    def test_button_marks_bitmap(self):
        src = _spacemouse_source()
        outputs = _run_source(
            src, translation=[0.0, 0.0, 0.0], pressed_buttons=[BUTTON_LEFT]
        )

        assert not outputs["spacemouse_buttons"].is_none
        bitmap = np.asarray(outputs["spacemouse_buttons"][0])
        assert bitmap[BUTTON_LEFT] == 1
        assert bitmap.sum() == 1

    def test_multiple_buttons_mark_bitmap(self):
        src = _spacemouse_source()
        outputs = _run_source(
            src,
            translation=[0.0, 0.0, 0.0],
            pressed_buttons=[BUTTON_LEFT, BUTTON_RIGHT],
        )

        assert not outputs["spacemouse_buttons"].is_none
        bitmap = np.asarray(outputs["spacemouse_buttons"][0])
        assert bitmap[BUTTON_LEFT] == 1
        assert bitmap[BUTTON_RIGHT] == 1
        assert bitmap.sum() == 2

    def test_inactive_device_yields_none(self):
        src = _spacemouse_source()
        outputs = _run_source(src, translation=None)
        assert outputs["spacemouse_translation"].is_none
        assert outputs["spacemouse_rotation"].is_none
        assert outputs["spacemouse_buttons"].is_none
