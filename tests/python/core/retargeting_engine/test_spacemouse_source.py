# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SpaceMouseSource DeviceIO converter.

Feeds SpaceMouseOutput payloads built through the real schema bindings into the converter, with no
plugin or OpenXR runtime involved.
"""

import numpy as np
import pytest

from isaaccapture.retargeting_engine.deviceio_source_nodes import SpaceMouseSource
from isaaccapture.retargeting_engine.interface.base_retargeter import _make_output_group
from isaaccapture.retargeting_engine.interface.tensor_group import TensorGroup
from isaaccapture.schema import Point, SpaceMouseOutput


def _run(state):
    src = SpaceMouseSource(name="spacemouse")
    tg = TensorGroup(src.input_spec()["deviceio_spacemouse"])
    tg[0] = state
    outputs = {name: _make_output_group(gt) for name, gt in src.output_spec().items()}
    src.compute({"deviceio_spacemouse": tg}, outputs)
    return outputs


def test_source_reads_the_generated_tracker():
    tracker = SpaceMouseSource(name="spacemouse").get_tracker()
    assert tracker.get_name() == "SpaceMouseTracker"


def test_axes_pass_through():
    outputs = _run(
        SpaceMouseOutput(Point(0.1, 0.2, 0.3), Point(-0.1, -0.2, -0.3), [], True)
    )

    assert np.asarray(outputs["spacemouse_translation"][0]) == pytest.approx(
        [0.1, 0.2, 0.3]
    )
    assert np.asarray(outputs["spacemouse_rotation"][0]) == pytest.approx(
        [-0.1, -0.2, -0.3]
    )


def test_buttons_fill_the_bitmap_and_extra_buttons_are_dropped():
    buttons = (
        [0, 1] + [0] * 6 + [1] * 8
    )  # a 16-button device holding buttons 1 and 8..15
    bitmap = np.asarray(
        _run(SpaceMouseOutput(Point(), Point(), buttons, True))["spacemouse_buttons"][0]
    )

    assert bitmap.tolist() == [0, 1, 0, 0, 0, 0, 0, 0]


def test_a_device_with_fewer_buttons_pads_the_bitmap():
    bitmap = np.asarray(
        _run(SpaceMouseOutput(Point(), Point(), [1, 1], True))["spacemouse_buttons"][0]
    )

    assert bitmap.tolist() == [1, 1, 0, 0, 0, 0, 0, 0]


@pytest.mark.parametrize(
    "state",
    [None, SpaceMouseOutput(Point(1.0, 1.0, 1.0), Point(), [1], False)],
    ids=["plugin-not-pushing", "no-device-open"],
)
def test_no_device_yields_absent_outputs(state):
    outputs = _run(state)

    assert outputs["spacemouse_translation"].is_none
    assert outputs["spacemouse_rotation"].is_none
    assert outputs["spacemouse_buttons"].is_none
