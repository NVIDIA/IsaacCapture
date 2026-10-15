# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

import numpy as np
import pytest

from isaaccapture_examples.tactile_visualizer.app import TactileVisualizer
from isaaccapture_examples.tactile_visualizer.model import default_openxr_shape
from isaaccapture_examples.tactile_visualizer.sources import (
    HandSample,
    MockSource,
    SourceSnapshot,
)
from isaaccapture_examples.tactile_visualizer.urdf_config import (
    load_urdf_config,
    map_raw_joints,
)


def test_mock_source_publishes_complete_latest_sample():
    source = MockSource(num_taxels=11, rate_hz=100.0)
    source.start()
    deadline = time.monotonic() + 2.0
    sequence = 0
    snapshot = None
    try:
        while snapshot is None and time.monotonic() < deadline:
            sequence, snapshot, error = source.mailbox.latest_after(sequence)
            assert error is None
            time.sleep(0.005)
    finally:
        source.stop()

    assert snapshot is not None
    assert snapshot.hands["right"].positions.shape == (26, 3)
    assert snapshot.forces["right"].shape == (11, 3)
    assert set(snapshot.raw_joints["right"]) == set(snapshot.raw_joints["left"])


def test_malformed_live_snapshot_is_rejected_before_replacing_display():
    malformed = SourceSnapshot(
        received_at=1.0,
        hands={
            "right": HandSample(
                positions=np.zeros((26, 2), np.float32),
                orientations=np.zeros((26, 4), np.float32),
                valid=np.ones(26, bool),
            )
        },
        shapes={"right": default_openxr_shape()},
        forces={"right": np.zeros((0, 3), np.float32)},
    )

    with pytest.raises(ValueError, match="hand pose"):
        TactileVisualizer._validate_snapshot(malformed)


def test_urdf_config_and_joint_mapping(tmp_path):
    path = tmp_path / "urdf.yaml"
    path.write_text(
        "scale: 0.01\noffset_cm: [1, 2, 3]\njoint_map:\n  robot_index: index_MCP_FE\n",
        encoding="utf-8",
    )

    config = load_urdf_config(path)
    values = map_raw_joints(
        ["robot_index", "same_name", "missing"],
        {"index_MCP_FE": 1.5, "same_name": 2.5},
        config.joint_map,
    )

    assert config.scale == 0.01
    assert config.offset_cm == (1.0, 2.0, 3.0)
    np.testing.assert_array_equal(values, (1.5, 2.5, 0.0))


def test_urdf_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("units: centimeters\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown"):
        load_urdf_config(path)
