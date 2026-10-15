# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from isaaccapture_examples.tactile_visualizer.model import (
    LatestMailbox,
    SessionController,
    SessionState,
    ShapeDocument,
    append_grid_taxels,
    atomic_write_within,
    default_openxr_shape,
    force_colors,
    project_points,
    resolve_current_joints,
    set_taxel_display_position,
    skin_taxels,
)


def test_default_shape_is_valid_openxr_layout():
    shape = default_openxr_shape(37)

    shape.validate()
    assert shape.num_joints == 26
    assert shape.num_taxels == 37
    assert int(shape.joint_parents[1]) == 0x7FFFFFFF


def test_default_new_shape_has_no_taxels():
    shape = default_openxr_shape()

    assert shape.num_joints == 26
    assert shape.num_taxels == 0


def test_append_taxels_uses_documented_grid_order_and_wrist_attachment():
    shape = default_openxr_shape()

    append_grid_taxels(shape, 6)

    np.testing.assert_allclose(shape.taxel_positions[0], (-0.04, -0.04, -0.04))
    np.testing.assert_allclose(shape.taxel_positions[1], (-0.04, -0.04, -0.02))
    np.testing.assert_allclose(shape.taxel_positions[5], (-0.04, -0.02, -0.04))
    np.testing.assert_array_equal(shape.taxel_joint_indices, 1)
    np.testing.assert_array_equal(shape.taxel_joint_weights[:, 0], 1.0)
    np.testing.assert_array_equal(shape.taxel_joint_weights[:, 1:], 0.0)


def test_shape_file_round_trip():
    shape = default_openxr_shape(13)

    restored = ShapeDocument.from_bytes(shape.to_bytes())

    for name, value in shape.arrays().items():
        np.testing.assert_array_equal(restored.arrays()[name], value)


def test_shape_file_rejects_malformed_bytes():
    with pytest.raises(ValueError, match="valid FlatBuffer"):
        ShapeDocument.from_bytes(b"not a shape")


def test_shape_validation_rejects_bad_weights():
    shape = default_openxr_shape(1)
    shape.taxel_joint_weights[0] = (0.5, 0.0, 0.0, 0.0)

    with pytest.raises(ValueError, match="sum to one"):
        shape.validate()


def test_shape_validation_rejects_self_parent_and_cycle():
    shape = default_openxr_shape()
    shape.joint_parents[3] = 3
    with pytest.raises(ValueError, match="cycle"):
        shape.validate()

    shape = default_openxr_shape()
    shape.joint_parents[1] = 2
    with pytest.raises(ValueError, match="cycle"):
        shape.validate()


def test_skinning_moves_taxel_with_attached_joint():
    shape = default_openxr_shape(1)
    current_positions, current_orientations = resolve_current_joints(
        shape, None, None, None
    )
    joint = int(shape.taxel_joint_indices[0, 0])
    current_positions[joint] += (0.01, -0.02, 0.03)

    positions, normals, tangents = skin_taxels(
        shape, current_positions, current_orientations
    )

    np.testing.assert_allclose(
        positions[0], shape.taxel_positions[0] + (0.01, -0.02, 0.03), atol=1e-6
    )
    np.testing.assert_allclose(normals[0], shape.taxel_normals[0], atol=1e-6)
    np.testing.assert_allclose(tangents[0], shape.taxel_tangents[0], atol=1e-6)


def test_missing_wrist_pose_preserves_bind_skeleton():
    shape = default_openxr_shape()
    live_positions = shape.joint_bind_positions + 1.0
    valid = np.ones(shape.num_joints, dtype=bool)
    valid[1] = False

    positions, orientations = resolve_current_joints(
        shape, live_positions, shape.joint_bind_orientations, valid
    )

    np.testing.assert_array_equal(positions, shape.joint_bind_positions)
    np.testing.assert_array_equal(orientations, shape.joint_bind_orientations)


def test_display_translation_inverts_skinning_and_left_mirror():
    shape = default_openxr_shape(1)
    current_positions, current_orientations = resolve_current_joints(
        shape, None, None, None
    )

    set_taxel_display_position(
        shape,
        current_positions,
        current_orientations,
        0,
        np.array([0.02, 0.03, 0.04]),
        "left",
    )

    np.testing.assert_allclose(shape.taxel_positions[0], (-0.02, 0.03, 0.04))


def test_force_color_gradient_and_missing_fallback():
    colors = force_colors(
        np.array([[0, 0, 0], [0, 0, 15], [0, 0, 30], [0, 0, 60]], np.float32),
        4,
    )

    np.testing.assert_array_equal(colors[0], (0, 0, 255))
    np.testing.assert_array_equal(colors[1], (128, 0, 128))
    np.testing.assert_array_equal(colors[2], (255, 0, 0))
    np.testing.assert_array_equal(colors[3], (255, 0, 0))
    assert force_colors(None, 3).shape == (3, 3)


def test_project_points_uses_viser_opencv_camera_convention():
    screen, visible = project_points(
        np.array([[0.0, 0.0, 1.0], [1.0, 0.0, -1.0]]),
        np.zeros(3),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.pi / 2,
        1.0,
    )

    np.testing.assert_allclose(screen[0], (0.5, 0.5))
    np.testing.assert_array_equal(visible, (True, False))


def test_state_machine_pause_disconnect_reconnect_and_resume():
    controller = SessionController(live_source=True, timeout_s=1.0)
    assert controller.state is SessionState.OFFLINE
    assert controller.message_received(9.0, ready=False) is SessionState.OFFLINE
    assert controller.message_received(10.0, ready=True) is SessionState.LIVE_PLAYING
    assert controller.pause() is SessionState.LIVE_PAUSED
    assert controller.tick(12.0) is SessionState.DISCONNECTED
    assert controller.editing_enabled
    assert controller.sample_received(12.1) is SessionState.LIVE_PAUSED
    assert controller.resume(12.2) is SessionState.LIVE_PLAYING
    assert controller.requirements_changed(12.3, ready=False) is SessionState.OFFLINE


def test_latest_mailbox_drops_unread_values():
    mailbox: LatestMailbox[str] = LatestMailbox()
    mailbox.put("old")
    mailbox.put("new")

    sequence, value, error = mailbox.latest_after(0)

    assert sequence == 2
    assert value == "new"
    assert error is None
    assert mailbox.latest_after(sequence) == (sequence, None, None)


def test_latest_mailbox_clears_transient_error_on_next_value():
    mailbox: LatestMailbox[str] = LatestMailbox()
    mailbox.fail(ValueError("bad frame"))
    mailbox.put("recovered")

    _, value, error = mailbox.latest_after(0)

    assert value == "recovered"
    assert error is None


def test_atomic_write_is_bounded_to_root(tmp_path):
    path = atomic_write_within(tmp_path, "nested/shape.bin", b"shape")

    assert path.read_bytes() == b"shape"
    with pytest.raises(ValueError, match="inside"):
        atomic_write_within(tmp_path, "../escape.bin", b"bad")
