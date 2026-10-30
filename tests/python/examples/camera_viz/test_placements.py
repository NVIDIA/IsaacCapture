# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lock-mode placement invariants.

The one that matters is facing: an OpenXR quad layer is single-sided, so a
plane rotated away from the viewer renders nothing at all. ``head`` shipped
with a 180-degree flip carried over from the old CameraPlane renderer and
showed a black feed.
"""

from __future__ import annotations

import math

import pytest

from isaaccapture_examples.camera_viz.placements import PlacementConfig, build
from isaaccapture_examples.camera_viz.placements._math import rotate_vec

MODES = ("world", "head", "gimbal", "lazy")

# Level, 45 deg right, 30 deg up, 45 deg right + 20 deg down.
HEAD_POSES = [
    (1.0, 0.0, 0.0, 0.0),
    (math.cos(math.radians(22.5)), 0.0, math.sin(math.radians(22.5)), 0.0),
    (math.cos(math.radians(15.0)), math.sin(math.radians(15.0)), 0.0, 0.0),
    (0.9186, -0.1622, 0.3564, 0.0629),
]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("head_q", HEAD_POSES)
def test_plane_faces_the_viewer(mode, head_q):
    """The plane's +z (its front) must point back toward the head."""
    head_pos = (0.0, 1.5, 0.0)
    placement = build(mode, PlacementConfig(distance=1.0)).update(head_pos, head_q)

    front = rotate_vec(placement.orientation, (0.0, 0.0, 1.0))
    to_head = tuple(head_pos[i] - placement.position[i] for i in range(3))
    norm = math.sqrt(_dot(to_head, to_head))
    assert norm > 0.1, "plane landed on top of the head"
    to_head = tuple(c / norm for c in to_head)

    # > 0 is "not facing away"; the strategies aim it squarely, so require
    # the front within 45 deg of the head direction.
    assert _dot(front, to_head) > math.cos(math.radians(45.0)), (
        f"{mode} faces away from the viewer: front={front} to_head={to_head}"
    )


@pytest.mark.parametrize("mode", MODES)
def test_plane_sits_the_configured_distance_away(mode):
    head_pos = (0.0, 1.5, 0.0)
    placement = build(mode, PlacementConfig(distance=1.25)).update(
        head_pos, (1.0, 0.0, 0.0, 0.0)
    )
    d = math.dist(placement.position, head_pos)
    assert d == pytest.approx(1.25, abs=1e-4)


def test_head_locked_tracks_pitch_but_the_yaw_only_modes_do_not():
    """head is the 6-DoF mode; the others deliberately stay level."""
    head_pos = (0.0, 1.5, 0.0)
    pitched = (math.cos(math.radians(15.0)), math.sin(math.radians(15.0)), 0.0, 0.0)

    head_y = build("head", PlacementConfig(distance=1.0)).update(head_pos, pitched)
    assert head_y.position[1] != pytest.approx(1.5, abs=1e-3)

    for mode in ("world", "gimbal", "lazy"):
        p = build(mode, PlacementConfig(distance=1.0)).update(head_pos, pitched)
        assert p.position[1] == pytest.approx(1.5, abs=1e-6)
