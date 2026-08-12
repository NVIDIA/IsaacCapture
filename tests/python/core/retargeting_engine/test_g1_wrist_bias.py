# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the per-headset G1 wrist bias.

The values are pinned because they were fitted from hardware captures and
tuned against operator trials; a silent change to them moves the robot's
wrists. PICO being exactly zero is the load-bearing case: that path has been
tuned by people over a long time and must stay untouched.
"""

import math

import pytest

from isaacteleop.retargeters.G1 import WRIST_BIAS_RAD, wrist_bias_for
from isaacteleop.retargeting_engine.utilities import SKELETON_PROFILES


class TestPico:
    def test_pico_is_exactly_zero(self):
        """PICO teleop is long-tuned; biasing it would be a regression."""
        assert wrist_bias_for("pico") == ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    def test_pico_zero_is_exact_not_approximate(self):
        for side in wrist_bias_for("pico"):
            for channel in side:
                assert channel == 0.0


class TestQuest:
    @pytest.mark.parametrize(
        "side,channel,degrees",
        [
            (0, 0, +24.24),  # left roll
            (0, 1, +11.44),  # left pitch
            (0, 2, -29.88),  # left yaw
            (1, 0, -19.47),  # right roll
            (1, 1, +16.87),  # right pitch
            (1, 2, +28.13),  # right yaw
        ],
    )
    def test_matches_the_measured_degrees(self, side, channel, degrees):
        """Guards the fit recorded in the module's table."""
        got = math.degrees(wrist_bias_for("quest")[side][channel])
        assert got == pytest.approx(degrees, abs=0.02)

    def test_quest_is_not_zero(self):
        """A zeroed table would silently disable the correction."""
        assert any(c != 0.0 for side in wrist_bias_for("quest") for c in side)

    def test_roll_is_opposite_between_sides(self):
        """Roll is driven toward zero from opposite starting points."""
        left_roll, right_roll = (
            wrist_bias_for("quest")[0][0],
            wrist_bias_for("quest")[1][0],
        )
        assert left_roll > 0 > right_roll


class TestTable:
    def test_unknown_profile_returns_zero_rather_than_raising(self):
        """Unbiased is the pre-existing behaviour and is safe; another
        headset's bias is not."""
        for unknown in ("quest3", "", "QUEST", "meta", "pico4"):
            assert wrist_bias_for(unknown) == ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    def test_shape_is_two_sides_of_three_channels(self):
        for bias in WRIST_BIAS_RAD.values():
            assert len(bias) == 2
            assert all(len(side) == 3 for side in bias)

    def test_values_are_plausible_radians(self):
        """Catches a table accidentally written in degrees."""
        for bias in WRIST_BIAS_RAD.values():
            for side in bias:
                for channel in side:
                    assert abs(channel) < math.pi / 2

    def test_keys_track_the_skeleton_profiles(self):
        """The bias and the skeleton correction are keyed by the same headset;
        adding a headset to one without the other leaves it half-corrected."""
        assert set(WRIST_BIAS_RAD) == set(SKELETON_PROFILES)
