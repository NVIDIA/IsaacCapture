# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for identifying a headset from its OpenXR interaction profile.

The two profiles asserted against real hardware are marked below. Everything
else must resolve to None rather than a guess: a wrong answer here feeds a
robot the wrong skeleton correction, and permuted joint axes are not a
recoverable error.
"""

import pytest

from isaacteleop.cloudxr.headset import (
    HEADSET_BY_INTERACTION_PROFILE,
    identify_headset,
)

# Measured: PICO 4 Ultra and Quest 3 over CloudXR, both hands, both sessions.
MEASURED = {
    "/interaction_profiles/bytedance/pico4_controller": "pico",
    "/interaction_profiles/oculus/touch_controller": "quest",
}


class TestMeasuredHardware:
    @pytest.mark.parametrize("profile,expected", sorted(MEASURED.items()))
    def test_matches_observed_hardware(self, profile, expected):
        assert identify_headset(profile) == expected

    def test_the_two_headsets_are_distinguished(self):
        """The whole point: these must not collapse to the same answer."""
        answers = {identify_headset(p) for p in MEASURED}
        assert answers == {"pico", "quest"}
        assert None not in answers


class TestUnknownInput:
    @pytest.mark.parametrize(
        "value",
        [
            None,
            "",
            "/interaction_profiles/khr/simple_controller",
            "/interaction_profiles/htc/vive_controller",
            "/interaction_profiles/valve/index_controller",
            "quest3",
            "pico",
            "bytedance/pico4_controller",
            "/INTERACTION_PROFILES/BYTEDANCE/PICO4_CONTROLLER",
            "  /interaction_profiles/bytedance/pico4_controller  ",
        ],
    )
    def test_unknown_returns_none(self, value):
        """Includes 'quest3', the CloudXR device-profile string that names the
        wrong vendor on a PICO session, and near-misses that must not fuzzy-match."""
        assert identify_headset(value) is None

    def test_empty_is_not_an_error(self):
        """Nothing is bound until actions sync, so empty is an expected state."""
        assert identify_headset("") is None


class TestTable:
    def test_every_value_is_a_known_headset(self):
        assert set(HEADSET_BY_INTERACTION_PROFILE.values()) <= {"pico", "quest"}

    def test_keys_are_full_openxr_paths(self):
        for profile in HEADSET_BY_INTERACTION_PROFILE:
            assert profile.startswith("/interaction_profiles/")

    def test_vendor_prefix_agrees_with_headset(self):
        """A bytedance path must never map to quest, or vice versa."""
        for profile, headset in HEADSET_BY_INTERACTION_PROFILE.items():
            if "/bytedance/" in profile:
                assert headset == "pico", profile
            else:
                assert headset == "quest", profile

    def test_no_duplicate_or_empty_keys(self):
        assert all(HEADSET_BY_INTERACTION_PROFILE)
        assert len(HEADSET_BY_INTERACTION_PROFILE) == len(
            set(HEADSET_BY_INTERACTION_PROFILE)
        )
