# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from capture_server.network_check import MAX_TOTAL_S, NetworkCheckPolicy


def test_defaults_are_valid_and_documented_values():
    policy = NetworkCheckPolicy()
    assert (policy.startup_s, policy.window_s, policy.valid_for_s) == (15.0, 30.0, 120.0)
    assert policy.min_delivery_fraction == pytest.approx(0.86)
    assert policy.max_low_windows == 2
    assert policy.startup_s + policy.window_s <= MAX_TOTAL_S


@pytest.mark.parametrize(
    "name", ["startup_s", "window_s", "max_capture_gap_s", "max_arrival_gap_s", "max_age_growth_s", "valid_for_s"]
)
@pytest.mark.parametrize(
    "bad", [float("inf"), float("-inf"), float("nan"), True, "5", None, 0, 0.0, -1, -0.5], ids=repr
)
def test_a_duration_must_be_a_finite_positive_number(name, bad):
    with pytest.raises(ValueError, match=name):
        NetworkCheckPolicy(**{name: bad})


@pytest.mark.parametrize("bad", [0, 0.0, -0.1, 1.01, 2, float("nan"), float("inf"), True, "0.5", None], ids=repr)
def test_min_delivery_fraction_must_be_in_zero_exclusive_one_inclusive(bad):
    with pytest.raises(ValueError, match="min_delivery_fraction"):
        NetworkCheckPolicy(min_delivery_fraction=bad)


@pytest.mark.parametrize("ok", [0.01, 0.5, 1, 1.0])
def test_min_delivery_fraction_accepts_values_up_to_one(ok):
    assert NetworkCheckPolicy(min_delivery_fraction=ok).min_delivery_fraction == ok


@pytest.mark.parametrize("bad", [-1, 1.5, 2.0, True, "2", None], ids=repr)
def test_max_low_windows_must_be_a_nonnegative_integer(bad):
    with pytest.raises(ValueError, match="max_low_windows"):
        NetworkCheckPolicy(max_low_windows=bad)


def test_max_low_windows_zero_is_allowed():
    assert NetworkCheckPolicy(max_low_windows=0).max_low_windows == 0


def test_startup_plus_window_may_equal_but_not_exceed_the_total_limit():
    assert NetworkCheckPolicy(startup_s=100.0, window_s=MAX_TOTAL_S - 100.0)
    with pytest.raises(ValueError, match="must not exceed"):
        NetworkCheckPolicy(startup_s=100.0, window_s=MAX_TOTAL_S - 99.0)


def test_from_mapping_accepts_a_valid_mapping():
    policy = NetworkCheckPolicy.from_mapping(
        {"startup_s": 5, "window_s": 10.5, "max_low_windows": 1, "min_delivery_fraction": 0.9}
    )
    assert (policy.startup_s, policy.window_s, policy.max_low_windows, policy.min_delivery_fraction) == (
        5.0,
        10.5,
        1,
        0.9,
    )
    assert policy.valid_for_s == 120.0


def test_from_mapping_of_nothing_is_the_defaults():
    assert NetworkCheckPolicy.from_mapping({}) == NetworkCheckPolicy()


def test_from_mapping_rejects_an_unknown_key():
    with pytest.raises(ValueError, match="unknown key"):
        NetworkCheckPolicy.from_mapping({"window": 10})


@pytest.mark.parametrize("raw", [None, [], "window_s", 5], ids=repr)
def test_from_mapping_rejects_a_non_mapping(raw):
    with pytest.raises(ValueError, match="must be a mapping"):
        NetworkCheckPolicy.from_mapping(raw)


@pytest.mark.parametrize("bad", [1.0, 2.5, True, "1", None], ids=repr)
def test_from_mapping_rejects_a_non_integer_count(bad):
    with pytest.raises(ValueError, match="max_low_windows must be an integer"):
        NetworkCheckPolicy.from_mapping({"max_low_windows": bad})


@pytest.mark.parametrize("bad", [True, False, "5", None, float("nan"), float("inf")], ids=repr)
def test_from_mapping_rejects_a_non_finite_or_non_numeric_value(bad):
    with pytest.raises(ValueError, match="window_s"):
        NetworkCheckPolicy.from_mapping({"window_s": bad})


def test_from_mapping_applies_the_same_range_checks_as_the_constructor():
    with pytest.raises(ValueError, match="min_delivery_fraction"):
        NetworkCheckPolicy.from_mapping({"min_delivery_fraction": 1.5})
    with pytest.raises(ValueError, match="must be positive"):
        NetworkCheckPolicy.from_mapping({"window_s": 0})


def test_a_window_that_rounds_to_no_nanoseconds_is_rejected():
    with pytest.raises(ValueError, match="at least one nanosecond"):
        NetworkCheckPolicy(window_s=5e-10)
    assert NetworkCheckPolicy(window_s=1e-9).window_s == 1e-9


def test_from_mapping_rejects_unknown_keys_of_mixed_types_with_a_value_error():
    with pytest.raises(ValueError, match="unknown key"):
        NetworkCheckPolicy.from_mapping({"bogus": 1, 7: 2})
