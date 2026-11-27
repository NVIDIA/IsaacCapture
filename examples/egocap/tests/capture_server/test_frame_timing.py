# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import pytest

from capture_server import frame_timing as ft


@pytest.mark.parametrize(
    ("age_s", "expected"),
    [
        (0.0, True),
        (0.5, True),
        (ft.MAX_FRAME_AGE_S, True),
        (ft.MAX_FRAME_AGE_S + 1e-9, False),
        (-ft.MAX_FRAME_LEAD_S, True),
        (-ft.MAX_FRAME_LEAD_S - 1e-9, False),
        (-10.0, False),
        (math.nan, False),
        (math.inf, False),
        (-math.inf, False),
    ],
)
def test_frame_age_is_plausible_at_its_edges(age_s, expected):
    assert ft.frame_age_is_plausible(age_s) is expected


def test_uncertainty_budget_defaults_when_unset(monkeypatch):
    monkeypatch.delenv(ft.BUDGET_ENV, raising=False)
    assert ft.uncertainty_budget_ns() == int(ft.DEFAULT_BUDGET_MS * 1e6)


@pytest.mark.parametrize("blank", ["", "   "])
def test_uncertainty_budget_defaults_when_blank(monkeypatch, blank):
    monkeypatch.setenv(ft.BUDGET_ENV, blank)
    assert ft.uncertainty_budget_ns() == int(ft.DEFAULT_BUDGET_MS * 1e6)


@pytest.mark.parametrize(("raw", "expected"), [("2", 2_000_000), (" 7.5 ", 7_500_000), ("0.5", 500_000)])
def test_uncertainty_budget_reads_milliseconds(monkeypatch, raw, expected):
    monkeypatch.setenv(ft.BUDGET_ENV, raw)
    assert ft.uncertainty_budget_ns() == expected


@pytest.mark.parametrize("raw", ["fast", "5ms", "0x10"])
def test_uncertainty_budget_rejects_a_non_number(monkeypatch, raw):
    monkeypatch.setenv(ft.BUDGET_ENV, raw)
    with pytest.raises(ValueError, match="must be a number"):
        ft.uncertainty_budget_ns()


@pytest.mark.parametrize("raw", ["0", "-1", "nan"])
def test_uncertainty_budget_rejects_a_non_positive_value(monkeypatch, raw):
    monkeypatch.setenv(ft.BUDGET_ENV, raw)
    with pytest.raises(ValueError, match="must be positive"):
        ft.uncertainty_budget_ns()
