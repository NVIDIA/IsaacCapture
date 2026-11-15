# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the clocks in isaaccapture_examples.agentic_remote.robot_app.clock."""

from __future__ import annotations

import math
import threading
import time

import pytest
from isaaccapture_examples.agentic_remote.robot_app.clock import (
    Clock,
    FakeClock,
    SystemClock,
)

BAD_DURATIONS = [-0.001, -1.0, math.nan, math.inf, -math.inf]


def test_fake_clock_defaults():
    fake = FakeClock()
    assert fake.monotonic() == 1000.0
    assert fake.wall() == 1000.0 + 1.7e9


def test_clock_fixture_is_a_fake_clock_at_its_defaults(clock):
    assert isinstance(clock, FakeClock)
    assert clock.monotonic() == 1000.0


def test_fake_clock_start_and_wall_offset():
    fake = FakeClock(start=5.0, wall_offset=100.0)
    assert fake.monotonic() == 5.0
    assert fake.wall() == 105.0


def test_fake_clock_stands_still_between_calls():
    fake = FakeClock()
    assert fake.monotonic() == fake.monotonic()
    assert fake.wall() == fake.wall()


def test_fake_clock_advance_moves_both_clocks():
    fake = FakeClock(start=10.0, wall_offset=100.0)
    fake.advance(2.5)
    assert fake.monotonic() == 12.5
    assert fake.wall() == 112.5


def test_fake_clock_sleep_advances_and_never_really_sleeps(monkeypatch):
    def no_real_sleep(seconds):
        raise AssertionError(f"time.sleep({seconds}) was called")

    monkeypatch.setattr(time, "sleep", no_real_sleep)
    fake = FakeClock(start=10.0)
    fake.sleep(3600.0)
    assert fake.monotonic() == 3610.0


def test_fake_clock_accepts_a_zero_sleep():
    fake = FakeClock(start=10.0)
    fake.sleep(0.0)
    fake.advance(0)
    assert fake.monotonic() == 10.0


@pytest.mark.parametrize("duration", BAD_DURATIONS)
def test_fake_clock_rejects_a_bad_sleep(duration):
    fake = FakeClock(start=10.0)
    with pytest.raises(ValueError, match=r"finite and not negative"):
        fake.sleep(duration)
    assert fake.monotonic() == 10.0


@pytest.mark.parametrize("duration", BAD_DURATIONS)
def test_fake_clock_rejects_a_bad_advance(duration):
    fake = FakeClock(start=10.0)
    with pytest.raises(ValueError, match=r"finite and not negative"):
        fake.advance(duration)
    assert fake.monotonic() == 10.0


@pytest.mark.parametrize(
    "kwargs", [{"start": math.nan}, {"start": math.inf}, {"wall_offset": math.nan}]
)
def test_fake_clock_rejects_a_non_finite_start(kwargs):
    with pytest.raises(ValueError, match=r"must be finite"):
        FakeClock(**kwargs)


def test_fake_clock_loses_no_sleep_across_threads():
    fake = FakeClock(start=0.0)
    threads_count, sleeps_each = 8, 2000
    ready = threading.Barrier(threads_count)

    def sleeper():
        ready.wait()
        for _ in range(sleeps_each):
            fake.sleep(0.25)

    threads = [threading.Thread(target=sleeper) for _ in range(threads_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    # 0.25 is exact in binary, so the sum is exact whatever the order.
    assert fake.monotonic() == threads_count * sleeps_each * 0.25


def test_both_clocks_satisfy_the_clock_protocol():
    assert isinstance(FakeClock(), Clock)
    assert isinstance(SystemClock(), Clock)
    assert not isinstance(object(), Clock)


def test_system_clock_reads_the_machine_clocks(monkeypatch):
    monkeypatch.setattr(time, "monotonic", lambda: 123.5)
    monkeypatch.setattr(time, "time", lambda: 1.7e9)
    system = SystemClock()
    assert system.monotonic() == 123.5
    assert system.wall() == 1.7e9


def test_system_clock_sleep_passes_the_duration_on(monkeypatch):
    slept = []
    monkeypatch.setattr(time, "sleep", slept.append)
    SystemClock().sleep(0.02)
    SystemClock().sleep(0)
    assert slept == [0.02, 0.0]


@pytest.mark.parametrize("duration", BAD_DURATIONS)
def test_system_clock_rejects_a_bad_sleep(monkeypatch, duration):
    slept = []
    monkeypatch.setattr(time, "sleep", slept.append)
    with pytest.raises(ValueError, match=r"finite and not negative"):
        SystemClock().sleep(duration)
    assert slept == []
