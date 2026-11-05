# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Clock exchange estimation: offset, uncertainty, window choice, rejection, drift, steps, staleness, reset."""

from __future__ import annotations

import dataclasses

import pytest

from egocap_web.clock_exchange import ClockEstimator, Exchange

MS = 1_000_000
S = 1_000_000_000


def make_exchange(
    t1: int,
    *,
    offset: int = 0,
    forward: int = MS,
    back: int = MS,
    processing: int = 0,
    rate_ppm: float = 0.0,
) -> Exchange:
    """An exchange between a local clock and a remote clock running ``offset`` (and ``rate_ppm``) ahead of it."""

    def remote(local_ns: int) -> int:
        return local_ns + offset + round(local_ns * rate_ppm * 1e-6)

    arrival = t1 + forward
    departure = arrival + processing
    return Exchange(t1=t1, t2=remote(arrival), t3=remote(departure), t4=departure + back)


def test_symmetric_delays_recover_offset_and_rtt_excludes_processing() -> None:
    ex = make_exchange(10 * S, offset=123_456_000, forward=3 * MS, back=3 * MS, processing=40 * MS)
    assert ex.offset_ns == 123_456_000
    assert ex.rtt_ns == 6 * MS

    estimator = ClockEstimator()
    assert estimator.add(ex) is None
    mapping = estimator.mapping(ex.t4)
    assert mapping is not None
    assert mapping.offset_ns == 123_456_000
    assert mapping.rtt_ns == 6 * MS
    assert mapping.uncertainty_ns == 3 * MS


@pytest.mark.parametrize("forward,back", [(1 * MS, 9 * MS), (9 * MS, 1 * MS), (2 * MS, 2 * MS)])
def test_asymmetric_delays_error_is_half_the_difference_within_uncertainty(forward: int, back: int) -> None:
    true_offset = 5 * MS
    ex = make_exchange(S, offset=true_offset, forward=forward, back=back, processing=7 * MS)
    estimator = ClockEstimator()
    estimator.add(ex)
    mapping = estimator.mapping(ex.t4)
    assert mapping is not None
    error = mapping.offset_ns - true_offset
    assert error == (forward - back) // 2
    assert abs(error) <= mapping.uncertainty_ns


def test_negative_offset_and_to_local_inverts_mapping() -> None:
    true_offset = -250 * MS
    ex = make_exchange(20 * S, offset=true_offset, forward=2 * MS, back=2 * MS, processing=MS)
    assert ex.offset_ns == true_offset

    estimator = ClockEstimator()
    estimator.add(ex)
    mapping = estimator.mapping(ex.t4)
    assert mapping is not None
    assert mapping.offset_ns == true_offset
    local = 21 * S
    assert mapping.to_local_ns(local + true_offset) == local


def test_high_rtt_outlier_does_not_replace_better_mapping() -> None:
    estimator = ClockEstimator()
    estimator.add(make_exchange(1 * S, offset=10 * MS, forward=MS, back=MS))
    good = estimator.mapping(2 * S)
    assert good is not None

    # Same clock, but a much slower exchange: it joins the window yet must not win.
    estimator.add(make_exchange(2 * S, offset=10 * MS, forward=20 * MS, back=20 * MS))
    assert estimator.mapping(3 * S) is good
    assert estimator.history == [good]


def test_best_of_window_is_the_lowest_rtt() -> None:
    estimator = ClockEstimator()
    for i, delay in enumerate([5 * MS, 3 * MS, 8 * MS, 2 * MS, 6 * MS]):
        estimator.add(make_exchange((i + 1) * S, offset=MS, forward=delay, back=delay))
    mapping = estimator.mapping(6 * S)
    assert mapping is not None
    assert mapping.rtt_ns == 4 * MS


def _malformed_and_slow() -> list[tuple[str, Exchange]]:
    return [
        ("slow", make_exchange(S, forward=200 * MS, back=100 * MS)),
        ("t4_before_t1", Exchange(t1=100, t2=0, t3=0, t4=50)),
        ("t3_before_t2", Exchange(t1=0, t2=100, t3=50, t4=1_000)),
        ("negative_rtt", Exchange(t1=0, t2=0, t3=900, t4=1_000 - 200)),
    ]


@pytest.mark.parametrize("name,bad", _malformed_and_slow(), ids=lambda v: v if isinstance(v, str) else "")
def test_rejected_exchange_reports_reason_and_leaves_mapping_unchanged(name: str, bad: Exchange) -> None:
    estimator = ClockEstimator(max_rtt_ns=250 * MS)
    estimator.add(make_exchange(S, offset=MS))
    before = estimator.mapping(2 * S)
    assert before is not None

    reason = estimator.add(bad)

    assert isinstance(reason, str) and reason
    assert estimator.mapping(2 * S) is before
    assert estimator.history == [before]
    assert not estimator.discontinuity


def test_drift_stays_continuous_and_tracks_within_uncertainty_plus_drift() -> None:
    ppm = 50.0
    forward = back = MS
    # Staleness is not under test: equal-rtt ties pick the oldest windowed exchange, which outlives the default.
    estimator = ClockEstimator(stale_after_ns=60 * S)
    for second in range(1, 31):
        ex = make_exchange(second * S, offset=0, forward=forward, back=back, rate_ppm=ppm)
        assert estimator.add(ex) is None
        mapping = estimator.mapping(ex.t4)
        assert mapping is not None
        # True offset at local time t is ppm * t; the mapping was taken at accepted_at_ns.
        true_offset = ppm * 1e-6 * ex.t4
        drift_since_best = ppm * 1e-6 * (ex.t4 - mapping.accepted_at_ns)
        assert abs(mapping.offset_ns - true_offset) <= mapping.uncertainty_ns + drift_since_best + 2
    assert not estimator.discontinuity
    assert {m.epoch for m in estimator.history} == {1}


def test_clock_step_sets_sticky_discontinuity_and_keeps_earlier_mapping() -> None:
    estimator = ClockEstimator()
    for second in range(1, 4):
        estimator.add(make_exchange(second * S, offset=0, forward=2 * MS, back=2 * MS))
    earlier = estimator.history[-1]
    snapshot = dataclasses.replace(earlier)
    assert not estimator.discontinuity

    # The remote clock jumps 50 ms; this exchange also has the lowest rtt so it becomes the best candidate.
    step = make_exchange(4 * S, offset=50 * MS, forward=MS, back=MS)
    assert estimator.add(step) is None

    assert estimator.discontinuity
    current = estimator.mapping(step.t4)
    assert current is not None
    assert current.epoch == earlier.epoch + 1
    assert current.offset_ns == 50 * MS
    assert estimator.history[-1] == current
    assert earlier in estimator.history
    assert earlier == snapshot
    with pytest.raises(dataclasses.FrozenInstanceError):
        earlier.offset_ns = 0  # type: ignore[misc]

    # Sticky: later well-behaved exchanges do not clear it.
    estimator.add(make_exchange(5 * S, offset=50 * MS, forward=MS, back=MS))
    assert estimator.discontinuity


def test_mapping_is_none_before_any_exchange_and_after_stale_window() -> None:
    estimator = ClockEstimator(stale_after_ns=5 * S)
    assert estimator.mapping(0) is None

    ex = make_exchange(S, offset=MS)
    estimator.add(ex)
    assert estimator.mapping(ex.t4) is not None
    assert estimator.mapping(ex.t4 + 5 * S) is not None
    assert estimator.mapping(ex.t4 + 5 * S + 1) is None


def test_reset_bumps_epoch_clears_mapping_and_keeps_history() -> None:
    estimator = ClockEstimator()
    first = make_exchange(S, offset=MS)
    estimator.add(first)
    adopted = list(estimator.history)
    assert adopted

    estimator.reset()

    assert estimator.mapping(first.t4) is None
    assert estimator.history == adopted

    # The same exchange after a reset is adopted afresh, in the new epoch.
    estimator.add(first)
    mapping = estimator.mapping(first.t4)
    assert mapping is not None
    assert mapping.epoch == adopted[0].epoch + 1
    assert estimator.history[: len(adopted)] == adopted
    assert len(estimator.history) == len(adopted) + 1
