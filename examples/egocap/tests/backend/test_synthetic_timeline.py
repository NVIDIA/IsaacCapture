# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""One synthetic event timeline recorded through a camera path and a tracking path, each on its own clock.

Truth events happen at known PC monotonic times. A camera and a glove-like tracking source observe every event
on their own device clock (own offset and drift) and deliver it after their own delay. Each device's clock is
mapped onto the PC's by the real ``ClockEstimator`` fed with simulated four-timestamp exchanges. The camera goes
through the real ``LiveKitRecordingFeed``; tracking samples are mapped with the same estimator API.

This proves plumbing and uncertainty accounting on synthetic clocks, not physical sensor alignment.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pytest

from capture_server import protocol as p
from egocap_web.clock_exchange import ClockEstimator, ClockMapping, Exchange
from egocap_web.livekit_receiver import ReceivedFrame
from egocap_web.livekit_recording import LiveKitRecordingFeed

S = 1_000_000_000
BASE_S = 5000.0  # the PC's monotonic time of the first event
EVENT_PERIOD_S = 0.25
EVENTS = 60
#: Frame timestamps are whole microseconds, so each stamp is up to 0.5 us off; allow a little more.
QUANTIZATION_S = 3e-6
EXCHANGE_PERIOD_S = 1.0
TRACK_LOST = "cam: the LiveKit track was lost"

#: (df, db): one-way delays of one exchange, in ns.
Link = Callable[[random.Random], tuple[int, int]]


def symmetric_jitter(rng: random.Random) -> tuple[int, int]:
    """Each direction 4 ms plus 0-6 ms of independent queueing, so asymmetry arises per exchange."""
    return 4_000_000 + int(rng.uniform(0, 6e6)), 4_000_000 + int(rng.uniform(0, 6e6))


def forward_heavy(rng: random.Random) -> tuple[int, int]:
    """All of the delay on the way out: the offset error reaches half the round trip."""
    return 30_000_000, 1_000


def backward_heavy(rng: random.Random) -> tuple[int, int]:
    return 1_000, 20_000_000


class DeviceClock:
    """A device clock as piecewise ``offset + t * (1 + ppm)``; a segment boundary is a step or a reboot."""

    def __init__(self, offset_ns: int, ppm: float) -> None:
        self.segments: list[tuple[int, int, float]] = [(0, offset_ns, ppm)]

    def add_segment(self, from_s: float, offset_ns: int, ppm: float) -> None:
        self.segments.append((round(from_s * S), offset_ns, ppm))

    def segment_at(self, pc_ns: int) -> tuple[int, int, float]:
        return [s for s in self.segments if s[0] <= pc_ns][-1]

    def read_ns(self, pc_ns: int) -> int:
        _, offset, ppm = self.segment_at(pc_ns)
        return round(offset + pc_ns + pc_ns * ppm * 1e-6)

    def ppm_at(self, pc_s: float) -> float:
        return self.segment_at(round(pc_s * S))[2]


@dataclass
class Device:
    """A device clock, its responder, and the real estimator fed with the exchanges that completed so far."""

    clock: DeviceClock
    link: Link
    rng: random.Random
    start_s: float
    end_s: float
    outage: tuple[float, float] | None = None
    resets_s: tuple[float, ...] = ()
    estimator: ClockEstimator = field(default_factory=ClockEstimator)
    now_ns: int = 0

    def __post_init__(self) -> None:
        # (completion time, order, reset-or-exchange); a reset is followed at once by an exchange so the
        # restarted publisher has a mapping again within milliseconds.
        self._items: list[tuple[int, int, Exchange | None]] = []
        times = [
            self.start_s + i * EXCHANGE_PERIOD_S
            for i in range(round((self.end_s - self.start_s) / EXCHANGE_PERIOD_S))
        ]
        times += [r + 0.01 for r in self.resets_s]
        for t in times:
            if self.outage is not None and self.outage[0] <= t < self.outage[1]:
                continue
            exchange = self._exchange(round(t * S))
            self._items.append((exchange.t4, 1, exchange))
        self._items += [(round(r * S), 0, None) for r in self.resets_s]
        self._items.sort(key=lambda item: item[:2])

    def _exchange(self, t1: int, processing_ns: int = 50_000) -> Exchange:
        """One exchange whose responder stamps t2/t3 on the device clock at the true arrival and send times."""
        df, db = self.link(self.rng)
        t2 = self.clock.read_ns(t1 + df)
        t3 = self.clock.read_ns(t1 + df + processing_ns)
        return Exchange(t1=t1, t2=t2, t3=t3, t4=t1 + df + processing_ns + db)

    def advance(self, now_ns: int) -> None:
        """Apply everything that has completed by ``now_ns`` on the PC clock."""
        while self._items and self._items[0][0] <= now_ns:
            _, _, exchange = self._items.pop(0)
            if exchange is None:
                self.estimator.reset()
            else:
                assert self.estimator.add(exchange) is None
        self.now_ns = now_ns

    def mapping(self) -> ClockMapping | None:
        return self.estimator.mapping(self.now_ns)


class SimClockSync:
    """The feed's ``receiver.clock_sync``: the estimator's mapping, remembering each one it handed out."""

    def __init__(self, device: Device) -> None:
        self._device = device
        self.used: list[ClockMapping | None] = []

    def mapping(self) -> ClockMapping | None:
        mapping = self._device.mapping()
        self.used.append(mapping)
        return mapping

    @property
    def discontinuity(self) -> bool:
        return self._device.estimator.discontinuity


class SimReceiver:
    """Serves frames in arrival order; the camera's clock-sync exchanges advance to each frame's arrival."""

    def __init__(self, frames: list[ReceivedFrame], device: Device) -> None:
        self._frames = sorted(frames, key=lambda f: f.receive_time_ns)
        self._device = device
        self.clock_sync = SimClockSync(device)

    async def next_frame_or_lost(self, stream_id: int) -> ReceivedFrame | None:
        if not self._frames:
            return None
        frame = self._frames.pop(0)
        self._device.advance(frame.receive_time_ns)
        return frame


class ListIngress:
    def __init__(self) -> None:
        self.offered: list = []

    def offer(self, item) -> bool:
        self.offered.append(item)
        return True


def event_time_s(k: int) -> float:
    return BASE_S + k * EVENT_PERIOD_S


@dataclass
class Recorded:
    """One recorded observation of event ``k``: the time written down and the mapping it was made under."""

    k: int
    time_s: float
    mapping: ClockMapping
    ppm: float
    discontinuity: bool = False

    @property
    def truth_s(self) -> float:
        return event_time_s(self.k)

    @property
    def error_s(self) -> float:
        return self.time_s - self.truth_s

    @property
    def tolerance_s(self) -> float:
        """Own uncertainty plus the drift accumulated since the exchange behind the mapping."""
        drift_s = abs(self.ppm) * 1e-6 * abs(self.truth_s - self.mapping.accepted_at_ns / S)
        return self.mapping.uncertainty_ns / S + drift_s + QUANTIZATION_S


@dataclass
class Result:
    camera: dict[int, Recorded]
    glove: dict[int, Recorded]
    feed_failure: str | None
    cam_device: Device
    glove_device: Device


async def run_timeline(
    *,
    cam_clock: DeviceClock,
    glove_clock: DeviceClock,
    cam_link: Link = symmetric_jitter,
    glove_link: Link = symmetric_jitter,
    cam_drop: frozenset[int] = frozenset(),
    glove_drop: frozenset[int] = frozenset(),
    cam_resets_s: tuple[float, ...] = (),
    cam_outage: tuple[float, float] | None = None,
    seed: int = 7,
) -> Result:
    rng = random.Random(seed)
    end_s = event_time_s(EVENTS) + 3.0
    cam = Device(cam_clock, cam_link, random.Random(rng.random()), BASE_S - 3.0, end_s, cam_outage, cam_resets_s)
    glove = Device(glove_clock, glove_link, random.Random(rng.random()), BASE_S - 3.0, end_s)

    # Camera: the stamp is the device clock at the true event time; delivery delay is jittery and independent.
    frames = []
    for k in range(EVENTS):
        if k in cam_drop:
            continue
        pc_ns = round(event_time_s(k) * S)
        delay_s = rng.uniform(0.02, 0.9)
        if any(event_time_s(k) < r <= event_time_s(k) + delay_s for r in cam_resets_s):
            continue  # in flight when the publisher restarted: lost with its connection
        frames.append(
            ReceivedFrame(
                stream_id=16,
                image=np.full((2, 2, 3), k, dtype=np.uint8),
                receive_time_ns=pc_ns + round(delay_s * S),
                transport_timestamp_us=0,
                user_timestamp_us=round(cam_clock.read_ns(pc_ns) / 1000),
                frame_id=k,
                user_data=None,
            )
        )
    receiver = SimReceiver(frames, cam)
    stream = p.LiveKitCameraStream(
        stream_id=16,
        name="cam",
        url="wss://lk",
        room="r",
        publisher="o",
        track="t",
        timestamp_basis=p.LiveKitTimestampBasis.SENSOR_CAPTURE_MONOTONIC_US,
    )
    ingress = ListIngress()
    feed = LiveKitRecordingFeed(
        # The simulated links are deliberately noisy; the budget gate has its own tests.
        receiver,
        [stream],
        ingress,
        {16: "cam"},
        max_uncertainty_ns=1_000_000_000,
    )
    feed.start()
    for _ in range(5000):
        await asyncio.sleep(0)
        if feed.failure:
            break
    await feed.stop()
    camera = {}
    for sample, mapping in zip(ingress.offered, receiver.clock_sync.used):
        k = int(sample.image[0, 0, 0])
        camera[k] = Recorded(k, sample.capture_time, mapping, cam_clock.ppm_at(event_time_s(k)))

    # Tracking: samples arrive in delivery order and are mapped with the mapping current on arrival.
    arrivals = sorted(
        (event_time_s(k) + rng.uniform(0.001, 0.015), k) for k in range(EVENTS) if k not in glove_drop
    )
    tracking = {}
    for arrival_s, k in arrivals:
        glove.advance(round(arrival_s * S))
        mapping = glove.mapping()
        assert mapping is not None
        stamp_ns = glove_clock.read_ns(round(event_time_s(k) * S))
        tracking[k] = Recorded(
            k,
            mapping.to_local_ns(stamp_ns) / S,
            mapping,
            glove_clock.ppm_at(event_time_s(k)),
            glove.estimator.discontinuity,
        )
    return Result(camera, tracking, feed.failure, cam, glove)


def default_clocks() -> tuple[DeviceClock, DeviceClock]:
    # Different boot offsets (one positive, one negative) and drift of opposite sign.
    return DeviceClock(3_200_000_000_000, 35.0), DeviceClock(-1_500_000_000_000, -80.0)


def paired(result: Result) -> list[int]:
    """Events both sources recorded."""
    return sorted(result.camera.keys() & result.glove.keys())


def assert_paired_within_uncertainty(result: Result, ks: list[int]) -> None:
    for k in ks:
        cam, glove = result.camera[k], result.glove[k]
        assert abs(cam.error_s) <= cam.tolerance_s, (k, cam.error_s, cam.tolerance_s)
        assert abs(glove.error_s) <= glove.tolerance_s, (k, glove.error_s, glove.tolerance_s)
        gap_s = abs(cam.time_s - glove.time_s)
        assert gap_s <= cam.tolerance_s + glove.tolerance_s, (k, gap_s)


async def test_independent_clocks_pair_within_summed_uncertainty_and_not_by_the_injected_offset():
    cam_clock, glove_clock = default_clocks()
    result = await run_timeline(cam_clock=cam_clock, glove_clock=glove_clock)
    assert result.feed_failure == TRACK_LOST
    ks = paired(result)
    assert len(ks) == EVENTS
    assert_paired_within_uncertainty(result, ks)
    assert not result.glove_device.estimator.discontinuity
    assert not result.cam_device.estimator.discontinuity

    # Sources really do disagree before mapping, and subtracting each device's injected boot offset does not pair
    # them either: the drift since boot is far beyond the uncertainty. Only the exchange-based mapping pairs them.
    k = ks[-1]
    raw_gap_s = (
        abs(cam_clock.read_ns(round(event_time_s(k) * S)) - glove_clock.read_ns(round(event_time_s(k) * S))) / S
    )
    assert raw_gap_s > 1000.0
    naive_cam = cam_clock.read_ns(round(event_time_s(k) * S)) - cam_clock.segments[0][1]
    naive_glove = glove_clock.read_ns(round(event_time_s(k) * S)) - glove_clock.segments[0][1]
    naive_gap_s = abs(naive_cam - naive_glove) / S
    allowed = result.camera[k].tolerance_s + result.glove[k].tolerance_s
    assert naive_gap_s > 10 * allowed


@pytest.mark.parametrize(
    ("cam_link", "glove_link"), [(forward_heavy, backward_heavy), (backward_heavy, forward_heavy)]
)
async def test_one_way_delay_asymmetry_reaches_half_the_round_trip_and_not_more(cam_link, glove_link):
    # No drift here, so the whole residual is the asymmetry the estimator cannot see.
    result = await run_timeline(
        cam_clock=DeviceClock(3_200_000_000_000, 0.0),
        glove_clock=DeviceClock(-1_500_000_000_000, 0.0),
        cam_link=cam_link,
        glove_link=glove_link,
    )
    ks = paired(result)
    assert_paired_within_uncertainty(result, ks)
    for source in (result.camera, result.glove):
        for k in ks:
            rec = source[k]
            # The error is (df - db) / 2 against an uncertainty of (df + db) / 2: the bound is nearly reached.
            assert abs(rec.error_s) > 0.98 * rec.mapping.uncertainty_ns / S - QUANTIZATION_S
    # The mapping is not an oracle: the recorded times carry the unseen asymmetry, so they disagree with truth.
    assert abs(result.camera[ks[0]].error_s) > 0.005


async def test_delivery_jitter_and_dropped_frames_leave_recorded_times_and_identity_intact():
    cam_clock, glove_clock = default_clocks()
    cam_drop, glove_drop = frozenset({3, 4, 17, 40}), frozenset({10, 11, 40, 55})
    result = await run_timeline(
        cam_clock=cam_clock, glove_clock=glove_clock, cam_drop=cam_drop, glove_drop=glove_drop
    )
    assert result.feed_failure == TRACK_LOST
    assert set(result.camera) == set(range(EVENTS)) - cam_drop
    assert set(result.glove) == set(range(EVENTS)) - glove_drop
    ks = paired(result)
    assert len(ks) == EVENTS - len(cam_drop | glove_drop)
    assert_paired_within_uncertainty(result, ks)


async def test_publisher_restart_starts_a_new_epoch_and_later_events_pair_under_the_new_clock():
    cam_clock, glove_clock = default_clocks()
    restart_s = BASE_S - 3.0 + 10.4  # between two periodic exchanges, mid-timeline
    cam_clock.add_segment(restart_s, offset_ns=9_000_000_000_000, ppm=-12.0)  # the publisher rebooted
    result = await run_timeline(cam_clock=cam_clock, glove_clock=glove_clock, cam_resets_s=(restart_s,))
    assert result.feed_failure == TRACK_LOST
    ks = paired(result)
    assert EVENTS - 4 <= len(ks) < EVENTS  # frames in flight at the restart are lost
    assert_paired_within_uncertainty(result, ks)
    before = [k for k in ks if event_time_s(k) < restart_s]
    after = [k for k in ks if event_time_s(k) > restart_s + 1.0]
    assert before and after
    assert {result.camera[k].mapping.epoch for k in before} == {1}
    assert all(result.camera[k].mapping.epoch >= 2 for k in after)
    assert not result.cam_device.estimator.discontinuity  # a reset is a new epoch, not a jump


async def test_a_clock_step_on_the_glove_sets_discontinuity_and_misplaces_samples_until_it_is_seen():
    cam_clock, glove_clock = default_clocks()
    step_s = BASE_S - 3.0 + 8.1
    segment = glove_clock.segments[0]
    glove_clock.add_segment(step_s, segment[1] + 40_000_000, segment[2])  # the device clock steps forward 40 ms
    result = await run_timeline(cam_clock=cam_clock, glove_clock=glove_clock)
    assert result.glove_device.estimator.discontinuity
    flagged = [k for k, r in result.glove.items() if r.discontinuity]
    assert flagged and min(flagged) > 0
    first_flagged = min(flagged)
    # Before the step, and from the first flagged sample on (a new epoch built from the newest exchange), the
    # recorded times pair within the uncertainty; the flag is what marks the take as spanning a step.
    clean = [k for k in paired(result) if event_time_s(k) < step_s - 0.5 or k >= first_flagged]
    assert_paired_within_uncertainty(result, clean)
    # Samples after the step but before the estimator saw it carry the full step as error, under the old mapping.
    unseen = [k for k in result.glove if event_time_s(k) > step_s and k < first_flagged]
    assert unseen
    for k in unseen:
        rec = result.glove[k]
        assert not rec.discontinuity
        assert abs(rec.error_s) > 0.03 > rec.tolerance_s
    assert result.glove[first_flagged].mapping.epoch > result.glove[0].mapping.epoch
    assert len(result.glove_device.estimator.history) >= 2


async def test_a_camera_clock_step_without_a_reset_fails_the_take_with_the_jump_reason():
    cam_clock, glove_clock = default_clocks()
    step_s = BASE_S - 3.0 + 8.1
    segment = cam_clock.segments[0]
    cam_clock.add_segment(step_s, segment[1] + 40_000_000, segment[2])
    result = await run_timeline(cam_clock=cam_clock, glove_clock=glove_clock)
    assert result.feed_failure is not None and "clock mapping jumped" in result.feed_failure
    # Frames recorded before the failure were mapped under the mapping that was good then.
    assert result.camera
    assert_paired_within_uncertainty(result, [k for k in paired(result) if event_time_s(k) < step_s - 0.5])
    assert len(result.camera) < EVENTS


async def test_exchanges_that_stop_for_longer_than_the_staleness_limit_fail_the_take_with_the_stale_reason():
    cam_clock, glove_clock = default_clocks()
    outage = (BASE_S + 4.0, BASE_S + 30.0)  # no replies from the publisher from here on
    result = await run_timeline(cam_clock=cam_clock, glove_clock=glove_clock, cam_outage=outage)
    assert result.feed_failure is not None and "no current clock mapping" in result.feed_failure
    assert result.camera and len(result.camera) < EVENTS
    # Everything recorded before the outage went stale still pairs within its uncertainty.
    assert_paired_within_uncertainty(result, paired(result))
