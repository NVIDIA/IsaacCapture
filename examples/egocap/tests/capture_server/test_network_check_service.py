# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""NetworkCheckService against a fake receiver whose clock is injected, so a 2 s window takes no real time.

The fake delivers scripted frames and raises TimeoutError once a stream's script is exhausted; the injected clock
only moves when a test moves it.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict

import numpy as np
import pytest

from capture_server import protocol as p
from capture_server.clock_exchange import ClockMapping
from capture_server.frame_timing import BUDGET_ENV
from capture_server.livekit_receiver import ReceivedFrame, StreamCounters
from capture_server.network_check import (
    LEGS_NOT_TESTED,
    LEGS_TESTED,
    NetworkCheckPolicy,
    NetworkCheckService,
    streams_generation,
)

SEC = 1_000_000_000
T0 = 1_000 * SEC
FPS = 30.0
WINDOW_S = 2.0


def _stream(stream_id: int, **overrides) -> p.LiveKitCameraStream:
    fields = dict(
        stream_id=stream_id,
        name=f"cam{stream_id}",
        url="wss://lk",
        room="room",
        publisher="orin",
        track=f"t{stream_id}",
        width=6,
        height=4,
        fps=FPS,
    )
    fields.update(overrides)
    return p.LiveKitCameraStream(**fields)


STREAMS = (_stream(16), _stream(17))


def _policy(**overrides) -> NetworkCheckPolicy:
    fields = dict(startup_s=1.0, window_s=WINDOW_S, valid_for_s=100.0)
    fields.update(overrides)
    return NetworkCheckPolicy(**fields)


class Clock:
    def __init__(self) -> None:
        self.ns = T0

    def __call__(self) -> int:
        return self.ns

    def advance(self, seconds: float) -> None:
        self.ns += int(seconds * SEC)


class FakeClockSync:
    def __init__(self) -> None:
        self.discontinuity = False
        self.uncertainty_ns: int | None = 1_000

    def mapping(self):
        if self.uncertainty_ns is None:
            return None
        return ClockMapping(
            epoch=1, offset_ns=0, rtt_ns=2_000, uncertainty_ns=self.uncertainty_ns, accepted_at_ns=0
        )


def frame(stream_id: int, receive_ns: int, frame_id: int, *, age_s: float = 0.05, stamped=True, shape=(4, 6, 3)):
    return ReceivedFrame(
        stream_id=stream_id,
        image=np.zeros(shape, dtype=np.uint8),
        receive_time_ns=receive_ns,
        transport_timestamp_us=1,
        user_timestamp_us=int((receive_ns - age_s * SEC) / 1000) if stamped else None,
        frame_id=frame_id,
        user_data=None,
    )


def healthy(stream_id: int, *, fps: float = FPS, age_s: float = 0.05, base: int = T0) -> list[ReceivedFrame]:
    return [frame(stream_id, base + int(i / fps * SEC), i, age_s=age_s) for i in range(int(WINDOW_S * fps))]


class FakeReceiver:
    """Delivers a startup frame per stream (or a startup failure), then each stream's measured script."""

    instances: list[FakeReceiver]

    def __init__(self, streams, token, clock_sync=False, *, scripts=None, startup=None):
        self.streams = streams
        self.token = token
        self.clock_sync = FakeClockSync() if clock_sync else None
        self.scripts = {k: list(v) for k, v in (scripts or {}).items()}
        self.startup = startup or {}
        self.counters = defaultdict(StreamCounters)
        self.measuring = False
        self.starts = self.stops = self.cleared = 0
        self.start_error: Exception | None = None
        self.join_gate: asyncio.Event | None = None
        self.stop_gate: asyncio.Event | None = None
        self.measure_gate: asyncio.Event | None = None
        self.entered_measure = asyncio.Event()
        self.on_measure = None
        self.lost: set[int] = set()

    async def start(self):
        self.starts += 1
        if self.join_gate is not None:
            await self.join_gate.wait()
        if self.start_error is not None:
            raise self.start_error

    async def stop(self):
        self.stops += 1
        if self.stop_gate is not None:
            await self.stop_gate.wait()

    def track_ids(self):
        return {s.stream_id: f"TR_{s.stream_id}" for s in self.streams}

    def clear_queued(self):
        self.cleared += 1
        self.measuring = True

    async def next_frame_or_lost(self, stream_id, timeout=None):
        await asyncio.sleep(0)
        if not self.measuring:
            mode = self.startup.get(stream_id, "frame")
            if mode == "timeout":
                raise asyncio.TimeoutError
            if mode == "lost":
                return None
            if isinstance(mode, ReceivedFrame):
                return mode
            return frame(stream_id, T0, 0)
        if not self.entered_measure.is_set():
            self.entered_measure.set()
            if self.on_measure is not None:
                self.on_measure(self)
        if self.measure_gate is not None:
            await self.measure_gate.wait()
        script = self.scripts.get(stream_id, [])
        if script:
            return script.pop(0)
        if stream_id in self.lost:
            return None
        raise asyncio.TimeoutError


class Harness:
    def __init__(self, streams=STREAMS, policy=None, **receiver_kwargs) -> None:
        self.clock = Clock()
        self.unix = 1_700_000_000 * SEC
        self.rooms: list[str] = []
        self.receivers: list[FakeReceiver] = []
        self.token_error: Exception | None = None
        self.receiver_kwargs = receiver_kwargs
        self.service = NetworkCheckService(
            streams,
            policy or _policy(),
            self._token,
            receiver_factory=self._factory,
            monotonic_ns=self.clock,
            unix_ns=lambda: self.unix,
        )

    def _token(self, room: str) -> str:
        if self.token_error is not None:
            raise self.token_error
        self.rooms.append(room)
        return f"token-for-{room}"

    def _factory(self, streams, token, clock_sync=False):
        receiver = FakeReceiver(streams, token, clock_sync, **self.receiver_kwargs)
        self.receivers.append(receiver)
        return receiver

    @property
    def receiver(self) -> FakeReceiver:
        return self.receivers[-1]

    async def finished(self, timeout_s: float = 5.0) -> p.NetworkCheckStatus:
        for _ in range(int(timeout_s / 0.005)):
            if not self.service.running:
                return self.service.snapshot()
            await asyncio.sleep(0.005)
        raise AssertionError("the check did not finish")


def _healthy_harness(**kwargs) -> Harness:
    return Harness(scripts={16: healthy(16), 17: healthy(17)}, **kwargs)


@pytest.fixture(autouse=True)
def _default_budget(monkeypatch):
    monkeypatch.delenv(BUDGET_ENV, raising=False)


def test_the_constructor_rejects_no_streams():
    with pytest.raises(ValueError, match="at least one"):
        NetworkCheckService((), _policy(), lambda room: "t")


@pytest.mark.parametrize("missing", ["width", "height", "fps"])
def test_the_constructor_rejects_a_stream_without_geometry_or_rate(missing):
    with pytest.raises(ValueError, match="width, height and fps"):
        NetworkCheckService((_stream(16), _stream(17, **{missing: None})), _policy(), lambda room: "t")


def test_streams_generation_is_order_independent_and_sensitive_to_every_identity_field():
    base = streams_generation(STREAMS)
    assert streams_generation(tuple(reversed(STREAMS))) == base
    for change in (
        {"fps": 25.0},
        {"width": 8},
        {"height": 6},
        {"track": "other"},
        {"name": "other"},
        {"url": "wss://other"},
        {"room": "other"},
        {"publisher": "other"},
        {"timestamp_basis": p.LiveKitTimestampBasis.SENSOR_CAPTURE_MONOTONIC_US},
    ):
        assert streams_generation((_stream(16, **change), STREAMS[1])) != base, change


async def test_the_initial_state_is_not_tested_with_every_camera_unjudged():
    harness = Harness()
    status = harness.service.snapshot()
    assert status.state is p.NetworkCheckState.NOT_TESTED
    assert status.generation == harness.service.generation == streams_generation(STREAMS)
    assert [(c.stream_id, c.passed, c.expected_fps) for c in status.cameras] == [(16, None, FPS), (17, None, FPS)]
    assert (status.phase, status.progress_s, status.completed_at_unix_ns, status.reason) == (None,) * 4
    assert (status.window_s, status.valid_for_s) == (WINDOW_S, 100.0)
    assert not harness.service.running
    assert harness.receivers == []


async def test_a_check_reports_starting_then_measuring_with_progress():
    harness = _healthy_harness()
    harness.service.begin_session("s1")
    await asyncio.sleep(0)
    starting = harness.service.snapshot()
    assert (starting.state, starting.phase, starting.progress_s) == (p.NetworkCheckState.CHECKING, "starting", 0)
    harness.receiver.measure_gate = measure_gate = asyncio.Event()
    await asyncio.wait_for(harness.receiver.entered_measure.wait(), 5)
    harness.clock.advance(1)
    measuring = harness.service.snapshot()
    assert (measuring.state, measuring.phase, measuring.progress_s) == (
        p.NetworkCheckState.CHECKING,
        "measuring",
        1,
    )
    assert measuring.completed_at_unix_ns is None and measuring.reason is None
    measure_gate.set()
    assert (await harness.finished()).state is p.NetworkCheckState.READY


async def test_the_receiver_is_built_for_the_room_with_clock_sync_and_a_token_for_it():
    harness = _healthy_harness()
    harness.service.begin_session("s1")
    await harness.finished()
    assert harness.rooms == ["room"]
    assert harness.receiver.token == "token-for-room"
    assert harness.receiver.clock_sync is not None
    assert harness.receiver.streams == STREAMS


async def test_a_healthy_check_is_ready_with_every_camera_passed_and_the_legs_listed():
    harness = _healthy_harness()
    harness.service.begin_session("s1")
    status = await harness.finished()
    assert status.state is p.NetworkCheckState.READY
    assert status.reason is None and status.phase is None and status.progress_s is None
    assert [c.passed for c in status.cameras] == [True, True]
    assert all(c.reasons == () and c.usable_frames == int(WINDOW_S * FPS) for c in status.cameras)
    assert status.legs_tested == LEGS_TESTED and status.legs_not_tested == LEGS_NOT_TESTED
    assert status.legs_tested and status.legs_not_tested
    assert status.generation == streams_generation(STREAMS)
    assert status.completed_at_unix_ns == harness.unix
    assert "LiveKit wss://lk" in status.endpoint and "room" in status.endpoint
    assert harness.receiver.cleared == 1


async def test_one_failing_camera_makes_the_check_unsuitable_and_names_it_with_the_others_reported():
    harness = Harness(scripts={16: healthy(16), 17: healthy(17, fps=5.0)})
    harness.service.begin_session("s1")
    status = await harness.finished()
    assert status.state is p.NetworkCheckState.UNSUITABLE
    by_id = {c.stream_id: c for c in status.cameras}
    assert by_id[16].passed is True and by_id[17].passed is False
    assert "cam17" in status.reason and "cam16" not in status.reason
    assert by_id[17].reasons and by_id[17].usable_frames == int(WINDOW_S * 5)


async def test_a_camera_stamped_in_the_future_is_unsuitable():
    harness = Harness(scripts={16: healthy(16), 17: healthy(17, age_s=-10.0)})
    harness.service.begin_session("s1")
    status = await harness.finished()
    assert status.state is p.NetworkCheckState.UNSUITABLE
    assert [c.passed for c in status.cameras] == [True, False]


async def test_a_lost_track_during_the_window_is_unsuitable():
    harness = Harness(scripts={16: healthy(16), 17: healthy(17)[:10]})
    harness.service.begin_session("s1")
    await asyncio.sleep(0)
    # Mark the track lost as soon as the window begins; its queued frames are still delivered first.
    harness.receiver.lost = {17}
    status = await harness.finished()
    assert status.state is p.NetworkCheckState.UNSUITABLE
    assert any("track was lost" in r for r in status.cameras[1].reasons)


async def test_a_camera_with_no_first_frame_within_startup_is_unsuitable_with_its_reason():
    harness = Harness(startup={17: "timeout"}, scripts={16: healthy(16)})
    harness.service.begin_session("s1")
    status = await harness.finished()
    assert status.state is p.NetworkCheckState.UNSUITABLE
    assert "cam17" in status.reason and "no frame arrived within the 1 s startup allowance" in status.reason
    assert [c.stream_id for c in status.cameras] == [17]
    assert status.cameras[0].passed is False
    assert harness.receiver.cleared == 0


async def test_a_track_lost_before_the_check_started_is_unsuitable():
    harness = Harness(startup={16: "lost"})
    harness.service.begin_session("s1")
    status = await harness.finished()
    assert status.state is p.NetworkCheckState.UNSUITABLE
    assert "lost before the check started" in status.reason


async def test_a_first_frame_without_a_capture_time_is_unsuitable():
    harness = Harness(startup={16: frame(16, T0, 0, stamped=False)})
    harness.service.begin_session("s1")
    status = await harness.finished()
    assert status.state is p.NetworkCheckState.UNSUITABLE
    assert "cam16" in status.reason and "no capture time" in status.reason


async def test_a_first_frame_of_the_wrong_geometry_is_unsuitable():
    harness = Harness(startup={17: frame(17, T0, 0, shape=(8, 12, 3))})
    harness.service.begin_session("s1")
    status = await harness.finished()
    assert status.state is p.NetworkCheckState.UNSUITABLE
    assert "cam17" in status.reason and "12x8, expected 6x4" in status.reason


async def test_a_clock_mapping_that_never_arrives_is_unsuitable_after_the_startup_allowance():
    harness = Harness(policy=_policy(startup_s=0.2))
    harness.service.begin_session("s1")
    # The receiver does not exist until the task runs.
    await asyncio.sleep(0)
    harness.receiver.clock_sync.uncertainty_ns = None
    status = await harness.finished()
    assert status.state is p.NetworkCheckState.UNSUITABLE
    assert "did not answer clock-sync" in status.reason
    assert harness.receiver.cleared == 0


async def test_a_clock_mapping_over_the_uncertainty_budget_is_unsuitable():
    harness = Harness(policy=_policy(startup_s=0.2))
    harness.service.begin_session("s1")
    await asyncio.sleep(0)
    harness.receiver.clock_sync.uncertainty_ns = 50_000_000
    status = await harness.finished()
    assert status.state is p.NetworkCheckState.UNSUITABLE
    assert "uncertain by 50.00 ms" in status.reason and "budget" in status.reason


async def test_the_uncertainty_budget_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv(BUDGET_ENV, "100")
    harness = _healthy_harness()
    harness.service.begin_session("s1")
    await asyncio.sleep(0)
    harness.receiver.clock_sync.uncertainty_ns = 50_000_000
    assert (await harness.finished()).state is p.NetworkCheckState.READY


async def test_a_mapping_jump_during_the_window_fails_every_camera():
    harness = _healthy_harness()
    harness.service.begin_session("s1")
    await asyncio.sleep(0)
    harness.receiver.on_measure = lambda receiver: setattr(receiver.clock_sync, "discontinuity", True)
    status = await harness.finished()
    assert status.state is p.NetworkCheckState.UNSUITABLE
    assert [c.passed for c in status.cameras] == [False, False]
    assert all(any("clock mapping" in r for r in c.reasons) for c in status.cameras)


async def test_a_join_failure_is_unsuitable_not_an_exception():
    harness = Harness()
    harness.service.begin_session("s1")
    await asyncio.sleep(0)
    harness.receiver.start_error = RuntimeError("connection refused")
    status = await harness.finished()
    assert status.state is p.NetworkCheckState.UNSUITABLE
    assert "could not join the LiveKit room: connection refused" in status.reason
    assert harness.receiver.stops == 1


async def _until(predicate, timeout_s: float = 3.0) -> None:
    for _ in range(int(timeout_s / 0.005)):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not met")


async def test_a_join_timeout_is_unsuitable_and_the_late_join_is_retired_exactly_once():
    harness = Harness(policy=_policy(startup_s=0.1))
    harness.service.begin_session("s1")
    await asyncio.sleep(0)
    gate = harness.receiver.join_gate = asyncio.Event()
    status = await harness.finished()
    assert status.state is p.NetworkCheckState.UNSUITABLE
    assert "took longer than 0.1 s" in status.reason
    assert harness.receiver.stops == 0
    gate.set()
    await _until(lambda: harness.receiver.stops >= 1)
    await asyncio.sleep(0.05)
    assert harness.receiver.stops == 1


async def test_missing_livekit_credentials_report_could_not_run(monkeypatch):
    from capture_server import server as server_module
    from capture_server.livekit_token import API_KEY_ENV, API_SECRET_ENV

    monkeypatch.delenv(API_KEY_ENV, raising=False)
    monkeypatch.delenv(API_SECRET_ENV, raising=False)
    harness = Harness()
    harness.service._make_token = server_module._viewer_token
    harness.service.begin_session("s1")
    status = await harness.finished()
    assert status.state is p.NetworkCheckState.UNSUITABLE
    assert status.reason.startswith("the check could not run:")
    assert API_KEY_ENV in status.reason
    assert harness.receivers == []


async def test_a_token_failure_in_the_callable_is_unsuitable_not_an_exception():
    harness = Harness()
    harness.token_error = RuntimeError("no credentials")
    harness.service.begin_session("s1")
    status = await harness.finished()
    assert status.state is p.NetworkCheckState.UNSUITABLE
    assert "the check could not run: no credentials" in status.reason


async def test_a_ready_result_turns_stale_after_valid_for_and_not_before():
    harness = _healthy_harness()
    harness.service.begin_session("s1")
    assert (await harness.finished()).state is p.NetworkCheckState.READY
    harness.clock.advance(99)
    assert harness.service.snapshot().state is p.NetworkCheckState.READY
    harness.clock.advance(2)
    stale = harness.service.snapshot()
    assert stale.state is p.NetworkCheckState.STALE
    assert [c.passed for c in stale.cameras] == [True, True]
    assert stale.completed_at_unix_ns == harness.unix


async def test_an_unsuitable_result_does_not_turn_stale():
    harness = Harness(startup={16: "timeout"})
    harness.service.begin_session("s1")
    await harness.finished()
    harness.clock.advance(10_000)
    assert harness.service.snapshot().state is p.NetworkCheckState.UNSUITABLE


async def test_a_new_start_drops_the_old_verdict_at_once():
    harness = _healthy_harness()
    harness.service.begin_session("s1")
    await harness.finished()
    harness.clock.advance(1000)
    assert harness.service.snapshot().state is p.NetworkCheckState.STALE
    base = harness.clock.ns
    harness.receiver_kwargs["scripts"] = {16: healthy(16, base=base), 17: healthy(17, base=base)}
    assert harness.service.start() is True
    status = harness.service.snapshot()
    assert status.state is p.NetworkCheckState.CHECKING and status.phase == "starting"
    assert status.completed_at_unix_ns is None and status.reason is None
    assert [c.passed for c in status.cameras] == [None, None]
    assert (await harness.finished()).state is p.NetworkCheckState.READY
    assert len(harness.receivers) == 2


async def test_start_returns_false_while_a_check_is_running():
    harness = Harness()
    harness.service.begin_session("s1")
    await asyncio.sleep(0)
    harness.receiver.join_gate = gate = asyncio.Event()
    assert harness.service.running
    assert harness.service.start() is False
    assert len(harness.receivers) == 1
    gate.set()
    await harness.service.stop()


async def test_start_after_a_finished_check_runs_a_new_one():
    harness = _healthy_harness()
    harness.service.begin_session("s1")
    await harness.finished()
    harness.receiver_kwargs["scripts"] = {16: healthy(16), 17: healthy(17)}
    assert harness.service.start() is True
    await harness.finished()
    assert len(harness.receivers) == 2


async def test_start_without_an_active_session_does_nothing():
    harness = _healthy_harness()
    assert harness.service.start() is False
    assert harness.receivers == []
    assert harness.service.snapshot().state is p.NetworkCheckState.NOT_TESTED
    harness.service.begin_session("s1")
    await harness.finished()
    await harness.service.end_session("s1")
    assert harness.service.start() is False
    assert len(harness.receivers) == 1


async def test_begin_session_reports_checking_with_the_session_id_at_once():
    harness = _healthy_harness()
    assert harness.service.snapshot().session_id is None
    harness.service.begin_session("s1")
    status = harness.service.snapshot()
    assert (status.state, status.session_id, status.phase) == (p.NetworkCheckState.CHECKING, "s1", "starting")
    ready = await harness.finished()
    assert (ready.state, ready.session_id) == (p.NetworkCheckState.READY, "s1")


async def test_a_new_session_drops_the_previous_sessions_ready_verdict():
    harness = _healthy_harness()
    harness.service.begin_session("A")
    assert (await harness.finished()).state is p.NetworkCheckState.READY
    harness.receiver_kwargs["scripts"] = {}
    harness.service.begin_session("B")
    status = harness.service.snapshot()
    assert (status.state, status.session_id) == (p.NetworkCheckState.CHECKING, "B")
    assert status.completed_at_unix_ns is None and [c.passed for c in status.cameras] == [None, None]


async def test_a_failing_check_for_one_session_is_not_shown_for_the_next():
    harness = Harness(startup={16: "timeout"})
    harness.service.begin_session("A")
    failed = await harness.finished()
    assert (failed.state, failed.session_id) == (p.NetworkCheckState.UNSUITABLE, "A")
    assert failed.reason
    harness.receiver_kwargs["startup"] = {}
    harness.receiver_kwargs["scripts"] = {16: healthy(16), 17: healthy(17)}
    harness.service.begin_session("B")
    status = harness.service.snapshot()
    assert (status.state, status.session_id, status.reason) == (p.NetworkCheckState.CHECKING, "B", None)
    ready = await harness.finished()
    assert (ready.state, ready.session_id, ready.reason) == (p.NetworkCheckState.READY, "B", None)


async def test_end_session_leaves_no_verdict_and_no_session():
    harness = _healthy_harness()
    harness.service.begin_session("s1")
    assert (await harness.finished()).state is p.NetworkCheckState.READY
    await harness.service.end_session("s1")
    status = harness.service.snapshot()
    assert (status.state, status.session_id) == (p.NetworkCheckState.NOT_TESTED, None)
    assert status.completed_at_unix_ns is None and status.reason is None
    assert [c.passed for c in status.cameras] == [None, None]
    assert harness.receiver.stops == 1


async def test_end_session_of_another_session_changes_nothing():
    harness = _healthy_harness()
    harness.service.begin_session("s1")
    assert (await harness.finished()).state is p.NetworkCheckState.READY
    await harness.service.end_session("other")
    status = harness.service.snapshot()
    assert (status.state, status.session_id) == (p.NetworkCheckState.READY, "s1")


async def test_end_session_during_a_running_check_cancels_it_and_leaves_the_room_once():
    harness = _healthy_harness()
    harness.service.begin_session("s1")
    await asyncio.sleep(0)
    harness.receiver.measure_gate = asyncio.Event()
    await asyncio.wait_for(harness.receiver.entered_measure.wait(), 5)
    await asyncio.wait_for(harness.service.end_session("s1"), 5)
    assert not harness.service.running
    status = harness.service.snapshot()
    assert (status.state, status.session_id, status.reason) == (p.NetworkCheckState.NOT_TESTED, None, None)
    assert harness.receiver.stops == 1
    await asyncio.sleep(0.05)
    assert harness.receiver.stops == 1


async def test_begin_session_while_a_check_runs_cancels_the_old_one_and_starts_a_new_one():
    harness = _healthy_harness()
    harness.service.begin_session("A")
    await asyncio.sleep(0)
    first = harness.receiver
    first.measure_gate = asyncio.Event()
    await asyncio.wait_for(first.entered_measure.wait(), 5)
    harness.service.begin_session("B")
    status = harness.service.snapshot()
    assert (status.state, status.session_id) == (p.NetworkCheckState.CHECKING, "B")
    await harness.finished()
    assert len(harness.receivers) == 2
    assert first.stops == 1
    assert harness.service.snapshot().session_id == "B"


async def test_a_check_whose_previous_leave_never_finishes_is_unsuitable_with_that_reason(monkeypatch):
    monkeypatch.setattr("capture_server.network_check._LEAVE_TIMEOUT_S", 0.1)
    harness = _healthy_harness()
    harness.service.begin_session("A")
    await asyncio.sleep(0)
    harness.receiver.stop_gate = stop_gate = asyncio.Event()
    assert (await harness.finished()).state is p.NetworkCheckState.READY
    assert harness.service._background
    harness.service.start()
    status = await harness.finished()
    assert status.state is p.NetworkCheckState.UNSUITABLE
    assert "previous check's connection" in status.reason and "has not closed yet" in status.reason
    assert len(harness.receivers) == 1
    stop_gate.set()


async def test_a_new_check_starts_once_the_previous_leave_has_finished(monkeypatch):
    monkeypatch.setattr("capture_server.network_check._LEAVE_TIMEOUT_S", 2.0)
    harness = _healthy_harness()
    harness.service.begin_session("A")
    await asyncio.sleep(0)
    harness.receiver.stop_gate = stop_gate = asyncio.Event()
    assert (await harness.finished()).state is p.NetworkCheckState.READY
    base = harness.clock.ns
    harness.receiver_kwargs["scripts"] = {16: healthy(16, base=base), 17: healthy(17, base=base)}
    harness.service.start()
    await asyncio.sleep(0.05)
    assert len(harness.receivers) == 1
    stop_gate.set()
    assert (await harness.finished()).state is p.NetworkCheckState.READY
    assert len(harness.receivers) == 2


async def test_stop_during_the_measurement_cancels_leaves_the_room_and_reports_no_verdict():
    harness = _healthy_harness()
    harness.service.begin_session("s1")
    await asyncio.sleep(0)
    harness.receiver.measure_gate = asyncio.Event()
    await asyncio.wait_for(harness.receiver.entered_measure.wait(), 5)
    await harness.service.stop()
    assert not harness.service.running
    assert harness.receiver.stops == 1
    status = harness.service.snapshot()
    assert status.state is p.NetworkCheckState.NOT_TESTED
    assert status.phase is None and status.completed_at_unix_ns is None and status.reason is None


async def test_a_check_cancelled_during_the_join_leaves_the_room_once_when_the_join_ends(monkeypatch):
    monkeypatch.setattr("capture_server.network_check._LEAVE_TIMEOUT_S", 0.2)
    harness = Harness()
    harness.service.begin_session("s1")
    await asyncio.sleep(0)
    gate = harness.receiver.join_gate = asyncio.Event()
    await asyncio.sleep(0.01)
    await asyncio.wait_for(harness.service.stop(), 5)
    assert not harness.service.running
    assert harness.service.snapshot().state is p.NetworkCheckState.NOT_TESTED
    assert harness.receiver.stops == 0
    gate.set()
    await _until(lambda: harness.receiver.stops >= 1)
    await asyncio.sleep(0.05)
    assert harness.receiver.stops == 1


async def test_stop_is_bounded_when_the_join_never_ends(monkeypatch):
    monkeypatch.setattr("capture_server.network_check._LEAVE_TIMEOUT_S", 0.2)
    harness = Harness()
    harness.service.begin_session("s1")
    await asyncio.sleep(0)
    gate = harness.receiver.join_gate = asyncio.Event()
    await asyncio.sleep(0.01)
    await asyncio.wait_for(harness.service.stop(), 3)
    gate.set()
    await _until(lambda: harness.receiver.stops == 1)


async def test_a_stuck_leave_does_not_hold_up_the_verdict_or_stop(monkeypatch):
    monkeypatch.setattr("capture_server.network_check._LEAVE_TIMEOUT_S", 0.2)
    harness = _healthy_harness()
    harness.service.begin_session("s1")
    await asyncio.sleep(0)
    stop_gate = harness.receiver.stop_gate = asyncio.Event()
    status = await harness.finished()
    assert status.state is p.NetworkCheckState.READY
    await asyncio.wait_for(harness.service.stop(), 3)
    assert harness.receiver.stops == 1
    stop_gate.set()


async def test_stop_with_nothing_running_is_harmless_and_keeps_a_verdict():
    harness = _healthy_harness()
    await harness.service.stop()
    assert harness.service.snapshot().state is p.NetworkCheckState.NOT_TESTED
    harness.service.begin_session("s1")
    await harness.finished()
    await harness.service.stop()
    assert harness.service.snapshot().state is p.NetworkCheckState.READY


@pytest.mark.parametrize(
    "build",
    [
        lambda: Harness(scripts={16: healthy(16), 17: healthy(17)}),
        lambda: Harness(scripts={16: healthy(16), 17: healthy(17, fps=5.0)}),
        lambda: Harness(startup={16: "timeout"}),
        lambda: Harness(startup={16: "lost"}),
    ],
    ids=["ready", "failing-camera", "no-first-frame", "lost-before-start"],
)
async def test_the_receiver_is_stopped_exactly_once_on_every_outcome(build):
    harness = build()
    harness.service.begin_session("s1")
    await harness.finished()
    assert harness.receiver.stops == 1


async def test_the_measured_cameras_carry_their_track_ids():
    harness = _healthy_harness()
    harness.service.begin_session("s1")
    status = await harness.finished()
    assert {c.stream_id: c.track_id for c in status.cameras} == {16: "TR_16", 17: "TR_17"}
