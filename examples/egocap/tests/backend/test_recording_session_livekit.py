# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The coordinator's optional LiveKit source: prepared before storage opens, fed after the ingress starts,
stopped before the ingress is retired."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from capture_server import protocol as p
from egocap_web.livekit_receiver import ReceivedFrame, StreamCounters
from egocap_web.livekit_recording import LiveKitRecordingFeed
from egocap_web.recording_session import CoordinatorBusyError, RecordingSessionCoordinator
from recording_service import (
    Disposition,
    FieldSpec,
    NumericSourceSpec,
    RecordingPhase,
    RecordingService,
    ServiceConfig,
)

NUMERIC = NumericSourceSpec("s", (FieldSpec("v", "float64"),))


class FakeSource:
    """Records the order of prepare/start/stop calls; can be told to fail prepare."""

    def __init__(self, events: list[str], prepare_error: Exception | None = None) -> None:
        self.events = events
        self.prepare_error = prepare_error
        self.ingress = None
        self.on_failure = None

    async def prepare(self) -> None:
        self.events.append("prepare")
        if self.prepare_error is not None:
            raise self.prepare_error

    def start(self, ingress, on_failure) -> None:
        self.events.append("start")
        self.ingress, self.on_failure = ingress, on_failure

    async def stop(self) -> None:
        self.events.append("stop")

    def timing_evidence(self):
        return None


def make_coordinator(tmp_path: Path, source) -> RecordingSessionCoordinator:
    def factory() -> ServiceConfig:
        return ServiceConfig(output_root=tmp_path / "out", numeric_sources=(NUMERIC,), video_sources=())

    return RecordingSessionCoordinator(
        factory, queue_capacity=8, readiness_check=lambda: (True, None), livekit_source=source
    )


async def test_a_prepare_failure_refuses_the_start_before_storage_opens(tmp_path, monkeypatch):
    events: list[str] = []
    coordinator = make_coordinator(tmp_path, FakeSource(events, RuntimeError("cam16 sent no frame")))
    await coordinator.open()
    coordinator.set_task("t")
    opened: list[bool] = []
    original = coordinator._open_storage_if_needed

    async def counting_open():
        opened.append(True)
        return await original()

    monkeypatch.setattr(coordinator, "_open_storage_if_needed", counting_open)
    with pytest.raises(CoordinatorBusyError, match="LiveKit video is not ready: cam16 sent no frame"):
        await coordinator.start_episode()
    assert opened == []
    assert events == ["prepare"]
    assert coordinator.snapshot().service.phase is RecordingPhase.IDLE
    await coordinator.close()


async def test_a_start_that_fails_after_prepare_stops_the_source(tmp_path, monkeypatch):
    events: list[str] = []
    coordinator = make_coordinator(tmp_path, FakeSource(events))
    await coordinator.open()
    coordinator.set_task("t")

    async def failing(_context):
        raise RuntimeError("storage exploded")

    monkeypatch.setattr(coordinator, "_start_episode_with_storage", failing)
    with pytest.raises(RuntimeError, match="storage exploded"):
        await coordinator.start_episode()
    assert events == ["prepare", "stop"]
    await coordinator.close()


async def test_a_start_cancelled_after_prepare_stops_the_source(tmp_path, monkeypatch):
    events: list[str] = []
    coordinator = make_coordinator(tmp_path, FakeSource(events))
    await coordinator.open()
    coordinator.set_task("t")

    async def cancelled(_context):
        raise asyncio.CancelledError

    monkeypatch.setattr(coordinator, "_start_episode_with_storage", cancelled)
    with pytest.raises(asyncio.CancelledError):
        await coordinator.start_episode()
    assert events == ["prepare", "stop"]
    await coordinator.close()


async def test_the_source_is_started_with_the_ingress_and_its_failure_settles_the_take(tmp_path):
    events: list[str] = []
    source = FakeSource(events)
    coordinator = make_coordinator(tmp_path, source)
    await coordinator.open()
    coordinator.set_task("t")
    await coordinator.start_episode()
    assert events == ["prepare", "start"]
    assert source.ingress is coordinator._current_ingress

    source.on_failure("cam16: the LiveKit track was lost")
    for _ in range(300):
        snap = coordinator.snapshot()
        if snap.service.phase is RecordingPhase.RECOVERY_REQUIRED and snap.active_operation is None:
            break
        await asyncio.sleep(0.01)
    assert coordinator.snapshot().service.phase is RecordingPhase.RECOVERY_REQUIRED
    await coordinator.close()


async def test_retiring_the_take_stops_the_source_before_the_ingress(tmp_path):
    events: list[str] = []
    coordinator = make_coordinator(tmp_path, FakeSource(events))
    await coordinator.open()
    coordinator.set_task("t")
    episode = await coordinator.start_episode()
    ingress = coordinator._current_ingress
    original_stop = ingress.stop

    async def recording_stop():
        events.append("ingress.stop")
        return await original_stop()

    ingress.stop = recording_stop
    await coordinator.stop_episode(episode.episode_id)
    assert events[-2:] == ["stop", "ingress.stop"]
    await coordinator.close()


class EvidenceSource(FakeSource):
    """A source whose timing evidence is read once per take, in the order given (None: no clock sync)."""

    def __init__(self, events: list[str], evidence: list) -> None:
        super().__init__(events)
        self.evidence = list(evidence)

    def timing_evidence(self):
        self.events.append("timing_evidence")
        return self.evidence.pop(0)


def spy_on_service(monkeypatch, events: list[str]) -> list[tuple[str, dict]]:
    """Record set_episode_timing calls (and their order against stop_episode) on the real service."""
    calls: list[tuple[str, dict]] = []
    set_timing, stop = RecordingService.set_episode_timing, RecordingService.stop_episode

    def recording_set_timing(self, episode_id, timing):
        events.append("set_timing")
        calls.append((episode_id, timing))
        return set_timing(self, episode_id, timing)

    async def recording_stop(self, episode_id):
        events.append("stop_episode")
        return await stop(self, episode_id)

    monkeypatch.setattr(RecordingService, "set_episode_timing", recording_set_timing)
    monkeypatch.setattr(RecordingService, "stop_episode", recording_stop)
    return calls


async def test_the_sources_timing_evidence_is_given_to_the_service_once_before_stop_and_not_reused(
    tmp_path, monkeypatch
):
    events: list[str] = []
    first = {"clock_mapping": {"mappings": [1]}, "event_time": {"cam16": {"qualified": True}}}
    tracking = {
        "pico_controller": {"meaning": "host_poll_time", "qualified": False},
        "manus_glove": {"meaning": "host_poll_time", "qualified": False},
    }
    calls = spy_on_service(monkeypatch, events)
    coordinator = make_coordinator(tmp_path, EvidenceSource(events, [first, None]))
    await coordinator.open()
    coordinator.set_task("t")

    one = await coordinator.start_episode()
    await coordinator.stop_episode(one.episode_id)
    merged = {**first, "event_time": {**first["event_time"], **tracking}}
    assert calls == [(one.episode_id, merged)]
    assert events.index("timing_evidence") < events.index("set_timing") < events.index("stop_episode")

    two = await coordinator.start_episode()
    await coordinator.stop_episode(two.episode_id)
    # the second take's source had none: only its tracking times are recorded, nothing carried over
    assert calls[1:] == [(two.episode_id, {"event_time": tracking})]
    await coordinator.close()


async def test_a_discarded_takes_evidence_does_not_describe_the_next_take(tmp_path, monkeypatch):
    events: list[str] = []
    calls = spy_on_service(monkeypatch, events)
    tracking = {
        "pico_controller": {"meaning": "host_poll_time", "qualified": False},
        "manus_glove": {"meaning": "host_poll_time", "qualified": False},
    }
    discarded = {"clock_mapping": {"mappings": ["discarded take"]}}
    coordinator = make_coordinator(tmp_path, EvidenceSource(events, [discarded, None]))
    await coordinator.open()
    coordinator.set_task("t")

    one = await coordinator.start_episode()
    await coordinator.discard_episode(one.episode_id)
    two = await coordinator.start_episode()
    await coordinator.stop_episode(two.episode_id)

    # the second take's source had none: only its tracking times are recorded, nothing inherited
    assert calls == [(two.episode_id, {"event_time": tracking})]
    await coordinator.close()


async def test_an_interrupted_takes_evidence_does_not_describe_the_next_take(tmp_path, monkeypatch):
    events: list[str] = []
    calls = spy_on_service(monkeypatch, events)
    tracking = {
        "pico_controller": {"meaning": "host_poll_time", "qualified": False},
        "manus_glove": {"meaning": "host_poll_time", "qualified": False},
    }
    interrupted = {"clock_mapping": {"mappings": ["interrupted take"]}}
    coordinator = make_coordinator(tmp_path, EvidenceSource(events, [interrupted, None]))
    await coordinator.open()
    coordinator.set_task("t")

    one = await coordinator.start_episode()
    coordinator._readiness_check = lambda: (False, "the camera went away")
    interrupted_result = await coordinator.stop_episode(one.episode_id)
    assert interrupted_result.disposition is not Disposition.KEPT
    _ = await coordinator.discard_pending(one.episode_id)  # a retained take must be settled before another starts
    coordinator._readiness_check = lambda: (True, None)
    two = await coordinator.start_episode()
    await coordinator.stop_episode(two.episode_id)

    assert calls == [(two.episode_id, {"event_time": tracking})]
    await coordinator.close()


class _UnstampedReceiver:
    """Delivers one frame with no publisher timestamp, then stays quiet."""

    class _Clock:
        discontinuity = False

        def mapping(self):
            return None

    def __init__(self) -> None:
        self.clock_sync = self._Clock()
        self.counters = {16: StreamCounters()}
        self._sent = False

    async def next_frame_or_lost(self, stream_id, timeout=None):
        if self._sent:
            await asyncio.Event().wait()
        self._sent = True
        return ReceivedFrame(
            stream_id=stream_id,
            image=np.zeros((4, 6, 3), dtype=np.uint8),
            receive_time_ns=0,
            transport_timestamp_us=1,
            user_timestamp_us=None,
            frame_id=None,
            user_data=None,
        )


class _RealFeedSource(FakeSource):
    """A source whose ``start`` runs the real feed, so its own failure reason reaches the coordinator."""

    def start(self, ingress, on_failure) -> None:
        super().start(ingress, on_failure)
        stream = p.LiveKitCameraStream(
            stream_id=16,
            name="head",
            url="ws://lk",
            room="r",
            publisher="o",
            track="t",
            timestamp_basis=p.LiveKitTimestampBasis.SENSOR_CAPTURE_MONOTONIC_US,
        )
        self.feed = LiveKitRecordingFeed(
            _UnstampedReceiver(), [stream], ingress, {16: "head"}, on_failure=on_failure
        )
        self.feed.start()

    async def stop(self) -> None:
        await super().stop()
        await self.feed.stop()


async def test_a_frame_without_a_publisher_timestamp_ends_the_take_and_leaves_it_recoverable(tmp_path):
    source = _RealFeedSource([])
    coordinator = make_coordinator(tmp_path, source)
    await coordinator.open()
    coordinator.set_task("t")
    await coordinator.start_episode()
    for _ in range(300):
        snap = coordinator.snapshot()
        if snap.service.phase is RecordingPhase.RECOVERY_REQUIRED and snap.active_operation is None:
            break
        await asyncio.sleep(0.01)
    snap = coordinator.snapshot()
    assert snap.service.phase is RecordingPhase.RECOVERY_REQUIRED
    assert "a frame arrived without the publisher's frame timestamp" in (coordinator._last_issue or "")
    assert snap.service.last_result is None or snap.service.last_result.disposition is not Disposition.KEPT
    await coordinator.close()
