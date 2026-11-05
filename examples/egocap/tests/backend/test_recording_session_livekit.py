# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The coordinator's optional LiveKit source: prepared before storage opens, fed after the ingress starts,
stopped before the ingress is retired."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from egocap_web.recording_session import CoordinatorBusyError, RecordingSessionCoordinator
from recording_service import FieldSpec, NumericSourceSpec, RecordingPhase, ServiceConfig

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
