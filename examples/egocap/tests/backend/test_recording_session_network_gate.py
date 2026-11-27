# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The coordinator's network gate: asked at admission, after storage opens and before activation, never during a
take."""

from __future__ import annotations

import pytest

from egocap_web.recording_session import CoordinatorBusyError, RecordingSessionCoordinator
from recording_service import FieldSpec, NumericSourceSpec, RecordingPhase, ServiceConfig

NUMERIC = NumericSourceSpec("s", (FieldSpec("v", "float64"),))


class Gate:
    def __init__(self, reason: str | None = None) -> None:
        self.reason = reason
        self.calls = 0

    def __call__(self) -> str | None:
        self.calls += 1
        return self.reason


def make_coordinator(tmp_path, gate) -> RecordingSessionCoordinator:
    def factory() -> ServiceConfig:
        return ServiceConfig(output_root=tmp_path / "out", numeric_sources=(NUMERIC,), video_sources=())

    return RecordingSessionCoordinator(
        factory, queue_capacity=8, readiness_check=lambda: (True, None), network_gate=gate
    )


def count_storage_opens(coordinator, monkeypatch, on_open=None) -> list[bool]:
    opened: list[bool] = []
    original = coordinator._open_storage_if_needed

    async def counting_open():
        opened.append(True)
        service = await original()
        if on_open is not None:
            on_open(service)
        return service

    monkeypatch.setattr(coordinator, "_open_storage_if_needed", counting_open)
    return opened


async def test_a_refusal_at_admission_opens_no_storage_and_starts_no_operation(tmp_path, monkeypatch):
    gate = Gate("the network check has not run yet")
    coordinator = make_coordinator(tmp_path, gate)
    await coordinator.open()
    coordinator.set_task("t")
    opened = count_storage_opens(coordinator, monkeypatch)
    with pytest.raises(CoordinatorBusyError, match="the network check has not run yet"):
        await coordinator.start_episode()
    assert opened == []
    snapshot = coordinator.snapshot()
    assert snapshot.active_operation is None
    assert snapshot.service.phase is RecordingPhase.IDLE
    await coordinator.close()


async def test_a_gate_that_allows_the_take_lets_it_start(tmp_path):
    gate = Gate()
    coordinator = make_coordinator(tmp_path, gate)
    await coordinator.open()
    coordinator.set_task("t")
    await coordinator.start_episode()
    assert coordinator.snapshot().service.phase is RecordingPhase.RECORDING
    assert gate.calls >= 2  # at admission and again after storage opened
    await coordinator.close()


async def test_no_gate_means_no_refusal(tmp_path):
    coordinator = make_coordinator(tmp_path, None)
    await coordinator.open()
    coordinator.set_task("t")
    await coordinator.start_episode()
    assert coordinator.snapshot().service.phase is RecordingPhase.RECORDING
    await coordinator.close()


async def test_a_verdict_that_changes_while_storage_opens_refuses_the_start_with_its_reason(tmp_path, monkeypatch):
    gate = Gate()
    coordinator = make_coordinator(tmp_path, gate)
    await coordinator.open()
    coordinator.set_task("t")

    def flip(_service) -> None:
        gate.reason = "the last network check is out of date; run it again"

    count_storage_opens(coordinator, monkeypatch, on_open=flip)
    with pytest.raises(CoordinatorBusyError, match="superseded while storage was opening: .*out of date"):
        await coordinator.start_episode()
    assert coordinator.snapshot().service.phase is RecordingPhase.IDLE
    await coordinator.close()


async def test_a_verdict_that_changes_before_activation_discards_the_take_with_its_reason(tmp_path, monkeypatch):
    gate = Gate()
    coordinator = make_coordinator(tmp_path, gate)
    await coordinator.open()
    coordinator.set_task("t")

    def flip_when_the_take_opens(service) -> None:
        original = service.start_episode

        async def start(*args, **kwargs):
            episode = await original(*args, **kwargs)
            gate.reason = "the cameras changed since the network check; run it again"
            return episode

        monkeypatch.setattr(service, "start_episode", start)

    count_storage_opens(coordinator, monkeypatch, on_open=flip_when_the_take_opens)
    with pytest.raises(CoordinatorBusyError, match="superseded before activation: .*cameras changed"):
        await coordinator.start_episode()
    snapshot = coordinator.snapshot()
    assert snapshot.service.phase is not RecordingPhase.RECORDING
    assert snapshot.active_operation is None
    await coordinator.close()


async def test_a_gate_change_mid_take_neither_interrupts_it_nor_is_asked_to_stop_it(tmp_path):
    gate = Gate()
    coordinator = make_coordinator(tmp_path, gate)
    await coordinator.open()
    coordinator.set_task("t")
    episode = await coordinator.start_episode()
    calls_when_recording = gate.calls
    gate.reason = "the last network check is out of date; run it again"
    assert coordinator.snapshot().service.phase is RecordingPhase.RECORDING
    await coordinator.stop_episode(episode.episode_id)
    assert gate.calls == calls_when_recording
    assert coordinator.snapshot().service.phase is RecordingPhase.IDLE
    await coordinator.close()
