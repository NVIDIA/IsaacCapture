# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""RecordingContext and the network check: published readiness, the run_network_check action."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

from egocap_web.recording_command import (
    RecordingCommandError,
    RecordingContext,
    handle_recording_command,
    parse_recording_command,
)
from egocap_web.recording_session import OperationKind, RecordingSessionCoordinator
from recording_service import FieldSpec, NumericSourceSpec, RecordingPhase, ServiceConfig

NUMERIC = NumericSourceSpec("s", (FieldSpec("v", "float64"),))
TOKEN = "<test-token>"
WAIT = "the network check is still running"


def _command(action: str, payload: dict, *, operator_token="op", control_token=TOKEN):
    return parse_recording_command(
        {
            "request_id": "r1",
            "action": action,
            "payload": payload,
            "operator_token": operator_token,
            "control_token": control_token,
        }
    )


async def _context(tmp_path: Path, *, readiness=lambda: (True, None), **kwargs) -> RecordingContext:
    def factory() -> ServiceConfig:
        return ServiceConfig(output_root=tmp_path / "out", numeric_sources=(NUMERIC,), video_sources=())

    coordinator = RecordingSessionCoordinator(
        factory,
        queue_capacity=8,
        readiness_check=readiness,
        network_gate=kwargs.get("network_gate"),
    )
    await coordinator.open()
    coordinator.set_task("Pour water.")
    return RecordingContext(coordinator=coordinator, control_token=TOKEN, **kwargs)


async def test_the_gate_reason_makes_the_take_not_ready_and_is_published(tmp_path):
    context = await _context(tmp_path, network_gate=lambda: WAIT)
    snapshot = context.snapshot()
    assert snapshot["ready_to_record"] is False
    assert snapshot["not_ready_reason"] == WAIT


async def test_a_gate_that_allows_the_take_publishes_ready(tmp_path):
    context = await _context(tmp_path, network_gate=lambda: None)
    snapshot = context.snapshot()
    assert snapshot["ready_to_record"] is True and snapshot["not_ready_reason"] is None


async def test_a_gate_clearing_is_announced_without_any_command(tmp_path):
    reasons: list[str | None] = [WAIT]
    context = await _context(tmp_path, network_gate=lambda: reasons[0])
    context.snapshot()
    revision = context.revision
    reasons[0] = None
    context.refresh()
    assert context.revision == revision + 1
    assert context.snapshot()["ready_to_record"] is True


async def test_a_refused_start_replies_with_the_gate_reason_and_records_nothing(tmp_path):
    context = await _context(tmp_path, network_gate=lambda: WAIT)
    reply = await handle_recording_command(_command("start_episode", {"batch_id": context.batch_id}), context)
    assert (reply["outcome"], reply["reason"]) == ("rejected", WAIT)
    assert reply["snapshot"]["episode_phase"] == "IDLE"


async def test_station_setup_outranks_the_network_gate(tmp_path):
    context = await _context(tmp_path, network_gate=lambda: WAIT, setup_gate=lambda: "select a station")
    assert context.snapshot()["not_ready_reason"] == "select a station"


async def test_a_capture_reason_outranks_the_network_gate(tmp_path):
    context = await _context(
        tmp_path,
        readiness=lambda: (False, "capture is not running"),
        network_gate=lambda: WAIT,
        setup_gate=lambda: None,
    )
    assert context.snapshot()["not_ready_reason"] == "capture is not running"


async def test_a_gate_change_mid_take_does_not_stop_the_take(tmp_path):
    reasons: list[str | None] = [None]
    context = await _context(tmp_path, network_gate=lambda: reasons[0])
    started = await handle_recording_command(_command("start_episode", {"batch_id": context.batch_id}), context)
    assert started["outcome"] == "ok"
    reasons[0] = WAIT
    episode_id = started["snapshot"]["current_episode_id"]
    stopped = await handle_recording_command(
        _command("stop_episode", {"batch_id": context.batch_id, "episode_id": episode_id}), context
    )
    assert (stopped["outcome"], stopped["snapshot"]["last_result"]["disposition"]) == ("ok", "KEPT")


async def test_the_snapshot_has_no_network_field_of_its_own(tmp_path):
    context = await _context(tmp_path, network_gate=lambda: WAIT)
    snapshot = context.snapshot()
    assert not [key for key in snapshot if "network" in key or "connection" in key]


async def test_the_network_gate_is_not_station_downtime(tmp_path):
    from types import SimpleNamespace

    from egocap_web import app as app_module

    context = await _context(tmp_path, network_gate=lambda: WAIT)
    assert context.snapshot()["ready_to_record"] is False
    stations = SimpleNamespace(label=lambda: "Bench A")
    state = app_module._station_state(context.coordinator, context, stations)
    assert state == ("Bench A", app_module.READY, None)


# --- run_network_check ------------------------------------------------------


def test_run_network_check_parses_like_new_batch():
    command = _command("run_network_check", {"batch_id": "b1"})
    assert command.action == "run_network_check"
    with pytest.raises(RecordingCommandError):
        parse_recording_command({"request_id": "r", "action": "run_network_check", "payload": {}})


async def test_run_network_check_awaits_the_callable_and_replies_ok(tmp_path):
    calls: list[str] = []

    async def run() -> None:
        calls.append("run")

    context = await _context(tmp_path, run_network_check=run)
    reply = await handle_recording_command(_command("run_network_check", {"batch_id": context.batch_id}), context)
    assert (reply["outcome"], reply["reason"]) == ("ok", None)
    assert calls == ["run"]


async def test_run_network_check_is_rejected_when_the_station_has_none(tmp_path):
    context = await _context(tmp_path)
    reply = await handle_recording_command(_command("run_network_check", {"batch_id": context.batch_id}), context)
    assert (reply["outcome"], reply["reason"]) == ("rejected", "this station has no network check to run")


async def test_run_network_check_for_another_batch_is_rejected_and_not_run(tmp_path):
    calls: list[str] = []

    async def run() -> None:
        calls.append("run")

    context = await _context(tmp_path, run_network_check=run)
    reply = await handle_recording_command(_command("run_network_check", {"batch_id": "stale"}), context)
    assert reply["outcome"] == "rejected"
    assert calls == []


@pytest.mark.parametrize(
    "kwargs,needle",
    [({"control_token": "x" * len(TOKEN)}, "control token"), ({"operator_token": None}, "operator")],
    ids=["control_token", "operator"],
)
async def test_run_network_check_needs_the_control_token_and_an_operator(tmp_path, kwargs, needle):
    calls: list[str] = []

    async def run() -> None:
        calls.append("run")

    context = await _context(tmp_path, run_network_check=run)
    reply = await handle_recording_command(
        _command("run_network_check", {"batch_id": context.batch_id}, **kwargs), context
    )
    assert reply["outcome"] == "rejected"
    assert needle in reply["reason"].lower()
    assert calls == []


async def test_a_failing_run_network_check_is_a_rejected_reply(tmp_path):
    async def run() -> None:
        raise RuntimeError("run_network_check needs an accepted capture session")

    context = await _context(tmp_path, run_network_check=run)
    reply = await handle_recording_command(_command("run_network_check", {"batch_id": context.batch_id}), context)
    assert reply["outcome"] == "rejected"


async def _run_guarded(tmp_path):
    calls: list[str] = []

    async def run() -> None:
        calls.append("run")

    return await _context(tmp_path, run_network_check=run), calls


BUSY = "finish or stop the current take before running the network check"


async def test_run_network_check_is_rejected_while_a_take_is_recording(tmp_path):
    context, calls = await _run_guarded(tmp_path)
    started = await handle_recording_command(_command("start_episode", {"batch_id": context.batch_id}), context)
    assert started["outcome"] == "ok"
    reply = await handle_recording_command(_command("run_network_check", {"batch_id": context.batch_id}), context)
    assert (reply["outcome"], reply["reason"]) == ("rejected", BUSY)
    assert calls == []


async def test_run_network_check_is_rejected_while_an_operation_is_in_flight(tmp_path, monkeypatch):
    context, calls = await _run_guarded(tmp_path)
    real = context.coordinator.snapshot()
    busy = dataclasses.replace(real, active_operation=OperationKind.STARTING)
    monkeypatch.setattr(context.coordinator, "snapshot", lambda: busy)
    reply = await handle_recording_command(_command("run_network_check", {"batch_id": context.batch_id}), context)
    assert (reply["outcome"], reply["reason"]) == ("rejected", BUSY)
    assert calls == []


async def test_run_network_check_is_rejected_with_an_interrupted_take(tmp_path, monkeypatch):
    context, calls = await _run_guarded(tmp_path)
    real = context.coordinator.snapshot()
    service = dataclasses.replace(
        real.service, phase=RecordingPhase.RECOVERY_REQUIRED, episode=SimpleNamespace(episode_id="e1")
    )
    interrupted = dataclasses.replace(real, service=service)
    monkeypatch.setattr(context.coordinator, "snapshot", lambda: interrupted)
    reply = await handle_recording_command(_command("run_network_check", {"batch_id": context.batch_id}), context)
    assert (reply["outcome"], reply["reason"]) == ("rejected", BUSY)
    assert calls == []


async def test_run_network_check_is_allowed_again_once_the_take_has_stopped(tmp_path):
    context, calls = await _run_guarded(tmp_path)
    started = await handle_recording_command(_command("start_episode", {"batch_id": context.batch_id}), context)
    episode_id = started["snapshot"]["current_episode_id"]
    await handle_recording_command(
        _command("stop_episode", {"batch_id": context.batch_id, "episode_id": episode_id}), context
    )
    reply = await handle_recording_command(_command("run_network_check", {"batch_id": context.batch_id}), context)
    assert reply["outcome"] == "ok"
    assert calls == ["run"]


async def test_a_run_network_check_that_cannot_reach_capture_server_is_rejected_without_recording_wording(
    tmp_path,
):
    async def run() -> None:
        raise ConnectionError("socket closed")

    context = await _context(tmp_path, run_network_check=run)
    reply = await handle_recording_command(_command("run_network_check", {"batch_id": context.batch_id}), context)
    assert (reply["outcome"], reply["reason"]) == (
        "rejected",
        "the network test could not be requested from CaptureServer",
    )
    assert "recording failed" not in reply["reason"].lower()
