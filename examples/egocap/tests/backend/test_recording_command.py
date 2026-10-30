# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for recording_command.py: both the structural payload
parsing/validation, and RecordingContext's real command handling against a
REAL RecordingSessionCoordinator (-> real RecordingService, same no-fixture
synthetic source pattern as test_recording_session.py) -- accept/reject
with a real reason, real identity assignment, real totals projected from
real service facts. No open_session/close_session case anywhere here: the
live recording strip only ever rotates batches via new_batch.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from egocap.recording.episode_layout import manifest_path
from egocap_web.recording_command import (
    RecordingCommandError,
    RecordingContext,
    handle_recording_command,
    parse_recording_command,
)
from egocap_web.recording_session import OperationKind, RecordingSessionCoordinator
from recording_service import FieldSpec, NumericSourceSpec, ServiceConfig

NUMERIC = NumericSourceSpec("s", (FieldSpec("v", "float64"),))
ALWAYS_READY = lambda: (True, None)  # noqa: E731

#: The one control_token every test RecordingContext is constructed with,
#: and the default every _command() call presents -- a test that wants to
#: prove the gate actually rejects an unmatched token passes a different
#: value explicitly, rather than this default ever being wrong by itself.
TEST_CONTROL_TOKEN = "<test-token>"


async def _open_context(tmp_path: Path, control_token: str = TEST_CONTROL_TOKEN) -> RecordingContext:
    def factory() -> ServiceConfig:
        return ServiceConfig(output_root=tmp_path / "out", numeric_sources=(NUMERIC,), video_sources=())

    coordinator = RecordingSessionCoordinator(factory, queue_capacity=8, readiness_check=ALWAYS_READY)
    await coordinator.open()
    coordinator.set_task("Pour water.")  # start_episode is rejected until a task is set
    return RecordingContext(coordinator=coordinator, control_token=control_token)


def _command(
    action: str,
    payload: dict,
    request_id: str = "r1",
    operator_token: str = "test-operator",
    control_token: str = TEST_CONTROL_TOKEN,
):
    return parse_recording_command(
        {
            "request_id": request_id,
            "action": action,
            "payload": payload,
            "operator_token": operator_token,
            "control_token": control_token,
        }
    )


# --- parsing/validation -----------------------------------------------------


def test_batch_target_actions_accept_flat_batch_id():
    raw = {"request_id": "r2", "action": "start_episode", "payload": {"batch_id": "b1"}}
    command = parse_recording_command(raw)
    assert command.action == "start_episode"


def test_episode_target_actions_accept_flat_batch_and_episode_id():
    command = parse_recording_command(
        {"request_id": "r3", "action": "stop_episode", "payload": {"batch_id": "b1", "episode_id": "e1"}}
    )
    assert command.action == "stop_episode"


def test_unknown_action_is_rejected():
    with pytest.raises(RecordingCommandError):
        parse_recording_command({"request_id": "r4", "action": "not_a_real_action", "payload": {}})


def test_close_session_is_not_a_known_action():
    with pytest.raises(RecordingCommandError):
        parse_recording_command({"request_id": "r5", "action": "close_session", "payload": {"batch_id": "b1"}})


def test_set_task_missing_task_field_is_rejected():
    raw = {"request_id": "r6", "action": "set_task", "payload": {"batch_id": "b1"}}
    with pytest.raises(RecordingCommandError):
        parse_recording_command(raw)


def test_set_task_empty_task_text_parses_fine_structurally():
    # Empty text is NOT a protocol violation -- see
    # test_set_task_empty_task_text_is_rejected_as_a_real_outcome_not_a_protocol_error
    # below for where it's actually rejected (an ordinary business outcome,
    # never a reason to tear down the connection).
    raw = {"request_id": "r7", "action": "set_task", "payload": {"batch_id": "b1", "task": ""}}
    command = parse_recording_command(raw)
    assert command.payload["task"] == ""


def test_set_task_wrong_typed_task_is_rejected():
    raw = {"request_id": "r8", "action": "set_task", "payload": {"batch_id": "b1", "task": 123}}
    with pytest.raises(RecordingCommandError):
        parse_recording_command(raw)


def test_new_batch_accepts_flat_batch_id():
    command = parse_recording_command({"request_id": "r9", "action": "new_batch", "payload": {"batch_id": "b1"}})
    assert command.action == "new_batch"


# --- RecordingContext real command handling (against a real coordinator/service) ---


async def test_fresh_context_starts_with_a_real_assigned_batch_id(tmp_path: Path):
    context = await _open_context(tmp_path)
    snapshot = context.snapshot()
    assert isinstance(snapshot["batch_id"], str) and snapshot["batch_id"] != ""
    assert snapshot["episode_phase"] == "IDLE"
    assert snapshot["saved_count"] == 0


async def test_two_fresh_contexts_get_distinct_batch_ids(tmp_path: Path):
    a = await _open_context(tmp_path / "a")
    b = await _open_context(tmp_path / "b")
    assert a.batch_id != b.batch_id


async def test_set_task_commits_real_text(tmp_path: Path):
    context = await _open_context(tmp_path)
    bid = context.batch_id
    reply = await handle_recording_command(_command("set_task", {"batch_id": bid, "task": "Pour water."}), context)
    assert reply["outcome"] == "ok"
    assert reply["reason"] is None
    assert reply["snapshot"]["task_text"] == "Pour water."


async def test_set_task_empty_task_text_is_rejected_as_a_real_outcome_not_a_protocol_error(tmp_path: Path):
    context = await _open_context(tmp_path)
    bid = context.batch_id
    reply = await handle_recording_command(_command("set_task", {"batch_id": bid, "task": ""}), context)
    # An ordinary rejected business outcome -- parse_recording_command
    # itself must never raise for this (that would be a protocol-level
    # close of the whole connection for a routine, recoverable mistake).
    assert reply["outcome"] == "rejected"
    assert reply["reason"] is not None
    assert reply["snapshot"]["task_text"] == "Pour water."  # unchanged, not replaced by empty


async def test_set_task_whitespace_only_task_text_is_rejected(tmp_path: Path):
    # Whitespace-only text is not a real instruction either, same as "".
    context = await _open_context(tmp_path)
    bid = context.batch_id
    reply = await handle_recording_command(_command("set_task", {"batch_id": bid, "task": "   "}), context)
    assert reply["outcome"] == "rejected"
    assert reply["reason"] is not None
    assert reply["snapshot"]["task_text"] == "Pour water."  # unchanged, not replaced by whitespace


async def test_start_episode_rejected_when_task_is_whitespace_only(tmp_path: Path):
    context = await _open_context(tmp_path)
    context.coordinator.set_task("   ")  # bypasses set_task's own validator directly
    bid = context.batch_id
    assert context.snapshot()["ready_to_record"] is False
    reply = await handle_recording_command(_command("start_episode", {"batch_id": bid}), context)
    assert reply["outcome"] == "rejected"
    assert reply["reason"] == "set a task before starting an episode"


async def test_set_task_rejected_while_an_episode_is_recording(tmp_path: Path):
    context = await _open_context(tmp_path)
    bid = context.batch_id
    await handle_recording_command(_command("start_episode", {"batch_id": bid}), context)
    reply = await handle_recording_command(
        _command("set_task", {"batch_id": bid, "task": "New task"}, "r2"), context
    )
    assert reply["outcome"] == "rejected"
    assert reply["reason"] is not None
    assert reply["snapshot"]["task_text"] == "Pour water."  # unchanged: the frozen take is not mutated


async def test_start_episode_assigns_a_real_episode_id_and_enters_recording(tmp_path: Path):
    context = await _open_context(tmp_path)
    reply = await handle_recording_command(_command("start_episode", {"batch_id": context.batch_id}), context)
    assert reply["outcome"] == "ok"
    assert reply["snapshot"]["episode_phase"] == "RECORDING"
    episode_id = reply["snapshot"]["current_episode_id"]
    assert episode_id is not None


async def test_start_episode_publishes_elapsed_ms_stop_records_the_final_duration_on_last_result(tmp_path: Path):
    context = await _open_context(tmp_path)
    bid = context.batch_id
    start_reply = await handle_recording_command(_command("start_episode", {"batch_id": bid}), context)
    # A real monotonic fact (immune to wall-clock drift between machines) a
    # fresh viewer can use to compute genuine elapsed time itself, not
    # something each viewer invents locally.
    elapsed_ms = start_reply["snapshot"]["current_episode_elapsed_ms"]
    assert isinstance(elapsed_ms, int)
    assert elapsed_ms >= 0

    episode_id = start_reply["snapshot"]["current_episode_id"]
    time.sleep(0.01)
    stop_reply = await handle_recording_command(
        _command("stop_episode", {"batch_id": bid, "episode_id": episode_id}, "r2"), context
    )
    # No longer "live" -- idle again.
    assert stop_reply["snapshot"]["current_episode_elapsed_ms"] is None
    # The finished episode's own duration is carried on the existing
    # last_result fact (alongside its episode_id/disposition) instead of a
    # second parallel "last completed" record -- just the data the UI
    # actually renders, so a just-finished take's duration stays visible
    # instead of vanishing the instant recording stops.
    last_result = stop_reply["snapshot"]["last_result"]
    assert last_result["episode_id"] == episode_id
    # The real service's own duration -- the actual submitted samples'
    # capture_time span, not wall-clock time between commands (this test
    # submits no samples, so a genuine, non-fabricated 0 is correct here).
    duration_ms = last_result["duration_ms"]
    assert isinstance(duration_ms, int)
    assert duration_ms >= 0


async def test_start_episode_rejected_when_already_recording(tmp_path: Path):
    context = await _open_context(tmp_path)
    bid = context.batch_id
    await handle_recording_command(_command("start_episode", {"batch_id": bid}), context)
    reply = await handle_recording_command(_command("start_episode", {"batch_id": bid}, "r2"), context)
    assert reply["outcome"] == "rejected"
    assert reply["reason"] is not None


async def test_stop_episode_increments_saved_count_and_sets_kept_result(tmp_path: Path):
    context = await _open_context(tmp_path)
    bid = context.batch_id
    start_reply = await handle_recording_command(_command("start_episode", {"batch_id": bid}), context)
    episode_id = start_reply["snapshot"]["current_episode_id"]
    stop_raw = {"batch_id": bid, "episode_id": episode_id}
    reply = await handle_recording_command(_command("stop_episode", stop_raw, "r2"), context)
    assert reply["outcome"] == "ok"
    assert reply["snapshot"]["saved_count"] == 1
    assert reply["snapshot"]["episode_phase"] == "IDLE"
    last_result = reply["snapshot"]["last_result"]
    assert last_result["episode_id"] == episode_id
    assert last_result["disposition"] == "KEPT"
    assert last_result["issues"] == []
    assert isinstance(last_result["duration_ms"], int)


async def test_stop_episode_rejected_when_nothing_is_recording(tmp_path: Path):
    context = await _open_context(tmp_path)
    raw = {"batch_id": context.batch_id, "episode_id": "e1"}
    reply = await handle_recording_command(_command("stop_episode", raw), context)
    assert reply["outcome"] == "rejected"
    assert context.snapshot()["saved_count"] == 0


async def test_stop_episode_rejected_for_a_mismatched_episode_id(tmp_path: Path):
    context = await _open_context(tmp_path)
    bid = context.batch_id
    await handle_recording_command(_command("start_episode", {"batch_id": bid}), context)
    reply = await handle_recording_command(
        _command("stop_episode", {"batch_id": bid, "episode_id": "not-the-real-one"}, "r2"), context
    )
    assert reply["outcome"] == "rejected"


async def test_discard_episode_does_not_increment_saved_count(tmp_path: Path):
    context = await _open_context(tmp_path)
    bid = context.batch_id
    start_reply = await handle_recording_command(_command("start_episode", {"batch_id": bid}), context)
    episode_id = start_reply["snapshot"]["current_episode_id"]
    reply = await handle_recording_command(
        _command("discard_episode", {"batch_id": bid, "episode_id": episode_id}, "r2"), context
    )
    assert reply["outcome"] == "ok"
    assert reply["snapshot"]["saved_count"] == 0
    assert reply["snapshot"]["last_result"]["disposition"] == "DISCARDED"


async def test_new_batch_assigns_a_fresh_id_and_resets_totals_but_preserves_task_text(tmp_path: Path):
    context = await _open_context(tmp_path)
    old_bid = context.batch_id
    await handle_recording_command(_command("set_task", {"batch_id": old_bid, "task": "Pour water."}), context)
    start_reply = await handle_recording_command(_command("start_episode", {"batch_id": old_bid}, "r2"), context)
    episode_id = start_reply["snapshot"]["current_episode_id"]
    stop_raw = {"batch_id": old_bid, "episode_id": episode_id}
    await handle_recording_command(_command("stop_episode", stop_raw, "r3"), context)
    assert context.snapshot()["saved_count"] == 1

    reply = await handle_recording_command(_command("new_batch", {"batch_id": old_bid}, "r4"), context)
    assert reply["outcome"] == "ok"
    new_bid = reply["snapshot"]["batch_id"]
    assert new_bid != old_bid
    assert reply["snapshot"]["saved_count"] == 0
    assert reply["snapshot"]["last_result"] is None
    assert reply["snapshot"]["task_text"] == "Pour water."  # preserved, not cleared


async def test_new_batch_rejected_while_an_episode_is_active(tmp_path: Path):
    context = await _open_context(tmp_path)
    bid = context.batch_id
    await handle_recording_command(_command("start_episode", {"batch_id": bid}), context)
    reply = await handle_recording_command(_command("new_batch", {"batch_id": bid}, "r2"), context)
    assert reply["outcome"] == "rejected"
    assert reply["reason"] is not None
    # Nothing was discarded implicitly -- still the same active episode.
    assert reply["snapshot"]["episode_phase"] == "RECORDING"
    assert reply["snapshot"]["batch_id"] == bid


async def test_a_second_requests_new_batch_against_a_now_stale_batch_id_is_rejected(tmp_path: Path):
    context = await _open_context(tmp_path)
    old_bid = context.batch_id
    await handle_recording_command(_command("new_batch", {"batch_id": old_bid}), context)
    # A second, concurrent request still targeting the now-superseded id.
    reply = await handle_recording_command(_command("new_batch", {"batch_id": old_bid}, "r2"), context)
    assert reply["outcome"] == "rejected"
    assert reply["reason"] is not None


async def test_commands_for_a_different_batch_id_are_rejected(tmp_path: Path):
    context = await _open_context(tmp_path)
    reply = await handle_recording_command(_command("start_episode", {"batch_id": "not-the-real-batch"}), context)
    assert reply["outcome"] == "rejected"


async def test_ready_to_record_reflects_command_admission_and_real_device_readiness(tmp_path: Path):
    context = await _open_context(tmp_path)
    bid = context.batch_id
    # nothing active, and the injected readiness check (ALWAYS_READY) says
    # yes -- a start_episode would be admitted right now
    assert context.snapshot()["ready_to_record"] is True
    await handle_recording_command(_command("start_episode", {"batch_id": bid}), context)
    # an episode is already active -- a second start_episode would be rejected
    assert context.snapshot()["ready_to_record"] is False


async def test_ready_to_record_reflects_a_real_readiness_check_failure(tmp_path: Path):
    """Not just phase==IDLE -- a real 'no camera selected'/stale-input
    rejection from the injected readiness check must also be reflected,
    since that's what start_episode itself would actually enforce."""

    def factory() -> ServiceConfig:
        return ServiceConfig(output_root=tmp_path / "out", numeric_sources=(NUMERIC,), video_sources=())

    not_ready = lambda: (False, "no camera selected")  # noqa: E731
    coordinator = RecordingSessionCoordinator(factory, queue_capacity=8, readiness_check=not_ready)
    context = RecordingContext(coordinator=coordinator, control_token=TEST_CONTROL_TOKEN)
    assert context.snapshot()["ready_to_record"] is False


async def test_apply_survives_requester_cancellation_projection_still_lands(tmp_path: Path):
    """Regression: the wrapper's own projection update (saved_count/
    last_result/elapsed-basis/revision) must not be skipped just because
    the REQUESTING coroutine was cancelled mid-operation -- the real
    coordinator operation (and this wrapper's bookkeeping of its result)
    both keep running to completion regardless."""
    context = await _open_context(tmp_path)
    bid = context.batch_id
    start_reply = await handle_recording_command(_command("start_episode", {"batch_id": bid}), context)
    episode_id = start_reply["snapshot"]["current_episode_id"]

    stop_command = _command("stop_episode", {"batch_id": bid, "episode_id": episode_id}, "r2")
    task = asyncio.create_task(handle_recording_command(stop_command, context))
    await asyncio.sleep(0)  # let it begin
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    for _ in range(200):
        if context.snapshot()["saved_count"] == 1:
            break
        await asyncio.sleep(0.01)
    assert context.snapshot()["saved_count"] == 1
    assert context.snapshot()["last_result"]["disposition"] == "KEPT"


async def test_revision_changes_mid_operation_not_only_at_completion(tmp_path: Path):
    """The sender loop must be able to notice a real phase change
    (STARTING/RECORDING/...) as it happens, not only once a whole command
    finishes -- revision is derived from live coordinator facts, not just
    a counter bumped at completion."""
    context = await _open_context(tmp_path)
    before = context.revision
    await handle_recording_command(_command("start_episode", {"batch_id": context.batch_id}), context)
    after = context.revision
    assert after != before


async def test_revision_advances_while_an_operation_is_still_in_flight(tmp_path: Path):
    """Real, direct proof: revision must already reflect active_operation
    == STARTING before start_episode's own on_episode_started callback
    ever fires (that callback only runs once the ingress is actually
    constructed, near the END of the operation) -- snapshot()'s own
    decision-state comparison is what catches this, not a callback."""
    context = await _open_context(tmp_path)
    context.snapshot()  # establish the baseline observed decision-state
    before = context.revision
    task = asyncio.create_task(
        handle_recording_command(_command("start_episode", {"batch_id": context.batch_id}), context)
    )
    await asyncio.sleep(0)  # let the operation begin and register itself
    # active_operation flips to STARTING synchronously, before the real
    # service's own phase transitions (that happens once the owned task
    # actually runs service.start_episode()) -- the real, first-observable
    # mid-flight signal is active_operation, not episode_phase yet.
    assert context.coordinator.snapshot().active_operation is OperationKind.STARTING
    context.snapshot()  # the call that actually advances revision
    mid_flight_revision = context.revision
    await task
    assert mid_flight_revision != before


async def test_start_episode_rejected_until_a_task_is_set_and_snapshot_carries_revision(tmp_path: Path):
    context = await _open_context(tmp_path)
    context.coordinator.set_task("")  # back to no task
    bid = context.batch_id
    assert context.snapshot()["ready_to_record"] is False
    reply = await handle_recording_command(_command("start_episode", {"batch_id": bid}), context)
    assert reply["outcome"] == "rejected"
    assert reply["reason"] == "set a task before starting an episode"
    before = reply["snapshot"]["revision"]
    reply = await handle_recording_command(_command("set_task", {"batch_id": bid, "task": "Pour water."}), context)
    assert reply["snapshot"]["revision"] > before
    assert reply["snapshot"]["ready_to_record"] is True


async def test_refresh_advances_revision_when_readiness_changes_without_any_snapshot_call(tmp_path: Path):
    state = {"ready": False}

    def factory() -> ServiceConfig:
        return ServiceConfig(output_root=tmp_path / "out", numeric_sources=(NUMERIC,), video_sources=())

    coordinator = RecordingSessionCoordinator(
        factory,
        queue_capacity=8,
        readiness_check=lambda: (state["ready"], None if state["ready"] else "no camera"),
    )
    context = RecordingContext(coordinator=coordinator, control_token=TEST_CONTROL_TOKEN)
    context.refresh()
    before = context.revision
    context.refresh()
    assert context.revision == before  # nothing changed
    state["ready"] = True
    context.refresh()
    assert context.revision == before + 1


async def test_an_unexpected_failure_is_a_rejected_reply_not_a_dead_receiver(tmp_path: Path, monkeypatch):
    context = await _open_context(tmp_path)

    async def boom(operator: str = ""):
        raise OSError("disk full")

    monkeypatch.setattr(context.coordinator, "start_episode", boom)
    reply = await handle_recording_command(_command("start_episode", {"batch_id": context.batch_id}), context)
    assert reply["outcome"] == "rejected" and "disk full" in reply["reason"]


# --- control-token gate -------------------------------------------------------


@pytest.mark.parametrize(
    "control_token",
    [
        None,
        "",
        123,
        "x" * (len(TEST_CONTROL_TOKEN) - 1),
        "x" * len(TEST_CONTROL_TOKEN),  # same length as the real token, but wrong
        "é" * len(TEST_CONTROL_TOKEN),
    ],
    ids=["missing", "empty", "wrong_type", "wrong_length", "incorrect", "non_ascii"],
)
async def test_a_mutating_command_with_an_invalid_control_token_is_rejected_and_has_no_effect(
    tmp_path: Path, control_token
):
    context = await _open_context(tmp_path)
    bid = context.batch_id
    reply = await handle_recording_command(
        _command("set_task", {"batch_id": bid, "task": "New task"}, control_token=control_token), context
    )
    assert reply["outcome"] == "rejected"
    assert "control token" in reply["reason"].lower()
    assert reply["snapshot"]["task_text"] == "Pour water."  # unchanged -- the coordinator was never reached


@pytest.mark.parametrize(
    "action,payload",
    [
        ("set_task", {"task": "New task"}),
        ("start_episode", {}),
        ("stop_episode", {"episode_id": "whatever"}),
        ("discard_episode", {"episode_id": "whatever"}),
        ("new_batch", {}),
    ],
)
async def test_every_mutating_action_requires_the_real_control_token(tmp_path: Path, action: str, payload: dict):
    """The gate is unconditional (not a per-action allowlist that could
    silently miss a future action) -- this proves it actually covers
    today's five, not just the one exercised above."""
    context = await _open_context(tmp_path)
    # Same length as TEST_CONTROL_TOKEN -- a mismatched length would only
    # exercise the length check, not compare_digest's real false branch.
    wrong_token = TEST_CONTROL_TOKEN[:-1] + "x"
    reply = await handle_recording_command(
        _command(action, {"batch_id": context.batch_id, **payload}, control_token=wrong_token), context
    )
    assert reply["outcome"] == "rejected"
    assert "control token" in reply["reason"].lower()


async def test_a_control_token_from_a_different_backend_run_is_rejected(tmp_path: Path):
    """Each backend process generates its own control_token (see app.py's
    lifespan); this proves a stale token from an earlier run doesn't carry
    over -- simulated here as a second RecordingContext with a different
    token, since that's the actual unit the real token lives on. The two
    tokens are deliberately the same LENGTH as each other: a mismatched
    length would only prove the length check works, not real cross-context
    isolation via the actual secret comparison."""
    first_run = await _open_context(tmp_path / "first")
    second_run_token = "<second-tok>"  # same length as TEST_CONTROL_TOKEN (12 chars)
    assert len(second_run_token) == len(first_run.control_token)
    second_run = await _open_context(tmp_path / "second", control_token=second_run_token)

    # The first run's own token is rejected against the second run's context...
    reply = await handle_recording_command(
        _command("new_batch", {"batch_id": second_run.batch_id}, control_token=first_run.control_token), second_run
    )
    assert reply["outcome"] == "rejected"
    assert "control token" in reply["reason"].lower()

    # ...while the second run's OWN token is accepted on that same context,
    # proving the first attempt was rejected because it's the wrong secret,
    # not because every command against a fresh context fails for some
    # other reason.
    reply = await handle_recording_command(
        _command("new_batch", {"batch_id": second_run.batch_id}, control_token=second_run_token), second_run
    )
    assert reply["outcome"] == "ok"


# --- operator gate -----------------------------------------------------------


@pytest.mark.parametrize(
    "operator_token",
    [None, "", "   ", 123, "x" * 129],
    ids=["missing", "empty", "whitespace_only", "wrong_type", "too_long"],
)
async def test_a_mutating_command_without_a_valid_operator_is_rejected_and_has_no_effect(
    tmp_path: Path, operator_token
):
    context = await _open_context(tmp_path)
    bid = context.batch_id
    reply = await handle_recording_command(
        parse_recording_command(
            {
                "request_id": "r1",
                "action": "set_task",
                "payload": {"batch_id": bid, "task": "New task"},
                "operator_token": operator_token,
                "control_token": TEST_CONTROL_TOKEN,
            }
        ),
        context,
    )
    assert reply["outcome"] == "rejected"
    assert "operator" in reply["reason"].lower()
    assert reply["snapshot"]["task_text"] == "Pour water."  # unchanged -- the coordinator was never reached


@pytest.mark.parametrize(
    "action,payload",
    [
        ("set_task", {"task": "New task"}),
        ("start_episode", {}),
        ("stop_episode", {"episode_id": "whatever"}),
        ("discard_episode", {"episode_id": "whatever"}),
        ("new_batch", {}),
    ],
)
async def test_every_mutating_action_is_gated_not_just_some(tmp_path: Path, action: str, payload: dict):
    """The gate in apply() is unconditional (not a per-action allowlist
    that could silently miss a future action) -- this proves it actually
    covers today's five, not just the one exercised above."""
    context = await _open_context(tmp_path)
    reply = await handle_recording_command(
        parse_recording_command(
            {
                "request_id": "r1",
                "action": action,
                "payload": {"batch_id": context.batch_id, **payload},
                "operator_token": None,
                "control_token": TEST_CONTROL_TOKEN,
            }
        ),
        context,
    )
    assert reply["outcome"] == "rejected"
    assert "operator" in reply["reason"].lower()


def _manifest_operator(tmp_path: Path, session_id: str, episode_index: int) -> str:
    manifest = json.loads(manifest_path(tmp_path / "out" / session_id, episode_index).read_text())
    return manifest["operator"]


async def test_operator_name_is_normalized_and_whitespace_collapsed_before_reaching_the_take(tmp_path: Path):
    context = await _open_context(tmp_path)
    raw_operator_name = "  Alice   Smith  "
    reply = await handle_recording_command(
        parse_recording_command(
            {
                "request_id": "r1",
                "action": "start_episode",
                "payload": {"batch_id": context.batch_id},
                "operator_token": raw_operator_name,
                "control_token": TEST_CONTROL_TOKEN,
            }
        ),
        context,
    )
    assert reply["outcome"] == "ok"
    session_id = reply["snapshot"]["current_episode_id"].split("/")[0]
    assert _manifest_operator(tmp_path, session_id, 0) == "Alice_Smith"


async def test_a_rejected_start_never_overwrites_the_frozen_operator_of_an_active_take(tmp_path: Path):
    context = await _open_context(tmp_path)
    bid = context.batch_id
    first_operator, second_operator = "alice", "mallory"
    first = await handle_recording_command(
        _command("start_episode", {"batch_id": bid}, operator_token=first_operator), context
    )
    assert first["outcome"] == "ok"
    session_id = first["snapshot"]["current_episode_id"].split("/")[0]
    # A second start while one is already RECORDING is rejected -- it must
    # not be able to smuggle a different operator into the active take,
    # NOR into the coordinator's own committed context (checking only the
    # manifest would miss this: the active take's manifest was already
    # frozen before the rejected call, so it would stay "alice" even if
    # the bug this guards against were reintroduced).
    second = await handle_recording_command(
        _command("start_episode", {"batch_id": bid}, "r2", operator_token=second_operator), context
    )
    assert second["outcome"] == "rejected"
    assert _manifest_operator(tmp_path, session_id, 0) == first_operator
    assert context.coordinator.snapshot().context.operator == ""


async def test_auto_fill_operator_name_defaults_off_and_reflects_the_constructor_argument(tmp_path: Path):
    off = await _open_context(tmp_path)
    assert off.snapshot()["auto_fill_operator_name"] is False

    def factory() -> ServiceConfig:
        return ServiceConfig(output_root=tmp_path / "out2", numeric_sources=(NUMERIC,), video_sources=())

    coordinator = RecordingSessionCoordinator(factory, queue_capacity=8, readiness_check=ALWAYS_READY)
    await coordinator.open()
    on = RecordingContext(coordinator=coordinator, control_token=TEST_CONTROL_TOKEN, auto_fill_operator_name=True)
    assert on.snapshot()["auto_fill_operator_name"] is True
