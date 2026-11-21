# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ownership, stage and confirmation rules of the controller, over a real fake-worker process."""

from __future__ import annotations

import asyncio

import pytest
from conftest import commands, wait_for
from isaaccapture_examples.manus_calibration.controller import (
    CalibrationController,
    NotOwner,
    Stage,
)
from isaaccapture_manus_calibration import protocol as proto
from isaaccapture_manus_calibration.worker import AttemptError, OutcomeKind

HOLD = {
    "steps": [
        {"title": "A", "instruction": "a", "time": 0.0},
        {"title": "B", "instruction": "b", "time": 0.0},
    ]
}
ONE_HOLD = {"steps": [{"title": "A", "instruction": "a", "time": 0.0}]}
LEFT = (proto.Side.LEFT,)
BOTH = (proto.Side.LEFT, proto.Side.RIGHT)
OWNER = "owner-client-0001"
OTHER = "other-client-0002"


def build(make_worker, scenario=None, **settings):
    """A controller over a fake worker, plus the views it announced."""
    worker, store = make_worker({**HOLD, **(scenario or {})}, **settings)
    views = []
    return CalibrationController(worker, views.append), store, views


async def at_prompt(controller: CalibrationController) -> None:
    await wait_for(
        lambda: (
            controller.view().stage is Stage.AWAITING_CONFIRMATION
            and controller.view().prompt is not None
        )
    )


async def finish(controller: CalibrationController, client: str = OWNER):
    """Confirm each prompt as ``client`` until the attempt settles; returns the last view."""
    confirmed: set[str] = set()
    async with asyncio.timeout(30):
        while controller.view().stage is not Stage.FINISHED:
            view = controller.view()
            prompt = view.prompt
            if (
                prompt
                and prompt.token not in confirmed
                and view.stage is Stage.AWAITING_CONFIRMATION
            ):
                confirmed.add(prompt.token)
                await controller.confirm(client, prompt.token)
            await asyncio.sleep(0.01)
    return controller.view()


def is_subsequence(wanted, seen) -> bool:
    iterator = iter(seen)
    return all(any(item == other for other in iterator) for item in wanted)


def test_a_new_controller_is_idle(make_worker):
    controller, _store, views = build(make_worker)
    view = controller.view()
    assert view.stage is Stage.IDLE
    assert view.owner is None
    assert view.prompt is None
    assert view.outcome is None
    assert controller.active is False
    assert views == []


def test_start_assigns_the_owner_and_reaches_the_first_prompt(make_worker):
    async def scenario():
        controller, _store, _views = build(make_worker)
        await controller.start(OWNER, LEFT)
        await at_prompt(controller)
        view = controller.view()
        assert controller.active is True
        assert view.owner == OWNER
        assert view.sides == LEFT
        assert view.prompt.side is proto.Side.LEFT
        assert view.prompt.index == 0
        assert [g.side for g in view.gloves] == [proto.Side.LEFT]
        await controller.aclose()

    asyncio.run(scenario())


def test_the_sides_are_reported_in_worker_order_whatever_order_was_asked(make_worker):
    async def scenario():
        controller, _store, _views = build(make_worker)
        await controller.start(OWNER, (proto.Side.RIGHT, proto.Side.LEFT))
        assert controller.view().sides == BOTH
        await controller.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("second", [OWNER, OTHER])
def test_a_second_start_is_refused_and_changes_nothing(make_worker, second):
    async def scenario():
        controller, _store, _views = build(make_worker)
        await controller.start(OWNER, LEFT)
        await at_prompt(controller)
        before = controller.view()
        with pytest.raises(AttemptError, match="already active"):
            await controller.start(second, BOTH)
        after = controller.view()
        assert after.owner == OWNER
        assert after.sides == before.sides
        assert after.prompt == before.prompt
        await controller.aclose()

    asyncio.run(scenario())


def test_only_the_owner_may_confirm_or_cancel(make_worker, commands_log):
    async def scenario():
        controller, _store, _views = build(make_worker)
        await controller.start(OWNER, LEFT)
        await at_prompt(controller)
        token = controller.view().prompt.token
        with pytest.raises(NotOwner):
            await controller.confirm(OTHER, token)
        with pytest.raises(NotOwner):
            await controller.cancel(OTHER)
        view = controller.view()
        assert view.stage is Stage.AWAITING_CONFIRMATION
        assert view.confirm_pending is False
        assert "confirm" not in commands(commands_log)
        assert "cancel" not in commands(commands_log)
        await controller.aclose()

    asyncio.run(scenario())


def test_confirm_and_cancel_with_nothing_active_are_refused(make_worker):
    async def scenario():
        controller, _store, _views = build(make_worker)
        with pytest.raises(AttemptError, match="no calibration attempt"):
            await controller.confirm(OWNER, "left-0")
        with pytest.raises(AttemptError, match="no calibration attempt"):
            await controller.cancel(OWNER)

    asyncio.run(scenario())


@pytest.mark.parametrize("token", ["wrong-token", "", "right-0"])
def test_a_wrong_or_stale_token_is_refused_without_a_step(
    make_worker, commands_log, token
):
    async def scenario():
        controller, _store, _views = build(make_worker)
        await controller.start(OWNER, LEFT)
        await at_prompt(controller)
        with pytest.raises(AttemptError, match="no longer current"):
            await controller.confirm(OWNER, token)
        assert controller.view().confirm_pending is False
        assert "confirm" not in commands(commands_log)
        await controller.aclose()

    asyncio.run(scenario())


def test_a_superseded_prompt_token_is_refused_after_the_next_prompt_appears(
    make_worker, commands_log
):
    async def scenario():
        controller, _store, _views = build(make_worker)
        await controller.start(OWNER, LEFT)
        await at_prompt(controller)
        first = controller.view().prompt
        await controller.confirm(OWNER, first.token)
        await wait_for(
            lambda: (
                controller.view().prompt is not None
                and controller.view().prompt.token != first.token
                and controller.view().stage is Stage.AWAITING_CONFIRMATION
            )
        )
        sent = commands(commands_log).count("confirm")
        with pytest.raises(AttemptError, match="no longer current"):
            await controller.confirm(OWNER, first.token)
        assert commands(commands_log).count("confirm") == sent
        await controller.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "sides", [LEFT, (proto.Side.RIGHT,), BOTH], ids=["left", "right", "both"]
)
def test_a_full_run_completes_publishes_and_releases_ownership(make_worker, sides):
    async def scenario():
        controller, store, _views = build(make_worker)
        await controller.start(OWNER, sides)
        view = await finish(controller)
        assert view.stage is Stage.FINISHED
        assert view.owner is None
        assert view.prompt is None
        assert view.outcome.kind is OutcomeKind.COMPLETED
        assert {f.side for f in view.outcome.files} == set(sides)
        assert len(store.saved_attempts()) == 1
        assert controller.active is False
        assert "completed" in view.notice

    asyncio.run(scenario())


def test_a_new_attempt_can_start_after_a_finished_one(make_worker):
    async def scenario():
        controller, store, _views = build(make_worker)
        await controller.start(OWNER, LEFT)
        await finish(controller)
        await controller.start(OTHER, LEFT)
        assert controller.view().owner == OTHER
        await finish(controller, OTHER)
        assert len(store.saved_attempts()) == 2

    asyncio.run(scenario())


def test_stages_advance_in_order_and_motion_reports_progress(make_worker):
    async def scenario():
        steps = {"steps": [{"title": "W", "instruction": "w", "time": 0.4}]}
        controller, _store, views = build(make_worker, steps)
        await controller.start(OWNER, LEFT)
        await finish(controller)
        stages = [v.stage for v in views]
        assert is_subsequence(
            [
                Stage.STARTING,
                Stage.DISCOVERING,
                Stage.AWAITING_CONFIRMATION,
                Stage.SAMPLING,
                Stage.FINISHED,
            ],
            stages,
        )
        elapsed = [v.elapsed_s for v in views if v.elapsed_s is not None]
        assert elapsed
        assert max(elapsed) <= 0.4 + 1e-6
        assert elapsed == sorted(elapsed)

    asyncio.run(scenario())


def test_confirm_pending_is_raised_while_the_worker_decides_then_cleared(
    make_worker,
):
    async def scenario():
        controller, _store, views = build(make_worker)
        await controller.start(OWNER, LEFT)
        await at_prompt(controller)
        await controller.confirm(OWNER, controller.view().prompt.token)
        assert any(v.confirm_pending for v in views)
        assert controller.view().confirm_pending is False
        await controller.aclose()

    asyncio.run(scenario())


def test_a_second_confirm_while_one_is_in_flight_sends_only_one_command(
    make_worker, commands_log
):
    async def scenario():
        controller, _store, _views = build(make_worker, ONE_HOLD)
        await controller.start(OWNER, LEFT)
        await at_prompt(controller)
        token = controller.view().prompt.token
        await asyncio.gather(
            controller.confirm(OWNER, token), controller.confirm(OWNER, token)
        )
        assert commands(commands_log).count("confirm") == 1
        await finish(controller)

    asyncio.run(scenario())


def test_a_refused_confirmation_becomes_a_notice_and_can_be_retried(make_worker):
    async def scenario():
        controller, _store, _views = build(
            make_worker, {**ONE_HOLD, "reject_first_confirm": True}
        )
        await controller.start(OWNER, LEFT)
        await at_prompt(controller)
        token = controller.view().prompt.token
        await controller.confirm(OWNER, token)
        view = controller.view()
        assert view.notice
        assert view.confirm_pending is False
        assert view.stage is Stage.AWAITING_CONFIRMATION
        await controller.confirm(OWNER, token)
        view = await finish(controller)
        assert view.outcome.kind is OutcomeKind.COMPLETED

    asyncio.run(scenario())


def test_cancel_by_the_owner_ends_cancelled_and_publishes_nothing(make_worker):
    async def scenario():
        controller, store, views = build(make_worker)
        await controller.start(OWNER, BOTH)
        await at_prompt(controller)
        await controller.cancel(OWNER)
        assert any(v.stage is Stage.STOPPING for v in views)
        await wait_for(lambda: controller.view().stage is Stage.FINISHED)
        view = controller.view()
        assert view.outcome.kind is OutcomeKind.CANCELLED
        assert view.outcome.files == ()
        assert store.saved_attempts() == []
        assert controller.active is False

    asyncio.run(scenario())


def test_only_the_owner_disconnecting_cancels_the_attempt(make_worker):
    async def scenario():
        controller, store, _views = build(make_worker)
        await controller.start(OWNER, LEFT)
        await at_prompt(controller)
        await controller.client_disconnected(OTHER)
        await asyncio.sleep(0.3)
        assert controller.view().stage is Stage.AWAITING_CONFIRMATION
        assert controller.active is True
        await controller.client_disconnected(OWNER)
        await wait_for(lambda: controller.view().stage is Stage.FINISHED)
        assert controller.view().outcome.kind is OutcomeKind.CANCELLED
        assert store.saved_attempts() == []

    asyncio.run(scenario())


def test_a_disconnect_with_nothing_active_is_ignored(make_worker):
    async def scenario():
        controller, _store, views = build(make_worker)
        await controller.client_disconnected(OWNER)
        assert views == []

    asyncio.run(scenario())


def test_a_completed_attempts_files_survive_the_owner_disconnecting_later(
    make_worker,
):
    async def scenario():
        controller, store, _views = build(make_worker)
        await controller.start(OWNER, LEFT)
        await finish(controller)
        await controller.client_disconnected(OWNER)
        assert len(store.saved_attempts()) == 1
        assert controller.view().outcome.kind is OutcomeKind.COMPLETED

    asyncio.run(scenario())


def test_aclose_cancels_an_active_attempt_and_waits_for_the_worker(make_worker):
    async def scenario():
        controller, store, _views = build(make_worker)
        await controller.start(OWNER, LEFT)
        await at_prompt(controller)
        await controller.aclose()
        outcome = controller.view().outcome
        assert outcome.kind is OutcomeKind.CANCELLED
        assert outcome.exit_code == 0
        assert outcome.forced_termination is False
        assert store.saved_attempts() == []

    asyncio.run(scenario())


def test_aclose_with_nothing_active_is_a_no_op(make_worker):
    async def scenario():
        controller, _store, views = build(make_worker)
        await controller.aclose()
        await controller.aclose()
        assert views == []

    asyncio.run(scenario())


def test_a_failed_run_reports_the_failure_and_publishes_nothing(make_worker):
    async def scenario():
        controller, store, _views = build(make_worker, {"fail_at_step": ["left", 0]})
        await controller.start(OWNER, LEFT)
        view = await finish(controller)
        assert view.outcome.kind is OutcomeKind.FAILED
        assert view.outcome.error.code == "step_failed"
        assert "failed" in view.notice
        assert view.outcome.files == ()
        assert store.saved_attempts() == []

    asyncio.run(scenario())


def test_a_worker_that_never_reports_a_result_is_uncertain_not_complete(
    make_worker,
):
    async def scenario():
        controller, store, _views = build(make_worker, {"no_result": True})
        await controller.start(OWNER, LEFT)
        view = await finish(controller)
        assert view.outcome.kind is OutcomeKind.UNCERTAIN
        assert "uncertain" in view.notice
        assert "completed" not in view.notice
        assert store.saved_attempts() == []

    asyncio.run(scenario())


def test_a_worker_that_cannot_start_finishes_failed_and_frees_the_controller(
    tmp_path, make_worker
):
    async def scenario():
        controller, _store, _views = build(make_worker, {"exit_before_ready": True})
        await controller.start(OWNER, LEFT)
        await wait_for(lambda: controller.view().stage is Stage.FINISHED)
        assert controller.view().outcome.kind is not OutcomeKind.COMPLETED
        assert controller.active is False

    asyncio.run(scenario())


def test_an_invalid_start_leaves_the_controller_usable(make_worker):
    async def scenario():
        controller, _store, _views = build(make_worker)
        with pytest.raises(proto.ProtocolError):
            await controller.start(OWNER, ())
        assert controller.active is False
        await controller.start(OWNER, LEFT)
        await controller.aclose()

    asyncio.run(scenario())


def test_owner_attempt_names_the_active_attempt_of_its_owner_only(make_worker):
    async def scenario():
        controller, _store, _views = build(make_worker, ONE_HOLD)
        assert controller.owner_attempt(OWNER) is None
        await controller.start(OWNER, LEFT)
        await at_prompt(controller)
        assert controller.owner_attempt(OWNER) is not None
        assert controller.owner_attempt(OTHER) is None
        await controller.aclose()
        assert controller.owner_attempt(OWNER) is None

    asyncio.run(scenario())


@pytest.mark.parametrize("second_owner", [OWNER, OTHER])
def test_a_late_disconnect_bound_to_an_earlier_attempt_cannot_cancel_a_later_one(
    make_worker, commands_log, second_owner
):
    async def scenario():
        controller, store, _views = build(make_worker, ONE_HOLD)
        await controller.start(OWNER, LEFT)
        await at_prompt(controller)
        earlier = controller.owner_attempt(OWNER)
        await controller.cancel(OWNER)
        await wait_for(lambda: controller.view().stage is Stage.FINISHED)
        await controller.start(second_owner, LEFT)
        await at_prompt(controller)
        later = controller.owner_attempt(second_owner)
        assert later is not None
        assert later is not earlier
        cancels = commands(commands_log).count("cancel")
        # The late timer of the earlier attempt fires now.
        await controller.client_disconnected(OWNER, attempt=earlier)
        await asyncio.sleep(0.3)
        assert controller.active is True
        assert controller.view().stage is Stage.AWAITING_CONFIRMATION
        assert commands(commands_log).count("cancel") == cancels
        # Bound to the right attempt, the same call does cancel it.
        await controller.client_disconnected(second_owner, attempt=later)
        await wait_for(lambda: controller.view().stage is Stage.FINISHED)
        assert controller.view().outcome.kind is OutcomeKind.CANCELLED
        assert store.saved_attempts() == []

    asyncio.run(scenario())


def test_a_disconnect_bound_to_the_current_attempt_cancels_it_and_reaps_the_worker(
    make_worker,
):
    async def scenario():
        controller, _store, _views = build(make_worker, ONE_HOLD)
        await controller.start(OWNER, LEFT)
        await at_prompt(controller)
        await controller.client_disconnected(
            OWNER, attempt=controller.owner_attempt(OWNER)
        )
        await wait_for(lambda: controller.view().stage is Stage.FINISHED)
        outcome = controller.view().outcome
        assert outcome.kind is OutcomeKind.CANCELLED
        assert outcome.exit_code == 0
        assert outcome.forced_termination is False

    asyncio.run(scenario())


def test_a_bound_disconnect_by_a_non_owner_is_ignored(make_worker):
    async def scenario():
        controller, _store, _views = build(make_worker, ONE_HOLD)
        await controller.start(OWNER, LEFT)
        await at_prompt(controller)
        await controller.client_disconnected(
            OTHER, attempt=controller.owner_attempt(OWNER)
        )
        await asyncio.sleep(0.3)
        assert controller.active is True
        await controller.aclose()

    asyncio.run(scenario())


def test_a_bound_disconnect_for_a_finished_attempt_is_ignored(make_worker):
    async def scenario():
        controller, store, _views = build(make_worker, ONE_HOLD)
        await controller.start(OWNER, LEFT)
        await at_prompt(controller)
        earlier = controller.owner_attempt(OWNER)
        await finish(controller)
        await controller.client_disconnected(OWNER, attempt=earlier)
        assert controller.view().outcome.kind is OutcomeKind.COMPLETED
        assert len(store.saved_attempts()) == 1

    asyncio.run(scenario())


def test_the_station_stays_unavailable_after_a_result_that_reports_the_sdk_unreleased(
    make_worker,
):
    async def scenario():
        controller, _store, _views = build(make_worker, {"not_released": True})
        await controller.start(OWNER, LEFT)
        view = await finish(controller)
        assert view.outcome.kind is OutcomeKind.UNCERTAIN
        assert "could not release the SDK" in view.outcome.message
        assert view.outcome.files == ()
        assert not controller.active
        # No new attempt may start while the gloves may still be in use.
        with pytest.raises(AttemptError, match="could not release the SDK"):
            await controller.start(OWNER, LEFT)

    asyncio.run(scenario())
