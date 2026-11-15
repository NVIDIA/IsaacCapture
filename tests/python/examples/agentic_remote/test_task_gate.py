# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TaskGate: how a task takes the arm, keeps it and gives it back.

The gate runs on a real AgentLink against the real robot app on the simulated
arm, in real time; the task, the skills, the arm adapter and the workstation
checker are stand-ins the test scripts.
"""

from __future__ import annotations

import types
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from isaaccapture_examples.agentic_remote.agent import task_gate
from isaaccapture_examples.agentic_remote.agent.config import TaskSettings
from isaaccapture_examples.agentic_remote.agent.link_client import AgentLink, Pinned
from isaaccapture_examples.agentic_remote.agent.task_gate import (
    READ_ONLY_TOOLS,
    REST_REFUSAL,
    RESTART,
    TaskGate,
)
from isaaccapture_examples.agentic_remote.protocol.carrier import loopback_pair
from link_bench import WAIT_S, until

apai = pytest.mark.apai

# APAI's motion skills, as far as these tests name them; localize_object is one
# of them that the experiment lists as read-only.
MOTION = {"grasp_object", "place_at", "move_home", "move_to_rest", "localize_object"}
TASK = "put the cube into the box"


def report(success: bool = False, summary: str = "", **fields: Any) -> Any:
    """What AgentOrchestrator.run_task returns, as far as the gate reads it."""
    given = {"stuck": None, "cancelled": False, **fields}
    return types.SimpleNamespace(task=TASK, success=success, summary=summary, **given)


class Rig:
    """A gate on a link to the real robot app, with a scripted task around it."""

    def __init__(self, harness: Any, **settings: Any) -> None:
        self.h = harness
        self.robot_end, agent_end = loopback_pair()
        self.link = AgentLink(agent_end)
        assert harness.app.attach_carrier(self.robot_end)
        self.link.wait_ready(WAIT_S)
        self.log: list[tuple] = []  # what happened, in order
        self.said: list[str] = []
        self.answers = ["yes"]  # one per prompt; then "no"
        self.on_ask: Callable[[], None] | None = None
        self.checker = types.SimpleNamespace(estopped=False)
        self.depth = 0  # how many skill scopes are open
        self.gate = TaskGate(
            self.link,
            None,
            TaskSettings(**settings),
            ask=self.ask,
            say=self.said.append,
            motion_skills=MOTION,
            read_only_tools=READ_ONLY_TOOLS,
        )
        self.gate.bind_arm(self, self.checker)
        real = self.link.request_owner

        def request_owner(request: str, **kwargs: Any) -> Any:
            self.log.append(("request", request, kwargs.get("quote")))
            result = real(request, **kwargs)
            self.log.append(("result", request, result.status))
            return result

        self.link.request_owner = request_owner

    # The arm adapter, as the gate uses it

    def open_task(self, pinned: Pinned) -> None:
        self.log.append(("open_task", pinned))

    def close_task(self) -> None:
        self.log.append(("close_task",))
        self.link.unpin()

    @contextmanager
    def skill_scope(self) -> Iterator[None]:
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1

    # The operator, the skills and the task

    def ask(self, prompt: str) -> str:
        self.log.append(("ask", prompt))
        if self.on_ask is not None:
            self.on_ask()
        return self.answers.pop(0) if self.answers else "no"

    def skill(self, result: dict | Callable[[], dict] | None = None) -> Callable:
        """SkillRuntime.execute: notes the call and who owns the arm then."""

        def original(name: str, args: dict) -> dict:
            self.log.append(("skill", name, self.owner(), self.depth))
            made = result() if callable(result) else result
            return {"ok": True} if made is None else dict(made)

        return original

    def execute(self, name: str, result: Any = None, **args: Any) -> dict:
        return self.gate.execute(self.skill(result), name, args)

    def task(self, body: Callable[[], Any], text: str = TASK) -> Any:
        """Run body() as one task: AgentOrchestrator.run_task with the gate."""
        return self.gate.bind_run_task(lambda task, **kwargs: body())(text)

    # What the robot app shows

    def owner(self) -> str:
        return self.h.app.snapshot().owner

    def shown(self, owner: str) -> Any:
        until(lambda: self.link.state().owner == owner, f"{owner} as owner")
        return self.link.state()

    def of(self, kind: str) -> list[tuple]:
        return [entry for entry in self.log if entry[0] == kind]

    def kinds(self) -> list[str]:
        return [entry[0] for entry in self.log]

    def help_reasons(self) -> list[str]:
        return [event[2]["reason"] for event in self.h.heard("help_needed")]

    def close(self) -> None:
        self.link.close()
        self.robot_end.close()


@pytest.fixture
def make_rig(make_realtime_robot_app: Callable[..., Any]) -> Iterator[Callable]:
    made: list[Rig] = []

    def make(**settings: Any) -> Rig:
        rig = Rig(make_realtime_robot_app(connected=False), **settings)
        made.append(rig)
        return rig

    yield make
    for rig in made:
        rig.close()


@pytest.fixture
def rig(make_rig: Callable[..., Rig]) -> Rig:
    return make_rig()


# The start


def test_thinking_and_looking_never_take_the_arm(rig: Rig) -> None:
    def body() -> Any:
        assert rig.execute("get_observation") == {"ok": True}
        assert rig.execute("localize_object", label="cube") == {"ok": True}
        assert rig.execute("task_done", success=True) == {"ok": True}
        return report(True)

    rig.task(body)
    # Each ran inside a skill scope of the arm; nothing was asked or sent.
    assert rig.kinds() == ["skill"] * 3
    assert [entry[2:] for entry in rig.log] == [("nobody", 1)] * 3
    assert rig.gate.requests == [] and rig.h.heard("owner") == []


def test_a_task_starts_only_after_the_typed_yes_and_the_thors_grant(rig: Rig) -> None:
    before = rig.link.state()

    def body() -> Any:
        rig.execute("get_observation")
        assert rig.execute("grasp_object", label="cube") == {"ok": True}
        assert rig.execute("place_at", x=0.1) == {"ok": True}
        return report(True)

    rig.task(body)
    # Looking, then: the typed yes, the request, the Thor's answer, the task
    # opened on the arm with the Thor's new numbers, and only then the skill.
    assert rig.kinds()[:7] == [
        "skill",
        "ask",
        "request",
        "result",
        "open_task",
        "skill",
        "skill",
    ]
    (asked,) = rig.of("ask")  # once per task
    assert (
        f'"{TASK}"' in asked[1]
        and 'First tool grasp_object: {"label": "cube"}' in asked[1]
    )
    assert (
        "Type yes to proceed" in asked[1]
        and "Ctrl+C at this terminal is STOP" in asked[1]
    )
    # The request quoted the numbers of the ROBOT_STATE read before the prompt.
    request = rig.of("request")[0]
    assert request[1:] == (
        "agent",
        Pinned(before.session_id, before.owner_number, before.stop_number),
    )
    assert rig.of("result")[0][1:] == ("agent", "completed")
    pinned = rig.of("open_task")[0][1]
    assert pinned == Pinned(before.session_id, before.owner_number + 1, 0)
    # Both motion skills ran with the agent as control owner on the Thor.
    assert [entry[1:3] for entry in rig.of("skill")[1:]] == [
        ("grasp_object", "agent"),
        ("place_at", "agent"),
    ]
    assert rig.said[0].startswith("robot: control owner nobody, stop level none")
    assert "jaws " in rig.said[0] and "torque on, assistance wait none" in rig.said[0]


@pytest.mark.parametrize("answer", ["", "no", "y", "yes!", "yes please", "ok"])
def test_any_answer_but_yes_refuses_and_nothing_is_sent(rig: Rig, answer: str) -> None:
    rig.answers = [answer]

    def body() -> Any:
        first = rig.execute("grasp_object", label="cube")
        second = rig.execute("place_at", x=0.1)  # refused again, not asked again
        assert first == second
        return first

    result = rig.task(body)
    assert result == {
        "ok": False,
        "error": "the operator did not type yes; no command sent",
        "refused": "consent",
        "motion_blocked": True,
        "motion_block_reason": "the operator did not type yes",
    }
    assert rig.kinds() == ["ask"]  # no request, no skill, no close of a task
    assert rig.gate.requests == [] and rig.h.heard("owner") == []
    assert rig.h.heard("help_needed") == [] and rig.owner() == "nobody"


@pytest.mark.parametrize("answer", ["yes", " Yes \n", "YES"])
def test_a_yes_is_taken_however_it_is_cased(rig: Rig, answer: str) -> None:
    rig.answers = [answer]
    rig.task(lambda: rig.execute("grasp_object") and report(True))
    assert rig.of("result")[0][1:] == ("agent", "completed")


def test_a_terminal_that_cannot_be_asked_refuses(rig: Rig) -> None:
    def nobody() -> None:
        raise RuntimeError("motion needs the operator's terminal")

    rig.on_ask = nobody
    result = rig.task(lambda: rig.execute("grasp_object"))
    assert result["refused"] == "consent"
    assert result["motion_block_reason"] == (
        "no typed yes: motion needs the operator's terminal"
    )
    assert rig.kinds() == ["ask"] and rig.owner() == "nobody"


def test_motion_outside_a_task_and_move_to_rest_are_always_refused(rig: Rig) -> None:
    outside = rig.execute("grasp_object")
    assert outside["refused"] == "consent" and "needs a task" in outside["error"]

    def body() -> Any:
        rig.execute("grasp_object")
        return rig.execute("move_to_rest")

    rest = rig.task(body)
    assert rest["ok"] is False and rest["motion_blocked"] is True
    assert rest["error"] == f"{REST_REFUSAL}; no command sent"
    assert (
        "type rest at the Thor terminal or run the console command rest"
        in rest["error"]
    )
    assert [entry[1] for entry in rig.of("skill")] == ["grasp_object"]
    assert rig.execute("move_to_rest")["motion_blocked"] is True  # also outside a task


def test_a_state_that_cannot_start_a_task_refuses_without_a_prompt(
    make_rig: Callable[..., Rig],
) -> None:
    # Somebody owns the arm already.
    rig = make_rig()
    assert rig.link.request_owner("agent", timed_return=False).status == "completed"
    rig.shown("agent")
    rig.log.clear()
    result = rig.task(lambda: rig.execute("grasp_object"))
    assert result["refused"] == "ownership"
    assert result["motion_block_reason"] == "control owner is agent"
    assert rig.kinds() == [] and rig.h.heard("help_needed") == []
    # A fault-stop is latched.
    rig = make_rig()
    rig.h.app._fault("motor fault")
    until(lambda: rig.link.state().stop_level == "fault", "the fault-stop")
    result = rig.task(lambda: rig.execute("grasp_object"))
    assert result["motion_block_reason"] == (
        "fault-stop (motor fault): run the console command reset"
    )
    assert rig.kinds() == []
    # The robot connection is lost.
    rig = make_rig()
    rig.robot_end.break_link()
    until(lambda: not rig.link._carrier.alive, "the lost link")
    result = rig.task(lambda: rig.execute("grasp_object"))
    assert result["motion_block_reason"].startswith("robot connection lost: ")
    assert rig.kinds() == []


def test_a_refusal_by_the_thor_names_the_remedy_and_ends_the_tasks_motion(
    rig: Rig,
) -> None:
    # A STOP while the operator answers: the stop number quoted is old.
    rig.on_ask = lambda: (
        rig.link.stop("pressed", "console"),
        until(lambda: rig.link.state().stop_number == 1, "the STOP"),
    )
    result = rig.task(lambda: rig.execute("grasp_object"))
    assert result["refused"] == "ownership" and result["motion_blocked"] is True
    reason = result["motion_block_reason"]
    assert reason.startswith("task refused by the robot: stop number 0 does not match")
    assert reason.endswith("(the robot's state changed meanwhile: type the task again)")
    assert rig.kinds() == ["ask", "request", "result"]
    assert rig.of("result")[0][1:] == ("agent", "refused")
    assert rig.owner() == "nobody" and rig.h.heard("help_needed") == []
    # The next task is a new start: it asks again and is granted.
    rig.log.clear()
    rig.on_ask = None
    rig.answers = ["yes"]
    rig.task(lambda: rig.execute("grasp_object") and report(True))
    assert rig.of("result")[0][1:] == ("agent", "completed")


@pytest.mark.parametrize(
    ("reason", "remedy"),
    [
        ("session id does not match", "restart the agent"),
        ("the motors are off; restart the robot app", "restart the robot app"),
        ("owner number 3 does not match the current 4", "type the task again"),
        ("control owner is human, not nobody", "give the arm back with right A"),
        ("fault-stop latched: motor fault", "run the console command reset"),
        ("busy: a jaw command is running", "wait for the robot app to finish"),
        ("the measured pose could not be read: timeout", "look at the Thor terminal"),
        ("type rest first: joint 2 outside", "run the console command rest"),
        ("a STOP or an owner change arrived while", "type the task again"),
        ("something new", ""),
    ],
)
def test_each_refusal_of_the_thor_has_its_remedy(reason: str, remedy: str) -> None:
    assert remedy in task_gate._remedy(reason)
    assert bool(task_gate._remedy(reason)) == bool(remedy)


# During the task


def test_three_failed_steps_in_a_row_end_the_task_as_stuck(rig: Rig) -> None:
    failed = {"ok": False, "error": "no grasp found"}

    def body() -> list[dict]:
        return [
            rig.execute("get_observation", {"ok": False}),  # before the agent owns
            rig.execute("grasp_object", failed),
            rig.execute("grasp_object", failed),
            rig.execute("get_observation"),  # a success resets the count
            rig.execute("grasp_object", failed),
            rig.execute("get_observation", {"ok": False, "error": "nothing seen"}),
            rig.execute("place_at", {"ok": False, "error": "unreachable", "x": 1}),
        ]

    results = rig.task(body)
    assert [result.get("outcome") for result in results] == [None] * 6 + ["stuck"]
    assert results[-1] == {
        "ok": False,
        "error": "unreachable",
        "x": 1,
        "outcome": "stuck",
        "ask": "three steps failed in a row: unreachable",
    }


def test_the_failed_steps_limit_is_the_settings(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig(failed_steps_limit=1)
    result = rig.task(lambda: rig.execute("grasp_object", {"ok": False}))
    assert result["outcome"] == "stuck"
    assert result["ask"] == "one step failed in a row: no reason given"


def test_a_blocked_step_is_not_rewritten_to_stuck(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig(failed_steps_limit=1)
    blocked = {"ok": False, "error": "arm stop latched", "motion_blocked": True}
    result = rig.task(lambda: rig.execute("grasp_object", blocked))
    assert "outcome" not in result


# The end


@pytest.mark.parametrize(
    ("ending", "sent", "reason"),
    [
        (lambda rig: report(True, "done"), "nobody", "task completed"),
        (
            lambda rig: report(
                stuck={"skill": "grasp_object", "ask": "move it closer"}
            ),
            "help_needed",
            "stuck: move it closer",
        ),
        (
            lambda rig: (
                rig.execute(
                    "place_at",
                    {
                        "ok": False,
                        "motion_blocked": True,
                        "motion_block_reason": "arm stop latched",
                    },
                )
                and report(summary="arm stop latched")
            ),
            "help_needed",
            "blocked: arm stop latched",
        ),
        (
            lambda rig: report(summary="step budget exhausted"),
            "help_needed",
            "step budget ran out",
        ),
        (
            lambda rig: report(summary="I give up: the box\nis not there"),
            "help_needed",
            "task ended without success: I give up: the box is not there",
        ),
    ],
    ids=["success", "stuck", "blocked", "budget", "gave-up"],
)
def test_the_end_of_a_task_gives_the_arm_back_or_asks_for_help(
    rig: Rig, ending: Callable[[Rig], Any], sent: str, reason: str
) -> None:
    def body() -> Any:
        rig.execute("grasp_object")
        pinned.append(rig.link.pinned)
        return ending(rig)

    pinned: list[Pinned] = []
    rig.task(body)
    # One end request, quoting the pinned numbers; its RESULT came before the
    # task was closed on the arm.
    assert rig.kinds()[-3:] == ["request", "result", "close_task"]
    assert rig.of("request")[-1][1:] == (sent, pinned[0])
    assert rig.of("result")[-1][1:] == (sent, "completed")
    assert rig.gate.requests[-1] == (sent, "completed", "")
    assert rig.link.pinned is None
    state = rig.shown("nobody")
    if sent == "nobody":
        assert rig.help_reasons() == [] and state.assistance_wait_s is None
        assert rig.said[-1].startswith("task completed: the arm holds where it is")
    else:
        assert rig.help_reasons() == [reason]
        until(lambda: rig.link.state().assistance_wait_s is not None, "the wait")
        assert rig.said[-1].startswith(f"HELP NEEDED: {reason}.")
        assert rig.h.heard("help_needed")[0][2]["timed_return"] is False


def test_an_error_of_the_agent_asks_for_help_and_is_raised_again(rig: Rig) -> None:
    def body() -> Any:
        rig.execute("grasp_object")
        raise ValueError("the model answered nothing")

    with pytest.raises(ValueError, match="the model answered nothing"):
        rig.task(body)
    assert rig.help_reasons() == ["agent error: ValueError: the model answered nothing"]
    assert rig.kinds()[-3:] == ["request", "result", "close_task"]
    assert rig.shown("nobody").assistance_wait_s is not None


@pytest.mark.parametrize("how", ["cancelled", "interrupt"])
def test_a_cancelled_task_sends_nothing_from_the_gate(rig: Rig, how: str) -> None:
    def body() -> Any:
        rig.execute("grasp_object")
        if how == "interrupt":
            raise KeyboardInterrupt
        return report(summary="command cancelled", cancelled=True)

    if how == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            rig.task(body)
    else:
        rig.task(body)
    # No owner request: the STOP of the agent terminal follows from APAI's path.
    assert [entry[1] for entry in rig.of("request")] == ["agent"]
    assert rig.kinds()[-1] == "close_task" and rig.h.heard("help_needed") == []
    assert rig.said[-1].startswith("the task was cancelled")
    assert rig.owner() == "agent"  # until that STOP


def test_a_help_request_the_thor_refuses_is_printed_and_nothing_else_is_sent(
    rig: Rig,
) -> None:
    def stopped() -> dict:
        # A STOP from the console during the skill: nobody owns the arm now.
        rig.link.stop("pressed", "console")
        rig.shown("nobody")
        return {"ok": False, "error": "halted", "motion_blocked": True}

    def body() -> Any:
        rig.execute("grasp_object", stopped)
        return report(summary="halted")

    rig.task(body)
    assert rig.kinds()[-3:] == ["request", "result", "close_task"]
    assert rig.of("result")[-1][1:] == ("help_needed", "refused")
    assert (
        "refused by the robot" in rig.said[-1]
        and "nothing else is sent" in rig.said[-1]
    )
    assert rig.h.heard("help_needed") == [] and rig.owner() == "nobody"
    assert rig.link.state().assistance_wait_s is None


def test_a_task_that_never_owned_the_arm_ends_without_a_request(rig: Rig) -> None:
    rig.task(lambda: rig.execute("get_observation") and report(False, "nothing seen"))
    assert rig.kinds() == ["skill"] and rig.h.heard("help_needed") == []


def test_the_timed_return_of_the_settings_goes_with_the_task(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig(timed_return=True)
    rig.task(lambda: rig.execute("grasp_object") and report(summary="gave up"))
    (event,) = rig.h.heard("help_needed")
    assert event[2]["timed_return"] is True


# The workstation latch (D4)


def test_a_latched_checker_refuses_the_start_of_motion(rig: Rig) -> None:
    rig.checker.estopped = True
    result = rig.gate.execute(rig.skill(), "grasp_object", {})
    assert result["motion_blocked"] is True  # outside a task: refused anyway
    token = rig.gate._current.set(task_gate._Task(TASK))
    try:
        result = rig.gate.execute(rig.skill(), "grasp_object", {})
    finally:
        rig.gate._current.reset(token)
    assert result["refused"] == "ownership"
    assert result["motion_block_reason"] == RESTART
    assert rig.kinds() == [] and rig.gate.restart_told is True


@apai
def test_after_a_skill_stop_the_gate_says_restart_and_runs_no_new_task(
    rig: Rig,
) -> None:
    def stop_in_a_skill() -> dict:
        rig.checker.estopped = True  # what SafeArm.stop() leaves behind
        return {
            "ok": False,
            "error": "home move failed; e-stop latched",
            "motion_blocked": True,
            "motion_block_reason": "arm stop latched",
        }

    def body() -> Any:
        rig.execute("grasp_object", stop_in_a_skill)
        return report(summary="home move failed")

    assert rig.gate.restart_needed is False
    rig.task(body)
    # The agent kept the arm through the skill stop, so the help request is taken.
    assert rig.help_reasons() == ["blocked: arm stop latched"]
    assert rig.said[-1] == RESTART and rig.gate.restart_told is True
    assert "restart the agent program" in RESTART
    # A new task is not run at all: no prompt, no request, no skill.
    rig.log.clear()
    ran: list[str] = []
    again = rig.gate.bind_run_task(lambda task, **kwargs: ran.append(task))("again")
    assert ran == [] and rig.log == []
    assert (again.success, again.summary, again.steps) == (False, RESTART, 0)
    # The latch stays for the run, whatever the checker says later.
    rig.checker.estopped = False
    assert rig.gate.restart_needed is True and rig.gate.tell_restart() is True
