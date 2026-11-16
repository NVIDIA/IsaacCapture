# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TaskGate: how a task takes the arm, keeps it and gives it back.

The gate runs on a real AgentLink against the real robot app on the simulated
arm, in real time; the task, the skills, the arm adapter and the workstation
checker are stand-ins the test scripts.
"""

from __future__ import annotations

import dataclasses
import threading
import time
import types
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from isaaccapture_examples.agentic_remote.agent import task_gate
from isaaccapture_examples.agentic_remote.agent.config import TaskSettings
from isaaccapture_examples.agentic_remote.agent.link_client import (
    AgentLink,
    CommandTimeout,
    Pinned,
)
from isaaccapture_examples.agentic_remote.agent.task_gate import (
    READ_ONLY_TOOLS,
    REST_REFUSAL,
    RESTART,
    SKILL_STOP_KEPT,
    SKILL_STOP_NOTE,
    SKILL_STOP_TOLD,
    TaskGate,
)
from isaaccapture_examples.agentic_remote.protocol.carrier import loopback_pair, send
from isaaccapture_examples.agentic_remote.protocol.messages import Result
from isaaccapture_examples.agentic_remote.robot_app import app as app_module
from isaaccapture_examples.agentic_remote.robot_app.interfaces import Owner
from link_bench import WAIT_S, robot_state, until

apai = pytest.mark.apai

# APAI's motion skills, as far as these tests name them; localize_object is one
# of them that the experiment lists as read-only.
MOTION = {"grasp_object", "place_at", "move_home", "move_to_rest", "localize_object"}
TASK = "put the cube into the box"


def report(success: bool = False, summary: str = "", **fields: Any) -> Any:
    """What AgentOrchestrator.run_task returns, as far as the gate reads it."""
    given = {"stuck": None, "cancelled": False, **fields}
    return types.SimpleNamespace(task=TASK, success=success, summary=summary, **given)


class Checker:
    """The workstation checker (APAI's SafetyHarness), as far as the gate uses it."""

    def __init__(self, log: list[tuple]) -> None:
        self.estopped = False
        self.stuck = False  # reset_estop() leaves the latch
        self._log = log
        self._observation_cancel_generation = 0  # APAI's count of its stops

    def estop(self) -> None:
        """What SafeArm.stop() leaves behind."""
        self.estopped = True
        self._observation_cancel_generation += 1

    def reset_estop(self) -> None:
        self._log.append(("reset",))
        if not self.stuck:
            self.estopped = False


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
        self.checker = Checker(self.log)
        self.depth = 0  # how many skill scopes are open
        # driver_check is the gate's own keyword, not a task setting.
        options = {
            name: settings.pop(name) for name in ("driver_check",) if name in settings
        }
        self.gate = TaskGate(
            self.link,
            None,
            TaskSettings(**settings),
            ask=self.ask,
            say=self.said.append,
            motion_skills=MOTION,
            read_only_tools=READ_ONLY_TOOLS,
            **options,
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
    # A person owns the arm.
    rig = make_rig()
    rig.h.app._authority.set_owner(Owner.HUMAN, "takeover")
    rig.shown("human")
    result = rig.task(lambda: rig.execute("grasp_object"))
    assert result["refused"] == "ownership"
    assert result["motion_block_reason"] == (
        "takeover not finished: give the arm back with right A at rest"
    )
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


# A takeover makes a person the control owner; it is open until the person
# gives the arm back at rest, also while a STOP has made nobody the owner.
TAKEOVER_OPEN = "takeover not finished: give the arm back with right A at rest"


def test_a_task_is_not_started_while_a_takeover_is_open(rig: Rig) -> None:
    # The gate's text is the robot app's own for the same refusal.
    assert task_gate.TAKEOVER_OPEN == app_module.TAKEOVER_OPEN == TAKEOVER_OPEN
    # What ROBOT_STATE's health["human"] says of a takeover, as the gate reads it.
    nobody = robot_state()
    for human, reason in (
        ({"takeover_open": True}, TAKEOVER_OPEN),
        ({"takeover_open": False}, None),
        ({}, None),
        (None, None),
    ):
        state = dataclasses.replace(nobody, health={"human": human})
        assert task_gate._not_startable(state) == reason
    assert task_gate._not_startable(nobody) is None  # a robot app without the key
    # A STOP made nobody the owner while a person's takeover is open.
    real, taken = rig.link.state, True

    def state_with_the_takeover(*args: Any, **kwargs: Any) -> Any:
        shown = real(*args, **kwargs)
        human = {**(shown.health.get("human") or {}), "takeover_open": taken}
        return dataclasses.replace(shown, health={**shown.health, "human": human})

    rig.link.state = state_with_the_takeover
    result = rig.task(lambda: rig.execute("grasp_object"))
    assert result["refused"] == "ownership"
    assert result["motion_block_reason"] == TAKEOVER_OPEN
    # Nobody was asked and nothing was sent.
    assert rig.kinds() == [] and rig.gate.requests == []
    assert rig.owner() == "nobody" and rig.h.heard("help_needed") == []
    assert rig.said[-1] == f"motion refused: {TAKEOVER_OPEN}; no command sent"
    # Once the arm was given back, the same task is asked for and granted.
    taken = False
    rig.task(lambda: rig.execute("grasp_object") and report(True))
    assert rig.kinds()[:3] == ["ask", "request", "result"]
    assert rig.of("result")[0][1:] == ("agent", "completed")


def answer_late(rig: Rig, delay_s: float) -> None:
    """The robot app answers the next OWNER_REQUEST "agent" delay_s late."""
    app = rig.h.app
    new_task = app._new_task

    def slow(message: Any, reply: Any) -> Any:
        app._new_task = new_task
        time.sleep(delay_s)  # on the dispatcher thread
        return new_task(message, reply)

    app._new_task = slow


def test_ownership_that_arrives_after_the_gate_gave_up_is_given_back_at_the_next_task(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig(end_wait_s=0.3)
    answer_late(rig, 0.6)
    refused = rig.task(lambda: rig.execute("grasp_object"))
    assert refused["refused"] == "ownership"
    assert "the robot did not answer" in refused["motion_block_reason"]
    assert rig.gate.requests == []  # the grant is still under way
    # The next task, typed at once: the gate waits for the late grant, gives
    # the arm back and starts as ever.
    rig.log.clear()
    rig.answers = ["yes"]
    done: list[dict] = []
    rig.task(lambda: done.append(rig.execute("grasp_object")) or report(True))
    assert done == [{"ok": True}]
    assert rig.kinds() == ["request", "result", "ask", "request", "result"] + [
        "open_task",
        "skill",
        "request",
        "result",
        "close_task",
    ]
    assert [entry[1:] for entry in rig.of("result")] == [
        ("nobody", "completed"),
        ("agent", "completed"),
        ("nobody", "completed"),
    ]
    assert rig.of("skill")[0][1:3] == ("grasp_object", "agent")
    assert rig.gate.requests[0][:2] == ("nobody", "completed")
    events = rig.h.heard("owner")
    assert events[0][2]["owner"] == "agent" and events[2][2]["owner"] == "agent"
    assert "the ownership arrived late" in events[1][1]
    assert any("the ownership arrived late" in line for line in rig.said)
    assert rig.h.heard("help_needed") == []


def unanswered_start(rig: Rig, command_id: str) -> None:
    """A task whose OWNER_REQUEST "agent" gets no RESULT in time."""
    real = rig.link.request_owner

    def no_answer(request: str, **kwargs: Any) -> Any:
        raise CommandTimeout("no final RESULT within 0.3 s", command_id)

    rig.link.request_owner = no_answer
    try:
        refused = rig.task(lambda: rig.execute("grasp_object"))
    finally:
        rig.link.request_owner = real
    assert "the robot did not answer" in refused["motion_block_reason"]
    rig.log.clear()


def test_a_late_refusal_of_the_thor_leaves_nothing_to_give_back(rig: Rig) -> None:
    unanswered_start(rig, "late-1")
    rig.link._take_result(
        Result("late-1", "refused", reason="stop number 0 does not match")
    )
    began = time.monotonic()
    rig.gate.return_late_ownership(5.0)
    rig.gate.return_late_ownership(5.0)
    assert time.monotonic() - began < 1.0  # the refusal ended the wait
    assert rig.kinds() == [] and rig.owner() == "nobody"


def test_the_wait_for_a_late_grant_is_spent_once(rig: Rig) -> None:
    unanswered_start(rig, "late-2")  # its RESULT never comes
    began = time.monotonic()
    rig.gate.return_late_ownership(0.3)
    assert 0.3 <= time.monotonic() - began < 1.0
    began = time.monotonic()
    rig.gate.return_late_ownership(5.0)
    assert time.monotonic() - began < 0.5
    assert rig.kinds() == []


def test_ownership_without_a_task_is_given_back_whenever_the_gate_looks(
    rig: Rig,
) -> None:
    # The agent owns the arm and no task of this gate does.
    assert rig.link.request_owner("agent", timed_return=False).status == "completed"
    state = rig.shown("agent")
    rig.log.clear()
    rig.gate.return_late_ownership()
    quote = Pinned(state.session_id, state.owner_number, state.stop_number)
    assert rig.log == [("request", "nobody", quote), ("result", "nobody", "completed")]
    assert rig.gate.requests == [("nobody", "completed", "")]
    assert rig.link.state().owner == "nobody"  # shown before the gate goes on
    (returned,) = [e for e in rig.h.heard("owner") if "arrived late" in e[1]]
    assert "the ownership arrived late" in returned[1]
    # Nothing to do then, and nothing while a task owns the arm.
    rig.log.clear()
    rig.gate.return_late_ownership()
    assert rig.log == []

    def body() -> Any:
        rig.execute("grasp_object")
        rig.gate.return_late_ownership()
        return report(True)

    rig.task(body)
    assert [entry[1] for entry in rig.of("request")] == ["agent", "nobody"]
    # After the task its own ownership is not taken for a late one, even while
    # ROBOT_STATE has not caught up.
    rig.log.clear()
    rig.gate.return_late_ownership()
    assert rig.log == []


def test_a_task_started_while_the_agent_owns_without_one_takes_the_arm(
    rig: Rig,
) -> None:
    assert rig.link.request_owner("agent", timed_return=False).status == "completed"
    rig.shown("agent")
    rig.log.clear()
    assert rig.task(lambda: rig.execute("grasp_object")) == {"ok": True}
    assert [entry[1:] for entry in rig.of("result")][:2] == [
        ("nobody", "completed"),
        ("agent", "completed"),
    ]


def test_an_end_request_without_an_answer_is_remembered(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig(owner_wait_s=0.5)
    assert rig.gate.end_unanswered is False

    def body() -> Any:
        rig.execute("grasp_object")
        asked = rig.link.request_owner

        def no_answer(request: str, **kwargs: Any) -> Any:
            if request == "agent":
                return asked(request, **kwargs)
            raise CommandTimeout("no final RESULT within 5 s", "c1")

        rig.link.request_owner = no_answer
        return report(True)

    rig.task(body)
    assert rig.gate.end_unanswered is True
    assert "was not answered" in rig.said[-1] and rig.owner() == "agent"
    # The gate does not ask again for an ownership it tried to end.
    rig.log.clear()
    rig.gate.return_late_ownership()
    assert rig.log == []
    # A task typed then is refused before its question, and told the remedy.
    rig.answers = ["yes"]
    refused = rig.task(lambda: rig.execute("grasp_object"))
    assert refused["motion_block_reason"] == (
        "control owner is agent (an owner request of this program got no "
        "answer: run the console command stop, then type the task again)"
    )
    assert rig.kinds() == [] and rig.owner() == "agent"


def test_a_task_typed_at_once_waits_for_the_state_that_shows_the_last_one_ended(
    rig: Rig,
) -> None:
    take = rig.link._take_state
    held: list[Any] = []

    def body() -> Any:
        rig.execute("grasp_object")
        rig.link._take_state = held.append  # from here the ROBOT_STATEs come late
        return report(True)

    rig.task(body)
    # The arm was given back; the newest ROBOT_STATE here still shows the agent.
    assert rig.owner() == "nobody" and rig.link.state().owner == "agent"

    def arrive() -> None:
        del rig.link._take_state
        for state in held:
            take(state)

    late = threading.Timer(0.3, arrive)
    late.start()
    rig.log.clear()
    rig.answers = ["yes"]
    began = time.monotonic()
    try:
        done = rig.task(lambda: rig.execute("grasp_object"))
    finally:
        late.join()
    # Not refused with "control owner is agent": the gate waited for the state.
    assert done == {"ok": True}
    assert 0.3 <= time.monotonic() - began < 2.0
    assert rig.kinds()[:4] == ["ask", "request", "result", "open_task"]


def test_the_gate_counts_the_tasks_that_ended_without_success(rig: Rig) -> None:
    rig.task(lambda: rig.execute("get_observation") and report(True))
    assert rig.gate.failed_tasks == 0
    rig.task(lambda: report(False, "nothing seen"))
    assert rig.gate.failed_tasks == 1
    with pytest.raises(ValueError, match="broken"):
        rig.task(lambda: (_ for _ in ()).throw(ValueError("broken")))
    assert rig.gate.failed_tasks == 2
    # A task the operator cancelled is not one that failed.
    rig.task(lambda: report(summary="command cancelled", cancelled=True))
    with pytest.raises(KeyboardInterrupt):
        rig.task(lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert rig.gate.failed_tasks == 2


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
        (
            "takeover not finished: give the arm back with right A at rest",
            "no controller: see the README",
        ),
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


def test_the_remedy_for_an_open_takeover_names_right_a(rig: Rig) -> None:
    remedy = (
        "hold right A for 1 s with the arm at rest; no controller: see the "
        'README, "A takeover that cannot be ended with right A"'
    )
    assert task_gate._remedy(TAKEOVER_OPEN) == remedy
    # It follows the Thor's reason in one line: it does not say it again.
    assert TAKEOVER_OPEN not in remedy
    # Its row stands before those of "control owner is": it wins where a
    # reason holds both pieces.
    both = f"control owner is nobody; {TAKEOVER_OPEN}"
    assert task_gate._remedy(both) == remedy
    # The Thor refuses the task: a person took the arm while the operator
    # answered the question. The refusal carries the remedy.
    real = rig.link.request_owner

    def taken_meanwhile(request: str, **kwargs: Any) -> Any:
        if request != "agent":
            return real(request, **kwargs)
        rig.log.append(("request", request, kwargs.get("quote")))
        return Result("agent-1", "refused", TAKEOVER_OPEN)

    rig.link.request_owner = taken_meanwhile
    result = rig.task(lambda: rig.execute("grasp_object"))
    assert result["refused"] == "ownership"
    assert result["motion_block_reason"] == (
        f"task refused by the robot: {TAKEOVER_OPEN} ({remedy})"
    )
    assert rig.kinds() == ["ask", "request"]
    assert rig.gate.requests == [("agent", "refused", TAKEOVER_OPEN)]
    assert rig.owner() == "nobody"


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


def end_in_a_skill_stop(rig: Rig) -> None:
    """One task whose motion skill fails and stops the arm; then nobody owns."""

    def stop_in_a_skill() -> dict:
        rig.checker.estop()
        return {
            "ok": False,
            "error": "home move failed; e-stop latched",
            "motion_blocked": True,
            "motion_block_reason": "arm stop latched",
        }

    def body() -> Any:
        rig.execute("grasp_object", stop_in_a_skill)
        return report(summary="home move failed")

    rig.task(body)
    rig.shown("nobody")
    rig.log.clear()
    rig.answers = ["yes"]


def grasp(rig: Rig) -> dict:
    """One task of one motion skill; what the skill returned."""
    results: list[dict] = []

    def body() -> Any:
        results.append(rig.execute("grasp_object", label="cube"))
        return report(results[0].get("ok") is True)

    rig.task(body)
    rig.shown("nobody")
    return results[0]


def test_a_task_after_a_skill_stop_runs_without_a_restart(rig: Rig) -> None:
    end_in_a_skill_stop(rig)
    # The agent kept the arm through the skill stop, so the help request was
    # taken. The latch stands until a person confirms the next task.
    assert rig.help_reasons() == ["blocked: arm stop latched"]
    assert rig.said[-1] == SKILL_STOP_TOLD and rig.checker.estopped is True
    assert rig.gate.restart_needed is False and rig.gate.tell_restart() is False

    assert grasp(rig) == {"ok": True}
    # The question names the skill stop; the latch is reset after the Thor's
    # grant and before the task is opened on the arm.
    assert rig.kinds() == [
        "ask",
        "request",
        "result",
        "reset",
        "open_task",
        "skill",
        "request",
        "result",
        "close_task",
    ]
    assert rig.of("ask")[0][1].startswith(f"Note: {SKILL_STOP_NOTE}. Authorize motion")
    assert [entry[1:] for entry in rig.of("result")] == [
        ("agent", "completed"),
        ("nobody", "completed"),
    ]
    assert rig.checker.estopped is False
    assert "the skill stop of the last task is cleared" in rig.said
    assert RESTART not in rig.said and rig.gate.restart_told is False
    assert rig.gate.failed_tasks == 1  # the task that ended in the skill stop

    # A task after it is asked as any other: nothing is named, nothing reset.
    rig.log.clear()
    rig.answers = ["yes"]
    assert grasp(rig) == {"ok": True}
    assert SKILL_STOP_NOTE not in rig.of("ask")[0][1] and rig.of("reset") == []


def test_the_latch_is_cleared_only_after_the_yes_and_the_grant(rig: Rig) -> None:
    end_in_a_skill_stop(rig)
    # Looking takes no arm and clears nothing.
    rig.task(lambda: rig.execute("get_observation") and report(True))
    assert rig.kinds() == ["skill"] and rig.checker.estopped is True
    # Any answer but yes: nothing is asked of the Thor, the latch stays.
    rig.log.clear()
    rig.answers = ["no"]
    assert grasp(rig)["refused"] == "consent"
    assert rig.kinds() == ["ask"] and rig.checker.estopped is True
    # A yes the Thor does not grant (a STOP came while the question stood).
    rig.log.clear()
    rig.answers = ["yes"]
    rig.on_ask = lambda: rig.h.app.stop("typed stop")
    refused = grasp(rig)
    assert "stop number" in refused["motion_block_reason"]
    assert rig.kinds() == ["ask", "request", "result"]
    assert rig.of("result")[0][1:] == ("agent", "refused")
    assert rig.checker.estopped is True and rig.gate.restart_needed is False
    # A fault-stop on the Thor: no question, and the latch stays.
    rig.log.clear()
    rig.on_ask = None
    rig.h.app._fault("motor fault")
    until(lambda: rig.link.state().stop_level == "fault", "the fault-stop")
    assert grasp(rig)["motion_block_reason"].startswith("fault-stop (motor fault)")
    assert rig.kinds() == [] and rig.checker.estopped is True
    assert rig.link.state().stop_level == "fault"  # nothing cleared it


def test_the_latch_is_cleared_while_the_agent_owns_with_stop_level_none(
    rig: Rig,
) -> None:
    end_in_a_skill_stop(rig)
    rig.h.app.stop("typed stop")  # a STOP the grant of the next task ends
    until(lambda: rig.link.state().stop_level == "stop", "the STOP")
    seen: list[tuple] = []
    reset = rig.checker.reset_estop

    def noted_reset() -> None:
        state = rig.link.state()
        seen.append((rig.owner(), state.owner, state.stop_level, rig.link.pinned))
        reset()

    rig.checker.reset_estop = noted_reset
    assert grasp(rig) == {"ok": True}
    # At the reset the Thor showed the agent as owner; nothing was pinned yet,
    # so no motion could have been sent before it.
    assert seen == [("agent", "agent", "none", None)]


def test_a_stop_of_the_checker_after_the_question_keeps_the_latch(rig: Rig) -> None:
    # Latched before the question and stopped again while it stood.
    end_in_a_skill_stop(rig)
    rig.on_ask = rig.checker.estop
    refused = grasp(rig)
    assert refused["refused"] == "ownership"
    assert refused["motion_block_reason"] == (
        f"the workstation checker was stopped again: {SKILL_STOP_KEPT}"
    )
    # The arm was granted and given back; no task was opened on it.
    assert rig.kinds() == ["ask", "request", "result", "request", "result"]
    assert [entry[1:] for entry in rig.of("result")] == [
        ("agent", "completed"),
        ("nobody", "completed"),
    ]
    assert rig.checker.estopped is True and rig.gate.restart_needed is False
    assert "the arm was given back: nobody owns it" in rig.said
    # The next task's question names the latch, and its yes clears it.
    rig.log.clear()
    rig.answers, rig.on_ask = ["yes"], None
    assert grasp(rig) == {"ok": True} and rig.checker.estopped is False


def test_a_latch_the_question_did_not_name_is_not_cleared(rig: Rig) -> None:
    rig.on_ask = rig.checker.estop  # not latched when the question was made
    refused = grasp(rig)
    assert SKILL_STOP_NOTE not in rig.of("ask")[0][1]
    assert SKILL_STOP_KEPT in refused["motion_block_reason"]
    assert rig.of("reset") == [] and rig.of("open_task") == []
    assert rig.checker.estopped is True and rig.owner() == "nobody"


def test_a_stop_on_the_thor_after_the_grant_keeps_the_latch(rig: Rig) -> None:
    end_in_a_skill_stop(rig)
    await_owner = rig.gate._await_owner

    def stop_after_the_grant(number: Any) -> Any:
        shown = await_owner(number)
        rig.h.app.stop("typed stop")
        rig.shown("nobody")
        return shown

    rig.gate._await_owner = stop_after_the_grant
    refused = grasp(rig)
    assert refused["motion_block_reason"] == (
        f"control owner nobody, stop level stop after the grant: {SKILL_STOP_KEPT}"
    )
    assert rig.of("reset") == [] and rig.of("open_task") == []
    assert rig.checker.estopped is True and rig.gate.restart_needed is False
    # The STOP had ended the ownership already: the give-back was refused.
    assert rig.of("result")[-1][1:] == ("nobody", "refused")
    assert any("the arm could not be given back" in line for line in rig.said)


@apai
@pytest.mark.parametrize("checker", ["stuck", "without a reset"])
def test_a_latch_that_does_not_clear_ends_in_the_restart_line(
    rig: Rig, checker: str
) -> None:
    end_in_a_skill_stop(rig)
    if checker == "stuck":
        rig.checker.stuck = True
    else:
        rig.gate.bind_arm(rig, types.SimpleNamespace(estopped=True))
    refused = grasp(rig)
    assert refused["refused"] == "ownership"
    assert refused["motion_block_reason"] == RESTART
    # Granted, not cleared, given back; no task was opened and no skill ran.
    assert [kind for kind in rig.kinds() if kind != "reset"] == [
        "ask",
        "request",
        "result",
        "request",
        "result",
    ]
    assert rig.of("result")[-1][1:] == ("nobody", "completed")
    assert rig.owner() == "nobody"
    assert rig.gate.restart_needed is True and rig.gate.restart_told is True
    # From here on as with clear_skill_stop false: no new task is run.
    rig.log.clear()
    ran: list[str] = []
    again = rig.gate.bind_run_task(lambda task, **kwargs: ran.append(task))("again")
    assert ran == [] and rig.log == [] and rig.said[-1] == RESTART
    assert (again.success, again.summary, again.steps) == (False, RESTART, 0)


@apai
def test_clear_skill_stop_false_keeps_the_restart(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig(clear_skill_stop=False)
    assert rig.gate.restart_needed is False
    end_in_a_skill_stop(rig)
    assert rig.help_reasons() == ["blocked: arm stop latched"]
    assert rig.said[-1] == RESTART and rig.gate.restart_told is True
    assert "restart the agent program" in RESTART
    # A new task is not run at all: no prompt, no request, no skill.
    ran: list[str] = []
    again = rig.gate.bind_run_task(lambda task, **kwargs: ran.append(task))("again")
    assert ran == [] and rig.log == [] and rig.of("reset") == []
    assert (again.success, again.summary, again.steps) == (False, RESTART, 0)
    # Motion inside a task that runs already is refused the same way.
    token = rig.gate._current.set(task_gate._Task(TASK))
    try:
        result = rig.gate.execute(rig.skill(), "grasp_object", {})
    finally:
        rig.gate._current.reset(token)
    assert result["refused"] == "ownership"
    assert result["motion_block_reason"] == RESTART and rig.kinds() == []
    # The latch stays for the run, whatever the checker says later.
    rig.checker.estopped = False
    assert rig.gate.restart_needed is True and rig.gate.tell_restart() is True


def test_a_stop_between_the_count_check_and_the_reset_is_not_erased(rig: Rig) -> None:
    end_in_a_skill_stop(rig)
    reset = rig.checker.reset_estop

    def stop_then_reset() -> None:
        rig.checker.estop()  # after the gate compared the counts
        reset()

    rig.checker.reset_estop = stop_then_reset
    count = rig.checker._observation_cancel_generation
    refused = grasp(rig)
    assert refused["refused"] == "ownership"
    assert refused["motion_block_reason"] == (
        f"the workstation checker was stopped again: {SKILL_STOP_KEPT}"
    )
    # Latched again by the checker's own stop; granted, given back, not opened.
    assert rig.checker.estopped is True
    assert rig.checker._observation_cancel_generation == count + 2
    assert rig.kinds() == ["ask", "request", "result", "reset", "request", "result"]
    assert rig.of("result")[-1][1:] == ("nobody", "completed")
    assert rig.gate.restart_needed is False
    # The next task's question names the latch, and its yes clears it.
    rig.checker.reset_estop = reset
    rig.log.clear()
    rig.answers = ["yes"]
    assert grasp(rig) == {"ok": True} and rig.checker.estopped is False


@pytest.mark.parametrize(
    "count", ["never", "at the question only", "at the grant only"]
)
def test_a_checker_without_the_stop_count_keeps_its_latch(rig: Rig, count: str) -> None:
    end_in_a_skill_stop(rig)
    log = rig.log

    class NoCount:
        estopped = True

        def estop(self) -> None:
            self.estopped = True

        def reset_estop(self) -> None:
            log.append(("reset",))
            self.estopped = False

    checker = NoCount()
    if count == "at the question only":
        checker._observation_cancel_generation = 7

    def while_the_question_stands() -> None:
        checker.estop()  # stopped again
        if count == "at the question only":
            del checker._observation_cancel_generation
        elif count == "at the grant only":
            checker._observation_cancel_generation = 7

    rig.gate.bind_arm(rig, checker)
    rig.on_ask = while_the_question_stands
    refused = grasp(rig)
    # Whether it was stopped again cannot be told: nothing is cleared.
    assert refused["motion_block_reason"] == RESTART
    assert checker.estopped is True and rig.of("reset") == []
    assert rig.kinds() == ["ask", "request", "result", "request", "result"]
    assert rig.of("result")[-1][1:] == ("nobody", "completed")
    assert rig.gate.restart_needed is True and rig.gate.restart_told is True


@apai
@pytest.mark.parametrize("how", ["raises", "does not latch"])
def test_a_stop_that_cannot_be_put_back_ends_in_the_restart_line(
    rig: Rig, how: str
) -> None:
    end_in_a_skill_stop(rig)
    reset, estop = rig.checker.reset_estop, rig.checker.estop
    calls: list[str] = []

    def broken() -> None:
        calls.append("estop")
        if how == "raises":
            raise RuntimeError("the checker is broken")

    def stop_then_reset() -> None:
        estop()  # after the gate compared the counts
        reset()
        rig.checker.estop = broken

    rig.checker.reset_estop = stop_then_reset
    refused = grasp(rig)
    # The stop was erased and could not be latched again: the checker's word
    # is no longer worth anything, so the program takes no new task.
    assert calls == ["estop"] and rig.checker.estopped is False
    assert refused["motion_block_reason"] == RESTART
    assert rig.kinds() == ["ask", "request", "result", "reset", "request", "result"]
    assert rig.of("result")[-1][1:] == ("nobody", "completed")
    assert rig.gate.restart_needed is True and rig.gate.restart_told is True
    rig.log.clear()
    ran: list[str] = []
    again = rig.gate.bind_run_task(lambda task, **kwargs: ran.append(task))("again")
    assert ran == [] and rig.log == [] and rig.said[-1] == RESTART
    assert (again.success, again.summary) == (False, RESTART)


@apai
def test_apais_checker_is_cleared_and_a_stop_meanwhile_is_put_back(rig: Rig) -> None:
    import numpy as np
    from apai.safety.harness import SafetyHarness, SafetyLimits

    real = SafetyHarness(SafetyLimits(np.full(3, -1.0), np.full(3, 1.0)))
    rig.gate.bind_arm(rig, real)
    rig.checker = real
    end_in_a_skill_stop(rig)
    assert real.estopped is True
    assert grasp(rig) == {"ok": True} and real.estopped is False
    assert rig.of("ask")[0][1].startswith(f"Note: {SKILL_STOP_NOTE}.")
    assert "the skill stop of the last task is cleared" in rig.said

    # Again, with a stop that comes while the gate resets the latch.
    rig.answers = ["yes"]
    end_in_a_skill_stop(rig)
    reset = real.reset_estop

    def stop_then_reset(**kwargs: Any) -> None:
        real.reset_estop = reset
        real.estop("a stop from another thread")
        reset(**kwargs)

    real.reset_estop = stop_then_reset
    count = real._observation_cancel_generation
    refused = grasp(rig)
    assert refused["motion_block_reason"] == (
        f"the workstation checker was stopped again: {SKILL_STOP_KEPT}"
    )
    # The stop and the gate's own latch after it: nothing was opened.
    assert real.estopped is True and real._observation_cancel_generation == count + 2
    assert rig.of("open_task") == [] and rig.owner() == "nobody"
    assert rig.gate.restart_needed is False
    rig.log.clear()
    rig.answers = ["yes"]
    assert grasp(rig) == {"ok": True} and real.estopped is False


def test_a_stop_of_the_checker_right_after_its_latch_was_cleared_ends_the_start(
    rig: Rig,
) -> None:
    end_in_a_skill_stop(rig)
    pin = rig.link.pin

    def stop_then_pin(pinned: Pinned) -> None:
        rig.checker.estop()  # after the gate's look that cleared the latch
        pin(pinned)

    rig.link.pin = stop_then_pin
    del rig.said[:]
    refused = grasp(rig)
    assert refused["refused"] == "ownership"
    assert refused["motion_block_reason"] == (
        f"the workstation checker was stopped again: {SKILL_STOP_KEPT}"
    )
    # The task was closed on the arm again and the arm given back: no skill ran.
    assert rig.kinds() == [
        "ask",
        "request",
        "result",
        "reset",
        "open_task",
        "close_task",
        "request",
        "result",
    ]
    assert rig.of("result")[-1][1:] == ("nobody", "completed")
    assert rig.checker.estopped is True and rig.link.pinned is None
    assert "the skill stop of the last task is cleared" not in rig.said
    assert not any("the agent owns the arm" in line for line in rig.said)
    assert rig.gate.restart_needed is False
    # The next task's question names the latch, and its yes clears it.
    rig.link.pin = pin
    rig.log.clear()
    rig.answers = ["yes"]
    assert grasp(rig) == {"ok": True} and rig.checker.estopped is False


def test_a_stop_of_the_checker_right_before_the_task_opens_ends_the_start(
    rig: Rig,
) -> None:
    pin = rig.link.pin

    def stop_then_pin(pinned: Pinned) -> None:
        rig.checker.estop()  # no latch stood before: the gate cleared nothing
        pin(pinned)

    rig.link.pin = stop_then_pin
    refused = grasp(rig)
    assert SKILL_STOP_KEPT in refused["motion_block_reason"]
    assert rig.kinds()[3:] == ["open_task", "close_task", "request", "result"]
    assert rig.of("reset") == [] and rig.of("skill") == []
    assert rig.checker.estopped is True and rig.owner() == "nobody"


def after_the_grant(rig: Rig, act: Callable[[], None]) -> None:
    """Call act() once the gate saw the agent as owner, before it goes on."""
    await_owner = rig.gate._await_owner

    def then_act(number: Any) -> Any:
        shown = await_owner(number)
        act()
        return shown

    rig.gate._await_owner = then_act


def test_a_stop_under_way_at_the_link_service_keeps_the_latch(rig: Rig) -> None:
    end_in_a_skill_stop(rig)

    def stop_under_way() -> None:
        # What the service's LOCAL_STATUS says between a STOP it forwarded
        # and the ROBOT_STATE that shows it.
        rig.link._carrier.status = types.SimpleNamespace(
            robot_connected=True, close_reason="", stop_unconfirmed=True
        )

    after_the_grant(rig, stop_under_way)
    refused = grasp(rig)
    assert refused["motion_block_reason"] == (
        f"a STOP was sent and the robot app has not shown it yet: {SKILL_STOP_KEPT}"
    )
    assert rig.of("reset") == [] and rig.of("open_task") == []
    assert rig.checker.estopped is True and rig.gate.restart_needed is False
    assert rig.of("result")[-1][1:] == ("nobody", "completed")


def test_a_robot_state_that_is_not_fresh_keeps_the_latch(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig(owner_wait_s=0.5)
    end_in_a_skill_stop(rig)
    real = rig.link._monotonic

    def states_stop() -> None:
        rig.link._take_state = lambda state: None  # no newer state arrives
        rig.link._monotonic = lambda: real() + 1.5

    after_the_grant(rig, states_stop)
    try:
        refused = rig.task(lambda: rig.execute("grasp_object"))
    finally:
        del rig.link._take_state
        rig.link._monotonic = real
    rig.shown("nobody")
    reason = refused["motion_block_reason"]
    assert reason.startswith("robot connection lost: the newest ROBOT_STATE is 1.5")
    assert reason.endswith("s old, over 1 s")
    assert rig.of("reset") == [] and rig.of("open_task") == []
    assert rig.checker.estopped is True and rig.gate.restart_needed is False


def test_the_setting_alone_keeps_a_latch_the_question_named(
    make_rig: Callable[..., Rig],
) -> None:
    """What only _clear_skill_stop's own look at the setting keeps."""
    for clear in (False, True):
        rig = make_rig(clear_skill_stop=clear)
        rig.checker.estop()
        mark = rig.checker._observation_cancel_generation
        assert rig.link.request_owner("agent", timed_return=False).status == "completed"
        state = rig.shown("agent")
        pinned = Pinned(state.session_id, state.owner_number, state.stop_number)
        rig.log.clear()
        kept = rig.gate._clear_skill_stop(pinned, True, mark)
        if clear:
            assert kept is None and rig.of("reset") == [("reset",)]
            assert rig.checker.estopped is False and rig.gate.restart_told is False
        else:
            assert kept == RESTART and rig.of("reset") == []
            assert rig.checker.estopped is True and rig.gate.restart_told is True


def lose_the_next_give_back(rig: Rig) -> None:
    """The robot app never gets the next OWNER_REQUEST "nobody": no RESULT."""
    app = rig.h.app
    give_up = app._give_up

    def lost(message: Any, reply: Any) -> None:
        app._give_up = give_up
        return None

    app._give_up = lost


def test_a_give_back_without_an_answer_is_tried_again(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig(end_wait_s=0.3)
    end_in_a_skill_stop(rig)
    rig.on_ask = rig.checker.estop  # the latch is kept: the arm is given back
    lose_the_next_give_back(rig)
    refused = rig.task(lambda: rig.execute("grasp_object"))
    assert SKILL_STOP_KEPT in refused["motion_block_reason"]
    assert rig.gate.end_unanswered is True and rig.owner() == "agent"
    assert any(
        line.startswith("the arm could not be given back: no final RESULT")
        for line in rig.said
    )
    # What the task prompt does while nothing is typed: the give-back again.
    rig.log.clear()
    rig.on_ask, rig.answers = None, ["yes"]
    rig.gate.return_late_ownership()
    assert [entry[1:] for entry in rig.of("result")] == [("nobody", "completed")]
    assert rig.said[-1] == "the ownership arrived late: the arm was given back"
    assert rig.owner() == "nobody"
    # Once per ownership: nothing more is sent for it.
    rig.log.clear()
    rig.gate.return_late_ownership()
    assert rig.log == []
    # The next task's question names the latch, and its yes clears it.
    assert grasp(rig) == {"ok": True} and rig.checker.estopped is False


def test_a_task_typed_after_a_lost_give_back_gives_the_arm_back_first(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig(end_wait_s=0.3)
    end_in_a_skill_stop(rig)
    rig.on_ask = rig.checker.estop
    lose_the_next_give_back(rig)
    rig.task(lambda: rig.execute("grasp_object"))
    assert rig.owner() == "agent"
    rig.log.clear()
    rig.on_ask, rig.answers = None, ["yes"]
    assert grasp(rig) == {"ok": True}
    assert rig.kinds()[:3] == ["request", "result", "ask"]
    assert rig.of("result")[0][1:] == ("nobody", "completed")


@pytest.mark.parametrize(
    "when",
    ["before the question", "at the status line", "at the question", "granted"],
)
def test_clear_skill_stop_false_clears_no_latch_seen_while_a_task_starts(
    make_rig: Callable[..., Rig], when: str
) -> None:
    rig = make_rig(clear_skill_stop=False)
    if when == "before the question":
        late = rig.gate.return_late_ownership

        def return_late_ownership(*args: Any, **kwargs: Any) -> None:
            rig.checker.estop()  # e.g. a stop from another thread
            late(*args, **kwargs)

        rig.gate.return_late_ownership = return_late_ownership
        expected: list[str] = []
    elif when == "at the status line":
        say = rig.gate._say

        def stop_then_say(text: str) -> None:
            if text.startswith("robot: control owner"):
                rig.checker.estop()  # after the gate's second look
            say(text)

        rig.gate._say = stop_then_say
        expected = []
    elif when == "at the question":
        rig.on_ask = rig.checker.estop
        expected = ["ask"]
    else:
        await_owner = rig.gate._await_owner

        def stop_after_the_grant(number: Any) -> Any:
            shown = await_owner(number)
            rig.checker.estop()
            return shown

        rig.gate._await_owner = stop_after_the_grant
        expected = ["ask", "request", "result", "request", "result"]
    refused = grasp(rig)
    assert refused["refused"] == "ownership"
    assert refused["motion_block_reason"] == RESTART
    assert rig.kinds() == expected
    assert all(SKILL_STOP_NOTE not in entry[1] for entry in rig.of("ask"))
    assert rig.checker.estopped is True and rig.owner() == "nobody"
    assert rig.gate.restart_needed is True and rig.gate.restart_told is True


# The driver the robot app declares


@apai
def test_the_driver_of_the_quoted_session_is_checked_before_the_question(
    make_rig: Callable[..., Rig],
) -> None:
    seen: list[str] = []

    def driver_check(driver: str) -> str | None:
        seen.append(driver)
        return None if driver == "sim" else f"the driver {driver!r} is not driven"

    rig = make_rig(driver_check=driver_check)
    assert grasp(rig) == {"ok": True} and seen == ["sim"]
    assert rig.gate.wrong_robot_app == "" and rig.gate.tell_end() is False
    # The robot app shows another driver for the same session.
    rig.log.clear()
    rig.answers = ["yes"]
    was = rig.link.welcome
    send(rig.robot_end, dataclasses.replace(was, driver="real"))
    until(lambda: rig.link.welcome.driver == "real", "the new WELCOME")
    refused = grasp(rig)
    assert refused["refused"] == "ownership"
    assert refused["motion_block_reason"] == "the driver 'real' is not driven"
    assert rig.kinds() == []  # no question, no owner request
    assert rig.gate.wrong_robot_app == "the driver 'real' is not driven"
    # No later task is run, whatever the robot app declares then.
    send(rig.robot_end, was)
    until(lambda: rig.link.welcome.driver == "sim", "the old WELCOME")
    ran: list[str] = []
    again = rig.gate.bind_run_task(lambda task, **kwargs: ran.append(task))("again")
    assert ran == [] and rig.log == [] and again.success is False
    assert rig.gate.tell_end() is True
    assert rig.said[-1] == (
        "the driver 'real' is not driven: restart the agent program; it takes "
        "no new task"
    )


def test_a_session_without_its_welcome_is_not_quoted(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig(driver_check=lambda driver: None, owner_wait_s=0.3)
    send(rig.robot_end, dataclasses.replace(rig.link.welcome, session_id="other"))
    until(lambda: rig.link.welcome.session_id == "other", "the other WELCOME")
    refused = grasp(rig)
    session = rig.link.state().session_id
    assert refused["motion_block_reason"] == (
        f"no WELCOME of session {session} has arrived (the newest is of session "
        "other): the driver of that robot app is not known"
    )
    assert rig.kinds() == [] and rig.gate.tell_end() is True


def test_the_welcome_of_a_new_session_is_waited_for_and_no_other_is_judged(
    make_rig: Callable[..., Rig],
) -> None:
    seen: list[str] = []

    def driver_check(driver: str) -> str | None:
        seen.append(driver)
        return None if driver == "sim" else f"the driver {driver!r} is not driven"

    rig = make_rig(driver_check=driver_check, owner_wait_s=2.0)
    was = rig.link.welcome
    # A restarted robot app: its ROBOT_STATE is there, and the newest WELCOME
    # is still that of the session before, which declared another driver.
    send(rig.robot_end, dataclasses.replace(was, session_id="before", driver="real"))
    until(lambda: rig.link.welcome.session_id == "before", "the old WELCOME")
    late = threading.Timer(0.3, send, (rig.robot_end, was))
    late.start()
    began = time.monotonic()
    try:
        assert grasp(rig) == {"ok": True}
    finally:
        late.join()
    # The gate waited for the WELCOME of the session it quotes and judged it
    # alone; the task then started as ever.
    assert 0.3 <= time.monotonic() - began < 2.0
    assert seen == ["sim"] and rig.gate.tell_end() is False
    assert rig.kinds()[:4] == ["ask", "request", "result", "open_task"]
    assert rig.of("request")[0][2].session_id == was.session_id


@apai
def test_the_arm_judges_the_rest_of_the_welcome_before_the_question(
    make_rig: Callable[..., Rig],
) -> None:
    judged: list[tuple[str, str]] = []

    def check_welcome(welcome: Any) -> str | None:
        """What the arm adapter compared when it connected."""
        judged.append((welcome.session_id, welcome.camera_calibration_id))
        if welcome.camera_calibration_id == "another":
            return "the robot app's camera calibration is 'another'"
        return None

    rig = make_rig(driver_check=lambda driver: None)
    rig.check_welcome = check_welcome
    was = rig.link.welcome
    assert grasp(rig) == {"ok": True}
    assert judged == [(was.session_id, was.camera_calibration_id)]
    # The robot app now names another camera calibration.
    rig.log.clear()
    rig.answers = ["yes"]
    send(rig.robot_end, dataclasses.replace(was, camera_calibration_id="another"))
    until(
        lambda: rig.link.welcome.camera_calibration_id == "another", "the new WELCOME"
    )
    refused = grasp(rig)
    assert refused["refused"] == "ownership"
    assert refused["motion_block_reason"] == (
        "the robot app's camera calibration is 'another'"
    )
    assert rig.kinds() == [] and rig.owner() == "nobody"  # no question, no request
    # No later task is run.
    ran: list[str] = []
    again = rig.gate.bind_run_task(lambda task, **kwargs: ran.append(task))("again")
    assert ran == [] and rig.log == [] and again.success is False
    assert rig.said[-1] == (
        "the robot app's camera calibration is 'another': restart the agent "
        "program; it takes no new task"
    )
    # A driver that is refused is not handed to the arm.
    judged.clear()
    other = make_rig(driver_check=lambda driver: "not driven")
    other.check_welcome = check_welcome
    assert grasp(other)["motion_block_reason"] == "not driven" and judged == []
