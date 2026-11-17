# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent's tools decide as the functional launcher decides.

The functional launcher is the experiment's thor_runtime.py: the program that
ran APAI's agent on the robot's own computer, with the motors in the same
process. Here the agent runs on the workstation and the motors on the Thor.
Each test compares a piece of this example with that launcher or with APAI's
own code. APAI's rest tool is apai.skills.rest_pose.move_to_rest.
"""

from __future__ import annotations

import ast
import os
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from isaaccapture_examples.agentic_remote.agent import task_gate
from isaaccapture_examples.agentic_remote.agent.link_client import Pinned
from isaaccapture_examples.agentic_remote.protocol.messages import (
    JAW_CLOSE_TORQUE,
    REQUEST_MOTORS_OFF,
    REQUEST_MOTORS_ON,
    Result,
)

apai = pytest.mark.apai
experiment = pytest.mark.experiment

PACKAGE = Path(task_gate.__file__).parents[1]
# The rig's poses, from the experiment's arm profile (configs/thor_operation).
PARK = [0.0] * 6
HOME = [-0.0197, 0.025, 0.025, 0.0078, -0.059, 0.0565]
RAISED = [0.4, 0.9, 0.7, 0.1, -0.3, 0.2]


def launcher_source() -> ast.Module:
    """The functional launcher, parsed: it imports the motor side when run."""
    folder = Path(os.environ["AGENTIC_REMOTE_EXPERIMENT_DIR"])
    return ast.parse((folder / "thor_runtime.py").read_text(encoding="utf-8"))


def texts(tree: ast.AST) -> list[str]:
    """Every piece of text in the source, the pieces of an f-string each."""
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


# The tool list


@experiment
def test_the_read_only_tools_are_the_functional_launchers() -> None:
    (assigned,) = (
        node
        for node in launcher_source().body
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == "READ_ONLY_TOOLS"
    )
    theirs = {
        node.value
        for node in ast.walk(assigned.value)
        if isinstance(node, ast.Constant)
    }
    assert task_gate.READ_ONLY_TOOLS == theirs and len(theirs) == 16


@apai
def test_the_tool_list_is_the_functional_one_with_the_rest_tool() -> None:
    from apai.skills.runtime import _MOTION_SKILLS
    from isaaccapture_examples.agentic_remote.agent import launcher

    # thor_runtime.specs() with --operate: its read-only tools and every
    # motion skill of APAI. The same two sets decide here, nothing taken away.
    assert launcher._motion_skills() == frozenset(_MOTION_SKILLS)
    assert task_gate.REST_SKILL in launcher._motion_skills()
    tree = ast.parse(Path(launcher.__file__).read_text(encoding="utf-8"))
    (specs,) = (
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "specs"
    )
    chosen = ast.unparse(specs)
    assert "motion = _motion_skills()\n" in chosen
    assert "item['name'] in READ_ONLY_TOOLS or item['name'] in motion" in chosen
    # The model is no longer told that the rest tool is missing.
    assert "move_to_rest" not in launcher.TOOLS_GUIDANCE
    assert "return to rest" not in launcher.TOOLS_GUIDANCE


@experiment
def test_the_gates_texts_are_the_functional_launchers() -> None:
    theirs = texts(launcher_source())
    not_admitted = task_gate.NOT_ADMITTED.format(name="")
    for ours in (not_admitted, task_gate.CONNECT_QUESTION, task_gate.TASK_COVERS):
        assert any(ours in text for text in theirs), ours
    assert any(text == task_gate.CONNECT_DECLINED for text in theirs)


# What a tool the gate does not let through gets back


class Rig:
    """A task gate with scripted answers on a link that only keeps a state."""

    def __init__(self, **state: Any) -> None:
        shown = {
            "session_id": "s1",
            "owner": "agent",
            "owner_number": 3,
            "stop_number": 0,
            "stop_level": "none",
            "fault_reason": "",
            "torque_on": True,
            "health": {},
        }
        self.state_now = types.SimpleNamespace(**{**shown, **state})
        self.pinned: Pinned | None = Pinned("s1", 3, 0)
        self.requests: list[str] = []
        self.refuse = ""
        self.asked: list[str] = []
        self.answers: list[str] = []
        self.said: list[str] = []
        self.connected = True
        self.woken = 0
        self.gate = task_gate.TaskGate(
            self,
            self,
            None,
            ask=self.ask,
            say=self.said.append,
            motion_skills={"grasp_object", "move_to_rest"},
        )

    # The link

    def state(self, max_age_s: float | None = None) -> Any:
        return self.state_now

    def request_owner(self, request: str, **more: Any) -> Result:
        self.requests.append(request)
        if self.refuse:
            return Result("c1", "refused", self.refuse)
        if request == REQUEST_MOTORS_ON:
            self.state_now.torque_on = True
        return Result("c1", "completed")

    def wait_change(self, timeout_s: float) -> None:
        return None

    # The arm adapter

    def motors_on(self) -> None:
        self.connected = True
        self.woken += 1

    def ask(self, prompt: str) -> str:
        self.asked.append(prompt)
        return self.answers.pop(0) if self.answers else "no"

    def owning(self) -> Any:
        """A task of this gate that owns the arm; returns the token to reset."""
        task = task_gate._Task("put the cube away")
        task.owner = True
        return self.gate._current.set(task)

    def run(self, name: str, **args: Any) -> Any:
        return self.gate.execute(
            lambda name, args: {"ok": True, "ran": name}, name, args
        )


def test_a_refused_tool_gets_the_functional_text_and_the_task_goes_on() -> None:
    rig = Rig()
    refused = rig.run("grasp_object")  # no task was typed
    functional = "grasp_object not admitted in this session; no command sent"
    assert refused["ok"] is False and refused["error"].startswith(functional)
    # ... followed by the reason of the split, and nothing ends the task.
    assert refused["error"] == f"{functional}: {task_gate.NO_TASK}"
    assert "motion_blocked" not in refused
    # Only the robot's own state, which needs the operator first, ends it.
    blocked = task_gate._refusal("grasp_object", "fault-stop", task_gate.BLOCKED)
    assert blocked["motion_blocked"] is True and blocked["error"].startswith(functional)


def test_a_tool_in_neither_list_is_not_admitted_as_in_the_functional_launcher() -> None:
    rig = Rig()
    token = rig.owning()
    try:
        result = rig.run("preview_grasp")
        looked = rig.run("get_observation")
    finally:
        rig.gate._current.reset(token)
    assert result["ok"] is False and "ran" not in result
    assert result["error"].startswith(
        "preview_grasp not admitted in this session; no command sent"
    )
    assert looked == {"ok": True, "ran": "get_observation"}


# Standby: after the rest tool the motors are off until the next motion skill


def test_the_next_motion_skill_asks_the_functional_question_before_motors_on() -> None:
    rig = Rig(torque_on=False)
    rig.connected = False  # standby: the rest tool switched the motors off
    rig.answers = ["yes"]
    token = rig.owning()
    try:
        result = rig.run("grasp_object")
    finally:
        rig.gate._current.reset(token)
    (asked,) = rig.asked
    assert task_gate.CONNECT_QUESTION in asked and "Type yes to proceed" in asked
    assert rig.requests == [REQUEST_MOTORS_ON] and rig.woken == 1
    assert result == {"ok": True, "ran": "grasp_object"}


def test_without_the_operators_yes_the_motors_stay_off() -> None:
    rig = Rig(torque_on=False)
    rig.connected = False
    rig.answers = ["no"]
    token = rig.owning()
    try:
        result = rig.run("grasp_object")
    finally:
        rig.gate._current.reset(token)
    assert rig.requests == [] and rig.woken == 0 and "ran" not in result
    assert result["error"].endswith(task_gate.CONNECT_DECLINED)
    assert "motion_blocked" not in result  # the model can report, as it does there


def test_the_rest_tool_in_standby_asks_nothing_and_switches_nothing_on() -> None:
    rig = Rig(torque_on=False)
    rig.connected = False
    token = rig.owning()
    try:
        result = rig.run("move_to_rest")
    finally:
        rig.gate._current.reset(token)
    # APAI's own tool answers "motors already off (standby)" by itself.
    assert result == {"ok": True, "ran": "move_to_rest"}
    assert rig.asked == [] and rig.requests == []


def test_motors_off_at_rest_do_not_refuse_a_task() -> None:
    state = Rig(torque_on=False, owner="nobody").state_now
    assert task_gate._not_startable(state) is None
    state.stop_level = "fault"
    assert "fault-stop" in task_gate._not_startable(state)


def test_a_task_the_model_ends_without_success_asks_for_no_help() -> None:
    # The functional launcher prints the report and asks for the next task.
    task = task_gate._Task("pick the cube")
    gave_up = types.SimpleNamespace(
        success=False, cancelled=False, stuck=None, summary="the cube is out of reach"
    )
    assert task_gate._ending(task, gave_up, None, False) == (
        "nobody",
        "task ended without success: the cube is out of reach",
    )
    # What needs a person still asks for one.
    stuck = types.SimpleNamespace(success=False, cancelled=False, stuck="no way")
    assert task_gate._ending(task, stuck, None, False)[0] == "help_needed"
    assert task_gate._ending(task, None, RuntimeError("x"), False)[0] == "help_needed"


# APAI's rest tool through the arm adapter


class RecordingArm:
    """APAI's SafeArm as its rest tool uses it; every move reaches its goal."""

    def __init__(self, raw: Any, start: list[float], log: list[tuple]) -> None:
        self.raw, self.n_joints, self.log = raw, len(start), log
        self.q = np.asarray(start, dtype=float)
        self.harness = types.SimpleNamespace(
            estopped=False,
            park_q=list(PARK),
            _pending_contact_episode=None,
            _pending_release_episode=None,
        )

    def get_state(self) -> Any:
        return types.SimpleNamespace(q=self.q.copy())

    def move_planned(self, target: Any, duration_s: float) -> bool:
        return self._move("move_planned", target, duration_s, None)

    def move_joints(
        self, target: Any, duration_s: float, joint_margin: float | None = None
    ) -> bool:
        return self._move("move_joints", target, duration_s, joint_margin)

    def _move(self, how: str, target: Any, duration_s: float, margin: Any) -> bool:
        goal = [float(value) for value in target]
        self.log.append((how, goal, duration_s, margin))
        self.q = np.asarray(goal, dtype=float)
        return True


def functional_rest(
    monkeypatch: pytest.MonkeyPatch,
    raw: Any,
    start: list[float],
    log: list[tuple],
    *,
    held: str | None = None,
) -> dict:
    """Run APAI's move_to_rest on a recording arm; its result.

    The two route checks of APAI need its checker and are left out: they
    move nothing.
    """
    from apai.safety import trajectory
    from apai.skills import rest_pose

    monkeypatch.setattr(trajectory, "plan_route", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        rest_pose,
        "vet_fold",
        lambda harness, start, park, rate: (
            rest_pose.fold_waypoints(start, park),
            None,
        ),
    )
    runtime = types.SimpleNamespace(
        arm=RecordingArm(raw, start, log),
        held_object=held,
        _held_provisional=None,
        _contact_episode=None,
        _carry_attachment=None,
        _release_episode=None,
        _profile_q=lambda name, why: np.asarray(HOME, dtype=float),
        memory=types.SimpleNamespace(add=lambda *args: None),
    )
    return rest_pose.move_to_rest(runtime)


class ThorLink:
    """The robot link as the arm adapter's rest methods use it."""

    def __init__(self, log: list[tuple]) -> None:
        self.log = log
        self.torque = True
        self.pinned: Pinned | None = Pinned("s1", 3, 0)
        self.refuse = ""

    def state(self, max_age_s: float | None = None) -> Any:
        return types.SimpleNamespace(torque_on=self.torque)

    def request_owner(self, request: str, **more: Any) -> Result:
        self.log.append(("owner_request", request, more.get("quote")))
        if self.refuse:
            return Result("c1", "refused", self.refuse)
        if request == REQUEST_MOTORS_OFF:
            self.torque = False  # the Thor found the arm at measured rest
        return Result("c1", "completed")

    def lost_reason(self) -> str | None:
        return None

    def unpin(self) -> None:
        self.pinned = None

    def final_result(self, command_id: str) -> None:
        return None


@pytest.fixture
def adapter() -> Any:
    """The arm adapter inside a task, with the Thor's motors on."""
    from isaaccapture_examples.agentic_remote.agent.arm_adapter import RobotLinkArm

    log: list[tuple] = []
    link = ThorLink(log)
    made = RobotLinkArm({"n_joints": 6}, None, link=link)
    made.connected = True
    made.open_task(link.pinned)

    def jaw(opening: float, effort: float, mode: str, *args: Any, **kwargs: Any) -> Any:
        log.append(("gripper", opening, mode))
        return Result("g1", "completed", detail={"outcome": "reached", "closed": True})

    made._jaw = jaw
    made.log, made.link = log, link
    return made


@apai
def test_the_adapter_is_the_kind_of_arm_apais_rest_tool_switches_off(
    adapter: Any,
) -> None:
    from apai.control.arm_base import ArmBase
    from apai.control.lazy_arm import LazyArm
    from isaaccapture_examples.agentic_remote.agent import arm_adapter

    # APAI's rest tool releases only a LazyArm whose drives a disconnect ends.
    assert isinstance(adapter, LazyArm)
    assert adapter.disconnect_preserves_drive_state is False
    # Its teardown must neither park this arm nor count its torque as cut.
    assert adapter.shutdown_park_required is False
    assert adapter.disconnect_cuts_torque is False
    # No call is passed on to a backend LazyArm would make: each name of
    # LazyArm is the adapter's own or ArmBase's, but the few that are kept.
    for name, theirs in vars(LazyArm).items():
        if name.startswith("__") or name in arm_adapter.LAZY_KEPT:
            continue
        if not (callable(theirs) or isinstance(theirs, property)):
            continue
        ours = getattr(type(adapter), name)
        assert ours is not theirs, name
        assert ours in (
            vars(arm_adapter.RobotLinkArm).get(name),
            getattr(ArmBase, name, None),
        ), name
    with pytest.raises(AttributeError):
        adapter.no_such_extra  # noqa: B018 - a probe of APAI must not wake a backend
    # APAI reads this name to end a task after a skill stop; its own arm hides
    # it, and so the functional agent goes on. The adapter's latch has another.
    assert not hasattr(adapter, "_stopped")
    source = Path(arm_adapter.__file__).read_text(encoding="utf-8")
    assert "_stopped" not in source


@apai
def test_apais_rest_tool_folds_closes_the_jaws_and_asks_for_the_motors_off(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from apai.skills import rest_pose

    result = functional_rest(monkeypatch, adapter, RAISED, adapter.log)
    kinds = [entry[0] for entry in adapter.log]
    # Up to the ready pose, the fold in APAI's steps, the driver's close,
    # then the motors off: in that order, each once.
    folds = rest_pose.SEGMENTS
    assert kinds == [
        "move_planned",
        *["move_joints"] * folds,
        "gripper",
        "owner_request",
    ]
    assert adapter.log[0][1:3] == (pytest.approx(HOME), rest_pose.HOME_S)
    assert adapter.log[folds][1] == pytest.approx(PARK)
    assert adapter.log[-2] == ("gripper", 0.0, JAW_CLOSE_TORQUE)
    assert adapter.log[-1] == ("owner_request", REQUEST_MOTORS_OFF, Pinned("s1", 3, 0))
    # APAI's own account of it, for the model.
    assert (result["at"], result["torque"]) == ("rest", "released")
    assert result["safe_to_power_off"] is True and result["arm_state"] == "standby"
    assert result["gripper"] == "closed" and result["route"] == ["initial_pose", "fold"]
    assert adapter.connected is False
    # In standby the tool answers by itself: nothing more is sent.
    sent = len(adapter.log)
    again = functional_rest(monkeypatch, adapter, PARK, adapter.log)
    assert again["at"] == "standby" and len(adapter.log) == sent


@apai
def test_the_rest_tool_refuses_with_a_held_load_in_apais_words(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = functional_rest(monkeypatch, adapter, RAISED, adapter.log, held="cube")
    assert result["ok"] is False and result["refused"] == "holding"
    assert result["error"].startswith("it is holding 'cube': put it down first")
    assert adapter.log == [] and adapter.connected is True  # nothing moved or sent


@apai
def test_where_the_thor_keeps_the_motors_on_the_model_is_told_so(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter.link.refuse = "the arm is not at rest: joint index 1 is at 0.2000 rad"
    result = functional_rest(monkeypatch, adapter, RAISED, adapter.log)
    assert result["ok"] is False and result["safe_to_power_off"] is False
    assert result["error"] == (
        "switching the motors off failed: RobotLinkError: the robot app kept "
        f"the motors on: {adapter.link.refuse}"
    )
    assert adapter.connected is True and adapter.link.torque is True


@apai
def test_outside_a_task_the_adapters_disconnect_sends_nothing(adapter: Any) -> None:
    adapter.close_task()  # as when the program ends: APAI's teardown follows
    adapter.disconnect()
    assert adapter.log == [] and adapter.link.torque is True
