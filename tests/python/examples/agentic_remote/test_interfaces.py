# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared robot app types in
isaaccapture_examples.agentic_remote.robot_app.interfaces."""

from __future__ import annotations

import dataclasses
import inspect
import json
from types import SimpleNamespace

import pytest
from isaaccapture_examples.agentic_remote.robot_app.interfaces import (
    ArmFault,
    ArmPort,
    AuthorityState,
    Checker,
    CheckerSet,
    ContactStop,
    FeedbackLost,
    ContactWatch,
    GripperOutcome,
    JawLoad,
    JointReading,
    MotionState,
    MoveOutcome,
    Owner,
    ReadFailed,
    Refusal,
    RestOutcome,
    StopLevel,
)

# Parameter names of every method of the three ports, without self.
SIGNATURES = {
    ArmPort: {
        "connect": [],
        "torque_on": [],
        "send_target": ["q"],
        "last_target": [],
        "read_joint": ["joint"],
        "read_all": [],
        "hold_measured": [],
        "release_torque": [],
        "release_confirmed": [],
        "faults": [],
        "temperatures": [],
        "gripper_referenced": [],
        "gripper_reference": ["cancel"],
        "gripper_read": [],
        "gripper_last_target": [],
        "gripper_command": ["opening", "max_effort", "mode", "cancel"],
        "gripper_light_hold": [],
    },
    ContactWatch: {
        "after_send": ["index", "q", "deadline_mono"],
        "hold": [],
        "finish": [],
    },
    Checker: {
        "feed_watchdog": ["t_mono"],
        "halt_count": [],
        "halt": ["reason"],
        "latch": ["reason"],
        "reset_latch": [],
        "latched": [],
        "begin_motion": ["halt_count"],
        "end_motion": [],
        "check_setpoint": ["q_prev", "q_next", "dt"],
        "check_pose": ["q"],
        "check_low_pose": ["q"],
        "check_path": ["q_start", "setpoints", "dt"],
        "check_grasp_region": ["region"],
        "set_grasp_region": ["region"],
        "note_contact": ["q"],
        "clear_contact_spots": [],
        "tip_blocked": ["q", "region"],
        "tip_state": ["q", "region"],
        "contact_watch": ["arm", "q_start", "history"],
    },
}

ARM_PORT_MEMBERS = {"n_joints", "joint_names", *SIGNATURES[ArmPort]}
CONTACT_WATCH_MEMBERS = set(SIGNATURES[ContactWatch])
CHECKER_MEMBERS = {"name", "version", *SIGNATURES[Checker]}


def _members(protocol: type) -> set[str]:
    """The public names a protocol declares: methods and annotated attributes."""
    names = set(inspect.get_annotations(protocol))
    names.update(name for name, value in vars(protocol).items() if callable(value))
    return {name for name in names if not name.startswith("_")}


def _stand_in(members: set[str]) -> SimpleNamespace:
    return SimpleNamespace(**dict.fromkeys(members, lambda *args, **kwargs: None))


@pytest.mark.parametrize(
    ("enum", "values"),
    [
        (
            Owner,
            {
                "NOBODY": "nobody",
                "AGENT": "agent",
                "HUMAN": "human",
                "RECOVERY": "recovery",
            },
        ),
        (StopLevel, {"NONE": "none", "STOP": "stop", "FAULT": "fault"}),
        (JawLoad, {"EMPTY": "empty", "LOADED": "loaded", "UNKNOWN": "unknown"}),
        (
            MotionState,
            {
                "IDLE": "idle",
                "MOVING": "moving",
                "HOLD": "hold",
                "REST_HOLD": "rest_hold",
                "REST_UNPOWERED": "rest_unpowered",
                "RELEASE_FAILED": "release_failed",
            },
        ),
    ],
)
def test_enum_names_and_values(enum, values):
    assert {member.name: member.value for member in enum} == values
    assert issubclass(enum, str)


def test_enums_behave_as_their_string_values():
    assert Owner.AGENT == "agent"
    assert Owner("recovery") is Owner.RECOVERY
    assert isinstance(StopLevel.FAULT, str)
    assert str(MotionState.REST_HOLD) == "rest_hold"
    assert f"{JawLoad.UNKNOWN}" == "unknown"
    assert json.dumps({"owner": Owner.NOBODY}) == '{"owner": "nobody"}'


def test_unknown_enum_value_is_rejected():
    with pytest.raises(ValueError, match=r"'robot' is not a valid Owner"):
        Owner("robot")


def test_refusal_carries_its_reason():
    refusal = Refusal("joint 2 above its limit")
    assert refusal.reason == "joint 2 above its limit"
    assert str(refusal) == "joint 2 above its limit"


def test_contact_stop_is_a_refusal_with_a_reason():
    with pytest.raises(Refusal) as caught:
        raise ContactStop("joint 4 trails its target")
    assert caught.value.reason == "joint 4 trails its target"


def test_lost_feedback_is_a_refusal_and_not_a_contact():
    lost = FeedbackLost("joint 1 gave no reading for 0.36 s")
    assert isinstance(lost, Refusal)
    assert not isinstance(lost, ContactStop)
    assert lost.reason == "joint 1 gave no reading for 0.36 s"


def test_read_and_arm_failures_are_not_refusals():
    assert not issubclass(ReadFailed, Refusal)
    assert not issubclass(ArmFault, Refusal)
    assert issubclass(ReadFailed, Exception)
    assert issubclass(ArmFault, Exception)


def test_joint_reading_is_valid_by_default_and_frozen():
    reading = JointReading(joint=2, value=0.5, t_start=1000.0, t_end=1000.008)
    assert reading.valid is True
    assert reading == JointReading(2, 0.5, 1000.0, 1000.008, True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        reading.value = 0.6


def test_gripper_outcome_defaults_and_is_frozen():
    outcome = GripperOutcome(status="stalled", opening=0.4)
    assert outcome.detail == ""
    assert GripperOutcome("refused", None, "fault latched").opening is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.status = "reached"


def test_authority_state_fields_and_frozen():
    state = AuthorityState(
        session_id="abc", owner=Owner.AGENT, owner_number=3, stop_number=1
    )
    assert [field.name for field in dataclasses.fields(AuthorityState)] == [
        "session_id",
        "owner",
        "owner_number",
        "stop_number",
        "stop_level",
        "fault_reason",
    ]
    assert state.stop_level is StopLevel.NONE
    assert state.fault_reason == ""
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.owner_number = 4


def test_move_outcome_fields():
    outcome = MoveOutcome("contact", "joint 4 trails its target", 17, [0.1, 0.2])
    assert outcome.status == "contact"
    assert outcome.reason == "joint 4 trails its target"
    assert outcome.index == 17
    assert outcome.end_q == [0.1, 0.2]
    assert MoveOutcome("completed") == MoveOutcome("completed", "", 0, None)


def test_rest_outcome_fields():
    outcome = RestOutcome("refused", "jaws may be holding")
    assert (outcome.status, outcome.reason) == ("refused", "jaws may be holding")
    assert RestOutcome("rest_hold").reason == ""


@pytest.mark.parametrize(
    ("protocol", "members"),
    [
        (ArmPort, ARM_PORT_MEMBERS),
        (ContactWatch, CONTACT_WATCH_MEMBERS),
        (Checker, CHECKER_MEMBERS),
    ],
)
def test_protocol_declares_the_listed_members(protocol, members):
    assert _members(protocol) == members


@pytest.mark.parametrize("protocol", SIGNATURES, ids=lambda cls: cls.__name__)
def test_protocol_methods_take_the_listed_parameters(protocol):
    declared = {
        name: list(inspect.signature(method).parameters)[1:]
        for name, method in vars(protocol).items()
        if inspect.isfunction(method) and not name.startswith("_")
    }
    assert declared == SIGNATURES[protocol]


def test_only_the_watchdog_time_and_the_reference_cancel_are_optional():
    # The start of the app references the jaws without a cancel.
    optional = {
        "feed_watchdog": {"t_mono": None},
        "gripper_reference": {"cancel": None},
    }
    for protocol in SIGNATURES:
        for name, method in vars(protocol).items():
            if not inspect.isfunction(method) or name.startswith("_"):
                continue
            defaults = {
                parameter.name: parameter.default
                for parameter in inspect.signature(method).parameters.values()
                if parameter.default is not inspect.Parameter.empty
            }
            assert defaults == optional.get(name, {}), f"{protocol.__name__}.{name}"


@pytest.mark.parametrize(
    ("protocol", "members"),
    [
        (ArmPort, ARM_PORT_MEMBERS),
        (ContactWatch, CONTACT_WATCH_MEMBERS),
        (Checker, CHECKER_MEMBERS),
    ],
)
def test_protocol_is_checked_at_runtime(protocol, members):
    assert isinstance(_stand_in(members), protocol)
    assert not isinstance(object(), protocol)
    for missing in sorted(members):
        assert not isinstance(_stand_in(members - {missing}), protocol)


def _checker_set() -> CheckerSet:
    return CheckerSet(
        agent=SimpleNamespace(name="agent"),
        human=SimpleNamespace(name="human"),
        recovery=SimpleNamespace(name="recovery"),
    )


@pytest.mark.parametrize("owner", [Owner.AGENT, Owner.HUMAN, Owner.RECOVERY])
def test_checker_set_for_owner(owner):
    checkers = _checker_set()
    assert checkers.for_owner(owner).name == owner.value
    assert checkers.for_owner(owner.value) is checkers.for_owner(owner)


def test_checker_set_has_no_checker_for_nobody():
    with pytest.raises(ValueError, match=r"nobody has no checker"):
        _checker_set().for_owner(Owner.NOBODY)


def test_checker_set_rejects_an_unknown_owner():
    with pytest.raises(ValueError, match=r"is not a valid Owner"):
        _checker_set().for_owner("operator")


def test_checker_set_all_is_agent_human_recovery():
    checkers = _checker_set()
    assert checkers.all() == (checkers.agent, checkers.human, checkers.recovery)
