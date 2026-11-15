# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Robot link messages: type numbers, JSON round trip and strict decoding."""

from __future__ import annotations

import dataclasses
import enum
import json
import sys
from pathlib import Path

import pytest

try:
    import isaaccapture_examples.agentic_remote.protocol  # noqa: F401
except ImportError:
    _root = next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "VERSION").is_file() and (parent / "CMakeLists.txt").is_file()
    )
    sys.path.insert(0, str(_root / "examples" / "agentic_remote" / "python"))

from isaaccapture_examples.agentic_remote.protocol import messages
from isaaccapture_examples.agentic_remote.protocol.messages import (
    COMMAND_TYPES,
    MAX_PAYLOAD_BYTES,
    CheckerState,
    Event,
    GraspRegion,
    Gripper,
    GripperState,
    JointSample,
    Move,
    Open,
    OwnerRequest,
    Permit,
    ProtocolError,
    Release,
    Reset,
    Result,
    RobotState,
    Shutdown,
    Stop,
    ViewStatus,
    Welcome,
    command_id_at_head,
    command_id_in,
    decode,
    is_reliable,
)

# Name and type number of every link message, by its delivery in section 6 of
# the design.
RELIABLE = {
    "Welcome": 64,
    "Move": 65,
    "Gripper": 66,
    "CheckerState": 67,
    "Stop": 68,
    "OwnerRequest": 69,
    "Reset": 70,
    "Release": 71,
    "Open": 72,
    "Shutdown": 73,
    "Result": 74,
    "Event": 75,
}
NEWEST_WINS = {"RobotState": 96, "Permit": 97, "ViewStatus": 98}

NAN = float("nan")
INF = float("inf")
SESSION = "3f9c1e0a7b2d4c5e8f6a9b0c1d2e3f40"
REST_Q = [0.0, -1.57, 1.57, 0.0, 0.3, 0.0]

WELCOME = Welcome(
    session_id=SESSION,
    robot_id="rebot-b601-01",
    profile_id="quest-right-controller",
    joint_names=["j1", "j2", "j3", "j4", "j5", "j6"],
    joint_min=[-2.9, -1.9, -2.2, -3.0, -1.6, -3.0],
    joint_max=[2.9, 1.9, 2.2, 3.0, 1.6, 3.0],
    rest_q=REST_Q,
    setpoint_period_s=0.02,
    max_move_setpoints=3000,
    limit_set_versions={"agent": "2026-11-13a", "human": "h3", "recovery": "r1"},
    camera_calibration_id="calib-7",
)
MOVE = Move(
    command_id="move-0001",
    session_id=SESSION,
    owner_number=3,
    stop_number=1,
    start_q=REST_Q,
    start_tol_rad=0.05,
    setpoints=[[0.001 * (i + 1) * (j + 1) for j in range(6)] for i in range(4)],
    period_s=0.02,
    grasp_region=GraspRegion(
        center_xy=(0.31, -0.05), radius_m=0.04, z_min=0.0, z_max=None
    ),
    evidence_wall_s=1700000123.25,
    expires_mono_s=1012.5,
)
MOVE_PLAIN = dataclasses.replace(
    MOVE, command_id="move-0002", grasp_region=None, evidence_wall_s=None
)
MOVE_CAPPED_REGION = dataclasses.replace(
    MOVE,
    command_id="move-0003",
    grasp_region=GraspRegion(
        center_xy=(-0.2, 0.15), radius_m=0.03, z_min=0.01, z_max=0.12
    ),
)
GRIPPER = Gripper(
    command_id="grip-1",
    session_id=SESSION,
    owner_number=3,
    stop_number=1,
    opening=0.0,
    max_effort=0.4,
    mode="close_until_contact",
)
CHECKER_STATE = CheckerState(
    command_id="chk-1",
    session_id=SESSION,
    owner_number=3,
    stop_number=1,
    action="clear_contact_spots",
)
STOP = Stop(reason="operator typed stop", sender="console")
OWNER_REQUEST = OwnerRequest(
    command_id="own-1",
    session_id=SESSION,
    owner_number=3,
    stop_number=1,
    request="help_needed",
    reason="three steps failed in a row",
)
OWNER_REQUEST_BARE = OwnerRequest(
    command_id="own-2",
    session_id=SESSION,
    owner_number=4,
    stop_number=2,
    request="agent",
)
OWNER_REQUEST_UNTIMED = dataclasses.replace(
    OWNER_REQUEST_BARE, command_id="own-3", timed_return=False
)
OWNER_REQUEST_REST = OwnerRequest(
    command_id="own-4",
    session_id=SESSION,
    owner_number=4,
    stop_number=2,
    request="rest",
    confirmed=True,
)
# What the operator's four other commands quote.
QUOTED = {"session_id": SESSION, "owner_number": 7, "stop_number": 4}
RESET = Reset(command_id="rst-1", **QUOTED)
RELEASE = Release(command_id="rel-1", **QUOTED)
OPEN = Open(command_id="opn-1", **QUOTED, confirmed=True)
SHUTDOWN = Shutdown(command_id="off-1", **QUOTED)
SHUTDOWN_CONFIRMED = dataclasses.replace(SHUTDOWN, command_id="off-2", confirmed=True)
RESULT = Result(
    command_id="move-0001",
    status="completed",
    reason="",
    end_q=[0.004, 0.008, 0.012, 0.016, 0.02, 0.024],
    detail={"index": 4, "settled": True, "late": [0.001, 0.0], "note": None},
)
RESULT_BARE = Result(command_id="move-0002", status="accepted")
RESULT_REFUSED = Result(
    command_id="move-0003", status="refused", reason="old stop number"
)
EVENT = Event(
    seq=17,
    kind="bell",
    text="ARM WILL MOVE TO REST",
    data={"remaining_s": 10.0, "marks": [30.0, 10.0, 5.0]},
)
EVENT_BARE = Event(seq=18, kind="help_needed", text="agent failed: stuck")
JOINTS = [
    JointSample(
        joint=i,
        value=REST_Q[i] + 0.01 * i,
        t_start=1000.0 + 0.008 * i,
        t_end=1000.008 + 0.008 * i,
        valid=i != 4,
    )
    for i in range(6)
]
STATE = RobotState(
    seq=412,
    session_id=SESSION,
    clock_mono_s=1000.06,
    clock_wall_s=1700000000.06,
    joints=JOINTS,
    recent=JOINTS + JOINTS[:2],
    target_q=[0.004, 0.008, 0.012, 0.016, 0.02, 0.024],
    target_t=1000.04,
    gripper=GripperState(
        opening=0.42,
        t=1000.05,
        valid=True,
        referenced=True,
        last_target=0.4,
        may_be_holding=True,
        load="loaded",
    ),
    owner="agent",
    owner_number=3,
    stop_number=1,
    stop_level="none",
    fault_reason="",
    limit_set_version="2026-11-13a",
    motion="moving",
    move_id="move-0001",
    move_index=2,
    move_total=4,
    last_result={"command_id": "grip-1", "status": "completed", "end_q": None},
    assistance_wait_s=87.5,
    torque_on=True,
    view_age_s=0.2,
    teleop_stage="",
    controller_age_s=0.04,
    health={"read_failures": 2, "oldest_reading_s": 0.11, "temperatures": [31.5]},
)
STATE_UNPOWERED = dataclasses.replace(
    STATE,
    seq=1,
    joints=[],
    recent=[],
    target_q=None,
    target_t=None,
    gripper=GripperState(
        opening=None,
        t=None,
        valid=False,
        referenced=False,
        last_target=None,
        may_be_holding=False,
        load="unknown",
    ),
    owner="nobody",
    stop_level="fault",
    fault_reason="joint feedback lost",
    motion="rest_unpowered",
    move_id="",
    move_index=0,
    move_total=0,
    last_result=None,
    assistance_wait_s=None,
    torque_on=False,
    view_age_s=None,
    controller_age_s=None,
    health={},
)
PERMIT = Permit(
    move_id="move-0001",
    session_id=SESSION,
    owner_number=3,
    stop_number=1,
    through_index=2,
    evidence_wall_s=1700000123.5,
    expires_mono_s=1000.9,
)
VIEW_STATUS = ViewStatus(live=True, age_s=0.12)

EXAMPLES = [
    WELCOME,
    MOVE,
    MOVE_PLAIN,
    MOVE_CAPPED_REGION,
    GRIPPER,
    CHECKER_STATE,
    STOP,
    OWNER_REQUEST,
    OWNER_REQUEST_BARE,
    OWNER_REQUEST_UNTIMED,
    OWNER_REQUEST_REST,
    RESET,
    RELEASE,
    OPEN,
    SHUTDOWN,
    SHUTDOWN_CONFIRMED,
    RESULT,
    RESULT_BARE,
    RESULT_REFUSED,
    EVENT,
    EVENT_BARE,
    STATE,
    STATE_UNPOWERED,
    PERMIT,
    VIEW_STATUS,
]


def _id(value: object) -> str | None:
    """Test id of a link message; None leaves other parameters to pytest."""
    return type(value).__name__ if isinstance(value, messages.Message) else None


def _where(message: messages.Message, path: tuple = ()) -> str:
    """The value at `path` as an error names it, such as Move.setpoints[3][2]."""
    parts = [f"[{key}]" if isinstance(key, int) else f".{key}" for key in path]
    return type(message).__name__ + "".join(parts)


def _changed(message: messages.Message, path: tuple, value: object) -> bytes:
    """The payload of `message` with the JSON value at `path` replaced."""
    obj = json.loads(message.to_payload())
    target = obj
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return json.dumps(obj).encode()


def _without(message: messages.Message, *names: str) -> bytes:
    """The payload of `message` with the named top-level fields left out."""
    obj = json.loads(message.to_payload())
    for name in names:
        del obj[name]
    return json.dumps(obj).encode()


def _required_fields() -> list[tuple[messages.Message, str]]:
    cases = {}
    for message in EXAMPLES:
        for item in dataclasses.fields(message):
            if (
                item.default is dataclasses.MISSING
                and item.default_factory is dataclasses.MISSING
            ):
                cases[type(message), item.name] = (message, item.name)
    return list(cases.values())


# What every command begins with (design section 6), in order.
QUOTING = {
    "command_id": ...,
    "session_id": ...,
    "owner_number": ...,
    "stop_number": ...,
}

# Every field of every link message in order; ... marks one without a default.
# Section 6 of the design says what each carries: every command quotes the owner
# number and the stop number.
MESSAGE_FIELDS = {
    Welcome: {
        "protocol_version": 1,
        "session_id": ...,
        "robot_id": ...,
        "profile_id": ...,
        "joint_names": ...,
        "joint_min": ...,
        "joint_max": ...,
        "rest_q": ...,
        "setpoint_period_s": ...,
        "max_move_setpoints": ...,
        "limit_set_versions": ...,
        "camera_calibration_id": "",
    },
    Move: {
        "command_id": ...,
        "session_id": ...,
        "owner_number": ...,
        "stop_number": ...,
        "start_q": ...,
        "start_tol_rad": ...,
        "setpoints": ...,
        "period_s": ...,
        "grasp_region": ...,
        "evidence_wall_s": ...,
        "expires_mono_s": ...,
    },
    Gripper: {
        "command_id": ...,
        "session_id": ...,
        "owner_number": ...,
        "stop_number": ...,
        "opening": ...,
        "max_effort": ...,
        "mode": ...,
    },
    CheckerState: {
        "command_id": ...,
        "session_id": ...,
        "owner_number": ...,
        "stop_number": ...,
        "action": ...,
    },
    Stop: {"reason": ..., "sender": ...},
    OwnerRequest: {
        **QUOTING,
        "request": ...,
        "reason": "",
        "timed_return": True,
        "confirmed": False,
    },
    Reset: QUOTING,
    Release: QUOTING,
    Open: {**QUOTING, "confirmed": ...},
    # A shutdown says whether the path was confirmed (design sections 6 and 11).
    Shutdown: {**QUOTING, "confirmed": False},
    # end_q follows a field with a default, so it has one too.
    Result: {
        "command_id": ...,
        "status": ...,
        "reason": "",
        "end_q": None,
        "detail": {},
    },
    Event: {"seq": ..., "kind": ..., "text": ..., "data": {}},
    RobotState: dict.fromkeys(
        (
            "seq",
            "session_id",
            "clock_mono_s",
            "clock_wall_s",
            "joints",
            "recent",
            "target_q",
            "target_t",
            "gripper",
            "owner",
            "owner_number",
            "stop_number",
            "stop_level",
            "fault_reason",
            "limit_set_version",
            "motion",
            "move_id",
            "move_index",
            "move_total",
            "last_result",
            "assistance_wait_s",
            "torque_on",
            "view_age_s",
            "teleop_stage",
            "controller_age_s",
            "health",
        ),
        ...,
    ),
    Permit: {
        "move_id": ...,
        "session_id": ...,
        "owner_number": ...,
        "stop_number": ...,
        "through_index": ...,
        "evidence_wall_s": ...,
        "expires_mono_s": ...,
    },
    ViewStatus: {"live": ..., "age_s": ...},
    GraspRegion: {"center_xy": ..., "radius_m": ..., "z_min": ..., "z_max": ...},
    JointSample: dict.fromkeys(("joint", "value", "t_start", "t_end", "valid"), ...),
    GripperState: dict.fromkeys(
        (
            "opening",
            "t",
            "valid",
            "referenced",
            "last_target",
            "may_be_holding",
            "load",
        ),
        ...,
    ),
}


def _default(item: dataclasses.Field) -> object:
    if item.default_factory is not dataclasses.MISSING:
        return item.default_factory()
    return ... if item.default is dataclasses.MISSING else item.default


@pytest.mark.parametrize("cls", MESSAGE_FIELDS, ids=lambda cls: cls.__name__)
def test_fields_and_defaults_match_the_table(cls):
    declared = {item.name: _default(item) for item in dataclasses.fields(cls)}
    assert list(declared.items()) == list(MESSAGE_FIELDS[cls].items())


def test_the_table_of_fields_covers_every_link_message():
    assert set(messages.MESSAGE_TYPES.values()) <= set(MESSAGE_FIELDS)


def test_type_numbers_match_the_two_tables():
    registered = {cls.__name__: n for n, cls in messages.MESSAGE_TYPES.items()}
    assert registered == RELIABLE | NEWEST_WINS
    for number, cls in messages.MESSAGE_TYPES.items():
        assert cls.TYPE == number


def test_every_message_class_is_registered_and_has_an_example():
    registered = set(messages.MESSAGE_TYPES.values())
    defined = {
        cls
        for cls in messages.Message.__subclasses__()
        if cls.__module__ == messages.__name__
    }
    assert defined == registered
    assert {type(message) for message in EXAMPLES} == registered


def test_reliable_type_numbers_are_64_to_95():
    assert [n for n in range(-1, 257) if is_reliable(n)] == list(range(64, 96))
    assert all(is_reliable(number) for number in RELIABLE.values())


def test_newest_wins_type_numbers_are_96_to_127():
    assert list(messages.NEWEST_WINS_TYPES) == list(range(96, 128))
    for number in NEWEST_WINS.values():
        assert not is_reliable(number)
        assert number in messages.NEWEST_WINS_TYPES


@pytest.mark.parametrize("message", EXAMPLES, ids=_id)
def test_round_trip(message):
    payload = message.to_payload()
    decoded = decode(message.TYPE, payload)
    assert type(decoded) is type(message)
    assert decoded == message
    assert decoded.to_payload() == payload


def test_tuples_and_nested_dataclasses_are_rebuilt():
    assert json.loads(MOVE.to_payload())["grasp_region"]["center_xy"] == [0.31, -0.05]
    move = decode(Move.TYPE, MOVE.to_payload())
    assert isinstance(move.grasp_region, GraspRegion)
    assert move.grasp_region.center_xy == (0.31, -0.05)
    assert isinstance(move.grasp_region.center_xy, tuple)
    assert move.grasp_region.z_max is None

    state = decode(RobotState.TYPE, STATE.to_payload())
    assert all(isinstance(sample, JointSample) for sample in state.joints)
    assert all(isinstance(sample, JointSample) for sample in state.recent)
    assert isinstance(state.gripper, GripperState)


def test_payload_is_compact_utf8_json():
    assert STOP.to_payload() == b'{"reason":"operator typed stop","sender":"console"}'
    assert VIEW_STATUS.to_payload() == b'{"live":true,"age_s":0.12}'
    for message in EXAMPLES:
        obj = json.loads(message.to_payload().decode("utf-8"))
        assert list(obj) == [item.name for item in dataclasses.fields(message)]


def test_non_ascii_text_round_trips():
    stop = Stop(reason="arrêt demandé 停止", sender="console")
    assert decode(Stop.TYPE, stop.to_payload()) == stop
    raw = '{"reason":"arrêt demandé 停止","sender":"console"}'.encode()
    assert decode(Stop.TYPE, raw) == stop


@pytest.mark.parametrize("msg_type", [-1, 0, 1, 32, 63, 76, 95, 99, 127, 128, 255])
def test_unknown_type_is_refused(msg_type):
    with pytest.raises(ProtocolError, match=r"^unknown link message type -?\d+$"):
        decode(msg_type, STOP.to_payload())


@pytest.mark.parametrize(
    ("payload", "problem"),
    [
        (b"", r"^Stop: bad JSON"),
        (b"stop", r"^Stop: bad JSON"),
        (b'{"reason":"x","sender":', r"^Stop: bad JSON"),
        (b'{"reason":"x","sender":"y"} trailing', r"^Stop: bad JSON"),
        (b"\xff\xfe{}", r"^Stop: bad JSON"),
        (b'{"reason":"\xc3","sender":"y"}', r"^Stop: bad JSON"),
        (b"[]", r"^Stop: expected an object, got list$"),
        (b"null", r"^Stop: expected an object, got NoneType$"),
        (b'"reason"', r"^Stop: expected an object, got str$"),
        (b"3", r"^Stop: expected an object, got int$"),
    ],
)
def test_bad_json_is_refused(payload, problem):
    with pytest.raises(ProtocolError, match=problem):
        decode(Stop.TYPE, payload)


@pytest.mark.parametrize(("message", "name"), _required_fields(), ids=_id)
def test_missing_required_field_is_refused(message, name):
    with pytest.raises(ProtocolError) as caught:
        decode(message.TYPE, _without(message, name))
    assert str(caught.value) == f"{_where(message)}: missing field {name!r}"


def test_missing_field_of_a_nested_dataclass_is_refused():
    obj = json.loads(STATE.to_payload())
    del obj["gripper"]["load"]
    with pytest.raises(ProtocolError) as caught:
        decode(RobotState.TYPE, json.dumps(obj).encode())
    assert str(caught.value) == "RobotState.gripper: missing field 'load'"

    obj = json.loads(STATE.to_payload())
    del obj["joints"][2]["t_end"]
    with pytest.raises(ProtocolError) as caught:
        decode(RobotState.TYPE, json.dumps(obj).encode())
    assert str(caught.value) == "RobotState.joints[2]: missing field 't_end'"

    region = {"center_xy": [0.1, 0.2], "radius_m": 0.05, "z_min": 0.0}
    with pytest.raises(ProtocolError) as caught:
        decode(Move.TYPE, _changed(MOVE, ("grasp_region",), region))
    assert str(caught.value) == "Move.grasp_region: missing field 'z_max'"


def test_fields_with_defaults_may_be_absent():
    result = decode(Result.TYPE, b'{"command_id":"c1","status":"accepted"}')
    assert result == Result("c1", "accepted", reason="", end_q=None, detail={})
    assert decode(Event.TYPE, _without(EVENT, "data")).data == {}
    assert decode(OwnerRequest.TYPE, _without(OWNER_REQUEST, "reason")).reason == ""
    request = decode(
        OwnerRequest.TYPE, _without(OWNER_REQUEST_REST, "timed_return", "confirmed")
    )
    assert (request.timed_return, request.confirmed) == (True, False)
    # A SHUTDOWN that does not say the path was confirmed is not confirmed.
    unconfirmed = decode(Shutdown.TYPE, _without(SHUTDOWN_CONFIRMED, "confirmed"))
    assert unconfirmed.confirmed is False

    payload = _without(WELCOME, "protocol_version", "camera_calibration_id")
    welcome = decode(Welcome.TYPE, payload)
    assert welcome.protocol_version == messages.PROTOCOL_VERSION == 1
    assert welcome.camera_calibration_id == ""


@pytest.mark.parametrize(
    ("message", "path", "value", "problem"),
    [
        (STOP, ("reason",), 5, "expected a string, got int"),
        (STOP, ("sender",), None, "expected a string, got NoneType"),
        (PERMIT, ("through_index",), 2.0, "expected an integer, got float"),
        (PERMIT, ("through_index",), "2", "expected an integer, got str"),
        (PERMIT, ("owner_number",), True, "expected an integer, got bool"),
        (PERMIT, ("stop_number",), None, "expected an integer, got NoneType"),
        (PERMIT, ("expires_mono_s",), False, "expected a number, got bool"),
        (PERMIT, ("evidence_wall_s",), "now", "expected a number, got str"),
        (PERMIT, ("evidence_wall_s",), None, "expected a number, got NoneType"),
        (VIEW_STATUS, ("live",), 1, "expected a boolean, got int"),
        (VIEW_STATUS, ("age_s",), [0.1], "expected a number, got list"),
        (OPEN, ("confirmed",), "yes", "expected a boolean, got str"),
        (SHUTDOWN_CONFIRMED, ("confirmed",), "yes", "expected a boolean, got str"),
        (SHUTDOWN, ("owner_number",), None, "expected an integer, got NoneType"),
        (RELEASE, ("owner_number",), 7.0, "expected an integer, got float"),
        (OWNER_REQUEST, ("timed_return",), None, "expected a boolean, got NoneType"),
        (OWNER_REQUEST_REST, ("confirmed",), 1, "expected a boolean, got int"),
        (WELCOME, ("protocol_version",), "1", "expected an integer, got str"),
        (WELCOME, ("joint_names", 0), 1, "expected a string, got int"),
        (WELCOME, ("joint_min",), {"j1": -1.0}, "expected a list, got dict"),
        (WELCOME, ("limit_set_versions",), ["a"], "expected an object, got list"),
        (WELCOME, ("limit_set_versions", "agent"), 3, "expected a string, got int"),
        (MOVE, ("start_q",), "0 0 0", "expected a list, got str"),
        (MOVE, ("start_q", 5), None, "expected a number, got NoneType"),
        (MOVE, ("setpoints", 1), 0.5, "expected a list, got float"),
        (MOVE, ("setpoints", 3, 2), "x", "expected a number, got str"),
        (MOVE, ("setpoints", 0, 0), True, "expected a number, got bool"),
        (MOVE, ("expires_mono_s",), None, "expected a number, got NoneType"),
        (MOVE, ("grasp_region",), [0.3, 0.1], "expected an object, got list"),
        (MOVE, ("grasp_region", "center_xy"), [0.3], "expected 2 values, got 1"),
        (MOVE, ("grasp_region", "center_xy"), [0, 0, 0], "expected 2 values, got 3"),
        (MOVE, ("grasp_region", "center_xy"), 0.3, "expected a list, got float"),
        (MOVE, ("grasp_region", "center_xy", 1), "y", "expected a number, got str"),
        (MOVE, ("grasp_region", "radius_m"), None, "expected a number, got NoneType"),
        (RESULT, ("end_q",), 0.0, "expected a list, got float"),
        (RESULT, ("detail",), [], "expected an object, got list"),
        (RESULT, ("detail",), None, "expected an object, got NoneType"),
        (EVENT, ("data",), "x", "expected an object, got str"),
        (EVENT, ("seq",), 17.5, "expected an integer, got float"),
        (STATE, ("joints", 0, "valid"), 1, "expected a boolean, got int"),
        (STATE, ("joints", 3), [3, 0.1], "expected an object, got list"),
        (STATE, ("recent", 7, "joint"), 1.0, "expected an integer, got float"),
        (STATE, ("gripper",), None, "expected an object, got NoneType"),
        (STATE, ("gripper", "load"), 0, "expected a string, got int"),
        (STATE, ("gripper", "opening"), "wide", "expected a number, got str"),
        (STATE, ("last_result",), "done", "expected an object, got str"),
        (STATE, ("health",), None, "expected an object, got NoneType"),
        (STATE, ("torque_on",), 0, "expected a boolean, got int"),
        (STATE, ("view_age_s",), "old", "expected a number, got str"),
        (STATE, ("teleop_stage",), None, "expected a string, got NoneType"),
        (STATE, ("controller_age_s",), True, "expected a number, got bool"),
    ],
    ids=_id,
)
def test_mistyped_field_is_refused(message, path, value, problem):
    with pytest.raises(ProtocolError) as caught:
        decode(message.TYPE, _changed(message, path, value))
    assert str(caught.value) == f"{_where(message, path)}: {problem}"


@pytest.mark.parametrize(
    ("msg_type", "payload"),
    [
        (ViewStatus.TYPE, b'{"live":true,"age_s":NaN}'),
        (ViewStatus.TYPE, b'{"live":true,"age_s":Infinity}'),
        (ViewStatus.TYPE, b'{"live":true,"age_s":-Infinity}'),
        (ViewStatus.TYPE, b'{"live":true,"age_s":1e999}'),
        (ViewStatus.TYPE, b'{"live":true,"age_s":-1.5E+400}'),
        (Result.TYPE, b'{"command_id":"c","status":"failed","end_q":[0.0,NaN]}'),
        (Result.TYPE, b'{"command_id":"c","status":"failed","detail":{"a":[NaN]}}'),
        (Event.TYPE, b'{"seq":1,"kind":"k","text":"t","data":{"v":{"w":1e400}}}'),
        (Stop.TYPE, b'{"reason":"r","sender":"s","not_a_field":Infinity}'),
    ],
)
def test_non_finite_number_is_refused(msg_type, payload):
    with pytest.raises(ProtocolError, match=r": non-finite number$"):
        decode(msg_type, payload)


def test_integer_too_large_for_a_float_is_refused():
    payload = b'{"live":true,"age_s":' + b"9" * 400 + b"}"
    with pytest.raises(
        ProtocolError, match=r"^ViewStatus\.age_s: number out of range$"
    ):
        decode(ViewStatus.TYPE, payload)


def test_payload_over_the_limit_is_refused():
    assert MAX_PAYLOAD_BYTES == 4 * 1024 * 1024
    payload = STOP.to_payload()
    at_limit = payload + b" " * (MAX_PAYLOAD_BYTES - len(payload))
    assert decode(Stop.TYPE, at_limit) == STOP
    with pytest.raises(ProtocolError, match=r"^Stop: \d+ bytes is over the limit$"):
        decode(Stop.TYPE, at_limit + b" ")


def test_the_longest_move_fits_in_a_payload():
    # 3000 setpoints is executor.max_setpoints in the shipped configuration.
    setpoints = [[-(1.0 + i / 3.0 + j / 7.0) for j in range(6)] for i in range(3000)]
    move = dataclasses.replace(MOVE, setpoints=setpoints)
    payload = move.to_payload()
    assert len(payload) < MAX_PAYLOAD_BYTES // 4
    assert decode(Move.TYPE, payload) == move


def test_deeply_nested_json_is_refused():
    depth = 200_000
    nested = b"[" * depth + b"]" * depth
    payload = b'{"seq":1,"kind":"k","text":"t","data":{"v":' + nested + b"}}"
    assert len(payload) < MAX_PAYLOAD_BYTES
    with pytest.raises(ProtocolError, match=r"^Event: bad JSON"):
        decode(Event.TYPE, payload)


def test_unknown_extra_fields_are_ignored():
    obj = json.loads(STATE.to_payload())
    obj["added_later"] = {"anything": [1, 2, 3]}
    obj["gripper"]["temperature"] = 31.5
    obj["joints"][0]["torque"] = 0.2
    assert decode(RobotState.TYPE, json.dumps(obj).encode()) == STATE

    payload = _changed(MOVE, ("grasp_region", "shape"), "disc")
    assert decode(Move.TYPE, payload) == MOVE


def test_integer_is_accepted_for_a_float_field():
    view = decode(ViewStatus.TYPE, b'{"live":false,"age_s":2}')
    assert view == ViewStatus(live=False, age_s=2.0)
    assert type(view.age_s) is float
    move = decode(Move.TYPE, _changed(MOVE, ("setpoints",), [[0, 1, 2, 3, 4, 5]]))
    assert move.setpoints == [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]
    assert all(type(value) is float for value in move.setpoints[0])


@pytest.mark.parametrize(
    ("message", "changes", "problem"),
    [
        (STOP, {"reason": 5}, "Stop.reason: expected a string, got int"),
        (VIEW_STATUS, {"live": 1}, "ViewStatus.live: expected a boolean, got int"),
        (VIEW_STATUS, {"age_s": True}, "ViewStatus.age_s: expected a number, got bool"),
        (VIEW_STATUS, {"age_s": NAN}, "ViewStatus.age_s: non-finite number"),
        (VIEW_STATUS, {"age_s": INF}, "ViewStatus.age_s: non-finite number"),
        (
            PERMIT,
            {"through_index": 2.0},
            "Permit.through_index: expected an integer, got float",
        ),
        (
            PERMIT,
            {"owner_number": True},
            "Permit.owner_number: expected an integer, got bool",
        ),
        (
            PERMIT,
            {"expires_mono_s": None},
            "Permit.expires_mono_s: expected a number, got NoneType",
        ),
        (
            MOVE,
            {"setpoints": [[0.0] * 6, [0.0, -INF]]},
            "Move.setpoints[1][1]: non-finite number",
        ),
        (MOVE, {"start_q": "rest"}, "Move.start_q: expected a list, got str"),
        (
            MOVE,
            {"grasp_region": {"radius_m": 0.1}},
            "Move.grasp_region: expected GraspRegion, got dict",
        ),
        (
            MOVE,
            {"grasp_region": GraspRegion((0.1,), 0.1, 0.0, None)},
            "Move.grasp_region.center_xy: expected 2 values, got 1",
        ),
        (
            STATE,
            {"gripper": None},
            "RobotState.gripper: expected GripperState, got NoneType",
        ),
        (
            STATE,
            {"joints": [JOINTS[0], (1, 0.0)]},
            "RobotState.joints[1]: expected JointSample, got tuple",
        ),
        (
            RESULT,
            {"detail": {1: "x"}},
            "Result.detail: expected a string key, got int",
        ),
        (
            RESULT,
            {"detail": {"x": object()}},
            "Result: cannot encode: object is not a JSON value",
        ),
    ],
    ids=_id,
)
def test_encoding_refuses_a_mistyped_or_non_finite_value(message, changes, problem):
    with pytest.raises(ProtocolError) as caught:
        dataclasses.replace(message, **changes).to_payload()
    assert str(caught.value) == problem


def test_encoding_refuses_a_non_finite_number_in_free_form_data():
    with pytest.raises(ProtocolError, match=r"^Result: cannot encode: "):
        Result("c", "failed", detail={"x": [NAN]}).to_payload()
    with pytest.raises(ProtocolError, match=r"^Event: cannot encode: "):
        Event(1, "k", "t", data={"x": {"y": -INF}}).to_payload()


def test_encoding_refuses_a_payload_over_the_limit():
    with pytest.raises(ProtocolError, match=r"^Event: \d+ bytes is over the limit$"):
        Event(1, "k", "x" * MAX_PAYLOAD_BYTES).to_payload()


def test_encoding_accepts_a_tuple_for_a_list():
    move = dataclasses.replace(MOVE, start_q=tuple(REST_Q))
    assert decode(Move.TYPE, move.to_payload()) == MOVE


def test_encoding_accepts_numpy_values():
    np = pytest.importorskip("numpy")
    move = dataclasses.replace(
        MOVE,
        owner_number=np.int64(3),
        start_q=np.array(REST_Q),
        start_tol_rad=np.float32(0.5),
        setpoints=np.full((3, 6), 0.25),
    )
    assert decode(Move.TYPE, move.to_payload()) == dataclasses.replace(
        MOVE, start_tol_rad=0.5, setpoints=[[0.25] * 6] * 3
    )

    sample = dataclasses.replace(JOINTS[0], value=np.float64(0.5), valid=np.bool_(True))
    state = dataclasses.replace(
        STATE, joints=[sample], health={"temperatures": np.array([30.5, 31.0])}
    )
    decoded = decode(RobotState.TYPE, state.to_payload())
    assert decoded.joints == [dataclasses.replace(JOINTS[0], value=0.5, valid=True)]
    assert decoded.health == {"temperatures": [30.5, 31.0]}

    with pytest.raises(ProtocolError, match=r"^ViewStatus\.age_s: expected a number"):
        ViewStatus(live=True, age_s=np.bool_(True)).to_payload()


def test_string_enum_members_are_sent_as_their_values():
    class Owner(enum.StrEnum):
        AGENT = "agent"

    class Level(str, enum.Enum):
        STOP = "stop"

    state = dataclasses.replace(STATE, owner=Owner.AGENT, stop_level=Level.STOP)
    decoded = decode(RobotState.TYPE, state.to_payload())
    assert (decoded.owner, decoded.stop_level) == ("agent", "stop")
    assert type(decoded.owner) is str
    assert type(decoded.stop_level) is str


def test_messages_are_frozen():
    for message in (STOP, STATE, STATE.gripper, STATE.joints[0], MOVE.grasp_region):
        name = dataclasses.fields(message)[0].name
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(message, name, getattr(message, name))


def test_default_dicts_are_not_shared():
    first, second = Result("a", "accepted"), Result("b", "accepted")
    assert first.detail == {} == second.detail
    assert first.detail is not second.detail
    assert Event(1, "k", "t").data is not Event(2, "k", "t").data


def test_protocol_error_is_a_value_error():
    assert issubclass(ProtocolError, ValueError)


# A receiver's own check, and the command id of a payload that does not decode

COMMANDS = [
    MOVE,
    GRIPPER,
    CHECKER_STATE,
    OWNER_REQUEST,
    RESET,
    RELEASE,
    OPEN,
    SHUTDOWN,
]


def test_the_commands_are_the_link_messages_to_the_thor_with_a_command_id():
    assert COMMAND_TYPES == {65, 66, 67, 69, 70, 71, 72, 73}
    with_id = {
        cls.TYPE
        for cls in messages.MESSAGE_TYPES.values()
        if "command_id" in {item.name for item in dataclasses.fields(cls)}
    }
    assert with_id - COMMAND_TYPES == {Result.TYPE}


def test_a_check_sees_the_parsed_object_before_any_value_is_converted():
    seen = []

    def check(obj: dict) -> str | None:
        seen.append(obj)
        return None

    assert decode(Move.TYPE, MOVE.to_payload(), check) == MOVE
    assert seen == [json.loads(MOVE.to_payload())]


def test_a_problem_the_check_names_is_the_error_and_nothing_is_converted(monkeypatch):
    def no_conversion(cls, obj):
        raise AssertionError("a value was converted")

    monkeypatch.setattr(messages, "_from_object", no_conversion)
    with pytest.raises(ProtocolError, match=r"^too many setpoints$") as refused:
        decode(Move.TYPE, MOVE.to_payload(), lambda obj: "too many setpoints")
    assert refused.value.command_id == (MOVE.command_id, True)


@pytest.mark.parametrize("payload", [b"[]", b"5", b"not json"])
def test_the_check_is_not_asked_about_a_payload_that_is_no_object(payload):
    def check(obj: dict) -> str | None:
        raise AssertionError("the check was asked")

    with pytest.raises(ProtocolError):
        decode(Move.TYPE, payload, check)


@pytest.mark.parametrize("message", COMMANDS, ids=_id)
def test_the_error_of_a_command_that_does_not_decode_carries_its_id(message):
    obj = json.loads(message.to_payload())
    obj["stop_number"] = "one"
    # Read from the parsed object, wherever the id stands in it.
    for ordered in (obj, dict(reversed(obj.items()))):
        with pytest.raises(ProtocolError) as refused:
            decode(message.TYPE, json.dumps(ordered).encode())
        assert refused.value.command_id == (message.command_id, True)
    # Read from the head of a payload that was not parsed.
    payload = message.to_payload()[:-1] + b',"x":NaN}'
    with pytest.raises(ProtocolError, match=r"non-finite number$") as refused:
        decode(message.TYPE, payload)
    assert refused.value.command_id == (message.command_id, True)


@pytest.mark.parametrize(
    ("msg_type", "payload"),
    [
        (Permit.TYPE, b'{"command_id":"c-1","move_id":5}'),
        (ViewStatus.TYPE, b'{"command_id":"c-1"}'),
        (Stop.TYPE, b'{"command_id":"c-1"}'),
        (Result.TYPE, b'{"command_id":"c-1"}'),
        (99, b'{"command_id":"c-1"}'),
    ],
    ids=["PERMIT", "VIEW_STATUS", "STOP", "RESULT", "an unknown type"],
)
def test_the_error_of_anything_but_a_command_carries_no_id(msg_type, payload):
    with pytest.raises(ProtocolError) as refused:
        decode(msg_type, payload)
    assert refused.value.command_id is None


@pytest.mark.parametrize(
    ("command_id", "read"),
    [
        (5, None),
        (None, None),
        (["c-1"], None),
        ("", ("", True)),
        ("x" * 256, ("x" * 256, True)),
        ("x" * 257, ("x" * 256, False)),
        ("é" * 700_000, ("é" * 256, False)),
    ],
    ids=["a number", "null", "a list", "empty", "256", "257", "megabytes"],
)
def test_the_id_of_a_parsed_command_is_given_back_whole_or_cut(command_id, read):
    obj = {"session_id": SESSION, "command_id": command_id}  # no number is quoted
    with pytest.raises(ProtocolError, match=r"^Reset") as refused:
        decode(Reset.TYPE, json.dumps(obj, ensure_ascii=False).encode())
    assert refused.value.command_id == read
    without = {"session_id": SESSION}
    with pytest.raises(ProtocolError) as refused:
        decode(Reset.TYPE, json.dumps(without).encode())
    assert refused.value.command_id is None


@pytest.mark.parametrize("message", [*COMMANDS, RESULT], ids=_id)
def test_the_id_is_read_at_the_head_of_what_to_payload_writes(message):
    assert command_id_at_head(message.to_payload()) == (message.command_id, True)


@pytest.mark.parametrize(
    "command_id",
    ["", "move 1", 'say "yes"', "back\\slash", "é ü 日本", "tab\tnew\nline", "a/b"],
)
def test_an_id_at_the_head_is_read_as_json_wrote_it(command_id):
    for ascii_only in (True, False):
        payload = json.dumps(
            {"command_id": command_id, "x": 1}, ensure_ascii=ascii_only
        ).encode()
        assert command_id_at_head(payload) == (command_id, True)
        assert command_id_at_head(b" \n\t" + payload) == (command_id, True)


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"not json",
        b"[]",
        b'{"session_id":"s","command_id":"c-1"}',
        b'{"command_id":5}',
        b'{"command_id":null}',
        b'["command_id","c-1"]',
        b'"command_id":"c-1"',
        b'{"command_id":"c-1"}'.rjust(4096),
        b"\xff\xfe" + b'{"command_id":"c-1"}',
    ],
    ids=[
        "empty",
        "no JSON",
        "a list",
        "the id is not first",
        "a number",
        "null",
        "a list of the words",
        "no brace",
        "behind 4 kB of spaces",
        "bytes that are not UTF-8 first",
    ],
)
def test_no_id_is_read_where_the_payload_does_not_begin_with_one(payload):
    assert command_id_at_head(payload) is None


@pytest.mark.parametrize(
    ("payload", "read"),
    [
        (b'{"command_id":"c-1', ("c-1", False)),
        (b'{"command_id":"c-1\\', ("c-1", False)),
        (b'{"command_id":"c-1\\x41"', ("c-1", False)),
        (b'{"command_id":"c-1\\u00e', ("c-1", False)),
        (b'{"command_id":"c-1\x01"', ("c-1", False)),
        (b'{"command_id":"' + b"x" * 257 + b'"}', ("x" * 256, False)),
        (b'{"command_id":"' + b"x" * 5_000_000, ("x" * 256, False)),
        (b'{"command_id":"' + "é".encode() * 5_000_000, ("é" * 256, False)),
        (b'{"command_id":"' + b"\\u00e9" * 5_000_000, ("é" * 256, False)),
    ],
    ids=[
        "no closing quote",
        "ends in a backslash",
        "an escape JSON does not have",
        "half an escape",
        "a control character",
        "257 characters",
        "megabytes",
        "megabytes that are not ASCII",
        "megabytes of escapes",
    ],
)
def test_an_id_that_is_long_or_broken_is_given_back_as_its_beginning(payload, read):
    assert command_id_at_head(payload) == read


def test_reading_the_id_at_the_head_parses_nothing_else(monkeypatch):
    parsed = []
    real = json.loads

    def loads(text, *args, **kwargs):
        parsed.append(len(text))
        return real(text, *args, **kwargs)

    payload = MOVE.to_payload()[:-1] + b',"more":[' + b"0," * 1_000_000 + b"0]}"
    monkeypatch.setattr(json, "loads", loads)
    assert command_id_at_head(payload) == (MOVE.command_id, True)
    assert command_id_in(payload) == (MOVE.command_id, True)
    assert parsed == [len(MOVE.command_id) + 2] * 2


@pytest.mark.parametrize(
    ("payload", "read"),
    [
        (b'{"session_id":"s","command_id":"c-1"}', ("c-1", True)),
        (b'{"x":NaN,"command_id":"c-1"}', ("c-1", True)),
        (b'{"x":[1,{"command_id":"inner"}],"command_id":"c-1"}', ("c-1", True)),
        (b'{"x":1,"command_id":"' + b"y" * 300 + b'"}', ("y" * 256, False)),
        (b'{"x":{"command_id":"inner"}}', None),
        (b'{"x":1,"command_id":5}', None),
        (b'[{"command_id":"c-1"}]', None),
        (b"not json", None),
        (b"", None),
        (b"\xff\xfe", None),
        (b"[" * 100_000, None),
    ],
    ids=[
        "the id is the last field",
        "NaN before the id",
        "another id deeper in",
        "a long id",
        "only a deeper id",
        "a number",
        "a list",
        "no JSON",
        "empty",
        "not UTF-8",
        "nested too deep",
    ],
)
def test_a_payload_that_does_not_begin_with_its_id_is_parsed_for_it(payload, read):
    assert command_id_at_head(payload) is None
    assert command_id_in(payload) == read
