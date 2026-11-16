# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Robot link messages and their JSON encoding, shared by the Thor and the workstation.

Type numbers 64 to 95 are reliable (queued, delivered in order); 96 to 127 are
newest-wins (an unsent older one of the same type is replaced). Decoding is strict
about the fields a link message declares and ignores fields it does not know, so a
newer peer can add fields without breaking an older one.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import json
import math
import re
import types
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

PROTOCOL_VERSION = 1
MAX_PAYLOAD_BYTES = 4 * 1024 * 1024
RELIABLE_TYPES = range(64, 96)
NEWEST_WINS_TYPES = range(96, 128)
# ROBOT_STATE.move_stage of a running move. Armed: it passed every check that
# needs no permit, sent nothing and waits for its first permit. Sending: a
# permit covered its first setpoint, or it needs none, and it went on.
MOVE_ARMED = "armed"
MOVE_SENDING = "sending"
# WELCOME.driver: what the robot app drives. "" means not declared, and every
# reader treats that as the real arm.
DRIVER_SIM = "sim"
DRIVER_APAI_MOCK = "apai-mock"
DRIVER_REAL = "real"
SIMULATED_DRIVERS = frozenset({DRIVER_SIM, DRIVER_APAI_MOCK})


class ProtocolError(ValueError):
    """A link message that cannot be encoded or decoded.

    When a command failed to decode, command_id is (its command id as far as it
    was read, whether that is all of it), and None when no id was found.
    """

    command_id: tuple[str, bool] | None = None


class Message:
    """Base of the link messages; each one is a frozen dataclass with a TYPE number."""

    TYPE: ClassVar[int]

    def to_payload(self) -> bytes:
        """This link message as UTF-8 JSON; ProtocolError if a value cannot be sent."""
        name = type(self).__name__
        try:
            text = json.dumps(
                _to_object(self),
                separators=(",", ":"),
                allow_nan=False,
                default=_json_default,
            )
        except _Mismatch as mismatch:
            raise _error(name, mismatch) from None
        except (TypeError, ValueError, RecursionError) as error:
            raise ProtocolError(f"{name}: cannot encode: {error}") from error
        payload = text.encode("utf-8")
        if len(payload) > MAX_PAYLOAD_BYTES:
            raise ProtocolError(f"{name}: {len(payload)} bytes is over the limit")
        return payload


@dataclass(frozen=True)
class GraspRegion:
    """Region around a grasped object where the table and obstacle rules are relaxed."""

    center_xy: tuple[float, float]
    radius_m: float
    z_min: float
    z_max: float | None


@dataclass(frozen=True)
class JointSample:
    """One measured joint value with the start and end of its read (Thor monotonic)."""

    joint: int
    value: float
    t_start: float
    t_end: float
    valid: bool


@dataclass(frozen=True)
class GripperState:
    """The jaw reading and what the robot app knows about a held object."""

    opening: float | None  # 0 closed .. 1 open
    t: float | None
    valid: bool
    referenced: bool
    last_target: float | None
    may_be_holding: bool
    load: str  # "empty", "loaded" or "unknown"


@dataclass(frozen=True)
class Welcome(Message):
    """Thor to workstation, first on a connection: session id, identity and limits."""

    TYPE: ClassVar[int] = 64

    # Keyword-only, so its default can stand ahead of the required fields.
    protocol_version: int = field(default=PROTOCOL_VERSION, kw_only=True)
    session_id: str
    robot_id: str
    profile_id: str
    joint_names: list[str]
    joint_min: list[float]
    joint_max: list[float]
    rest_q: list[float]
    setpoint_period_s: float
    max_move_setpoints: int
    limit_set_versions: dict[str, str]
    camera_calibration_id: str = ""
    driver: str = ""  # DRIVER_SIM, DRIVER_APAI_MOCK, DRIVER_REAL or "" (not declared)


@dataclass(frozen=True)
class Move(Message):
    """Agent to Thor: timed setpoints from an expected start pose."""

    TYPE: ClassVar[int] = 65

    command_id: str
    session_id: str
    owner_number: int
    stop_number: int
    start_q: list[float]
    start_tol_rad: float
    setpoints: list[list[float]]
    period_s: float
    grasp_region: GraspRegion | None
    evidence_wall_s: float | None  # camera time of the grid the move was checked on
    expires_mono_s: float  # Thor monotonic clock


@dataclass(frozen=True)
class Gripper(Message):
    """Agent to Thor: one jaw command."""

    TYPE: ClassVar[int] = 66

    command_id: str
    session_id: str
    owner_number: int
    stop_number: int
    opening: float  # 0 closed .. 1 open
    max_effort: float  # 0 .. 1
    mode: str  # "plain" or "close_until_contact"


@dataclass(frozen=True)
class CheckerState(Message):
    """Agent to Thor: a change the agent's skills made to the agent checker's state."""

    TYPE: ClassVar[int] = 67

    command_id: str
    session_id: str
    owner_number: int
    stop_number: int
    action: str  # "clear_contact_spots" or "close_grasp"


@dataclass(frozen=True)
class Stop(Message):
    """Any client to Thor: STOP. Needs no control ownership and quotes no numbers."""

    TYPE: ClassVar[int] = 68

    reason: str
    sender: str


@dataclass(frozen=True)
class OwnerRequest(Message):
    """Agent or console to Thor: change the control owner, or ask for help."""

    TYPE: ClassVar[int] = 69

    command_id: str
    session_id: str
    owner_number: int
    stop_number: int
    request: str  # "agent", "human", "rest", "nobody" or "help_needed"
    reason: str = ""
    timed_return: bool = True  # with "agent": this task may end in a timed return
    confirmed: bool = False  # with "rest": the operator typed that the path is clear


@dataclass(frozen=True)
class Reset(Message):
    """Console to Thor: clear a latched fault-stop."""

    TYPE: ClassVar[int] = 70

    command_id: str
    session_id: str
    owner_number: int
    stop_number: int


@dataclass(frozen=True)
class Release(Message):
    """Console to Thor: release torque at rest."""

    TYPE: ClassVar[int] = 71

    command_id: str
    session_id: str
    owner_number: int
    stop_number: int


@dataclass(frozen=True)
class Open(Message):
    """Console to Thor: open the jaws through the guarded opening."""

    TYPE: ClassVar[int] = 72

    command_id: str
    session_id: str
    owner_number: int
    stop_number: int
    confirmed: bool


@dataclass(frozen=True)
class Shutdown(Message):
    """Console to Thor: return to rest, release torque, then end the robot app."""

    TYPE: ClassVar[int] = 73

    command_id: str
    session_id: str
    owner_number: int
    stop_number: int
    # The operator typed that the path is clear; needed when the arm has to move.
    confirmed: bool = False


@dataclass(frozen=True)
class Result(Message):
    """Thor to workstation: what became of the command with this id."""

    TYPE: ClassVar[int] = 74

    command_id: str
    status: str  # "accepted", "refused", "completed", "cancelled" or "failed"
    reason: str = ""
    end_q: list[float] | None = None  # measured
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Event(Message):
    """Thor to workstation: text for the operator."""

    TYPE: ClassVar[int] = 75

    seq: int
    kind: str
    text: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RobotState(Message):
    """Thor to workstation: measured state, authority and health with a clock pair."""

    TYPE: ClassVar[int] = 96

    seq: int
    session_id: str
    clock_mono_s: float
    clock_wall_s: float
    joints: list[JointSample]  # newest reading per joint
    recent: list[JointSample]  # every reading of the recent window
    target_q: list[float] | None  # last target sent
    target_t: float | None
    gripper: GripperState
    owner: str
    owner_number: int
    stop_number: int
    stop_level: str  # "none", "stop" or "fault"
    fault_reason: str
    limit_set_version: str
    motion: str  # "idle", "moving", "hold", "rest_hold" or "rest_unpowered"
    move_id: str
    move_index: int
    move_total: int
    last_result: dict[str, Any] | None
    assistance_wait_s: float | None  # time left, None when no wait runs
    torque_on: bool
    view_age_s: float | None  # age of the operator's live view, None if unknown
    teleop_stage: str  # "" until human takeover is built
    controller_age_s: float | None  # age of the newest controller sample
    health: dict[str, Any]
    # How far the move named by move_id has got: "" (none runs, or its start
    # checks are not done), MOVE_ARMED or MOVE_SENDING.
    move_stage: str = ""


@dataclass(frozen=True)
class Permit(Message):
    """Agent to Thor: the running move may be sent up to and including through_index."""

    TYPE: ClassVar[int] = 97

    move_id: str
    session_id: str
    owner_number: int
    stop_number: int
    through_index: int
    evidence_wall_s: float  # camera time of the grid this part was checked on
    expires_mono_s: float  # Thor monotonic clock


@dataclass(frozen=True)
class ViewStatus(Message):
    """Workstation to Thor: whether the operator's live view is fresh."""

    TYPE: ClassVar[int] = 98

    live: bool
    age_s: float


MESSAGE_TYPES: dict[int, type[Message]] = {
    cls.TYPE: cls
    for cls in (
        Welcome,
        Move,
        Gripper,
        CheckerState,
        Stop,
        OwnerRequest,
        Reset,
        Release,
        Open,
        Shutdown,
        Result,
        Event,
        RobotState,
        Permit,
        ViewStatus,
    )
}


# The commands: each carries a command id and is answered with a RESULT.
COMMAND_TYPES = frozenset(
    cls.TYPE
    for cls in (
        Move,
        Gripper,
        CheckerState,
        OwnerRequest,
        Reset,
        Release,
        Open,
        Shutdown,
    )
)

# How much of a payload command_id_at_head() reads, and of an id it gives back.
_ID_HEAD_BYTES = 2048
_ID_HINT_CHARS = 256
_ID_AT_HEAD = re.compile(
    r'\s*\{\s*"command_id"\s*:\s*"'
    r'((?:[^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*)("?)'
)

# Sees a parsed JSON object before its values are converted; returns a problem.
Check = Callable[[dict[str, Any]], str | None]


def is_reliable(msg_type: int) -> bool:
    """True for a type number delivered in order; the others are newest-wins."""
    return msg_type in RELIABLE_TYPES


def decode(msg_type: int, payload: bytes, check: Check | None = None) -> Message:
    """The link message in `payload`; ProtocolError if it is not a valid one.

    check is a receiver's own limit on what it converts: it sees the parsed JSON
    object first and returns a problem, which becomes the error's text, or None.
    """
    cls = MESSAGE_TYPES.get(msg_type)
    if cls is None:
        raise ProtocolError(f"unknown link message type {msg_type!r}")
    parsed: list[dict[str, Any]] = []
    try:
        return _decoded(cls, payload, check, parsed)
    except ProtocolError as error:
        if msg_type in COMMAND_TYPES:
            error.command_id = _command_id(parsed, payload)
        raise


def command_id_at_head(payload: bytes) -> tuple[str, bool] | None:
    """The command id a payload begins with, as to_payload() writes it.

    Returns (the id, or the beginning of a long one; whether that is all of
    it), or None when the payload does not begin with one. Only the head of
    the payload is read, and nothing of it is checked.
    """
    head = str(payload[:_ID_HEAD_BYTES], "utf-8", "replace")
    found = _ID_AT_HEAD.match(head)
    if found is None:
        return None
    try:
        text = json.loads(f'"{found.group(1)}"')
    except ValueError:
        return None
    return text[:_ID_HINT_CHARS], bool(found.group(2)) and len(text) <= _ID_HINT_CHARS


def command_id_in(payload: bytes) -> tuple[str, bool] | None:
    """The command id of a payload that is not decoded, as command_id_at_head().

    Where the payload does not begin with the id, the payload is parsed and
    the id read from the JSON object: for a payload small enough to parse.
    """
    at_head = command_id_at_head(payload)
    if at_head is not None:
        return at_head
    try:
        obj = json.loads(payload)
    except (ValueError, RecursionError):
        return None
    return _command_id([obj], payload) if isinstance(obj, dict) else None


def _decoded(
    cls: type[Message], payload: bytes, check: Check | None, parsed: list
) -> Message:
    """decode() for a known type; the JSON object is put into `parsed`."""
    name = cls.__name__
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ProtocolError(f"{name}: {len(payload)} bytes is over the limit")
    try:
        obj = json.loads(
            str(payload, "utf-8"), parse_float=_finite, parse_constant=_finite
        )
        if not isinstance(obj, dict):
            raise _expected("an object", obj)
        parsed.append(obj)
        problem = None if check is None else check(obj)
        if problem is None:
            return _from_object(cls, obj)
    except _Mismatch as mismatch:
        raise _error(name, mismatch) from None
    except (ValueError, RecursionError) as error:
        raise ProtocolError(f"{name}: bad JSON: {error}") from error
    raise ProtocolError(problem)  # in the receiver's own words


def _command_id(parsed: list, payload: bytes) -> tuple[str, bool] | None:
    """The command id of a payload that failed to decode, read without checks."""
    if not parsed:
        return command_id_at_head(payload)
    value = parsed[0].get("command_id")
    if not isinstance(value, str):
        return None
    return value[:_ID_HINT_CHARS], len(value) <= _ID_HINT_CHARS


# A converter checks one value against its field type: (value, to_wire) -> value.
# to_wire is True when encoding (dataclass to JSON object), False when decoding.
_Convert = Callable[[Any, bool], Any]


class _Mismatch(Exception):
    """A value that does not fit its field; `path` is filled from the value outwards."""

    def __init__(self, problem: str) -> None:
        super().__init__(problem)
        self.problem = problem
        self.path: list[str] = []


def _error(name: str, mismatch: _Mismatch) -> ProtocolError:
    where = name + "".join(reversed(mismatch.path))
    return ProtocolError(f"{where}: {mismatch.problem}")


def _expected(what: str, value: Any) -> _Mismatch:
    return _Mismatch(f"expected {what}, got {type(value).__name__}")


def _plain(value: Any) -> Any:
    """A numpy array or scalar as plain Python values; anything else unchanged."""
    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else value


def _json_default(value: Any) -> Any:
    plain = _plain(value)
    if plain is value:
        raise TypeError(f"{type(value).__name__} is not a JSON value")
    return plain


def _finite(text: str) -> float:
    """JSON number parser that refuses NaN, the infinities and overflowing literals."""
    number = float(text)
    if not math.isfinite(number):
        raise _Mismatch("non-finite number")
    return number


def _bool(value: Any, to_wire: bool) -> bool:
    if to_wire:
        value = _plain(value)
    if not isinstance(value, bool):
        raise _expected("a boolean", value)
    return value


def _int(value: Any, to_wire: bool) -> int:
    if to_wire:
        value = _plain(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _expected("an integer", value)
    return value


def _float(value: Any, to_wire: bool) -> float:
    if to_wire:
        value = _plain(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _expected("a number", value)
    try:
        number = float(value)
    except OverflowError:
        raise _Mismatch("number out of range") from None
    if not math.isfinite(number):
        raise _Mismatch("non-finite number")
    return number


def _str(value: Any, to_wire: bool) -> str:
    if not isinstance(value, str):
        raise _expected("a string", value)
    return value


def _any(value: Any, to_wire: bool) -> Any:
    # Free-form JSON: the JSON encoder or parser has the last word on it.
    return value


_SCALARS: dict[Any, _Convert] = {
    bool: _bool,
    int: _int,
    float: _float,
    str: _str,
    Any: _any,
}


def _sequence(of: _Convert | tuple[_Convert, ...]) -> _Convert:
    """Converter for a list of `of`, or for a tuple with one converter per place."""
    fixed = isinstance(of, tuple)

    def convert(value: Any, to_wire: bool) -> Any:
        if to_wire:
            value = _plain(value)
        if not isinstance(value, (list, tuple)):
            raise _expected("a list", value)
        if fixed and len(value) != len(of):
            raise _Mismatch(f"expected {len(of)} values, got {len(value)}")
        out = []
        places = zip(of if fixed else itertools.repeat(of), value)
        for index, (item, element) in enumerate(places):
            try:
                out.append(item(element, to_wire))
            except _Mismatch as mismatch:
                mismatch.path.append(f"[{index}]")
                raise
        # JSON has no tuple: one travels as a list and is rebuilt on decoding.
        return tuple(out) if fixed and not to_wire else out

    return convert


def _mapping(item: _Convert) -> _Convert:
    def convert(value: Any, to_wire: bool) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise _expected("an object", value)
        out = {}
        for key, element in value.items():
            if not isinstance(key, str):
                raise _expected("a string key", key)
            try:
                out[key] = item(element, to_wire)
            except _Mismatch as mismatch:
                mismatch.path.append(f".{key}")
                raise
        return out

    return convert


def _nested(cls: type) -> _Convert:
    def convert(value: Any, to_wire: bool) -> Any:
        if to_wire:
            if not isinstance(value, cls):
                raise _expected(cls.__name__, value)
            return _to_object(value)
        if not isinstance(value, dict):
            raise _expected("an object", value)
        return _from_object(cls, value)

    return convert


def _converter(hint: Any) -> _Convert:
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin is types.UnionType:  # only "X | None" is used
        inner = _converter(next(arg for arg in args if arg is not types.NoneType))
        return lambda value, to_wire: None if value is None else inner(value, to_wire)
    if origin is list:
        return _sequence(_converter(args[0]))
    if origin is tuple:
        return _sequence(tuple(_converter(arg) for arg in args))
    if origin is dict:
        return _mapping(_converter(args[1]))
    if dataclasses.is_dataclass(hint):
        return _nested(hint)
    return _SCALARS[hint]


@functools.cache
def _schema(cls: type) -> tuple[tuple[str, _Convert, bool], ...]:
    """(name, converter, required) for each field of a dataclass."""
    hints = typing.get_type_hints(cls)
    return tuple(
        (
            item.name,
            _converter(hints[item.name]),
            item.default is dataclasses.MISSING
            and item.default_factory is dataclasses.MISSING,
        )
        for item in dataclasses.fields(cls)
    )


def _to_object(value: Any) -> dict[str, Any]:
    out = {}
    for name, convert, _ in _schema(type(value)):
        try:
            out[name] = convert(getattr(value, name), True)
        except _Mismatch as mismatch:
            mismatch.path.append(f".{name}")
            raise
    return out


def _from_object(cls: type, obj: dict[str, Any]) -> Any:
    values = {}
    for name, convert, required in _schema(cls):
        if name in obj:
            try:
                values[name] = convert(obj[name], False)
            except _Mismatch as mismatch:
                mismatch.path.append(f".{name}")
                raise
        elif required:
            raise _Mismatch(f"missing field {name!r}")
    return cls(**values)
