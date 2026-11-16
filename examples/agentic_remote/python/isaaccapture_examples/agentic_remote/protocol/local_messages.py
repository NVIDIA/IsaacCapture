# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The local API's own messages, between the robot link service and its clients.

They use type numbers 192 to 199 and never travel on the robot connection.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

from .messages import PROTOCOL_VERSION, Message, ProtocolError

LOCAL_TYPES = range(192, 200)
MAX_LOCAL_PAYLOAD_BYTES = 16 * 1024

ROLE_AGENT = "agent"
ROLE_CONSOLE = "console"
ROLE_VIEWER = "viewer"
ROLE_TOOL = "tool"
# The human program: the workstation program that turns a person's controller
# data into joint targets for the arm.
ROLE_HUMAN = "human"
ROLES = (ROLE_AGENT, ROLE_CONSOLE, ROLE_VIEWER, ROLE_TOOL, ROLE_HUMAN)


@dataclass(frozen=True)
class LocalHello(Message):
    """Client to service, first on a local connection: who the client is."""

    TYPE: ClassVar[int] = 192

    role: str  # one of ROLES
    name: str
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class LocalStatus(Message):
    """Service to client, newest wins: the robot connection and who is attached."""

    TYPE: ClassVar[int] = 193

    robot_connected: bool
    close_reason: str  # why the last robot connection ended; "" while one is up
    session_id: str  # of the newest WELCOME; "" before the first one
    # The Thor's monotonic clock minus this machine's, None while not known.
    clock_offset_ns: int | None
    clock_two_way: bool
    roles: dict[str, int] = field(default_factory=dict)  # role -> clients attached
    # A STOP was forwarded and no ROBOT_STATE has shown a higher stop number yet.
    stop_unconfirmed: bool = False


@dataclass(frozen=True)
class LocalNotice(Message):
    """Service to client: a refusal of something that has no command id."""

    TYPE: ClassVar[int] = 194

    text: str
    about_type: int | None = None  # type number of what was refused, if any


LOCAL_MESSAGE_TYPES: dict[int, type[Message]] = {
    cls.TYPE: cls for cls in (LocalHello, LocalStatus, LocalNotice)
}


def is_local(msg_type: int) -> bool:
    """True for a type number that belongs to the local API alone."""
    return msg_type in LOCAL_TYPES


def decode_local(msg_type: int, payload: bytes) -> Message:
    """The local API message in `payload`; ProtocolError if it is not a valid one.

    Fields the message does not declare are ignored.
    """
    cls = LOCAL_MESSAGE_TYPES.get(msg_type)
    if cls is None:
        raise ProtocolError(f"unknown local API message type {msg_type!r}")
    name = cls.__name__
    if len(payload) > MAX_LOCAL_PAYLOAD_BYTES:
        raise ProtocolError(f"{name}: {len(payload)} bytes is over the limit")
    try:
        obj = json.loads(str(payload, "utf-8"))
    except (ValueError, RecursionError) as error:
        raise ProtocolError(f"{name}: bad JSON: {error}") from error
    if not isinstance(obj, dict):
        raise ProtocolError(f"{name}: expected an object")
    values = {}
    for key, (check, required) in _FIELDS[cls].items():
        if key not in obj:
            if required:
                raise ProtocolError(f"{name}: missing field {key!r}")
            continue
        if not check(obj[key]):
            raise ProtocolError(f"{name}.{key}: wrong kind of value")
        values[key] = obj[key]
    return cls(**values)


def _is_str(value: Any) -> bool:
    return isinstance(value, str)


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_int_or_none(value: Any) -> bool:
    return value is None or _is_int(value)


def _is_counts(value: Any) -> bool:
    return isinstance(value, dict) and all(_is_int(n) for n in value.values())


# Field name -> (check of the JSON value, whether the field must be there).
_FIELDS: dict[type, dict[str, tuple[Callable[[Any], bool], bool]]] = {
    LocalHello: {
        "role": (_is_str, True),
        "name": (_is_str, True),
        "protocol_version": (_is_int, False),
    },
    LocalStatus: {
        "robot_connected": (_is_bool, True),
        "close_reason": (_is_str, True),
        "session_id": (_is_str, True),
        "clock_offset_ns": (_is_int_or_none, True),
        "clock_two_way": (_is_bool, True),
        "roles": (_is_counts, False),
        "stop_unconfirmed": (_is_bool, False),
    },
    LocalNotice: {
        "text": (_is_str, True),
        "about_type": (_is_int_or_none, False),
    },
}
