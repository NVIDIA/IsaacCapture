# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A robot link service on loopback carriers, for the tests of the local API."""

from __future__ import annotations

import queue
import socket
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    import isaaccapture_examples.agentic_remote.protocol  # noqa: F401
except ImportError:
    _root = next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "VERSION").is_file() and (parent / "CMakeLists.txt").is_file()
    )
    sys.path.insert(0, str(_root / "examples" / "agentic_remote" / "python"))

from isaaccapture_examples.agentic_remote.link_service.local import (
    HEADER,
    LocalCarrier,
    connect_local,
    pack_record,
    read_exact,
)
from isaaccapture_examples.agentic_remote.link_service.service import (
    Incoming,
    RobotLinkService,
)
from isaaccapture_examples.agentic_remote.protocol.carrier import (
    LoopbackCarrier,
    loopback_pair,
    send,
)
from isaaccapture_examples.agentic_remote.protocol.local_messages import (
    LocalHello,
    decode_local,
    is_local,
)
from isaaccapture_examples.agentic_remote.protocol.messages import (
    CheckerState,
    Event,
    Gripper,
    GripperState,
    HumanSetpoint,
    Message,
    Move,
    Open,
    OwnerRequest,
    Permit,
    Release,
    Reset,
    Result,
    RobotState,
    Shutdown,
    Stop,
    ViewStatus,
    Welcome,
    decode,
)

# Real seconds a test gives another thread before it fails.
WAIT_S = 5.0
# Real seconds a test waits to see that nothing arrives.
QUIET_S = 0.05

QUOTED = {"session_id": "s1", "owner_number": 1, "stop_number": 0}


def until(done: Callable[[], Any], what: str = "the condition") -> None:
    """Wait in real time until done() holds."""
    end = time.monotonic() + WAIT_S
    while not done():
        assert time.monotonic() < end, f"{what} did not come true"
        time.sleep(0.002)


def welcome(session_id: str = "s1") -> Welcome:
    return Welcome(
        session_id=session_id,
        robot_id="sim-arm",
        profile_id="sim-6dof",
        joint_names=["j1"],
        joint_min=[-1.0],
        joint_max=[1.0],
        rest_q=[0.0],
        setpoint_period_s=0.02,
        max_move_setpoints=100,
        limit_set_versions={"agent": "sim-1"},
    )


def robot_state(
    seq: int = 1,
    owner: str = "nobody",
    owner_number: int = 1,
    stop_number: int = 0,
    session_id: str = "s1",
) -> RobotState:
    return RobotState(
        seq=seq,
        session_id=session_id,
        clock_mono_s=1000.0,
        clock_wall_s=1.7e9,
        joints=[],
        recent=[],
        target_q=None,
        target_t=None,
        gripper=GripperState(None, None, False, False, None, False, "unknown"),
        owner=owner,
        owner_number=owner_number,
        stop_number=stop_number,
        stop_level="none",
        fault_reason="",
        limit_set_version="sim-1",
        motion="rest_hold",
        move_id="",
        move_index=0,
        move_total=0,
        last_result=None,
        assistance_wait_s=None,
        torque_on=True,
        view_age_s=None,
        teleop_stage="",
        controller_age_s=None,
        health={},
    )


def move(command_id: str = "move-1", **quoted: Any) -> Move:
    return Move(
        command_id=command_id,
        **{**QUOTED, **quoted},
        start_q=[0.0],
        start_tol_rad=0.02,
        setpoints=[[0.1]],
        period_s=0.02,
        grasp_region=None,
        evidence_wall_s=None,
        expires_mono_s=1005.0,
    )


def gripper(command_id: str = "grip-1", **quoted: Any) -> Gripper:
    fields = {**QUOTED, **quoted}
    return Gripper(command_id, **fields, opening=0.5, max_effort=0.5, mode="plain")


def permit(move_id: str = "move-1", **quoted: Any) -> Permit:
    return Permit(
        move_id=move_id,
        **{**QUOTED, **quoted},
        through_index=0,
        evidence_wall_s=1.7e9,
        expires_mono_s=1002.0,
    )


def owner_request(
    request: str, command_id: str = "own-1", **quoted: Any
) -> OwnerRequest:
    return OwnerRequest(command_id, **{**QUOTED, **quoted}, request=request)


# One of each link message a client can try to send, by name.
CLIENT_MESSAGES: dict[str, Message] = {
    "MOVE": move(),
    "GRIPPER": gripper(),
    "PERMIT": permit(),
    "CHECKER_STATE": CheckerState("chk-1", **QUOTED, action="clear_contact_spots"),
    "OWNER_REQUEST agent": owner_request("agent"),
    "OWNER_REQUEST help_needed": owner_request("help_needed"),
    "OWNER_REQUEST nobody": owner_request("nobody"),
    "OWNER_REQUEST rest": owner_request("rest"),
    "OWNER_REQUEST human": owner_request("human"),
    "OWNER_REQUEST handback": owner_request("handback"),
    "HUMAN_SETPOINT": HumanSetpoint(
        **QUOTED,
        seq=1,
        source="synthetic",
        stage="ready",
        sample_age_s=0.01,
        engaged=False,
        grip_released=True,
        q=None,
        jaw_opening=None,
        state_clock_mono_s=1000.0,
    ),
    "RESET": Reset("reset-1", **QUOTED),
    "RELEASE": Release("release-1", **QUOTED),
    "OPEN": Open("open-1", **QUOTED, confirmed=True),
    "SHUTDOWN": Shutdown("shutdown-1", **QUOTED),
    "VIEW_STATUS": ViewStatus(live=True, age_s=0.1),
    "STOP": Stop(reason="pressed", sender="test"),
    "WELCOME": welcome(),
    "ROBOT_STATE": robot_state(),
    "RESULT": Result("x-1", "completed"),
    "EVENT": Event(seq=1, kind="note", text="hello"),
}


class FakeThor:
    """The robot's end of a loopback robot connection, driven by the test."""

    def __init__(self, address: str) -> None:
        self.carrier, self.service_end = loopback_pair()
        self.refused = threading.Event()
        self.incoming = Incoming(address, lambda: self.service_end, self._refuse)
        self._seq = 0

    def _refuse(self) -> None:
        self.refused.set()
        self.service_end.close()

    def send(self, message: Message) -> None:
        send(self.carrier, message)

    def state(self, **fields: Any) -> RobotState:
        """Send a ROBOT_STATE with the next sequence number; returns it."""
        self._seq += 1
        state = robot_state(seq=self._seq, **fields)
        self.send(state)
        return state

    def next(self, timeout_s: float = WAIT_S) -> tuple[int, bytes]:
        """The next link message the service forwarded, as (type number, payload)."""
        received = self.carrier.receive(timeout_s)
        assert received is not None, "the service forwarded nothing"
        return received

    def next_message(self) -> Message:
        return decode(*self.next())

    def nothing(self, wait_s: float = QUIET_S) -> None:
        """Fail if the service forwards anything within wait_s."""
        received = self.carrier.receive(wait_s)
        assert received is None, f"forwarded: {received}"

    def drain(self) -> list[tuple[int, bytes]]:
        received = []
        while (item := self.carrier.receive(0)) is not None:
            received.append(item)
        return received


class Bench:
    """A started robot link service whose robot connections the test makes."""

    def __init__(self, folder: Path, **options: Any) -> None:
        self.path = str(folder / "s.sock")
        assert len(self.path) < 100, f"socket path too long: {self.path}"
        self._incoming: queue.Queue[Incoming] = queue.Queue()
        options = {"status_period_s": 0.05, "poll_s": 0.01, **options}
        self.service = RobotLinkService(self._accept, socket_path=self.path, **options)
        self.service.start()
        self._carriers: list[LocalCarrier] = []
        self._sockets: list[socket.socket] = []
        self.thors: list[FakeThor] = []

    def _accept(self, timeout_s: float) -> Incoming | None:
        try:
            return self._incoming.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def dial(self, address: str = "127.0.0.1") -> FakeThor:
        """Offer the service a robot connection from this address."""
        thor = FakeThor(address)
        self.thors.append(thor)
        self._incoming.put(thor.incoming)
        return thor

    def robot(self, session_id: str | None = "s1", **state: Any) -> FakeThor:
        """A robot connection the service took; sends WELCOME and a ROBOT_STATE."""
        thor = self.dial()
        until(lambda: self.service._carrier is thor.service_end, "the robot connection")
        if session_id is not None:
            thor.send(welcome(session_id))
            sent = thor.state(session_id=session_id, **state)
            until(lambda: self.service.robot_state == sent, "the ROBOT_STATE")
        return thor

    def client(self, role: str, name: str = "test") -> LocalCarrier:
        carrier = connect_local(self.path, role, name, WAIT_S)
        self._carriers.append(carrier)
        return carrier

    def raw(self, role: str | None = None) -> socket.socket:
        """A plain socket on the local API; with a role it has said LOCAL_HELLO."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(WAIT_S)
        sock.connect(self.path)
        self._sockets.append(sock)
        if role is not None:
            write(sock, LocalHello(role=role, name="raw"))
            msg_type, _ = read_record(sock)
            assert msg_type == 193
        return sock

    def close(self) -> None:
        for carrier in self._carriers:
            carrier.close()
        for sock in self._sockets:
            sock.close()
        self.service.close()
        for thor in self.thors:
            thor.carrier.close()


def write(sock: socket.socket, message: Message) -> None:
    sock.sendall(pack_record(message.TYPE, message.to_payload()))


def read_record(sock: socket.socket) -> tuple[int, bytes]:
    """The next record of a plain socket; EOFError once the service closed it."""
    msg_type, length = HEADER.unpack(read_exact(sock, HEADER.size))
    return msg_type, read_exact(sock, length)


def read_message(sock: socket.socket) -> Message:
    msg_type, payload = read_record(sock)
    if is_local(msg_type):
        return decode_local(msg_type, payload)
    return decode(msg_type, payload)


def closed_by_service(sock: socket.socket) -> list[Message]:
    """Read a plain socket until the service closes it; what it sent before."""
    messages = []
    try:
        while True:
            messages.append(read_message(sock))
    except EOFError:
        return messages


def received(
    carrier: LocalCarrier | LoopbackCarrier,
    kind: type,
    never: type | None = None,
    **fields: Any,
) -> Any:
    """The next link message of this kind with these field values; others pass by.

    Fails if one of the kind `never` passes by first.
    """
    end = time.monotonic() + WAIT_S
    while True:
        left = end - time.monotonic()
        assert left > 0, f"no {kind.__name__} with {fields} arrived"
        got = carrier.receive(left)
        if got is None:
            continue
        message = decode(*got)
        if isinstance(message, kind) and all(
            getattr(message, name) == value for name, value in fields.items()
        ):
            return message
        assert never is None or not isinstance(message, never), f"got {message}"


def quiet(carrier: LocalCarrier, kind: type, wait_s: float = QUIET_S) -> None:
    """Fail if a link message of this kind arrives within wait_s."""
    end = time.monotonic() + wait_s
    while (left := end - time.monotonic()) > 0:
        got = carrier.receive(left)
        if got is not None:
            assert not isinstance(decode(*got), kind), f"got {decode(*got)}"
