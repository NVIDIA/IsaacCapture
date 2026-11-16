# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The robot link service on a loopback robot connection: routing, roles, STOP."""

from __future__ import annotations

import dataclasses
import random
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from link_bench import (
    CLIENT_MESSAGES,
    QUIET_S,
    WAIT_S,
    Bench,
    closed_by_service,
    gripper,
    move,
    owner_request,
    permit,
    quiet,
    read_message,
    received,
    robot_state,
    until,
    welcome,
    write,
)

import isaaccapture_examples.agentic_remote.link_service.__main__ as service_main
import isaaccapture_examples.agentic_remote.link_service.service as service_module
from isaaccapture_examples.agentic_remote.link_service.config import (
    ConfigError,
    LinkServiceConfig,
    link_service_config,
    load_link_service_config,
)
from isaaccapture_examples.agentic_remote.link_service.local import (
    HEADER,
    LocalCarrier,
    LocalRefused,
    connect_local,
    pack_record,
)
from isaaccapture_examples.agentic_remote.link_service.service import (
    AGENT_GONE,
    BAD_STOP,
    LOST,
    OWNER_REQUESTS,
    ROLE_TABLE,
    STOP_UNDELIVERED,
    Incoming,
    RobotLinkService,
    incoming_carrier,
    link_message_name,
    peer_host,
)
from isaaccapture_examples.agentic_remote.protocol.carrier import loopback_pair, send
from isaaccapture_examples.agentic_remote.protocol.local_messages import (
    ROLES,
    LocalHello,
    LocalNotice,
    LocalStatus,
)
from isaaccapture_examples.agentic_remote.protocol.messages import (
    COMMAND_TYPES,
    MESSAGE_TYPES,
    PROTOCOL_VERSION,
    CheckerState,
    Event,
    HumanSetpoint,
    Message,
    Move,
    OwnerRequest,
    Permit,
    ProtocolError,
    Reset,
    Result,
    RobotState,
    Stop,
    ViewStatus,
    Welcome,
    decode,
)
from isaaccapture_examples.agentic_remote.testing.client import WorkstationClient

LOGGER = "isaaccapture.examples.agentic_remote.link_service"
MARKER = Stop(reason="marker", sender="test")

# What each role may send, by the names of link_bench.CLIENT_MESSAGES.
ALLOWED = {
    "agent": {
        "MOVE",
        "GRIPPER",
        "PERMIT",
        "CHECKER_STATE",
        "OWNER_REQUEST agent",
        "OWNER_REQUEST help_needed",
        "OWNER_REQUEST nobody",
        "STOP",
    },
    "console": {
        "STOP",
        "OWNER_REQUEST rest",
        "OWNER_REQUEST nobody",
        "RESET",
        "RELEASE",
        "OPEN",
        "SHUTDOWN",
    },
    "viewer": {"VIEW_STATUS", "STOP"},
    "tool": {"STOP"},
    # The human program: it turns a person's controller data into joint targets.
    "human": {
        "HUMAN_SETPOINT",
        "OWNER_REQUEST human",
        "OWNER_REQUEST handback",
        "STOP",
    },
}


def human_setpoint(seq: int = 1, **fields) -> HumanSetpoint:
    """The HUMAN_SETPOINT of CLIENT_MESSAGES with these field values instead."""
    return dataclasses.replace(CLIENT_MESSAGES["HUMAN_SETPOINT"], seq=seq, **fields)


def refusal_text(client: LocalCarrier, message: Message) -> str:
    """What the service tells the sender of a link message it does not forward."""
    if message.TYPE in COMMAND_TYPES:
        refusal = received(client, Result, command_id=message.command_id)
        assert refusal.status == "refused"
        return refusal.reason
    return client.wait_notice(lambda notice: notice.about_type == message.TYPE).text


@pytest.fixture
def folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A short private folder: a socket path holds 107 bytes at most."""
    return tmp_path_factory.mktemp("ls")


@pytest.fixture
def bench(folder: Path) -> Iterator[Bench]:
    made = Bench(folder)
    yield made
    made.close()


class Wrapped:
    """A carrier around another one, with what a real link adds or a test needs."""

    def __init__(self, inner, close_reason="", offset=None, hold=None):
        self._inner = inner
        self.close_reason = close_reason
        self._offset = offset
        self._hold = hold  # type number whose send waits for self.release
        self.holding = threading.Event()
        self.release = threading.Event()
        # Every send asked of it, in order: ("reliable" or "latest", type number).
        self.calls: list[tuple[str, int]] = []

    def clock_offset(self):
        return self._offset

    def send_reliable(self, msg_type, payload):
        self.calls.append(("reliable", msg_type))
        if msg_type == self._hold:
            self.holding.set()
            self.release.wait(WAIT_S)
        self._inner.send_reliable(msg_type, payload)

    def send_latest(self, msg_type, payload):
        self.calls.append(("latest", msg_type))
        self._inner.send_latest(msg_type, payload)

    def receive(self, timeout_s):
        return self._inner.receive(timeout_s)

    @property
    def alive(self):
        return self._inner.alive

    def close(self):
        self._inner.close()


def _wrapped_robot(bench: Bench, **options):
    """A robot connection through a Wrapped carrier: (robot end, the wrapper)."""
    robot_end, service_end = loopback_pair()
    wrapped = Wrapped(service_end, **options)
    bench._incoming.put(incoming_carrier("127.0.0.1", wrapped))
    until(lambda: bench.service._carrier is wrapped, "the robot connection")
    return robot_end, wrapped


# Names and addresses


def test_link_message_names_are_those_of_the_design():
    assert link_message_name(65) == "MOVE"
    assert link_message_name(69) == "OWNER_REQUEST"
    assert link_message_name(96) == "ROBOT_STATE"
    assert link_message_name(98) == "VIEW_STATUS"
    assert link_message_name(99) == "HUMAN_SETPOINT"
    assert link_message_name(192) == "LOCAL_HELLO"
    assert link_message_name(7) == "type 7"


@pytest.mark.parametrize(
    ("address", "host"),
    [
        ("127.0.0.1", "127.0.0.1"),
        ("127.0.0.1:5000", "127.0.0.1"),
        (" 10.0.0.2:47610 ", "10.0.0.2"),
        ("[::1]:5000", "::1"),
        ("::1", "::1"),
        ("thor.local:1", "thor.local"),
    ],
)
def test_peer_host_drops_the_port(address, host):
    assert peer_host(address) == host


@pytest.mark.parametrize(
    ("address", "host"),
    [
        ("0:0:0:0:0:0:0:1", "::1"),
        ("[0:0:0:0:0:0:0:1]:5000", "::1"),
        # A link-local peer comes with the zone of the interface it dialed in on.
        ("[FE80::1%eth0]:5000", "fe80::1%eth0"),
        ("::ffff:192.168.7.20", "192.168.7.20"),
        ("[::FFFF:192.168.7.20]:5000", "192.168.7.20"),
        ("[::ffff:c0a8:714]:5000", "192.168.7.20"),
        ("THOR.local:1", "THOR.local"),  # no IP address: left as it is
    ],
)
def test_peer_host_gives_an_ip_address_in_one_spelling(address, host):
    assert peer_host(address) == host


def test_the_role_table_is_the_contracts():
    for role in ROLES:
        for name, message in CLIENT_MESSAGES.items():
            allowed = message.TYPE in ROLE_TABLE[role]
            if isinstance(message, OwnerRequest):
                allowed = allowed and message.request in OWNER_REQUESTS[role]
            assert allowed == (name in ALLOWED[role]), (role, name)


# Thor to clients


def test_welcome_state_and_event_go_to_every_client(bench):
    thor = bench.robot()
    clients = [bench.client(role) for role in ROLES]
    for client in clients:
        assert received(client, Welcome) == welcome()
    state = thor.state(owner="agent", owner_number=3)
    event = Event(seq=5, kind="note", text="for everyone")
    thor.send(event)
    for client in clients:
        assert received(client, RobotState, owner_number=3) == state
        assert received(client, Event) == event


def test_a_welcome_from_the_thor_reaches_attached_clients(bench):
    viewer = bench.client("viewer")
    assert viewer.status.robot_connected is False
    bench.robot("s2")
    assert received(viewer, Welcome).session_id == "s2"
    viewer.wait_status(lambda s: s.robot_connected and s.session_id == "s2")
    assert bench.service.welcome == welcome("s2")


def test_a_result_goes_to_its_sender_and_to_every_console(bench):
    thor = bench.robot()
    agent = bench.client("agent")
    console_a = bench.client("console")
    console_b = bench.client("console")
    viewer = bench.client("viewer")
    send(agent, move("move-7"))
    assert thor.next_message() == move("move-7")
    accepted = Result("move-7", "accepted")
    done = Result("move-7", "completed", end_q=[0.1])
    thor.send(accepted)
    thor.send(done)
    for client in (agent, console_a, console_b):
        assert received(client, Result) == accepted
        assert received(client, Result) == done
    thor.send(Event(seq=1, kind="note", text="after"))
    assert received(viewer, Event, never=Result).text == "after"


def test_a_consoles_own_result_arrives_once(bench):
    thor = bench.robot()
    console = bench.client("console")
    send(console, Reset("reset-3", session_id="s1", owner_number=1, stop_number=0))
    thor.next()
    thor.send(Result("reset-3", "refused", "no fault-stop"))
    assert received(console, Result).reason == "no fault-stop"
    quiet(console, Result)


def test_a_result_nobody_here_sent_goes_to_the_consoles_only(bench):
    thor = bench.robot()
    agent = bench.client("agent")
    console = bench.client("console")
    thor.send(Result("thor-terminal-1", "completed"))
    assert received(console, Result).command_id == "thor-terminal-1"
    quiet(agent, Result)


def test_what_the_thor_may_not_send_is_dropped(bench, caplog):
    caplog.set_level("WARNING", logger=LOGGER)
    thor = bench.robot()
    console = bench.client("console")
    thor.carrier.send_reliable(Move.TYPE, move().to_payload())
    thor.carrier.send_reliable(Result.TYPE, b"not json\n\x1b[31m")
    thor.carrier.send_reliable(200, b"{}")
    thor.send(Event(seq=1, kind="note", text="fine"))
    assert received(console, Event).text == "fine"
    assert console.receive(QUIET_S) is None
    lines = [r.getMessage() for r in caplog.records]
    assert any("dropped a MOVE from the Thor" in line for line in lines)
    assert any("dropped a RESULT from the Thor" in line for line in lines)
    assert not any("\n" in line or "\x1b" in line for line in lines)


def test_the_workstation_client_runs_on_a_local_carrier(bench):
    thor = bench.robot(owner_number=6, stop_number=2)
    client = WorkstationClient(bench.client("agent"), time, name="apai")
    try:
        assert client.wait_welcome(WAIT_S) == welcome()
        command_id = client.request_owner("agent")
        request = thor.next_message()
        assert request.command_id == command_id
        assert (request.owner_number, request.stop_number) == (6, 2)
        thor.send(Result(command_id, "completed", detail={"owner_number": 7}))
        assert client.wait_result(command_id, timeout_s=WAIT_S).status == "completed"
        thor.send(Event(seq=1, kind="note", text="hello"))
        assert client.wait_event("note", timeout_s=WAIT_S).text == "hello"
        assert client.unexpected == []
    finally:
        client.close()


# Roles


@pytest.mark.parametrize("name", sorted(CLIENT_MESSAGES))
@pytest.mark.parametrize("role", ROLES)
def test_a_role_sends_what_it_may_and_nothing_else(bench, role, name):
    thor = bench.robot()
    client = bench.client(role)
    message = CLIENT_MESSAGES[name]
    payload = message.to_payload()
    send(client, message)
    if name in ALLOWED[role]:
        # Forwarded as it is: the service does not rewrite it.
        assert thor.next() == (message.TYPE, payload)
        return
    kind = name.split()[0]
    if message.TYPE in COMMAND_TYPES:
        refusal = received(client, Result, command_id=message.command_id)
        assert refusal.status == "refused"
        text = refusal.reason
    else:
        notice = client.wait_notice()
        assert notice.about_type == message.TYPE
        text = notice.text
    assert f"the {role} role may not send {kind}" in text
    if kind == "OWNER_REQUEST" and message.TYPE in ROLE_TABLE[role]:
        assert f'"{message.request}"' in text
    send(client, MARKER)
    assert thor.next_message() == MARKER  # nothing went out ahead of it
    assert client.alive


def test_the_human_role_may_send_its_three_link_messages_and_nothing_else(bench):
    mine = {99, 69, 68}  # HUMAN_SETPOINT, OWNER_REQUEST and STOP
    assert ROLE_TABLE["human"] == mine
    assert {message.TYPE for message in CLIENT_MESSAGES.values()} == set(MESSAGE_TYPES)
    thor = bench.robot()
    human = bench.client("human")
    for name, message in sorted(CLIENT_MESSAGES.items()):
        if message.TYPE not in mine:
            send(human, message)
            assert refusal_text(human, message) == f"the human role may not send {name}"
    # The three go out as they are, and nothing went out ahead of them.
    for name in ("HUMAN_SETPOINT", "OWNER_REQUEST human", "STOP"):
        message = CLIENT_MESSAGES[name]
        send(human, message)
        assert thor.next() == (message.TYPE, message.to_payload())
    assert human.alive


def test_the_human_role_may_ask_for_a_takeover_and_a_handback_only(bench):
    # A takeover makes a person the control owner of the arm; the hand-back
    # ends the takeover with the arm at rest.
    assert OWNER_REQUESTS["human"] == {"human", "handback"}
    thor = bench.robot()
    human = bench.client("human")
    for word in ("agent", "help_needed", "nobody", "rest", "recovery"):
        send(human, owner_request(word, command_id=f"own-{word}"))
        refusal = received(human, Result, command_id=f"own-{word}")
        assert refusal.status == "refused"
        assert refusal.reason == f'the human role may not send OWNER_REQUEST "{word}"'
    # The takeover and the hand-back go out as they are, ahead of anything else.
    takeover = owner_request("human", "own-b", reason="right B held 1.0 s")
    handback = owner_request("handback", "own-a", reason="right A held 1.0 s")
    for request in (takeover, handback):
        send(human, request)
        assert thor.next() == (OwnerRequest.TYPE, request.to_payload())
    # The RESULT of its request comes back to the human program.
    thor.send(Result("own-b", "refused", "no fresh controller data: stage stale"))
    answer = received(human, Result, command_id="own-b")
    assert answer.reason == "no fresh controller data: stage stale"


def test_a_second_human_program_is_refused(bench):
    thor = bench.robot()
    first = bench.client("human", "one")
    with pytest.raises(
        LocalRefused, match=r"a human program is attached already; one at a time"
    ):
        connect_local(bench.path, "human", "two", WAIT_S)
    assert first.alive
    assert bench.service.roles() == {"human": 1}
    # The first one is still served, and the other roles attach beside it.
    send(first, human_setpoint(seq=7))
    assert thor.next_message() == human_setpoint(seq=7)
    bench.client("agent")
    bench.client("console")
    assert bench.service.roles() == {"human": 1, "agent": 1, "console": 1}
    first.close()
    until(lambda: "human" not in bench.service.roles())
    third = bench.client("human", "three")
    send(third, human_setpoint(seq=8))
    assert thor.next_message() == human_setpoint(seq=8)


def test_the_console_may_not_ask_for_a_takeover_unless_the_setting_allows_it(folder):
    # The typed fallback: with typed_takeover the console role may ask for a
    # takeover and a hand-back too.
    # Off unless the setting is True itself: a word such as "false" is not.
    for options in ({}, {"typed_takeover": "false"}, {"typed_takeover": True}):
        bench = Bench(folder, **options)
        try:
            thor = bench.robot()
            console = bench.client("console")
            agent = bench.client("agent")
            for word in ("human", "handback"):
                request = owner_request(
                    word, f"typed-{word}", reason="typed at the operator console"
                )
                send(console, request)
                if options.get("typed_takeover") is True:
                    assert thor.next() == (OwnerRequest.TYPE, request.to_payload())
                    continue
                refusal = received(console, Result, command_id=request.command_id)
                assert refusal.status == "refused"
                assert refusal.reason == (
                    f'the console role may not send OWNER_REQUEST "{word}"'
                )
            # The setting is the console's alone, and it adds these two only.
            for client, role, word in (
                (agent, "agent", "human"),
                (agent, "agent", "handback"),
                (console, "console", "agent"),
            ):
                send(client, owner_request(word, f"{role}-{word}"))
                refusal = received(client, Result, command_id=f"{role}-{word}")
                assert refusal.reason == (
                    f'the {role} role may not send OWNER_REQUEST "{word}"'
                )
            send(console, MARKER)
            assert thor.next_message() == MARKER  # nothing else went out
        finally:
            bench.close()


def test_the_setting_of_one_service_does_not_change_the_module_tables(
    tmp_path_factory,
):
    tables = (dict(ROLE_TABLE), dict(OWNER_REQUESTS))
    with_setting = Bench(tmp_path_factory.mktemp("ls"), typed_takeover=True)
    without = Bench(tmp_path_factory.mktemp("ls"))  # made second, runs beside it
    try:
        takeover = owner_request("human", "typed-1")
        allowed = with_setting.robot()
        send(with_setting.client("console"), takeover)
        assert allowed.next_message() == takeover
        refused = without.robot()
        console = without.client("console")
        send(console, takeover)
        refusal = received(console, Result, command_id="typed-1")
        assert refusal.reason == 'the console role may not send OWNER_REQUEST "human"'
        send(console, MARKER)
        assert refused.next_message() == MARKER
    finally:
        with_setting.close()
        without.close()
    assert (ROLE_TABLE, OWNER_REQUESTS) == tables
    assert OWNER_REQUESTS["console"] == {"rest", "nobody"}


def test_the_typed_fallback_is_switched_on_in_the_services_file(folder):
    loaded = link_service_config({"operator": {"typed_takeover": True}})
    options = loaded.service_options()
    del options["socket_path"]  # the bench names its own
    bench = Bench(folder, **options)
    try:
        thor = bench.robot()
        handback = owner_request("handback", "typed-2")
        send(bench.client("console"), handback)
        assert thor.next_message() == handback
    finally:
        bench.close()


def test_operator_commands_keep_their_owner_number(bench):
    thor = bench.robot()
    console = bench.client("console")
    reset = Reset("reset-9", session_id="s1", owner_number=41, stop_number=17)
    send(console, reset)
    assert decode(*thor.next()).owner_number == 41


def test_a_command_that_cannot_be_read_is_refused_and_not_forwarded(bench):
    thor = bench.robot()
    sock = bench.raw("agent")
    sock.sendall(pack_record(65, b'{"command_id":"move-9","setpoints":"all"}'))
    refusal = read_message(sock)
    while not isinstance(refusal, Result):
        refusal = read_message(sock)
    assert refusal.command_id == "move-9"
    assert refusal.status == "refused"
    assert "the MOVE is not valid" in refusal.reason
    sock.sendall(pack_record(97, b"\x00\xff"))  # a PERMIT without an id
    notice = read_message(sock)
    while not isinstance(notice, LocalNotice):
        notice = read_message(sock)
    assert "the PERMIT is not valid" in notice.text
    write(sock, MARKER)
    assert thor.next_message() == MARKER
    assert bench.service.roles() == {"agent": 1}


@pytest.mark.parametrize("command_id", ["", "x" * 129, "bad\nid"])
def test_a_command_id_that_cannot_be_kept_is_refused(bench, command_id):
    thor = bench.robot()
    agent = bench.client("agent")
    send(agent, owner_request("agent", command_id=command_id))
    assert "no usable command id" in agent.wait_notice().text
    send(agent, MARKER)
    assert thor.next_message() == MARKER


def test_too_many_commands_without_a_result_are_refused(bench, monkeypatch):
    monkeypatch.setattr(service_module, "COMMANDS_KEPT", 2)
    thor = bench.robot()
    agent = bench.client("agent")
    for number in (1, 2, 3):
        send(agent, gripper(f"grip-{number}"))
    refusal = received(agent, Result, command_id="grip-3")
    assert refusal.status == "refused"
    assert "too many commands" in refusal.reason
    assert [decode(*item).command_id for item in (thor.next(), thor.next())] == [
        "grip-1",
        "grip-2",
    ]
    thor.send(Result("grip-1", "completed"))
    received(agent, Result, command_id="grip-1")
    send(agent, gripper("grip-4"))
    assert thor.next_message().command_id == "grip-4"


# Reliable and newest-wins
#
# The robot connection takes a link message in one of two ways: send_reliable
# queues every one in order; send_latest keeps only the newest unsent one of
# its type (newest-wins).


def _three_go_out_newest_wins(bench: Bench, role: str, messages: list) -> None:
    """A client in this role sends three link messages of one newest-wins type."""
    robot_end, wrapped = _wrapped_robot(bench, offset=(0, True, True))
    client = bench.client(role)
    for message in messages:
        send(client, message)
    send(client, MARKER)  # the service reads it behind the three
    until(lambda: ("reliable", Stop.TYPE) in wrapped.calls, "the STOP")
    kind = messages[0].TYPE
    assert wrapped.calls == [("latest", kind)] * 3 + [("reliable", Stop.TYPE)]
    # The Thor read nothing meanwhile: only the newest one waits for it.
    arrived = [decode(*robot_end.receive(WAIT_S)) for _ in range(2)]
    assert arrived == [messages[-1], MARKER]


def test_a_human_setpoint_goes_to_the_thor_newest_wins(bench):
    # The human program sends up to 50 a second: none may wait in a queue.
    setpoints = [human_setpoint(seq=seq) for seq in (1, 2, 3)]
    _three_go_out_newest_wins(bench, "human", setpoints)


@pytest.mark.parametrize(
    ("role", "messages"),
    [
        ("agent", [permit(f"move-{n}") for n in (1, 2, 3)]),
        ("viewer", [ViewStatus(live=True, age_s=n / 10) for n in (1, 2, 3)]),
    ],
    ids=["PERMIT", "VIEW_STATUS"],
)
def test_a_permit_and_a_view_status_go_to_the_thor_newest_wins(bench, role, messages):
    _three_go_out_newest_wins(bench, role, messages)


# STOP


@pytest.mark.parametrize("role", ROLES)
def test_a_stop_is_forwarded_as_it_is_with_the_clients_own_sender(bench, role):
    thor = bench.robot()
    client = bench.client(role)
    stop = Stop(reason="x" * 2000, sender="somebody else")
    send(client, stop)
    assert thor.next() == (Stop.TYPE, stop.to_payload())


def test_a_stop_does_not_wait_behind_a_send_that_hangs(bench):
    robot_end, wrapped = _wrapped_robot(bench, hold=Move.TYPE)
    agent = bench.client("agent")
    console = bench.client("console")
    send(agent, move())
    assert wrapped.holding.wait(WAIT_S)
    send(console, Stop("now", "console"))
    assert decode(*robot_end.receive(WAIT_S)) == Stop("now", "console")
    wrapped.release.set()
    assert decode(*robot_end.receive(WAIT_S)) == move()


def test_a_stop_with_a_bad_payload_is_still_a_stop(bench):
    thor = bench.robot()
    sock = bench.raw("agent")
    sock.sendall(pack_record(Stop.TYPE, b"\xffnot json"))
    assert thor.next_message() == Stop(BAD_STOP, "agent")
    write(sock, MARKER)  # the client is still served
    assert thor.next_message() == MARKER


@pytest.mark.parametrize(
    "record",
    [HEADER.pack(Stop.TYPE, 70_000), HEADER.pack(Stop.TYPE, 50) + b"half"],
    ids=["oversized", "incomplete"],
)
def test_a_stop_whose_payload_never_comes_is_still_a_stop(bench, record):
    thor = bench.robot()
    sock = bench.raw("tool")
    sock.sendall(record)
    assert thor.next_message() == Stop(BAD_STOP, "tool")
    assert "STOP whose payload" in closed_by_service(sock)[-1].text


def test_a_stop_ahead_of_the_hello_is_forwarded(bench):
    thor = bench.robot()
    sock = bench.raw()
    write(sock, Stop("first thing", "panic button"))
    assert thor.next_message() == Stop("first thing", "panic button")
    sock.sendall(pack_record(Stop.TYPE, b""))
    assert thor.next_message() == Stop(BAD_STOP, "unknown")


# A LOCAL_HELLO the service refuses, by why: (the role of the client that is
# attached already, the hello, what its sender is told).
REFUSED_HELLOS = {
    "a second human program": (
        "human",
        LocalHello(role="human", name="two"),
        "a human program is attached already; one at a time",
    ),
    "a second agent client": (
        "agent",
        LocalHello(role="agent", name="two"),
        "an agent client is attached already; one at a time",
    ),
    "an unknown role": (
        "tool",
        LocalHello(role="pilot", name="two"),
        "unknown role 'pilot'",
    ),
    "another protocol version": (
        "tool",
        LocalHello(role="tool", name="two", protocol_version=PROTOCOL_VERSION + 1),
        f"protocol version {PROTOCOL_VERSION + 1}",
    ),
}


def _in_one_write(sock: socket.socket, *messages: Message) -> None:
    """Write the records of these messages together, as a raw STOP is written."""
    sock.sendall(b"".join(pack_record(m.TYPE, m.to_payload()) for m in messages))


@pytest.mark.parametrize("case", sorted(REFUSED_HELLOS))
def test_a_stop_behind_a_refused_hello_is_forwarded(bench, case):
    attached, hello, why = REFUSED_HELLOS[case]
    thor = bench.robot()
    first = bench.client(attached, "one")
    sock = bench.raw()
    # The STOP of the human program for right B, a button of the controller.
    stop = Stop(reason="right B", sender="controller")
    _in_one_write(sock, hello, stop)
    assert thor.next_message() == stop
    told = closed_by_service(sock)
    assert [type(message) for message in told] == [LocalNotice]
    assert why in told[0].text
    # The refused client was not attached, and the first one is still served.
    assert bench.service.roles() == {attached: 1}
    send(first, MARKER)
    assert thor.next_message() == MARKER


def test_a_stop_behind_a_refused_hello_that_is_not_delivered_is_answered(bench):
    bench.client("human", "one")  # there is no robot connection
    sock = bench.raw()
    stop = Stop(reason="right B", sender="controller")
    _in_one_write(sock, LocalHello(role="human", name="two"), stop)
    told = closed_by_service(sock)
    # The STOP's notice is the last one: the one a raw STOP reports.
    assert [notice.text for notice in told] == [
        "a human program is attached already; one at a time",
        STOP_UNDELIVERED,
    ]
    assert told[-1].about_type == Stop.TYPE


@pytest.mark.parametrize(
    ("role", "message"),
    [
        ("agent", move()),
        ("human", human_setpoint(engaged=True, q=[0.1])),
        ("human", owner_request("human")),
    ],
    ids=["MOVE", "HUMAN_SETPOINT", "OWNER_REQUEST"],
)
def test_what_is_not_a_stop_behind_a_refused_hello_stays_here(bench, role, message):
    thor = bench.robot()
    bench.client(role, "one")
    sock = bench.raw()
    _in_one_write(sock, LocalHello(role=role, name="two"), message)
    assert "one at a time" in closed_by_service(sock)[-1].text
    thor.nothing()


def test_after_a_stop_permits_are_dropped_and_old_moves_refused(bench):
    thor = bench.robot(stop_number=4)
    agent = bench.client("agent")
    console = bench.client("console")
    send(console, Stop("pressed", "console"))
    assert thor.next_message() == Stop("pressed", "console")
    agent.wait_status(lambda s: s.stop_unconfirmed)

    send(agent, permit(stop_number=4))
    send(agent, permit(stop_number=5))
    send(agent, move("move-old", stop_number=4))
    send(agent, gripper("grip-old", stop_number=3))
    for command_id in ("move-old", "grip-old"):
        refusal = received(agent, Result, command_id=command_id)
        assert refusal.status == "refused"
        assert "a STOP was sent after stop number 4" in refusal.reason
    quiet(console, Result)  # a local refusal is the sender's alone
    # What is not motion still goes out, and it is the first thing that does.
    checker = CheckerState("chk", "s1", 1, 4, "clear_contact_spots")
    send(agent, checker)
    assert thor.next_message() == checker

    thor.state(stop_number=5)
    agent.wait_status(lambda s: not s.stop_unconfirmed)
    send(agent, permit(stop_number=5))
    send(agent, move("move-new", stop_number=5))
    assert thor.next_message() == permit(stop_number=5)
    assert thor.next_message() == move("move-new", stop_number=5)


def test_a_human_setpoint_stays_here_while_the_stop_gate_is_set(bench):
    thor = bench.robot(owner="human", owner_number=6, stop_number=4)
    human = bench.client("human")
    quoted = {"owner_number": 6, "stop_number": 4}
    driving = human_setpoint(seq=1, **quoted, engaged=True, q=[0.1])
    send(human, driving)
    assert thor.next_message() == driving
    # Right B of the controller: the human program's own STOP sets the gate, and
    # it goes out with the sender the human program wrote.
    pressed = Stop(reason="right B", sender="controller")
    send(human, pressed)
    assert thor.next_message() == pressed
    human.wait_status(lambda s: s.stop_unconfirmed)

    send(human, human_setpoint(seq=2, **quoted, engaged=True, q=[0.2]))
    send(human, human_setpoint(seq=3, owner_number=6, stop_number=5))
    send(human, MARKER)
    assert thor.next_message() == MARKER  # neither went out ahead of it
    time.sleep(QUIET_S)
    assert human.notices == []  # dropped without an answer, as a PERMIT is

    # The ROBOT_STATE that shows the STOP ends the gate.
    thor.state(owner="nobody", owner_number=7, stop_number=5)
    human.wait_status(lambda s: not s.stop_unconfirmed)
    after = human_setpoint(seq=4, owner_number=7, stop_number=5)
    send(human, after)
    assert thor.next_message() == after


def test_a_state_with_the_old_stop_number_does_not_end_the_gate(bench):
    thor = bench.robot(stop_number=4)
    agent = bench.client("agent")
    send(agent, Stop("self", "agent"))
    thor.next()
    sent = thor.state(stop_number=4, owner_number=9)
    until(lambda: bench.service.robot_state == sent)
    assert bench.service.status().stop_unconfirmed
    send(agent, permit(stop_number=4))
    send(agent, MARKER)
    assert thor.next_message() == MARKER


def test_the_gate_outlives_the_robot_connection_until_a_stop_is_seen(bench):
    thor = bench.robot(stop_number=4)
    agent = bench.client("agent")
    send(agent, Stop("self", "agent"))
    thor.next()
    agent.wait_status(lambda s: s.stop_unconfirmed)
    thor.carrier.break_link()
    agent.wait_status(lambda s: not s.robot_connected)
    # The STOP may be lost with the connection: the same session comes back
    # with the old stop number, and motion from here stays shut.
    again = bench.robot(stop_number=4)
    assert bench.service.status().stop_unconfirmed
    send(agent, move("move-old", stop_number=4))
    assert received(agent, Result, command_id="move-old").status == "refused"
    send(agent, Stop("again", "agent"))
    assert again.next_message() == Stop("again", "agent")
    again.state(stop_number=5)
    agent.wait_status(lambda s: not s.stop_unconfirmed)


def test_a_new_session_ends_the_gate(bench):
    thor = bench.robot(stop_number=4)
    agent = bench.client("agent")
    send(agent, Stop("self", "agent"))
    thor.next()
    thor.carrier.break_link()
    agent.wait_status(lambda s: not s.robot_connected)
    bench.robot("s2", stop_number=0)
    agent.wait_status(lambda s: s.robot_connected and not s.stop_unconfirmed)


def test_a_stop_that_cannot_be_delivered_is_answered_with_a_notice(bench):
    console = bench.client("console")
    send(console, Stop("pressed", "console"))
    notice = console.wait_notice()
    assert notice.text == STOP_UNDELIVERED
    assert notice.about_type == Stop.TYPE
    assert not bench.service.status().stop_unconfirmed


# The agent program is gone


def test_the_agent_client_gone_while_it_owns_the_arm_asks_for_help(bench):
    thor = bench.robot(owner="agent", owner_number=8, stop_number=3)
    console = bench.client("console")
    agent = bench.client("agent")
    agent.close()
    request = thor.next_message()
    assert isinstance(request, OwnerRequest)
    assert request.request == "help_needed"
    assert request.reason == AGENT_GONE == "agent program gone"
    assert (request.session_id, request.owner_number, request.stop_number) == (
        "s1",
        8,
        3,
    )
    thor.nothing()
    # Its RESULT goes to the consoles.
    thor.send(Result(request.command_id, "completed"))
    assert received(console, Result).command_id == request.command_id


def test_a_dropped_agent_client_counts_as_gone(bench):
    thor = bench.robot(owner="agent")
    sock = bench.raw("agent")
    sock.sendall(pack_record(201, b""))
    assert thor.next_message().reason == AGENT_GONE


@pytest.mark.parametrize("owner", ["nobody", "human", "recovery"])
def test_the_agent_client_gone_while_another_owns_the_arm_sends_nothing(bench, owner):
    thor = bench.robot(owner=owner)
    agent = bench.client("agent")
    console = bench.client("console")
    agent.close()
    until(lambda: bench.service.roles() == {"console": 1})
    send(console, MARKER)
    assert thor.next_message() == MARKER


def test_a_console_gone_sends_nothing(bench):
    thor = bench.robot(owner="agent")
    agent = bench.client("agent")
    console = bench.client("console")
    console.close()
    until(lambda: bench.service.roles() == {"agent": 1})
    send(agent, MARKER)
    assert thor.next_message() == MARKER


@pytest.mark.parametrize("owner", ["human", "agent", "nobody"])
@pytest.mark.parametrize("how", ["closed", "dropped"])
def test_a_human_program_that_leaves_sends_nothing_to_the_thor(bench, owner, how):
    thor = bench.robot(owner=owner, owner_number=6, stop_number=2)
    console = bench.client("console")
    sock = bench.raw("human")
    driving = human_setpoint(owner_number=6, stop_number=2, engaged=True, q=[0.1])
    write(sock, driving)
    assert thor.next_message() == driving
    if how == "closed":
        sock.close()
    else:
        sock.sendall(pack_record(201, b""))  # the service drops it for this
    until(lambda: bench.service.roles() == {"console": 1})
    # Not on the next ROBOT_STATE either: the Thor ends the drive by itself.
    last = thor.state(owner=owner, owner_number=6, stop_number=2)
    until(lambda: bench.service.robot_state == last)
    send(console, MARKER)
    assert thor.next_message() == MARKER  # no STOP and no help request before it
    thor.nothing()


def test_an_agent_program_that_leaves_while_a_person_owns_the_arm_sends_nothing_to_the_thor(
    bench,
):
    thor = bench.robot(owner="human", owner_number=6)
    human = bench.client("human")
    agent = bench.client("agent")
    agent.close()
    until(lambda: bench.service.roles() == {"human": 1})
    last = thor.state(owner="human", owner_number=6)
    until(lambda: bench.service.robot_state == last)
    # The person's next target is the first thing the Thor gets.
    target = human_setpoint(owner_number=6, engaged=True, q=[0.1])
    send(human, target)
    assert thor.next_message() == target
    thor.nothing()


def test_the_agent_client_gone_without_a_robot_connection_sends_nothing(bench):
    thor = bench.robot(owner="agent")
    agent = bench.client("agent")
    thor.carrier.break_link()
    agent.wait_status(lambda s: not s.robot_connected)
    agent.close()
    until(lambda: bench.service.roles() == {})
    again = bench.robot(owner="nobody")
    again.nothing()


def test_ownership_seen_after_the_agent_client_left_asks_for_help_once(bench):
    thor = bench.robot(owner="nobody", owner_number=1)
    agent = bench.client("agent")
    agent.close()
    until(lambda: bench.service.roles() == {})
    # The owner change the agent asked for shows only now.
    thor.state(owner="agent", owner_number=2)
    request = thor.next_message()
    assert (request.request, request.reason) == ("help_needed", AGENT_GONE)
    assert request.owner_number == 2
    for _ in range(5):
        thor.state(owner="agent", owner_number=2)
    thor.nothing()


def test_no_help_request_while_an_agent_client_is_attached(bench):
    thor = bench.robot(owner="agent")
    agent = bench.client("agent")
    for _ in range(5):
        thor.state(owner="agent")
    send(agent, MARKER)
    assert thor.next_message() == MARKER


# The robot connection


def test_the_status_tells_the_loss_with_its_reason(bench):
    robot_end, _ = _wrapped_robot(bench, close_reason="peer silent\nfor 3 s")
    console = bench.client("console")
    assert console.status.robot_connected is True
    assert console.status.close_reason == ""
    robot_end.break_link()
    status = console.wait_status(lambda s: not s.robot_connected)
    assert status.close_reason == "peer silent for 3 s"
    assert bench.service.robot_connected is False
    assert bench.service.robot_state is None


def test_a_carrier_without_a_reason_is_lost_in_the_services_words(bench):
    thor = bench.robot()
    viewer = bench.client("viewer")
    thor.carrier.break_link()
    assert viewer.wait_status(lambda s: not s.robot_connected).close_reason == LOST


def test_the_status_carries_the_clock_offset(bench):
    _, wrapped = _wrapped_robot(bench, offset=(-250_000, True, True))
    viewer = bench.client("viewer")
    status = viewer.wait_status(lambda s: s.clock_offset_ns is not None)
    assert (status.clock_offset_ns, status.clock_two_way) == (-250_000, True)
    assert bench.service.clock_offset == (-250_000, True)
    wrapped._offset = (-260_000, True, True)
    viewer.wait_status(lambda s: s.clock_offset_ns == -260_000)
    wrapped._offset = (0, False, False)  # not valid
    viewer.wait_status(lambda s: s.clock_offset_ns is None)


def test_commands_under_way_fail_when_the_robot_connection_is_lost(bench):
    thor = bench.robot()
    agent = bench.client("agent")
    console = bench.client("console")
    viewer = bench.client("viewer")
    send(agent, gripper("grip-done"))
    send(agent, move("move-running"))
    send(
        console, Reset("reset-waiting", session_id="s1", owner_number=1, stop_number=0)
    )
    for _ in range(3):
        thor.next()
    thor.send(Result("grip-done", "completed"))
    thor.send(Result("move-running", "accepted"))
    received(agent, Result, command_id="move-running")
    thor.carrier.break_link()

    failed = received(agent, Result, command_id="move-running", status="failed")
    assert failed.reason == LOST == "robot connection lost"
    quiet(agent, Result)  # not the console's command, not the finished one
    lost = {}
    for _ in range(2):
        result = received(console, Result, status="failed")
        lost[result.command_id] = result.reason
    assert lost == {"move-running": LOST, "reset-waiting": LOST}
    quiet(viewer, Result)


def test_without_a_robot_connection_commands_are_refused(bench):
    agent = bench.client("agent")
    send(agent, permit())
    send(agent, move("move-1"))
    refusal = received(agent, Result, command_id="move-1")
    assert (refusal.status, refusal.reason) == ("refused", LOST)
    assert agent.notices == []


def test_a_new_robot_connection_replays_nothing(bench):
    thor = bench.robot()
    agent = bench.client("agent")
    send(agent, move("move-1"))
    thor.next()
    thor.carrier.break_link()
    received(agent, Result, command_id="move-1", status="failed")
    send(agent, move("move-2"))
    assert received(agent, Result, command_id="move-2").status == "refused"

    again = bench.robot("s1", owner_number=2)
    agent.wait_status(lambda s: s.robot_connected)
    assert received(agent, Welcome) == welcome()
    again.nothing()
    send(agent, move("move-3", owner_number=2))
    assert again.next_message().command_id == "move-3"
    again.nothing()
    # A late RESULT for a command of the old connection is not the agent's.
    again.send(Result("move-1", "completed"))
    again.send(Event(seq=1, kind="note", text="after"))
    received(agent, Event)
    assert agent.receive(0) is None


def test_a_peer_from_a_wrong_address_is_refused_at_once(bench, caplog):
    caplog.set_level("WARNING", logger=LOGGER)
    stranger = bench.dial("10.9.9.9:4000\nx")
    assert stranger.refused.wait(WAIT_S)
    assert not stranger.carrier.alive
    assert bench.service.robot_connected is False

    def logged() -> bool:
        lines = [r.getMessage() for r in caplog.records]
        return any("not an allowed address" in s and "\n" not in s for s in lines)

    until(logged, "the log line")
    # The Thor is still taken afterwards, with or without a port in its address.
    thor = bench.dial("127.0.0.1:51234")
    until(lambda: bench.service.robot_connected)
    assert not thor.refused.is_set()


def test_allowed_peers_are_the_configured_ones(folder):
    bench = Bench(folder, allowed_peers=["192.168.7.20"])
    try:
        assert bench.dial("127.0.0.1").refused.wait(WAIT_S)
        thor = bench.dial("192.168.7.20:9")
        until(lambda: bench.service.robot_connected)
        assert not thor.refused.is_set()
    finally:
        bench.close()


@pytest.mark.parametrize(
    ("allowed", "dialing"),
    [
        ("0:0:0:0:0:0:0:1", "[::1]:5000"),
        ("::ffff:127.0.0.1", "127.0.0.1:5000"),
        ("FE80::1%eth0", "[fe80::1%eth0]:5000"),
        ("169.254.7.20", "[::ffff:169.254.7.20]:9"),
        ("192.168.7.20", "[::ffff:192.168.7.20]:9"),
        (" 192.168.7.20 ", "192.168.7.20:9"),
    ],
)
@pytest.mark.parametrize("through_the_file", [True, False])
def test_an_allowed_peer_is_taken_in_any_spelling_of_its_address(
    folder, allowed, dialing, through_the_file
):
    peers = [allowed]
    if through_the_file:
        peers = list(link_service_config({"allowed_peers": peers}).allowed_peers)
    bench = Bench(folder, allowed_peers=peers)
    try:
        assert bench.dial("192.0.2.9:5000").refused.wait(WAIT_S)
        thor = bench.dial(dialing)
        until(lambda: bench.service.robot_connected)
        assert not thor.refused.is_set()
    finally:
        bench.close()


def test_allowed_peers_are_kept_in_one_spelling():
    loaded = link_service_config(
        {"allowed_peers": ["0:0:0:0:0:0:0:1", " 127.0.0.1 ", "FE80::1%eth0", "::1"]}
    )
    assert loaded.allowed_peers == ("::1", "127.0.0.1", "fe80::1%eth0", "::1")
    assert LinkServiceConfig(allowed_peers=("::FFFF:10.0.0.2",)).allowed_peers == (
        "::ffff:10.0.0.2",
    )
    # An IPv4 link-local address has no zone and is taken as it is.
    assert link_service_config(
        {"allowed_peers": ["169.254.7.20", "::ffff:169.254.7.20"]}
    ).allowed_peers == ("169.254.7.20", "::ffff:169.254.7.20")


def test_a_link_local_peer_is_reported_with_the_zone_of_its_interface():
    # What the remote link's transport asks of the C library for a peer's text.
    try:
        index = socket.if_nametoindex("lo")
    except (AttributeError, OSError):
        pytest.skip("no interface named lo to take a zone from")
    flags = socket.NI_NUMERICHOST | socket.NI_NUMERICSERV
    host, port = socket.getnameinfo(("fe80::1", 5000, 0, index), flags)
    assert (host, port) == ("fe80::1%lo", "5000")
    assert peer_host(f"[{host}]:{port}") == "fe80::1%lo"
    # The entry that takes this peer names the interface, not its number.
    named = link_service_config({"allowed_peers": ["FE80::1%lo"]}).allowed_peers
    assert named == ("fe80::1%lo",)


def test_a_link_local_peer_is_taken_only_on_the_interface_its_entry_names(folder):
    peers = list(link_service_config({"allowed_peers": ["fe80::1%eth0"]}).allowed_peers)
    bench = Bench(folder, allowed_peers=peers)
    try:
        assert bench.dial("[fe80::1%eth1]:5000").refused.wait(WAIT_S)
        assert bench.dial("[fe80::2%eth0]:5000").refused.wait(WAIT_S)
        thor = bench.dial("[fe80::1%eth0]:5000")
        until(lambda: bench.service.robot_connected)
        assert not thor.refused.is_set()
    finally:
        bench.close()


def test_a_second_robot_connection_is_refused_while_one_is_alive(bench):
    first = bench.robot()
    console = bench.client("console")
    second = bench.dial()
    assert second.refused.wait(WAIT_S)
    send(console, MARKER)
    assert first.next_message() == MARKER  # the first one is still the one
    first.carrier.break_link()
    third = bench.robot("s3")
    send(console, MARKER)
    assert third.next_message() == MARKER
    assert not third.refused.is_set()


def test_a_robot_connection_that_cannot_be_opened_is_passed_over(bench, caplog):
    caplog.set_level("WARNING", logger=LOGGER)

    def fail():
        raise OSError("handshake failed")

    bench._incoming.put(Incoming("127.0.0.1", fail, lambda: None))
    until(lambda: any("could not be opened" in r.getMessage() for r in caplog.records))
    assert bench.robot().carrier.alive


def test_an_accept_that_raises_is_logged_once_and_tried_again(folder, caplog):
    caplog.set_level("ERROR", logger=LOGGER)
    calls = []

    def accept(timeout_s):
        calls.append(timeout_s)
        raise OSError("listener gone")

    service = RobotLinkService(accept, socket_path=str(folder / "s.sock"), poll_s=0.01)
    service.start()
    try:
        until(lambda: len(calls) >= 2)
    finally:
        service.close()
    lines = [
        r for r in caplog.records if "accepting a robot connection" in r.getMessage()
    ]
    assert len(lines) == 1
    assert len(calls) <= 5  # it pauses between tries


def test_closing_the_service_ends_the_robot_connection_and_sends_nothing(folder):
    bench = Bench(folder)
    thor = bench.robot(owner="agent")
    bench.client("agent")
    bench.client("console")
    bench.service.close()
    assert not thor.service_end.alive
    assert thor.drain() == []  # no STOP, no help request: the Thor holds by itself
    bench.close()
    with pytest.raises(RuntimeError):
        bench.service.start()


# The service originates nothing


def _allowed_from(role: str, msg_type: int, payload: bytes) -> bool:
    """Whether the service may forward this record of a client in this role."""
    if msg_type not in ROLE_TABLE.get(role, {Stop.TYPE}):
        return False
    try:
        message = decode(msg_type, payload)
    except ProtocolError:
        return False
    if isinstance(message, OwnerRequest):
        return message.request in OWNER_REQUESTS[role]
    return True


@pytest.mark.parametrize("seed", range(6))
def test_the_service_makes_no_link_message_of_its_own(folder, seed):
    """Random client input: all the Thor gets was sent by a client that may send it.

    The exceptions are the two the service is built to make: a STOP in place of
    a client's STOP with a bad payload, and the "agent program gone" request.
    """
    rng = random.Random(seed)
    bench = Bench(folder)
    try:
        thor = bench.robot()
        socks = {role: bench.raw(role) for role in ROLES}
        socks["none"] = bench.raw()  # never says LOCAL_HELLO
        sent: set[tuple[str, int, bytes]] = set()
        bad_stops = 0
        names = sorted(CLIENT_MESSAGES)
        reconnects = 0
        for step in range(150):
            role = rng.choice(sorted(socks))
            kind = rng.random()
            if kind < 0.6:
                message = CLIENT_MESSAGES[rng.choice(names)]
                if isinstance(message, HumanSetpoint):
                    message = human_setpoint(seq=step)  # one payload per sender
                msg_type, payload = message.TYPE, message.to_payload()
                if hasattr(message, "command_id"):
                    payload = payload.replace(b'-1"', f'-{step}"'.encode(), 1)
            elif kind < 0.75:
                msg_type = rng.choice([65, 66, 67, 69, 70, 97, 98, 99])
                payload = rng.randbytes(rng.randrange(0, 40))
            elif kind < 0.9:
                msg_type, payload = Stop.TYPE, rng.randbytes(rng.randrange(0, 40))
                bad_stops += 1
            elif kind < 0.95:
                thor.state(
                    owner=rng.choice(["agent", "nobody"]),
                    owner_number=rng.randrange(1, 4),
                    stop_number=rng.randrange(0, 3),
                )
                continue
            else:
                msg_type, payload = rng.choice([0, 64, 96, 150, 192, 193, 255]), b"{}"
            sent.add((role, msg_type, payload))
            try:
                socks[role].sendall(pack_record(msg_type, payload))
            except OSError:
                # The service dropped this client; a few come back, with a pause.
                if role != "none" and reconnects < 4:
                    reconnects += 1
                    time.sleep(0.02)
                    until(lambda role=role: role not in bench.service.roles())
                    socks[role] = bench.raw(role)

        # The agent owns the arm when every client leaves.
        last = thor.state(owner="agent", owner_number=9)
        until(lambda: bench.service.robot_state == last)
        for sock in socks.values():
            sock.close()
        until(lambda: not bench.service._clients and not bench.service._guests)
        time.sleep(QUIET_S)

        forwarded = thor.drain()
        help_requests = 0
        for msg_type, payload in forwarded:
            if any(
                (role, msg_type, payload) in sent
                and _allowed_from(role, msg_type, payload)
                for role in socks
            ):
                continue
            message = decode(msg_type, payload)
            if isinstance(message, Stop):
                assert message.reason == BAD_STOP and bad_stops
                assert message.sender in (*ROLES, "unknown")
            else:
                assert isinstance(message, OwnerRequest), message
                assert message.request == "help_needed"
                assert message.reason == AGENT_GONE
                assert message.command_id.startswith("link-service-")
                help_requests += 1
        assert forwarded, "nothing was forwarded at all"
        assert help_requests >= 1
        # Never a motion link message that no agent client sent.
        for msg_type, payload in forwarded:
            if msg_type in (Move.TYPE, 66, Permit.TYPE):
                assert ("agent", msg_type, payload) in sent
            if msg_type == ViewStatus.TYPE:
                assert ("viewer", msg_type, payload) in sent
            # Never a joint target that no human program sent.
            if msg_type == HumanSetpoint.TYPE:
                assert ("human", msg_type, payload) in sent
    finally:
        bench.close()


# Settings


def _shipped_config() -> Path:
    root = next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "VERSION").is_file() and (parent / "CMakeLists.txt").is_file()
    )
    return root / "examples" / "agentic_remote" / "configs" / "link_service.yaml"


def test_the_shipped_settings_load_and_match_the_defaults():
    loaded = load_link_service_config(_shipped_config())
    assert loaded == LinkServiceConfig()
    assert loaded.listen_endpoint == "127.0.0.1:47610"
    assert loaded.allowed_peers == ("127.0.0.1",)
    assert loaded.socket_path is None
    assert loaded.log_level == "info"


def test_settings_become_the_services_options(folder):
    loaded = link_service_config(
        {
            "listen": {"address": "0.0.0.0", "port": 5000},
            "allowed_peers": ["192.168.7.20", "192.168.7.21"],
            "socket_path": str(folder / "s.sock"),
            "log_level": "debug",
            "local": {"max_clients": 4, "send_timeout_s": 0.5},
            "operator": {"typed_takeover": True},
        }
    )
    assert loaded.listen_endpoint == "0.0.0.0:5000"
    options = loaded.service_options()
    assert options["allowed_peers"] == ("192.168.7.20", "192.168.7.21")
    assert options["max_clients"] == 4
    assert options["typed_takeover"] is True
    assert LinkServiceConfig().service_options()["typed_takeover"] is False
    service = RobotLinkService(lambda timeout_s: time.sleep(timeout_s), **options)
    assert service.socket_path == str(folder / "s.sock")


@pytest.mark.link
@pytest.mark.parametrize("setting", [True, False])
def test_the_program_says_at_its_start_that_the_typed_fallback_is_on(
    folder, monkeypatch, setting
):
    """The service's program on 127.0.0.1 with port 0; nothing dials it."""
    monkeypatch.setattr(service_main, "set_console_level", lambda level: None)
    file = folder / "link_service.yaml"
    settings = {
        "listen": {"address": "127.0.0.1", "port": 0},
        "socket_path": str(folder / "s.sock"),
        "operator": {"typed_takeover": setting},
    }
    file.write_text(yaml.safe_dump(settings), encoding="utf-8")
    said: list[str] = []
    code = service_main.main(
        ["--config", str(file)], wait=lambda program: "ended", say=said.append
    )
    assert code == 0
    assert said[0].startswith("robot link service: waits for the robot app on ")
    assert said[-1].startswith("robot link service: ended; closing.")
    on = (
        "robot link service: operator.typed_takeover is on: the operator console "
        "may ask for a takeover and for the hand-back"
    )
    assert said[1:-1] == ([on] if setting else [])


def test_an_empty_file_gives_the_defaults(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    assert load_link_service_config(path) == LinkServiceConfig()


@pytest.mark.parametrize(
    ("data", "problem"),
    [
        ({"listen": {"port": 70000}}, r"listen\.port must be from 0 to 65535"),
        ({"listen": {"port": "47610"}}, r"listen\.port must be a whole number"),
        ({"listen": {"address": ""}}, r"listen\.address must be a non-empty string"),
        ({"listen": {"host": "x"}}, r"unknown setting\(s\): listen\.host"),
        ({"listen": []}, r"listen must be a mapping"),
        ({"allowed_peers": []}, r"allowed_peers must be a non-empty list"),
        ({"allowed_peers": "127.0.0.1"}, r"allowed_peers must be a list"),
        ({"allowed_peers": [7]}, r"allowed_peers\[0\] must be a non-empty string"),
        ({"allowed_peers": ["thor"]}, r"allowed_peers\[0\] must be an IP address"),
        (
            {"allowed_peers": ["127.0.0.1", "192.0.2.10:47610"]},
            r"allowed_peers\[1\] must be an IP address, got '192\.0\.2\.10:47610'",
        ),
        # A link-local peer is reported with its interface's name: an entry
        # without one, or with the interface's number, would never match.
        (
            {"allowed_peers": ["fe80::1"]},
            r"allowed_peers\[0\] is a link-local address: write it with the name "
            r"of this workstation's interface, as in fe80::1%eth0, got 'fe80::1'",
        ),
        (
            {"allowed_peers": ["127.0.0.1", " FE80::1%2 "]},
            r"allowed_peers\[1\] is a link-local address: write it with the name "
            r"of this workstation's interface, as in fe80::1%eth0, got ' FE80::1%2 '",
        ),
        ({"log_level": "loud"}, r"log_level must be one of"),
        ({"local": {"max_clients": 0}}, r"local\.max_clients must be from 1"),
        ({"local": {"hello_timeout_s": 0}}, r"local\.hello_timeout_s must be above 0"),
        ({"local": {"queue_bytes": True}}, r"local\.queue_bytes must be a whole"),
        (
            {"operator": {"typed_takeover": "true"}},
            r"operator\.typed_takeover must be true or false, got 'true'",
        ),
        (
            {"operator": {"typed_takeover": 1}},
            r"operator\.typed_takeover must be true or false, got 1",
        ),
        (
            {"operator": {"typed_takover": True}},
            r"unknown setting\(s\): operator\.typed_takover",
        ),
        ({"operator": None}, r"operator must be a mapping"),
        ({"typed_takeover": True}, r"unknown setting\(s\): typed_takeover"),
        ({"thor": "x", "local": {}}, r"unknown setting\(s\): thor"),
        ([1], r"the file must be a mapping"),
    ],
)
def test_a_wrong_setting_is_named(data, problem):
    with pytest.raises(ConfigError, match=problem):
        link_service_config(data)


def test_a_file_that_cannot_be_read_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match=r"cannot read"):
        load_link_service_config(tmp_path / "none.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("listen: [1\n")
    with pytest.raises(ConfigError, match=r"cannot read"):
        load_link_service_config(bad)


def test_the_too_many_clients_are_turned_away(folder):
    bench = Bench(folder, max_clients=2)
    try:
        thor = bench.robot()
        bench.client("console")
        bench.client("viewer")
        extra = bench.raw()
        write(extra, MARKER)  # it is not served, but its STOP goes out
        assert thor.next_message() == MARKER
        assert "too many local clients" in closed_by_service(extra)[-1].text
        assert bench.service.roles() == {"console": 1, "viewer": 1}
    finally:
        bench.close()


def _full_bench(folder: Path) -> Bench:
    """A service with a robot connection and both of its two client slots held."""
    bench = Bench(folder, max_clients=2)
    bench.robot()
    bench.client("console")
    bench.client("viewer")
    return bench


@pytest.mark.parametrize("writes", [1, 2])
def test_a_turned_away_clients_stop_after_its_hello_goes_out(folder, writes):
    bench = _full_bench(folder)
    try:
        extra = bench.raw()
        hello = LocalHello(role="tool", name="stop-button")
        stop = Stop("operator pressed stop", "tool")
        records = [pack_record(m.TYPE, m.to_payload()) for m in (hello, stop)]
        for part in records if writes == 2 else [b"".join(records)]:
            extra.sendall(part)
        assert bench.thors[0].next_message() == stop
        assert "too many local clients" in closed_by_service(extra)[-1].text
        assert bench.service.roles() == {"console": 1, "viewer": 1}
    finally:
        bench.close()


def test_a_turned_away_clients_move_after_its_hello_stays_here(folder):
    bench = _full_bench(folder)
    try:
        extra = bench.raw()
        write(extra, LocalHello(role="agent", name="raw"))
        write(extra, move())
        assert "too many local clients" in closed_by_service(extra)[-1].text
        bench.thors[0].nothing()
    finally:
        bench.close()


def test_turned_away_clients_do_not_hold_up_the_next_ones_stop(folder):
    bench = _full_bench(folder)
    try:
        for _ in range(3):  # they say nothing
            bench.raw()
            time.sleep(0.02)
        extra = bench.raw()
        began = time.monotonic()
        write(extra, MARKER)
        assert bench.thors[0].next_message() == MARKER
        assert time.monotonic() - began < 0.4
    finally:
        bench.close()


def test_a_stop_goes_out_when_many_are_turned_away_at_once(folder):
    bench = _full_bench(folder)
    try:
        for _ in range(service_module.TURN_AWAY_HELPERS + 1):  # they say nothing
            bench.raw()
            time.sleep(0.02)
        extra = bench.raw()
        write(extra, MARKER)
        assert bench.thors[0].next_message() == MARKER
        assert "too many local clients" in closed_by_service(extra)[-1].text
    finally:
        bench.close()


def test_a_stop_before_the_first_state_is_not_confirmed_by_that_state(bench):
    thor = bench.robot(session_id=None)  # connected, no ROBOT_STATE yet
    agent = bench.client("agent")
    console = bench.client("console")
    send(console, Stop("pressed", "console"))
    assert thor.next_message() == Stop("pressed", "console")
    agent.wait_status(lambda s: s.stop_unconfirmed)
    # A ROBOT_STATE the Thor may have built before it saw the STOP.
    thor.send(welcome())
    sent = thor.state(stop_number=4)
    until(lambda: bench.service.robot_state == sent)
    assert bench.service.status().stop_unconfirmed
    send(agent, permit(stop_number=4))
    send(agent, move("move-old", stop_number=4))
    refusal = received(agent, Result, command_id="move-old")
    assert "a STOP was sent after stop number 4" in refusal.reason
    send(agent, MARKER)
    assert thor.next_message() == MARKER

    thor.state(stop_number=5)
    agent.wait_status(lambda s: not s.stop_unconfirmed)
    send(agent, move("move-new", stop_number=5))
    assert thor.next_message() == move("move-new", stop_number=5)


def test_a_stop_before_the_first_state_is_confirmed_by_a_later_one(bench):
    thor = bench.robot(session_id=None)
    agent = bench.client("agent")
    send(agent, Stop("self", "agent"))
    thor.next()
    thor.send(welcome())
    first = thor.state(stop_number=0)
    until(lambda: bench.service.robot_state == first)
    assert bench.service.status().stop_unconfirmed
    thor.state(stop_number=1)
    agent.wait_status(lambda s: s.session_id == "s1" and not s.stop_unconfirmed)


def test_a_stop_before_the_first_state_holds_human_setpoints_back_too(bench):
    thor = bench.robot(session_id=None)  # connected, no ROBOT_STATE yet
    human = bench.client("human")
    pressed = Stop(reason="right B", sender="controller")
    send(human, pressed)
    assert thor.next_message() == pressed
    human.wait_status(lambda s: s.stop_unconfirmed)
    # The Thor may have counted that STOP before it built its first ROBOT_STATE:
    # the stop number then does not rise again by itself.
    thor.send(welcome())
    first = thor.state(stop_number=3)
    until(lambda: bench.service.robot_state == first)
    assert bench.service.status().stop_unconfirmed
    send(human, human_setpoint(seq=1, stop_number=3))
    # The request for a takeover is no motion: it goes out, and first.
    takeover = owner_request("human", "own-b", stop_number=3)
    send(human, takeover)
    assert thor.next_message() == takeover
    # Right B again: the stop number that follows this STOP ends the gate.
    send(human, pressed)
    assert thor.next_message() == pressed
    thor.state(stop_number=4)
    human.wait_status(lambda s: not s.stop_unconfirmed)
    after = human_setpoint(seq=2, stop_number=4)
    send(human, after)
    assert thor.next_message() == after


def test_a_stop_shuts_the_gate_before_it_goes_out(bench):
    robot_end, wrapped = _wrapped_robot(bench, hold=Stop.TYPE)
    send(robot_end, welcome())
    state = robot_state(stop_number=4)
    send(robot_end, state)
    until(lambda: bench.service.robot_state == state)
    agent = bench.client("agent")
    console = bench.client("console")
    send(console, Stop("now", "console"))
    assert wrapped.holding.wait(WAIT_S)  # the STOP is inside its send
    send(agent, permit(stop_number=4))
    send(agent, gripper("grip-old", stop_number=4))
    send(agent, move("move-old", stop_number=4))
    for command_id in ("grip-old", "move-old"):
        refusal = received(agent, Result, command_id=command_id)
        assert refusal.status == "refused"
    wrapped.release.set()
    assert decode(*robot_end.receive(WAIT_S)) == Stop("now", "console")
    send(agent, MARKER)
    assert decode(*robot_end.receive(WAIT_S)) == MARKER  # nothing else went out


def test_a_command_id_another_client_has_under_way_is_refused(bench):
    thor = bench.robot()
    agent = bench.client("agent")
    console = bench.client("console")
    send(agent, move("m1"))
    assert thor.next_message() == move("m1")
    send(console, owner_request("rest", command_id="m1"))
    refusal = received(console, Result, command_id="m1")
    assert refusal.status == "refused"
    assert "command id in use" in refusal.reason
    quiet(agent, Result)  # the refusal is the console's alone
    send(console, MARKER)
    assert thor.next_message() == MARKER  # the OWNER_REQUEST did not go out
    thor.send(Result("m1", "completed"))
    assert received(agent, Result, command_id="m1").status == "completed"
    assert received(console, Result, command_id="m1").status == "completed"
    # With its final RESULT in, the id is free again.
    send(console, owner_request("rest", command_id="m1"))
    assert thor.next_message() == owner_request("rest", command_id="m1")
    thor.send(Result("m1", "refused", "not now"))
    assert received(console, Result, command_id="m1").reason == "not now"
    quiet(agent, Result)


def test_a_client_may_send_its_own_command_id_again(bench):
    thor = bench.robot()
    agent = bench.client("agent")
    send(agent, move("m1"))
    send(agent, move("m1"))
    assert [thor.next_message(), thor.next_message()] == [move("m1"), move("m1")]
    thor.send(Result("m1", "completed"))
    assert received(agent, Result, command_id="m1").status == "completed"


def test_a_status_is_a_local_status(bench):
    assert bench.service.status() == LocalStatus(False, "", "", None, False, {}, False)
