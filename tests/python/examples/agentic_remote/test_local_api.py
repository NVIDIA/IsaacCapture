# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The local API: its records, its own messages, its socket and its clients."""

from __future__ import annotations

import os
import re
import select
import socket
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from link_bench import (
    QUIET_S,
    WAIT_S,
    Bench,
    closed_by_service,
    read_message,
    received,
    until,
    welcome,
    write,
)
from repo_paths import repo_root

import isaaccapture_examples.agentic_remote.link_service.local as local
from isaaccapture_examples.agentic_remote.link_service.config import (
    load_link_service_config,
)
from isaaccapture_examples.agentic_remote.link_service.local import (
    HEADER,
    LocalCarrier,
    LocalListener,
    LocalRefused,
    LocalSocketError,
    LocalSocketInUse,
    RecordQueue,
    Stalled,
    connect_local,
    default_socket_path,
    pack_record,
    read_exact,
)
from isaaccapture_examples.agentic_remote.link_service.service import RobotLinkService
from isaaccapture_examples.agentic_remote.protocol.carrier import Carrier
from isaaccapture_examples.agentic_remote.protocol.local_messages import (
    LOCAL_MESSAGE_TYPES,
    LOCAL_TYPES,
    MAX_LOCAL_PAYLOAD_BYTES,
    ROLES,
    LocalHello,
    LocalNotice,
    LocalStatus,
    decode_local,
    is_local,
)
from isaaccapture_examples.agentic_remote.protocol.messages import (
    MAX_PAYLOAD_BYTES,
    MESSAGE_TYPES,
    NEWEST_WINS_TYPES,
    PROTOCOL_VERSION,
    RELIABLE_TYPES,
    Event,
    ProtocolError,
    RobotState,
    Stop,
    Welcome,
)

CONFIGS = repo_root() / "examples" / "agentic_remote" / "configs"

STATUS = LocalStatus(
    robot_connected=True,
    close_reason="",
    session_id="s1",
    clock_offset_ns=-12345,
    clock_two_way=True,
    roles={"agent": 1, "console": 2},
    stop_unconfirmed=True,
)


@pytest.fixture
def folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A short private folder: a socket path holds 107 bytes at most."""
    return tmp_path_factory.mktemp("ls")


@pytest.fixture
def bench(folder: Path) -> Iterator[Bench]:
    made = Bench(folder)
    yield made
    made.close()


def _never(timeout_s: float) -> None:
    time.sleep(timeout_s)


# The local API's own messages


def test_the_local_messages_have_the_numbers_of_the_contract():
    assert (LocalHello.TYPE, LocalStatus.TYPE, LocalNotice.TYPE) == (192, 193, 194)
    assert set(LOCAL_MESSAGE_TYPES) == {192, 193, 194}
    assert list(LOCAL_TYPES) == list(range(192, 200))
    assert not set(LOCAL_TYPES) & set(MESSAGE_TYPES)
    assert not set(LOCAL_TYPES) & (set(RELIABLE_TYPES) | set(NEWEST_WINS_TYPES))
    assert is_local(192) and is_local(199)
    assert not is_local(191) and not is_local(200) and not is_local(68)
    assert ROLES == ("agent", "console", "viewer", "tool", "human")


@pytest.mark.parametrize(
    "message",
    [
        LocalHello(role="agent", name="apai"),
        STATUS,
        LocalStatus(False, "peer closed", "", None, False),
        LocalNotice(text="the tool role may not send MOVE", about_type=65),
        LocalNotice(text="no"),
    ],
)
def test_a_local_message_survives_encoding(message):
    assert decode_local(message.TYPE, message.to_payload()) == message


def test_hello_says_the_protocol_version_by_default():
    assert LocalHello(role="tool", name="x").protocol_version == PROTOCOL_VERSION
    old = decode_local(192, b'{"role":"tool","name":"x"}')
    assert old.protocol_version == PROTOCOL_VERSION


@pytest.mark.parametrize(
    ("msg_type", "payload", "problem"),
    [
        (192, b"not json", r"bad JSON"),
        (192, b"\xff\xfe", r"bad JSON"),
        (192, b"[1]", r"expected an object"),
        (192, b'{"role":"agent"}', r"missing field 'name'"),
        (192, b'{"role":7,"name":"x"}', r"LocalHello\.role"),
        (192, b'{"role":"agent","name":"x","protocol_version":true}', r"protocol_ver"),
        (193, b'{"robot_connected":1}', r"LocalStatus\.robot_connected"),
        (194, b'{"text":"x","about_type":"MOVE"}', r"LocalNotice\.about_type"),
        (195, b"{}", r"unknown local API message type 195"),
        (68, b"{}", r"unknown local API message type 68"),
        (192, b" " * (MAX_LOCAL_PAYLOAD_BYTES + 1), r"over the limit"),
    ],
)
def test_a_bad_local_message_is_a_protocol_error(msg_type, payload, problem):
    with pytest.raises(ProtocolError, match=problem):
        decode_local(msg_type, payload)


def test_a_local_message_ignores_fields_it_does_not_know():
    payload = b'{"text":"x","later":[1,2]}'
    assert decode_local(194, payload) == LocalNotice(text="x")


# Records


def test_a_record_is_type_length_payload():
    record = pack_record(68, b"abc")
    assert record == b"\x44\x00\x00\x00\x03abc"
    assert HEADER.size == 5
    assert HEADER.unpack(record[:5]) == (68, 3)
    assert pack_record(255, b"") == b"\xff\x00\x00\x00\x00"


@pytest.mark.parametrize("msg_type", [-1, 256])
def test_a_record_needs_a_type_number_of_one_byte(msg_type):
    with pytest.raises(ProtocolError, match=r"does not fit one byte"):
        pack_record(msg_type, b"")


def test_a_record_refuses_a_payload_over_the_limit():
    with pytest.raises(ProtocolError, match=r"over the limit"):
        pack_record(65, bytes(MAX_PAYLOAD_BYTES + 1))


def test_read_exact_joins_parts_and_sees_the_end():
    a, b = socket.socketpair()
    with a, b:
        a.sendall(b"ab")
        a.sendall(b"cde")
        assert read_exact(b, 5) == b"abcde"
        a.sendall(b"x")
        with pytest.raises(Stalled):
            read_exact(b, 2, time.monotonic() + QUIET_S)
        a.close()
        with pytest.raises(EOFError):
            read_exact(b, 1)


def test_the_record_queue_keeps_order_and_only_the_newest_state():
    waiting = RecordQueue(max_messages=8, max_bytes=1000)
    assert waiting.add(75, b"e1")
    assert waiting.add(96, b"state 1")
    assert waiting.add(193, b"status 1")
    assert waiting.add(75, b"e2")
    assert waiting.add(96, b"state 2")
    assert waiting.add(193, b"status 2")
    assert len(waiting) == 4
    taken = [waiting.take() for _ in range(5)]
    assert taken == [
        (75, b"e1"),
        (75, b"e2"),
        (96, b"state 2"),
        (193, b"status 2"),
        None,
    ]


def test_the_record_queue_is_bounded_in_count_and_bytes():
    waiting = RecordQueue(max_messages=2, max_bytes=10)
    assert waiting.add(75, b"1234")
    assert not waiting.add(75, b"1234567")  # 11 bytes in all
    assert waiting.add(96, b"123456")
    assert not waiting.add(75, b"")  # a third record
    assert waiting.add(96, b"12")  # replaces the waiting state
    assert waiting.take() == (75, b"1234")
    assert waiting.add(75, b"12345678")
    waiting.clear()
    assert len(waiting) == 0 and waiting.take() is None
    assert waiting.add(75, b"1234567890")


# The socket


def test_the_default_socket_path(monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/77")
    assert default_socket_path() == "/run/user/77/agentic_remote/robot_link.sock"
    monkeypatch.delenv("XDG_RUNTIME_DIR")
    expected = f"/tmp/agentic_remote-{os.getuid()}/robot_link.sock"
    assert default_socket_path() == expected


def test_the_folder_is_made_private_and_the_socket_too(folder):
    old = os.umask(0)  # a wide-open umask must not widen either
    try:
        listener = LocalListener(str(folder / "made" / "s.sock"))
    finally:
        os.umask(old)
    try:
        assert stat.S_IMODE(os.stat(folder / "made").st_mode) == 0o700
        mode = os.stat(listener.path).st_mode
        assert stat.S_ISSOCK(mode)
        assert stat.S_IMODE(mode) == 0o600
    finally:
        listener.close()
    assert not os.path.exists(listener.path)


def test_a_folder_open_to_others_is_refused(folder):
    shared = folder / "shared"
    shared.mkdir()
    shared.chmod(0o755)
    with pytest.raises(LocalSocketError, match=r"open to other users \(mode 0755\)"):
        LocalListener(str(shared / "s.sock"))
    assert not (shared / "s.sock").exists()


def test_a_path_too_long_for_a_socket_is_refused(folder):
    with pytest.raises(LocalSocketError, match=r"longer than 107 bytes"):
        LocalListener(str(folder / ("x" * 120)))


def test_a_stale_socket_file_is_replaced(folder):
    path = str(folder / "s.sock")
    dead = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    dead.bind(path)
    dead.close()  # the file stays, nobody listens
    assert os.path.exists(path)
    listener = LocalListener(path)
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(path)
        client.close()
        assert listener.accept(WAIT_S) is not None
    finally:
        listener.close()


def test_a_file_that_is_not_a_socket_is_not_replaced(folder):
    path = folder / "s.sock"
    path.write_text("keep me")
    with pytest.raises(LocalSocketError, match=r"is not a socket"):
        LocalListener(str(path))
    assert path.read_text() == "keep me"


def test_a_second_service_on_a_live_socket_fails_with_a_clear_line(bench):
    second = RobotLinkService(_never, socket_path=bench.path)
    with pytest.raises(LocalSocketInUse, match=r"a robot link service already runs on"):
        second.start()
    second.close()
    # The first one still serves.
    client = bench.client("tool")
    assert client.alive


def test_a_live_listener_without_the_lock_is_seen_too(folder):
    path = str(folder / "s.sock")
    other = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    other.bind(path)
    other.listen(1)
    try:
        with pytest.raises(LocalSocketInUse):
            LocalListener(path)
        assert os.path.exists(path)
    finally:
        other.close()


def test_closing_the_service_removes_the_socket_and_frees_the_path(folder):
    first = Bench(folder)
    first.close()
    assert not os.path.exists(first.path)
    second = Bench(folder)
    try:
        assert second.client("tool").alive
    finally:
        second.close()


# LOCAL_HELLO


def test_a_client_is_a_carrier_and_gets_the_status_first(bench):
    client = bench.client("console", "desk")
    assert isinstance(client, LocalCarrier)
    assert isinstance(client, Carrier)
    status = client.status
    assert status.robot_connected is False
    assert status.session_id == ""
    assert status.clock_offset_ns is None
    assert status.roles == {"console": 1}
    assert status.stop_unconfirmed is False
    assert bench.service.roles() == {"console": 1}


def test_every_client_sees_who_is_attached(bench):
    console = bench.client("console")
    agent = bench.client("agent")
    viewer = bench.client("viewer")
    everyone = {"console": 1, "agent": 1, "viewer": 1}
    for client in (console, agent, viewer):
        client.wait_status(lambda s: s.roles == everyone)
    viewer.close()
    console.wait_status(lambda s: s.roles == {"console": 1, "agent": 1})


def test_a_new_client_gets_the_welcome_and_the_newest_state(bench):
    thor = bench.robot()
    sent = thor.state(owner_number=4)
    until(lambda: bench.service.robot_state == sent)
    client = bench.client("viewer")
    assert client.status.robot_connected is True
    assert client.status.session_id == "s1"
    assert received(client, Welcome) == welcome()
    assert received(client, RobotState) == sent


@pytest.mark.parametrize(
    ("hello", "problem"),
    [
        (LocalHello(role="root", name="x"), r"unknown role 'root'"),
        (LocalHello(role="agent", name="x", protocol_version=99), r"version 99"),
    ],
)
def test_a_bad_hello_is_refused_with_the_reason(bench, hello, problem):
    sock = bench.raw()
    write(sock, hello)
    messages = closed_by_service(sock)
    assert [type(m) for m in messages] == [LocalNotice]
    assert re.search(problem, messages[0].text)
    assert bench.service.roles() == {}


def test_connect_local_raises_the_refusal(bench):
    with pytest.raises(LocalRefused, match=r"unknown role 'boss'"):
        connect_local(bench.path, "boss", "x", WAIT_S)


def test_connect_local_fails_when_nothing_listens(folder):
    with pytest.raises(OSError):
        connect_local(str(folder / "none.sock"), "tool", "x", 0.5)


def test_one_agent_at_a_time_and_the_next_after_it_left(bench):
    first = bench.client("agent", "one")
    with pytest.raises(LocalRefused, match=r"an agent client is attached already"):
        connect_local(bench.path, "agent", "two", WAIT_S)
    assert first.alive
    assert bench.service.roles() == {"agent": 1}
    first.close()
    until(lambda: bench.service.roles() == {})
    assert bench.client("agent", "three").alive


def test_two_consoles_may_attach(bench):
    bench.client("console")
    bench.client("console")
    assert bench.service.roles() == {"console": 2}


def test_a_client_that_does_not_start_with_hello_is_dropped(bench):
    sock = bench.raw()
    write(sock, Event(seq=1, kind="x", text="x"))
    messages = closed_by_service(sock)
    assert "before its LOCAL_HELLO" in messages[-1].text


def test_a_client_that_says_nothing_is_dropped(folder):
    bench = Bench(folder, hello_timeout_s=0.1)
    try:
        sock = bench.raw()
        messages = closed_by_service(sock)
        assert "no LOCAL_HELLO in time" in messages[-1].text
    finally:
        bench.close()


def test_a_hello_name_is_cleaned_before_it_is_logged(bench, caplog):
    caplog.set_level("INFO", logger="isaaccapture.examples.agentic_remote")
    sock = bench.raw()
    write(sock, LocalHello(role="tool", name="evil\nline\x1b[31m" + "x" * 500))
    assert isinstance(read_message(sock), LocalStatus)
    until(lambda: any("attached" in r.getMessage() for r in caplog.records))
    line = next(r.getMessage() for r in caplog.records if "attached" in r.getMessage())
    assert "\n" not in line and "\x1b" not in line
    assert len(line) < 200


# Malformed input


@pytest.mark.parametrize(
    ("record", "problem"),
    [
        (pack_record(7, b"{}"), "type 7"),
        (pack_record(200, b"{}"), "type 200"),
        (pack_record(193, STATUS.to_payload()), "LOCAL_STATUS"),
        (pack_record(192, b'{"role":"tool","name":"again"}'), "LOCAL_HELLO"),
        (HEADER.pack(75, 70_000), "over the limit"),
        (HEADER.pack(65, MAX_PAYLOAD_BYTES + 1), "over the limit"),
    ],
)
def test_malformed_input_drops_that_client_only(bench, record, problem):
    thor = bench.robot()
    other = bench.client("console")
    sock = bench.raw("tool")
    sock.sendall(record)
    messages = closed_by_service(sock)
    assert isinstance(messages[-1], LocalNotice)
    assert problem in messages[-1].text
    thor.nothing()
    # The other client is still served.
    thor.send(Event(seq=1, kind="note", text="still here"))
    assert received(other, Event).text == "still here"


def test_a_hello_over_its_limit_drops_the_client(bench):
    sock = bench.raw()
    sock.sendall(HEADER.pack(192, MAX_LOCAL_PAYLOAD_BYTES + 1))
    assert "over the limit" in closed_by_service(sock)[-1].text


def test_half_a_record_holds_only_its_own_client(bench):
    thor = bench.robot()
    stuck = bench.raw("tool")
    stuck.sendall(HEADER.pack(75, 100) + b"half")  # the rest never comes
    console = bench.client("console")
    console.send_reliable(Stop.TYPE, Stop("go on", "console").to_payload())
    assert thor.next_message() == Stop("go on", "console")


# Slow and stuck clients


def test_a_client_that_does_not_read_is_dropped_and_blocks_nobody(folder):
    bench = Bench(folder, queue_messages=16, send_timeout_s=0.5)
    try:
        thor = bench.robot()
        stuck = bench.raw("viewer")
        reader = bench.client("console")
        text = "x" * 60_000
        began = time.monotonic()
        for seq in range(200):  # 12 MB, in step with the client that reads
            thor.send(Event(seq=seq, kind="note", text=text))
            assert received(reader, Event).seq == seq
        assert time.monotonic() - began < WAIT_S
        until(lambda: bench.service.roles() == {"console": 1}, "the drop")
        # The stuck client's socket was closed by the service.
        with pytest.raises(EOFError):
            while True:
                read_exact(stuck, 65536)
    finally:
        bench.close()


def test_states_do_not_pile_up_for_a_client_that_does_not_read(folder):
    bench = Bench(folder, queue_messages=8, send_timeout_s=30.0)
    try:
        thor = bench.robot()
        bench.raw("viewer")  # never reads
        reader = bench.client("console")
        for _ in range(500):
            thor.state()
        last = thor.state(owner_number=77)
        assert received(reader, RobotState, owner_number=77) == last
        # Newest wins: 500 states did not overflow the queue of 8.
        assert bench.service.roles() == {"viewer": 1, "console": 1}
    finally:
        bench.close()


def test_a_carrier_that_is_not_read_stops_reading_its_socket(bench):
    thor = bench.robot()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(bench.path)
    lazy = LocalCarrier(sock, inbox_messages=4)
    try:
        lazy.send_reliable(192, LocalHello(role="viewer", name="lazy").to_payload())
        lazy.wait_status()
        for seq in range(50):
            thor.send(Event(seq=seq, kind="note", text="x"))
        time.sleep(QUIET_S)
        assert len(lazy._inbox) <= 4
        # Reading again gives every one of them, in order.
        for seq in range(50):
            assert received(lazy, Event).seq == seq
    finally:
        lazy.close()


# The client's carrier


def test_the_carrier_dies_with_the_service(folder):
    bench = Bench(folder)
    client = bench.client("tool")
    bench.service.close()
    assert client.wait_dead(WAIT_S)
    assert not client.alive
    assert client.close_reason == "closed by the robot link service"
    client.send_reliable(Stop.TYPE, Stop("x", "tool").to_payload())  # ignored
    began = time.monotonic()
    assert client.receive(QUIET_S) is None
    assert time.monotonic() - began >= QUIET_S * 0.9  # a dead carrier still paces
    bench.close()


def test_closing_the_carrier_detaches_the_client(bench):
    client = bench.client("viewer")
    client.close()
    assert not client.alive
    assert client.close_reason == "closed by this end"
    until(lambda: bench.service.roles() == {})


def test_the_carrier_refuses_a_payload_over_the_limit(bench):
    client = bench.client("agent")
    with pytest.raises(ProtocolError, match=r"over the limit"):
        client.send_reliable(65, bytes(MAX_PAYLOAD_BYTES + 1))
    assert client.alive


def test_local_messages_do_not_pass_through_receive(bench):
    client = bench.client("viewer")
    bench.client("console")  # a LOCAL_STATUS for the viewer
    client.wait_status(lambda s: s.roles.get("console") == 1)
    assert client.receive(QUIET_S) is None
    assert client.notices == []


# A STOP and a socket nobody reads

FILLER = pack_record(Event.TYPE, Event(seq=1, kind="note", text="x" * 60).to_payload())
STOP_PAYLOAD = Stop("x", "console").to_payload()


def unread_carrier(**options) -> tuple[LocalCarrier, socket.socket]:
    """(a carrier whose socket takes no further send at once, the other end).

    Nobody reads the other end. The filler is written straight to the socket,
    until a send with a timeout, as the carrier's, would wait.
    """
    near, far = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    carrier = LocalCarrier(near, **options)
    while select.select([], [near], [], 0)[1]:
        near.send(FILLER, socket.MSG_DONTWAIT)
    return carrier, far


def records_at(far: socket.socket) -> list[int]:
    """The type number of every whole record that waits at the other end."""
    far.setblocking(False)
    data = b""
    while True:
        try:
            part = far.recv(65536)
        except BlockingIOError:
            break
        if not part:
            break
        data += part
    types = []
    while len(data) >= HEADER.size:
        msg_type, length = HEADER.unpack(data[: HEADER.size])
        assert len(data) >= HEADER.size + length, "half a record was written"
        data = data[HEADER.size + length :]
        types.append(msg_type)
    assert data == b""
    return types


def test_a_bounded_stop_gives_up_on_a_socket_nobody_reads():
    carrier, far = unread_carrier()
    try:
        carrier.bound_stop(0.2)
        began = time.monotonic()
        carrier.send_reliable(Stop.TYPE, STOP_PAYLOAD)
        took = time.monotonic() - began
        # It waited its limit, not the 2 s of any other send, and wrote nothing.
        assert 0.2 <= took < 1.0, f"the STOP's send took {took:.2f} s"
        assert not carrier.alive
        assert carrier.close_reason == "the socket did not take a STOP within 0.2 s"
        assert Stop.TYPE not in records_at(far)
    finally:
        carrier.close()
        far.close()


def test_a_bounded_stop_does_not_wait_out_another_send_that_is_stuck():
    carrier, far = unread_carrier()
    done: list[float] = []

    def stuck_send() -> None:
        carrier.send_reliable(
            Event.TYPE, Event(seq=2, kind="note", text="y").to_payload()
        )
        done.append(time.monotonic())

    try:
        carrier.bound_stop(0.2)
        other = threading.Thread(target=stuck_send, daemon=True)
        other.start()
        until(carrier._send_lock.locked, "the other send to hold the socket")
        began = time.monotonic()
        carrier.send_reliable(Stop.TYPE, STOP_PAYLOAD)
        took = time.monotonic() - began
        assert 0.2 <= took < 1.0, f"the STOP's send took {took:.2f} s"
        assert not carrier.alive
        assert carrier.close_reason == "a STOP waited 0.2 s behind another send"
        # The carrier's end released the stuck send too.
        other.join(WAIT_S)
        assert done and done[0] - began < 1.0
        assert Stop.TYPE not in records_at(far)
    finally:
        carrier.close()
        far.close()


def test_a_stop_without_the_bound_waits_as_any_other_send():
    # The agent program's carrier: it has no other way to write a STOP.
    carrier, far = unread_carrier(send_timeout_s=0.6)
    try:
        began = time.monotonic()
        carrier.send_reliable(Stop.TYPE, STOP_PAYLOAD)
        took = time.monotonic() - began
        assert 0.6 <= took < 2.0
        assert carrier.close_reason == "send failed: timed out"
    finally:
        carrier.close()
        far.close()


def test_a_bounded_stop_is_written_at_once_while_the_socket_takes_it(bench):
    thor = bench.robot()
    client = bench.client("console")
    client.bound_stop(0.2)
    began = time.monotonic()
    client.send_reliable(Stop.TYPE, STOP_PAYLOAD)
    assert time.monotonic() - began < 0.1 and client.alive
    assert thor.next_message() == Stop("x", "console")


def test_the_package_imports_without_isaac_capture():
    root = Path(local.__file__).resolve().parents[3]
    code = (
        "import sys\n"
        "import isaaccapture_examples.agentic_remote.link_service.service\n"
        "import isaaccapture_examples.agentic_remote.link_service.local\n"
        "import isaaccapture_examples.agentic_remote.link_service.config\n"
        "import isaaccapture_examples.agentic_remote.protocol.local_messages\n"
        "assert not [m for m in sys.modules if m.split('.')[0] == 'isaaccapture']\n"
    )
    env = {**os.environ, "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr


# The shipped service files


def test_the_shipped_service_files_keep_typed_takeover_off(tmp_path):
    """The typed fallback, a takeover asked for at the operator console, ships off."""
    demo = CONFIGS / "link_service.yaml"
    rig = CONFIGS / "link_service_rebot.yaml"
    for path in (demo, rig):
        shipped = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert shipped["operator"]["typed_takeover"] is False, path.name
    assert load_link_service_config(demo).typed_takeover is False
    # The rig file loads only with the Thor's address written in.
    text = rig.read_text(encoding="utf-8")
    assert text.count("\nallowed_peers: []\n") == 1
    written = tmp_path / rig.name
    written.write_text(
        text.replace("\nallowed_peers: []\n", "\nallowed_peers: [192.0.2.10]\n"),
        encoding="utf-8",
    )
    loaded = load_link_service_config(written)
    assert loaded.typed_takeover is False
    assert loaded.service_options()["typed_takeover"] is False
