# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LinkCarrier: the example's three calls over a real isaaccapture.remote_link.Link.

The tests marked "link" repeat the loopback carrier's behaviour tests of
test_carrier.py on pairs of real links. What the real link does differently:

- Delivery takes time. A test waits until everything it sent has arrived, or was
  replaced on the way, before it looks at the order.
- A newest-wins link message replaces an older one of its type while that one is
  unsent, where it keeps the older one's place among the unsent, or unread at
  the peer, where it takes the newest place. Which of the two happens depends on
  timing, so the order of newest-wins link messages of different types is not
  fixed. One per type is left, the newest, and none passes a reliable link
  message sent before it.
- A reliable link message may pass a newest-wins one sent before it: the link
  sends what is queued as reliable first.
- close() lets what is queued as reliable go out; an unsent newest-wins link
  message may be dropped.
- The peer keeps only the newest unread link message of the types 96 to 127. A
  newest-wins send of another type is replaced only while it is unsent.
- There is no break_link(). A link that dies, or is closed, still hands out what
  arrived before; the loopback carrier's break_link() drops it.
- The peer of a closed link learns of it a moment later, not at once.

Most pairs are the two ends of a Unix socket pair: no TCP connection is made.
The tests of connect() make one TCP connection each, to 127.0.0.1.
"""

from __future__ import annotations

import ast
import socket
import sys
import threading
import time
import types
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from isaaccapture_examples.agentic_remote.protocol import link_carrier
from isaaccapture_examples.agentic_remote.protocol.carrier import (
    Carrier,
    receive_message,
    send,
)
from isaaccapture_examples.agentic_remote.protocol.link_carrier import (
    LATEST_TYPES,
    LinkCarrier,
)
from isaaccapture_examples.agentic_remote.protocol.messages import (
    MAX_PAYLOAD_BYTES,
    NEWEST_WINS_TYPES,
    Permit,
    ProtocolError,
    Stop,
    ViewStatus,
)

link = pytest.mark.link

# Upper bound, in real seconds, on a wait that is expected to end sooner.
WAIT_S = 10.0
JOIN_S = 10.0
# Longer than any join below: a receive that has to be woken cannot pass by timing out.
BLOCK_S = 60.0

STOP = Stop(reason="operator typed stop", sender="console")
PERMIT = Permit(
    move_id="move-1",
    session_id="s1",
    owner_number=2,
    stop_number=1,
    through_index=10,
    evidence_wall_s=1700000000.5,
    expires_mono_s=1001.0,
)


def _wait_for(predicate: Callable[[], bool], timeout_s: float = WAIT_S) -> bool:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.002)
    return True


def _settled(sender: LinkCarrier, receiver: LinkCarrier, sent: int) -> None:
    """Wait until all `sent` link messages of a fresh pair arrived or were replaced."""

    def accounted() -> int:
        there = receiver.link.stats()
        replaced_unsent = sender.link.stats().latest_replaced
        return there.received_queued + there.received_replaced + replaced_unsent

    assert _wait_for(lambda: accounted() == sent), f"{accounted()} of {sent} arrived"


def _drain(carrier: LinkCarrier) -> list[tuple[int, bytes]]:
    received = []
    while (item := carrier.receive(0)) is not None:
        received.append(item)
    return received


def _take(carrier: LinkCarrier, count: int) -> list[tuple[int, bytes]]:
    """The next `count` link messages, each waited for."""
    taken = []
    for _ in range(count):
        received = carrier.receive(WAIT_S)
        assert received is not None, f"only {len(taken)} of {count} arrived"
        taken.append(received)
    return taken


@pytest.fixture
def pair() -> Iterator[tuple[LinkCarrier, LinkCarrier]]:
    """Two LinkCarriers on the two ends of one Unix socket pair."""
    a_sock, b_sock = socket.socketpair()
    a = LinkCarrier.adopt(a_sock.detach(), name="test-a")
    b = LinkCarrier.adopt(b_sock.detach(), name="test-b")
    yield a, b
    a.close()
    b.close()


# Without Isaac Capture: the module, and the carrier over a stand-in link.


class FakeLink:
    """Records what a LinkCarrier asks of its link."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.alive = True
        self.close_reason = ""
        self.inbox: list[types.SimpleNamespace] = []

    def send_reliable(self, msg_type: int, payload: bytes) -> None:
        self.calls.append(("reliable", msg_type, payload))

    def send_latest(self, msg_type: int, payload: bytes) -> None:
        self.calls.append(("latest", msg_type, payload))

    def receive(self, timeout_ms: int) -> types.SimpleNamespace | None:
        self.calls.append(("receive", timeout_ms))
        return self.inbox.pop(0) if self.inbox else None

    def clock(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(offset_ns=-1234, two_way=True, valid=True)

    def close(self) -> None:
        self.alive = False
        self.close_reason = "closed"


def test_the_module_imports_isaaccapture_only_inside_connect_and_adopt():
    tree = ast.parse(Path(link_carrier.__file__).read_text(encoding="utf-8"))
    inside: dict[ast.AST, str] = {}
    for function in ast.walk(tree):
        if isinstance(function, ast.FunctionDef):
            for node in ast.walk(function):
                inside.setdefault(node, function.name)
    places = set()
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        if any(name.split(".")[0] == "isaaccapture" for name in names):
            places.add(inside.get(node, "the module"))
    assert places and places <= {"connect", "adopt", "_link_class"}
    assert "the module" not in places


def test_a_link_carrier_is_a_carrier():
    carrier = LinkCarrier(FakeLink())
    assert isinstance(carrier, Carrier)
    assert carrier.alive
    assert carrier.close_reason == ""


def test_the_newest_wins_types_of_the_link_are_the_protocols():
    assert LATEST_TYPES == NEWEST_WINS_TYPES == range(96, 128)


def test_sends_reach_the_link_with_their_payload_as_bytes():
    fake = FakeLink()
    carrier = LinkCarrier(fake)
    carrier.send_reliable(74, bytearray(b"result"))
    carrier.send_latest(96, memoryview(b"state"))
    assert fake.calls == [("reliable", 74, b"result"), ("latest", 96, b"state")]
    assert all(type(call[2]) is bytes for call in fake.calls)


@pytest.mark.parametrize("call", ["send_reliable", "send_latest"])
def test_a_payload_over_the_limit_is_refused_before_it_reaches_the_link(call):
    fake = FakeLink()
    carrier = LinkCarrier(fake)
    with pytest.raises(
        ProtocolError, match=r"^4194305 bytes is over the limit of 4194304 bytes$"
    ):
        getattr(carrier, call)(74, bytes(MAX_PAYLOAD_BYTES + 1))
    assert fake.calls == []
    getattr(carrier, call)(74, bytes(MAX_PAYLOAD_BYTES))
    assert len(fake.calls) == 1


@pytest.mark.parametrize("msg_type", [-1, 256, 11, 13, 14, 74.0, "74", True])
def test_a_type_number_the_link_cannot_carry_is_refused(msg_type):
    fake = FakeLink()
    carrier = LinkCarrier(fake)
    with pytest.raises(ProtocolError, match=r"is not a type number a link carries$"):
        carrier.send_reliable(msg_type, b"")
    with pytest.raises(ProtocolError, match=r"is not a type number a link carries$"):
        carrier.send_latest(msg_type, b"")
    assert fake.calls == []


@pytest.mark.parametrize(
    ("timeout_s", "timeout_ms"),
    [
        (0, 0),
        (0.0, 0),
        (-1.0, 0),  # the link would take a negative time as "no limit"
        (float("nan"), 0),
        (0.0002, 1),
        (0.001, 1),
        (0.0101, 11),
        (0.5, 500),
        (2, 2000),
        (float("inf"), 3_600_000),
        (1e12, 3_600_000),
    ],
)
def test_receive_gives_the_link_its_timeout_in_whole_milliseconds(
    timeout_s, timeout_ms
):
    fake = FakeLink()
    assert LinkCarrier(fake).receive(timeout_s) is None
    assert fake.calls == [("receive", timeout_ms)]


def test_receive_hands_out_type_number_and_payload():
    fake = FakeLink()
    fake.inbox.append(types.SimpleNamespace(type=74, payload=b"result"))
    assert LinkCarrier(fake).receive(0.01) == (74, b"result")


def test_clock_offset_close_and_the_link_itself():
    fake = FakeLink()
    carrier = LinkCarrier(fake)
    assert carrier.link is fake
    assert carrier.clock_offset() == (-1234, True, True)
    carrier.close()
    assert not carrier.alive
    assert carrier.close_reason == "closed"


class FakeLinkClass:
    """Stands in for isaaccapture.remote_link.Link: records connect() and adopt()."""

    calls: list[tuple]

    @classmethod
    def connect(cls, *args, **options) -> FakeLink:
        cls.calls.append(("connect", args, options))
        return FakeLink()

    @classmethod
    def adopt(cls, *args, **options) -> FakeLink:
        cls.calls.append(("adopt", args, options))
        return FakeLink()


@pytest.fixture
def fake_link_class(monkeypatch) -> type[FakeLinkClass]:
    FakeLinkClass.calls = []
    package = types.ModuleType("isaaccapture")
    module = types.ModuleType("isaaccapture.remote_link")
    module.Link = FakeLinkClass
    package.remote_link = module
    monkeypatch.setitem(sys.modules, "isaaccapture", package)
    monkeypatch.setitem(sys.modules, "isaaccapture.remote_link", module)
    return FakeLinkClass


def test_connect_and_adopt_build_the_link_with_the_examples_options(fake_link_class):
    dialled = LinkCarrier.connect("127.0.0.1:5000", 2.5, name="robot-app")
    adopted = LinkCarrier.adopt(7, name="link-service")
    assert isinstance(dialled, LinkCarrier) and isinstance(adopted, LinkCarrier)
    # Both ends ask for clock pings. The heartbeat period and the peer timeout
    # are not given: the link's own, 0.5 s and 3 s, apply.
    options = {"send_pings": True, "latest_types": range(96, 128)}
    assert fake_link_class.calls == [
        ("connect", ("127.0.0.1:5000", 2500), {"name": "robot-app", **options}),
        ("adopt", (7,), {"name": "link-service", **options}),
    ]


@pytest.mark.parametrize("timeout_s", [0, -1.0, float("nan"), float("inf")])
def test_connect_needs_a_time_limit(fake_link_class, timeout_s):
    with pytest.raises(ValueError, match=r"^timeout_s must be finite and above 0"):
        LinkCarrier.connect("127.0.0.1:5000", timeout_s)
    assert fake_link_class.calls == []


def test_connect_rounds_a_short_time_limit_up_to_a_millisecond(fake_link_class):
    LinkCarrier.connect("127.0.0.1:5000", 0.0001)
    assert fake_link_class.calls[0][1] == ("127.0.0.1:5000", 1)


# With Isaac Capture: pairs of real links.


@link
def test_both_ends_are_live_carriers(pair):
    a, b = pair
    assert isinstance(a, Carrier) and isinstance(b, Carrier)
    assert a.alive and b.alive
    assert a.close_reason == "" and b.close_reason == ""


@link
def test_reliable_link_messages_arrive_in_order(pair):
    a, b = pair
    sent = [(74, b"%d" % n) for n in range(50)] + [(75, b"event"), (74, b"last")]
    for msg_type, payload in sent:
        a.send_reliable(msg_type, payload)
    assert _take(b, len(sent)) == sent
    assert b.receive(0) is None
    assert a.receive(0) is None


@link
def test_each_direction_has_its_own_queue(pair):
    a, b = pair
    a.send_reliable(65, b"to b")
    b.send_reliable(74, b"to a")
    b.send_latest(96, b"state")
    assert _take(a, 2) == [(74, b"to a"), (96, b"state")]
    assert _take(b, 1) == [(65, b"to b")]
    assert a.receive(0) is None and b.receive(0) is None


@link
def test_receive_returns_none_when_nothing_arrived(pair):
    a, b = pair
    assert b.receive(0) is None
    assert b.receive(-1.0) is None
    assert b.receive(0.01) is None
    a.send_reliable(68, b"x")
    assert b.receive(WAIT_S) == (68, b"x")
    assert b.receive(0.01) is None


@link
def test_the_links_own_heartbeats_and_pings_are_never_handed_out(pair):
    a, b = pair
    # Longer than the heartbeat period of 0.5 s; pings travel from the start.
    assert b.receive(0.7) is None
    assert a.receive(0) is None
    assert b.link.stats().messages_received > 0
    assert a.alive and b.alive


@link
def test_newest_wins_replaces_an_unreceived_link_message_of_its_type(pair):
    a, b = pair
    for n in range(5):
        a.send_latest(96, b"state %d" % n)
    _settled(a, b, 5)
    assert _drain(b) == [(96, b"state 4")]


@link
def test_newest_wins_keeps_one_link_message_per_type(pair):
    a, b = pair
    a.send_latest(96, b"state 0")
    a.send_latest(97, b"permit 0")
    a.send_latest(96, b"state 1")
    a.send_latest(98, b"view 0")
    a.send_latest(97, b"permit 1")
    _settled(a, b, 5)
    # The order among the types depends on what was still unsent when replaced.
    assert sorted(_drain(b)) == [(96, b"state 1"), (97, b"permit 1"), (98, b"view 0")]


@link
def test_a_received_newest_wins_link_message_is_not_replaced(pair):
    a, b = pair
    a.send_latest(96, b"state 0")
    assert b.receive(WAIT_S) == (96, b"state 0")
    a.send_latest(96, b"state 1")
    a.send_latest(96, b"state 2")
    # One of the three was received before the count began.
    assert _wait_for(
        lambda: b.link.stats().received_replaced + a.link.stats().latest_replaced == 1
    )
    assert _take(b, 1) == [(96, b"state 2")]
    assert b.receive(0) is None


@link
def test_a_replacement_never_passes_a_reliable_link_message_sent_before_it(pair):
    a, b = pair
    a.send_latest(96, b"state 0")
    a.send_reliable(74, b"result")
    a.send_latest(96, b"state 1")
    a.send_reliable(75, b"event")
    _settled(a, b, 4)
    received = _drain(b)
    # The reliable one sent after it may pass it, unlike on the loopback link.
    assert received in (
        [(74, b"result"), (96, b"state 1"), (75, b"event")],
        [(74, b"result"), (75, b"event"), (96, b"state 1")],
    )


@link
def test_many_replacements_each_arrive_after_the_reliable_one_sent_before(pair):
    a, b = pair
    count = 200
    for n in range(count):
        a.send_reliable(74, b"%d" % n)
        a.send_latest(96, b"%d" % n)
    reliable_seen = latest_seen = -1
    while (reliable_seen, latest_seen) != (count - 1, count - 1):
        (received,) = _take(b, 1)
        index = int(received[1])
        if received[0] == 74:
            assert index == reliable_seen + 1
            reliable_seen = index
        else:
            assert latest_seen < index <= reliable_seen
            latest_seen = index
    assert b.receive(0) is None


@link
def test_reliable_link_messages_are_never_replaced(pair):
    a, b = pair
    a.send_reliable(74, b"first")
    a.send_reliable(74, b"second")
    a.send_latest(74, b"newest-wins 0")
    a.send_latest(74, b"newest-wins 1")
    a.send_reliable(74, b"third")
    received = _take(b, 4)
    if (74, b"newest-wins 0") in received:
        # Type 74 is not newest-wins at the peer: sent before the next one came,
        # the older one arrives as well.
        received += _take(b, 1)
        received.remove((74, b"newest-wins 0"))
    reliable = [(74, b"first"), (74, b"second"), (74, b"third")]
    assert [one for one in received if one in reliable] == reliable
    # After the two reliable ones sent before it; the third may pass it.
    assert received.index((74, b"newest-wins 1")) >= 2
    assert len(received) == 4
    assert b.receive(0.01) is None


@link
def test_the_payload_is_copied_when_sent(pair):
    a, b = pair
    payload = bytearray(b"abc")
    a.send_reliable(68, payload)
    payload[0] = ord("z")
    assert b.receive(WAIT_S) == (68, b"abc")


@link
def test_the_largest_payload_travels_and_a_larger_one_is_refused(pair):
    a, b = pair
    largest = bytes(range(256)) * (MAX_PAYLOAD_BYTES // 256)
    a.send_reliable(65, largest)
    with pytest.raises(ProtocolError, match=r"is over the limit of 4194304 bytes$"):
        a.send_reliable(65, largest + b"x")
    with pytest.raises(ProtocolError, match=r"is over the limit of 4194304 bytes$"):
        a.send_latest(96, largest + b"x")
    assert b.receive(WAIT_S) == (65, largest)
    assert b.receive(0.01) is None
    assert a.alive and b.alive


@link
@pytest.mark.parametrize("msg_type", [11, 13, 14, 256, -1])
def test_the_links_own_type_numbers_are_refused_and_the_link_lives(pair, msg_type):
    a, b = pair
    with pytest.raises(ProtocolError):
        a.send_reliable(msg_type, b"")
    with pytest.raises(ProtocolError):
        a.send_latest(msg_type, b"")
    a.send_reliable(74, b"still alive")
    assert b.receive(WAIT_S) == (74, b"still alive")


@link
def test_close_ends_the_link_and_the_peer_still_receives_what_was_sent_before(pair):
    a, b = pair
    a.send_latest(96, b"final state")
    a.send_reliable(74, b"final result")
    a.send_reliable(75, b"final event")
    a.close()
    assert not a.alive
    assert a.close_reason == "closed"
    # The peer learns of it a moment later.
    assert _wait_for(lambda: not b.alive)
    assert b.close_reason != ""
    a.send_reliable(74, b"too late")
    b.send_reliable(65, b"too late")
    received = _drain(b)
    # close() sends what is queued as reliable; an unsent newest-wins link
    # message may be dropped.
    assert [one for one in received if one[0] != 96] == [
        (74, b"final result"),
        (75, b"final event"),
    ]
    assert set(received) <= {
        (74, b"final result"),
        (75, b"final event"),
        (96, b"final state"),
    }
    assert _drain(a) == []
    a.close()
    b.close()
    assert not a.alive and not b.alive
    assert _drain(b) == []


@link
def test_a_closed_link_still_hands_out_what_arrived_before(pair):
    a, b = pair
    b.send_reliable(65, b"move")
    assert _wait_for(lambda: a.link.stats().received_queued == 1)
    a.close()
    # The loopback carrier drops it; the app drops it when it handles the loss.
    assert a.receive(0) == (65, b"move")
    assert a.receive(0) is None


@link
def test_a_peer_that_goes_away_ends_the_link():
    a_sock, b_sock = socket.socketpair()
    carrier = LinkCarrier.adopt(a_sock.detach(), name="test-a")
    try:
        assert carrier.alive
        b_sock.close()
        assert _wait_for(lambda: not carrier.alive)
        assert carrier.close_reason != ""
        carrier.send_reliable(74, b"ignored")
        carrier.send_latest(96, b"ignored")
        assert carrier.receive(0) is None
        assert not carrier.alive
    finally:
        carrier.close()


@link
def test_a_dead_links_receive_still_waits_out_its_timeout(pair):
    a, b = pair
    a.close()
    assert _wait_for(lambda: not b.alive)
    for end in (a, b):
        began = time.monotonic()
        assert end.receive(0.02) is None
        # A loop paced by receive() keeps its pace when the link dies.
        assert time.monotonic() - began >= 0.02


@link
def test_a_blocked_receive_gets_a_link_message_sent_from_another_thread(pair):
    a, b = pair
    received = []
    receiver = threading.Thread(
        target=lambda: received.append(b.receive(BLOCK_S)), daemon=True
    )
    receiver.start()
    a.send_reliable(74, b"done")
    receiver.join(JOIN_S)
    assert not receiver.is_alive()
    assert received == [(74, b"done")]


@link
def test_concurrent_senders_each_keep_their_order(pair):
    a, b = pair
    senders, per_sender = 4, 250

    def run(sender: int) -> None:
        for n in range(per_sender):
            a.send_reliable(64 + sender, n.to_bytes(2, "big"))

    threads = [
        threading.Thread(target=run, args=(sender,), daemon=True)
        for sender in range(senders)
    ]
    for thread in threads:
        thread.start()
    seen: dict[int, list[int]] = {64 + sender: [] for sender in range(senders)}
    for msg_type, payload in _take(b, senders * per_sender):
        seen[msg_type].append(int.from_bytes(payload, "big"))
    for thread in threads:
        thread.join(JOIN_S)
        assert not thread.is_alive()
    assert all(numbers == list(range(per_sender)) for numbers in seen.values())
    assert b.receive(0) is None


@link
def test_two_ends_can_talk_from_two_threads(pair):
    a, b = pair
    rounds = 200
    echoed = []

    def echo() -> None:
        for _ in range(rounds):
            received = b.receive(BLOCK_S)
            if received is None:
                return
            b.send_reliable(received[0] + 1, received[1])

    echoer = threading.Thread(target=echo, daemon=True)
    echoer.start()
    for n in range(rounds):
        a.send_reliable(65, b"%d" % n)
        echoed.append(a.receive(BLOCK_S))
    echoer.join(JOIN_S)
    assert not echoer.is_alive()
    assert echoed == [(66, b"%d" % n) for n in range(rounds)]


@link
def test_send_and_receive_message_carry_link_messages(pair):
    a, b = pair
    view = ViewStatus(live=False, age_s=0.9)
    send(a, STOP)
    send(a, view)
    send(b, PERMIT)
    assert [receive_message(b, WAIT_S) for _ in range(2)] == [STOP, view]
    assert receive_message(b, 0.01) is None
    assert receive_message(a, WAIT_S) == PERMIT


@link
def test_sends_on_a_dead_link_are_ignored(pair):
    a, b = pair
    b.close()
    assert _wait_for(lambda: not a.alive)
    send(a, STOP)
    send(a, PERMIT)
    send(b, PERMIT)
    assert receive_message(b, 0) is None
    assert receive_message(a, 0) is None


@link
def test_both_ends_of_a_pair_get_a_two_way_clock_estimate(pair):
    a, b = pair
    # Both ends send pings: each gets its own two-way estimate.
    for end in (a, b):
        assert _wait_for(lambda end=end: end.clock_offset()[1:] == (True, True))
    a_offset, b_offset = a.clock_offset()[0], b.clock_offset()[0]
    # One machine, one monotonic clock: local minus remote is about zero.
    assert abs(a_offset) < 50_000_000
    assert abs(b_offset) < 50_000_000


@link
def test_connect_dials_a_listener_and_adopt_takes_what_it_accepts():
    from isaaccapture.remote_link import LinkListener

    dialled: list[LinkCarrier | Exception] = []

    def dial(port: int) -> None:
        try:
            dialled.append(
                LinkCarrier.connect(f"127.0.0.1:{port}", WAIT_S, name="test-dial")
            )
        except Exception as error:  # shown below, on the test's thread
            dialled.append(error)

    with LinkListener("127.0.0.1", 0) as listener:
        dialer = threading.Thread(target=dial, args=(listener.port,), daemon=True)
        dialer.start()
        accepted = listener.accept(int(WAIT_S * 1000))
        assert accepted is not None, "nothing connected"
        fd, peer = accepted
        assert peer.startswith("127.0.0.1:")
        server = LinkCarrier.adopt(fd, name="test-accept")
    dialer.join(JOIN_S)
    assert not dialer.is_alive()
    client = dialled[0]
    try:
        assert isinstance(client, LinkCarrier), client
        send(client, STOP)
        assert receive_message(server, WAIT_S) == STOP
        # Newest wins on both ends, and a clock estimate from pings on both.
        for sender, receiver in ((client, server), (server, client)):
            for n in range(3):
                sender.send_latest(96, b"state %d" % n)
            _settled(sender, receiver, 3)
            assert _drain(receiver) == [(96, b"state 2")]
            assert _wait_for(lambda end=sender: end.clock_offset()[1:] == (True, True))
    finally:
        server.close()
        if isinstance(client, LinkCarrier):
            client.close()


@link
def test_connect_to_a_port_nothing_listens_on_raises_at_once():
    from isaaccapture.remote_link import LinkListener

    with LinkListener("127.0.0.1", 0) as listener:
        port = listener.port
    began = time.monotonic()
    # One attempt: the port was just freed, so nothing accepts on it.
    with pytest.raises(RuntimeError, match=r"^cannot connect to "):
        LinkCarrier.connect(f"127.0.0.1:{port}", WAIT_S, name="test-dial")
    assert time.monotonic() - began < WAIT_S


@link
def test_connect_refuses_a_bad_endpoint_without_dialling():
    with pytest.raises(ValueError):
        LinkCarrier.connect("no-port-here", 1.0)
