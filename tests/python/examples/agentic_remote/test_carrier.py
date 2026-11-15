# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The loopback carrier and the send / receive_message helpers."""

from __future__ import annotations

import logging
import sys
import threading
import time
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

from isaaccapture_examples.agentic_remote.protocol.carrier import (
    Carrier,
    LoopbackCarrier,
    loopback_pair,
    receive_message,
    send,
)
from isaaccapture_examples.agentic_remote.protocol.messages import (
    Event,
    Permit,
    ProtocolError,
    Result,
    Stop,
    ViewStatus,
)

LOGGER = "isaaccapture.examples.agentic_remote.carrier"
# Longer than any join below: a receive that has to be woken cannot pass by timing out.
BLOCK_S = 60.0
JOIN_S = 10.0

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


class RecordingCarrier:
    """A carrier that records which send call each link message used."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, bytes]] = []
        self.alive = True

    def send_reliable(self, msg_type: int, payload: bytes) -> None:
        self.calls.append(("reliable", msg_type, payload))

    def send_latest(self, msg_type: int, payload: bytes) -> None:
        self.calls.append(("latest", msg_type, payload))

    def receive(self, timeout_s: float) -> tuple[int, bytes] | None:
        return None

    def close(self) -> None:
        self.alive = False


def _drain(carrier: LoopbackCarrier) -> list[tuple[int, bytes]]:
    received = []
    while (item := carrier.receive(0)) is not None:
        received.append(item)
    return received


def test_loopback_ends_are_carriers():
    a, b = loopback_pair()
    assert isinstance(a, LoopbackCarrier)
    assert isinstance(b, LoopbackCarrier)
    assert isinstance(a, Carrier)
    assert isinstance(RecordingCarrier(), Carrier)
    assert a.alive
    assert b.alive


def test_carrier_pair_fixture_gives_both_ends_of_one_link(request):
    try:
        robot_end, workstation_end = request.getfixturevalue("carrier_pair")
    except pytest.FixtureLookupError:
        pytest.skip("the carrier_pair fixture is in conftest.py")
    assert robot_end.alive
    assert workstation_end.alive
    send(workstation_end, STOP)
    send(robot_end, PERMIT)
    assert receive_message(robot_end, 0) == STOP
    assert receive_message(workstation_end, 0) == PERMIT


def test_reliable_link_messages_arrive_in_order():
    a, b = loopback_pair()
    sent = [(74, b"%d" % n) for n in range(50)] + [(75, b"event"), (74, b"last")]
    for msg_type, payload in sent:
        a.send_reliable(msg_type, payload)
    assert _drain(b) == sent
    assert _drain(a) == []


def test_each_direction_has_its_own_queue():
    a, b = loopback_pair()
    a.send_reliable(65, b"to b")
    b.send_reliable(74, b"to a")
    b.send_latest(96, b"state")
    assert _drain(a) == [(74, b"to a"), (96, b"state")]
    assert _drain(b) == [(65, b"to b")]


def test_receive_returns_none_when_nothing_arrived():
    a, b = loopback_pair()
    assert b.receive(0) is None
    assert b.receive(-1.0) is None
    assert b.receive(0.01) is None
    a.send_reliable(68, b"x")
    assert b.receive(0.01) == (68, b"x")
    assert b.receive(0.01) is None


def test_newest_wins_replaces_an_unreceived_link_message_of_its_type():
    a, b = loopback_pair()
    for n in range(5):
        a.send_latest(96, b"state %d" % n)
    assert _drain(b) == [(96, b"state 4")]


def test_newest_wins_keeps_one_link_message_per_type():
    a, b = loopback_pair()
    a.send_latest(96, b"state 0")
    a.send_latest(97, b"permit 0")
    a.send_latest(96, b"state 1")
    a.send_latest(98, b"view 0")
    a.send_latest(97, b"permit 1")
    assert _drain(b) == [(96, b"state 1"), (98, b"view 0"), (97, b"permit 1")]


def test_a_received_newest_wins_link_message_is_not_replaced():
    a, b = loopback_pair()
    a.send_latest(96, b"state 0")
    assert b.receive(0) == (96, b"state 0")
    a.send_latest(96, b"state 1")
    a.send_latest(96, b"state 2")
    assert _drain(b) == [(96, b"state 2")]


def test_a_replacement_never_passes_a_reliable_link_message_sent_before_it():
    a, b = loopback_pair()
    a.send_latest(96, b"state 0")
    a.send_reliable(74, b"result")
    a.send_latest(96, b"state 1")
    a.send_reliable(75, b"event")
    assert _drain(b) == [(74, b"result"), (96, b"state 1"), (75, b"event")]


def test_reliable_link_messages_are_never_replaced():
    a, b = loopback_pair()
    a.send_reliable(74, b"first")
    a.send_reliable(74, b"second")
    a.send_latest(74, b"newest-wins 0")
    a.send_latest(74, b"newest-wins 1")
    a.send_reliable(74, b"third")
    assert _drain(b) == [
        (74, b"first"),
        (74, b"second"),
        (74, b"newest-wins 1"),
        (74, b"third"),
    ]


def test_the_payload_is_copied_when_sent():
    a, b = loopback_pair()
    payload = bytearray(b"abc")
    a.send_reliable(68, payload)
    payload[0] = ord("z")
    assert b.receive(0) == (68, b"abc")


def test_break_link_ends_both_ends_and_drops_what_was_under_way():
    a, b = loopback_pair()
    a.send_reliable(74, b"result")
    a.send_latest(96, b"state")
    b.send_reliable(65, b"move")
    b.break_link()
    assert not a.alive
    assert not b.alive
    for end in (a, b):
        assert end.receive(0) is None
        assert end.receive(0.01) is None


@pytest.mark.parametrize("end_link", ["break_link", "close"])
def test_a_dead_links_receive_still_waits_out_its_timeout(end_link):
    a, b = loopback_pair()
    getattr(a, end_link)()
    for end in (a, b):
        began = time.monotonic()
        assert end.receive(0.02) is None
        # A loop paced by receive() keeps its pace when the link dies.
        assert time.monotonic() - began >= 0.02


def test_sends_on_a_dead_link_are_ignored():
    a, b = loopback_pair()
    a.break_link()
    a.send_reliable(74, b"result")
    a.send_latest(96, b"state")
    b.send_reliable(65, b"move")
    b.send_latest(97, b"permit")
    assert _drain(a) == []
    assert _drain(b) == []
    a.break_link()
    a.close()
    b.close()
    assert not a.alive
    assert not b.alive


def test_close_ends_the_link_and_the_peer_still_receives_what_was_sent_before():
    a, b = loopback_pair()
    a.send_reliable(74, b"final result")
    a.send_latest(96, b"final state")
    b.send_reliable(65, b"move")
    a.close()
    assert not a.alive
    assert not b.alive
    assert a.receive(0) is None
    a.send_reliable(74, b"too late")
    b.send_reliable(65, b"too late")
    assert _drain(b) == [(74, b"final result"), (96, b"final state")]
    assert _drain(a) == []
    a.close()
    b.close()
    assert _drain(b) == []


def test_a_blocked_receive_gets_a_link_message_sent_from_another_thread():
    a, b = loopback_pair()
    received = []
    receiver = threading.Thread(
        target=lambda: received.append(b.receive(BLOCK_S)), daemon=True
    )
    receiver.start()
    a.send_reliable(74, b"done")
    receiver.join(JOIN_S)
    assert not receiver.is_alive()
    assert received == [(74, b"done")]


def test_a_blocked_receive_gets_a_newest_wins_link_message():
    a, b = loopback_pair()
    started = threading.Event()

    def publish() -> None:
        started.wait(JOIN_S)
        a.send_latest(96, b"state")

    publisher = threading.Thread(target=publish, daemon=True)
    publisher.start()
    started.set()
    assert b.receive(BLOCK_S) == (96, b"state")
    publisher.join(JOIN_S)
    assert not publisher.is_alive()


def test_concurrent_senders_each_keep_their_order():
    a, b = loopback_pair()
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
    for _ in range(senders * per_sender):
        received = b.receive(BLOCK_S)
        assert received is not None
        seen[received[0]].append(int.from_bytes(received[1], "big"))
    for thread in threads:
        thread.join(JOIN_S)
        assert not thread.is_alive()
    assert all(numbers == list(range(per_sender)) for numbers in seen.values())
    assert b.receive(0) is None


def test_concurrent_newest_wins_senders_leave_one_link_message():
    a, b = loopback_pair()

    def run(sender: int) -> None:
        for n in range(250):
            a.send_latest(96, b"%d:%d" % (sender, n))

    threads = [
        threading.Thread(target=run, args=(sender,), daemon=True) for sender in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(JOIN_S)
        assert not thread.is_alive()
    (remaining,) = _drain(b)
    assert remaining[0] == 96
    assert remaining[1].endswith(b":249")


def test_two_ends_can_talk_from_two_threads():
    a, b = loopback_pair()
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


def test_send_picks_the_delivery_from_the_type_number():
    carrier = RecordingCarrier()
    view = ViewStatus(live=True, age_s=0.1)
    result = Result(command_id="move-1", status="accepted")
    event = Event(seq=1, kind="help_needed", text="agent failed")
    for message in (STOP, PERMIT, result, view, event):
        send(carrier, message)
    assert carrier.calls == [
        ("reliable", Stop.TYPE, STOP.to_payload()),
        ("latest", Permit.TYPE, PERMIT.to_payload()),
        ("reliable", Result.TYPE, result.to_payload()),
        ("latest", ViewStatus.TYPE, view.to_payload()),
        ("reliable", Event.TYPE, event.to_payload()),
    ]


def test_send_and_receive_message_carry_link_messages():
    a, b = loopback_pair()
    older = ViewStatus(live=True, age_s=0.1)
    newer = ViewStatus(live=False, age_s=0.9)
    send(a, STOP)
    send(a, older)
    send(a, STOP)
    send(a, newer)
    send(b, PERMIT)
    assert [receive_message(b, 0) for _ in range(4)] == [STOP, STOP, newer, None]
    assert receive_message(a, 0.01) == PERMIT
    assert receive_message(a, 0.01) is None


def test_send_on_a_dead_carrier_is_ignored():
    a, b = loopback_pair()
    b.break_link()
    send(a, STOP)
    send(a, PERMIT)
    assert receive_message(b, 0) is None
    assert receive_message(a, 0) is None


def test_send_raises_for_a_link_message_that_cannot_be_encoded():
    a, b = loopback_pair()
    with pytest.raises(ProtocolError, match=r"^ViewStatus\.age_s: non-finite number$"):
        send(a, ViewStatus(live=True, age_s=float("nan")))
    assert b.receive(0) is None


def test_receive_message_drops_a_link_message_that_does_not_decode(caplog):
    a, b = loopback_pair()
    a.send_reliable(Result.TYPE, b"not json")
    a.send_reliable(1, b"{}")
    a.send_latest(ViewStatus.TYPE, b'{"live":true}')
    send(a, PERMIT)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert [receive_message(b, 0) for _ in range(4)] == [None, None, None, PERMIT]
    assert [record.name for record in caplog.records] == [LOGGER] * 3
    assert [record.levelno for record in caplog.records] == [logging.WARNING] * 3
    dropped = [record.getMessage() for record in caplog.records]
    assert "type 74" in dropped[0]
    assert "bad JSON" in dropped[0]
    assert "unknown link message type 1" in dropped[1]
    assert "missing field 'age_s'" in dropped[2]


@pytest.mark.parametrize(
    "payload",
    [b"", b"not json", b"[]", b'{"reason":"x"}', b'{"reason":5,"sender":"console"}'],
)
def test_a_stop_with_a_bad_payload_is_still_a_stop(payload, caplog):
    a, b = loopback_pair()
    a.send_reliable(Stop.TYPE, payload)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        received = receive_message(b, 0)
    assert received == Stop(reason="STOP with a bad payload", sender="unknown")
    assert [record.name for record in caplog.records] == [LOGGER]
    assert receive_message(b, 0) is None
