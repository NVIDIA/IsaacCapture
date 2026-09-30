# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wire correctness, bounded buffering and command freshness contracts."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from remote_assistance.bridge import Outbox, sonic_stop_message
from remote_assistance.connection import token_for
from remote_assistance.data import (
    CHUNK_BYTES,
    HEADER,
    MAX_MESSAGE,
    DataSession,
    Reassembler,
    fragments,
)


def test_fragmented_message_reorders_and_deduplicates():
    receiver = Reassembler()
    lease = receiver.issue_lease()
    payload = bytes(range(256)) * 256
    packets = list(fragments(payload, lease, 10))
    assert max(map(len, packets)) <= 1300
    for packet in packets[:0:-1]:
        assert receiver.accept("pose", packet) is None
        assert receiver.accept("pose", packet) is None
    assert receiver.accept("pose", packets[0]) == payload
    assert receiver.accept("pose", packets[0]) is None
    assert receiver.pending == {}


def test_partial_message_expires_and_old_sequence_cannot_overwrite():
    now = [0.0]
    receiver = Reassembler(clock=lambda: now[0])
    lease = receiver.issue_lease()
    packets = list(fragments(b"x" * 5000, lease, 1))
    assert receiver.accept("pose", packets[0]) is None
    now[0] = 0.31
    assert receiver.accept("pose", packets[1]) is None
    new_lease = receiver.issue_lease()
    assert receiver.pending == {}
    assert receiver.accept("pose", next(fragments(b"new", new_lease, 5))) == b"new"
    assert receiver.accept("pose", next(fragments(b"old", new_lease, 4))) is None
    receiver.reset()
    assert (
        receiver.accept("pose", next(fragments(b"old connection", new_lease, 6)))
        is None
    )


def test_malformed_packets_and_memory_bounds():
    receiver = Reassembler()
    lease = receiver.issue_lease()
    for packet in (
        b"",
        b"bad",
        HEADER.pack(b"RA01", lease, 1, 0, 65535, MAX_MESSAGE),
        HEADER.pack(b"RA01", lease, 1, 0, 1, MAX_MESSAGE + 1),
    ):
        assert receiver.accept("pose", packet) is None
    for seq in range(100):
        packet = next(fragments(b"x" * (CHUNK_BYTES + 1), lease, seq))
        assert receiver.accept("pose", packet) is None
    assert len(receiver.pending) <= 32
    with pytest.raises(ValueError):
        list(fragments(b"x" * (MAX_MESSAGE + 1), lease, 1))


def test_untrusted_participant_and_topic_are_rejected():
    async def scenario():
        session = DataSession(
            {},
            "robot",
            "edge",
            {"ra/v1/pose"},
            lambda *_: pytest.fail("Must not accept packet"),
        )
        for peer, topic in (("stranger", "ra/v1/pose"), ("edge", "ra/v1/unknown")):
            session._received(
                SimpleNamespace(
                    participant=SimpleNamespace(identity=peer), topic=topic, data=b"bad"
                )
            )
        assert session.stats["wrong_peer"] == 1
        assert session.stats["wrong_topic"] == 1

    asyncio.run(scenario())


def test_outbox_keeps_latest_state_but_orders_commands_and_expires():
    box = Outbox(0.3)
    state = {"topic": "pose", "latest": True}
    command = {"topic": "command", "latest": False}
    box.put(state, b"old", 0)
    box.put(state, b"fresh", 0.1)
    box.put(command, b"start", 0.1)
    box.put(command, b"stop", 0.2)
    assert box.pop(0.2)[1] == b"fresh"
    assert box.pop(0.2)[1] == b"start"
    assert box.pop(0.2)[1] == b"stop"
    box.put(command, b"expired start", 0.2)
    assert box.pop(0.6) is None
    for _ in range(32):
        box.put(command, b"event", 1)
    with pytest.raises(RuntimeError):
        box.put(command, b"overflow", 1)


def test_stop_wire_schema_and_development_credential_scope():
    message = sonic_stop_message()
    assert message[:7] == b"command"
    header = json.loads(message[7:1287].rstrip(b"\0"))
    assert [field["name"] for field in header["fields"]] == ["start", "stop", "planner"]
    assert message[1287:] == bytes((0, 1, 0))
    with pytest.raises(ValueError, match="loopback"):
        token_for(
            {"development": True, "url": "ws://remote.example:7880", "room": "r"},
            "robot",
        )


def test_mode_switch_is_a_state_coalescing_boundary():
    box = Outbox(0.3)
    state = {"topic": "pose", "latest": True}
    command = {"topic": "command", "latest": False}
    box.put(state, b"old target", 0)
    box.put(state, b"target before switch", 0.01)
    box.put(command, b"switch", 0.02)
    box.put(state, b"target after switch", 0.03)
    assert [box.pop(0.04)[1] for _ in range(3)] == [
        b"target before switch",
        b"switch",
        b"target after switch",
    ]
    assert box.pop(0.04) is None
    box.clear()
    box.put(state, b"new session", 1)
    assert box.pop(1)[1] == b"new session"
