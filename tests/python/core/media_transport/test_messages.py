# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from isaac_media_transport.message_protocol import (
    Reassembler,
    fragments,
    HEADER,
    CHUNK_BYTES,
    MAX_MESSAGE,
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
