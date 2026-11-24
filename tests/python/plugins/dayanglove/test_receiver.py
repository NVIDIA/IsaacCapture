# SPDX-FileCopyrightText: Copyright (c) 2026 mwang259. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import socket
import time

import pytest

from receiver import StreamState, UdpReceiver, decode_frame, main


@pytest.mark.parametrize("side", [1, 2])
def test_preserves_every_wire_field_and_unknown_fields(frame_type, make_frame, side):
    original = make_frame(side, sequence=257)
    # Future field 100, varint 7: unknown additive fields must survive decoding.
    payload = original.SerializeToString() + b"\xa0\x06\x07"
    decoded = decode_frame(payload, frame_type)
    assert decoded.SerializeToString() == payload
    for field in frame_type.DESCRIPTOR.fields:
        assert getattr(decoded, field.name) == getattr(original, field.name)
    assert decoded.device_sequence == 1
    assert decoded.received_unix_ns == 1793440000123456789


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": 2},
        {"hand": 0},
        {"hand": 3},
        {"mirror_x": False},
        {"glove_id": ""},
        {"session_id": ""},
        {"output_sequence": 0},
        {"frame_number": 0},
        {"status": 256},
        {"device_sequence": 256},
        {"palm_adc": [0] * 209},
        {"palm_adc": [4096] * 210},
        {"mano_pose45": [0] * 44},
        {"joints21_xyz_m": [0] * 64},
        {"mano_pose45": [float("nan")] * 45},
        {"joints21_xyz_m": [float("inf")] * 63},
        {"source_timestamp": float("nan")},
        {"pose_valid": False},
    ],
)
def test_rejects_invalid_contract(frame_type, make_frame, changes):
    with pytest.raises(ValueError):
        decode_frame(make_frame(**changes).SerializeToString(), frame_type)


@pytest.mark.parametrize("payload", [b"", b"\xff", b"\x12\x05ab", b"x" * 1401])
def test_rejects_malformed_or_oversize(frame_type, payload):
    with pytest.raises(ValueError):
        decode_frame(payload, frame_type)


def test_tactile_only_replaces_old_valid_pose(frame_type, make_frame):
    state = StreamState(frame_type)
    peer = ("127.0.0.1", 40000)
    state.feed(make_frame().SerializeToString(), peer, 0)
    invalid = make_frame(
        sequence=2, pose_valid=False, mano_pose45=[], joints21_xyz_m=[]
    )
    sample = state.feed(invalid.SerializeToString(), peer, 1)
    assert state.latest[1] is sample
    assert not sample.frame.pose_valid
    assert not sample.frame.mano_pose45
    assert len(sample.frame.palm_adc) == 210


def test_per_hand_ordering_allows_cross_hand_reordering(frame_type, make_frame):
    state = StreamState(frame_type)
    peer = ("127.0.0.1", 40000)
    for side, sequence in [(1, 1), (1, 3), (2, 2), (2, 4)]:
        assert state.feed(
            make_frame(side, sequence).SerializeToString(), peer, sequence
        )
    for sequence in (3, 1):
        assert state.feed(make_frame(1, sequence).SerializeToString(), peer, 10) is None
    assert state.hand_counts == {1: 2, 2: 2}
    assert state.counts["out_of_order_or_duplicate"] == 2
    # A jump observed before reordered packet 2 arrived is not proven packet loss.
    assert state.counts["sequence_jumps"] == 1
    assert state.latest[1].frame.output_sequence == 3


def test_one_hand_expires_while_other_continues(frame_type, make_frame):
    state = StreamState(frame_type, stale_after=1)
    peer = ("127.0.0.1", 40000)
    state.feed(make_frame(1, 1).SerializeToString(), peer, 0)
    state.feed(make_frame(2, 2).SerializeToString(), peer, 900_000_000)
    assert state.expire(1_000_000_000) == [1]
    assert set(state.latest) == {2}
    # Repeated old samples cannot make the expired hand live again.
    assert state.feed(make_frame(1, 1).SerializeToString(), peer, 1_100_000_000) is None
    # frame_number may restart after a device reconnect within a sender session.
    assert state.feed(
        make_frame(1, 3, frame_number=1).SerializeToString(), peer, 1_200_000_000
    )
    assert set(state.latest) == {1, 2}


def test_restart_waits_for_silence_and_rejects_retired_session(frame_type, make_frame):
    state = StreamState(frame_type, stale_after=1)
    old_peer, new_peer = ("127.0.0.1", 40000), ("127.0.0.1", 40001)
    old = make_frame(sequence=20).SerializeToString()
    state.feed(old, old_peer, 0)
    new = make_frame(sequence=1, session_id="session-b").SerializeToString()
    assert state.feed(new, new_peer, 500_000_000) is None
    assert state.feed(new, new_peer, 1_000_000_000)
    assert state.latest[1].frame.output_sequence == 1
    assert state.feed(old, old_peer, 2_000_000_000) is None
    assert state.active[1] == "session-b"
    assert state.counts["session_changes"] == 1
    assert state.counts["retired_session"] == 1


def test_invalid_packets_do_not_refresh_freshness_or_take_over(frame_type, make_frame):
    state = StreamState(frame_type, stale_after=1)
    peer = ("127.0.0.1", 40000)
    state.feed(make_frame().SerializeToString(), peer, 0)
    assert state.feed(b"\xff", peer, 1_000_000_000) is None
    assert not state.latest
    assert state.last_receive_ns == 0
    assert state.active == (peer, "session-a")
    assert state.counts["invalid"] == 1


def test_real_udp_truncation_recovery_and_socket_cleanup(frame_type, make_frame):
    with UdpReceiver(frame_type, port=0, stale_after=0.03) as receiver:
        address = receiver.address
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            sender.sendto(b"x" * 3000, address)
            assert receiver.poll() is None
            for side in (1, 2):
                message = make_frame(side, side)
                sender.sendto(message.SerializeToString(), address)
                sample = receiver.poll()
                assert sample.frame == message
                assert sample.received_monotonic_ns <= time.monotonic_ns()
            assert receiver.state.counts["invalid"] == 1
            assert receiver.poll(timeout=0.06) is None
            assert not receiver.state.latest
    with UdpReceiver(frame_type, port=address[1]):
        pass


def test_cli_no_data_fails_explicitly(frame_type, request, tmp_path):
    protocol = request.config.getoption("--protocol-python")
    summary = tmp_path / "summary.json"
    assert (
        main(
            [
                "--protocol-python",
                protocol,
                "--port",
                "0",
                "--duration",
                "0.01",
                "--summary-file",
                str(summary),
            ]
        )
        == 2
    )
    assert summary.is_file()
