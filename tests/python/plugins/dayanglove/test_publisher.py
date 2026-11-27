# SPDX-FileCopyrightText: Copyright (c) 2026 mwang259. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
import subprocess
import time

import pytest

from receiver import ReceivedFrame
from source import encode_line, encode_sample


@pytest.fixture(scope="module")
def publisher():
    binary = os.environ.get("DAYANGLOVE_PUBLISHER")
    if not binary or not Path(binary).is_file():
        pytest.fail(
            "Build dayanglove_plugin and set DAYANGLOVE_PUBLISHER before running test_publisher.py"
        )
    return binary


def run(publisher, data):
    return subprocess.run(
        [publisher, "--validate-only", "--roundtrip-json"],
        input=data,
        capture_output=True,
        check=False,
        timeout=5,
    )


@pytest.mark.parametrize("side", [1, 2])
@pytest.mark.parametrize("pose_valid", [True, False])
def test_native_schema_roundtrip_exact(publisher, make_frame, side, pose_valid):
    changes = (
        {} if pose_valid else dict(pose_valid=False, mano_pose45=[], joints21_xyz_m=[])
    )
    original = encode_sample(
        ReceivedFrame(
            make_frame(side, **changes), ("127.0.0.1", 1), time.monotonic_ns()
        )
    )
    result = run(publisher, original)
    assert result.returncode == 0, result.stderr.decode()
    assert json.loads(result.stdout) == json.loads(original)


@pytest.mark.parametrize(
    "changes",
    [
        dict(hand=3),
        dict(palm_adc=[4096] * 210),
        dict(mano_pose45=[0]),
        dict(mirror_x=False),
        dict(available=True, received_monotonic_ns=0),
    ],
)
def test_native_rejects_invalid_contract(publisher, make_frame, changes):
    packet = json.loads(
        encode_sample(
            ReceivedFrame(make_frame(), ("127.0.0.1", 1), time.monotonic_ns())
        )
    )
    packet.update(changes)
    assert run(publisher, encode_line(packet)).returncode != 0


@pytest.mark.parametrize("payload", [b"{bad}\n", b"x" * 16385, b'{"hand":1}', b"{}\n"])
def test_native_rejects_malformed_and_truncated_ipc(publisher, payload):
    assert run(publisher, payload).returncode != 0


def test_native_ordering_allows_device_counter_reset_and_new_session(
    publisher, make_frame
):
    now = time.monotonic_ns()
    packets = [
        encode_sample(ReceivedFrame(make_frame(sequence=10), ("127.0.0.1", 1), now)),
        encode_sample(
            ReceivedFrame(
                make_frame(sequence=11, frame_number=1), ("127.0.0.1", 1), now + 1
            )
        ),
        encode_sample(
            ReceivedFrame(
                make_frame(sequence=1, session_id="new"), ("127.0.0.1", 1), now + 2
            )
        ),
    ]
    result = run(publisher, b"".join(packets))
    assert result.returncode == 0, result.stderr.decode()
    assert len(result.stdout.splitlines()) == 3
    assert run(publisher, packets[0] + packets[0]).returncode != 0


def test_native_lifecycle_accepts_new_peer_with_same_session_id(publisher, make_frame):
    now = time.monotonic_ns()
    old = encode_sample(ReceivedFrame(make_frame(sequence=10), ("127.0.0.1", 1), now))
    offline = encode_line(dict(hand=1, available=False, received_monotonic_ns=now + 1))
    new = encode_sample(
        ReceivedFrame(make_frame(sequence=1), ("127.0.0.1", 2), now + 2)
    )
    result = run(publisher, old + offline + new)
    assert result.returncode == 0, result.stderr.decode()
