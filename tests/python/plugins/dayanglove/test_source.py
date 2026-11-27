# SPDX-FileCopyrightText: Copyright (c) 2026 mwang259. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

from receiver import ReceivedFrame, StreamState
from source import PublisherBridge, PublisherPipe, encode_sample


class Sink:
    def __init__(self):
        self.packets = []

    def send(self, line):
        self.packets.append(json.loads(line))


def test_json_preserves_all_fields_without_float_or_uint64_rounding(make_frame):
    original = make_frame(source_timestamp=0.12345678901234567)
    arrival = time.monotonic_ns()
    packet = json.loads(
        encode_sample(ReceivedFrame(original, ("127.0.0.1", 50051), arrival))
    )
    for field in original.DESCRIPTOR.fields:
        expected = getattr(original, field.name)
        assert packet[field.name] == (
            list(expected)
            if field.name in {"mano_pose45", "joints21_xyz_m", "palm_adc"}
            else expected
        )
    assert packet["received_monotonic_ns"] == arrival
    assert packet["available"] is True


def test_per_hand_disconnect_and_sender_restart(frame_type, make_frame, monkeypatch):
    clock = 1_000_000_000
    monkeypatch.setattr(time, "monotonic_ns", lambda: clock)
    sink = Sink()
    bridge = PublisherBridge(sink)
    state = StreamState(frame_type, stale_after=1)
    peer = ("127.0.0.1", 40000)
    for side in (1, 2):
        sample = state.feed(make_frame(side, side).SerializeToString(), peer)
        bridge.update(state, sample)
    clock += 800_000_000
    sample = state.feed(make_frame(2, 3).SerializeToString(), peer)
    bridge.update(state, sample)
    clock += 300_000_000
    bridge.update(state, None)
    assert sink.packets[-1]["hand"] == 1
    assert sink.packets[-1]["available"] is False
    assert bridge.active_hands == {2}
    clock += 1_000_000_000
    sample = state.feed(make_frame(1, 1, session_id="new").SerializeToString(), peer)
    bridge.update(state, sample)
    assert sink.packets[-2]["hand"] == 2
    assert sink.packets[-2]["available"] is False
    assert sink.packets[-1]["session_id"] == "new"
    bridge.close()
    assert not bridge.active_hands
    assert sink.packets[-1]["available"] is False


def test_relocated_folder_loads_bundled_protocol_without_repository(tmp_path):
    import receiver

    folder = tmp_path / "delivery"
    shutil.copytree(
        Path(receiver.__file__).parent,
        folder,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    script = "from receiver import load_frame_type; assert load_frame_type().DESCRIPTOR.full_name == 'dayanglove.v1.GloveFrame'"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=folder,
        capture_output=True,
        check=False,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr


def test_dead_publisher_fails_and_reaps():
    pipe = PublisherPipe([sys.executable, "-c", "raise SystemExit(7)"])
    pipe.process.wait(timeout=5)
    with pytest.raises(RuntimeError, match="exited: 7"):
        pipe.send(b"{}\n")
    with pytest.raises(RuntimeError, match="exited: 7"):
        pipe.close()


def test_stalled_publisher_write_is_bounded():
    pipe = PublisherPipe([sys.executable, "-c", "import time; time.sleep(10)"])
    try:
        with pytest.raises(TimeoutError, match="stalled"):
            pipe.send(b"x" * 1_000_000)
    finally:
        pipe.process.terminate()
        pipe.process.wait(timeout=5)
        pipe.process.stdin.close()
