# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent program's two sources of stereo pairs: the bridge child and a folder.

No GPU and no CuPy: the bridge child runs the real bridge program on the fake
receiver's file slots.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from camera_bench import BRIDGE_CHILD, STREAMS, FakeReceiver

from isaaccapture_examples.agentic_remote.agent.camera import frame_source
from isaaccapture_examples.agentic_remote.agent.camera.frame_source import (
    BridgeSource,
    CameraError,
    RecordedPairSource,
    StereoPair,
    bgr_to_nv12,
)

RAW_PAIR = ".state/thor/captures/20261106T212519Z"
GONE = r"camera frame bridge (is gone|ended)"


class Publishing:
    """A FakeReceiver that publishes a pair every few milliseconds."""

    def __init__(self, receiver: FakeReceiver, period_s: float = 0.005) -> None:
        self.receiver = receiver
        self._period_s = period_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> FakeReceiver:
        self._thread.start()
        return self.receiver

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.receiver.publish()
            time.sleep(self._period_s)


@pytest.fixture
def receiver(tmp_path):
    fake = FakeReceiver(tmp_path)
    yield fake
    fake.stop()


@pytest.fixture
def make_source(tmp_path):
    """BridgeSource objects on the bridge program without CuPy; closed at the end."""
    script = tmp_path / "bridge_child.py"
    script.write_text(BRIDGE_CHILD, encoding="utf-8")
    made: list[BridgeSource] = []

    def make(receiver: FakeReceiver, **settings) -> BridgeSource:
        fields = {
            "python": sys.executable,
            "receiver_socket": receiver.socket_path,
            "run_dir": tmp_path / "run",
            "width": receiver.width,
            "height": receiver.height,
            "command": [sys.executable, str(script)],
            **settings,
        }
        source = BridgeSource(**fields)
        made.append(source)
        return source

    yield make
    for source in made:
        source.close()


# BridgeSource


def test_the_bridge_child_gives_verified_pairs_and_never_one_twice(
    receiver, make_source
):
    source = make_source(receiver)
    source.open()
    child = source._process
    assert source.hello["ring_depth"] == 4 and source.hello["streams"] == list(STREAMS)
    # Its own session: the terminal's Ctrl+C does not reach the bridge.
    assert os.getpgid(child.pid) != os.getpgid(0)

    # Nothing published yet: None within the time asked for.
    asked = time.monotonic()
    assert source.next_pair(0.2) is None
    assert 0.2 <= time.monotonic() - asked < 1.0
    assert source.last_none.startswith("no verified stereo pair within 0.2 s")

    seen = []
    with Publishing(receiver):
        for _ in range(6):
            before = time.monotonic()
            pair = source.next_pair(2.0)
            assert isinstance(pair, StereoPair)
            assert pair.left.shape == pair.right.shape == (72, 64)
            assert pair.left.dtype == np.uint8
            assert (pair.left == receiver.pixel(pair.seq, 0)).all()
            assert (pair.right == receiver.pixel(pair.seq, 1)).all()
            assert pair.session_id == 41 and pair.rotation == 0
            assert pair.device_index == (5, 7)
            assert pair.sof_wall_ns[1] - pair.sof_wall_ns[0] == 1000
            # The arrival stamp is this machine's monotonic clock, in the past.
            assert before - 1.0 < pair.receiver_rx_mono_s <= time.monotonic()
            check = pair.check
            assert check["source"] == "bridge" and check["ring_depth"] == 4
            assert check["copy_start_mono_ns"] <= check["copy_end_mono_ns"]
            assert pair.seq < check["confirmed_by_seq"] <= pair.seq + 3
            seen.append(pair.seq)
    assert seen == sorted(set(seen))

    # The receiver went quiet: no pair, and the source stays usable.
    assert source.next_pair(0.2) is None
    source.close()
    assert child.poll() == 0
    assert not list((source.log_path.parent).glob("*.sock"))
    source.close()
    with pytest.raises(CameraError, match="not started"):
        source.next_pair(0.1)


def test_a_pair_published_while_the_request_waits_is_handed_out(receiver, make_source):
    source = make_source(receiver)
    source.open()

    def later() -> None:
        time.sleep(0.15)
        receiver.publish()
        time.sleep(0.02)
        receiver.publish()

    thread = threading.Thread(target=later)
    thread.start()
    pair = source.next_pair(2.0)
    thread.join()
    assert pair is not None and pair.seq == 1
    assert pair.check["confirmed_by_seq"] == 2


def test_the_childs_environment_names_gpu_0_and_only_the_package(
    receiver, make_source, monkeypatch
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/somewhere")
    monkeypatch.setenv("PYTHONPATH", "/apai/src")
    env = make_source(receiver, camera_pr="/the/pr")._environment()
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert "LD_LIBRARY_PATH" not in env
    assert env["PYTHONPATH"] == str(frame_source.PACKAGE_ROOT)
    assert (frame_source.PACKAGE_ROOT / "isaaccapture_examples").is_dir()
    assert env["AGENTIC_REMOTE_CAMERA_PR"] == "/the/pr"
    # Without `command`, the bridge is the module, run by the consumer's Python.
    plain = BridgeSource(python="/venv/bin/python", receiver_socket="r", run_dir="d")
    assert plain._command == ["/venv/bin/python", "-m", frame_source.BRIDGE_MODULE]


def test_a_killed_bridge_is_a_camera_error_that_stays(receiver, make_source):
    source = make_source(receiver)
    source.open()
    with Publishing(receiver):
        assert source.next_pair(2.0) is not None
        os.kill(source._process.pid, signal.SIGKILL)
        with pytest.raises(CameraError, match=GONE):
            for _ in range(20):
                source.next_pair(0.5)
        with pytest.raises(CameraError, match=GONE):
            source.next_pair(0.5)


def test_other_cameras_than_the_calibrated_ones_are_a_camera_error(
    tmp_path, make_source
):
    receiver = FakeReceiver(tmp_path, devices=(0, 1))
    try:
        source = make_source(receiver)
        source.open()
        with Publishing(receiver), pytest.raises(CameraError) as refused:
            source.next_pair(2.0)
        text = str(refused.value)
        assert text.startswith("camera frame bridge: the stereo pair comes from Argus")
        assert "devices 0 and 1; the calibration is for 5 (left) and 7 (right)" in text
        source._process.wait(5.0)
        assert source._process.returncode == 1
        # The line is in the child's log as well.
        assert "Argus devices 0 and 1" in source.log_path.read_text()
    finally:
        receiver.stop()


def test_open_fails_with_the_bridges_own_line(tmp_path, make_source, receiver):
    # No camera receiver at the socket.
    receiver.stop()
    source = make_source(receiver)
    with pytest.raises(CameraError) as failed:
        source.open()
    assert "ended with code 1" in str(failed.value)
    assert "camera frame bridge: no camera receiver at" in str(failed.value)
    assert source._process is None

    # A receiver whose images are not the calibration's size.
    other = tmp_path / "other"
    other.mkdir()
    small = FakeReceiver(other)
    try:
        source = make_source(small, width=2560, height=1984)
        with pytest.raises(CameraError, match="the calibration needs 2560 x 1984"):
            source.open()
    finally:
        small.stop()

    # An interpreter that is not there.
    source = make_source(small, command=[str(tmp_path / "no-python")])
    with pytest.raises(CameraError, match="cannot start the camera frame bridge"):
        source.open()


def test_close_ends_a_bridge_that_was_never_asked(receiver, make_source):
    source = make_source(receiver)
    source.open()
    child = source._process
    started = time.monotonic()
    source.close()
    assert child.poll() == 0 and time.monotonic() - started < 2.0


def test_close_from_another_thread_ends_a_waiting_request(receiver, make_source):
    source = make_source(receiver)
    source.open()
    threading.Timer(0.2, source.close).start()
    started = time.monotonic()
    child = source._process
    with pytest.raises(CameraError, match="was closed"):
        source.next_pair(5.0)
    assert time.monotonic() - started < 3.0
    # The bridge saw the goodbye although it was in the middle of a request.
    assert child.wait(3.0) == 0


def test_a_pair_of_the_real_size_crosses_the_bridge(tmp_path, make_source):
    """2560 x 1984 NV12: 7.6 MB per eye through both sockets, on the CPU."""
    receiver = FakeReceiver(tmp_path, width=2560, height=1984, ring_depth=3)
    try:
        source = make_source(receiver)
        source.open()
        with Publishing(receiver, period_s=0.03):
            times = []
            for _ in range(3):
                started = time.monotonic()
                pair = source.next_pair(5.0)
                times.append(time.monotonic() - started)
                assert pair is not None and pair.left.shape == (2976, 2560)
                assert (
                    pair.left[0, 0] == pair.left[-1, -1] == receiver.pixel(pair.seq, 0)
                )
                assert (pair.right == receiver.pixel(pair.seq, 1)).all()
                copy_ms = (
                    pair.check["copy_end_mono_ns"] - pair.check["copy_start_mono_ns"]
                ) / 1e6
        print(
            f"next_pair at full size: {min(times) * 1e3:.1f} ms, copy {copy_ms:.1f} ms"
        )
    finally:
        receiver.stop()


# RecordedPairSource


def picture(width: int = 64, height: int = 48, seed: int = 0) -> np.ndarray:
    """A smooth colour image: NV12 keeps it almost unchanged."""
    yy, xx = np.mgrid[0:height, 0:width]
    blue = 40 + xx * 2 + seed * 20
    green = 60 + yy * 3
    red = 200 - xx - yy
    return np.clip(np.dstack([blue, green, red]), 0, 255).astype(np.uint8)


class Clocks:
    """A monotonic clock, a Thor wall clock and a sleep that moves both."""

    def __init__(self) -> None:
        self.mono = 500.0
        self.wall = 1_790_000_000.0
        self.slept: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.mono += seconds
        self.wall += seconds

    def settings(self) -> dict:
        return {
            "monotonic": lambda: self.mono,
            "thor_wall_now": lambda: self.wall,
            "sleep": self.sleep,
        }


@pytest.fixture
def folder(tmp_path) -> Path:
    for name, seed in (("left_raw", 0), ("right_raw", 1)):
        assert cv2.imwrite(str(tmp_path / f"{name}.png"), picture(seed=seed))
    # What the experiment's capture folders also hold: not raw pairs.
    for name in ("left_rectified", "right_rectified"):
        assert cv2.imwrite(str(tmp_path / f"{name}.png"), picture(32, 24))
    return tmp_path


def test_bgr_to_nv12_is_what_the_receivers_images_look_like():
    bgr = picture()
    nv12 = bgr_to_nv12(bgr)
    assert nv12.shape == (72, 64) and nv12.dtype == np.uint8
    back = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
    assert np.abs(back.astype(int) - bgr.astype(int)).mean() < 3.0
    # The chroma plane is interleaved U, V: a pure blue image has U above V.
    blue = np.zeros((4, 4, 3), np.uint8)
    blue[..., 0] = 255
    chroma = bgr_to_nv12(blue)[4:].reshape(-1, 2)
    assert (chroma[:, 0] > 200).all() and (chroma[:, 1] < 128).all()
    with pytest.raises(CameraError, match="even sides"):
        bgr_to_nv12(np.zeros((3, 4, 3), np.uint8))


def test_recorded_pairs_come_with_invented_times_that_grow(folder):
    clocks = Clocks()
    source = RecordedPairSource(
        folder, period_s=0.1, exposure_age_s=0.06, **clocks.settings()
    )
    with pytest.raises(CameraError, match="not open"):
        source.next_pair(1.0)
    source.open()
    with pytest.raises(CameraError, match="already open"):
        source.open()

    first = source.next_pair(1.0)
    assert isinstance(first, StereoPair)
    assert (first.session_id, first.seq) == (1, 1)
    assert first.left.shape == first.right.shape == (72, 64)
    assert not first.left.flags.writeable
    back = cv2.cvtColor(first.left, cv2.COLOR_YUV2BGR_NV12)
    assert np.abs(back.astype(int) - picture().astype(int)).mean() < 3.0
    assert not np.array_equal(first.left, first.right)
    assert first.device_index == (5, 7) and first.rotation == 0
    # Captured 0.06 s ago on the Thor's wall clock; arrived now.
    assert first.sof_wall_ns[0] == first.sof_wall_ns[1]
    assert first.sof_wall_ns[0] == pytest.approx(1_789_999_999_940_000_000, abs=1e3)
    assert first.receiver_rx_mono_s == 500.0
    assert first.check == {
        "source": "recorded",
        "files": ["left_raw.png", "right_raw.png"],
        "invented_times": True,
    }
    assert clocks.slept == []

    # The next pair is due one period later: a shorter wait gives None.
    assert source.next_pair(0.04) is None
    assert clocks.slept == [0.04]
    assert source.last_none == "the next recorded pair is due in 0.100 s"
    second = source.next_pair(1.0)
    assert clocks.slept == [0.04, pytest.approx(0.06)]
    assert second.seq == 2 and second.receiver_rx_mono_s == pytest.approx(500.1)
    assert second.sof_wall_ns[0] - first.sof_wall_ns[0] == pytest.approx(1e8, abs=1e3)

    # A Thor wall clock that steps back does not make a capture time repeat.
    clocks.wall -= 5.0
    clocks.mono += 1.0
    third = source.next_pair(1.0)
    assert third.seq == 3 and third.sof_wall_ns[0] == second.sof_wall_ns[0] + 1

    source.close()
    with pytest.raises(CameraError, match="not open"):
        source.next_pair(1.0)


def test_recorded_pairs_can_end(folder):
    for name, seed in (("left_0002", 2), ("right_0002", 3)):
        assert cv2.imwrite(str(folder / f"{name}.png"), picture(seed=seed))
    clocks = Clocks()
    source = RecordedPairSource(folder, period_s=0.0, repeat=False, **clocks.settings())
    source.open()
    files = [source.next_pair(0.1).check["files"] for _ in range(2)]
    assert files == [
        ["left_0002.png", "right_0002.png"],
        ["left_raw.png", "right_raw.png"],
    ]
    assert source.next_pair(0.1) is None and clocks.slept == [0.1]
    assert source.last_none == "the recorded pairs are used up"

    again = RecordedPairSource(folder, period_s=0.0, **clocks.settings())
    again.open()
    assert [again.next_pair(0.1).seq for _ in range(5)] == [1, 2, 3, 4, 5]


def test_a_folder_without_a_raw_pair_is_a_camera_error(tmp_path):
    with pytest.raises(CameraError, match="cannot read"):
        RecordedPairSource(tmp_path / "missing").open()
    cv2.imwrite(str(tmp_path / "left_raw.png"), picture())
    with pytest.raises(CameraError, match="no left_raw/right_raw stereo pair"):
        RecordedPairSource(tmp_path).open()
    cv2.imwrite(str(tmp_path / "right_raw.png"), picture(32, 24))
    with pytest.raises(CameraError, match="differ in size"):
        RecordedPairSource(tmp_path).open()
    (tmp_path / "right_raw.png").write_bytes(b"not an image")
    with pytest.raises(CameraError, match="cannot read"):
        RecordedPairSource(tmp_path).open()


def test_raw_nv12_files_are_served_as_they_are(tmp_path):
    nv12 = bgr_to_nv12(picture())
    nv12.tofile(tmp_path / "left_raw.nv12")
    nv12[::-1].copy().tofile(tmp_path / "right_raw.nv12")
    source = RecordedPairSource(tmp_path, size_wh=(64, 48))
    source.open()
    pair = source.next_pair(1.0)
    assert np.array_equal(pair.left, nv12) and np.array_equal(pair.right, nv12[::-1])
    with pytest.raises(CameraError, match="not a 2560 x 1984 NV12 image"):
        RecordedPairSource(tmp_path).open()


def test_both_sources_have_the_pair_source_calls():
    for kind in (BridgeSource, RecordedPairSource):
        for name in ("open", "close", "next_pair"):
            assert callable(getattr(kind, name)), (kind, name)


def test_camera_error_is_apais_own_when_apai_is_there():
    try:
        from apai.perception.camera_base import CameraError as theirs
    except ImportError:
        assert issubclass(CameraError, RuntimeError)
    else:
        assert CameraError is theirs


@pytest.mark.experiment
def test_the_experiments_raw_pair_is_served_at_full_size(experiment_dir):
    capture = experiment_dir / RAW_PAIR
    source = RecordedPairSource(capture)
    source.open()
    started = time.monotonic()
    wall = time.time()
    pair = source.next_pair(1.0)
    assert pair.left.shape == pair.right.shape == (1984 * 3 // 2, 2560)
    assert pair.check["files"] == ["left_raw.jpg", "right_raw.jpg"]
    # What unit E does with it gives back the experiment's own image.
    bgr = cv2.cvtColor(pair.left, cv2.COLOR_YUV2BGR_NV12)
    original = cv2.imread(str(capture / "left_raw.jpg"))
    assert bgr.shape == original.shape == (1984, 2560, 3)
    assert np.abs(bgr.astype(np.int16) - original).mean() < 2.0
    # Invented times: captured just before now, arrived now.
    assert wall - 0.2 < pair.sof_wall_ns[0] / 1e9 < time.time()
    assert started <= pair.receiver_rx_mono_s <= time.monotonic()
