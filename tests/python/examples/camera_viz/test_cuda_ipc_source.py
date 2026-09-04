# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the CUDA IPC source (``type: cuda_ipc``).

Wire-format and config checks run everywhere. The end-to-end tests drive the
real producer, ``sensing_ipc_testsrc``, and skip unless it has been built and a
CUDA device is present:

    cmake --build build --target sensing_ipc_testsrc

``sensing_ipc_testsrc`` paints a 16-bit frame counter into the top-left of
every frame, so a test can tell a fresh frame from a stale or torn one by
reading pixels rather than trusting the sequence number the producer sends.
"""

from __future__ import annotations

import os
import signal
import struct
import subprocess
import time
from pathlib import Path

import pytest

from repo_paths import repo_root  # noqa: E402
from sources import build_local_camera  # noqa: E402
from sources.cuda_ipc import (  # noqa: E402
    _FRAME_FMT,
    _HELLO_FMT,
    _RELEASE_FMT,
    CudaIpcSource,
)

_REPO_ROOT = repo_root()
_TESTSRC_REL = "src/plugins/sensing/tools/sensing_ipc_testsrc"


def _find_testsrc() -> Path | None:
    override = os.environ.get("SENSING_IPC_TESTSRC")
    if override:
        p = Path(override)
        return p if p.is_file() else None
    for build_dir in (_REPO_ROOT / "build", _REPO_ROOT / "cmake-build-debug"):
        candidate = build_dir / _TESTSRC_REL
        if candidate.is_file():
            return candidate
    return None


def _cuda_available() -> bool:
    try:
        import cupy as cp
    except ImportError:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


_testsrc = _find_testsrc()

requires_producer = pytest.mark.skipif(
    _testsrc is None or not _cuda_available(),
    reason="needs cupy + a CUDA device + a built sensing_ipc_testsrc",
)


# ── wire format ───────────────────────────────────────────────────────


def test_struct_sizes_match_the_cpp_static_asserts():
    """core/cuda_ipc_protocol.hpp static_asserts these exact sizes."""
    assert struct.calcsize(_HELLO_FMT) == 56
    assert struct.calcsize(_FRAME_FMT) == 24
    assert struct.calcsize(_RELEASE_FMT) == 8


def test_magics_are_the_ascii_tags_the_producer_sends():
    from sources import cuda_ipc

    assert struct.pack("<I", cuda_ipc._HELLO_MAGIC) == b"ICUD"
    assert struct.pack("<I", cuda_ipc._FRAME_MAGIC) == b"IFRM"
    assert struct.pack("<I", cuda_ipc._RELEASE_MAGIC) == b"IREL"


# ── config plumbing ───────────────────────────────────────────────────


def test_build_local_camera_rejects_stereo():
    spec = {
        "type": "cuda_ipc",
        "name": "cam",
        "socket": "/tmp/nope.sock",
        "width": 640,
        "height": 480,
        "stereo": True,
    }
    with pytest.raises(ValueError, match="cannot be stereo"):
        build_local_camera(spec)


def test_shipped_config_parses():
    yaml = pytest.importorskip("yaml")
    cfg_path = _REPO_ROOT / "examples" / "camera_viz" / "configs" / "cuda_ipc.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    (cam,) = [c for c in cfg["cameras"] if c.get("enabled", True)]
    assert cam["type"] == "cuda_ipc"
    assert cam["socket"] and cam["width"] and cam["height"]


# ── end to end ────────────────────────────────────────────────────────


class _Producer:
    """Runs sensing_ipc_testsrc for the duration of a `with` block."""

    def __init__(self, socket_path: str, width: int, height: int, fps: int = 60):
        self._argv = [
            str(_testsrc),
            f"--socket={socket_path}",
            f"--width={width}",
            f"--height={height}",
            f"--fps={fps}",
        ]
        self._proc: subprocess.Popen | None = None

    def __enter__(self) -> "_Producer":
        self._proc = subprocess.Popen(
            self._argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(1.5)  # let it allocate + bind before anyone connects
        assert self._proc.poll() is None, "producer exited during startup"
        return self

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.send_signal(signal.SIGINT)
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def __exit__(self, *exc) -> None:
        self.stop()


def _frame_counter(image) -> int:
    """Read the producer's 16-bit counter bar: 16 cells, 24 px each."""
    import cupy as cp

    centres = cp.asnumpy(image[12, 12 : 16 * 24 : 24, 0])
    return int("".join("1" if v > 127 else "0" for v in centres), 2)


def _collect(source: CudaIpcSource, want: int, timeout_s: float = 15.0) -> list:
    counters = []
    deadline = time.monotonic() + timeout_s
    while len(counters) < want and time.monotonic() < deadline:
        frame = source.latest()
        if frame is None:
            time.sleep(0.002)
            continue
        counters.append(_frame_counter(frame.image))
    return counters


@requires_producer
def test_frames_arrive_and_always_advance(tmp_path):
    """The headline property: every frame handed out is fresh, never torn.

    A slot the producer reused while we were reading it would show a counter
    that stalls or goes backwards.
    """
    sock = str(tmp_path / "ipc.sock")
    with _Producer(sock, 640, 480) as producer:
        source = CudaIpcSource(name="cam", socket_path=sock, width=640, height=480)
        source.start()
        try:
            counters = _collect(source, 60)
        finally:
            source.stop()
        producer.stop()

    assert len(counters) >= 60, f"only {len(counters)} frames arrived"
    deltas = [b - a for a, b in zip(counters, counters[1:])]
    assert all(d > 0 for d in deltas), f"stale or torn frame: deltas={deltas}"


@requires_producer
def test_image_is_a_gpu_rgba_view_of_the_declared_size(tmp_path):
    import cupy as cp

    sock = str(tmp_path / "ipc.sock")
    with _Producer(sock, 320, 240) as producer:
        source = CudaIpcSource(name="cam", socket_path=sock, width=320, height=240)
        source.start()
        try:
            deadline = time.monotonic() + 10.0
            frame = None
            while frame is None and time.monotonic() < deadline:
                frame = source.latest()
                if frame is None:
                    time.sleep(0.002)
            assert frame is not None, "no frame within 10s"
            assert frame.image.shape == (240, 320, 4)
            assert frame.image.dtype == cp.uint8
            assert hasattr(frame.image, "__cuda_array_interface__")
            assert frame.source_id == "cam"
            assert frame.timestamp_ns > 0
        finally:
            source.stop()
        producer.stop()


@requires_producer
def test_waits_for_a_late_producer_then_recovers_from_its_restart(tmp_path):
    sock = str(tmp_path / "ipc.sock")
    source = CudaIpcSource(name="cam", socket_path=sock, width=320, height=240)
    source.start()
    try:
        # Started before any producer exists: must idle, not raise.
        time.sleep(1.0)
        assert source.latest() is None

        with _Producer(sock, 320, 240):
            assert len(_collect(source, 10)) >= 10

        # Producer gone; the source should be idle again rather than serving
        # pixels out of an allocation that no longer has an owner.
        time.sleep(1.0)

        with _Producer(sock, 320, 240):
            assert len(_collect(source, 10)) >= 10, "did not recover"
    finally:
        source.stop()


@requires_producer
def test_geometry_mismatch_is_refused(tmp_path):
    """A config that disagrees with the producer must yield no frames rather
    than reinterpreting its bytes at the wrong stride."""
    sock = str(tmp_path / "ipc.sock")
    with _Producer(sock, 320, 240):
        source = CudaIpcSource(name="cam", socket_path=sock, width=640, height=480)
        source.start()
        try:
            time.sleep(3.0)
            assert source.latest() is None
        finally:
            source.stop()
