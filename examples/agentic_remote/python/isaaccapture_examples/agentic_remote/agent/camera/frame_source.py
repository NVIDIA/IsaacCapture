# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where the agent program's stereo pairs come from.

BridgeSource starts the camera frame bridge and asks it for one verified pair at
a time. RecordedPairSource serves raw pairs from a folder with invented capture
times, for tests and dry runs.
"""

from __future__ import annotations

import contextlib
import itertools
import os
import re
import socket
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from . import bridge_wire as wire

try:
    # APAI's own error when APAI is there: CameraBase.get_frame() counts it.
    from apai.perception.camera_base import CameraError
except ImportError:

    class CameraError(RuntimeError):  # type: ignore[no-redef]
        """No stereo pair can be had; the text says why."""


BRIDGE_MODULE = "isaaccapture_examples.agentic_remote.agent.camera.frame_bridge"
CAMERA_PR_ENV = "AGENTIC_REMOTE_CAMERA_PR"
# The folder that holds isaaccapture_examples: the bridge's whole Python path.
PACKAGE_ROOT = Path(__file__).resolve().parents[4]
# Seconds past a request's own timeout the bridge has for its answer, and for the
# bytes of a pair after their record.
_REPLY_GRACE_S = 1.0
_BYTES_GRACE_S = 2.0
_RECORDED_STEM = re.compile(r"^left((?:_raw)?(?:_\d+)?)$")
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
_bridge_numbers = itertools.count(1)


@dataclass(frozen=True)
class StereoPair:
    """The left and the right camera frame of one capture, in host memory."""

    session_id: int
    seq: int
    left: np.ndarray  # uint8 NV12, shape (height * 3 // 2, width)
    right: np.ndarray
    sof_wall_ns: tuple[int, int]  # start of exposure of each eye, Thor wall clock
    device_index: tuple[int, int]
    rotation: int
    # The receiver's stamp of the pair's arrival, this machine's monotonic clock.
    receiver_rx_mono_s: float
    check: dict[str, Any]  # what the source verified


class PairSource(Protocol):
    def open(self) -> None: ...

    def close(self) -> None: ...

    def next_pair(self, timeout_s: float) -> StereoPair | None:
        """A pair newer than every earlier one, or None after `timeout_s`.

        Raises CameraError when the source cannot give pairs any more.
        """
        ...


class BridgeSource:
    """Stereo pairs from the camera receiver, through a camera frame bridge child.

    `python` is the camera consumer's interpreter (it has CuPy). One request is
    open at a time; next_pair() may be called from any one thread.
    """

    def __init__(
        self,
        *,
        python: str | os.PathLike[str],
        receiver_socket: str | os.PathLike[str],
        run_dir: str | os.PathLike[str],
        camera_pr: str | os.PathLike[str] | None = None,
        streams: tuple[str, str] = ("head.left", "head.right"),
        width: int = 2560,
        height: int = 1984,
        rotation: int = 0,
        devices: tuple[int, int] = (5, 7),
        max_skew_ms: float = 5.0,
        start_timeout_s: float = 20.0,
        command: Sequence[str] | None = None,
    ) -> None:
        self._command = (
            list(command) if command else [os.fspath(python), "-m", BRIDGE_MODULE]
        )
        self._receiver_socket = os.fspath(receiver_socket)
        self._run_dir = Path(run_dir)
        self._camera_pr = camera_pr or os.environ.get(CAMERA_PR_ENV)
        self._streams = streams
        self._size = (width, height)
        self._rotation = rotation
        self._devices = devices
        self._max_skew_ms = max_skew_ms
        self._start_timeout_s = start_timeout_s
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._sock: socket.socket | None = None
        self._reader: wire.Reader | None = None
        self._path = ""
        self._failure = ""
        self._closing = False
        self._last: tuple[int, int] | None = None  # session id and seq handed out
        self.hello: dict[str, Any] = {}
        self.last_none = ""  # why the last next_pair() gave None

    @property
    def log_path(self) -> Path:
        return self._run_dir / "frame_bridge.log"

    def open(self) -> None:
        if self._process is not None:
            raise CameraError("the camera frame bridge is already started")
        self._closing = False
        self._failure = ""
        self._run_dir.mkdir(parents=True, exist_ok=True)
        name = f"frame_bridge-{os.getpid()}-{next(_bridge_numbers)}.sock"
        try:
            self._path = wire.socket_path(str(self._run_dir), name)
        except OSError as problem:
            raise CameraError(f"no place for the bridge's socket: {problem}") from None
        argv = [
            *self._command,
            "--receiver-socket",
            self._receiver_socket,
            "--listen",
            self._path,
            "--streams",
            *self._streams,
            "--width",
            str(self._size[0]),
            "--height",
            str(self._size[1]),
            "--rotation",
            str(self._rotation),
            "--devices",
            *(str(device) for device in self._devices),
            "--max-skew-ms",
            str(self._max_skew_ms),
        ]
        try:
            with self.log_path.open("ab") as log:
                self._process = subprocess.Popen(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=self._environment(),
                    # The terminal's Ctrl+C belongs to the agent program's teardown.
                    start_new_session=True,
                )
        except OSError as problem:
            raise CameraError(
                f"cannot start the camera frame bridge ({argv[0]}): {problem}"
            ) from None
        try:
            self._attach()
        except BaseException:
            self.close()
            raise

    def next_pair(self, timeout_s: float) -> StereoPair | None:
        with self._lock:
            sock, reader = self._sock, self._reader
            if self._failure:
                raise CameraError(self._failure)
            if sock is None or reader is None:
                raise CameraError("the camera frame bridge is not started")
            session, seq = self._last or (None, None)
            try:
                wire.send(sock, wire.next_request(session, seq, timeout_s))
                deadline = time.monotonic() + timeout_s + _REPLY_GRACE_S
                record = reader.record(deadline)
                if record["type"] == wire.NONE:
                    self.last_none = record["reason"]
                    return None
                if record["type"] == wire.ERROR:
                    raise self._fail(f"camera frame bridge: {record['reason']}")
                if record["type"] != wire.PAIR:
                    raise self._fail(f"the bridge sent a {record['type']} record")
                return self._read_pair(record, reader)
            except wire.WireClosed:
                raise self._fail(self._gone()) from None
            except wire.WireTimeout:
                raise self._fail(
                    f"the camera frame bridge did not answer within "
                    f"{timeout_s + _REPLY_GRACE_S:g} s"
                ) from None
            except wire.WireError as problem:
                raise self._fail(f"bad record from the bridge: {problem}") from None

    def close(self) -> None:
        """End the bridge child; safe to call twice and from another thread."""
        self._closing = True
        sock = self._sock
        if sock is not None:
            with contextlib.suppress(wire.WireClosed, wire.WireError):
                wire.send(sock, wire.close())
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
        with self._lock:
            self._sock = None
            self._reader = None
            if sock is not None:
                sock.close()
            process, self._process = self._process, None
        if process is not None:
            for ask in (None, process.terminate, process.kill):
                if ask is not None:
                    ask()
                try:
                    process.wait(2.0)
                    break
                except subprocess.TimeoutExpired:
                    continue
        if self._path:
            with contextlib.suppress(OSError):
                os.unlink(self._path)
            self._path = ""

    def _environment(self) -> dict[str, str]:
        env = dict(os.environ)
        env.pop("LD_LIBRARY_PATH", None)
        env.pop("PYTHONHOME", None)
        env["PYTHONPATH"] = str(PACKAGE_ROOT)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = "0"
        if self._camera_pr:
            env[CAMERA_PR_ENV] = os.fspath(self._camera_pr)
        return env

    def _attach(self) -> None:
        process = self._process
        assert process is not None
        deadline = time.monotonic() + self._start_timeout_s
        sock = None
        while sock is None:
            if process.poll() is not None:
                raise CameraError(self._gone())
            if time.monotonic() > deadline:
                raise CameraError(
                    f"the camera frame bridge did not start within "
                    f"{self._start_timeout_s:g} s (log: {self.log_path})"
                )
            if os.path.exists(self._path):
                try:
                    sock = wire.dial(self._path)
                except OSError:
                    time.sleep(0.3)
            else:
                time.sleep(0.05)
        self._sock = sock
        self._reader = wire.Reader(sock)
        try:
            hello = self._reader.record(time.monotonic() + 5.0)
        except (wire.WireClosed, wire.WireTimeout, wire.WireError) as problem:
            raise CameraError(
                f"no hello from the camera frame bridge: {problem!r}"
            ) from None
        size = (hello.get("width"), hello.get("height"))
        if hello["type"] != wire.HELLO or hello.get("version") != wire.VERSION:
            raise CameraError(
                f"the camera frame bridge speaks another version: {hello}"
            )
        if size != self._size or hello.get("pixel_format") != "nv12":
            raise CameraError(
                f"the camera frame bridge serves {hello}, not NV12 {size}"
            )
        self.hello = hello

    def _read_pair(self, record: dict[str, Any], reader: wire.Reader) -> StereoPair:
        width, height = self._size
        eye_bytes = width * height * 3 // 2
        if (record["left_bytes"], record["right_bytes"]) != (eye_bytes, eye_bytes):
            raise self._fail("the bridge sent a pair of another size")
        key = (record["session_id"], record["seq"])
        if (
            self._last is not None
            and key[0] == self._last[0]
            and key[1] <= self._last[1]
        ):
            raise self._fail(f"the bridge repeated stereo pair {key}")
        deadline = time.monotonic() + _BYTES_GRACE_S
        eyes = []
        for _ in range(2):
            pixels = np.frombuffer(reader.exact(eye_bytes, deadline), dtype=np.uint8)
            eyes.append(pixels.reshape(height * 3 // 2, width))
        self._last = key
        sof, devices = record["sof_wall_ns"], record["device_index"]
        return StereoPair(
            session_id=key[0],
            seq=key[1],
            left=eyes[0],
            right=eyes[1],
            sof_wall_ns=(sof[0], sof[1]),
            device_index=(devices[0], devices[1]),
            rotation=record["rotation"],
            receiver_rx_mono_s=record["receiver_rx_mono_ns"] / 1e9,
            check={"source": "bridge", **record["check"]},
        )

    def _fail(self, text: str) -> CameraError:
        """Remember why this source is finished; every later call says the same."""
        if self._closing:
            text = "the camera frame bridge was closed"
        self._failure = text
        return CameraError(text)

    def _gone(self) -> str:
        process = self._process
        code = None if process is None else process.poll()
        how = "is gone" if code is None else f"ended with code {code}"
        return f"the camera frame bridge {how}: {self._log_tail()}"

    def _log_tail(self) -> str:
        try:
            with self.log_path.open("rb") as log:
                log.seek(max(0, log.seek(0, os.SEEK_END) - 2000))
                lines = log.read().decode("utf-8", "replace").strip().splitlines()
        except OSError:
            lines = []
        return lines[-1] if lines else f"nothing in {self.log_path}"


class RecordedPairSource:
    """Raw stereo pairs from a folder, served again and again with invented times.

    The folder holds left_raw.jpg and right_raw.jpg (or left_0001.png and
    right_0001.png and so on; .nv12 files are taken as they are). Each hand-out
    is a new pair: a new seq, a capture time `exposure_age_s` before
    `thor_wall_now()` and an arrival stamp of now. Nothing here was captured at
    those times.
    """

    def __init__(
        self,
        folder: str | os.PathLike[str],
        *,
        period_s: float = 1.0 / 7.0,
        exposure_age_s: float = 0.06,
        thor_wall_now: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        device_index: tuple[int, int] = (5, 7),
        size_wh: tuple[int, int] = (2560, 1984),
        repeat: bool = True,
        session_id: int = 1,
    ) -> None:
        self._folder = Path(folder)
        self._period_s = period_s
        self._exposure_age_s = exposure_age_s
        self._thor_wall_now = thor_wall_now
        self._monotonic = monotonic
        self._sleep = sleep
        self._device_index = device_index
        self._size = size_wh
        self._repeat = repeat
        self._session_id = session_id
        self._lock = threading.Lock()
        self._pairs: list[tuple[np.ndarray, np.ndarray, list[str]]] | None = None
        self._seq = 0
        self._due = 0.0
        self._last_sof_ns = 0
        self.last_none = ""  # why the last next_pair() gave None

    def open(self) -> None:
        with self._lock:
            if self._pairs is not None:
                raise CameraError("the recorded pairs are already open")
            pairs = []
            for left, right in _recorded_files(self._folder):
                eyes = [_read_nv12(path, self._size) for path in (left, right)]
                if eyes[0].shape != eyes[1].shape:
                    raise CameraError(f"{left.name} and {right.name} differ in size")
                pairs.append((eyes[0], eyes[1], [left.name, right.name]))
            if not pairs:
                raise CameraError(
                    f"no left_raw/right_raw stereo pair in {self._folder}"
                )
            self._pairs = pairs
            self._due = self._monotonic()

    def close(self) -> None:
        with self._lock:
            self._pairs = None

    def next_pair(self, timeout_s: float) -> StereoPair | None:
        with self._lock:
            pairs = self._pairs
            if pairs is None:
                raise CameraError("the recorded pairs are not open")
            wait = self._due - self._monotonic()
            ended = not self._repeat and self._seq >= len(pairs)
            if ended or wait > timeout_s:
                self.last_none = (
                    "the recorded pairs are used up"
                    if ended
                    else f"the next recorded pair is due in {wait:.3f} s"
                )
                self._sleep(max(0.0, timeout_s))
                return None
            if wait > 0:
                self._sleep(wait)
            now = self._monotonic()
            self._due = now + self._period_s
            left, right, names = pairs[self._seq % len(pairs)]
            self._seq += 1
            sof_ns = int((self._thor_wall_now() - self._exposure_age_s) * 1e9)
            sof_ns = self._last_sof_ns = max(sof_ns, self._last_sof_ns + 1)
            return StereoPair(
                session_id=self._session_id,
                seq=self._seq,
                left=left,
                right=right,
                sof_wall_ns=(sof_ns, sof_ns),
                device_index=self._device_index,
                rotation=0,
                receiver_rx_mono_s=now,
                check={"source": "recorded", "files": names, "invented_times": True},
            )


def _recorded_files(folder: Path) -> list[tuple[Path, Path]]:
    try:
        entries = sorted(folder.iterdir())
    except OSError as problem:
        raise CameraError(f"cannot read {folder}: {problem}") from None
    found = []
    for left in entries:
        match = _RECORDED_STEM.match(left.stem)
        if match is None or left.suffix.lower() not in (*_IMAGE_SUFFIXES, ".nv12"):
            continue
        right = left.with_name(f"right{match.group(1)}{left.suffix}")
        if right.is_file():
            found.append((left, right))
    return found


def _read_nv12(path: Path, size_wh: tuple[int, int]) -> np.ndarray:
    """The image file as a read-only NV12 array of shape (height * 3 // 2, width)."""
    if path.suffix.lower() == ".nv12":
        width, height = size_wh
        data = np.fromfile(path, dtype=np.uint8)
        if data.size != width * height * 3 // 2:
            raise CameraError(f"{path} is not a {width} x {height} NV12 image")
        nv12 = data.reshape(height * 3 // 2, width)
    else:
        import cv2

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise CameraError(f"cannot read {path}")
        nv12 = bgr_to_nv12(bgr)
    nv12.setflags(write=False)
    return nv12


def bgr_to_nv12(bgr: np.ndarray) -> np.ndarray:
    """A BGR image (even width and height) as NV12, as the camera receiver gives it."""
    import cv2

    height, width = bgr.shape[:2]
    if height % 2 or width % 2:
        raise CameraError(f"an NV12 image needs even sides, not {width} x {height}")
    i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    nv12 = np.empty_like(i420)
    nv12[:height] = i420[:height]
    quarter = (height // 2) * (width // 2)
    chroma = i420[height:].reshape(-1)
    interleaved = nv12[height:].reshape(height // 2, width // 2, 2)
    interleaved[..., 0] = chroma[:quarter].reshape(height // 2, width // 2)
    interleaved[..., 1] = chroma[quarter:].reshape(height // 2, width // 2)
    return nv12
