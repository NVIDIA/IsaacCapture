# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""python -m isaaccapture_examples.agentic_remote.agent.camera.frame_bridge.

The camera frame bridge: a child of the agent program that runs in the camera
consumer's Python environment. It copies one stereo pair at a time out of the
camera receiver's GPU slots, proves that no slot was written again during the
copy, and hands the bytes to the agent program (records: bridge_wire.py).

The proof. D is the ring depth and S the pair's seq. The receiver writes the
slot of S again only after the (D - 1)-th publish after S, and seq grows by at
least one per publish. So a later notification with S < seq <= S + D - 1 whose
timestamp is after the end of the copy shows that the copy was done first.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import select
import signal
import socket
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import bridge_wire as wire

EXIT_OK = 0
EXIT_FAILED = 1  # the receiver, the stream or the agent program's socket failed
EXIT_NOT_RUN = 2  # the arguments or the installation cannot be used

PROGRAM = "camera frame bridge"
CAMERA_PR_ENV = "AGENTIC_REMOTE_CAMERA_PR"
# The one clock domain whose capture times are the Thor's wall clock.
WALL_CLOCK = "realtime"
_RECV_BYTES = 65536
_MAX_BACKLOG_BYTES = 8 * 1024 * 1024


class StreamRefused(Exception):
    """The receiver's stream is not the one the calibration belongs to."""


@dataclass(frozen=True)
class StreamRules:
    """What the bridge accepts; the defaults are this rig's calibrated head pair."""

    streams: tuple[str, str] = ("head.left", "head.right")
    width: int = 2560
    height: int = 1984
    pixel_format: str = "nv12"
    rotation: int = 0
    devices: tuple[int, int] = (5, 7)  # Argus device of the left and the right eye
    max_skew_ns: int = 5_000_000
    confirm_timeout_s: float = 0.25

    @property
    def eye_bytes(self) -> int:
        return self.width * self.height * 3 // 2


def check_catalog(subscribed: Mapping[str, Any], rules: StreamRules) -> tuple[int, int]:
    """The (ring depth, GPU id) of the two streams; StreamRefused if they do not fit."""
    listed = subscribed.get("streams")
    by_name = {
        entry.get("name"): entry
        for entry in (listed if isinstance(listed, list) else [])
        if isinstance(entry, dict)
    }
    depths, gpus = set(), set()
    for name in rules.streams:
        entry = by_name.get(name)
        if entry is None:
            raise StreamRefused(f"the camera receiver has no stream {name}")
        got = (entry.get("width"), entry.get("height"), entry.get("pixel_format"))
        want = (rules.width, rules.height, rules.pixel_format)
        if got != want:
            raise StreamRefused(
                f"stream {name} is {got[0]} x {got[1]} {got[2]}; the calibration "
                f"needs {want[0]} x {want[1]} {want[2]}"
            )
        slots = entry.get("slots")
        if not isinstance(slots, list) or len(slots) < 3:
            raise StreamRefused(f"stream {name} has fewer than 3 slots")
        if not all(isinstance(slot, dict) for slot in slots):
            raise StreamRefused(f"stream {name} lists slots that are no objects")
        numbers = [slot.get("slot") for slot in slots]
        if not all(map(_is_whole, numbers)) or sorted(numbers) != list(
            range(len(slots))
        ):
            raise StreamRefused(f"stream {name}: the slots are not numbered 0 to n-1")
        shape = [rules.height * 3 // 2, rules.width]
        for slot in slots:
            if (
                slot.get("shape") != shape
                or slot.get("strides") != [rules.width, 1]
                or slot.get("nbytes") != rules.eye_bytes
            ):
                raise StreamRefused(
                    f"stream {name} slot {slot.get('slot')} is not a packed NV12 "
                    f"image of {rules.width} x {rules.height}"
                )
        depths.add(len(slots))
        gpus.add(entry.get("gpu_id"))
    if len(depths) != 1:
        raise StreamRefused("the two eyes have rings of different depth")
    if gpus != {0}:
        raise StreamRefused(
            f"the camera receiver's slots are on GPU {sorted(gpus, key=str)}; start "
            "it with CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES=0 and gpu_id 0"
        )
    return depths.pop(), 0


class SlotReader:
    """The one call that copies a slot to host memory.

    `to_host` is cupy.asnumpy in the bridge program; the tests pass numpy arrays
    and a plain copy.
    """

    def __init__(
        self, slots: Mapping[tuple[str, int], Any], to_host: Callable[[Any], Any]
    ) -> None:
        self._slots = slots
        self._to_host = to_host

    def __call__(self, stream: str, slot: int) -> Any:
        return self._to_host(self._slots[(stream, slot)])


@dataclass(frozen=True)
class _Note:
    """One notification of the receiver: one eye of one publish."""

    stream: str
    slot: int
    seq: int
    timestamp_ns: int  # the receiver's monotonic clock after the pixels were written
    session: int
    generation: int
    meta: Mapping[str, Any]


@dataclass
class _Held:
    """A copied pair that waits for the notification that confirms it."""

    record: dict[str, Any]
    left: Any
    right: Any
    session: int
    seq: int
    copy_end_ns: int


@dataclass
class _Request:
    after_session: int | None
    after_seq: int | None
    timeout_s: float
    deadline_ns: int
    refused: str = ""  # why the last pair since the request was not handed out


@dataclass(frozen=True)
class Outcome:
    """The answer to one request: the record and the raw bytes after it."""

    record: dict[str, Any]
    payloads: tuple[Any, ...] = ()


class PairTracker:
    """Pairs the two eyes, copies on request and verifies the copy. No GPU code.

    Not thread-safe: the bridge calls it under one lock. Times are those of
    `monotonic_ns`, which must be the receiver's monotonic clock (same machine).
    """

    COUNTERS = (
        "notifications",
        "malformed",
        "unpaired",  # one eye came, the other never did
        "eyes_differ",  # the two eyes disagree on slot or timestamp
        "no_capture_time",
        "clock_domain",
        "skew",
        "no_arrival_stamp",
        "pairs",  # complete and acceptable
        "copies",
        "overwritten",  # a copy discarded: its slot may have been written again
        "unconfirmed",  # a copy discarded: nothing confirmed it in time
        "session_changes",
        "handed_out",
    )

    def __init__(
        self,
        rules: StreamRules,
        ring_depth: int,
        read_slot: Callable[[str, int], Any],
        *,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if ring_depth < 3:
            raise ValueError("ring depth below 3")
        self._rules = rules
        self._depth = ring_depth
        self._read = read_slot
        self._now = monotonic_ns
        self.counters = dict.fromkeys(self.COUNTERS, 0)
        self.refusal: str | None = None  # set once: this stream is never handed out
        self._session: int | None = None
        self._half: _Note | None = None
        self._held: _Held | None = None
        self._ready: _Held | None = None
        self._request: _Request | None = None

    # The agent program's side

    def request(
        self, after_session: int | None, after_seq: int | None, timeout_s: float
    ) -> None:
        """Ask for one pair newer than the one named; replaces an open request."""
        self._drop_copy()
        self._request = _Request(
            after_session,
            after_seq,
            timeout_s,
            self._now() + int(timeout_s * 1e9),
        )

    def cancel(self) -> None:
        self._drop_copy()
        self._request = None

    def take(self) -> Outcome | None:
        """The answer to the open request, or None while there is none yet."""
        request = self._request
        if request is None:
            return None
        if self.refusal is not None:
            self.cancel()
            return Outcome(wire.error(self.refusal))
        now = self._now()
        held = self._held
        if held is not None and now - held.copy_end_ns > int(
            self._rules.confirm_timeout_s * 1e9
        ):
            self._discard(
                "unconfirmed",
                f"nothing confirmed the copy of seq {held.seq} within "
                f"{self._rules.confirm_timeout_s:g} s",
            )
        ready = self._ready
        if ready is not None:
            self._ready = None
            self._request = None
            self.counters["handed_out"] += 1
            return Outcome(ready.record, (ready.left, ready.right))
        if now >= request.deadline_ns:
            reason = f"no verified stereo pair within {request.timeout_s:g} s"
            if request.refused:
                reason += f"; last: {request.refused}"
            self.cancel()
            return Outcome(wire.none(reason, self.counters))
        return None

    # The receiver's side

    def on_notification(self, message: Mapping[str, Any]) -> None:
        """Take one "frame" notification of the receiver; others are ignored."""
        if message.get("type") != "frame" or self.refusal is not None:
            return
        if message.get("stream") not in self._rules.streams:
            return
        meta = message.get("metadata")
        try:
            if not isinstance(meta, Mapping):
                raise TypeError("metadata is no object")
            note = _Note(
                stream=message["stream"],
                slot=_whole(message["slot"]),
                seq=_whole(message["seq"]),
                timestamp_ns=_whole(message["timestamp_ns"]),
                session=_whole(meta["session_id"]),
                generation=_whole(message.get("slot_generation", 0)),
                meta=meta,
            )
        except (KeyError, TypeError, ValueError):
            self.counters["malformed"] += 1
            return
        self.counters["notifications"] += 1
        if note.session != self._session:
            self._new_session(note.session)
        self._judge(note)
        self._pair(note)

    def _new_session(self, session: int) -> None:
        if self._session is not None:
            self.counters["session_changes"] += 1
        self._session = session
        self._half = None
        if self._held is not None:
            self._discard("unconfirmed", "the sender's session changed during a copy")

    def _judge(self, note: _Note) -> None:
        """Confirm or discard the held copy by a later notification."""
        held = self._held
        if held is None or note.seq <= held.seq:
            return
        last = held.seq + self._depth - 1  # the last publish before the slot is reused
        after_copy = note.timestamp_ns > held.copy_end_ns
        if note.seq <= last and after_copy:
            held.record["check"].update(
                confirmed_by_seq=note.seq,
                confirmed_by_timestamp_ns=note.timestamp_ns,
            )
            self._held = None
            self._ready = held
        elif note.seq >= last and not after_copy:
            self._discard(
                "overwritten",
                f"seq {note.seq} was published before the copy of seq {held.seq} "
                f"ended (ring depth {self._depth})",
            )
        elif note.seq > last:
            self._discard(
                "unconfirmed",
                f"seq {note.seq} came before any confirmation of seq {held.seq}",
            )

    def _pair(self, note: _Note) -> None:
        half = self._half
        if half is not None and half.seq != note.seq:
            # Also when the seq went back: the receiver began its numbers again.
            self.counters["unpaired"] += 1
            half = None
        if half is None or half.stream == note.stream:
            self._half = note
            return
        self._half = None
        left, right = (
            (half, note) if half.stream == self._rules.streams[0] else (note, half)
        )
        if left.slot != right.slot or left.timestamp_ns != right.timestamp_ns:
            self._refuse("eyes_differ", f"seq {note.seq}: slot or timestamp differ")
            return
        fault = self._stream_fault(left, right)
        if fault is not None:
            self.refusal = fault
            return
        refused = self._pair_fault(left)
        if refused is not None:
            self._refuse(*refused)
            return
        self.counters["pairs"] += 1
        request = self._request
        if request is None or self._held is not None or self._ready is not None:
            return
        if left.session == request.after_session and (
            request.after_seq is not None and left.seq <= request.after_seq
        ):
            request.refused = (
                f"seq {left.seq} is not after seq {request.after_seq} of the "
                "same session"
            )
            return
        self._copy(left, right)

    def _stream_fault(self, left: _Note, right: _Note) -> str | None:
        """Why this stream is not the calibrated one, or None."""
        rules = self._rules
        for note, role in ((left, "left"), (right, "right")):
            meta = note.meta
            rotation = meta.get("stream_rotation")
            if not _is_whole(rotation) or rotation != rules.rotation:
                return (
                    f"{note.stream} is sent with rotation "
                    f"{meta.get('stream_rotation')!r}; the calibration needs "
                    f"{rules.rotation} (configs/camera_sender_rebot.yaml)"
                )
            if meta.get("role") != role:
                return f"{note.stream} carries the {meta.get('role')!r} eye, not {role}"
            if (meta.get("width"), meta.get("height")) != (rules.width, rules.height):
                return (
                    f"{note.stream} is {meta.get('width')} x {meta.get('height')}, "
                    f"not {rules.width} x {rules.height}"
                )
        devices = (left.meta.get("device_index"), right.meta.get("device_index"))
        if not all(map(_is_whole, devices)) or devices != rules.devices:
            return (
                f"the stereo pair comes from Argus devices {devices[0]!r} and "
                f"{devices[1]!r}; the calibration is for {rules.devices[0]} (left) "
                f"and {rules.devices[1]} (right) (configs/camera_sender_rebot.yaml)"
            )
        return None

    def _pair_fault(self, left: _Note) -> tuple[str, str] | None:
        """(counter, text) for a pair that is not handed out, or None."""
        meta = left.meta
        seq = left.seq
        sof = (meta.get("left_sensor_sof_ns"), meta.get("right_sensor_sof_ns"))
        if not all(_is_whole(value) and value > 0 for value in sof):
            return "no_capture_time", f"seq {seq} has no capture time"
        if meta.get("sender_wall_clock") != WALL_CLOCK:
            return (
                "clock_domain",
                f"seq {seq}: capture times are on the sender's "
                f"{meta.get('sender_wall_clock')!r} clock, not {WALL_CLOCK!r}",
            )
        delta = meta.get("tsc_delta_ns")
        if (
            not _is_whole(delta)
            or abs(delta) > self._rules.max_skew_ns
            or abs(sof[1] - sof[0]) > self._rules.max_skew_ns
        ):
            return (
                "skew",
                f"seq {seq}: the eyes started exposure {delta!r} ns apart "
                f"(limit {self._rules.max_skew_ns})",
            )
        if not _is_whole(meta.get("receiver_rx_payload_done_mono_ns")):
            return "no_arrival_stamp", f"seq {seq} has no arrival stamp"
        return None

    def _copy(self, left: _Note, right: _Note) -> None:
        start = self._now()
        try:
            pixels = (
                self._read(left.stream, left.slot),
                self._read(right.stream, right.slot),
            )
            sizes = tuple(memoryview(eye).nbytes for eye in pixels)
        except Exception as problem:
            self.refusal = f"copying slot {left.slot} failed: {problem!r}"
            return
        end = self._now()
        if sizes != (self._rules.eye_bytes, self._rules.eye_bytes):
            self.refusal = f"slot {left.slot} gave {sizes} bytes"
            return
        self.counters["copies"] += 1
        meta = left.meta
        record = {
            "type": wire.PAIR,
            "session_id": left.session,
            "seq": left.seq,
            "sof_wall_ns": [meta["left_sensor_sof_ns"], meta["right_sensor_sof_ns"]],
            "device_index": [meta["device_index"], right.meta["device_index"]],
            "rotation": meta["stream_rotation"],
            "receiver_rx_mono_ns": meta["receiver_rx_payload_done_mono_ns"],
            "check": {
                "slot": left.slot,
                "slot_generation": [left.generation, right.generation],
                "ring_depth": self._depth,
                "publish_mono_ns": left.timestamp_ns,
                "copy_start_mono_ns": start,
                "copy_end_mono_ns": end,
                "tsc_delta_ns": meta["tsc_delta_ns"],
            },
            "left_bytes": sizes[0],
            "right_bytes": sizes[1],
        }
        self._held = _Held(record, pixels[0], pixels[1], left.session, left.seq, end)

    def _refuse(self, counter: str, text: str) -> None:
        self.counters[counter] += 1
        if self._request is not None:
            self._request.refused = text

    def _discard(self, counter: str, text: str) -> None:
        self._held = None
        self._refuse(counter, text)

    def _drop_copy(self) -> None:
        self._held = None
        self._ready = None


def _is_whole(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _whole(value: Any) -> int:
    if not _is_whole(value):
        raise TypeError(f"not a whole number: {value!r}")
    return value


class ReceiverLink:
    """The bridge's end of the camera receiver's notification socket.

    Speaks the receiver's protocol (hello, subscribe, subscribed, then "frame"
    notifications, one JSON object per line) without any GPU code.
    """

    def __init__(self, socket_path: str, streams: Sequence[str]) -> None:
        self._path = socket_path
        self._streams = list(streams)
        self._sock: socket.socket | None = None
        self._buffer = bytearray()
        self.hello: dict[str, Any] = {}

    def attach(self, timeout_s: float = 5.0) -> dict[str, Any]:
        """Subscribe to the streams; the receiver's "subscribed" message.

        Raises OSError when no receiver listens, RuntimeError when it refuses.
        """
        sock = wire.dial(self._path)
        self._sock = sock
        deadline = time.monotonic() + timeout_s
        self.hello = self._one(deadline)
        if self.hello.get("type") != "hello":
            raise RuntimeError(f"expected hello, got {self.hello.get('type')!r}")
        request = {"type": "subscribe", "streams": self._streams}
        sock.sendall(json.dumps(request, separators=(",", ":")).encode() + b"\n")
        subscribed = self._one(deadline)
        if subscribed.get("type") == "error":
            raise RuntimeError(str(subscribed.get("error", "subscription refused")))
        if subscribed.get("type") != "subscribed":
            raise RuntimeError(f"expected subscribed, got {subscribed.get('type')!r}")
        sock.setblocking(False)
        return subscribed

    def read_messages(self, timeout_s: float) -> list[dict[str, Any]] | None:
        """The messages that are in; [] after `timeout_s` with none; None at the end."""
        sock = self._sock
        if sock is None:
            return None
        try:
            ready, _, _ = select.select([sock], [], [], timeout_s)
        except (OSError, ValueError):
            return None
        if not ready:
            return []
        while True:
            try:
                chunk = sock.recv(_RECV_BYTES)
            except BlockingIOError:
                break
            except OSError:
                return None
            if not chunk:
                return None
            self._buffer += chunk
            if len(self._buffer) > _MAX_BACKLOG_BYTES:
                break
        return self._lines()

    def close(self) -> None:
        sock, self._sock = self._sock, None
        if sock is not None:
            sock.close()

    def _lines(self) -> list[dict[str, Any]]:
        messages = []
        while True:
            end = self._buffer.find(b"\n")
            if end < 0:
                return messages
            line = bytes(self._buffer[:end])
            del self._buffer[: end + 1]
            message = json.loads(line.decode("utf-8"))
            if isinstance(message, dict):
                messages.append(message)

    def _one(self, deadline: float) -> dict[str, Any]:
        assert self._sock is not None
        while b"\n" not in self._buffer:
            left = deadline - time.monotonic()
            if left <= 0:
                raise RuntimeError("the camera receiver did not answer")
            self._sock.settimeout(left)
            try:
                chunk = self._sock.recv(_RECV_BYTES)
            except TimeoutError:
                continue
            if not chunk:
                raise RuntimeError("the camera receiver closed the socket")
            self._buffer += chunk
        end = self._buffer.find(b"\n")
        line = bytes(self._buffer[:end])
        del self._buffer[: end + 1]
        message = json.loads(line.decode("utf-8"))
        if not isinstance(message, dict):
            raise RuntimeError("the camera receiver sent no JSON object")
        return message


class Bridge:
    """Runs a PairTracker between a ReceiverLink and one agent program socket.

    Two threads: pump() reads the receiver's notifications all the time (the
    receiver drops a client that does not read for 2 s) and copies; serve()
    answers the agent program's requests.
    """

    def __init__(
        self, tracker: PairTracker, receiver: ReceiverLink, hello: dict[str, Any]
    ) -> None:
        self._tracker = tracker
        self._receiver = receiver
        self._hello = hello
        self._changed = threading.Condition()
        self._receiver_gone = ""
        self.stopping = threading.Event()

    def pump(self) -> None:
        """Feed the tracker until the receiver's socket ends or stop is asked."""
        gone = "the camera receiver closed its socket"
        try:
            while not self.stopping.is_set():
                messages = self._receiver.read_messages(0.05)
                if messages is None:
                    break
                if messages:
                    with self._changed:
                        for message in messages:
                            self._tracker.on_notification(message)
                        self._changed.notify_all()
            else:
                return
        except Exception as problem:
            gone = f"reading the camera receiver failed: {problem!r}"
        with self._changed:
            self._receiver_gone = gone
            self._changed.notify_all()

    def serve(self, client: socket.socket) -> int:
        """Answer one agent program until it closes; the bridge's exit code."""
        reader = wire.Reader(client)
        try:
            wire.send(client, self._hello)
            while not self.stopping.is_set():
                try:
                    record = reader.record(time.monotonic() + 0.5)
                except wire.WireTimeout:
                    continue
                if record["type"] == wire.CLOSE:
                    return EXIT_OK
                if record["type"] != wire.NEXT:
                    raise wire.WireError(f"unexpected {record['type']} record")
                outcome = self._answer(record, client)
                if outcome is None:
                    continue
                wire.send(client, outcome.record, *outcome.payloads)
                if outcome.record["type"] == wire.ERROR:
                    _complain(outcome.record["reason"])
                    return EXIT_FAILED
        except wire.WireClosed:
            return EXIT_OK
        except wire.WireError as problem:
            _complain(f"the agent program sent a bad record: {problem}")
            return EXIT_FAILED
        return EXIT_OK

    def _answer(self, request: dict[str, Any], client: socket.socket) -> Outcome | None:
        """The answer to one request; None when the agent program gave up on it."""
        with self._changed:
            self._tracker.request(
                request["after_session"], request["after_seq"], request["timeout_s"]
            )
            while True:
                if self._receiver_gone:
                    self._tracker.cancel()
                    return Outcome(wire.error(self._receiver_gone))
                if self.stopping.is_set():
                    self._tracker.cancel()
                    return Outcome(wire.error("the camera frame bridge is ending"))
                outcome = self._tracker.take()
                if outcome is not None:
                    return outcome
                # One request at a time: anything it sends now is its goodbye.
                if select.select([client], [], [], 0)[0]:
                    self._tracker.cancel()
                    return None
                self._changed.wait(0.01)


def load_cuda(camera_pr: Path, gpu_id: int) -> tuple[Any, Any]:
    """The pull request's consumer module and CuPy, with the CUDA context made.

    The part of open_cuda_slots that takes time; it needs no receiver.
    """
    source = camera_pr / "cuda_ipc_consumer" / "consumer.py"
    spec = importlib.util.spec_from_file_location("agentic_remote_camera_pr", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    import cupy

    with cupy.cuda.Device(gpu_id):
        cupy.cuda.runtime.memGetInfo()  # the first CUDA call makes the context
    return module, cupy


def open_cuda_slots(
    subscribed: Mapping[str, Any],
    camera_pr: Path,
    gpu_id: int,
    loaded: tuple[Any, Any] | None = None,
) -> SlotReader:
    """Open the receiver's slots with the pull request's CudaIpcClient (CuPy).

    `loaded` is what load_cuda() gave, when it was called beforehand.
    """
    module, cupy = loaded or load_cuda(camera_pr, gpu_id)
    client = module.CudaIpcClient("", [])
    client._open_streams(dict(subscribed))
    arrays = {key: slot.array for key, slot in client.slots.items()}

    def to_host(array: Any) -> Any:
        with cupy.cuda.Device(gpu_id):
            return cupy.asnumpy(array)

    reader = SlotReader(arrays, to_host)
    reader.owner = client  # keeps the shared buffers open
    return reader


def _cuda_slots_loaded_first(
    camera_pr: Path,
) -> Callable[[Mapping[str, Any], Path, int], SlotReader]:
    """open_cuda_slots with load_cuda() done now, for GPU 0 (check_catalog takes no other).

    What load_cuda() raises now is raised when the slots are asked for.
    """
    try:
        loaded = load_cuda(camera_pr, 0)
    except Exception as problem:
        failure = problem

        def fail(subscribed: Mapping[str, Any], camera_pr: Path, gpu_id: int) -> Any:
            raise failure

        return fail
    return lambda subscribed, camera_pr, gpu_id: open_cuda_slots(
        subscribed, camera_pr, gpu_id, loaded
    )


def _listen(path: str) -> socket.socket:
    """A listening Unix socket at `path`, mode 0600; the path must be free."""
    if len(os.fsencode(path)) > wire.MAX_SOCKET_PATH_BYTES:
        raise OSError(f"socket path too long: {path}")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    mask = os.umask(0o177)
    try:
        server.bind(path)
    finally:
        os.umask(mask)
    server.listen(1)
    return server


def _complain(text: str) -> None:
    print(f"{PROGRAM}: {text}", file=sys.stderr, flush=True)


def _parser() -> argparse.ArgumentParser:
    rules = StreamRules()
    parser = argparse.ArgumentParser(
        prog="python -m isaaccapture_examples.agentic_remote.agent.camera.frame_bridge",
        description=(
            "The camera frame bridge of the agentic remote execution example: "
            "verified stereo pairs from the camera receiver for the agent program."
        ),
    )
    parser.add_argument("--receiver-socket", required=True, metavar="PATH")
    parser.add_argument(
        "--listen", required=True, metavar="PATH", help="the bridge's own Unix socket"
    )
    parser.add_argument(
        "--camera-pr",
        default=os.environ.get(CAMERA_PR_ENV),
        metavar="DIR",
        help=f"the camera streamer example's folder (default: ${CAMERA_PR_ENV})",
    )
    parser.add_argument(
        "--streams", nargs=2, default=list(rules.streams), metavar=("LEFT", "RIGHT")
    )
    parser.add_argument("--width", type=int, default=rules.width)
    parser.add_argument("--height", type=int, default=rules.height)
    parser.add_argument("--rotation", type=int, default=rules.rotation)
    parser.add_argument(
        "--devices",
        nargs=2,
        type=int,
        default=list(rules.devices),
        metavar=("LEFT", "RIGHT"),
        help="the Argus devices the calibration manifest names",
    )
    parser.add_argument("--max-skew-ms", type=float, default=rules.max_skew_ns / 1e6)
    parser.add_argument(
        "--confirm-timeout-s", type=float, default=rules.confirm_timeout_s
    )
    parser.add_argument(
        "--accept-timeout-s",
        type=float,
        default=30.0,
        help="how long the bridge waits for the agent program",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    open_slots: Callable[[Mapping[str, Any], Path, int], Callable[[str, int], Any]]
    | None = None,
) -> int:
    """Run the bridge for one agent program. `open_slots` is for the tests."""
    try:
        args = _parser().parse_args(argv)
    except SystemExit as stop:
        return EXIT_OK if stop.code in (0, None) else EXIT_NOT_RUN
    # Lead's decision D10: the program itself picks the card, before CuPy loads.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if open_slots is None:
        if not args.camera_pr:
            _complain(f"--camera-pr or {CAMERA_PR_ENV} is needed")
            return EXIT_NOT_RUN
        # CuPy and the CUDA context before the receiver is subscribed to: the
        # receiver waits on a subscriber that does not read, up to 2 s, and its
        # other consumers get no notification meanwhile.
        open_slots = _cuda_slots_loaded_first(Path(args.camera_pr))
    rules = StreamRules(
        streams=(args.streams[0], args.streams[1]),
        width=args.width,
        height=args.height,
        rotation=args.rotation,
        devices=(args.devices[0], args.devices[1]),
        max_skew_ns=int(args.max_skew_ms * 1e6),
        confirm_timeout_s=args.confirm_timeout_s,
    )
    receiver = ReceiverLink(args.receiver_socket, rules.streams)
    server = None
    try:
        try:
            subscribed = receiver.attach()
        except (OSError, RuntimeError, ValueError) as problem:
            _complain(f"no camera receiver at {args.receiver_socket}: {problem}")
            return EXIT_FAILED
        try:
            ring_depth, gpu_id = check_catalog(subscribed, rules)
        except StreamRefused as problem:
            _complain(str(problem))
            return EXIT_FAILED
        try:
            read_slot = open_slots(subscribed, Path(args.camera_pr or ""), gpu_id)
        except Exception as problem:
            _complain(f"cannot open the camera receiver's slots: {problem!r}")
            return EXIT_NOT_RUN
        bridge = Bridge(
            PairTracker(rules, ring_depth, read_slot),
            receiver,
            wire.hello(
                width=rules.width,
                height=rules.height,
                pixel_format=rules.pixel_format,
                ring_depth=ring_depth,
                gpu_id=gpu_id,
                streams=rules.streams,
            ),
        )
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, lambda *_: bridge.stopping.set())
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            server = _listen(args.listen)
        except OSError as problem:
            _complain(f"cannot listen on {args.listen}: {problem}")
            return EXIT_FAILED
        pump = threading.Thread(target=bridge.pump, name="frame_bridge_pump")
        pump.daemon = True
        pump.start()
        try:
            client = _accept(server, bridge, args.accept_timeout_s)
            if client is None:
                _complain("no agent program came")
                return EXIT_FAILED
            with client:
                return bridge.serve(client)
        finally:
            bridge.stopping.set()
            pump.join(2.0)
    finally:
        receiver.close()
        if server is not None:
            server.close()
            try:
                os.unlink(args.listen)
            except OSError:
                pass


def _accept(
    server: socket.socket, bridge: Bridge, timeout_s: float
) -> socket.socket | None:
    deadline = time.monotonic() + timeout_s
    server.settimeout(0.2)
    while time.monotonic() < deadline and not bridge.stopping.is_set():
        try:
            client, _ = server.accept()
        except TimeoutError:
            continue
        except OSError:
            return None
        return client
    return None


if __name__ == "__main__":
    sys.exit(main())
