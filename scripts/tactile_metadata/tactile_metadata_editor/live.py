# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owned latest snapshots from one authenticated local receiving worker."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import secrets
import selectors
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from uuid import uuid4

from .checks import application_binding_issues
from .compiler import canonical_values, metadata_fingerprint
from .domain import FrozenMap, Issue, LiveSnapshot, thaw
from .profiles.hand import JOINT_NAMES

MAX_MESSAGE = 8 * 1024 * 1024
_BOOTSTRAP = """
import json, runpy, sys
request_path = sys.argv[1]
with open(request_path, encoding="utf-8") as stream:
    request = json.load(stream)
sys.path[:] = request["module_search_path"]
sys.argv[:] = ["tactile_metadata_editor.live_worker", request_path]
runpy.run_module("tactile_metadata_editor.live_worker", run_name="__main__")
"""


@dataclass(frozen=True)
class ConnectConfig:
    bindings: FrozenMap
    demo: bool = False
    demo_values: FrozenMap = field(default_factory=FrozenMap)

    def __post_init__(self):
        bindings = FrozenMap(self.bindings)
        if not bindings or any(side not in ("left", "right") for side in bindings):
            raise ValueError("Configure explicit left/right receiving sources")
        if type(self.demo) is not bool:
            raise ValueError("Demo mode must be boolean")
        collections, vendors, joint_devices = set(), set(), {}
        for side, binding in bindings.items():
            issues = application_binding_issues(binding, side)
            if issues:
                raise ValueError("; ".join(issue.message for issue in issues))
            if not isinstance(binding, Mapping):
                raise ValueError("Each side needs a source configuration")
            collection = binding.get("tactile_collection_id")
            if not isinstance(collection, str) or not collection:
                raise ValueError("Tactile collection ID must be nonempty")
            if collection in collections:
                raise ValueError("Each side needs a distinct tactile collection ID")
            collections.add(collection)
            hand = binding.get("hand_source")
            if not isinstance(hand, Mapping) or hand.get("side") != side:
                raise ValueError("Hand source must declare the same side")
            vendors.add((hand.get("vendor_id"), FrozenMap(
                hand.get("vendor_params", {})
            )))
            joint = binding.get("joint_source")
            if joint is not None:
                previous = joint_devices.setdefault(
                    joint["collection_id"], joint["device_id"],
                )
                if previous != joint["device_id"]:
                    raise ValueError(
                        "Shared joint collection has conflicting device IDs"
                    )
        if len(vendors) != 1:
            raise ValueError(
                "Both hands must share one HandTracker vendor configuration"
            )
        values = {}
        for collection, value in self.demo_values.items():
            if collection not in collections:
                raise ValueError("Synthetic metadata has an unconfigured collection")
            values[collection] = canonical_values(value)
        if values and not self.demo:
            raise ValueError("Synthetic metadata is only available in demo mode")
        object.__setattr__(self, "bindings", bindings)
        object.__setattr__(self, "demo_values", FrozenMap(values))


@dataclass(frozen=True)
class ConnectionStatus:
    state: str
    epoch: str
    issues: tuple[Issue, ...] = ()
    synthetic: bool = False


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Protocol numbers must be finite")
    return number


def _reject_constant(value):
    raise ValueError("Protocol numbers must be finite")


def send_message(stream, message):
    payload = json.dumps(
        message, ensure_ascii=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    if not 0 < len(payload) <= MAX_MESSAGE:
        raise ValueError("Local receiving message exceeds 8 MiB")
    stream.sendall(struct.pack("!I", len(payload)) + payload)


def _read_exact(stream, count, stop=None, deadline=None):
    chunks = bytearray()
    while len(chunks) < count:
        if stop is not None and stop.is_set():
            raise EOFError("Local receiving connection closed")
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError("Local receiving handshake timed out")
        try:
            chunk = stream.recv(count - len(chunks))
        except socket.timeout:
            continue
        if not chunk:
            raise EOFError("Local receiving worker closed the connection")
        chunks.extend(chunk)
    return bytes(chunks)


def receive_message(stream, stop=None, deadline=None):
    size = struct.unpack("!I", _read_exact(stream, 4, stop, deadline))[0]
    if not 0 < size <= MAX_MESSAGE:
        raise ValueError("Invalid local receiving message length")
    payload = _read_exact(stream, size, stop, deadline).decode("utf-8")
    message = json.loads(
        payload, parse_constant=_reject_constant, parse_float=_finite_float,
    )
    if not isinstance(message, dict):
        raise ValueError("Local receiving message must be an object")
    return message


def _number(value):
    try:
        return not isinstance(value, bool) and isinstance(
            value, (int, float)
        ) and math.isfinite(value)
    except OverflowError:
        return False


def _rows(value, width):
    return isinstance(value, (list, tuple)) and all(
        isinstance(row, (list, tuple)) and len(row) == width
        and all(_number(item) for item in row) for row in value
    )


class LiveConnection:
    def __init__(self, config: ConnectConfig):
        if not isinstance(config, ConnectConfig):
            raise TypeError("Expected ConnectConfig")
        self.config = config
        self.epoch = str(uuid4())
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._closing = threading.Event()
        self._latest = None
        self._metadata = {}
        self._last_received = time.monotonic()
        self._status = ConnectionStatus("connecting", self.epoch, synthetic=config.demo)
        self._diagnostic = bytearray()
        self._stream = None
        self._process = None
        self._temporary = tempfile.TemporaryDirectory(
            prefix="tactile-live-", dir="/tmp",
        )
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.settimeout(0.2)
        try:
            self._listener.bind(("127.0.0.1", 0))
            self._listener.listen(1)
            self._token = secrets.token_urlsafe(32)
            request_path = Path(self._temporary.name) / "request.json"
            request = {
                "version": 1, "port": self._listener.getsockname()[1],
                "token": self._token, "epoch": self.epoch,
                "bindings": thaw(config.bindings), "demo": config.demo,
                "demo_values": thaw(config.demo_values),
                "module_search_path": list(sys.path),
            }
            descriptor = os.open(
                request_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump(request, output, allow_nan=False)
            environment = os.environ.copy()
            environment["ISAACCAPTURE_LOGGING"] = "off"
            self._process = subprocess.Popen(
                [sys.executable, "-B", "-c", _BOOTSTRAP, str(request_path)],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, env=environment,
            )
        except BaseException:
            self._listener.close()
            self._temporary.cleanup()
            raise
        self._diagnostic_thread = threading.Thread(
            target=self._drain_diagnostics, daemon=True,
        )
        self._thread = threading.Thread(target=self._receive, daemon=True)
        try:
            self._diagnostic_thread.start()
            self._thread.start()
        except BaseException:
            self._closing.set()
            self._listener.close()
            self._reap(0)
            if self._diagnostic_thread.is_alive():
                self._diagnostic_thread.join(timeout=0.5)
            else:
                self._process.stdout.close()
            self._temporary.cleanup()
            raise

    def _drain_diagnostics(self):
        stream = self._process.stdout
        try:
            os.set_blocking(stream.fileno(), False)
            with selectors.DefaultSelector() as selector:
                selector.register(stream, selectors.EVENT_READ)
                while True:
                    ready = selector.select(0.1)
                    for key, _ in ready:
                        chunk = os.read(key.fd, 8192)
                        if not chunk:
                            return
                        with self._lock:
                            self._diagnostic.extend(chunk)
                            del self._diagnostic[:-8192]
                    if not ready and self._process.poll() is not None:
                        return
        except (OSError, ValueError):
            return
        finally:
            stream.close()

    @property
    def diagnostics(self):
        with self._lock:
            return self._diagnostic.decode("utf-8", errors="replace")

    @property
    def status(self):
        with self._lock:
            status = self._status
            stale = time.monotonic() - self._last_received > 1
        if status.state in ("connecting", "connected") and stale:
            return ConnectionStatus(
                "interrupted", self.epoch, (Issue(
                    "warning", "live_heartbeat",
                    "No local worker heartbeat for one second",
                    field="live",
                ),), self.config.demo,
            )
        return status

    def latest(self):
        with self._lock:
            snapshot = self._latest
        if snapshot is None or self.status.state == "connected":
            return snapshot
        sources = thaw(snapshot.sources)
        for group in sources.values():
            for source in group.values():
                source["valid"] = False
                source["reason"] = "Local receiving connection is interrupted"
        return LiveSnapshot(
            snapshot.epoch, snapshot.seq, snapshot.read_monotonic_ns,
            snapshot.metadata, FrozenMap(sources),
        )

    def _set_status(self, state, issues=()):
        with self._lock:
            self._status = ConnectionStatus(
                state, self.epoch, tuple(issues), self.config.demo,
            )

    def _receive(self):
        try:
            deadline = time.monotonic() + 30
            while not self._closing.is_set():
                if self._process.poll() is not None:
                    raise RuntimeError("Receiving worker exited before its handshake")
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        "Receiving worker did not connect within 30 seconds"
                    )
                try:
                    stream, address = self._listener.accept()
                except socket.timeout:
                    continue
                stream.settimeout(0.2)
                self._stream = stream
                if address[0] != "127.0.0.1":
                    raise ValueError("Receiving connection is not local")
                hello = receive_message(
                    stream, self._closing, deadline=time.monotonic() + 5,
                )
                if (
                    type(hello.get("version")) is not int or hello["version"] != 1
                    or hello.get("type") != "hello"
                    or not isinstance(hello.get("token"), str)
                    or not secrets.compare_digest(hello["token"], self._token)
                    or hello.get("epoch") != self.epoch
                ):
                    raise ValueError("Receiving worker authentication failed")
                with self._send_lock:
                    send_message(stream, {
                        "type": "status", "status": "accepted", "version": 1,
                    })
                self._listener.close()
                while not self._closing.is_set():
                    self._message(receive_message(stream, self._closing))
                break
        except Exception as error:
            if not self._closing.is_set():
                self._set_status("error", (Issue(
                    "error", "live_receive", f"{type(error).__name__}: {error}",
                    field="live",
                ),))
        finally:
            self._listener.close()
            if not self._closing.is_set():
                if self._stream is not None:
                    self._stream.close()
                self._reap(0.2)
                self._temporary.cleanup()

    def _message(self, message):
        if message.get("epoch") != self.epoch:
            raise ValueError("Receiving message belongs to a different epoch")
        kind = message.get("type")
        if kind == "error":
            detail = message.get("message")
            if not isinstance(detail, str):
                raise ValueError("Receiving error has no reason")
            raise RuntimeError(detail)
        if kind == "status":
            if message.get("status") not in ("connecting", "connected"):
                raise ValueError("Invalid receiving status")
            self._set_status(message["status"])
        elif kind == "metadata":
            collection = message.get("collection_id")
            allowed = {
                item["tactile_collection_id"]
                for item in self.config.bindings.values()
            }
            if collection not in allowed:
                raise ValueError("Metadata belongs to an unconfigured collection")
            values = canonical_values(message.get("values"))
            fingerprint = metadata_fingerprint(values)
            if message.get("fingerprint") != fingerprint:
                raise ValueError(
                    "Received metadata fingerprint differs from its values"
                )
            previous = self._metadata.get(collection)
            if previous is not None and previous["fingerprint"] != fingerprint:
                raise ValueError(
                    "Received metadata changed within the receiving session"
                )
            self._metadata[collection] = {
                "values": values, "fingerprint": fingerprint,
                "synthetic": self.config.demo,
            }
        elif kind == "frame":
            self._frame(message)
        else:
            raise ValueError("Unsupported receiving message type")
        with self._lock:
            self._last_received = time.monotonic()

    def _frame(self, message):
        seq, read_time = message.get("seq"), message.get("read_monotonic_ns")
        if (
            type(seq) is not int or seq < 0
            or type(read_time) is not int or read_time < 0
        ):
            raise ValueError("Invalid receiving frame sequence or local read time")
        if self._latest is not None and seq <= self._latest.seq:
            raise ValueError("Receiving frame sequence did not increase")
        if self._latest is not None and read_time < self._latest.read_monotonic_ns:
            raise ValueError("Local frame read time moved backwards")
        sources = message.get("sources")
        if not isinstance(sources, dict) or set(sources) != {
            "tactile", "hands", "joints",
        }:
            raise ValueError("Receiving frame must contain all source groups")
        expected = {
            "tactile": {
                binding["tactile_collection_id"]
                for binding in self.config.bindings.values()
            },
            "hands": set(self.config.bindings),
            "joints": {
                binding["joint_source"]["collection_id"]
                for binding in self.config.bindings.values()
                if binding.get("joint_source")
            },
        }
        for group, names in expected.items():
            records = sources[group]
            if not isinstance(records, dict) or set(records) != names:
                raise ValueError(
                    f"Receiving {group} sources differ from the configuration"
                )
            for name, record in records.items():
                if (
                    not isinstance(record, dict)
                    or type(record.get("valid")) is not bool
                ):
                    raise ValueError(
                        "Receiving source needs independent boolean validity"
                    )
                if group == "tactile" and record["valid"]:
                    baseline = self._metadata.get(name)
                    if baseline is None or not _rows(record.get("forces"), 3):
                        raise ValueError(
                            "Valid forces require finite rows and metadata"
                        )
                    count = len(
                        baseline["values"]["geometry_mapping"]["taxel_bind_poses"]
                    )
                    if len(record["forces"]) != count:
                        raise ValueError("Force count differs from producer metadata")
                if group == "hands" and record["valid"]:
                    count = len(JOINT_NAMES)
                    for field, width in (("positions", 3),
                                         ("orientations_xyzw", 4)):
                        rows = record.get(field)
                        if (
                            not isinstance(rows, (tuple, list)) or len(rows) != count
                            or any(row is not None and not _rows([row], width)
                                   for row in rows)
                        ):
                            raise ValueError("Hand pose rows must be finite or null")
                    radii, flags = record.get("radii"), record.get("is_valid")
                    if (
                        not isinstance(radii, (tuple, list)) or len(radii) != count
                        or any(value is not None and not _number(value)
                               for value in radii)
                        or not isinstance(flags, (tuple, list)) or len(flags) != count
                        or any(type(value) is not bool for value in flags)
                    ):
                        raise ValueError("Invalid hand radius or producer validity")
        snapshot = LiveSnapshot(
            self.epoch, seq, read_time, FrozenMap(self._metadata), FrozenMap(sources),
        )
        with self._lock:
            self._latest = snapshot
            self._status = ConnectionStatus(
                "connected", self.epoch, synthetic=self.config.demo,
            )

    def _reap(self, timeout):
        if self._process is None:
            return
        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                self._process.terminate()
                self._process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            except ProcessLookupError:
                self._process.wait()

    def disconnect(self):
        if self._closing.is_set():
            return
        self._closing.set()
        if self._stream is not None:
            try:
                with self._send_lock:
                    send_message(self._stream, {"type": "stop"})
            except (OSError, ValueError):
                pass
        self._reap(5)
        if self._stream is not None:
            try:
                self._stream.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._stream.close()
        self._listener.close()
        self._thread.join(timeout=1)
        self._diagnostic_thread.join(timeout=0.5)
        self._temporary.cleanup()
        self._set_status("disconnected")


def connect(config: ConnectConfig) -> LiveConnection:
    return LiveConnection(config)
