# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Starts the whole Ego Capture stack with one command and stops it together on Ctrl-C.

    uv run python tools/start_system.py --orin user@orin --deploy-script PATH --bridge PATH --livekit-server PATH

In order: deploy and start the Orin sender over ssh (for the SIPL backend this first stops nvargus-daemon, which
holds the cameras), a LiveKit server, CaptureServer, the backend (which owns the recorder), the camera receiver
and the camera-to-LiveKit bridge, then time sync through the timesync tool: ``ensure-local`` (which may install
chrony on this PC), an unprivileged follower here that measures the Orin's clock, and ``configure-remote`` on
the Orin. Without ``--orin`` no sender is started, so only the PC side comes up.
The servers, the backend and the receiver are waited on until they are ready before the next one starts, and all
children are stopped, last started first, when this script ends on Ctrl-C, on SIGTERM or after an error. The
Orin sender runs for ``--seconds``; when it ends the whole stack is stopped.

The sender settings come from the checked-in profile ``configs/default-camera-settings.yaml`` (see
CAMERA_SETTINGS.md). CaptureServer runs real acquisition (a built IsaacCapture checkout and tracking hardware)
unless ``--mock`` gives it synthetic tracking and cameras. ``--mock`` also replaces the Orin sender with a local
mock sender (no ``--orin``), so nothing real is started; the receiver, bridge and page are the same.

The sender release the profile describes, its deploy tool, the timesync tool, the LiveKit server, the bridge and
the mock sender are not part of this tree's base, so the tools' paths are options (or ``EGOCAP_DEPLOY_SCRIPT``,
``EGOCAP_TIMESYNC_TOOL``, ``EGOCAP_LIVEKIT_SERVER``, ``EGOCAP_BRIDGE``, ``EGOCAP_MOCK_SENDER``). The pure
helpers sit above the supervisor so they can be tested without
starting anything.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import ipaddress
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
import webbrowser
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

EGOCAP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EGOCAP_ROOT.parents[1]
LIVEKIT_DEV_KEY = "devkey"
LIVEKIT_DEV_SECRET = "secret"
CONTROL_TOKEN_MARKER = "Recording control token for this run:"
TIMESYNC_TOOL = REPO_ROOT / "src/core/timesync/tools/timesync_tool.py"
DEFAULT_SENDER_CONFIG = EGOCAP_ROOT / "configs/default-camera-settings.yaml"
DEFAULT_STATION_CONFIG = EGOCAP_ROOT / "recording/egocap/stations/ego_stations.example.yaml"
# The page reads latency from the sensor's capture time; see CAMERA_SETTINGS.md.
TIMESTAMP_BASIS = "capture_monotonic_us"
# Seconds a child gets to exit after SIGINT before it is killed. The backend closes an in-progress take, which its
# coordinator bounds at 35 s and the aligner bounds again afterwards, so it gets a margin over both.
# A deploy-tool command that only signals or queries the Orin; the deploy itself is not bounded by this.
ORIN_COMMAND_TIMEOUT_S = 60.0
# Installing chrony may download packages, so it gets longer than a query.
ENSURE_LOCAL_TIMEOUT_S = 300.0
DEFAULT_GRACE_S = 15.0
BACKEND_GRACE_S = 90.0
# Signals that mean "stop the stack"; while it is stopping they mean "stop faster".
STOP_SIGNALS = tuple(sig for sig in (signal.SIGTERM, getattr(signal, "SIGHUP", None)) if sig is not None)
# The mock sender sends no capture time, so the bridge stamps the frame-ready time on its own clock instead.
MOCK_TIMESTAMP_BASIS = "monotonic_us"
# The bridge shrinks the picture by this integer factor before encoding, so the descriptor declares the shrunken
# size: the recorder refuses a take whose frames differ from the declared one.
BRIDGE_DOWNSCALE = 2
DEFAULT_SENSOR_WIDTH = 2560
DEFAULT_SENSOR_HEIGHT = 1984
# The backend serves the page from this build output and mounts nothing if it is missing.
FRONTEND_INDEX = EGOCAP_ROOT / "frontend" / "dist" / "index.html"


class StartupError(Exception):
    """A step failed in a way the operator should read as is."""


def _b64(data: bytes) -> str:
    """URL-safe base64 without padding, as JWT segments require."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def publisher_token(key: str, secret: str, room: str, identity: str, hours: int = 24) -> str:
    """Sign a LiveKit join token that may publish and subscribe in one room.

    Done here rather than with the LiveKit CLI so the launcher needs nothing installed beyond the server itself.
    """
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    now = int(time.time())
    claims = {
        "iss": key,
        "sub": identity,
        "name": identity,
        "nbf": now,
        "exp": now + hours * 3600,
        "video": {
            "room": room,
            "roomJoin": True,
            "canPublish": True,
            "canSubscribe": True,
            "canPublishData": True,
        },
    }
    body = _b64(json.dumps(claims).encode())
    signature = _b64(hmac.new(secret.encode(), f"{header}.{body}".encode(), hashlib.sha256).digest())
    return f"{header}.{body}.{signature}"


def stream_track(name: str) -> str:
    """The LiveKit track the bridge publishes the sender stream ``name`` on."""
    return f"sensing-{name}"


def streams_yaml(
    url: str, room: str, publisher: str, streams: Sequence[tuple[str, int, int]], fps: int, basis: str
) -> str:
    """The CaptureServer ``--livekit-streams`` file: one (name, width, height) per stream the bridge publishes."""
    # Dumped from a mapping because the stream name comes from a user-edited profile.
    descriptor = {
        "livekit": {
            "url": url,
            "room": room,
            "publisher": publisher,
            "timestamp_basis": basis,
            "streams": [
                {"name": name, "track": stream_track(name), "width": width, "height": height, "fps": fps}
                for name, width, height in streams
            ],
        }
    }
    return yaml.safe_dump(descriptor, sort_keys=False)


def receiver_yaml(host: str, port: int, socket_path: str, streams: Sequence[tuple[str, int, int]]) -> str:
    """The camera receiver's config: connect to the sender and expose each (name, width, height) mono stream."""
    config = {
        "version": 1,
        "receiver": {
            "host": host,
            "port": port,
            "gpu_id": 0,
            "max_record_bytes": 268435456,
            "stats_log_interval_s": 2,
        },
        "ipc": {"socket_path": socket_path, "owner_name": "egocap_start_system", "queue_size": 1024},
        "streams": [
            {
                "index": index,
                "name": name,
                "type": "mono",
                "width": width,
                "height": height,
                "output_format": "nv12",
                "ring_depth": 6,
                "ipc": {"mono": name},
            }
            for index, (name, width, height) in enumerate(streams)
        ],
    }
    return yaml.safe_dump(config, sort_keys=False)


@dataclass(frozen=True)
class SenderSettings:
    """The effective sender values the rest of the stack must agree with."""

    port: int
    fps: int
    bitrate_mbps: int
    stream_ids: tuple[str, ...]
    full_range: bool
    sipl: bool
    # Each stream's output size; None, for a lone stream only, leaves the sensor mode's native size, which --width
    # and --height describe.
    sizes: tuple[tuple[int | None, int | None], ...]


def _section(parent: dict, key: str, path: Path) -> dict:
    """The mapping under ``parent[key]``, created when absent or null; another type is a startup error."""
    value = parent.get(key)
    if value is None:
        value = parent[key] = {}
    if not isinstance(value, dict):
        raise StartupError(f"{path}: {key} must be a mapping")
    return value


def load_sender_config(
    path: Path,
    fps: int | None = None,
    bitrate_mbps: int | None = None,
    sender_port: int | None = None,
) -> dict:
    """Read the checked-in sender profile and apply the command-line overrides to the loaded mapping.

    The bridge, the receiver and the CaptureServer descriptor are all built from the result, so one value
    cannot differ between the sender and its consumers.
    """
    try:
        config = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise StartupError(f"cannot read the sender config {path}: {error}; pass --sender-config") from error
    if not isinstance(config, dict) or not isinstance(config.get("streams"), list) or not config["streams"]:
        raise StartupError(f"{path}: the sender config needs at least one stream")
    network = _section(config, "network", path)
    for index, stream in enumerate(config["streams"]):
        if not isinstance(stream, dict):
            raise StartupError(f"{path}: streams[{index}] must be a mapping")
        # The receiver stream the launcher writes is mono, so another type would deploy a sender it cannot receive.
        if stream.get("type") != "mono":
            raise StartupError(f"{path}: the launcher supports only mono streams, got {stream.get('type')!r}")
        output = _section(stream, "output", path)
        # An override replaces the profile's value; the profile stays the source of everything else.
        if fps is not None:
            output["fps"] = fps
        if bitrate_mbps is not None:
            output["bitrate_mbps"] = bitrate_mbps
    if sender_port is not None:
        network["port"] = sender_port
    return config


def sender_settings(config: dict) -> SenderSettings:
    """The values other components need from a loaded sender config, refusing one that cannot give a latency."""
    network = config.get("network", {})
    streams = config["streams"]
    outputs = [stream.get("output", {}) for stream in streams]
    # The deploy builds the SIPL backend only for a SIPL profile, and the bridge's range flag follows the backend,
    # so neither may be guessed from an `auto` or missing value.
    if config.get("capture_backend") not in ("sipl", "argus"):
        raise StartupError("the sender config must set capture_backend to sipl or argus")
    for index, stream in enumerate(streams):
        if not stream.get("id"):
            raise StartupError(f"the sender config must set streams[{index}].id")
        # The bridge and the descriptor need a number, and a native-rate profile (fps 0) has none until runtime.
        if not outputs[index].get("fps"):
            raise StartupError(
                f"the sender config must set streams[{index}].output.fps; the bridge and descriptor need a rate"
            )
    # The bridge takes one rate for every stream.
    if len({output["fps"] for output in outputs}) != 1:
        raise StartupError("every stream in the sender config must share one output.fps")
    # The id names the stream's track and its IPC stream, so two streams cannot share one.
    stream_ids = tuple(str(stream["id"]) for stream in streams)
    if len(set(stream_ids)) != len(stream_ids):
        raise StartupError("every stream in the sender config must have its own id")
    sizes = tuple(
        (_output_dimension(output, "width", index), _output_dimension(output, "height", index))
        for index, output in enumerate(outputs)
    )
    # --width and --height describe one sensor, so they cannot stand in for several streams' sizes.
    if len(streams) > 1 and any(None in size for size in sizes):
        raise StartupError("a sender config with several streams must set every stream's output.width and height")
    return SenderSettings(
        port=int(network.get("port", 5000)),
        fps=int(outputs[0]["fps"]),
        bitrate_mbps=sum(int(output.get("bitrate_mbps", 0)) for output in outputs),
        stream_ids=stream_ids,
        # SIPL produces full-range pictures; the bridge must say so or the encoded stream signals limited range.
        full_range=config["capture_backend"] == "sipl",
        sipl=config["capture_backend"] == "sipl",
        sizes=sizes,
    )


def _output_dimension(output: dict, key: str, index: int) -> int | None:
    value = output.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise StartupError(
            f"the sender config's streams[{index}].output.{key} must be a positive integer, got {value!r}"
        )
    return value


def frame_size(profile: tuple[int | None, int | None], width: int | None, height: int | None) -> tuple[int, int]:
    """The picture size of one stream: the profile's output size when it sets one, else the sensor's.

    The receiver, the bridge and the descriptor all follow this, so a profile that resizes its output cannot
    disagree with them. A --width or --height that contradicts the profile is refused.
    """
    for name, flag in (("--width", width), ("--height", height)):
        if flag is not None and flag <= 0:
            raise StartupError(f"{name} must be positive, got {flag}")
    if (profile[0] is None) != (profile[1] is None):
        raise StartupError("the sender config must set both streams[*].output.width and height, or neither")
    if profile[0] is not None:
        for name, flag, value in (("--width", width, profile[0]), ("--height", height, profile[1])):
            if flag is not None and flag != value:
                raise StartupError(f"{name} {flag} contradicts the sender config's output size {value}")
        size = profile
    else:
        size = (width or DEFAULT_SENSOR_WIDTH, height or DEFAULT_SENSOR_HEIGHT)
    if size[0] % (2 * BRIDGE_DOWNSCALE) or size[1] % (2 * BRIDGE_DOWNSCALE):
        raise StartupError(f"the picture size {size[0]}x{size[1]} must divide by {2 * BRIDGE_DOWNSCALE}")
    return size


def mock_sender_command(
    python: str, script: Path, settings: SenderSettings, streams: Sequence[tuple[str, int, int]]
) -> list[str]:
    """The local mock sender that stands in for the Orin: HEVC streams shaped like the real ones."""
    return [
        python,
        str(script),
        "--port",
        str(settings.port),
        *(
            arg
            for name, width, height in streams
            for arg in ("--stream", f"{name}:{width}x{height}@{settings.fps}")
        ),
    ]


def bridge_timing_arguments(
    settings: SenderSettings, mock: bool, orin_host: str | None, reference_id: str | None, chrony_port: int
) -> list[str]:
    """The bridge flags that say what the frame timestamps mean and which chrony instances can vouch for them."""
    if mock:
        # No sender clock to qualify and no SIPL range to signal.
        return ["--timestamp-basis", MOCK_TIMESTAMP_BASIS]
    flags = [
        "--timestamp-basis",
        TIMESTAMP_BASIS,
        "--chrony-sender",
        str(orin_host),
        "--chrony-sender-refid",
        str(reference_id),
        "--chrony-pc-port",
        str(chrony_port),
    ]
    # SIPL pictures are full range; the bridge must say so or the encoded stream signals limited range.
    return [*flags, "--source-full-range"] if settings.full_range else flags


def check_station_cameras(station_config: Path, capture_cameras: Sequence[str]) -> None:
    """Refuse a station file that requests CaptureServer cameras which were not given with ``--capture-camera``."""
    try:
        data = yaml.safe_load(station_config.read_text())
        # Only the first station is ever read by the backend.
        station = next(iter(data["stations"].values()))
        requested = [source["name"] for source in station.get("video_sources") or []]
    except (OSError, yaml.YAMLError, KeyError, StopIteration, TypeError, AttributeError) as error:
        raise StartupError(f"cannot read the station file {station_config}: {error}") from error
    given = {camera.split("=", 1)[0] for camera in capture_cameras}
    missing = [name for name in requested if name not in given]
    if missing:
        raise StartupError(
            f"the station file {station_config} requests cameras {missing} that CaptureServer was not given; pass "
            "--capture-camera NAME=DEVICE_ID for each, or use a station file whose video_sources name none "
            "(the stream from the Orin comes through LiveKit)"
        )


def load_timesync_tool(path: Path):
    """The timesync module's tool, imported so its address and reference-id helpers have one copy."""
    import importlib.util

    if not path.is_file():
        raise StartupError(f"the timesync tool {path} does not exist (--timesync-tool or EGOCAP_TIMESYNC_TOOL)")
    spec = importlib.util.spec_from_file_location("timesync_tool", path)
    if spec is None or spec.loader is None:
        raise StartupError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    # The tool's dataclasses look their module up by name, so it is registered while it runs.
    sys.modules["timesync_tool"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop("timesync_tool", None)
        raise
    return module


def port_open(host: str, port: int) -> bool:
    """Whether something accepts TCP connections at host:port."""
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


@dataclass
class Managed:
    """One child process and the log file its output goes to."""

    name: str
    process: subprocess.Popen
    log: Path
    grace_s: float = DEFAULT_GRACE_S
    leader_only: bool = False
    # An optional child may exit without stopping the stack; see watch().
    optional: bool = False


def raise_if_a_child_exited(stack: "Stack") -> None:
    """Stop startup when a required child has already exited; an optional one is left to ``watch``.

    A sender that ended cleanly is the requested end of its run, which ``watch`` reports, not a startup failure.
    """
    for child in stack.children:
        if child.optional or child.process.poll() is None:
            continue
        if child.name == "sender" and child.process.returncode == 0:
            continue
        raise StartupError(f"{child.name} exited with {child.process.returncode}; see {child.log}")


def run_supervised(stack: "Stack", command: list[str]) -> int:
    """Run a helper to completion while checking the stack's children once a second.

    A required child that dies meanwhile stops the helper and raises, instead of going unnoticed until it returns.
    """
    process = subprocess.Popen(command)
    try:
        while process.poll() is None:
            raise_if_a_child_exited(stack)
            time.sleep(1)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    return process.returncode


def watch(stack: "Stack") -> Managed:
    """Block until a required child exits and return it.

    An optional child that exits is reported once and the stack keeps running.
    """
    reported: set[str] = set()
    while True:
        for child in stack.children:
            if child.process.poll() is None:
                continue
            if not child.optional:
                return child
            if child.name not in reported:
                reported.add(child.name)
                print(
                    f"WARNING: {child.name} exited with {child.process.returncode}; the preview stays up and "
                    f"capture times are unavailable; see {child.log}"
                )
        time.sleep(1)


class Stack:
    """Starts child processes one at a time and stops every one of them, last first, on exit."""

    def __init__(self, workdir: Path) -> None:
        """Keep each child's log under ``workdir`` so a failed step can point at the file to read."""
        self.workdir = workdir
        self.children: list[Managed] = []

    def start(
        self,
        name: str,
        command: Sequence[str],
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
        grace_s: float = DEFAULT_GRACE_S,
        leader_only: bool = False,
        optional: bool = False,
    ) -> Managed:
        """Launch a child in its own process group so stopping it also stops anything it spawned.

        ``leader_only`` sends the polite SIGINT to the child itself and not to its whole group, for a child that
        closes its own helper processes in an order of its own (the backend's video writers). ``optional`` marks a
        child whose exit is reported but does not stop the stack.
        """
        # Each child writes to its own file so a failed step can point the operator at the output to read.
        log = self.workdir / f"{name}.log"
        # A new session makes the child the leader of its own process group, so a signal to the group reaches
        # anything it spawned and never this script.
        with log.open("w") as handle:
            try:
                process = subprocess.Popen(
                    list(command),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, **(env or {})},
                    cwd=cwd,
                    start_new_session=True,
                )
            except OSError as error:
                # A wrong tool path is an operator mistake, so say which tool instead of printing a traceback.
                raise StartupError(f"cannot start {name} ({command[0]}): {error}") from error
        # Registering the child here is what makes stop_all() cover it even if a later step fails.
        child = Managed(name, process, log, grace_s, leader_only, optional)
        self.children.append(child)
        print(f"started {name} (pid {process.pid}); log {log}")
        return child

    def wait_until(self, child: Managed, ready, what: str, timeout_s: float) -> None:
        """Block until ``ready()`` is true, failing early if the child exits first."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if child.process.poll() is not None:
                raise StartupError(
                    f"{child.name} exited with {child.process.returncode} before {what}; see {child.log}"
                )
            if ready():
                print(f"{child.name}: {what}")
                return
            time.sleep(0.5)
        raise StartupError(f"{child.name} was not ready ({what}) within {timeout_s:.0f} s; see {child.log}")

    @staticmethod
    def _signal_group(child: Managed, sig: int) -> bool:
        """Send ``sig`` to the child's process group; False when the group is already gone."""
        try:
            os.killpg(child.process.pid, sig)
        except ProcessLookupError:
            # The last process of the group exited after poll() looked, which is a stop, not an error.
            return False
        return True

    def _kill(self, child: Managed) -> None:
        """Kill the child's group and reap it, so no zombie or orphan outlives this script."""
        self._signal_group(child, signal.SIGKILL)
        try:
            child.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print(f"{child.name} did not die after SIGKILL; leaving it", file=sys.stderr)

    @staticmethod
    def _wait(child: Managed, timeout_s: float, hurry: list[bool]) -> bool:
        """Wait for the child to exit; False when its grace ran out or the operator asked to hurry."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not hurry[0]:
            try:
                child.process.wait(timeout=min(0.2, max(0.0, deadline - time.monotonic())))
                return True
            except subprocess.TimeoutExpired:
                continue
        return child.process.poll() is not None

    def stop_all(self) -> None:
        """Stop the children last started first, waiting for each to exit before signalling the next.

        A child later in the start order depends on the ones before it (the backend needs CaptureServer's live
        connection to close a take), so it must be gone before its dependencies are signalled. Each child gets its
        own grace period before it is killed, and whatever is left of its process group once it has exited is
        killed too. A second Ctrl-C, or a SIGTERM or SIGHUP, while waiting kills the rest at once instead of
        aborting.
        """
        hurry = [False]

        def hurry_up(signum, frame) -> None:
            hurry[0] = True

        # A signal while stopping must hurry the cleanup, not raise out of it and leave children running.
        previous = {sig: signal.signal(sig, hurry_up) for sig in STOP_SIGNALS}
        try:
            for child in reversed(self.children):
                if child.process.poll() is not None:
                    self._signal_group(child, signal.SIGKILL)
                    continue
                if hurry[0]:
                    self._kill(child)
                    continue
                # SIGINT lets the receiver and sender drain and free the GPU and cameras. A leader-only child gets
                # it alone, so its helper processes are not interrupted behind its back.
                try:
                    if child.leader_only:
                        child.process.send_signal(signal.SIGINT)
                    else:
                        self._signal_group(child, signal.SIGINT)
                except ProcessLookupError:
                    # The leader exited between poll() and the signal, which is a stop and not an error.
                    pass
                try:
                    exited = self._wait(child, child.grace_s, hurry)
                except KeyboardInterrupt:
                    # An impatient operator: stop waiting for this child and for every one after it.
                    hurry[0] = True
                    exited = False
                if exited:
                    # The leader is gone; anything it left in the group goes with it.
                    self._signal_group(child, signal.SIGKILL)
                else:
                    if not hurry[0]:
                        print(
                            f"{child.name} did not exit within {child.grace_s:.0f} s; killing it", file=sys.stderr
                        )
                    self._kill(child)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def _raise_interrupt(signum, frame) -> None:
    """Turn SIGTERM and SIGHUP into the Ctrl-C path, so a closed terminal stops the stack like Ctrl-C does."""
    raise KeyboardInterrupt


def control_token(log: Path) -> str | None:
    """The recording control token the backend prints at startup, once it has."""
    for line in log.read_text(errors="replace").splitlines():
        if CONTROL_TOKEN_MARKER in line:
            return line.split(CONTROL_TOKEN_MARKER, 1)[1].strip()
    return None


def ensure_receiver_environment() -> None:
    """Run the receiver's own ``setup.sh`` when its venv or its native decoder is missing; it is idempotent."""
    receiver_dir = REPO_ROOT / "examples/camera_streamer/camera_receiver"
    has_venv = receiver_venv_python(receiver_dir).is_file()
    has_codec = any((receiver_dir / "codec").glob("_camera_receiver_codec*.so"))
    if has_venv and has_codec:
        return
    print("setting up the camera receiver environment (first run)", flush=True)
    subprocess.run(["bash", str(receiver_dir / "setup.sh")], check=True)


def receiver_venv_python(receiver_dir: Path) -> Path:
    """The Python of the receiver's own environment, where its ``setup.sh`` put it.

    ``setup.sh`` honours ``CAMERA_RECEIVER_VENV`` and ``run.sh`` ``CAMERA_RECEIVER_PYTHON``, so they decide here
    too.
    """
    explicit = os.environ.get("CAMERA_RECEIVER_PYTHON")
    if explicit:
        return Path(explicit)
    return Path(os.environ.get("CAMERA_RECEIVER_VENV", receiver_dir / ".venv")) / "bin/python"


def default_receiver_command() -> str:
    """Run the receiver with the Python from its own ``setup.sh`` environment, which has CuPy, when it exists."""
    receiver_dir = REPO_ROOT / "examples/camera_streamer/camera_receiver"
    venv_python = receiver_venv_python(receiver_dir)
    python = str(venv_python) if venv_python.is_file() else "python3"
    return shlex.join([python, str(receiver_dir / "camera_receiver.py")])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """The command line; every external tool's path may also come from an environment variable."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--orin", help="user@host of the Jetson; omit to start only the PC side")
    parser.add_argument(
        "--deploy-script", default=os.environ.get("EGOCAP_DEPLOY_SCRIPT"), help="the sender's deploy_sender.py"
    )
    parser.add_argument(
        "--livekit-server", default=os.environ.get("EGOCAP_LIVEKIT_SERVER"), help="livekit-server binary"
    )
    parser.add_argument("--bridge", default=os.environ.get("EGOCAP_BRIDGE"), help="camera_livekit_bridge binary")
    parser.add_argument(
        "--receiver",
        default=os.environ.get("EGOCAP_RECEIVER", default_receiver_command()),
        help="command that runs the camera receiver (takes --config FILE)",
    )
    parser.add_argument("--release", default="egocap-start", help="release name for the Orin deploy")
    parser.add_argument("--skip-deploy", action="store_true", help="reuse the release named by --release")
    parser.add_argument(
        "--sender-config", type=Path, default=DEFAULT_SENDER_CONFIG, help="sender profile uploaded to the Orin"
    )
    parser.add_argument("--station-config", type=Path, default=DEFAULT_STATION_CONFIG)
    parser.add_argument(
        "--width",
        type=int,
        help="sensor width when the profile sets no output size (default 2560); the receiver expects it and the "
        "descriptor declares it divided by the bridge shrink",
    )
    parser.add_argument(
        "--height",
        type=int,
        help="sensor height when the profile sets no output size (default 1984); the receiver expects it and the "
        "descriptor declares it divided by the bridge shrink",
    )
    parser.add_argument("--fps", type=int, help="override the profile's output.fps")
    parser.add_argument("--bitrate-mbps", type=int, help="override every stream's HEVC bitrate")
    parser.add_argument("--sender-port", type=int, help="override the profile's TCP video port")
    parser.add_argument(
        "--timesync-tool",
        type=Path,
        default=Path(os.environ.get("EGOCAP_TIMESYNC_TOOL", TIMESYNC_TOOL)),
        help="the timesync module's timesync_tool.py",
    )
    parser.add_argument("--chrony-cmd-port", type=int, default=12301, help="command port of this PC's chronyd")
    parser.add_argument(
        "--chrony-wait",
        type=float,
        default=120.0,
        help="seconds to wait for this PC to settle on the Orin's clock",
    )
    parser.add_argument(
        "--require-chrony", action="store_true", help="stop instead of starting before the clocks have settled"
    )
    parser.add_argument("--host-lan-ip", help="this PC's address toward the Orin (default: found from the route)")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="no hardware: synthetic tracking and cameras from CaptureServer, and a local mock sender instead of "
        "the Orin (with --bridge and --livekit-server the mock video also reaches the page)",
    )
    parser.add_argument(
        "--mock-sender",
        type=Path,
        default=Path(os.environ["EGOCAP_MOCK_SENDER"]) if os.environ.get("EGOCAP_MOCK_SENDER") else None,
        help="camera_streamer's mock_sender.py (not in this tree), used with --mock (needs ffmpeg with libx265)",
    )
    parser.add_argument(
        "--capture-camera",
        action="append",
        default=[],
        metavar="NAME=DEVICE_ID",
        help="a camera for CaptureServer to open (repeatable); the station file may only name cameras given here",
    )
    parser.add_argument("--backend-port", type=int, default=8010)
    parser.add_argument(
        "--seconds",
        type=int,
        default=3600,
        help="how long the Orin sender runs; when it ends the whole stack stops, so a take in progress is cut off",
    )
    parser.add_argument("--open", action="store_true", help="open the page in a browser once it is up")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Bring the stack up in dependency order, print how to use it, then wait until interrupted."""
    args = parse_args(argv)
    orin_host = args.orin.rsplit("@", 1)[-1] if args.orin else None
    room, publisher = "egocap-room", "egocap-bridge"
    livekit_url = "ws://127.0.0.1:7880"
    workdir = Path(tempfile.mkdtemp(prefix="egocap-start-"))
    ipc_socket = str(workdir / "camera_receiver.sock")
    stack = Stack(workdir)
    orin_contacted = False
    # Children run in their own sessions, so a signal to this script reaches none of them; turn the ones that mean
    # stop into the Ctrl-C path so the finally below always runs stop_all().
    for sig in STOP_SIGNALS:
        signal.signal(sig, _raise_interrupt)
    try:
        # The deploy is checked first so a missing tool fails before anything is running.
        if args.orin and args.mock:
            raise StartupError("--mock replaces the Orin sender with a local mock sender; do not pass --orin")
        if args.orin and not args.deploy_script:
            raise StartupError("--orin needs --deploy-script (or EGOCAP_DEPLOY_SCRIPT)")
        if args.orin and not (args.bridge and args.livekit_server):
            raise StartupError("--orin needs --bridge and --livekit-server to carry the video to the page")
        # The camera chain (sender, receiver, bridge, LiveKit) runs for the Orin, or for --mock when the tools to
        # carry its video are given; without them --mock is just CaptureServer's own synthetic cameras.
        chain = bool(args.orin) or (args.mock and bool(args.bridge and args.livekit_server))
        if chain and args.mock and not (args.mock_sender and args.mock_sender.is_file()):
            raise StartupError(
                "--mock with --bridge and --livekit-server needs --mock-sender (or EGOCAP_MOCK_SENDER)"
            )
        # Real acquisition opens only the cameras it is given, and a station file naming any other one is rejected
        # for the whole session, so catch it here instead of at the first Begin.
        if not args.mock:
            check_station_cameras(args.station_config, args.capture_camera)
        basis = MOCK_TIMESTAMP_BASIS if args.mock else TIMESTAMP_BASIS
        # One loaded profile is the source for the sender, the receiver, the bridge and the descriptor.
        sender_config = load_sender_config(args.sender_config, args.fps, args.bitrate_mbps, args.sender_port)
        sender = sender_settings(sender_config)
        streams = [
            (name, *frame_size(size, args.width, args.height))
            for name, size in zip(sender.stream_ids, sender.sizes, strict=True)
        ]
        # The bridge identifies the Orin's chrony by the reference id of its IPv4 address.
        if args.orin:
            try:
                ipaddress.IPv4Address(orin_host)
            except ValueError as error:
                raise StartupError(f"--orin needs an IPv4 address for the Orin, got {orin_host!r}") from error
        (workdir / "sender.yaml").write_text(yaml.safe_dump(sender_config, sort_keys=False))
        (workdir / "streams.yaml").write_text(
            streams_yaml(
                livekit_url,
                room,
                publisher,
                [(name, width // BRIDGE_DOWNSCALE, height // BRIDGE_DOWNSCALE) for name, width, height in streams],
                sender.fps,
                basis,
            )
        )
        (workdir / "receiver.yaml").write_text(
            receiver_yaml(orin_host or "127.0.0.1", sender.port, ipc_socket, streams)
        )
        if args.mock:
            print(f"mock sender: {sender.fps} fps streams {sender.stream_ids}, timestamps are frame-ready time")
        else:
            print(
                f"sender config {args.sender_config}: {sender.fps} fps, {sender.bitrate_mbps} Mbit/s total, "
                f"release {args.release}"
            )
            if args.orin:
                stop_at = time.strftime("%H:%M:%S", time.localtime(time.time() + args.seconds))
                print(f"the Orin sender stops after {args.seconds} s (about {stop_at}), which stops this stack")
        # Without a build the backend serves no page, and announcing a URL that answers 404 would mislead.
        if not FRONTEND_INDEX.is_file():
            raise StartupError(
                f"the frontend is not built ({FRONTEND_INDEX} is missing); run `npm install && npm run build` "
                f"in {EGOCAP_ROOT / 'frontend'}"
            )
        token_file = workdir / "publisher-token.txt"
        token_file.write_text(publisher_token(LIVEKIT_DEV_KEY, LIVEKIT_DEV_SECRET, room, publisher))
        token_file.chmod(0o600)
        python = sys.executable
        pythonpath = os.pathsep.join(
            [
                *(str(EGOCAP_ROOT / sub) for sub in ("backend", "recording", "capture_server", ".")),
                # CaptureServer imports the Manus calibration worker, which is not part of this folder.
                str(REPO_ROOT / "src/plugins/manus_calibration/python"),
            ]
        )

        if chain and args.receiver == default_receiver_command():
            ensure_receiver_environment()
            # The default was resolved before this setup created the receiver's environment, so resolve it again.
            # An explicit --receiver or EGOCAP_RECEIVER is not the default and is never replaced.
            args.receiver = default_receiver_command()
        if args.orin:
            tool = load_timesync_tool(args.timesync_tool)
            lan_ip = args.host_lan_ip or tool.lan_address_toward(orin_host)
        # 1. The Orin sender: deploy a release (never activating a service), then run it for a bounded time.
        if args.orin:
            orin_contacted = True
            deploy = [python, args.deploy_script, args.orin]
            # nvargus-daemon returns at boot and holds the cameras, which SIPL capture needs free; a sender left
            # by an earlier run outlives its ssh session.
            subprocess.run([*deploy, "stop"], check=False, timeout=ORIN_COMMAND_TIMEOUT_S)
            if sender.sipl:
                subprocess.run([*deploy, "stop-daemon"], check=True, timeout=ORIN_COMMAND_TIMEOUT_S)
            if not args.skip_deploy:
                print("deploying the sender to", args.orin)
                subprocess.run(
                    [*deploy, "deploy", *(["--with-sipl"] if sender.sipl else []), "--name", args.release],
                    check=True,
                )
            stack.start(
                "sender",
                [
                    *deploy,
                    "run",
                    "--name",
                    args.release,
                    "--config",
                    str(workdir / "sender.yaml"),
                    "--seconds",
                    str(args.seconds),
                ],
            )
        # 1b. With --mock a local mock sender stands in for the Orin, so nothing real is started.
        mock_sender = None
        if chain and args.mock:
            mock_sender = stack.start(
                "mock-sender", mock_sender_command(python, args.mock_sender, sender, streams)
            )
        # 2. LiveKit carries the video from the bridge to the browser and the recorder.
        if args.livekit_server:
            server = stack.start("livekit", [args.livekit_server, "--dev", "--bind", "127.0.0.1"])
            stack.wait_until(server, lambda: port_open("127.0.0.1", 7880), "listening on 7880", 30)
        # 3. CaptureServer announces the stream and serves tracking.
        capture = stack.start(
            "capture-server",
            [
                python,
                "-m",
                "capture_server",
                *(["--mock"] if args.mock else []),
                *(arg for camera in args.capture_camera for arg in ("--camera", camera)),
                # The descriptor advertises a LiveKit stream, so it is given only when something publishes one.
                *(["--livekit-streams", str(workdir / "streams.yaml")] if chain else []),
            ],
            env={"PYTHONPATH": pythonpath},
            cwd=EGOCAP_ROOT,
        )
        stack.wait_until(capture, lambda: port_open("127.0.0.1", 8082), "listening on 8082", 60)
        # 4. The backend serves the page and runs the recorder.
        backend = stack.start(
            "backend",
            [
                python,
                "-m",
                "uvicorn",
                "egocap_web.app:app",
                "--app-dir",
                "backend",
                "--port",
                str(args.backend_port),
            ],
            env={
                "PYTHONPATH": pythonpath,
                "EGOCAP_STATION_CONFIG": str(args.station_config),
                "LIVEKIT_API_KEY": LIVEKIT_DEV_KEY,
                "LIVEKIT_API_SECRET": LIVEKIT_DEV_SECRET,
            },
            cwd=EGOCAP_ROOT,
            grace_s=BACKEND_GRACE_S,
            leader_only=True,
        )
        stack.wait_until(
            backend, lambda: port_open("127.0.0.1", args.backend_port), f"listening on {args.backend_port}", 60
        )
        # 5. The receiver decodes the sender's stream; the bridge republishes it into the LiveKit room.
        if chain:
            if mock_sender is not None:
                # Building the mock's test streams with ffmpeg can take a while on the first run.
                stack.wait_until(
                    mock_sender, lambda: port_open("127.0.0.1", sender.port), f"listening on {sender.port}", 300
                )
            receiver = stack.start(
                "receiver", [*shlex.split(args.receiver), "--config", str(workdir / "receiver.yaml")]
            )
            stack.wait_until(receiver, lambda: Path(ipc_socket).exists(), "IPC socket ready", 60)
            stack.start(
                "bridge",
                [
                    args.bridge,
                    "--livekit-url",
                    livekit_url,
                    "--livekit-token-file",
                    str(token_file),
                    "--socket",
                    ipc_socket,
                    *(
                        arg
                        for name in sender.stream_ids
                        for arg in ("--stream", f"{name}={stream_track(name)}:{BRIDGE_DOWNSCALE}")
                    ),
                    "--fps",
                    str(sender.fps),
                    *bridge_timing_arguments(
                        sender,
                        args.mock,
                        orin_host,
                        tool.reference_id_for(orin_host) if args.orin else None,
                        args.chrony_cmd_port,
                    ),
                ],
            )
        url = f"http://127.0.0.1:{args.backend_port}/"
        token = control_token(backend.log)
        print(f"\nEgo Capture is up at {url}\nrecording control token: {token}\nlogs: {workdir}")
        if args.mock:
            # Everything the page shows and records in this mode is synthetic.
            print("--mock: tracking, the CaptureServer cameras and the camera stream are synthetic.")
        print("Ctrl-C stops everything.")
        if args.open:
            webbrowser.open(url)
        if args.orin:
            # The page address is already printed and the browser open, so a slow or failed step never holds the
            # preview. Each step has a timeout; a failure only degrades capture times (the bridge refuses
            # unqualified times, so a take cannot start until time sync is up). A child that dies meanwhile is
            # noticed between steps.
            print("setting up time sync; capture times are unavailable until it finishes", flush=True)
            settled = False
            try:
                provisioned = (
                    subprocess.run(
                        [python, str(args.timesync_tool), "ensure-local"],
                        check=False,
                        timeout=ENSURE_LOCAL_TIMEOUT_S,
                    ).returncode
                    == 0
                )
            except (OSError, subprocess.TimeoutExpired):
                provisioned = False
            raise_if_a_child_exited(stack)
            if provisioned:
                # An unprivileged chronyd on this PC measures the Orin's clock and never adjusts this PC's. Nothing
                # has to reach this PC from the Orin, so a firewall here does not matter.
                stack.start(
                    "timesync-follower",
                    [
                        python,
                        str(args.timesync_tool),
                        "follow",
                        "--server",
                        orin_host,
                        "--cmd-port",
                        str(args.chrony_cmd_port),
                        "--workdir",
                        str(workdir / "chrony"),
                    ],
                    # Without --require-chrony a failed follower leaves the preview up and the bridge refusing
                    # capture times, which is what an unavailable clock measurement means.
                    optional=not args.require_chrony,
                )
                print("configuring the Orin's chrony to serve its clock to", lan_ip, flush=True)
                try:
                    configured = (
                        subprocess.run(
                            [
                                python,
                                str(args.timesync_tool),
                                "configure-remote",
                                args.orin,
                                "--host-lan-ip",
                                lan_ip,
                            ],
                            check=False,
                            timeout=90,
                        ).returncode
                        == 0
                    )
                except subprocess.TimeoutExpired:
                    configured = False
                raise_if_a_child_exited(stack)
                settled = (
                    configured
                    and run_supervised(
                        stack,
                        [
                            python,
                            str(args.timesync_tool),
                            "status",
                            "--remote",
                            orin_host,
                            "--cmd-port",
                            str(args.chrony_cmd_port),
                            "--wait",
                            str(args.chrony_wait),
                        ],
                    )
                    == 0
                )
            if not settled:
                message = "the clocks have not settled, so the bridge refuses capture times until they do"
                if args.require_chrony:
                    raise StartupError(message)
                print("WARNING:", message)
            else:
                print("time sync: settled", flush=True)
        # Stay up until a child dies or the operator interrupts.
        dead = watch(stack)
        if dead.name == "sender" and dead.process.returncode == 0:
            # The sender runs for --seconds and then exits cleanly: that is the requested end, not a failure.
            print(f"the Orin sender finished its {args.seconds} s run; stopping the stack")
            return 0
        raise StartupError(f"{dead.name} exited with {dead.process.returncode}; see {dead.log}")
    except KeyboardInterrupt:
        return 0
    except (StartupError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        stack.stop_all()
        if orin_contacted:
            # Ending the local ssh does not end the remote sender; ask it to stop so it frees the cameras.
            try:
                subprocess.run(
                    [sys.executable, args.deploy_script, args.orin, "stop"],
                    check=False,
                    timeout=ORIN_COMMAND_TIMEOUT_S,
                )
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                print(
                    "could not confirm that the Orin sender stopped; check it with the deploy tool",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    sys.exit(main())
