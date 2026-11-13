# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Starts the whole Ego Capture stack with one command and stops it together on Ctrl-C.

    uv run python tools/start_system.py --orin user@orin --deploy-script PATH --bridge PATH --livekit-server PATH

In order: deploy and start the Orin sender over ssh, a LiveKit server, CaptureServer, the backend (which owns
the recorder), the camera receiver and the camera-to-LiveKit bridge. Without ``--orin`` no sender is started, so
only the PC side comes up. The servers, the backend and the receiver are waited on until they are ready before the
next one starts, and all children are stopped, last started first, when this script ends on Ctrl-C, on SIGTERM or
after an error. The Orin sender runs for ``--seconds``; when it ends the whole stack is stopped.

The sender settings come from the checked-in profile ``configs/default-camera-settings.yaml`` (see
CAMERA_SETTINGS.md). CaptureServer runs real acquisition (a built IsaacCapture checkout and tracking hardware)
unless ``--mock`` gives it synthetic tracking and cameras. ``--mock`` also replaces the Orin sender with a local
mock sender (no ``--orin``), so nothing real is started; the receiver, bridge and page are the same.

The sender release the profile describes, its deploy tool, the LiveKit server, the bridge and the mock sender are
not part of this tree, so the tools' paths are options (or ``EGOCAP_DEPLOY_SCRIPT``, ``EGOCAP_LIVEKIT_SERVER``,
``EGOCAP_BRIDGE``, ``EGOCAP_MOCK_SENDER``). The pure helpers sit above the supervisor so they can be tested without
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
DEFAULT_SENDER_CONFIG = EGOCAP_ROOT / "configs/default-camera-settings.yaml"
DEFAULT_STATION_CONFIG = EGOCAP_ROOT / "recording/egocap/stations/ego_stations.example.yaml"
# The page reads latency from the sensor's capture time; see CAMERA_SETTINGS.md.
TIMESTAMP_BASIS = "capture_monotonic_us"
# Seconds a child gets to exit after SIGINT before it is killed. The backend closes an in-progress take, which its
# coordinator bounds at 35 s and the aligner bounds again afterwards, so it gets a margin over both.
DEFAULT_GRACE_S = 15.0
BACKEND_GRACE_S = 90.0
# Signals that mean "stop the stack"; while it is stopping they mean "stop faster".
STOP_SIGNALS = tuple(sig for sig in (signal.SIGTERM, getattr(signal, "SIGHUP", None)) if sig is not None)
# The mock sender has no clock responder, so the bridge stamps the frame-ready time on its own clock instead.
MOCK_TIMESTAMP_BASIS = "monotonic_us"
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


def streams_yaml(
    url: str, room: str, publisher: str, track: str, width: int, height: int, fps: int, basis: str, name: str
) -> str:
    """The CaptureServer ``--livekit-streams`` file for one camera published by the bridge."""
    # Dumped from a mapping because the stream name comes from a user-edited profile.
    descriptor = {
        "livekit": {
            "url": url,
            "room": room,
            "publisher": publisher,
            "timestamp_basis": basis,
            "streams": [{"name": name, "track": track, "width": width, "height": height, "fps": fps}],
        }
    }
    return yaml.safe_dump(descriptor, sort_keys=False)


def receiver_yaml(host: str, port: int, socket_path: str, width: int, height: int, name: str) -> str:
    """The camera receiver's config: connect to the sender and expose one mono stream over CUDA IPC."""
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
                "index": 0,
                "name": name,
                "type": "mono",
                "width": width,
                "height": height,
                "output_format": "nv12",
                "ring_depth": 6,
                "ipc": {"mono": name},
            }
        ],
    }
    return yaml.safe_dump(config, sort_keys=False)


@dataclass(frozen=True)
class SenderSettings:
    """The effective sender values the rest of the stack must agree with."""

    port: int
    clock_port: int
    fps: int
    bitrate_mbps: int
    stream_id: str
    full_range: bool
    sipl: bool


def load_sender_config(
    path: Path,
    fps: int | None = None,
    bitrate_mbps: int | None = None,
    clock_port: int | None = None,
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
    if not isinstance(config, dict) or not isinstance(config.get("streams"), list) or len(config["streams"]) != 1:
        raise StartupError(f"{path}: the launcher supports exactly one stream in the sender config")
    stream = config["streams"][0]
    network = config.setdefault("network", {})
    output = stream.setdefault("output", {})
    # An override replaces the profile's value; the profile stays the source of everything else.
    if fps is not None:
        output["fps"] = fps
    if bitrate_mbps is not None:
        output["bitrate_mbps"] = bitrate_mbps
    if clock_port is not None:
        network["clock_port"] = clock_port
    if sender_port is not None:
        network["port"] = sender_port
    return config


def sender_settings(config: dict) -> SenderSettings:
    """The values other components need from a loaded sender config, refusing one that cannot give a latency."""
    network = config.get("network", {})
    stream = config["streams"][0]
    output = stream.get("output", {})
    # The deploy builds the SIPL backend only for a SIPL profile, and the bridge's range flag follows the backend,
    # so neither may be guessed from an `auto` or missing value.
    if config.get("capture_backend") not in ("sipl", "argus"):
        raise StartupError("the sender config must set capture_backend to sipl or argus")
    if not stream.get("id"):
        raise StartupError("the sender config must set streams[0].id")
    # The bridge and the descriptor need a number, and a native-rate profile (fps 0) has none until runtime.
    if not output.get("fps"):
        raise StartupError(
            "the sender config must set streams[0].output.fps; the bridge and descriptor need a rate"
        )
    # Without both, the page has no sensor capture time and no clock to map it with.
    if not network.get("timing_extension") or not network.get("clock_port"):
        raise StartupError("the sender config must set network.timing_extension: true and network.clock_port")
    return SenderSettings(
        port=int(network.get("port", 5000)),
        clock_port=int(network["clock_port"]),
        fps=int(output["fps"]),
        bitrate_mbps=int(output.get("bitrate_mbps", 0)),
        stream_id=str(stream["id"]),
        # SIPL produces full-range pictures; the bridge must say so or the encoded stream signals limited range.
        full_range=config["capture_backend"] == "sipl",
        sipl=config["capture_backend"] == "sipl",
    )


def mock_sender_command(python: str, script: Path, settings: SenderSettings, width: int, height: int) -> list[str]:
    """The local mock sender that stands in for the Orin: one HEVC stream shaped like the real one."""
    return [
        python,
        str(script),
        "--port",
        str(settings.port),
        "--stream",
        f"{settings.stream_id}:{width}x{height}@{settings.fps}",
    ]


def bridge_timing_arguments(settings: SenderSettings, mock: bool, orin_host: str | None) -> list[str]:
    """The bridge flags that say what the frame timestamps mean and where the sender's clock can be measured."""
    if mock:
        # No sender clock to measure and no SIPL range to signal.
        return ["--timestamp-basis", MOCK_TIMESTAMP_BASIS]
    flags = ["--timestamp-basis", TIMESTAMP_BASIS, "--sender-clock", f"{orin_host}:{settings.clock_port}"]
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
    ) -> Managed:
        """Launch a child in its own process group so stopping it also stops anything it spawned.

        ``leader_only`` sends the polite SIGINT to the child itself and not to its whole group, for a child that
        closes its own helper processes in an order of its own (the backend's video writers).
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
        child = Managed(name, process, log, grace_s, leader_only)
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
        default=os.environ.get(
            "EGOCAP_RECEIVER",
            shlex.join(
                ["python3", str(REPO_ROOT / "examples/camera_streamer/camera_receiver/camera_receiver.py")]
            ),
        ),
        help="command that runs the camera receiver (takes --config FILE)",
    )
    parser.add_argument("--release", default="egocap-start", help="release name for the Orin deploy")
    parser.add_argument("--skip-deploy", action="store_true", help="reuse the release named by --release")
    parser.add_argument(
        "--sender-config", type=Path, default=DEFAULT_SENDER_CONFIG, help="sender profile uploaded to the Orin"
    )
    parser.add_argument("--station-config", type=Path, default=DEFAULT_STATION_CONFIG)
    parser.add_argument("--width", type=int, default=2560, help="sensor width the receiver and descriptor expect")
    parser.add_argument(
        "--height", type=int, default=1984, help="sensor height the receiver and descriptor expect"
    )
    parser.add_argument("--fps", type=int, help="override the profile's output.fps")
    parser.add_argument("--bitrate-mbps", type=int, help="override the profile's HEVC bitrate")
    parser.add_argument("--clock-port", type=int, help="override the profile's UDP clock port (sender and bridge)")
    parser.add_argument("--sender-port", type=int, help="override the profile's TCP video port")
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
    room, publisher, track = "egocap-room", "egocap-bridge", "sensing-head"
    livekit_url = "ws://127.0.0.1:7880"
    workdir = Path(tempfile.mkdtemp(prefix="egocap-start-"))
    ipc_socket = str(workdir / "camera_receiver.sock")
    stack = Stack(workdir)
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
        sender_config = load_sender_config(
            args.sender_config, args.fps, args.bitrate_mbps, args.clock_port, args.sender_port
        )
        sender = sender_settings(sender_config)
        # The bridge's sender-clock address must be a literal IPv4 address, not a name.
        if args.orin:
            try:
                ipaddress.IPv4Address(orin_host)
            except ValueError as error:
                raise StartupError(
                    f"--orin needs an IPv4 address for the sender clock, got {orin_host!r}"
                ) from error
        (workdir / "sender.yaml").write_text(yaml.safe_dump(sender_config, sort_keys=False))
        (workdir / "streams.yaml").write_text(
            streams_yaml(
                livekit_url, room, publisher, track, args.width, args.height, sender.fps, basis, sender.stream_id
            )
        )
        (workdir / "receiver.yaml").write_text(
            receiver_yaml(
                orin_host or "127.0.0.1", sender.port, ipc_socket, args.width, args.height, sender.stream_id
            )
        )
        if args.mock:
            print(f"mock sender: {sender.fps} fps stream {sender.stream_id!r}, timestamps are frame-ready time")
        else:
            print(
                f"sender config {args.sender_config}: {sender.fps} fps, {sender.bitrate_mbps} Mbit/s, "
                f"timing extension on, clock port {sender.clock_port}, release {args.release}"
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
            str(EGOCAP_ROOT / sub) for sub in ("backend", "recording", "capture_server", ".")
        )

        # 1. The Orin sender: deploy a release (never activating a service), then run it for a bounded time.
        if args.orin:
            deploy = [python, args.deploy_script, args.orin]
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
                "mock-sender", mock_sender_command(python, args.mock_sender, sender, args.width, args.height)
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
                    "--stream",
                    f"{sender.stream_id}={track}:2",
                    "--fps",
                    str(sender.fps),
                    *bridge_timing_arguments(sender, args.mock, orin_host),
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
        # Stay up until a child dies or the operator interrupts.
        while all(child.process.poll() is None for child in stack.children):
            time.sleep(1)
        dead = next(child for child in stack.children if child.process.poll() is not None)
        raise StartupError(f"{dead.name} exited with {dead.process.returncode}; see {dead.log}")
    except KeyboardInterrupt:
        return 0
    except (StartupError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        stack.stop_all()


if __name__ == "__main__":
    sys.exit(main())
