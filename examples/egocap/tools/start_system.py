# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Starts the whole Ego Capture stack with one command and stops it together on Ctrl-C.

    uv run python tools/start_system.py --orin user@orin --deploy-script PATH --bridge PATH --livekit-server PATH

In order: deploy and start the Orin sender over ssh, a LiveKit server, CaptureServer, the backend (which owns
the recorder), the camera receiver and the camera-to-LiveKit bridge. Without ``--orin`` no sender is started, so
only the PC side comes up. Each process is waited on until it is ready before the next one starts, and all of
them are stopped, last started first, when this script exits for any reason.

The sender settings come from the checked-in profile ``configs/default-camera-settings.yaml`` (see
CAMERA_SETTINGS.md). CaptureServer runs real acquisition (a built IsaacCapture checkout and tracking hardware)
unless ``--mock`` gives it synthetic tracking and cameras.

The sender's deploy tool, the LiveKit server and the bridge are not part of this tree, so their paths are options
(or ``EGOCAP_DEPLOY_SCRIPT``, ``EGOCAP_LIVEKIT_SERVER``, ``EGOCAP_BRIDGE``). The pure helpers sit above the
supervisor so they can be tested without starting anything.
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
    url: str, room: str, publisher: str, track: str, width: int, height: int, fps: int, basis: str
) -> str:
    """The CaptureServer ``--livekit-streams`` file for one camera published by the bridge."""
    return (
        "livekit:\n"
        f"  url: {url}\n"
        f"  room: {room}\n"
        f"  publisher: {publisher}\n"
        f"  timestamp_basis: {basis}\n"
        "  streams:\n"
        f"    - {{name: head, track: {track}, width: {width}, height: {height}, fps: {fps}}}\n"
    )


def receiver_yaml(host: str, port: int, socket_path: str, width: int, height: int, name: str) -> str:
    """The camera receiver's config: connect to the sender and expose one mono stream over CUDA IPC."""
    return (
        "version: 1\n"
        "receiver:\n"
        f"  host: {host}\n"
        f"  port: {port}\n"
        "  gpu_id: 0\n"
        "  max_record_bytes: 268435456\n"
        "  stats_log_interval_s: 2\n"
        "ipc:\n"
        f"  socket_path: {socket_path}\n"
        "  owner_name: egocap_start_system\n"
        "  queue_size: 1024\n"
        "streams:\n"
        "  - index: 0\n"
        f"    name: {name}\n"
        "    type: mono\n"
        f"    width: {width}\n"
        f"    height: {height}\n"
        "    output_format: nv12\n"
        "    ring_depth: 6\n"
        "    ipc:\n"
        f"      mono: {name}\n"
    )


@dataclass(frozen=True)
class SenderSettings:
    """The effective sender values the rest of the stack must agree with."""

    port: int
    clock_port: int
    fps: int
    bitrate_mbps: int
    stream_id: str
    full_range: bool


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
    except OSError as error:
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
        full_range=config.get("capture_backend") == "sipl",
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
    ) -> Managed:
        """Launch a child in its own process group so stopping it also stops anything it spawned."""
        # Each child writes to its own file so a failed step can point the operator at the output to read.
        log = self.workdir / f"{name}.log"
        handle = log.open("w")
        # A new session makes the child the leader of its own process group, so a signal to the group reaches
        # anything it spawned and never this script.
        process = subprocess.Popen(
            list(command),
            stdout=handle,
            stderr=subprocess.STDOUT,
            env={**os.environ, **(env or {})},
            cwd=cwd,
            start_new_session=True,
        )
        # Registering the child here is what makes stop_all() cover it even if a later step fails.
        child = Managed(name, process, log, grace_s)
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

    def stop_all(self) -> None:
        """Stop the children last started first, waiting for each to exit before signalling the next.

        A child later in the start order depends on the ones before it (the backend needs CaptureServer's live
        connection to close a take), so it must be gone before its dependencies are signalled. Each child gets its
        own grace period before it is killed. A second Ctrl-C while waiting kills the rest at once.
        """
        hard = False
        for child in reversed(self.children):
            if child.process.poll() is not None:
                continue
            if hard:
                self._kill(child)
                continue
            # SIGINT lets the receiver and sender drain and free the GPU and cameras.
            if not self._signal_group(child, signal.SIGINT):
                child.process.poll()
                continue
            try:
                child.process.wait(timeout=child.grace_s)
            except subprocess.TimeoutExpired:
                print(f"{child.name} did not exit within {child.grace_s:.0f} s; killing it", file=sys.stderr)
                self._kill(child)
            except KeyboardInterrupt:
                # An impatient operator: stop waiting for this child and for every one after it.
                hard = True
                self._kill(child)


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
            f"python3 {REPO_ROOT / 'examples/camera_streamer/camera_receiver/camera_receiver.py'}",
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
        help="run CaptureServer with synthetic tracking and cameras, no hardware needed",
    )
    parser.add_argument("--backend-port", type=int, default=8010)
    parser.add_argument("--seconds", type=int, default=3600, help="how long the Orin sender runs")
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
    # Children must stop however this script ends, including an exception in the middle of startup.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    try:
        # The deploy is checked first so a missing tool fails before anything is running.
        if args.orin and not args.deploy_script:
            raise StartupError("--orin needs --deploy-script (or EGOCAP_DEPLOY_SCRIPT)")
        if args.orin and not (args.bridge and args.livekit_server):
            raise StartupError("--orin needs --bridge and --livekit-server to carry the video to the page")
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
            streams_yaml(livekit_url, room, publisher, track, args.width, args.height, sender.fps, TIMESTAMP_BASIS)
        )
        (workdir / "receiver.yaml").write_text(
            receiver_yaml(
                orin_host or "127.0.0.1", sender.port, ipc_socket, args.width, args.height, sender.stream_id
            )
        )
        print(
            f"sender config {args.sender_config}: {sender.fps} fps, {sender.bitrate_mbps} Mbit/s, "
            f"timing extension on, clock port {sender.clock_port}, release {args.release}"
        )
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
                subprocess.run([*deploy, "deploy", "--with-sipl", "--name", args.release], check=True)
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
                "--livekit-streams",
                str(workdir / "streams.yaml"),
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
        )
        stack.wait_until(
            backend, lambda: port_open("127.0.0.1", args.backend_port), f"listening on {args.backend_port}", 60
        )
        # 5. The receiver decodes the sender's stream; the bridge republishes it into the LiveKit room.
        if args.orin:
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
                    "--timestamp-basis",
                    TIMESTAMP_BASIS,
                    "--sender-clock",
                    f"{orin_host}:{sender.clock_port}",
                    *(["--source-full-range"] if sender.full_range else []),
                ],
            )
        url = f"http://127.0.0.1:{args.backend_port}/"
        token = control_token(backend.log)
        print(f"\nEgo Capture is up at {url}\nrecording control token: {token}\nlogs: {workdir}")
        if args.mock:
            # A recording made here mixes synthetic tracking and cameras with the real stream.
            print("--mock: tracking and the CaptureServer cameras are synthetic.")
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
