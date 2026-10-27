# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Teleop Session Server as a program: a server subprocess and robot apps over loopback.

Servers run with --placeholder-openxr (no OpenXR runtime, KeyboardTracker only) and
--no-launch-cloudxr-runtime. Every process a server starts inherits a marker variable, so
teardown can show that none of them outlived it.
"""

import json
import os
import queue
import re
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path

import pytest

import isaaccapture
from isaaccapture.deviceio_session import (
    RemotePeerClosedError,
    RemoteSession,
    RemoteSessionListener,
)
from isaaccapture.deviceio_trackers import KeyboardTracker
from isaaccapture.retargeting_engine.deviceio_source_nodes import HeadSource
from isaaccapture.teleop_session_manager import (
    DeviceState,
    ProviderState,
    SessionMode,
    StatusReason,
    TeleopSession,
    TeleopSessionConfig,
)
from isaaccapture.teleop_session_server import __main__ as server_main
from isaaccapture.teleop_session_server import _session

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the Teleop Session Server runs on Linux only"
)

WAIT_MS = 10_000
BANNER = re.compile(r"^Teleop Session Server listening on ((tcp|ws)://\S+)$")
MARKER = "ISAACCAPTURE_TEST_SERVER_MARKER"
BUSY = "busy: another robot app is connected to this Teleop Session Server"
SHUTDOWN = "The Teleop Session Server is shutting down"
WAITING = re.compile(r"Robot app at \S+ waits up to 4 s for the one at \S+ to leave")
# What a server without an OpenXR runtime reports: no headset is tracked.
UNTRACKED_HEADSET = {
    "state": "AVAILABLE",
    "headset_state": "DISCONNECTED",
    "reason": "TRACKING_UNAVAILABLE",
}
KEYBOARD = {"name": "0", "type": "KeyboardTracker", "params": {}, "vendor": None}


def _link_message(message_type, payload=b""):
    """A link message (remote_link/message_header.hpp): magic, type, version 0, no sequence."""
    return (
        struct.pack("<IBBHIIq", 0x30535452, message_type, 0, 0, 0, len(payload), 0)
        + payload
    )


HEARTBEAT = _link_message(11)


def _wait_for(predicate, timeout_s=10.0):
    """The first truthy value of predicate(), or its last value after timeout_s."""
    deadline = time.monotonic() + timeout_s
    while True:
        value = predicate()
        if value or time.monotonic() > deadline:
            return value
        time.sleep(0.005)


# ============================================================================
# Processes
# ============================================================================


def _running(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return text[text.rindex(")") + 2 :].split()[0] != "Z"


def _marked(marker):
    """Running processes whose environment carries `marker`."""
    entry = f"{MARKER}={marker}".encode()
    found = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            environ = Path(f"/proc/{name}/environ").read_bytes()
        except OSError:
            continue
        if entry in environ.split(b"\0") and _running(int(name)):
            found.append(int(name))
    return found


class _Server:
    """A Teleop Session Server subprocess, started in its own process group."""

    def __init__(self, tmp_path, *args, websocket=True):
        self.marker = uuid.uuid4().hex
        self.log_path = tmp_path / f"server-{self.marker[:8]}.log"
        env = dict(os.environ)
        # Leads its own logging, so its console output lands in log_path.
        env.pop("ISAACCAPTURE_LOG_SOCKET", None)
        env["ISAACCAPTURE_LOG_DIR"] = str(tmp_path / "logs")
        env[MARKER] = self.marker
        command = [
            sys.executable,
            "-m",
            "isaaccapture.teleop_session_server",
            "--no-launch-cloudxr-runtime",
            "--placeholder-openxr",
            "--listen",
            "tcp://127.0.0.1:0",
            *(["--ws-listen", "ws://127.0.0.1:0"] if websocket else []),
            *args,
        ]
        with open(self.log_path, "wb") as log:
            self.proc = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=log,
                env=env,
                start_new_session=True,
                text=True,
            )
        self.endpoints = {}
        self._lines = queue.Queue()
        threading.Thread(target=self._read_stdout, daemon=True).start()
        self._read_banners(2 if websocket else 1)

    def _read_stdout(self):
        for line in self.proc.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def _read_banners(self, count):
        deadline = time.monotonic() + 60.0
        while len(self.endpoints) < count:
            try:
                line = self._lines.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                line = None
            if line is None:
                self.kill()
                pytest.fail(f"the server printed no listening banner:\n{self.log()}")
            match = BANNER.match(line.rstrip("\n"))
            if match:
                self.endpoints[match.group(2)] = match.group(1)

    def address(self):
        """(host, port) of the TCP listener."""
        host, port = self.endpoints["tcp"].removeprefix("tcp://").rsplit(":", 1)
        return host, int(port)

    def log(self):
        return self.log_path.read_text(errors="replace")

    def stop(self, sig=signal.SIGTERM):
        """Signal the server and wait for it to exit; its exit code."""
        if self.proc.poll() is None:
            self.proc.send_signal(sig)
        try:
            return self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.kill()
            pytest.fail(f"the server did not stop:\n{self.log()}")

    def kill(self):
        for pid in [self.proc.pid, *_marked(self.marker)]:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.proc.wait()

    def close(self):
        """Stop the server; fail if it, or anything it started, did not end cleanly."""
        code = self.stop()
        if not _wait_for(lambda: not _marked(self.marker)):
            self.kill()
            pytest.fail(f"processes outlived the server:\n{self.log()}")
        assert code == 0, self.log()


@pytest.fixture
def start_server(tmp_path):
    servers = []

    def start(*args, websocket=True):
        server = _Server(tmp_path, *args, websocket=websocket)
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.close()


# ============================================================================
# Robot apps
# ============================================================================


def _robot():
    keyboard = KeyboardTracker()
    return keyboard, RemoteSession.create([(keyboard, "0")])


def _hello(session, protocol=1, **fields):
    hello = {
        "protocol": protocol,
        "isaaccapture_version": isaaccapture.__version__,
        "app_name": "ServerTest",
        "tick_rate_hz": 120.0,
        "trackers": [KEYBOARD],
        "topics": session.subscribed_topics(),
    }
    hello.update(fields)
    return json.dumps(hello)


@contextmanager
def _ready_robot(server, transport="tcp"):
    """A robot app's RemoteSession that the server reported READY to."""
    _, session = _robot()
    with session:
        session.connect(server.endpoints[transport], _hello(session), WAIT_MS)
        assert session.wait_ready(WAIT_MS), server.log()
        yield session


def _status(session):
    return json.loads(session.latest_status())


def _fresh(session):
    session.update()
    return session.is_fresh and session.tick_age_ns is not None


def _update_error(session):
    try:
        session.update()
    except RuntimeError as error:
        return str(error)
    return None


@contextmanager
def _peer_without_hello(server):
    """A connection that keeps its link alive but sends no HELLO."""
    stop = threading.Event()

    def beat(sock):
        while not stop.wait(0.5):
            try:
                sock.sendall(HEARTBEAT)
            except OSError:
                return

    with socket.create_connection(server.address()) as sock:
        sock.sendall(HEARTBEAT)
        thread = threading.Thread(target=beat, args=(sock,), daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join()


# ============================================================================
# Streaming
# ============================================================================


@pytest.mark.parametrize("transport", ["tcp", "ws"])
def test_robot_app_reads_the_server(start_server, transport):
    server = start_server()
    keyboard, session = _robot()
    with session:
        session.connect(server.endpoints[transport], _hello(session), WAIT_MS)
        ack = json.loads(session.hello_ack())
        assert re.fullmatch(r"[0-9a-f]{32}", ack["session_id"])
        assert ack["server_version"] == isaaccapture.__version__
        assert session.wait_ready(WAIT_MS)
        # READY, or already the first "running" STATUS after it.
        status = _status(session)
        assert status.pop("stage") in ("ready", "running")
        assert status == {"message": "", "headset": UNTRACKED_HEADSET}

        assert _wait_for(lambda: _fresh(session)), server.log()
        assert abs(session.tick_age_ns) < 250_000_000
        # The server's keyboard has no provider, so every tick records it as absent.
        assert keyboard.get_keyboard_data(session) is None
        assert _wait_for(lambda: _status(session)["stage"] == "running")
    assert _wait_for(lambda: "disconnected" in server.log()), server.log()


@pytest.mark.parametrize("transport", ["tcp", "ws"])
def test_a_robot_app_can_reconnect_right_after_leaving(start_server, transport):
    server = start_server()
    for _ in range(3):
        with _ready_robot(server, transport) as session:
            assert _wait_for(lambda: _fresh(session))


def test_a_robot_app_right_after_a_connection_that_sent_nothing_is_served(
    start_server,
):
    server = start_server(websocket=False)
    # A health probe.
    socket.create_connection(server.address()).close()
    with _ready_robot(server) as session:
        assert _wait_for(lambda: _fresh(session))


def test_a_robot_app_that_leaves_before_hello_frees_the_server(start_server):
    server = start_server(websocket=False)
    with socket.create_connection(server.address()):
        assert _wait_for(lambda: "Robot app connected from" in server.log())
    assert _wait_for(lambda: "left before HELLO" in server.log()), server.log()
    with _ready_robot(server) as session:
        assert _wait_for(lambda: _fresh(session))


def test_a_hello_that_requests_plugins_is_refused(start_server):
    server = start_server(websocket=False)
    plugin = {
        "plugin_name": "manus_hand_plugin",
        "plugin_root_id": "manus",
        "plugin_args": [],
        "required": True,
    }
    _, session = _robot()
    with session:
        with pytest.raises(RuntimeError) as raised:
            session.connect(
                server.endpoints["tcp"], _hello(session, plugins=[plugin]), WAIT_MS
            )
    assert str(raised.value) == (
        "The robot app requested plugins ['manus_hand_plugin'], but plugins are started on "
        "the Teleop Session Server machine (by hand or a script), not requested by the "
        "robot app"
    )
    with _ready_robot(server) as session:
        assert _wait_for(lambda: _fresh(session))


# ============================================================================
# One robot app at a time
# ============================================================================


@pytest.mark.parametrize("transport", ["tcp", "ws"])
def test_a_robot_app_is_refused_busy_while_another_is_connected(
    start_server, transport
):
    server = start_server()
    with _ready_robot(server) as first:
        _, waiting = _robot()
        _, third = _robot()
        with waiting, third:
            waiting.start(server.endpoints[transport], _hello(waiting), WAIT_MS)
            assert _wait_for(lambda: WAITING.search(server.log())), server.log()
            waiting_since = time.monotonic()
            # One already waits, so the next is refused at once.
            with pytest.raises(RuntimeError) as raised:
                third.connect(server.endpoints[transport], _hello(third), WAIT_MS)
            assert str(raised.value) == BUSY
            assert time.monotonic() - waiting_since < 2.0, server.log()
            with pytest.raises(RuntimeError) as raised:
                waiting.wait_hello_ack(WAIT_MS)
            assert str(raised.value) == BUSY
            waited = time.monotonic() - waiting_since
            grace = server_main.PENDING_GRACE_S
            assert grace - 0.5 <= waited < grace + 3.0, server.log()
        # The first robot app kept its SAMPLES throughout.
        assert _wait_for(lambda: _fresh(first))
    refusals = re.findall(
        r"Refused the robot app at \S+: the one at \S+ is connected", server.log()
    )
    assert len(refusals) == 2, server.log()


def test_a_waiting_robot_app_is_served_once_the_connected_one_leaves(start_server):
    server = start_server(websocket=False)
    _, first = _robot()
    _, waiting = _robot()
    with waiting:
        with first:
            first.connect(server.endpoints["tcp"], _hello(first), WAIT_MS)
            assert first.wait_ready(WAIT_MS)
            waiting.start(server.endpoints["tcp"], _hello(waiting), WAIT_MS)
            assert _wait_for(lambda: WAITING.search(server.log())), server.log()
        assert waiting.wait_hello_ack(WAIT_MS), server.log()
        assert waiting.wait_ready(WAIT_MS)
        assert _wait_for(lambda: _fresh(waiting))


def test_a_robot_app_reconnecting_after_its_link_dropped_is_served(start_server):
    """The dropped link times out after 3 s, within the reconnecting robot app's wait."""
    server = start_server(websocket=False)
    _, template = _robot()
    hello = _hello(template).encode()
    with socket.create_connection(server.address()) as dropped:
        # Sends its HELLO, then nothing: a robot app whose network went away.
        dropped.sendall(_link_message(1, hello))
        assert _wait_for(lambda: "Ready; streaming" in server.log()), server.log()
        with _ready_robot(server) as session:
            assert _wait_for(lambda: _fresh(session))
    log = server.log()
    assert WAITING.search(log), log
    assert re.search(r"Robot app at \S+ disconnected: no messages for \d+ ms", log), log
    assert "is connected" not in log


# ============================================================================
# Refusals
# ============================================================================


@pytest.mark.parametrize(
    "scheme, port_of, error",
    [
        ("tcp", "ws", r"^Remote session ended before HELLO_ACK: "),
        ("ws", "tcp", r"^WebSocket upgrade to \S+ refused: HTTP/1\.1 400 Bad Request$"),
    ],
    ids=["tcp_on_ws_port", "ws_on_tcp_port"],
)
@pytest.mark.parametrize("busy", [False, True], ids=["idle", "busy"])
def test_the_wrong_scheme_for_the_port_fails(
    start_server, scheme, port_of, error, busy
):
    server = start_server()
    port = server.endpoints[port_of].rsplit(":", 1)[1]
    with ExitStack() as stack:
        first = stack.enter_context(_ready_robot(server)) if busy else None
        _, robot = _robot()
        with robot:
            start = time.monotonic()
            with pytest.raises(RuntimeError, match=error) as raised:
                robot.connect(f"{scheme}://127.0.0.1:{port}", _hello(robot), WAIT_MS)
            # A busy server holds it as a robot app that waits for the connected one.
            limit = 5.0 + (server_main.PENDING_GRACE_S if busy else 0.0)
            assert time.monotonic() - start < limit, server.log()
            # Answered, not closed: a robot app retries a close.
            assert not isinstance(raised.value, RemotePeerClosedError)
        if first is not None:
            assert _wait_for(lambda: _fresh(first))
    # The server still answers a stop signal at once.
    start = time.monotonic()
    assert server.stop() == 0
    assert time.monotonic() - start < 5.0, server.log()


def test_another_protocol_is_refused(start_server):
    server = start_server(websocket=False)
    _, session = _robot()
    with session:
        with pytest.raises(RuntimeError, match=r"^Remote session protocol 2 is not"):
            session.connect(
                server.endpoints["tcp"], _hello(session, protocol=2), WAIT_MS
            )


def test_a_hello_that_is_not_utf8_is_refused_at_once(start_server):
    server = start_server(websocket=False)
    _, session = _robot()
    hello = _hello(session).encode().replace(b'"ServerTest"', b'"\xff"')
    with session:
        with pytest.raises(RuntimeError) as raised:
            session.connect(server.endpoints["tcp"], hello, WAIT_MS)
    # An answer, not a close, which a robot app would retry.
    assert type(raised.value) is RuntimeError
    assert str(raised.value).startswith(
        "HELLO is not valid UTF-8: 'utf-8' codec can't decode byte 0xff"
    ), server.log()


class _Pipeline:
    def __init__(self, leaf_nodes):
        self._leaf_nodes = list(leaf_nodes)

    def get_leaf_nodes(self):
        return self._leaf_nodes

    def execute_pipeline(self, inputs, context=None):
        return {}


def test_teleop_session_raises_the_server_error_verbatim(start_server):
    """TeleopSession in SessionMode.REMOTE against the server: its HELLO parses, its error returns."""
    server = start_server(websocket=False)
    config = TeleopSessionConfig(
        app_name="ServerTest",
        pipeline=_Pipeline([HeadSource(name="head")]),
        mode=SessionMode.REMOTE,
        remote_endpoint=server.endpoints["tcp"],
        remote_connect_timeout_s=30.0,
    )
    with pytest.raises(RuntimeError) as raised:
        with TeleopSession(config):
            pass
    assert str(raised.value) == (
        "This Teleop Session Server runs with --placeholder-openxr and serves only "
        "KeyboardTracker, not HeadTracker (tracker '0')"
    )


def test_teleop_session_shows_a_server_without_openxr_as_no_headset(start_server):
    """A placeholder server's headset entry is well formed: DISCONNECTED, not a poll error."""
    server = start_server(websocket=False)
    config = TeleopSessionConfig(
        app_name="ServerTest",
        pipeline=_Pipeline([]),
        mode=SessionMode.REMOTE,
        remote_endpoint=server.endpoints["tcp"],
        remote_connect_timeout_s=30.0,
    )

    def openxr_status(session):
        return (
            session.get_provider_status("openxr/runtime"),
            session.get_device_status("openxr/headset"),
        )

    with TeleopSession(config) as session:
        before_step = openxr_status(session)
        session.step()
        after_step = openxr_status(session)
    for provider, headset in (before_step, after_step):
        assert (provider.status, provider.reason) == (
            ProviderState.AVAILABLE,
            StatusReason.NONE,
        )
        assert (headset.status, headset.reason) == (
            DeviceState.DISCONNECTED,
            StatusReason.NO_CURRENT_DATA,
        )
        assert "headset DISCONNECTED (TRACKING_UNAVAILABLE)" in headset.error


@pytest.mark.parametrize(
    "scheme, port_of, error_type, error",
    [
        ("tcp", "ws", RuntimeError, r"^Remote session ended before HELLO_ACK: "),
        (
            "ws",
            "tcp",
            TimeoutError,
            r"server did not answer \(WebSocket upgrade to \S+ refused: "
            r"HTTP/1\.1 400 Bad Request\)",
        ),
    ],
    ids=["tcp_on_ws_port", "ws_on_tcp_port"],
)
def test_teleop_session_with_the_wrong_scheme_for_the_port_fails_at_once(
    start_server, scheme, port_of, error_type, error
):
    server = start_server()
    port = server.endpoints[port_of].rsplit(":", 1)[1]
    config = TeleopSessionConfig(
        app_name="ServerTest",
        pipeline=_Pipeline([]),
        mode=SessionMode.REMOTE,
        remote_endpoint=f"{scheme}://127.0.0.1:{port}",
        remote_connect_timeout_s=30.0,
    )
    start = time.monotonic()
    with pytest.raises(error_type, match=error):
        with TeleopSession(config):
            pass
    assert time.monotonic() - start < 5.0, server.log()


# ============================================================================
# Stopping
# ============================================================================


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT], ids=["term", "int"])
def test_stopping_the_server_tells_its_robot_apps(start_server, sig):
    server = start_server(websocket=False)
    with _ready_robot(server) as session:
        _, waiting = _robot()
        with waiting:
            waiting.start(server.endpoints["tcp"], _hello(waiting), WAIT_MS)
            assert _wait_for(lambda: WAITING.search(server.log())), server.log()
            start = time.monotonic()
            assert server.stop(sig) == 0
            assert time.monotonic() - start < 5.0, server.log()
            assert _wait_for(lambda: _update_error(session)) == SHUTDOWN
            with pytest.raises(RuntimeError) as raised:
                waiting.wait_hello_ack(WAIT_MS)
            assert str(raised.value) == SHUTDOWN
    log = server.log()
    assert f"Received {sig.name}; stopping the Teleop Session Server" in log
    assert "KeyboardInterrupt" not in log
    # Listening stopped, so a robot app keeps retrying until a restarted server serves it.
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(server.address(), timeout=5.0)


def test_a_stop_signal_while_a_connection_sends_no_hello_stops_promptly(start_server):
    server = start_server(websocket=False)
    with _peer_without_hello(server):
        assert _wait_for(lambda: "Robot app connected from" in server.log())
        start = time.monotonic()
        assert server.stop() == 0
        assert time.monotonic() - start < _session.HELLO_TIMEOUT_S, server.log()


def test_the_server_refuses_a_port_in_use():
    taken = RemoteSessionListener("127.0.0.1", 0)
    argv = ["--no-launch-cloudxr-runtime", "--listen", f"tcp://127.0.0.1:{taken.port}"]
    assert server_main.main(argv) == 1


# ============================================================================
# The loop in process
# ============================================================================


class _Launcher:
    def __init__(self, on_check):
        self.checks = []
        self._on_check = on_check

    def health_check(self):
        self.checks.append(time.monotonic())
        self._on_check(len(self.checks))


def _listening():
    return server_main._Listening(
        RemoteSessionListener("127.0.0.1", 0), False, "tcp://127.0.0.1"
    )


def test_a_stopping_server_refuses_waiting_robot_apps_and_stops_listening():
    stop = server_main._StopRequest()
    stop.signal = signal.SIGTERM
    listening = _listening()
    port = listening.listener.port
    server = server_main._Server(
        [listening], _Launcher(lambda _count: None), stop, placeholder=True
    )
    _, robot = _robot()
    with robot:
        # In the listen backlog when the server stops.
        robot.start(f"127.0.0.1:{port}", _hello(robot), WAIT_MS)
        assert server.run() == 0
        with pytest.raises(RuntimeError) as raised:
            robot.wait_hello_ack(WAIT_MS)
        assert str(raised.value) == SHUTDOWN
    # Refused, so a robot app keeps retrying until a restarted server serves it.
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(("127.0.0.1", port), timeout=5.0)


def test_the_server_stops_with_code_1_when_cloudxr_stops():
    """The robot app it serves hears that the server is shutting down."""
    cloudxr_stopped = threading.Event()

    def check(_count):
        if cloudxr_stopped.is_set():
            raise RuntimeError("The CloudXR runtime serving /run has stopped")

    listening = _listening()
    port = listening.listener.port
    server = server_main._Server(
        [listening], _Launcher(check), server_main._StopRequest(), placeholder=True
    )
    result = {}
    thread = threading.Thread(target=lambda: result.update(code=server.run()))
    thread.start()
    try:
        _, session = _robot()
        with session:
            session.connect(f"127.0.0.1:{port}", _hello(session), WAIT_MS)
            assert session.wait_ready(WAIT_MS)
            cloudxr_stopped.set()
            assert _wait_for(lambda: _update_error(session)) == SHUTDOWN
    finally:
        cloudxr_stopped.set()
        thread.join(timeout=30)
    assert result == {"code": 1}


def test_the_server_checks_cloudxr_about_once_a_second():
    stop = server_main._StopRequest()

    def check(count):
        if count == 3:
            stop.signal = signal.SIGTERM

    launcher = _Launcher(check)
    server = server_main._Server([_listening()], launcher, stop, placeholder=True)
    assert server.run() == 0
    gaps = [b - a for a, b in zip(launcher.checks, launcher.checks[1:])]
    assert len(gaps) == 2
    assert all(0.9 <= gap < 2.0 for gap in gaps), gaps
