# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Teleop Session Server as a program: a server subprocess and robot apps over loopback.

Servers run with --placeholder-openxr (no OpenXR runtime, KeyboardTracker only) and
--no-launch-cloudxr-runtime. Every process a server starts inherits a marker variable, so
teardown can show that none of them outlived it.
"""

import json
import logging
import os
import queue
import re
import select
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
from isaaccapture.plugin_manager import PluginCrashException
from isaaccapture.retargeting_engine.deviceio_source_nodes import HeadSource
from isaaccapture.teleop_session_manager import (
    DeviceState,
    ProviderState,
    SessionMode,
    StatusReason,
    TeleopSession,
    TeleopSessionConfig,
    _remote,
)
from isaaccapture.teleop_session_server import __main__ as server_main
from isaaccapture.teleop_session_server import _session

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the Teleop Session Server runs on Linux only"
)

WAIT_MS = 10_000
BANNER = re.compile(r"^Teleop Session Server listening on ((tcp|ws)://\S+)$")
MARKER = "ISAACCAPTURE_TEST_SERVER_MARKER"
SESSION_MODULE = b"isaaccapture.teleop_session_server._session"
BUSY = "busy: another robot app is connected to this Teleop Session Server"
SHUTDOWN = "The Teleop Session Server is shutting down"
# What a server without an OpenXR runtime reports: no headset is tracked.
UNTRACKED_HEADSET = {
    "state": "AVAILABLE",
    "headset_state": "DISCONNECTED",
    "reason": "TRACKING_UNAVAILABLE",
}
KEYBOARD = {"name": "0", "type": "KeyboardTracker", "params": {}, "vendor": None}
# A HEARTBEAT link message (remote_link/message_header.hpp): magic, type 11, version 0, no payload.
HEARTBEAT = struct.pack("<IBBHIIq", 0x30535452, 11, 0, 0, 0, 0, 0)

# Runs until a file named fail_now appears in its directory, then exits with code 3. A stop
# signal leaves a file named after it there.
PROBE_PLUGIN = """\
import os, signal, sys, time

def stop(signum, frame):
    open("stopped_by_" + signal.Signals(signum).name, "w").close()
    sys.exit(0)

signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)
while not os.path.exists("fail_now"):
    time.sleep(0.02)
sys.exit(3)
"""


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


def _stat(pid):
    """(state, ppid) of a process, or None once it is gone."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    fields = text[text.rindex(")") + 2 :].split()
    return fields[0], int(fields[1])


def _running(pid):
    stat = _stat(pid)
    return stat is not None and stat[0] != "Z"


def _cmdline(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return []


def _pids():
    return [int(entry) for entry in os.listdir("/proc") if entry.isdigit()]


def _marked(marker):
    """Running processes whose environment carries `marker`."""
    entry = f"{MARKER}={marker}".encode()
    found = []
    for pid in _pids():
        try:
            environ = Path(f"/proc/{pid}/environ").read_bytes()
        except OSError:
            continue
        if entry in environ.split(b"\0") and _running(pid):
            found.append(pid)
    return found


class _Server:
    """A Teleop Session Server subprocess, started in its own process group."""

    def __init__(self, tmp_path, *args, websocket=True):
        self.marker = uuid.uuid4().hex
        self.log_path = tmp_path / f"server-{self.marker[:8]}.log"
        self.expected_exit_code = 0
        env = dict(os.environ)
        # Leads its own logging, so its console output lands in log_path.
        env.pop("ISAACCAPTURE_LOG_SOCKET", None)
        env.pop("ISAAC_TELEOP_PLUGIN_PATH", None)
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

    def log(self):
        return self.log_path.read_text(errors="replace")

    def session_pids(self):
        """Running session processes of this server.

        An exiting process loses its cmdline before it becomes a zombie, so to wait for one to
        end, wait on its pid (_running) rather than on this list.
        """
        pids = []
        for pid in _pids():
            stat = _stat(pid)
            if stat is None or stat[0] == "Z" or stat[1] != self.proc.pid:
                continue
            if SESSION_MODULE in _cmdline(pid):
                pids.append(pid)
        return pids

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
            strays = {pid: _cmdline(pid) for pid in _marked(self.marker)}
            self.kill()
            pytest.fail(f"processes outlived the server: {strays}\n{self.log()}")
        assert code == self.expected_exit_code, self.log()


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


def _hello(session, plugins=(), protocol=1):
    return json.dumps(
        {
            "protocol": protocol,
            "isaaccapture_version": isaaccapture.__version__,
            "app_name": "ServerTest",
            "tick_rate_hz": 120.0,
            "trackers": [KEYBOARD],
            "topics": session.subscribed_topics(),
            "plugins": list(plugins),
        }
    )


@contextmanager
def _ready_robot(server, transport="tcp", plugins=()):
    """A robot app's RemoteSession that the server reported READY to."""
    _, session = _robot()
    with session:
        session.connect(server.endpoints[transport], _hello(session, plugins), WAIT_MS)
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


def _plugin_dir(tmp_path, command=None):
    """A plugin search directory holding probe_plugin; (search directory, plugin directory)."""
    root = tmp_path / "plugins"
    plugin = root / "probe_plugin"
    plugin.mkdir(parents=True)
    (plugin / "probe.py").write_text(PROBE_PLUGIN)
    if command is None:
        assert " " not in sys.executable, "PluginManager splits the command on spaces"
        command = f"{sys.executable} probe.py"
    (plugin / "plugin.yaml").write_text(
        f"name: probe_plugin\ndescription: test plugin\ncommand: {command}\n"
    )
    return root, plugin


def _stopped_by(plugin_dir):
    """The signals that stopped probe_plugin."""
    return sorted(
        path.name.removeprefix("stopped_by_")
        for path in plugin_dir.glob("stopped_by_*")
    )


def _request(name="probe_plugin", required=True, args=()):
    return {
        "plugin_name": name,
        "plugin_root_id": "probe",
        "plugin_args": list(args),
        "required": required,
    }


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
        assert status == {"message": "", "plugins": [], "headset": UNTRACKED_HEADSET}

        assert _wait_for(lambda: _fresh(session)), server.log()
        assert abs(session.tick_age_ns) < 250_000_000
        # The server's keyboard has no provider, so every tick records it as absent.
        assert keyboard.get_keyboard_data(session) is None
        assert _wait_for(lambda: _status(session)["stage"] == "running")
        [session_pid] = server.session_pids()
    assert _wait_for(lambda: not _running(session_pid)), server.log()


def test_a_second_robot_app_is_refused_while_one_is_connected(start_server):
    server = start_server()
    with _ready_robot(server) as first:
        for transport in ("tcp", "ws"):
            _, second = _robot()
            with second:
                with pytest.raises(RuntimeError) as raised:
                    second.connect(server.endpoints[transport], _hello(second), WAIT_MS)
            assert str(raised.value) == BUSY
        assert _wait_for(lambda: _fresh(first))
        assert len(server.session_pids()) == 1
    # Each refused robot app: one INFO line on its wait, then the refusal.
    lines = server.log().splitlines()
    waits = [i for i, line in enumerate(lines) if "for the previous session" in line]
    refusals = [i for i, line in enumerate(lines) if "Refused the robot app" in line]
    assert len(waits) == len(refusals) == 2, server.log()
    assert waits[0] < refusals[0] < waits[1] < refusals[1], server.log()
    waiting = re.compile(
        rf"\[INFO \].* waits up to {server_main.HANDOVER_TIMEOUT_S:g} s for the "
        r"previous session \(pid \d+\) to end$"
    )
    assert all(waiting.search(lines[i]) for i in waits), server.log()
    assert all("is connected" in lines[i] for i in refusals), server.log()


def test_the_session_ends_with_its_robot_app_and_frees_the_server(
    start_server, tmp_path
):
    search_dir, plugin_dir = _plugin_dir(tmp_path)
    server = start_server("--plugin-dir", str(search_dir), websocket=False)
    plugins = [_request(required=False, args=["--side=left"])]
    with _ready_robot(server, plugins=plugins) as session:
        [session_pid] = server.session_pids()
        [entry] = _status(session)["plugins"]
        assert (entry["plugin_name"], entry["plugin_root_id"]) == (
            "probe_plugin",
            "probe",
        )
        assert (entry["state"], entry["reason"]) == ("RUNNING", "NONE")
        plugin_pid = entry["pid"]
        assert _running(plugin_pid)
        assert {b"--plugin-root-id=probe", b"--side=left"} <= set(_cmdline(plugin_pid))
    assert _wait_for(lambda: not _running(session_pid) and not _running(plugin_pid))
    assert _stopped_by(plugin_dir) == ["SIGINT"]

    with _ready_robot(server) as session:
        assert _wait_for(lambda: _fresh(session))


@pytest.mark.parametrize("transport", ["tcp", "ws"])
def test_a_robot_app_can_reconnect_right_after_leaving(
    start_server, tmp_path, transport
):
    search_dir, _ = _plugin_dir(tmp_path)
    server = start_server("--plugin-dir", str(search_dir))
    for _ in range(3):
        # The previous session may still be stopping its plugin; the server waits for it.
        with _ready_robot(server, transport, plugins=[_request()]) as session:
            assert _wait_for(lambda: _fresh(session))


def test_a_robot_app_right_after_a_connection_that_sent_nothing_is_served(
    start_server,
):
    server = start_server(websocket=False)
    host, port = server.endpoints["tcp"].removeprefix("tcp://").rsplit(":", 1)
    # A health probe: its session process is still starting when the robot app connects.
    socket.create_connection((host, int(port))).close()
    with _ready_robot(server) as session:
        assert _wait_for(lambda: _fresh(session))


def test_a_robot_app_that_leaves_before_hello_frees_the_server(start_server):
    server = start_server(websocket=False)
    host, port = server.endpoints["tcp"].removeprefix("tcp://").rsplit(":", 1)
    with socket.create_connection((host, int(port))):
        [session_pid] = _wait_for(server.session_pids) or [None]
        assert session_pid is not None, server.log()
    assert _wait_for(lambda: not _running(session_pid)), server.log()
    with _ready_robot(server) as session:
        assert _wait_for(lambda: _fresh(session))


# ============================================================================
# Plugins
# ============================================================================


def test_a_plugin_failure_reaches_the_robot_app(start_server, tmp_path):
    search_dir, plugin_dir = _plugin_dir(tmp_path)
    server = start_server("--plugin-dir", str(search_dir), websocket=False)
    with _ready_robot(server, plugins=[_request()]) as session:
        (plugin_dir / "fail_now").touch()

        def failed_entry():
            [entry] = _status(session)["plugins"]
            return entry if entry["state"] != "RUNNING" else None

        entry = _wait_for(failed_entry)
        assert entry, server.log()
        assert (entry["state"], entry["reason"], entry["exit_code"]) == (
            "EXITED",
            "NONZERO_EXIT",
            3,
        )
        assert entry["error"] == "Plugin process unexpectedly exited with code 3"
        # What the robot app's TeleopSession makes of it.
        proxy = _remote.RemotePluginContext(
            _remote.RemoteStatus(session), "probe_plugin", "probe"
        )
        with pytest.raises(PluginCrashException, match="exited with code 3"):
            proxy.check_health()
        # SAMPLES keep coming: the robot app decides what a plugin failure means.
        assert _wait_for(lambda: _fresh(session))
    assert _wait_for(
        lambda: (
            "is no longer running: Plugin process unexpectedly exited with code 3"
            in server.log()
        )
    ), server.log()


def test_a_required_plugin_the_server_lacks_is_an_error(start_server):
    server = start_server(websocket=False)
    _, session = _robot()
    with session:
        with pytest.raises(RuntimeError) as raised:
            session.connect(
                server.endpoints["tcp"],
                _hello(session, [_request("no_such_plugin")]),
                WAIT_MS,
            )
    assert str(raised.value).startswith(
        "Required plugin 'no_such_plugin' was not discovered on the Teleop Session "
        "Server in ["
    )
    assert _wait_for(lambda: not server.session_pids())


def test_an_optional_plugin_the_server_lacks_is_skipped(start_server):
    server = start_server(websocket=False)
    plugins = [_request("no_such_plugin", required=False)]
    with _ready_robot(server, plugins=plugins) as session:
        assert _status(session)["plugins"] == []


def test_a_plugin_that_cannot_start_is_an_error(start_server, tmp_path):
    search_dir, _ = _plugin_dir(tmp_path, command="./no_such_binary")
    server = start_server("--plugin-dir", str(search_dir), websocket=False)
    _, session = _robot()
    with session:
        session.connect(server.endpoints["tcp"], _hello(session, [_request()]), WAIT_MS)
        with pytest.raises(RuntimeError) as raised:
            session.wait_ready(WAIT_MS)
    # PluginManager's own words follow, as TeleopSession reports them locally.
    assert str(raised.value).startswith(
        "Plugin 'probe_plugin' could not start on the Teleop Session Server: "
        "Plugin process exited immediately: failed to execute plugin command "
        "'./no_such_binary'"
    )


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
def test_the_wrong_scheme_for_the_port_fails_at_once(
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
            assert time.monotonic() - start < 5.0, server.log()
            # Answered, not closed: a robot app retries a close.
            assert not isinstance(raised.value, RemotePeerClosedError)
        if first is not None:
            assert _wait_for(lambda: _fresh(first))
    # The main process still answers a stop signal at once.
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
def test_stopping_the_server_ends_the_robot_session(start_server, tmp_path, sig):
    search_dir, plugin_dir = _plugin_dir(tmp_path)
    server = start_server("--plugin-dir", str(search_dir), websocket=False)
    with _ready_robot(server, plugins=[_request()]) as session:
        [session_pid] = server.session_pids()
        plugin_pid = _status(session)["plugins"][0]["pid"]
        assert server.stop(sig) == 0
        assert not _running(session_pid)
        assert _wait_for(lambda: not _running(plugin_pid))
        # The session's teardown stopped it, as TeleopSession stops a local plugin.
        assert _stopped_by(plugin_dir) == ["SIGINT"]
        assert _wait_for(lambda: _update_error(session)) == SHUTDOWN


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT], ids=["term", "int"])
def test_a_stop_signal_to_the_session_process_tells_the_robot_app(start_server, sig):
    server = start_server(websocket=False)
    with _ready_robot(server) as session:
        [session_pid] = server.session_pids()
        os.kill(session_pid, sig)
        assert _wait_for(lambda: _update_error(session)) == SHUTDOWN, server.log()
    exited = re.compile(
        rf"Session process {session_pid} for \S+ exited with code (-?\d+)"
    )
    match = _wait_for(lambda: exited.search(server.log()))
    assert match is not None and match.group(1) == "0", server.log()
    assert "KeyboardInterrupt" not in server.log()
    with _ready_robot(server) as session:
        assert _wait_for(lambda: _fresh(session))


def test_the_session_and_its_plugins_end_when_the_main_process_dies(
    start_server, tmp_path
):
    search_dir, plugin_dir = _plugin_dir(tmp_path)
    server = start_server("--plugin-dir", str(search_dir), websocket=False)
    with _ready_robot(server, plugins=[_request()]) as session:
        [session_pid] = server.session_pids()
        plugin_pid = _status(session)["plugins"][0]["pid"]
        server.expected_exit_code = -signal.SIGKILL
        assert server.stop(signal.SIGKILL) == -signal.SIGKILL
        # PR_SET_PDEATHSIG stops the session, which stops its plugin and tells the robot app.
        assert _wait_for(lambda: not _running(session_pid) and not _running(plugin_pid))
        assert _stopped_by(plugin_dir) == ["SIGINT"]
        assert _wait_for(lambda: _update_error(session)) == SHUTDOWN


@contextmanager
def _peer_without_hello(endpoint):
    """A connection that keeps its link alive but sends no HELLO.

    Its session process reports nothing until its HELLO wait runs out.
    """
    host, port = endpoint.removeprefix("tcp://").rsplit(":", 1)
    stop = threading.Event()

    def beat(sock):
        while not stop.wait(0.5):
            try:
                sock.sendall(HEARTBEAT)
            except OSError:
                return

    with socket.create_connection((host, int(port))) as sock:
        sock.sendall(HEARTBEAT)
        thread = threading.Thread(target=beat, args=(sock,), daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join()


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT], ids=["term", "int"])
def test_a_stop_signal_during_a_slow_handover_stops_the_server_promptly(
    start_server, sig
):
    server = start_server("--verbose", websocket=False)
    with _peer_without_hello(server.endpoints["tcp"]):
        assert _wait_for(lambda: "Waiting for HELLO from" in server.log()), server.log()
        hello_wait_started = time.monotonic()
        [first_pid] = server.session_pids()
        _, robot = _robot()
        with robot:
            robot.start(server.endpoints["tcp"], _hello(robot), WAIT_MS)
            assert _wait_for(lambda: "for the previous session" in server.log()), (
                server.log()
            )
            assert server.stop(sig) == 0
            # Before the first session's HELLO wait would have run out.
            stopped_after = time.monotonic() - hello_wait_started
            assert stopped_after < _session.HELLO_TIMEOUT_MS / 1000, server.log()
            assert not _running(first_pid)
            assert not _marked(server.marker)
            with pytest.raises(RuntimeError) as raised:
                robot.wait_hello_ack(WAIT_MS)
            assert str(raised.value) == SHUTDOWN


def test_a_robot_app_waiting_for_the_previous_session_is_logged_once(start_server):
    server = start_server(websocket=False)
    _, robot = _robot()
    with robot:
        with _peer_without_hello(server.endpoints["tcp"]):
            [first_pid] = _wait_for(server.session_pids) or [None]
            assert first_pid is not None, server.log()
            robot.start(server.endpoints["tcp"], _hello(robot), WAIT_MS)
            waiting = (
                f"waits up to {server_main.HANDOVER_TIMEOUT_S:g} s for the previous "
                f"session (pid {first_pid}) to end"
            )
            assert _wait_for(lambda: waiting in server.log()), server.log()
            # The wait goes on for several of the main process's wait slices.
            time.sleep(10 * server_main.WAIT_SLICE_S)
        # The first connection closed, so its session ends and the robot app is served.
        assert robot.wait_hello_ack(WAIT_MS), server.log()
    log = server.log()
    [line] = [line for line in log.splitlines() if "for the previous session" in line]
    assert re.search(rf"\[INFO \].* Robot app at \S+ {re.escape(waiting)}$", line), log


@pytest.mark.parametrize("stopped_by, exit_code", [("signal", 0), ("cloudxr", 1)])
def test_a_handover_wait_checks_for_stop_and_cloudxr(
    monkeypatch, stopped_by, exit_code
):
    monkeypatch.setattr(server_main, "HANDOVER_TIMEOUT_S", 30.0)
    stop = server_main._StopRequest()

    def check(count):
        # The robot app's handover wait spans the second and third checks.
        if count < 3:
            return
        if stopped_by == "signal":
            stop.signal = signal.SIGTERM
        else:
            raise RuntimeError("The CloudXR runtime serving /run has stopped")

    launcher = _Launcher(check)
    listening = server_main._Listening(
        RemoteSessionListener("127.0.0.1", 0), False, "tcp://127.0.0.1"
    )
    # A session process that never reports, as one still starting.
    command = [sys.executable, "-c", "import time; time.sleep(60)"]
    server = server_main._Server([listening], command, launcher, stop)
    endpoint = f"127.0.0.1:{listening.listener.port}"
    _, robot = _robot()
    # Both wait in the listen backlog: the first gets the session, the robot app waits for it.
    with socket.create_connection(("127.0.0.1", listening.listener.port)), robot:
        robot.start(endpoint, _hello(robot), WAIT_MS)
        start = time.monotonic()
        assert server.run() == exit_code
        assert time.monotonic() - start < 10.0
        with pytest.raises(RuntimeError) as raised:
            robot.wait_hello_ack(WAIT_MS)
        assert str(raised.value) == SHUTDOWN
    gaps = [b - a for a, b in zip(launcher.checks, launcher.checks[1:])]
    assert len(gaps) == 2
    assert all(0.9 <= gap < 2.0 for gap in gaps), gaps


def test_a_stopping_server_refuses_waiting_robot_apps_and_stops_listening():
    stop = server_main._StopRequest()
    stop.signal = signal.SIGTERM
    listening = server_main._Listening(
        RemoteSessionListener("127.0.0.1", 0), False, "tcp://127.0.0.1"
    )
    port = listening.listener.port
    server = server_main._Server(
        [listening],
        [sys.executable, "-c", "pass"],
        _Launcher(lambda _count: None),
        stop,
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


def test_a_session_waiting_for_hello_stops_within_the_grace_period(start_server):
    server = start_server("--verbose", websocket=False)
    host, port = server.endpoints["tcp"].removeprefix("tcp://").rsplit(":", 1)
    with socket.create_connection((host, int(port))):
        assert _wait_for(lambda: "Waiting for HELLO from" in server.log()), server.log()
        start = time.monotonic()
        assert server.stop() == 0
        stopped_in = time.monotonic() - start
    log = server.log()
    assert "did not stop within" not in log, log
    assert stopped_in < server_main.SESSION_STOP_TIMEOUT_S, log


def test_the_server_refuses_a_port_in_use():
    taken = RemoteSessionListener("127.0.0.1", 0)
    argv = ["--no-launch-cloudxr-runtime", "--listen", f"tcp://127.0.0.1:{taken.port}"]
    assert server_main.main(argv) == 1


class _Launcher:
    def __init__(self, on_check):
        self.checks = []
        self._on_check = on_check

    def health_check(self):
        self.checks.append(time.monotonic())
        self._on_check(len(self.checks))


def _accept_loop(launcher):
    stop = server_main._StopRequest()
    listening = server_main._Listening(
        RemoteSessionListener("127.0.0.1", 0), False, "tcp://127.0.0.1"
    )
    command = [sys.executable, "-c", "pass"]
    return server_main._Server([listening], command, launcher, stop), stop


def test_a_session_process_stuck_after_its_robot_app_left_is_killed(monkeypatch):
    monkeypatch.setattr(server_main, "HANDOVER_TIMEOUT_S", 1.0)
    # Reports its robot app gone at once, then never exits.
    script = (
        "import os, sys, time; "
        "os.close(int(sys.argv[sys.argv.index('--serving-fd') + 1])); "
        "time.sleep(60)"
    )
    listening = server_main._Listening(
        RemoteSessionListener("127.0.0.1", 0), False, "tcp://127.0.0.1"
    )
    server = server_main._Server(
        [listening],
        [sys.executable, "-c", script],
        _Launcher(lambda _count: None),
        server_main._StopRequest(),
    )
    pairs = [socket.socketpair() for _ in range(2)]
    try:
        server._admit(listening, pairs[0][0].detach(), "first")
        stuck = server._session.popen
        start = time.monotonic()
        server._admit(listening, pairs[1][0].detach(), "second")
        assert time.monotonic() - start < 5.0
        assert stuck.returncode == -signal.SIGKILL
        assert server._session.peer == "second"
    finally:
        server._end_session()
        for _, robot_end in pairs:
            robot_end.close()


# Reports a 30 s teardown time (0.5 s late with --late), and takes argv[1] seconds over its
# teardown after SIGTERM; with --leave, its robot app leaves at once and the teardown starts.
SLOW_TEARDOWN_SESSION = """\
import os, signal, sys, time

def teardown(signum=None, frame=None):
    time.sleep(float(sys.argv[1]))
    sys.exit(0)

signal.signal(signal.SIGTERM, teardown)
fd = int(sys.argv[sys.argv.index("--serving-fd") + 1])
if "--late" in sys.argv:
    time.sleep(0.5)
os.write(fd, b"30.0")
if "--leave" in sys.argv:
    os.close(fd)
    teardown()
time.sleep(60)
"""


@contextmanager
def _slow_teardown_server(*args):
    """A main process whose session processes run SLOW_TEARDOWN_SESSION with ``args``.

    Yields (server, admit), where admit(peer) hands the server a connection from ``peer``.
    """
    listening = server_main._Listening(
        RemoteSessionListener("127.0.0.1", 0), False, "tcp://127.0.0.1"
    )
    server = server_main._Server(
        [listening],
        [sys.executable, "-c", SLOW_TEARDOWN_SESSION, *args],
        _Launcher(lambda _count: None),
        server_main._StopRequest(),
    )
    robot_ends = []

    def admit(peer):
        server_end, robot_end = socket.socketpair()
        robot_ends.append(robot_end)
        server._admit(listening, server_end.detach(), peer)
        return server._session

    try:
        yield server, admit
    finally:
        server._end_session(kill=True)
        for robot_end in robot_ends:
            robot_end.close()


def _reported(session):
    """Wait for the session process's report without reading it; whether it came."""
    poller = select.poll()
    poller.register(session.serving_fd, select.POLLIN)
    return bool(poller.poll(WAIT_MS))


def test_a_robot_app_waits_for_the_previous_session_to_stop_its_plugins(monkeypatch):
    monkeypatch.setattr(server_main, "HANDOVER_TIMEOUT_S", 0.2)
    with _slow_teardown_server("1.0", "--leave") as (server, admit):
        ending = admit("first")
        assert _reported(ending)
        admit("second")
        # Its teardown ran to the end, past HANDOVER_TIMEOUT_S, within the 30 s it reported.
        assert ending.popen.returncode == 0
        assert server._session.peer == "second"


def test_a_handover_announces_a_longer_bound_once_a_starting_session_reports_it(
    monkeypatch, caplog
):
    monkeypatch.setattr(server_main, "BUSY_GRACE_S", 10.0)
    with _slow_teardown_server("1.0", "--late", "--leave") as (server, admit):
        ending = admit("first")
        with caplog.at_level(logging.INFO, logger="isaaccapture.teleop_session_server"):
            admit("second")
        assert ending.popen.returncode == 0
        assert server._session.peer == "second"
    bounds = [
        re.search(r"waits up to (\S+) s", message)[1]
        for message in caplog.messages
        if "for the previous session" in message
    ]
    assert bounds == [f"{server_main.HANDOVER_TIMEOUT_S:g}", "30"], caplog.messages


def test_stopping_waits_for_the_session_to_stop_its_plugins(monkeypatch):
    monkeypatch.setattr(server_main, "SESSION_STOP_TIMEOUT_S", 0.2)
    with _slow_teardown_server("1.0") as (server, admit):
        session = admit("robot")
        assert _reported(session)
        server._end_session()
        assert session.teardown_s == 30.0
        assert session.popen.returncode == 0


def test_another_stop_signal_kills_a_stopping_session_at_once():
    with _slow_teardown_server("60") as (server, admit):
        session = admit("robot")
        assert _reported(session)
        threading.Timer(0.5, server._stop, (signal.SIGINT, None)).start()
        start = time.monotonic()
        server._end_session()
        assert time.monotonic() - start < 5.0
        assert session.popen.returncode == -signal.SIGKILL


def test_the_signal_that_stops_the_server_leaves_the_session_its_teardown():
    with _slow_teardown_server("1.0") as (server, admit):
        session = admit("robot")
        assert _reported(session)
        server._stop(signal.SIGINT, None)
        assert server._stopping()
        server._end_session()
        assert session.popen.returncode == 0


def test_a_second_stop_signal_before_the_wait_kills_the_session_at_once():
    with _slow_teardown_server("60") as (server, admit):
        session = admit("robot")
        assert _reported(session)
        server._stop(signal.SIGINT, None)
        assert server._stopping()
        # As while the server refuses the robot apps in its listen backlog.
        server._stop(signal.SIGINT, None)
        start = time.monotonic()
        server._end_session()
        assert time.monotonic() - start < 5.0
        assert session.popen.returncode == -signal.SIGKILL


def test_each_stop_signal_counts_when_python_handles_them_once():
    """Python runs the handler once for both signals, which arrive during one native call."""
    wakeup_read, wakeup_write = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
    stop = server_main._StopRequest(wakeup_read)
    listener = RemoteSessionListener("127.0.0.1", 0)
    previous = signal.signal(signal.SIGTERM, stop)
    previous_wakeup_fd = signal.set_wakeup_fd(wakeup_write, warn_on_full_buffer=False)

    def send_two():
        time.sleep(0.2)
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.03)
        os.kill(os.getpid(), signal.SIGTERM)

    sender = threading.Thread(target=send_two)
    try:
        sender.start()
        # Holds no GIL until it times out, well after both signals.
        assert listener.accept(1000) is None
        sender.join()
        assert stop.signal == signal.SIGTERM
        assert stop.count == 2
    finally:
        signal.set_wakeup_fd(previous_wakeup_fd)
        signal.signal(signal.SIGTERM, previous)
        listener.close()
        os.close(wakeup_read)
        os.close(wakeup_write)


# A session process whose PluginManager cannot be created: a failure before HELLO_ACK.
BROKEN_SESSION = """\
import sys
import isaaccapture.plugin_manager as pm
from isaaccapture.teleop_session_server import _session

def broken(search_paths):
    raise OSError("no plugin directory today")

pm.PluginManager = broken
sys.exit(_session.main(sys.argv[1:]))
"""


def test_a_session_that_fails_before_hello_ack_tells_the_robot_app(caplog):
    """The robot app raises the server's ERROR at once instead of taking it for a crash."""
    stop = server_main._StopRequest()
    listening = server_main._Listening(
        RemoteSessionListener("127.0.0.1", 0), False, "tcp://127.0.0.1"
    )
    command = [
        sys.executable,
        "-c",
        BROKEN_SESSION,
        "--parent-pid",
        str(os.getpid()),
        "--placeholder-openxr",
    ]
    config = TeleopSessionConfig(
        app_name="ServerTest",
        pipeline=_Pipeline([]),
        mode=SessionMode.REMOTE,
        remote_endpoint=f"tcp://127.0.0.1:{listening.listener.port}",
        remote_connect_timeout_s=30.0,
    )
    outcome = {}

    def robot_app():
        start = time.monotonic()
        try:
            with TeleopSession(config):
                pass
        except Exception as error:  # noqa: BLE001 - checked below
            outcome["error"] = error
        outcome["after_s"] = time.monotonic() - start

    robot = threading.Thread(target=robot_app)

    def check(count):
        # Once the robot app is done and the session process has exited; a minute at most.
        exited = any("exited with code" in m for m in caplog.messages)
        if (not robot.is_alive() and exited) or count > 60:
            stop.signal = signal.SIGTERM

    server = server_main._Server([listening], command, _Launcher(check), stop)
    robot.start()
    with caplog.at_level(logging.INFO, logger="isaaccapture.teleop_session_server"):
        try:
            assert server.run() == 0
        finally:
            robot.join(timeout=60)
    error = outcome.get("error")
    assert type(error) is RuntimeError, outcome
    assert str(error) == (
        "The Teleop Session Server failed: OSError: no plugin directory today"
    )
    # Not retried: one connection, one session process.
    assert outcome["after_s"] < 20.0
    messages = caplog.messages
    assert len([m for m in messages if m.startswith("Robot app connected from")]) == 1
    assert any(
        re.fullmatch(r"Session process \d+ for \S+ exited with code 1", m)
        for m in messages
    ), messages


def test_the_server_stops_when_cloudxr_stops():
    def stopped(_count):
        raise RuntimeError("The CloudXR runtime serving /run has stopped")

    server, _ = _accept_loop(_Launcher(stopped))
    assert server.run() == 1


def test_the_server_checks_cloudxr_about_once_a_second():
    def check(count):
        if count == 3:
            stop.signal = signal.SIGTERM

    launcher = _Launcher(check)
    server, stop = _accept_loop(launcher)
    assert server.run() == 0
    gaps = [b - a for a, b in zip(launcher.checks, launcher.checks[1:])]
    assert len(gaps) == 2
    assert all(0.9 <= gap < 2.0 for gap in gaps), gaps
