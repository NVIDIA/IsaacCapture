# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Robot state on the Teleop Session Server: the HELLO key, HELLO_ACK, and the store it hosts.

The HELLO and RobotConnection tests run in process against stand-in hosts. The last tests start
the server (``--placeholder-openxr``, no OpenXR runtime) on a free loopback port, with its reader
socket in a temporary folder, and read what a robot app sends with ``RobotState.connect()``. The
other programs that serve a reader socket are processes the test starts and ends; a local robot
app runs with mocks for OpenXR, DeviceIO and replay.
"""

import dataclasses
import json
import logging
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from test_reader_socket_retry import TAKEN, OtherProgram

import isaaccapture
import isaaccapture.robot_state
import isaaccapture.teleop_session_server as server_package
from isaaccapture.deviceio_session import RemoteSession, RemoteSessionListener
from isaaccapture.robot_state import (
    RobotState,
    RobotStateStore,
    default_socket_path,
    parse_robot_description,
    records,
)
from isaaccapture.teleop_session_manager import (
    SessionMode,
    TeleopSession,
    TeleopSessionConfig,
)
from isaaccapture.teleop_session_server import __main__ as server_main
from isaaccapture.teleop_session_server._session import (
    HelloError,
    RobotConnection,
    SessionRequest,
    parse_hello,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the Teleop Session Server runs on Linux only"
)

WAIT_MS = 10_000
BANNER = re.compile(r"^Teleop Session Server listening on (tcp://\S+)$")
BAD_ROBOT_STATE = r"^robot_state must be an object with a version number$"
NOT_ACTIVE = r"^TeleopSession is not active$"
NO_READERS = "Reader programs cannot get robot state from this Teleop Session Server: "
# What a local robot app's entry raises while another program serves its reader socket's path.
LOCAL_TAKEN = (
    "Another program on this machine already serves a robot's state at {path}: a Teleop "
    "Session Server or another robot app. One robot per machine is served; stop that "
    "program, then start this robot app again."
)
# How often a robot app starts a connection again that the server closed unanswered.
CONNECT_RETRY_S = 0.5
DESCRIPTION_TOML = """
robot = "test_arm"

[series.arm]
shape = "joints"
joints = [
  { name = "joint1",  kind = "revolute", min = -2.0, max = 2.0 },
  { name = "gripper", kind = "gripper",  min = 0.0,  max = 1.0 },
]

[series.status]
shape = "status"
stale_after_s = 0
readings = ["motors_on"]
"""
DESCRIPTION = parse_robot_description(DESCRIPTION_TOML)
OTHER_DESCRIPTION = parse_robot_description(
    DESCRIPTION_TOML.replace("test_arm", "other_arm").replace("joint1", "base")
)


def _hello(**fields):
    hello = {
        "protocol": 1,
        "isaaccapture_version": isaaccapture.__version__,
        "app_name": "Robot",
        "tick_rate_hz": 90.0,
        "trackers": [],
        "topics": [],
    }
    hello.update(fields)
    return json.dumps(hello)


def _wait_for(predicate, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while True:
        value = predicate()
        if value or time.monotonic() > deadline:
            return value
        time.sleep(0.01)


# ============================================================================
# HELLO
# ============================================================================


def test_a_hello_without_the_key_offers_no_robot_state():
    request = parse_hello(_hello(), placeholder=True)
    assert request.robot_state is False


@pytest.mark.parametrize("version", [1, 2, 7])
def test_a_hello_with_the_key_offers_robot_state(version):
    """A later version is served too: the answer names the version this server takes."""
    request = parse_hello(
        _hello(robot_state={"version": version, "later": "ignored"}), placeholder=True
    )
    assert request.robot_state is True


@pytest.mark.parametrize(
    "value",
    [None, True, 1, "1", [], [1], {}, {"version": None}, {"version": "1"}]
    + [{"version": True}, {"version": 1.0}, {"version": 0}, {"version": -1}],
    ids=repr,
)
def test_a_bad_robot_state_value_is_refused(value):
    with pytest.raises(HelloError, match=BAD_ROBOT_STATE):
        parse_hello(_hello(robot_state=value), placeholder=True)


def test_other_hello_errors_come_first():
    with pytest.raises(HelloError, match="topics must be a list"):
        parse_hello(_hello(topics=1, robot_state=True), placeholder=True)


def test_the_request_field_is_the_last_and_off_by_default():
    fields = dataclasses.fields(SessionRequest)
    assert fields[-1].name == "robot_state"
    assert fields[-1].default is False


# ============================================================================
# HELLO_ACK
# ============================================================================


class _HostBefore:
    """RemoteSessionHost stand-in of a build without robot state: no attach_robot_state()."""

    alive = True
    close_reason = ""

    def __init__(self, hello):
        self.hello = hello
        # (kind, text)
        self.sent = []

    def wait_hello(self, timeout_ms):
        return self.hello

    def send_hello_ack(self, text):
        self.sent.append(("hello_ack", text))

    def send_error(self, text):
        self.sent.append(("error", text))


class _Host(_HostBefore):
    def attach_robot_state(self, store):
        self.sent.append(("attach", store))


def _connect(host, **kwargs):
    connection = RobotConnection(
        host, "robot:1", placeholder=True, extensions=None, **kwargs
    )
    connection.step(None, None)
    return connection


def _plain_ack(text):
    """Whether a HELLO_ACK is the one a server built before robot state sends."""
    ack = json.loads(text)
    return (
        list(ack) == ["session_id", "server_version"]
        and ack["server_version"] == isaaccapture.__version__
        and text
        == json.dumps(
            {"session_id": ack["session_id"], "server_version": ack["server_version"]}
        )
    )


def test_a_robot_app_without_the_key_is_answered_as_before():
    """Old robot app, new server: a host without the new call serves it, the store is untouched."""
    store = RobotStateStore()
    called = []
    host = _HostBefore(_hello())
    connection = _connect(
        host, robot_state=store, on_robot_state=lambda: called.append(1)
    )
    ((kind, text),) = host.sent
    assert kind == "hello_ack"
    assert _plain_ack(text)
    assert not connection.ended
    assert called == []
    assert (store.connected, store.session_count) == (False, 0)


def test_a_connection_made_without_the_new_arguments_agrees_nothing():
    host = _Host(_hello(robot_state={"version": 1}))
    connection = _connect(host)
    ((kind, text),) = host.sent
    assert kind == "hello_ack"
    assert _plain_ack(text)
    assert not connection.ended


def test_a_server_without_a_store_agrees_nothing():
    called = []
    host = _Host(_hello(robot_state={"version": 1}))
    _connect(host, robot_state=None, on_robot_state=lambda: called.append(1))
    ((kind, text),) = host.sent
    assert kind == "hello_ack"
    assert _plain_ack(text)
    assert called == []


@pytest.mark.parametrize("version", [1, 3])
def test_a_robot_app_with_the_key_is_agreed_to(version, caplog):
    caplog.set_level(logging.INFO)
    store = RobotStateStore()
    called = []
    host = _Host(_hello(robot_state={"version": version}))
    connection = _connect(
        host, robot_state=store, on_robot_state=lambda: called.append(len(host.sent))
    )
    # The server hears of it first; then the store is attached, before the answer leaves.
    assert [kind for kind, _ in host.sent] == ["attach", "hello_ack"]
    assert host.sent[0][1] is store
    ack = json.loads(host.sent[1][1])
    assert list(ack) == ["session_id", "server_version", "robot_state"]
    assert ack["robot_state"] == {"version": 1}
    assert called == [0]
    assert not connection.ended
    assert "Robot app 'Robot' at robot:1 sends robot state" in caplog.text


def test_agreeing_needs_no_callback():
    host = _Host(_hello(robot_state={"version": 1}))
    _connect(host, robot_state=RobotStateStore())
    assert [kind for kind, _ in host.sent] == ["attach", "hello_ack"]


def _no_place():
    raise HelloError("no place for this robot's state")


def test_a_robot_app_the_server_refuses_gets_the_error_and_nothing_else(caplog):
    caplog.set_level(logging.INFO)
    store = RobotStateStore()
    host = _Host(_hello(robot_state={"version": 1}))
    connection = _connect(host, robot_state=store, on_robot_state=_no_place)
    # No store attached, no HELLO_ACK.
    assert host.sent == [("error", "no place for this robot's state")]
    assert connection.ended
    assert (store.connected, store.session_count, store.series()) == (False, 0, [])
    assert [
        (entry.levelno, entry.getMessage())
        for entry in caplog.records
        if entry.levelno >= logging.INFO
    ] == [
        (
            logging.WARNING,
            "Refused the robot app at robot:1: no place for this robot's state",
        )
    ]


def test_a_robot_app_without_the_key_is_served_where_one_with_it_is_refused():
    store = RobotStateStore()
    host = _HostBefore(_hello())
    connection = _connect(host, robot_state=store, on_robot_state=_no_place)
    ((kind, text),) = host.sent
    assert kind == "hello_ack"
    assert _plain_ack(text)
    assert not connection.ended
    assert store.session_count == 0


def test_a_hello_that_cannot_be_parsed_is_refused_before_the_server_hears_of_it():
    called = []
    host = _Host(_hello(topics=1, robot_state={"version": 1}))
    connection = _connect(
        host, robot_state=RobotStateStore(), on_robot_state=lambda: called.append(1)
    )
    assert host.sent == [("error", "topics must be a list")]
    assert connection.ended
    assert called == []


def test_a_bad_robot_state_value_ends_the_connection_with_its_reason():
    store = RobotStateStore()
    host = _Host(_hello(robot_state={"version": "1"}))
    connection = _connect(host, robot_state=store)
    assert host.sent == [
        ("error", "robot_state must be an object with a version number")
    ]
    assert connection.ended
    assert store.session_count == 0


# ============================================================================
# The server's store, in process
# ============================================================================


class _Launcher:
    def health_check(self):
        pass


def _server(socket_path=None):
    listening = server_main._Listening(
        RemoteSessionListener("127.0.0.1", 0), False, "tcp://127.0.0.1"
    )
    stop = server_main._StopRequest()
    server = server_main._Server(
        [listening],
        _Launcher(),
        stop,
        placeholder=True,
        robot_state_socket=socket_path,
    )
    return server, stop, listening.listener.port


@pytest.fixture
def socket_path(tmp_path_factory):
    # Short: a Unix socket path holds about a hundred bytes.
    return str(tmp_path_factory.mktemp("rs") / "rs.sock")


class _Running:
    """A _Server running on a thread until the block ends."""

    def __init__(self, socket_path):
        self.server, self._stop, self.port = _server(socket_path)
        self.code = None
        self._thread = threading.Thread(target=self._run)

    def _run(self):
        self.code = self.server.run()

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.signal = signal.SIGTERM
        self._thread.join(timeout=30)
        assert not self._thread.is_alive()
        return False


def _robot_hello(session, **fields):
    return _hello(app_name="ServerTest", topics=session.subscribed_topics(), **fields)


def test_no_server_in_this_process_hosts_no_robot_state():
    assert server_package.hosted_robot_state() is None


def test_a_server_hosts_one_store_for_its_life(socket_path):
    with _Running(socket_path) as running:
        hosted = server_package.hosted_robot_state()
        assert isinstance(hosted, RobotState)
        assert hosted.server_reachable
        assert not hosted.connected
        assert hosted.description() is None
        # Nothing is opened for readers until a robot app sends robot state.
        assert not os.path.exists(socket_path)

        with RemoteSession.create([]) as session:
            session.set_robot_state(records.declarations(DESCRIPTION))
            session.connect(
                f"127.0.0.1:{running.port}",
                _robot_hello(session, robot_state={"version": 1}),
                WAIT_MS,
            )
            assert json.loads(session.hello_ack())["robot_state"] == {"version": 1}
            session.start_robot_state()
            now_ns = time.monotonic_ns()
            session.publish_robot_record(
                records.DESCRIPTION_TOPIC,
                records.encode_description(DESCRIPTION, now_ns),
                now_ns,
            )
            session.publish_robot_record(
                "arm",
                records.encode_reading(DESCRIPTION, "arm", [0.5, 1.0], time_ns=now_ns),
                now_ns,
            )
            assert _wait_for(lambda: hosted.joints("arm"))
            assert hosted.connected
            assert hosted.description() == DESCRIPTION
            assert hosted.joints("arm")["joint1"].position == 0.5
            assert _wait_for(lambda: os.path.exists(socket_path))
            # A reader made later reads the same store.
            assert server_package.hosted_robot_state().joints("arm").positions == (
                0.5,
                1.0,
            )
        # The robot app left: the same reader says so, and keeps what the robot is.
        assert _wait_for(lambda: not hosted.connected)
        assert hosted.joints("arm") is None
        assert hosted.joints("arm", allow_stale=True)["gripper"].position == 1.0
        assert hosted.description() == DESCRIPTION
    assert running.code == 0
    assert server_package.hosted_robot_state() is None
    assert not os.path.exists(socket_path)


def test_a_robot_app_without_robot_state_opens_nothing(socket_path):
    with _Running(socket_path) as running:
        with RemoteSession.create([]) as session:
            session.connect(f"127.0.0.1:{running.port}", _robot_hello(session), WAIT_MS)
            assert _plain_ack(session.hello_ack())
            assert session.wait_ready(WAIT_MS)
            hosted = server_package.hosted_robot_state()
            assert not hosted.connected
            assert not os.path.exists(socket_path)
    assert running.code == 0


def test_a_reader_socket_that_cannot_be_opened_does_not_stop_the_server(
    socket_path, monkeypatch, caplog
):
    caplog.set_level(logging.WARNING)

    class _Refusing:
        def __init__(self, store, path):
            self.path = path

        def start(self):
            raise OSError(f"cannot bind {self.path}")

        def close(self):
            raise AssertionError("a service that never started is not closed")

    monkeypatch.setattr(isaaccapture.robot_state, "RobotStateLocalService", _Refusing)
    with _Running(socket_path) as running:
        with RemoteSession.create([]) as session:
            session.connect(
                f"127.0.0.1:{running.port}",
                _robot_hello(session, robot_state={"version": 1}),
                WAIT_MS,
            )
            assert session.wait_ready(WAIT_MS)
            # Robot state still reaches the server's own store.
            assert server_package.hosted_robot_state().connected
    assert running.code == 0
    assert (
        "Reader programs cannot get robot state from this Teleop Session Server: "
        f"cannot bind {socket_path}"
    ) in caplog.text


def test_the_default_reader_socket_is_named_only_when_it_is_opened():
    """A server made without a path, as the existing tests make it, touches no socket."""
    server, stop, _port = _server()
    assert server._robot_state_socket is None
    assert server._robot_state_service is None
    stop.signal = signal.SIGTERM
    assert server.run() == 0
    assert server_package.hosted_robot_state() is None


def test_the_option_is_optional():
    parser = server_main._build_parser()
    assert parser.parse_args([]).robot_state_socket is None
    args = parser.parse_args(["--robot-state-socket", "/tmp/somewhere/rs.sock"])
    assert args.robot_state_socket == "/tmp/somewhere/rs.sock"
    assert "--robot-state-socket PATH" in parser.format_help()


# ============================================================================
# The server program
# ============================================================================


class _Program:
    """A Teleop Session Server subprocess with its reader socket at `socket_path`.

    Without one, at the default path.
    """

    def __init__(self, tmp_path, socket_path=None):
        self.log_path = tmp_path / "server.log"
        env = dict(os.environ)
        # Leads its own logging, so its console output lands in log_path.
        env.pop("ISAACCAPTURE_LOG_SOCKET", None)
        env["ISAACCAPTURE_LOG_DIR"] = str(tmp_path / "logs")
        command = [
            sys.executable,
            "-m",
            "isaaccapture.teleop_session_server",
            "--no-launch-cloudxr-runtime",
            "--placeholder-openxr",
            "--listen",
            "tcp://127.0.0.1:0",
        ]
        if socket_path is not None:
            command += ["--robot-state-socket", socket_path]
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
        lines = queue.Queue()
        threading.Thread(
            target=lambda: [lines.put(line) for line in self.proc.stdout], daemon=True
        ).start()
        self.endpoint = None
        deadline = time.monotonic() + 60.0
        while self.endpoint is None:
            try:
                line = lines.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                self.proc.kill()
                self.proc.wait()
                pytest.fail(f"the server printed no listening banner:\n{self.log()}")
            match = BANNER.match(line.rstrip("\n"))
            if match:
                self.endpoint = match.group(1)

    def log(self):
        return self.log_path.read_text(errors="replace")

    def stop(self):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
        try:
            return self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
            pytest.fail(f"the server did not stop:\n{self.log()}")


@pytest.fixture
def program(tmp_path, socket_path):
    server = _Program(tmp_path, socket_path)
    yield server
    server.stop()


class _Pipeline:
    def get_leaf_nodes(self):
        return []

    def execute_pipeline(self, inputs, context=None):
        return {}


def _robot_app(program, **kwargs):
    return TeleopSession(
        TeleopSessionConfig(
            app_name="ServerTest",
            pipeline=_Pipeline(),
            mode=SessionMode.REMOTE,
            remote_endpoint=program.endpoint,
            remote_connect_timeout_s=30.0,
            verbose=False,
            **kwargs,
        )
    )


def test_the_server_program_hosts_the_robot_state_of_its_robot_app(
    program, socket_path
):
    """New robot app, new server: a reader program sees what the robot is and its readings."""
    with RobotState.connect(socket_path) as reader:
        # The reader socket appears with the first robot app that sends robot state.
        assert not reader.server_reachable
        assert reader.description() is None

        with _robot_app(program, robot_description=DESCRIPTION) as session:
            assert session.robot_state_active, program.log()
            assert _wait_for(lambda: reader.description()), program.log()
            assert reader.server_reachable
            assert reader.connected
            assert reader.description() == DESCRIPTION
            assert session.publish_robot_state("arm", [0.25, 1.0])
            assert session.publish_robot_state("status", [True])
            arm = _wait_for(lambda: reader.joints("arm"))
            assert arm is not None, program.log()
            assert arm.names == ("joint1", "gripper")
            assert arm["joint1"].position == 0.25
            assert 0.0 <= arm.age_s < 2.0
            assert reader.gripper("gripper").opening == 1.0
            assert _wait_for(lambda: reader.status())["motors_on"] is True
            session.step()

        # The robot app left; the server and the reader stay.
        assert _wait_for(lambda: not reader.connected), program.log()
        assert reader.server_reachable
        assert reader.status() is None
        assert reader.status(allow_stale=True)["motors_on"] is True

        # Another robot takes its place on the same server, for the same reader.
        with _robot_app(program, robot_description=OTHER_DESCRIPTION) as session:
            assert _wait_for(
                lambda: (reader.description() or DESCRIPTION).robot == "other_arm"
            ), program.log()
            assert reader.joints("arm") is None
            assert session.publish_robot_state("arm", [-0.5, 0.0])
            arm = _wait_for(lambda: reader.joints("arm"))
            assert arm.names == ("base", "gripper")
            assert arm.positions == (-0.5, 0.0)

    assert program.stop() == 0, program.log()
    assert not os.path.exists(socket_path)
    # Stopped: a reader finds no server.
    with RobotState.connect(socket_path) as reader:
        assert not reader.server_reachable
        assert reader.description() is None


def test_the_server_program_serves_a_robot_app_without_robot_state_as_before(
    program, socket_path
):
    """Old robot app, new server: nothing about robot state happens on the server."""
    with _robot_app(program) as session:
        assert not session.robot_state_active
        session.step()
        with RobotState.connect(socket_path) as reader:
            assert not reader.server_reachable
    assert not os.path.exists(socket_path)
    assert program.stop() == 0, program.log()
    assert "robot state" not in program.log()


# ============================================================================
# One robot per reader socket path
# ============================================================================


@pytest.fixture
def other_program():
    """Start other programs that serve a path; they are killed when the test ends."""
    programs = []

    def start(path):
        other = OtherProgram(path)
        programs.append(other)
        return other

    yield start
    for other in programs:
        if other.running:
            other.kill()


def _robot_at(path=None):
    """The robot a new reader at `path` reads; None when nothing answers."""
    with RobotState.connect(path) as reader:
        description = reader.description()
        return None if description is None else description.robot


def _open_files():
    """How many file descriptors this process holds."""
    return len(os.listdir("/proc/self/fd"))


def test_the_server_program_refuses_a_robot_while_another_process_serves_the_path(
    program, socket_path, other_program
):
    other = other_program(socket_path)
    text = TAKEN.format(path=socket_path)
    refused = _robot_app(program, robot_description=DESCRIPTION)
    for count in (1, 2):
        files, threads = _open_files(), threading.active_count()
        start = time.monotonic()
        with pytest.raises(RuntimeError) as raised:
            refused.__enter__()
        # The server's answer, long before remote_connect_timeout_s.
        assert time.monotonic() - start < 10.0, program.log()
        assert type(raised.value) is RuntimeError
        assert str(raised.value) == text
        assert not refused.robot_state_active
        with pytest.raises(RuntimeError, match=NOT_ACTIVE):
            refused.publish_robot_state("arm", [0.0, 0.0])
        # Raised once: the robot app does not connect again by itself.
        time.sleep(3 * CONNECT_RETRY_S)
        log = program.log()
        assert log.count("Robot app connected from") == count, log
        assert log.count("Refused the robot app at") == count, log
        assert log.count(f": {text}") == count, log
        if count == 2:
            # The first entry loaded what a process loads once; the second left nothing.
            assert _open_files() == files
            assert threading.active_count() == threads
    assert program.proc.poll() is None
    assert NO_READERS not in program.log()

    # The same robot app without a description is served, whoever holds the path.
    with _robot_app(program) as session:
        assert not session.robot_state_active
        session.step()
    # The other program and its readers are undisturbed.
    assert other.running
    assert _robot_at(socket_path) == "other_arm"

    other.leave()
    with RobotState.connect(socket_path, robot="test_arm") as reader:
        with refused as session:
            assert session.robot_state_active, program.log()
            assert _wait_for(lambda: reader.connected), program.log()
            assert reader.description() == DESCRIPTION
            assert session.publish_robot_state("arm", [0.25, 1.0])
            arm = _wait_for(lambda: reader.joints("arm"))
            assert arm is not None, program.log()
            assert arm["joint1"].position == 0.25
            session.step()
    assert program.stop() == 0, program.log()
    log = program.log()
    assert log.count("Refused the robot app at") == 2, log
    assert NO_READERS not in log


def test_the_server_program_serves_a_robot_whose_reader_socket_cannot_be_opened(
    program, socket_path
):
    """Not another program at the path: the robot app is served and the socket tried again."""
    with open(socket_path, "w") as notes:
        notes.write("notes")
    reason = f"{socket_path} exists and is not a socket"
    with _robot_app(program, robot_description=DESCRIPTION) as session:
        assert session.robot_state_active, program.log()
        assert session.publish_robot_state("arm", [0.25, 1.0])
        session.step()
        time.sleep(2.5 * server_main.HEALTH_CHECK_PERIOD_S)
        assert program.log().count(NO_READERS + reason) == 1, program.log()
        assert "Refused the robot app" not in program.log()

        os.unlink(socket_path)
        with RobotState.connect(socket_path, robot="test_arm") as reader:
            assert _wait_for(lambda: reader.connected), program.log()
            assert session.publish_robot_state("arm", [0.5, 1.0])
            assert _wait_for(
                lambda: (
                    (arm := reader.joints("arm")) is not None
                    and arm["joint1"].position == 0.5
                )
            ), program.log()
    assert program.stop() == 0, program.log()
    assert program.log().count(NO_READERS) == 1


@contextmanager
def _local_backends():
    """Replace OpenXR, DeviceIO and replay for a robot app in a local session."""
    with (
        patch("isaaccapture.oxr.OpenXRSession") as openxr,
        patch("isaaccapture.deviceio.DeviceIOSession.run") as live_run,
        patch(
            "isaaccapture.deviceio.DeviceIOSession.get_required_extensions",
            return_value=[],
        ),
        patch("isaaccapture.deviceio.ReplaySession.run") as replay_run,
        patch("isaaccapture.deviceio.McapReplayConfig"),
    ):
        yield openxr, live_run, replay_run


def _local_robot_app(mode, **kwargs):
    if mode == "replay":
        kwargs.update(mode=SessionMode.REPLAY, mcap_config=MagicMock())
    return TeleopSession(
        TeleopSessionConfig(
            app_name="LocalTest", pipeline=_Pipeline(), verbose=False, **kwargs
        )
    )


@pytest.mark.parametrize("mode", ["live", "replay"])
def test_a_local_robot_app_is_refused_while_the_server_program_keeps_the_path(
    tmp_path, tmp_path_factory, monkeypatch, mode
):
    """The server keeps its reader socket between two robots; both use the default path."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path_factory.mktemp("rt")))
    socket_path = default_socket_path()
    assert socket_path.startswith(os.environ["XDG_RUNTIME_DIR"])
    program = _Program(tmp_path)
    try:
        # A robot agrees robot state with the server, then leaves.
        with _robot_app(program, robot_description=DESCRIPTION) as session:
            assert session.robot_state_active, program.log()
            assert session.publish_robot_state("arm", [0.25, 1.0])
            assert _wait_for(lambda: _robot_at() == "test_arm"), program.log()

        with RobotState.connect() as reader, _local_backends() as backends:
            assert _wait_for(lambda: reader.server_reachable and not reader.connected)
            local = _local_robot_app(mode, robot_description=OTHER_DESCRIPTION)
            for _ in range(2):
                with pytest.raises(RuntimeError) as raised:
                    local.__enter__()
                assert str(raised.value) == LOCAL_TAKEN.format(path=socket_path)
                assert not local.robot_state_active
                with pytest.raises(RuntimeError, match=NOT_ACTIVE):
                    local.publish_robot_state("arm", [0.0, 0.0])
            # No backend of the session started.
            assert [backend.call_count for backend in backends] == [0, 0, 0]
            # The server and its reader are undisturbed.
            assert program.proc.poll() is None
            assert reader.server_reachable
            assert reader.joints("arm", allow_stale=True)["joint1"].position == 0.25

            # Without a description the local robot app enters.
            with _local_robot_app(mode) as session:
                assert not session.robot_state_active
            assert [backend.call_count for backend in backends] == [
                mode == "live",
                mode == "live",
                mode == "replay",
            ]

        # The server stopped: the place is free for the local robot app.
        assert program.stop() == 0, program.log()
        with _local_backends(), local:
            assert local.robot_state_active
            assert _robot_at() == "other_arm"
    finally:
        program.stop()
