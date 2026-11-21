# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Teleop Session Server's reader socket: one robot per path, and a socket that cannot be
opened is tried again.

The server runs in process (``placeholder=True``, no OpenXR runtime) on a free loopback port;
its reader socket is in a temporary folder. The other programs that serve the path are
processes the test starts and ends, or services of this process.
"""

import json
import logging
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import isaaccapture
import isaaccapture.robot_state
import isaaccapture.teleop_session_server as server_package
from isaaccapture.deviceio_session import RemoteSession, RemoteSessionListener
from isaaccapture.robot_state import (
    RobotState,
    RobotStateLocalService,
    RobotStateStore,
    parse_robot_description,
    records,
)
from isaaccapture.robot_state._local import ReaderSocketTaken
from isaaccapture.teleop_session_server import __main__ as server_main
from isaaccapture.teleop_session_server._session import HelloError

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the Teleop Session Server runs on Linux only"
)

WAIT_MS = 10_000
SERVER_LOGGER = "isaaccapture.teleop_session_server"
NO_READERS = (
    "Reader programs cannot get robot state from this Teleop Session Server: {reason}"
)
# The ERROR for a robot app that offers robot state while another program serves the path.
TAKEN = (
    "Another program on the Teleop Session Server's machine already serves a robot's state "
    "at {path}. One robot per machine is served; stop that program, or connect this robot "
    "app without a robot description."
)
# Far above what one try of a taken path costs the server's loop, far under a wait for it.
TRY_LIMIT_S = 0.5
DESCRIPTION_TOML = """
robot = "test_arm"

[series.arm]
shape = "joints"
joints = [
  { name = "joint1",  kind = "revolute", min = -2.0, max = 2.0 },
  { name = "gripper", kind = "gripper",  min = 0.0,  max = 1.0 },
]
"""
DESCRIPTION = parse_robot_description(DESCRIPTION_TOML)
OTHER_DESCRIPTION = parse_robot_description(
    DESCRIPTION_TOML.replace("test_arm", "other_arm")
)


def _hello(**fields):
    hello = {
        "protocol": 1,
        "isaaccapture_version": isaaccapture.__version__,
        "app_name": "RetryTest",
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


def _fill(store, description):
    """What a robot app that agreed robot state leaves in a store."""
    store.begin_session()
    for series in records.declarations(description):
        store.declare_series(
            series.name,
            series.shape,
            series.schema_name,
            series.schema_encoding,
            series.schema_data,
            series.message_encoding,
            series.stale_after_ns,
            series.in_order,
            series.rate_hz,
        )
    now_ns = time.monotonic_ns()
    store.put_record(
        records.DESCRIPTION_TOPIC,
        records.encode_description(description, now_ns),
        now_ns,
        robot_minus_server_ns=0,
        clock_two_way=True,
    )
    return store


def _robot_at(path):
    """The robot a new reader at `path` reads; None when nothing answers."""
    with RobotState.connect(path) as reader:
        description = reader.description()
        return None if description is None else description.robot


class _Launcher:
    def health_check(self):
        pass


def _server(socket_path):
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


@pytest.fixture
def socket_path(tmp_path_factory):
    # Short: a Unix socket path holds about a hundred bytes.
    return str(tmp_path_factory.mktemp("rs") / "rs.sock")


@pytest.fixture
def server(socket_path):
    """A server that is not running: the test calls what its loop calls."""
    server, _stop, _port = _server(socket_path)
    yield server
    server._shut_down(server_main.SHUTDOWN_MESSAGE)
    assert server_package.hosted_robot_state() is None


@pytest.fixture
def others():
    """Services of another program's making, closed when the test ends."""
    services = []

    def start(path):
        service = RobotStateLocalService(
            _fill(RobotStateStore(), OTHER_DESCRIPTION), path
        )
        services.append(service)
        service.start()
        return service

    yield start
    for service in services:
        service.close()


_OTHER_PROGRAM = (
    "import sys; sys.path.insert(0, sys.argv[1]); "
    "import test_reader_socket_retry; test_reader_socket_retry.serve_other(sys.argv[2])"
)


def serve_other(path):
    """What an `OtherProgram` runs: serve OTHER_DESCRIPTION at `path` until its input ends."""
    service = RobotStateLocalService(_fill(RobotStateStore(), OTHER_DESCRIPTION), path)
    service.start()
    print("serving", flush=True)
    sys.stdin.read()
    service.close()


class OtherProgram:
    """A process of its own that serves robot state at `path`."""

    def __init__(self, path):
        self._process = subprocess.Popen(
            [sys.executable, "-c", _OTHER_PROGRAM, str(Path(__file__).parent), path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        line = self._process.stdout.readline()
        if line.strip() != "serving":
            self.kill()
            pytest.fail(f"the other program did not serve {path}: {line!r}")

    @property
    def running(self):
        return self._process.poll() is None

    def leave(self):
        """End it in order: it closes its service."""
        self._process.stdin.close()
        assert self._process.wait(30.0) == 0
        self._process.stdout.close()

    def kill(self):
        """End it at once (SIGKILL): its socket file and its lock file stay."""
        self._process.kill()
        self._process.wait(30.0)
        self._process.stdin.close()
        self._process.stdout.close()

    def end(self, how):
        if how == "killed":
            self.kill()
        else:
            self.leave()


@pytest.fixture
def other_program():
    """Start other programs that serve a path; they are killed when the test ends."""
    programs = []

    def start(path):
        program = OtherProgram(path)
        programs.append(program)
        return program

    yield start
    for program in programs:
        if program.running:
            program.kill()


@pytest.fixture
def starts(monkeypatch):
    """Every start() of a reader socket the server tries, as the service it was tried on."""
    tried = []

    class Counted(RobotStateLocalService):
        def start(self):
            tried.append(self)
            super().start()

    monkeypatch.setattr(isaaccapture.robot_state, "RobotStateLocalService", Counted)
    return tried


@pytest.fixture
def warnings(caplog):
    caplog.set_level(logging.INFO)

    def texts():
        return [
            entry.getMessage()
            for entry in caplog.records
            if entry.levelno == logging.WARNING and entry.name == SERVER_LOGGER
        ]

    return texts


@pytest.fixture
def refusals(caplog):
    """The warnings of the robot apps' connections: whom the server refused, and why."""
    caplog.set_level(logging.INFO)

    def texts():
        return [
            entry.getMessage()
            for entry in caplog.records
            if entry.levelno == logging.WARNING
            and entry.name == SERVER_LOGGER + ".session"
        ]

    return texts


def _a_period_passes(server):
    """The health check as the loop makes it once HEALTH_CHECK_PERIOD_S passed."""
    server._next_health_check = -math.inf
    server._check_health()


# ============================================================================
# What the loop calls: a HELLO that offers robot state, and the health check
# ============================================================================


def test_nothing_is_tried_before_a_robot_app_agreed_robot_state(
    server, socket_path, starts, warnings
):
    for _ in range(3):
        _a_period_passes(server)
    assert starts == []
    assert server._robot_state_service is None
    assert os.listdir(Path(socket_path).parent) == []
    assert warnings() == []


def test_a_reader_socket_that_cannot_be_opened_is_tried_once_per_period(
    server, socket_path, starts, warnings
):
    # Not another program: a file that is no socket lies at the path.
    Path(socket_path).write_text("notes")
    reason = f"{socket_path} exists and is not a socket"

    # A robot app's HELLO offers robot state: it is not refused for this.
    server._serve_robot_state()
    assert len(starts) == 1
    assert server._robot_state_service is None
    assert server._robot_state_wanted is True
    assert warnings() == [NO_READERS.format(reason=reason)]

    # The loop passes the health check many times in one period.
    server._next_health_check = -math.inf
    for _ in range(5):
        server._check_health()
    assert len(starts) == 2
    for count in (3, 4, 5):
        _a_period_passes(server)
        assert len(starts) == count
    # The same reason is said once; the file is untouched.
    assert warnings() == [NO_READERS.format(reason=reason)]
    assert Path(socket_path).read_text() == "notes"

    # The file is gone: the next period's try opens the socket.
    Path(socket_path).unlink()
    _fill(server._robot_state, DESCRIPTION)
    _a_period_passes(server)
    assert len(starts) == 6
    assert server._robot_state_service is starts[-1]
    assert server._robot_state_service.serving is True
    assert _robot_at(socket_path) == "test_arm"

    # A socket that is served is not tried again.
    for _ in range(3):
        _a_period_passes(server)
    assert len(starts) == 6
    assert len(warnings()) == 1


@pytest.mark.parametrize("other_is", ["a process", "a service of this process"])
def test_a_hello_with_robot_state_is_refused_while_another_program_serves_the_path(
    server, socket_path, other_program, others, starts, warnings, other_is
):
    start_other = other_program if other_is == "a process" else others
    other = start_other(socket_path)

    for tried in (1, 2):
        start = time.monotonic()
        with pytest.raises(HelloError) as raised:
            server._serve_robot_state()
        # The try does not hold up the server's loop.
        assert time.monotonic() - start < TRY_LIMIT_S
        assert str(raised.value) == TAKEN.format(path=socket_path)
        assert type(raised.value.__cause__) is ReaderSocketTaken
        assert len(starts) == tried
        assert server._robot_state_service is None

        # No robot app agreed robot state: the health check tries nothing.
        for _ in range(3):
            _a_period_passes(server)
        assert len(starts) == tried
        assert server._robot_state_wanted is False
    # The connection logs the refusal; the other program's readers are untouched.
    assert warnings() == []
    assert _robot_at(socket_path) == "other_arm"

    # The other program left: the next HELLO opens the socket.
    if other_is == "a process":
        other.leave()
    else:
        other.close()
    _fill(server._robot_state, DESCRIPTION)
    server._serve_robot_state()
    assert len(starts) == 3
    assert server._robot_state_service.serving is True
    assert server._robot_state_wanted is True
    assert _robot_at(socket_path) == "test_arm"
    assert warnings() == []


@pytest.mark.skipif(
    hasattr(os, "getuid") and os.getuid() == 0,
    reason="root connects to a socket file whatever its mode",
)
def test_a_hello_is_not_refused_for_a_socket_file_closed_to_the_server_s_user(
    server, socket_path, other_program, starts, warnings
):
    """A killed program's socket file the server may not connect to: nobody is known to serve it."""
    other_program(socket_path).kill()
    os.chmod(socket_path, 0)
    reason = f"{socket_path} cannot be connected to ([Errno 13] Permission denied)"

    for _ in range(2):
        server._serve_robot_state()
    assert len(starts) == 2
    assert server._robot_state_service is None
    assert server._robot_state_wanted is True
    assert warnings() == [NO_READERS.format(reason=reason)]

    # It can be connected to again and nobody answers: the next period's try replaces it.
    os.chmod(socket_path, 0o600)
    _fill(server._robot_state, DESCRIPTION)
    _a_period_passes(server)
    assert len(starts) == 3
    assert server._robot_state_service.serving is True
    assert _robot_at(socket_path) == "test_arm"
    assert len(warnings()) == 1


def test_the_warning_is_repeated_for_another_reason_and_after_a_success(
    server, socket_path, others, starts, warnings
):
    taken = NO_READERS.format(
        reason=f"another program serves robot state at {socket_path}"
    )
    no_socket = NO_READERS.format(reason=f"{socket_path} exists and is not a socket")

    Path(socket_path).write_text("notes")
    server._serve_robot_state()
    _a_period_passes(server)
    assert warnings() == [no_socket]

    # Another program takes the path meanwhile: met by a later try, that is a warning.
    Path(socket_path).unlink()
    other = others(socket_path)
    for _ in range(3):
        _a_period_passes(server)
    assert warnings() == [no_socket, taken]
    assert server._robot_state_wanted is True
    assert other.serving is True

    other.close()
    _a_period_passes(server)
    served = server._robot_state_service
    assert served.serving is True
    assert warnings() == [no_socket, taken]

    # The server's service ends and another program takes the path: said again.
    served.close()
    others(socket_path)
    for _ in range(3):
        _a_period_passes(server)
    assert server._robot_state_service is None
    assert warnings() == [no_socket, taken, taken]

    # A robot app that connects in this state is the second robot at the path.
    tried = len(starts)
    with pytest.raises(HelloError) as raised:
        server._serve_robot_state()
    assert str(raised.value) == TAKEN.format(path=socket_path)
    assert len(starts) == tried + 1
    assert warnings() == [no_socket, taken, taken]
    # The server goes on trying for the place it had.
    _a_period_passes(server)
    assert len(starts) == tried + 2


class _Exploding:
    """A store that cannot be read."""

    def export_snapshot(self, known_setup_generation):
        raise RuntimeError("the store exploded")


def test_a_reader_socket_whose_thread_ended_is_opened_again(
    server, socket_path, starts, warnings
):
    _fill(server._robot_state, DESCRIPTION)
    server._serve_robot_state()
    first = server._robot_state_service
    assert first.serving is True
    assert _robot_at(socket_path) == "test_arm"
    _a_period_passes(server)
    assert server._robot_state_service is first

    # The next request ends the service's thread.
    first._store = _Exploding()
    assert _robot_at(socket_path) is None
    assert _wait_for(lambda: not first.serving)
    assert _wait_for(lambda: not os.path.exists(socket_path))

    _a_period_passes(server)
    second = server._robot_state_service
    assert second is not first
    assert second.serving is True
    assert _robot_at(socket_path) == "test_arm"
    assert starts == [first, second]
    assert warnings() == []
    # The dead one was closed.
    with pytest.raises(RuntimeError, match="RobotStateLocalService is closed"):
        first.start()


def test_a_reader_socket_whose_thread_cannot_start_does_not_stop_the_server(
    server, socket_path, monkeypatch, warnings
):
    class NoThread(RobotStateLocalService):
        def start(self):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(isaaccapture.robot_state, "RobotStateLocalService", NoThread)
    server._serve_robot_state()
    for _ in range(3):
        _a_period_passes(server)
    assert server._robot_state_service is None
    assert warnings() == [NO_READERS.format(reason="can't start new thread")]


# ============================================================================
# The server running, with a robot app
# ============================================================================


def _send_robot_state(session, port):
    """Connect as a robot app that sends DESCRIPTION and one reading of its arm."""
    session.set_robot_state(records.declarations(DESCRIPTION))
    hello = _hello(topics=session.subscribed_topics(), robot_state={"version": 1})
    session.connect(f"127.0.0.1:{port}", hello, WAIT_MS)
    assert json.loads(session.hello_ack())["robot_state"] == {"version": 1}
    session.start_robot_state()
    now_ns = time.monotonic_ns()
    session.publish_robot_record(
        records.DESCRIPTION_TOPIC,
        records.encode_description(DESCRIPTION, now_ns),
        now_ns,
    )
    _publish_arm(session, 0.5)


def _publish_arm(session, joint1):
    now_ns = time.monotonic_ns()
    session.publish_robot_record(
        "arm",
        records.encode_reading(DESCRIPTION, "arm", [joint1, 1.0], time_ns=now_ns),
        now_ns,
    )


def _refused(session, port):
    """Connect as a robot app that offers robot state; the ERROR the server answers."""
    session.set_robot_state(records.declarations(DESCRIPTION))
    hello = _hello(topics=session.subscribed_topics(), robot_state={"version": 1})
    with pytest.raises(RuntimeError) as raised:
        session.connect(f"127.0.0.1:{port}", hello, WAIT_MS)
    # Answered, not closed: a robot app tries a closed connection again.
    assert type(raised.value) is RuntimeError
    return str(raised.value)


@pytest.mark.parametrize("ending", ["left", "killed"])
def test_a_robot_app_with_robot_state_is_refused_while_another_process_serves_the_path(
    socket_path, other_program, starts, warnings, refusals, ending
):
    other = other_program(socket_path)
    text = TAKEN.format(path=socket_path)
    with _Running(socket_path) as running:
        store = running.server._robot_state
        for count in (1, 2):
            with RemoteSession.create([]) as session:
                assert _refused(session, running.port) == text
            assert _wait_for(lambda count=count: len(refusals()) == count)
            # Tried once for its HELLO, and not by the health check.
            time.sleep(1.5 * server_main.HEALTH_CHECK_PERIOD_S)
            assert len(starts) == count
        # Nothing of the refused robot app reached the store.
        assert (store.connected, store.session_count, store.series()) == (False, 0, [])
        assert server_package.hosted_robot_state().description() is None
        for refusal in refusals():
            assert refusal.startswith("Refused the robot app at 127.0.0.1:")
            assert refusal.endswith(": " + text)
        assert warnings() == []

        # The same robot app without robot state is served, whoever holds the path.
        with RemoteSession.create([]) as session:
            hello = _hello(topics=session.subscribed_topics())
            session.connect(f"127.0.0.1:{running.port}", hello, WAIT_MS)
            assert "robot_state" not in json.loads(session.hello_ack())
            assert session.wait_ready(WAIT_MS)
            time.sleep(1.5 * server_main.HEALTH_CHECK_PERIOD_S)
        assert len(starts) == 2
        assert running.server._robot_state_wanted is False
        assert (store.connected, store.session_count) == (False, 0)
        # The other program and its readers are undisturbed.
        assert other.running
        assert _robot_at(socket_path) == "other_arm"

        other.end(ending)
        assert os.path.exists(socket_path) == (ending == "killed")
        with RemoteSession.create([]) as session:
            _send_robot_state(session, running.port)
            with RobotState.connect(socket_path, robot="test_arm") as reader:
                assert _wait_for(lambda: reader.connected)
                assert reader.description() == DESCRIPTION
                assert _wait_for(lambda: reader.joints("arm"))["joint1"].position == 0.5
                _publish_arm(session, 0.25)
                assert _wait_for(
                    lambda: (
                        (arm := reader.joints("arm")) is not None
                        and arm["joint1"].position == 0.25
                    )
                )
            assert len(starts) == 3
            assert running.server._robot_state_service is starts[-1]
            assert running.server._robot_state_wanted is True
    assert running.code == 0
    assert len(refusals()) == 2
    assert warnings() == []
    assert not os.path.exists(socket_path)


def test_a_robot_app_that_is_served_stays_served_when_its_server_loses_the_path(
    socket_path, others, starts, warnings, refusals
):
    """Only a robot app that starts as the second one is refused."""
    reason = f"another program serves robot state at {socket_path}"
    with _Running(socket_path) as running:
        with RemoteSession.create([]) as session:
            _send_robot_state(session, running.port)
            hosted = server_package.hosted_robot_state()
            assert _wait_for(lambda: hosted.joints("arm"))
            assert _robot_at(socket_path) == "test_arm"

            # The server's service ends and another program takes the path.
            running.server._robot_state_service.close()
            other = others(socket_path)
            assert _wait_for(warnings) == [NO_READERS.format(reason=reason)]
            # Tried again every period, said once; the robot app's state still arrives.
            tried = len(starts)
            time.sleep(2.5 * server_main.HEALTH_CHECK_PERIOD_S)
            assert len(starts) >= tried + 2
            assert warnings() == [NO_READERS.format(reason=reason)]
            _publish_arm(session, 0.25)
            assert _wait_for(
                lambda: (
                    (arm := hosted.joints("arm")) is not None
                    and arm["joint1"].position == 0.25
                )
            )
            assert session.wait_ready(WAIT_MS)
            assert _robot_at(socket_path) == "other_arm"

            # The other program left: the server takes its path again within a period.
            other.close()
            with RobotState.connect(socket_path, robot="test_arm") as reader:
                assert _wait_for(lambda: reader.connected, 3.0)
                arm = reader.joints("arm", allow_stale=True)
                assert arm["joint1"].position == 0.25
    assert running.code == 0
    assert refusals() == []
    assert warnings() == [NO_READERS.format(reason=reason)]
    assert not os.path.exists(socket_path)


def test_a_robot_app_without_robot_state_makes_no_socket_and_no_lock_file(
    socket_path, starts, warnings
):
    with _Running(socket_path) as running:
        with RemoteSession.create([]) as session:
            hello = _hello(topics=session.subscribed_topics())
            session.connect(f"127.0.0.1:{running.port}", hello, WAIT_MS)
            assert "robot_state" not in json.loads(session.hello_ack())
            assert session.wait_ready(WAIT_MS)
            # Several health checks pass.
            time.sleep(2.5 * server_main.HEALTH_CHECK_PERIOD_S)
            assert not server_package.hosted_robot_state().connected
            assert running.server._robot_state_wanted is False
    assert running.code == 0
    assert starts == []
    assert os.listdir(Path(socket_path).parent) == []
    assert warnings() == []
