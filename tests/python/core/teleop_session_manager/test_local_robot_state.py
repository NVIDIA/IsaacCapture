# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Robot state sent by TeleopSession in SessionMode.LIVE and SessionMode.REPLAY.

The robot app keeps the robot's state and serves it to reader programs itself. OpenXR, DeviceIO
and replay are replaced by mocks; the store, the reader socket and the reader are real. The socket
is the default one, under a temporary XDG_RUNTIME_DIR; the other robot apps are processes the
test starts and ends.
"""

import json
import logging
import math
import os
import select
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from test_remote_mode import mock_remote_dependencies, remote_config
from test_remote_no_headset import (
    READER_THREAD,
    local_backends,
    local_config,
    reader_threads,
    robot_state_files,
)
from test_remote_robot_state import (
    AGREED,
    DESCRIPTION,
    DESCRIPTION_TOML,
    SERIES_NAMES,
    _ack,
    _loopback_config,
    _MiniServer,
    _Session,
    _wait_for,
)

from isaaccapture.robot_state import (
    RobotDescriptionError,
    RobotState,
    RobotStateLocalService,
    RobotStateStore,
    _host,
    default_socket_path,
    parse_robot_description,
    records,
)
from isaaccapture.teleop_session_manager import (
    PluginConfig,
    SessionMode,
    TeleopSession,
    teleop_session,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="robot state is served over a Unix socket"
)

MODES = ["live", "replay"]
NOT_ACTIVE = r"^TeleopSession is not active$"
LOCK_FILE = "robot_state.sock.lock"
MIB = 1024 * 1024
ROUNDS = 20
RETRY_THREAD = "isaaccapture-robot-state-socket-retry"
# What entering raises while another program serves robot state at the reader socket's path.
TAKEN = (
    "Another program on this machine already serves a robot's state at {path}: a Teleop "
    "Session Server or another robot app. One robot per machine is served; stop that "
    "program, then start this robot app again."
)
# The description of the user page's calls: velocities, a number reading, a shape of its own.
DRIVE_TOML = """
robot = "drive_arm"

[shapes.force]
fields = ["fx", "fy", { name = "contact", kind = "flag" }]

[series.arm]
shape = "joints"
has_velocity = true
joints = [
  { name = "joint1",  kind = "revolute", min = -2.0, max = 2.0 },
  { name = "gripper", kind = "gripper",  min = 0.0,  max = 1.0 },
]

[series.waist]
shape = "joints"
joints = [{ name = "waist_yaw", kind = "revolute", min = -1.5, max = 1.5 }]

[series.status]
shape = "status"
readings = ["motors_on", "fault", { name = "battery", kind = "number", unit = "V" }]

[series.wrist_force]
shape = "force"
"""
DRIVE = parse_robot_description(DRIVE_TOML)


@pytest.fixture(autouse=True)
def runtime_dir(tmp_path_factory, monkeypatch):
    """The session serves its reader socket under this folder, not the user's own."""
    path = tmp_path_factory.mktemp("rt")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(path))
    assert default_socket_path().startswith(str(path))
    return path


@pytest.fixture
def description_file(tmp_path):
    path = tmp_path / "test_arm.toml"
    path.write_text(DESCRIPTION_TOML)
    return path


def _config(mode, **kwargs):
    if mode == "replay":
        mcap_config = MagicMock()
        mcap_config.filename = "session.mcap"
        kwargs.update(mode=SessionMode.REPLAY, mcap_config=mcap_config)
    return local_config(**kwargs)


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


# ============================================================================
# Entry, publishing, exit
# ============================================================================


@pytest.mark.parametrize("mode", MODES)
class TestLocalSession:
    def test_entry_serves_the_robots_state(self, mode, description_file, runtime_dir):
        session = TeleopSession(_config(mode, robot_description=description_file))
        assert not session.robot_state_active
        with local_backends() as mocks, session:
            assert session.robot_state_active
            assert robot_state_files(runtime_dir) == ["robot_state.sock", LOCK_FILE]
            assert len(reader_threads()) == 1
            read_ns = time.monotonic_ns()
            assert session.publish_robot_state(
                "arm", [0.25, -0.5, 1.0], time_ns=read_ns
            )
            assert session.publish_robot_record(
                "wrist_force", {"fx": 1.0, "fy": 2.0, "contact": True}
            )
            with RobotState.connect() as robot:
                assert robot.server_reachable
                assert robot.connected
                assert robot.description() == DESCRIPTION
                arm = robot.joints("arm")
                assert arm.positions == (0.25, -0.5, 1.0)
                assert (arm.sample_time_ns, arm.fresh, arm.clock_two_way) == (
                    read_ns,
                    True,
                    True,
                )
                assert 0.0 <= arm.age_s < 5.0
                assert robot.record("wrist_force").values == {
                    "fx": 1.0,
                    "fy": 2.0,
                    "contact": True,
                }
                assert robot.status() is None

                session.step()
                assert session.publish_robot_state("status", [True, "none"])
                assert robot.status()["motors_on"] is True
            assert mocks.oxr_cls.call_count == (mode == "live")
            assert mocks.replay_run.call_count == (mode == "replay")

        assert not session.robot_state_active
        assert robot_state_files(runtime_dir) == [LOCK_FILE]
        assert reader_threads() == []
        with pytest.raises(RuntimeError, match=NOT_ACTIVE):
            session.publish_robot_state("arm", [0.0, 0.0, 0.0])
        with pytest.raises(RuntimeError, match=NOT_ACTIVE):
            session.publish_robot_record("wrist_force", [0.0, 0.0, False])

    def test_a_loaded_description_works_as_a_file_does(self, mode):
        config = _config(mode, robot_description=DESCRIPTION)
        with local_backends(), TeleopSession(config) as session:
            assert session.robot_state_active
            with RobotState.connect(robot="test_arm") as robot:
                assert robot.connected
                assert list(robot.series()) == SERIES_NAMES[1:]

    def test_a_reader_kept_across_two_entries_reads_the_second(self, mode):
        session = TeleopSession(_config(mode, robot_description=DESCRIPTION))
        with local_backends(), RobotState.connect() as robot:
            assert not robot.server_reachable
            with session:
                assert session.publish_robot_state("arm", [0.25, 0.0, 0.0])
                assert robot.joints("arm").positions == (0.25, 0.0, 0.0)
            assert not robot.server_reachable
            assert not robot.connected
            assert robot.description() is None

            with session:
                assert robot.connected
                assert robot.description() == DESCRIPTION
                # Nothing of the first entry is read as the second's.
                assert robot.joints("arm") is None
                assert robot.joints("arm", allow_stale=True) is None
                assert session.publish_robot_state("arm", [0.5, 0.0, 1.0])
                assert robot.joints("arm").positions == (0.5, 0.0, 1.0)
            assert not robot.server_reachable

    def test_publishing_from_another_thread(self, mode):
        results = []
        config = _config(mode, robot_description=DESCRIPTION)
        with local_backends(), TeleopSession(config) as session:
            thread = threading.Thread(
                target=lambda: results.append(
                    session.publish_robot_state("arm", [0.0, 0.0, 0.0])
                )
            )
            thread.start()
            thread.join(timeout=10)
        assert results == [True]

    def test_hosting_starts_before_the_backend(self, mode):
        """A reader sees the robot while the session waits for its backend."""
        session = TeleopSession(_config(mode, robot_description=DESCRIPTION))
        seen = {}

        def backend(*args, **kwargs):
            with RobotState.connect() as robot:
                seen["connected"] = robot.connected
                seen["description"] = robot.description()
            seen["active"] = session.robot_state_active
            with pytest.raises(RuntimeError, match=NOT_ACTIVE):
                session.publish_robot_state("arm", [0.0, 0.0, 0.0])
            seen["calls"] = seen.get("calls", 0) + 1
            return made

        with local_backends() as mocks:
            if mode == "live":
                made = mocks.openxr
                mocks.oxr_cls.side_effect = backend
            else:
                made = MagicMock()
                mocks.replay_run.side_effect = backend
            with session:
                assert session.robot_state_active
        assert seen == {
            "connected": True,
            "description": DESCRIPTION,
            "active": False,
            "calls": 1,
        }

    def test_a_bad_description_fails_before_the_backend(
        self, mode, tmp_path, runtime_dir
    ):
        broken = tmp_path / "broken.toml"
        broken.write_text('robot = "r"\n[series.arm]\nshape = "wheels"\n')
        for path, error in (
            (tmp_path / "missing.toml", FileNotFoundError),
            (broken, RobotDescriptionError),
        ):
            session = TeleopSession(_config(mode, robot_description=path))
            with local_backends() as mocks:
                with pytest.raises(error):
                    session.__enter__()
                mocks.oxr_cls.assert_not_called()
                mocks.live_run.assert_not_called()
                mocks.replay_run.assert_not_called()
            assert not session._in_context
            assert not session.robot_state_active
            assert robot_state_files(runtime_dir) == []
            assert reader_threads() == []
            with pytest.raises(RuntimeError, match=NOT_ACTIVE):
                session.publish_robot_state("arm", [0.0, 0.0, 0.0])

    def test_a_failed_backend_rolls_the_host_back(self, mode, runtime_dir):
        session = TeleopSession(_config(mode, robot_description=DESCRIPTION))
        with local_backends() as mocks:
            run = mocks.live_run if mode == "live" else mocks.replay_run
            run.side_effect = RuntimeError("run failed")
            with pytest.raises(RuntimeError, match=r"^run failed$"):
                session.__enter__()
            assert not session._in_context
            assert not session.robot_state_active
            assert robot_state_files(runtime_dir) == [LOCK_FILE]
            assert reader_threads() == []
            assert mocks.openxr.exited == (mode == "live")
            with RobotState.connect() as robot:
                assert not robot.server_reachable
            with pytest.raises(RuntimeError, match=NOT_ACTIVE):
                session.publish_robot_state("arm", [0.0, 0.0, 0.0])

            run.side_effect = None
            with session:
                assert session.robot_state_active
                assert session.publish_robot_state("arm", [0.0, 0.0, 0.0])


# ============================================================================
# One robot app, every mode
# ============================================================================


def drive(session):
    """Every publish call and every wrong call of the user page, with what each one gave."""
    calls = [
        lambda: session.publish_robot_state("arm", [0.25, 1.0], velocities=[0.5, 0.0]),
        lambda: session.publish_robot_state(
            "arm", [0.5, 0.0], velocities=[0.0, 0.0], valid=[True, False], time_ns=1234
        ),
        lambda: session.publish_robot_state("waist", [0.5]),
        lambda: session.publish_robot_state(
            "status",
            [True, "", 47.5],
            faults=[("overtemperature", "joint1 is hot", "joint1", "warning")],
        ),
        lambda: session.publish_robot_state("status", [None, "stopped", None]),
        lambda: session.publish_robot_state("wrist_force", [1.5, -2.0, True]),
        lambda: session.publish_robot_record(
            "wrist_force", {"fx": 0.5, "fy": 0.25, "contact": False}, time_ns=99
        ),
        # A series the description does not have.
        lambda: session.publish_robot_state("leg", [0.0]),
        # A wrong number of values.
        lambda: session.publish_robot_state("arm", [0.0], velocities=[0.0, 0.0]),
        lambda: session.publish_robot_state("arm", [0.0, 0.0], velocities=[0.0]),
        # Missing and undeclared velocities and efforts.
        lambda: session.publish_robot_state("arm", [0.0, 0.0]),
        lambda: session.publish_robot_state(
            "arm", [0.0, 0.0], velocities=[0.0, 0.0], efforts=[0.0, 0.0]
        ),
        lambda: session.publish_robot_state("waist", [0.0], velocities=[0.0]),
        # A number that is not finite.
        lambda: session.publish_robot_state("waist", [math.nan]),
        lambda: session.publish_robot_state("waist", [math.inf]),
        # A status value of another kind than declared, and a fault of no known form.
        lambda: session.publish_robot_state("status", [True, "", "full"]),
        lambda: session.publish_robot_state(
            "status", [True, "", 47.5], faults=[("code", "text", "source", "bad")]
        ),
        lambda: session.publish_robot_state("waist", [0.0], faults=[]),
        # A record with missing or unknown fields.
        lambda: session.publish_robot_record("wrist_force", {"fx": 1.0}),
        lambda: session.publish_robot_record(
            "wrist_force", {"fx": 0, "fy": 0, "contact": True, "fz": 0}
        ),
        # A record above 1 MiB.
        lambda: session.publish_robot_state("status", [True, "x" * (MIB + 1), 47.5]),
    ]
    outcomes = []
    for index, publish in enumerate(calls):
        try:
            outcomes.append(("returns", publish()))
        except Exception as error:
            # The refusal of a record above 1 MiB is worded by whoever keeps the records.
            text = None if index == len(calls) - 1 else str(error)
            outcomes.append(("raises", type(error).__name__, text))
    return outcomes


VALID_CALLS = 7
EXPECTED = [("returns", True)] * VALID_CALLS + [
    ("raises", "ValueError", text)
    for text in (
        "robot 'drive_arm' has no series 'leg' (it has: arm, waist, status, wrist_force)",
        "series 'arm' takes 2 values (joint1, gripper), got 1",
        "series 'arm': velocities takes 2 values, got 1",
        "series 'arm' declares has_velocity: velocities are required",
        "series 'arm' does not declare has_effort",
        "series 'waist' does not declare has_velocity",
        "series 'waist': 'waist_yaw' is not a finite number",
        "series 'waist': 'waist_yaw' is not a finite number",
        "series 'status': reading 'battery' is declared number, got str",
        "series 'status': fault severity 'bad' is not one of warning, error, fatal",
        "series 'waist' has shape 'joints': faults does not apply",
        "series 'wrist_force': record lacks fy, contact",
        "series 'wrist_force': record has unknown fields fz",
        None,
    )
]


def drive_outside(session):
    """The publish calls of a session that is not entered."""
    outcomes = []
    for publish in (session.publish_robot_state, session.publish_robot_record):
        with pytest.raises(RuntimeError) as raised:
            publish("waist", [0.0])
        outcomes.append(str(raised.value))
    return outcomes


class _SizedSession(_Session):
    """The fake RemoteSession, refusing a record above 1 MiB as the real one does."""

    def publish_robot_record(self, series, data, sample_time_ns):
        if len(data) > MIB:
            raise ValueError(f"RobotStatePublisher: a record of {len(data)} bytes")
        return super().publish_robot_record(series, data, sample_time_ns)


class TestOneRobotApp:
    """The test of the aim: one robot app, written once, runs in a local and a remote session."""

    @pytest.mark.parametrize("mode", [*MODES, "remote", "remote_loopback"])
    def test_the_same_calls_give_the_same_in_every_mode(self, mode):
        server = None
        if mode == "remote_loopback":
            server = _MiniServer(answer=AGREED, store=RobotStateStore())
            session = TeleopSession(_loopback_config(server, robot_description=DRIVE))
            backends = local_backends()
        elif mode == "remote":
            session = TeleopSession(remote_config(robot_description=DRIVE))
            backends = mock_remote_dependencies(_SizedSession(_ack(**AGREED)))
        else:
            session = TeleopSession(_config(mode, robot_description=DRIVE))
            backends = local_backends()
        try:
            assert drive_outside(session) == ["TeleopSession is not active"] * 2
            with backends, session:
                assert session.robot_state_active
                assert drive(session) == EXPECTED
                session.step()
                assert drive(session) == EXPECTED
                assert session.robot_state_active
            assert not session.robot_state_active
            assert drive_outside(session) == ["TeleopSession is not active"] * 2
        finally:
            if server is not None:
                server.stop()

    @pytest.mark.parametrize("mode", [*MODES, "remote"])
    def test_without_a_description_the_calls_say_what_is_missing(self, mode):
        missing = ["TeleopSessionConfig.robot_description is not set"] * 2
        if mode == "remote":
            session = TeleopSession(remote_config())
            backends = mock_remote_dependencies(_SizedSession(_ack(**AGREED)))
        else:
            session = TeleopSession(_config(mode))
            backends = local_backends()
        assert drive_outside(session) == missing
        with backends, session:
            assert not session.robot_state_active
            assert drive_outside(session) == missing
        assert drive_outside(session) == missing

    @pytest.mark.parametrize("mode", MODES)
    def test_with_a_socket_that_cannot_be_opened_the_calls_return_false(
        self, mode, caplog, monkeypatch
    ):
        monkeypatch.setattr(_host, "RETRY_S", 0.05)
        # Not another program: a file that is no socket lies at the reader socket's path.
        in_the_way = Path(default_socket_path())
        in_the_way.parent.mkdir(mode=0o700)
        in_the_way.write_text("notes")
        session = TeleopSession(_config(mode, robot_description=DRIVE))
        with (
            local_backends() as mocks,
            caplog.at_level(logging.WARNING, logger="isaaccapture"),
            session,
        ):
            # The robot app runs; only its robot state reaches nobody.
            assert mocks.oxr_cls.call_count == (mode == "live")
            assert mocks.replay_run.call_count == (mode == "replay")
            assert not session.robot_state_active
            # A valid call returns False and a record above 1 MiB is not looked at; a call
            # that does not fit the description is refused all the same.
            refused = [("returns", False)] * VALID_CALLS + EXPECTED[VALID_CALLS:-1]
            assert drive(session) == [*refused, ("returns", False)]
            session.step()
            assert not session.robot_state_active
            with RobotState.connect() as robot:
                assert not robot.server_reachable
            assert _warnings(caplog) == [
                "Reader programs cannot get robot state from this robot app: "
                f"{in_the_way} exists and is not a socket. Publish calls return False "
                "until the socket can be opened; trying again every 0.05 s"
            ]

            # The file is gone: the next publish calls have the socket opened.
            in_the_way.unlink()
            assert _wait_for(
                lambda: session.publish_robot_state("waist", [0.5]), timeout_s=10.0
            )
            assert session.robot_state_active
            with RobotState.connect(robot="drive_arm") as robot:
                assert robot.joints("waist").positions == (0.5,)
            assert drive(session) == EXPECTED
        assert len(_warnings(caplog)) == 1


# ============================================================================
# One robot per machine: a second robot app is refused at entry
# ============================================================================

# A robot app in a program of its own: on command it enters and leaves LIVE and REPLAY sessions
# with a robot description, one command per line of its input. OpenXR, DeviceIO and replay are
# mocks; the store and the reader socket are real.
_ROBOT_APP = """
import json, sys, threading, time
from unittest.mock import MagicMock, patch

from isaaccapture.robot_state import parse_robot_description
from isaaccapture.teleop_session_manager import SessionMode, TeleopSession, TeleopSessionConfig

ROBOT = '''
robot = "{name}"

[series.waist]
shape = "joints"
joints = [{{ name = "waist_yaw", kind = "revolute", min = -1.5, max = 1.5 }}]
'''


class Pipeline:
    def get_leaf_nodes(self):
        return []

    def execute_pipeline(self, inputs, context=None):
        return {}


def config(mode, robot):
    kwargs = {"robot_description": parse_robot_description(ROBOT.format(name=robot))}
    if mode == "replay":
        kwargs.update(mode=SessionMode.REPLAY, mcap_config=MagicMock())
    return TeleopSessionConfig(app_name="a", pipeline=Pipeline(), verbose=False, **kwargs)


session = None
with (
    patch("isaaccapture.oxr.OpenXRSession") as openxr,
    patch("isaaccapture.deviceio.DeviceIOSession.run") as live_run,
    patch("isaaccapture.deviceio.DeviceIOSession.get_required_extensions", return_value=[]),
    patch("isaaccapture.deviceio.ReplaySession.run") as replay_run,
    patch("isaaccapture.deviceio.McapReplayConfig"),
):
    backends = (openxr, live_run, replay_run)
    for line in sys.stdin:
        command = json.loads(line)
        do = command["do"]
        answer = "done"
        if do == "enter":
            before = sum(backend.call_count for backend in backends)
            session = TeleopSession(config(command["mode"], command["robot"]))
            # Two programs told the same moment of this machine's clock enter together.
            while time.monotonic_ns() < command.get("at_ns", 0):
                pass
            try:
                session.__enter__()
            except RuntimeError as error:
                answer = {"raised": str(error)}
            else:
                answer = {"entered": session.robot_state_active}
            answer["backends"] = sum(backend.call_count for backend in backends) - before
            answer["threads"] = sorted(
                thread.name for thread in threading.enumerate() if "robot-state" in thread.name
            )
        elif do == "publish":
            answer = session.publish_robot_state("waist", [command["position"]])
        elif do == "leave":
            session.__exit__(None, None, None)
        print(json.dumps({"do": do, "answer": answer}), flush=True)
"""


class RobotAppProgram:
    """A process that runs `_ROBOT_APP`."""

    def __init__(self):
        self._process = subprocess.Popen(
            [sys.executable, "-c", _ROBOT_APP],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )

    def send(self, do, **command):
        self._process.stdin.write(json.dumps({"do": do, **command}) + "\n")
        self._process.stdin.flush()

    def answer(self, do):
        while True:
            ready, _, _ = select.select([self._process.stdout], [], [], 120.0)
            assert ready, f"the robot app did not answer '{do}'"
            line = self._process.stdout.readline()
            assert line, f"the robot app ended (exit code {self._process.poll()})"
            if line.startswith("{"):
                reply = json.loads(line)
                if reply["do"] == do:
                    return reply["answer"]

    def do(self, do, **command):
        self.send(do, **command)
        return self.answer(do)

    def kill(self):
        """End it at once (SIGKILL): it closes nothing."""
        self._process.kill()
        self._process.wait(30.0)
        self._process.stdin.close()
        self._process.stdout.close()


@pytest.fixture
def robot_app():
    """Start robot apps in programs of their own; they are killed when the test ends."""
    programs = []

    def start():
        program = RobotAppProgram()
        programs.append(program)
        return program

    yield start
    for program in programs:
        if program._process.poll() is None:
            program.kill()


@pytest.fixture
def plugins(tmp_path):
    """A configured plugin and the manager that would start it."""
    manager = MagicMock()
    manager.get_plugin_names.return_value = ["test_plugin"]
    config = PluginConfig(
        plugin_name="test_plugin", plugin_root_id="root", search_paths=[tmp_path]
    )
    with patch(
        "isaaccapture.plugin_manager.PluginManager", return_value=manager
    ) as manager_cls:
        yield SimpleNamespace(config=config, manager=manager, manager_cls=manager_cls)


def robot_state_threads():
    return [
        thread
        for thread in threading.enumerate()
        if thread.name in (READER_THREAD, RETRY_THREAD)
    ]


def lock_descriptors():
    """This program's open descriptors of the lock file; None where they cannot be listed."""
    if not os.path.isdir("/proc/self/fd"):
        return None
    lock_path = os.path.realpath(default_socket_path() + ".lock")
    found = []
    for name in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{name}")
        except OSError:
            continue
        if target == lock_path:
            found.append(int(name))
    return found


def lock_is_free():
    """Whether no program holds the lock of the reader socket's path."""
    import fcntl

    fd = os.open(default_socket_path() + ".lock", os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    finally:
        os.close(fd)


def host_warnings(caplog):
    """What the robot app's side of robot state warned of; a reader's lines are not among them."""
    names = ("isaaccapture.robot_state._host", "isaaccapture.robot_state._local")
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING and r.name in names
    ]


def assert_nothing_started(session, mocks):
    """A refused entry made no OpenXR session, no DeviceIO or replay session and left no host."""
    mocks.oxr_cls.assert_not_called()
    mocks.live_run.assert_not_called()
    mocks.replay_run.assert_not_called()
    mocks.start_twin.assert_not_called()
    assert not session._in_context
    assert not session.robot_state_active
    assert session._robot_state_sender is None
    assert session._robot_description is None
    assert session.oxr_session is None
    with pytest.raises(RuntimeError, match=NOT_ACTIVE):
        session.publish_robot_state("arm", [0.0, 0.0, 0.0])


@pytest.mark.parametrize("mode", MODES)
class TestASecondRobotIsRefused:
    def test_entry_raises_while_another_program_serves_the_path(
        self, mode, robot_app, plugins, runtime_dir, caplog
    ):
        other = robot_app()
        entered = other.do("enter", mode="live", robot="other")
        assert entered["entered"] is True
        assert other.do("publish", position=0.25) is True
        session = TeleopSession(
            _config(mode, robot_description=DESCRIPTION, plugins=[plugins.config])
        )
        with (
            RobotState.connect(robot="other") as kept,
            local_backends() as mocks,
            caplog.at_level(logging.INFO, logger="isaaccapture"),
        ):
            assert kept.joints("waist").positions == (0.25,)

            with pytest.raises(RuntimeError) as raised:
                session.__enter__()
            assert str(raised.value) == TAKEN.format(path=default_socket_path())
            cause = raised.value.__cause__
            assert type(cause).__name__ == "ReaderSocketTaken"
            assert isinstance(cause, OSError)
            assert str(cause) == (
                f"another program serves robot state at {default_socket_path()}"
            )
            # Nothing of the session started, and nothing of its host is left.
            assert_nothing_started(session, mocks)
            plugins.manager_cls.assert_not_called()
            plugins.manager.start.assert_not_called()
            assert robot_state_threads() == []
            assert lock_descriptors() in (None, [])
            assert not lock_is_free()
            assert robot_state_files(runtime_dir) == ["robot_state.sock", LOCK_FILE]
            # The exception is the whole report.
            assert host_warnings(caplog) == []

            # The other program and its reader are undisturbed.
            assert other.do("publish", position=0.5) is True
            assert kept.connected
            waist = kept.joints("waist")
            assert (waist.positions, waist.sequence) == ((0.5,), 2)

            # A second attempt is refused the same way.
            with pytest.raises(RuntimeError) as raised:
                session.__enter__()
            assert str(raised.value) == TAKEN.format(path=default_socket_path())
            assert_nothing_started(session, mocks)

            # The other program left: the same session enters, with all that belongs to it.
            other.do("leave")
            assert not kept.server_reachable
            assert lock_is_free()
            with session:
                assert session.robot_state_active
                assert mocks.oxr_cls.call_count == (mode == "live")
                assert mocks.live_run.call_count == (mode == "live")
                assert mocks.replay_run.call_count == (mode == "replay")
                assert plugins.manager.start.call_count == 1
                assert session.publish_robot_state("arm", [0.25, 0.0, 0.0])
                with RobotState.connect(robot="test_arm") as robot:
                    assert robot.joints("arm").positions == (0.25, 0.0, 0.0)
                # The reader that expects the other robot reads nothing of this one.
                assert kept.server_reachable
                assert not kept.connected
            assert robot_state_threads() == []
            assert host_warnings(caplog) == []

    def test_entry_raises_while_this_program_serves_the_path(self, mode, runtime_dir):
        other = RobotStateLocalService(RobotStateStore(), default_socket_path())
        other.start()
        try:
            held = lock_descriptors()
            session = TeleopSession(_config(mode, robot_description=DESCRIPTION))
            with local_backends() as mocks:
                with pytest.raises(RuntimeError) as raised:
                    session.__enter__()
                assert str(raised.value) == TAKEN.format(path=default_socket_path())
                assert_nothing_started(session, mocks)
                assert other.serving
                assert len(reader_threads()) == 1
                assert lock_descriptors() == held

                other.close()
                with session:
                    assert session.robot_state_active
                    assert session.publish_robot_state("arm", [0.0, 0.0, 0.0])
            assert robot_state_files(runtime_dir) == [LOCK_FILE]
        finally:
            other.close()

    def test_a_killed_robot_app_refuses_nobody(self, mode, robot_app, runtime_dir):
        other = robot_app()
        assert other.do("enter", mode=mode, robot="killed")["entered"] is True
        with RobotState.connect() as reader:
            assert reader.description().robot == "killed"
        other.kill()
        # Its socket file and its lock file are still there; nobody holds either.
        assert robot_state_files(runtime_dir) == ["robot_state.sock", LOCK_FILE]
        assert lock_is_free()
        with RobotState.connect() as reader:
            assert not reader.server_reachable

        config = _config(mode, robot_description=DESCRIPTION)
        with local_backends(), TeleopSession(config) as session:
            assert session.robot_state_active
            assert session.publish_robot_state("arm", [0.5, 0.0, 1.0])
            with RobotState.connect(robot="test_arm") as robot:
                assert robot.joints("arm").positions == (0.5, 0.0, 1.0)
        assert robot_state_files(runtime_dir) == [LOCK_FILE]

    @pytest.mark.skipif(
        hasattr(os, "getuid") and os.getuid() == 0,
        reason="root connects to a socket file whatever its mode",
    )
    def test_a_killed_robot_app_s_socket_file_closed_to_this_user_refuses_nobody(
        self, mode, robot_app, runtime_dir, caplog, monkeypatch
    ):
        monkeypatch.setattr(_host, "RETRY_S", 0.05)
        other = robot_app()
        assert other.do("enter", mode=mode, robot="killed")["entered"] is True
        other.kill()
        left = default_socket_path()
        os.chmod(left, 0)
        assert lock_is_free()

        session = TeleopSession(_config(mode, robot_description=DESCRIPTION))
        with (
            local_backends() as mocks,
            caplog.at_level(logging.WARNING, logger="isaaccapture"),
            session,
        ):
            # No program holds the path: the robot app runs, and its robot state reaches nobody.
            assert mocks.oxr_cls.call_count == (mode == "live")
            assert mocks.replay_run.call_count == (mode == "replay")
            assert not session.robot_state_active
            assert session.publish_robot_state("arm", [0.5, 0.0, 1.0]) is False
            assert host_warnings(caplog) == [
                "Reader programs cannot get robot state from this robot app: "
                f"{left} cannot be connected to ([Errno 13] Permission denied). Publish "
                "calls return False until the socket can be opened; trying again every 0.05 s"
            ]

            # The file can be connected to again and nobody answers: it is replaced.
            os.chmod(left, 0o600)
            assert _wait_for(
                lambda: session.publish_robot_state("arm", [0.5, 0.0, 1.0]),
                timeout_s=10.0,
            )
            assert session.robot_state_active
            with RobotState.connect(robot="test_arm") as robot:
                assert robot.joints("arm").positions == (0.5, 0.0, 1.0)
        assert len(host_warnings(caplog)) == 1
        assert robot_state_files(runtime_dir) == [LOCK_FILE]


class TestASecondRobotIsRefusedInLive:
    def test_no_openxr_session_is_asked_of_a_factory(self):
        factory = MagicMock()
        other = RobotStateLocalService(RobotStateStore(), default_socket_path())
        other.start()
        try:
            config = local_config(
                robot_description=DESCRIPTION, oxr_session_factory=factory
            )
            session = TeleopSession(config)
            with local_backends() as mocks:
                with pytest.raises(RuntimeError) as raised:
                    session.__enter__()
                assert str(raised.value) == TAKEN.format(path=default_socket_path())
                factory.assert_not_called()
                assert_nothing_started(session, mocks)
        finally:
            other.close()

    def test_a_session_without_a_description_enters_beside_another_robot(
        self, robot_app, runtime_dir
    ):
        other = robot_app()
        assert other.do("enter", mode="live", robot="other")["entered"] is True
        with local_backends() as mocks, TeleopSession(local_config()) as session:
            session.step()
            assert mocks.oxr_cls.call_count == 1
            assert not session.robot_state_active
            assert robot_state_threads() == []
            assert lock_descriptors() in (None, [])
        assert robot_state_files(runtime_dir) == ["robot_state.sock", LOCK_FILE]
        assert other.do("publish", position=0.25) is True


def test_of_two_robot_apps_that_enter_at_once_exactly_one_does(robot_app, runtime_dir):
    apps = {"left": robot_app(), "right": robot_app()}
    for app in apps.values():
        assert app.do("ready") == "done"  # both have finished importing
    refusal = TAKEN.format(path=default_socket_path())
    winners = []
    folder = runtime_dir / "isaaccapture"
    for index in range(ROUNDS):
        modes = {"left": MODES[index % 2], "right": MODES[(index // 2) % 2]}
        if index % 2 == 0 and folder.exists():
            # As on a machine that has served no robot state yet: no folder, no lock file.
            (folder / LOCK_FILE).unlink()
            folder.rmdir()
        at_ns = time.monotonic_ns() + 30_000_000
        for name, app in apps.items():
            app.send("enter", mode=modes[name], robot=name, at_ns=at_ns)
        answers = {name: app.answer("enter") for name, app in apps.items()}

        entered = [name for name, answer in answers.items() if "entered" in answer]
        assert len(entered) == 1, answers
        (winner,) = entered
        (loser,) = set(apps) - {winner}
        assert answers[winner]["entered"] is True
        assert answers[winner]["backends"] > 0
        # The one refused made no OpenXR, DeviceIO or replay session and keeps no thread.
        assert answers[loser] == {"raised": refusal, "backends": 0, "threads": []}
        # The path is the winner's: a reader reads its robot.
        assert apps[winner].do("publish", position=0.5) is True
        with RobotState.connect(robot=winner) as reader:
            assert reader.joints("waist").positions == (0.5,)
        winners.append(winner)
        apps[winner].do("leave")
        assert robot_state_files(runtime_dir) == [LOCK_FILE]
    assert len(winners) == ROUNDS


# ============================================================================
# A session without a description
# ============================================================================

# Enters a LIVE and a REPLAY session without a description, then constructs one of each with a
# description; prints what of robot state exists at each point.
_NO_DESCRIPTION = """
import json, os, sys, threading
from unittest.mock import MagicMock, patch

from isaaccapture.teleop_session_manager import SessionMode, TeleopSession, TeleopSessionConfig


class Pipeline:
    def get_leaf_nodes(self):
        return []

    def execute_pipeline(self, inputs, context=None):
        return {}


def robot_state():
    folder = os.path.join(os.environ["XDG_RUNTIME_DIR"], "isaaccapture")
    files = os.listdir(folder) if os.path.isdir(folder) else []
    return {
        "modules": sorted(m for m in sys.modules if m.startswith("isaaccapture.robot_state.")),
        "threads": sorted(t.name for t in threading.enumerate() if "robot-state" in t.name),
        "files": sorted(name for name in files if "robot_state" in name),
    }


def config(mode, **kwargs):
    if mode == "replay":
        kwargs.update(mode=SessionMode.REPLAY, mcap_config=MagicMock())
    return TeleopSessionConfig(app_name="a", pipeline=Pipeline(), verbose=False, **kwargs)


seen = {}
with (
    patch("isaaccapture.oxr.OpenXRSession"),
    patch("isaaccapture.deviceio.DeviceIOSession.run"),
    patch("isaaccapture.deviceio.DeviceIOSession.get_required_extensions", return_value=[]),
    patch("isaaccapture.deviceio.ReplaySession.run"),
    patch("isaaccapture.deviceio.McapReplayConfig"),
):
    for mode in ("live", "replay"):
        with TeleopSession(config(mode)) as session:
            session.step()
            seen[mode] = robot_state()
            seen[mode]["active"] = session.robot_state_active
        seen[mode + " left"] = robot_state()
    for mode in ("live", "replay"):
        session = TeleopSession(config(mode, robot_description=sys.argv[1]))
        seen[mode + " constructed"] = robot_state()
        seen[mode + " constructed"]["active"] = session.robot_state_active
print(json.dumps(seen))
"""
# The store's native module comes with the session bindings; nothing written in Python does.
_NOTHING = {
    "modules": ["isaaccapture.robot_state._robot_state"],
    "threads": [],
    "files": [],
}


class TestWithoutADescription:
    def test_nothing_of_robot_state_is_loaded_or_created(
        self, description_file, runtime_dir
    ):
        result = subprocess.run(
            [sys.executable, "-c", _NO_DESCRIPTION, str(description_file)],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        seen = json.loads(result.stdout.strip().splitlines()[-1])
        assert seen == {
            "live": {**_NOTHING, "active": False},
            "live left": _NOTHING,
            "replay": {**_NOTHING, "active": False},
            "replay left": _NOTHING,
            "live constructed": {**_NOTHING, "active": False},
            "replay constructed": {**_NOTHING, "active": False},
        }
        assert robot_state_files(runtime_dir) == []

    @pytest.mark.parametrize("mode", MODES)
    def test_a_session_entered_without_one_serves_nothing(self, mode, runtime_dir):
        with local_backends(), TeleopSession(_config(mode)) as session:
            session.step()
            assert not session.robot_state_active
            assert session._robot_state_sender is None
            assert reader_threads() == []
            assert robot_state_files(runtime_dir) == []
            with RobotState.connect() as robot:
                assert not robot.server_reachable


# ============================================================================
# The same robot state as a remote session delivers
# ============================================================================

SERIES_FIELDS = (
    "name",
    "shape",
    "schema_name",
    "schema_encoding",
    "schema_data",
    "message_encoding",
    "stale_after_ns",
    "in_order",
    "rate_hz",
)
PUBLISHES = [
    ("arm", lambda s: s.publish_robot_state("arm", [0.25, -0.5, 1.0], time_ns=11)),
    ("arm", lambda s: s.publish_robot_state("arm", [0.5, 0.0, 0.0], time_ns=12)),
    (
        "status",
        lambda s: s.publish_robot_state(
            "status", [True, False], faults=[("overtemperature", "hot")], time_ns=13
        ),
    ),
    (
        "wrist_force",
        lambda s: s.publish_robot_record(
            "wrist_force", {"fx": 1.0, "fy": 2.0, "contact": True}, time_ns=14
        ),
    ),
    (
        "arm",
        lambda s: s.publish_robot_state(
            "arm", [0.75, 0.0, 0.5], valid=[True, False, True], time_ns=15
        ),
    ),
    ("wrist_force", lambda s: s.publish_robot_state("wrist_force", [0.0, 0.0, False])),
]


class _StillClock:
    """The `time` module with a monotonic_ns() that stands still: equal publishes then encode
    to equal bytes."""

    def __init__(self):
        self._now_ns = time.monotonic_ns()

    def monotonic_ns(self):
        return self._now_ns

    def __getattr__(self, name):
        return getattr(time, name)


def _contents(store):
    return {
        "series": [
            {name: getattr(info, name) for name in SERIES_FIELDS}
            for info in store.series()
        ],
        "records": {
            name: (
                store.latest(name).data,
                store.latest(name).sequence,
                store.latest(name).sample_time_ns,
            )
            for name in SERIES_NAMES
        },
    }


def test_a_local_session_holds_what_a_remote_one_delivers(monkeypatch):
    clock = _StillClock()
    monkeypatch.setattr(teleop_session, "time", clock)
    monkeypatch.setattr(_host, "time", clock)

    config = local_config(robot_description=DESCRIPTION)
    with local_backends(), TeleopSession(config) as session:
        for _series, publish in PUBLISHES:
            assert publish(session)
        local = _contents(session._robot_state_sender._store)

    store = RobotStateStore()
    server = _MiniServer(answer=AGREED, store=store)
    try:
        config = _loopback_config(server, robot_description=DESCRIPTION)
        with TeleopSession(config) as session:
            assert _wait_for(lambda: store.latest("robot_description").data is not None)
            for series, publish in PUBLISHES:
                # Each record has arrived before the next is published: none is dropped.
                before = store.latest(series).sequence
                assert publish(session)
                assert _wait_for(
                    lambda series=series, before=before: (
                        store.latest(series).sequence != before
                    )
                )
            remote = _contents(store)
            assert store.dropped_records == 0
    finally:
        server.stop()

    assert [entry["name"] for entry in local["series"]] == SERIES_NAMES
    assert all(data is not None for data, _, _ in local["records"].values())
    assert records.decode_description(local["records"]["robot_description"][0]) == (
        DESCRIPTION
    )
    assert local == remote
