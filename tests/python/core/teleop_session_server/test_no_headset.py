# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Teleop Session Server with a robot app that does not wait for a tracked headset.

The server's loop runs in process on a thread, with its real listener and RemoteSessionHost on a
free loopback port; stand-ins the test steers replace oxr.OpenXRSession and CloudXR. The robot
app is a TeleopSession with an empty pipeline; a reader reads at the default socket path, under
a temporary XDG_RUNTIME_DIR. No OpenXR runtime and no headset is used.

The server's retry delays are the real ones, so a test that waits for a retry takes that long.
"""

import logging
import os
import signal
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from isaaccapture import oxr
from isaaccapture.deviceio_session import RemoteSessionHost, RemoteSessionListener
from isaaccapture.robot_state import (
    RobotState,
    default_socket_path,
    parse_robot_description,
)
from isaaccapture.teleop_session_manager import (
    DeviceState,
    ProviderState,
    SessionMode,
    StatusReason,
    TeleopSession,
    TeleopSessionConfig,
)
from isaaccapture.teleop_session_server import __main__ as server_main
from isaaccapture.teleop_session_server import _common, _session

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the Teleop Session Server runs on Linux only"
)

# How oxr.OpenXRSession(wait_for_system=False) fails while no headset has reached the runtime.
NO_HEADSET_ERROR = (
    "Failed to get OpenXR system: XR_ERROR_FORM_FACTOR_UNAVAILABLE (-35). The OpenXR runtime "
    "is up but no headset is connected to it"
)
OTHER_OPENXR_ERROR = "Failed to create OpenXR instance: -51"
WAITING = _session.WAITING_FOR_HEADSET
READY_AFTER_S = _session.READY_AFTER_CONNECTED_S
# The specified minimum a headset stays tracked before READY.
MIN_TRACKED_S = 1.1
# The robot app's headset error before the server's OpenXR session exists, and while that
# session tracks no headset.
CONNECTING_OPENXR = f"Teleop Session Server: connecting_openxr ({WAITING})"
NOT_TRACKED = (
    "Teleop Session Server reports OpenXR AVAILABLE, headset DISCONNECTED "
    "(TRACKING_UNAVAILABLE)"
)
SERVER_WAITS = "No headset is connected to CloudXR; waiting for one"
SERVER_READY = "Ready; streaming to the robot app at "
# Zero handles, which RemoteSessionHost.start_session() refuses before it uses them.
NO_HANDLES = (0, 0, 0, 0)
START_FAILURE = (
    "The Teleop Session Server could not start the trackers: "
    "RemoteSessionHost.start_session: invalid OpenXRSessionHandles (instance, session, space "
    "must be non-null handles and xrGetInstanceProcAddr must be set)"
)
OPEN_FAILURE = (
    "The Teleop Session Server could not open an OpenXR session: " + OTHER_OPENXR_ERROR
)
# What the stand-in OpenXR session reports once it failed, and the server's reason for it.
INSTANCE_LOST = "xrPollEvent failed because the OpenXR instance was lost"
SESSION_FAILURE = f"The OpenXR session on the Teleop Session Server failed (INSTANCE_LOST): {INSTANCE_LOST}"
# What the stand-in CloudXR's health check raises while it is down.
CLOUDXR_STOPPED = "The CloudXR runtime serving /run has stopped"
KEEPS_SERVING = (
    "Robot apps are served without headset data until this is fixed; the Teleop Session "
    "Server keeps checking and needs no restart"
)
START_HINT = ". Start CloudXR with: python -m isaaccapture.cloudxr.service start"
STAYS_CONNECTED = (
    "It stays connected without tracker data; the server starts the trackers again "
    "by itself"
)
TRACKERS_RUN_AGAIN = "The trackers of the robot app at "
# The robot app's lines when SAMPLES stop and when they arrive again.
SAMPLES_STALE = (
    "RemoteSession: samples stale (tick age over 250 ms); trackers report no data"
)
SAMPLES_FRESH = "RemoteSession: receiving fresh samples"
RETRY_S = _common.RETRY_S
# How far past a retry delay the server's retry may show on a loaded test machine.
RETRY_SLACK_S = 5.0

ROBOT = "test_arm"
# stale_after_s leaves room for a loaded test machine; MAX_WAIT_S below is the stricter check.
DESCRIPTION = parse_robot_description(
    f"""
robot = "{ROBOT}"

[series.arm]
shape = "joints"
stale_after_s = 2.0
joints = [
  {{ name = "joint1",  kind = "revolute", min = -2.0, max = 2.0 }},
  {{ name = "gripper", kind = "gripper",  min = 0.0,  max = 1.0 }},
]
"""
)
STEP_S = 0.005
# Longest a reader may go without a newer reading while the robot app publishes every STEP_S.
MAX_WAIT_S = 1.0
# Above the link's 3 s peer timeout: the server sends one STATUS and then nothing that long.
NO_HEADSET_S = 3.5
# How far past READY_AFTER_S the robot app may learn of READY on a loaded test machine.
READY_SLACK_S = 2.0


@pytest.fixture(autouse=True)
def runtime_dir(tmp_path_factory, monkeypatch):
    """The default reader socket is under this folder."""
    path = tmp_path_factory.mktemp("rt")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(path))
    assert default_socket_path().startswith(str(path))
    return path


class _OpenXR:
    """oxr.OpenXRSession stand-in the test steers.

    Construction fails with ``error`` until ``reached``; the session then tracks a headset
    while ``tracked``, and reports that it failed while ``failed``.
    """

    def __init__(self):
        self.reached = False
        self.tracked = False
        self.failed = False
        self.error = NO_HEADSET_ERROR
        self.handles = (7, 7, 7, 7)
        # wait_for_system of every construction.
        self.attempts = []
        self.opened = 0
        self.closed = 0

    def __call__(self, app_name, extensions, wait_for_system=True):
        self.attempts.append(wait_for_system)
        if not self.reached:
            raise RuntimeError(self.error)
        self.opened += 1
        return self

    def get_handles(self):
        return oxr.OpenXRSessionHandles(*self.handles)

    def get_provider_snapshot(self):
        if self.failed:
            return SimpleNamespace(
                state=oxr.OpenXRProviderState.FAILED,
                headset_state=oxr.OpenXRHeadsetState.DISCONNECTED,
                reason=oxr.OpenXRProviderReason.INSTANCE_LOST,
                error=INSTANCE_LOST,
            )
        tracked = self.tracked
        return SimpleNamespace(
            state=oxr.OpenXRProviderState.AVAILABLE,
            headset_state=oxr.OpenXRHeadsetState.CONNECTED
            if tracked
            else oxr.OpenXRHeadsetState.DISCONNECTED,
            reason=oxr.OpenXRProviderReason.NONE
            if tracked
            else oxr.OpenXRProviderReason.TRACKING_UNAVAILABLE,
            error="",
        )

    def close(self):
        self.closed += 1


class _Launcher:
    """CloudXR stand-in the test steers: its health check fails while ``down``."""

    def __init__(self):
        self.down = False

    def health_check(self):
        if self.down:
            raise RuntimeError(CLOUDXR_STOPPED)


class _Running:
    """A _Server with an OpenXR session of its own (no placeholder), on a thread until stop()."""

    def __init__(self):
        listening = server_main._Listening(
            RemoteSessionListener("127.0.0.1", 0), False, "tcp://127.0.0.1"
        )
        self._stop = server_main._StopRequest()
        self.launcher = _Launcher()
        self.server = server_main._Server([listening], self.launcher, self._stop)
        self.endpoint = f"127.0.0.1:{listening.listener.port}"
        self.code = None
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def _run(self):
        self.code = self.server.run()

    @property
    def running(self):
        return self._thread.is_alive()

    def wait(self, timeout_s=30.0):
        """The exit code once run() has returned by itself."""
        self._thread.join(timeout=timeout_s)
        assert not self._thread.is_alive()
        return self.code

    def stop(self):
        self._stop.signal = signal.SIGTERM
        return self.wait()


@pytest.fixture
def openxr(monkeypatch):
    scripted = _OpenXR()
    monkeypatch.setattr(oxr, "OpenXRSession", scripted)
    return scripted


@pytest.fixture
def logs(caplog):
    """The log texts so far, the ones of the server's start (a fixture) included."""
    caplog.set_level(logging.INFO, logger="isaaccapture")
    return lambda: [
        record.getMessage()
        for when in ("setup", "call")
        for record in caplog.get_records(when)
    ]


@pytest.fixture
def server(openxr, logs):
    running = _Running()
    yield running
    running.stop()


class _Pipeline:
    def get_leaf_nodes(self):
        return []

    def execute_pipeline(self, inputs, context=None):
        return {}


def _robot_app(server, **kwargs):
    kwargs.setdefault("remote_connect_timeout_s", 10.0)
    return TeleopSession(
        TeleopSessionConfig(
            app_name="NoHeadsetTest",
            pipeline=_Pipeline(),
            mode=SessionMode.REMOTE,
            remote_endpoint=server.endpoint,
            robot_description=DESCRIPTION,
            verbose=False,
            **kwargs,
        )
    )


def _early(server):
    return _robot_app(server, wait_for_headset=False)


def _wait_for(predicate, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(STEP_S)
    return True


def _step_for(session, seconds):
    """Step the robot app for `seconds`; a step that raises ends it."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        session.step()
        time.sleep(STEP_S)


def _headset(session):
    return session.get_device_status("openxr/headset")


def _tick_age_s(session):
    return session.last_step_info.remote_tick_age_s


def _reading(count):
    """A reading of the arm that tells `count` apart from its neighbours, exact in a float."""
    return [(count % 1024) / 1024.0, 1.0]


def _is_ready(server):
    return f"Teleop Session Server at {server.endpoint} is ready"


class _Robot:
    """The robot app's loop: each step() steps the session and publishes one reading of the arm.

    It also reads the arm as a reader program does. Once a reading has shown, every step must
    find a fresh one; ``longest_wait_s`` is the longest the reader went without a newer one.
    """

    def __init__(self, session, reader):
        self.session = session
        self.reader = reader
        self.published = 0
        self.sequence = None
        self.advanced_at = None
        self.longest_wait_s = 0.0

    def step(self):
        assert self.session.step() == {}
        self.published += 1
        assert self.session.publish_robot_state("arm", _reading(self.published))
        arm = self.reader.joints("arm")
        now = time.monotonic()
        if arm is None:
            assert self.sequence is None, "the reader lost the arm's fresh reading"
            return
        if self.advanced_at is not None:
            self.longest_wait_s = max(self.longest_wait_s, now - self.advanced_at)
        if self.sequence is None or arm.sequence > self.sequence:
            self.sequence = arm.sequence
            self.advanced_at = now
        assert arm.sequence == self.sequence, "the reader went back to an older reading"

    def run_until(self, predicate, timeout_s=10.0):
        deadline = time.monotonic() + timeout_s
        while True:
            self.step()
            if predicate():
                return True
            if time.monotonic() > deadline:
                return False
            time.sleep(STEP_S)

    def run_for(self, seconds):
        self.run_until(lambda: False, seconds)

    def reads(self):
        return self.run_until(lambda: self.sequence is not None)

    def newest_shows(self):
        """Whether the reader gets the reading published last, with no further publish."""
        newest = _reading(self.published)[0]
        return _wait_for(
            lambda: (
                (arm := self.reader.joints("arm")) is not None
                and arm["joint1"].position == newest
            )
        )


# ============================================================================
# Early entry, then a headset
# ============================================================================


def test_readings_flow_from_entry_through_every_change_of_the_headset(
    server, openxr, logs
):
    ready = _is_ready(server)
    with RobotState.connect(robot=ROBOT) as reader:
        # The server opens the reader socket with the first robot app that sends robot state.
        assert not reader.server_reachable

        with _early(server) as session:
            # Phase 1: no headset has reached the server's OpenXR runtime.
            assert (
                f"Teleop Session Server at {server.endpoint} answered; "
                "not waiting for a tracked headset"
            ) in logs()
            assert openxr.opened == 0
            assert session.robot_state_active
            robot = _Robot(session, reader)
            assert robot.reads()
            assert reader.server_reachable
            assert reader.connected
            assert reader.description() == DESCRIPTION
            assert server.server._robot_state_service.path == default_socket_path()

            robot.run_for(NO_HEADSET_S)
            headset = _headset(session)
            assert (headset.status, headset.reason, headset.error) == (
                DeviceState.DISCONNECTED,
                StatusReason.NO_CURRENT_DATA,
                CONNECTING_OPENXR,
            )
            provider = session.get_provider_status("openxr/runtime")
            assert provider.status == ProviderState.AVAILABLE
            assert _tick_age_s(session) is None
            assert openxr.opened == 0
            assert len(openxr.attempts) >= 3
            assert set(openxr.attempts) == {False}
            assert logs().count(SERVER_WAITS) == 1
            assert ready not in logs()
            assert robot.newest_shows()

            # Phase 2: the server's OpenXR session opens; it tracks no headset.
            openxr.reached = True
            assert robot.run_until(lambda: _headset(session).error == NOT_TRACKED)
            assert openxr.opened == 1
            attempts = len(openxr.attempts)
            assert _headset(session).status == DeviceState.DISCONNECTED
            robot.run_for(MIN_TRACKED_S + 0.4)
            assert _headset(session).error == NOT_TRACKED
            assert robot.run_until(lambda: _tick_age_s(session) is not None)
            assert ready not in logs()
            assert [text for text in logs() if text.startswith(SERVER_READY)] == []

            # Phase 3: the headset is tracked; READY follows once it has stayed tracked.
            tracked_at = time.monotonic()
            openxr.tracked = True
            assert robot.run_until(lambda: ready in logs())
            waited_s = time.monotonic() - tracked_at
            assert MIN_TRACKED_S <= waited_s < READY_AFTER_S + READY_SLACK_S
            assert robot.run_until(
                lambda: _headset(session).status == DeviceState.CONNECTED
            )
            assert _headset(session).reason == StatusReason.OPENXR_SESSION_READY
            assert robot.run_until(
                lambda: (age := _tick_age_s(session)) is not None and 0.0 <= age < 0.25
            )

            # Phase 4: the headset is taken off and put on again.
            openxr.tracked = False
            assert robot.run_until(lambda: _headset(session).error == NOT_TRACKED)
            assert _headset(session).status == DeviceState.DISCONNECTED
            openxr.tracked = True
            assert robot.run_until(
                lambda: _headset(session).status == DeviceState.CONNECTED
            )
            robot.run_for(0.3)

            assert logs().count(ready) == 1
            assert len([t for t in logs() if t.startswith(SERVER_READY)]) == 1
            assert session.robot_state_active
            assert robot.newest_shows()
            assert robot.longest_wait_s < MAX_WAIT_S
            # One OpenXR session for the server's life.
            assert len(openxr.attempts) == attempts

        # The robot app is gone; the server, its OpenXR session and the reader stay.
        assert _wait_for(lambda: not reader.connected)
        assert reader.server_reachable
        assert reader.description() == DESCRIPTION
        assert reader.joints("arm") is None
        newest = reader.joints("arm", allow_stale=True)
        assert newest["joint1"].position == _reading(robot.published)[0]
        assert server.running
        assert (openxr.opened, openxr.closed) == (1, 0)

    assert server.stop() == 0
    assert openxr.closed == 1
    assert not os.path.exists(default_socket_path())


def test_with_the_default_the_same_server_is_not_ready(server, openxr, logs):
    """The twin of early entry: only wait_for_headset differs."""
    with RobotState.connect(robot=ROBOT) as reader:
        started = time.monotonic()
        with pytest.raises(TimeoutError) as raised:
            with _robot_app(server, remote_connect_timeout_s=0.5):
                pytest.fail("entry returned without a tracked headset")
        assert str(raised.value) == (
            f"Teleop Session Server at {server.endpoint} was not ready after 0.5 s: "
            f"connecting_openxr ({WAITING})"
        )
        assert time.monotonic() - started >= 0.5
        assert _wait_for(lambda: not reader.connected)
        assert reader.joints("arm", allow_stale=True) is None
        assert openxr.opened == 0

        # The same server, still without a headset, serves a robot app that does not wait.
        with _early(server) as session:
            robot = _Robot(session, reader)
            assert robot.reads()
            assert reader.connected
            assert _headset(session).status == DeviceState.DISCONNECTED
    assert server.stop() == 0


def test_a_robot_app_entering_while_a_headset_is_tracked_is_ready_each_time(
    server, openxr, logs
):
    ready = _is_ready(server)
    openxr.reached = True
    openxr.tracked = True
    assert _wait_for(lambda: openxr.opened == 1)
    session = _early(server)

    def enter(reader, entry):
        entered_at = time.monotonic()
        with session:
            robot = _Robot(session, reader)
            assert logs().count(ready) == entry - 1
            assert robot.reads()
            assert robot.run_until(lambda: logs().count(ready) == entry)
            waited_s = time.monotonic() - entered_at
            assert MIN_TRACKED_S <= waited_s < READY_AFTER_S + READY_SLACK_S
            assert robot.run_until(
                lambda: _headset(session).status == DeviceState.CONNECTED
            )
            robot.run_for(0.2)
            assert logs().count(ready) == entry
            assert robot.newest_shows()
            assert robot.longest_wait_s < MAX_WAIT_S
        assert _wait_for(lambda: not reader.connected)

    with RobotState.connect(robot=ROBOT) as reader:
        enter(reader, 1)
        enter(reader, 2)
    assert openxr.opened == 1
    assert server.stop() == 0


# ============================================================================
# A headset side that fails: the server, the session and robot state go on
# ============================================================================


@pytest.fixture
def starts(monkeypatch):
    """When the server called RemoteSessionHost.start_session(), failed calls included."""
    tried = []
    start_session = RemoteSessionHost.start_session

    def counted(self, *args, **kwargs):
        tried.append(time.monotonic())
        return start_session(self, *args, **kwargs)

    monkeypatch.setattr(RemoteSessionHost, "start_session", counted)
    return tried


def _before_openxr(reason):
    """The robot app's headset error while the server has no trackers and sent no READY."""
    return f"Teleop Session Server: connecting_openxr ({reason})"


def _warning(reason, start_hint=False):
    """The server's warning about a headset side that cannot be used."""
    return f"{reason}. {KEEPS_SERVING}{START_HINT if start_hint else ''}"


def _robot_app_said(logs, text):
    """How often the robot app's RemoteSession logged a line that starts with ``text``."""
    return len([line for line in logs() if line.startswith(text)])


def _is_fresh(session):
    age = _tick_age_s(session)
    return age is not None and 0.0 <= age < 0.25


def _no_headset_data(session):
    """A headset that is not tracked, with stale samples: every tracker reports no data."""
    headset = _headset(session)
    return (
        headset.status == DeviceState.DISCONNECTED
        and headset.error == NOT_TRACKED
        and _tick_age_s(session) is None
    )


def test_a_tracker_start_failure_once_a_headset_arrives_keeps_the_session_and_robot_state(
    server, openxr, logs, starts
):
    failure = _before_openxr(START_FAILURE)
    with RobotState.connect(robot=ROBOT) as reader:
        with _early(server) as session:
            robot = _Robot(session, reader)
            assert robot.reads()
            robot.run_for(0.3)
            assert robot.newest_shows()

            # A headset reaches the runtime; the server cannot start the robot app's trackers.
            openxr.handles = NO_HANDLES
            openxr.reached = True
            assert robot.run_until(lambda: _headset(session).error == failure)
            headset = _headset(session)
            assert (headset.status, headset.reason) == (
                DeviceState.DISCONNECTED,
                StatusReason.NO_CURRENT_DATA,
            )
            provider = session.get_provider_status("openxr/runtime")
            assert provider.status == ProviderState.AVAILABLE

            # The server starts them again by itself, on the OpenXR session it keeps.
            assert robot.run_until(lambda: len(starts) >= 2, RETRY_S + RETRY_SLACK_S)
            assert starts[1] - starts[0] >= RETRY_S - 0.1
            assert _headset(session).error == failure
            assert _tick_age_s(session) is None
            assert (openxr.opened, openxr.closed) == (1, 0)
            said = [text for text in logs() if text.endswith(f". {STAYS_CONNECTED}")]
            assert len(said) == 1
            assert said[0].endswith(f": {START_FAILURE}. {STAYS_CONNECTED}")

            # CloudXR is started again with a headset whose trackers start.
            openxr.handles = (7, 7, 7, 7)
            server.launcher.down = True
            assert robot.run_until(lambda: openxr.closed == 1)
            server.launcher.down = False
            assert robot.run_until(lambda: _headset(session).error == NOT_TRACKED)
            assert robot.run_until(lambda: _is_fresh(session))
            assert openxr.opened == 2
            assert [t for t in logs() if t.startswith(TRACKERS_RUN_AGAIN)]

            assert session.robot_state_active
            assert reader.connected
            assert robot.newest_shows()
            assert robot.longest_wait_s < MAX_WAIT_S
        assert server.running
    assert server.stop() == 0


def test_an_openxr_failure_keeps_the_server_the_session_and_robot_state(
    server, openxr, logs
):
    ready = _is_ready(server)
    with RobotState.connect(robot=ROBOT) as reader:
        with _early(server) as session:
            robot = _Robot(session, reader)
            assert robot.reads()
            robot.run_for(0.3)

            # The server's next try to open its OpenXR session fails for another reason.
            openxr.error = OTHER_OPENXR_ERROR
            assert robot.run_until(
                lambda: _headset(session).error == _before_openxr(OPEN_FAILURE)
            )
            assert _headset(session).status == DeviceState.DISCONNECTED
            attempts = len(openxr.attempts)
            assert server.running

            # Its next try, a retry delay later, finds a tracked headset.
            openxr.reached = True
            openxr.tracked = True
            assert robot.run_until(
                lambda: _headset(session).status == DeviceState.CONNECTED,
                RETRY_S + READY_AFTER_S + RETRY_SLACK_S,
            )
            assert len(openxr.attempts) == attempts + 1
            assert robot.run_until(lambda: _is_fresh(session))
            assert robot.run_until(lambda: ready in logs())

            assert logs().count(_warning(OPEN_FAILURE)) == 1
            assert session.robot_state_active
            assert robot.newest_shows()
            assert robot.longest_wait_s < MAX_WAIT_S
        assert _wait_for(lambda: not reader.connected)
        # The server and its reader socket are still there.
        assert server.running
        assert reader.server_reachable
        assert reader.joints("arm", allow_stale=True) is not None
    assert server.stop() == 0
    assert not os.path.exists(default_socket_path())
    stopping = [t for t in logs() if "stopping the Teleop Session Server" in t]
    assert stopping == ["Received SIGTERM; stopping the Teleop Session Server"]


def test_cloudxr_stopping_and_returning_keeps_the_session_and_robot_state(
    server, openxr, logs
):
    ready = _is_ready(server)
    openxr.reached = True
    openxr.tracked = True
    with RobotState.connect(robot=ROBOT) as reader:
        with _early(server) as session:
            robot = _Robot(session, reader)
            assert robot.reads()
            assert robot.run_until(lambda: ready in logs())
            assert robot.run_until(
                lambda: _headset(session).status == DeviceState.CONNECTED
            )
            assert robot.run_until(lambda: _is_fresh(session))
            stale = _robot_app_said(logs, SAMPLES_STALE)

            # CloudXR stops: the trackers and the OpenXR session they read through go.
            server.launcher.down = True
            assert robot.run_until(lambda: _no_headset_data(session))
            # The robot app says so; its C++ log lines reach Python a moment later.
            assert robot.run_until(
                lambda: _robot_app_said(logs, SAMPLES_STALE) == stale + 1
            )
            assert _headset(session).reason == StatusReason.NO_CURRENT_DATA
            assert (openxr.opened, openxr.closed) == (1, 1)
            attempts = len(openxr.attempts)
            # It stays so, and OpenXR is not called, while CloudXR is down.
            robot.run_for(1.5 * server_main.HEALTH_CHECK_PERIOD_S)
            assert _no_headset_data(session)
            assert len(openxr.attempts) == attempts
            assert reader.connected
            # Said once for the outage, however long it lasts.
            assert _robot_app_said(logs, SAMPLES_STALE) == stale + 1
            fresh = _robot_app_said(logs, SAMPLES_FRESH)

            # CloudXR is started again: the same session gets tracker data again.
            server.launcher.down = False
            assert robot.run_until(
                lambda: _headset(session).status == DeviceState.CONNECTED
            )
            assert robot.run_until(lambda: _is_fresh(session))
            assert (openxr.opened, openxr.closed) == (2, 1)
            assert len(openxr.attempts) == attempts + 1
            assert robot.run_until(
                lambda: _robot_app_said(logs, SAMPLES_FRESH) == fresh + 1
            )

            assert logs().count(_warning(CLOUDXR_STOPPED, start_hint=True)) == 1
            assert logs().count("CloudXR is available again") == 1
            # READY is sent once for a connection.
            assert logs().count(ready) == 1
            assert len([t for t in logs() if t.startswith(SERVER_READY)]) == 1
            assert session.robot_state_active
            assert robot.newest_shows()
            assert robot.longest_wait_s < MAX_WAIT_S
        assert server.running
    assert server.stop() == 0
    assert openxr.closed == 2


def test_a_robot_app_that_waited_for_the_headset_survives_an_outage_after_ready(
    server, openxr, logs
):
    ready = _is_ready(server)
    openxr.reached = True
    openxr.tracked = True
    with RobotState.connect(robot=ROBOT) as reader:
        # The default: entry returns at READY.
        with _robot_app(server) as session:
            assert logs().count(ready) == 1
            robot = _Robot(session, reader)
            assert robot.reads()
            assert robot.run_until(lambda: _is_fresh(session))
            assert _headset(session).status == DeviceState.CONNECTED

            # The server's OpenXR session fails; it reads as a headset taken off.
            openxr.failed = True
            assert robot.run_until(lambda: openxr.closed == 1)
            openxr.failed = False
            assert robot.run_until(lambda: _no_headset_data(session))
            provider = session.get_provider_status("openxr/runtime")
            assert provider.status == ProviderState.AVAILABLE

            # A retry delay later the server has a new OpenXR session.
            assert robot.run_until(
                lambda: _headset(session).status == DeviceState.CONNECTED,
                RETRY_S + RETRY_SLACK_S,
            )
            assert robot.run_until(lambda: _is_fresh(session))
            assert (openxr.opened, openxr.closed) == (2, 1)

            assert logs().count(_warning(SESSION_FAILURE)) == 1
            assert logs().count(ready) == 1
            assert robot.newest_shows()
            assert robot.longest_wait_s < MAX_WAIT_S
        assert server.running
    assert server.stop() == 0


def test_a_robot_app_that_waits_for_the_headset_times_out_with_the_reason_while_cloudxr_is_down(
    server, openxr, logs
):
    server.launcher.down = True
    warning = _warning(CLOUDXR_STOPPED, start_hint=True)
    assert _wait_for(lambda: warning in logs())
    attempts = len(openxr.attempts)

    with RobotState.connect(robot=ROBOT) as reader:
        with pytest.raises(TimeoutError) as raised:
            with _robot_app(server, remote_connect_timeout_s=1.0):
                pytest.fail("entry returned while CloudXR is down")
        assert str(raised.value) == (
            f"Teleop Session Server at {server.endpoint} was not ready after 1 s: "
            f"connecting_openxr ({CLOUDXR_STOPPED})"
        )
        assert _wait_for(lambda: not reader.connected)

        # The same server serves a robot app that does not wait.
        with _early(server) as session:
            robot = _Robot(session, reader)
            assert robot.reads()
            assert robot.run_until(
                lambda: _headset(session).error == _before_openxr(CLOUDXR_STOPPED)
            )
            assert robot.newest_shows()
        assert _wait_for(lambda: not reader.connected)
        assert len(openxr.attempts) == attempts

        # CloudXR is started and a headset connects: a robot app that waits enters.
        openxr.reached = True
        openxr.tracked = True
        server.launcher.down = False
        with _robot_app(server) as session:
            assert _headset(session).status == DeviceState.CONNECTED
            robot = _Robot(session, reader)
            assert robot.reads()
            assert robot.run_until(lambda: _is_fresh(session))
    assert logs().count(warning) == 1
    assert server.stop() == 0
