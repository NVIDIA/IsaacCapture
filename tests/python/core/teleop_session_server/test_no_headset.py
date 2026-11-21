# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Teleop Session Server with a robot app that does not wait for a tracked headset.

The server's loop runs in process on a thread, with its real listener and RemoteSessionHost on a
free loopback port; stand-ins the test steers replace oxr.OpenXRSession and CloudXR. The robot
app is a TeleopSession with an empty pipeline; a reader reads at the default socket path, under
a temporary XDG_RUNTIME_DIR. No OpenXR runtime and no headset is used.
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
from isaaccapture.deviceio_session import RemoteSessionListener
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
from isaaccapture.teleop_session_server import _session

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
    while ``tracked``.
    """

    def __init__(self):
        self.reached = False
        self.tracked = False
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
    def health_check(self):
        pass


class _Running:
    """A _Server with an OpenXR session of its own (no placeholder), on a thread until stop()."""

    def __init__(self):
        listening = server_main._Listening(
            RemoteSessionListener("127.0.0.1", 0), False, "tcp://127.0.0.1"
        )
        self._stop = server_main._StopRequest()
        self.server = server_main._Server([listening], _Launcher(), self._stop)
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
# The server's ERROR after early entry
# ============================================================================


def test_a_tracker_start_failure_once_a_headset_arrives_ends_the_session_in_step(
    server, openxr, logs
):
    with RobotState.connect(robot=ROBOT) as reader:
        with _early(server) as session:
            robot = _Robot(session, reader)
            assert robot.reads()
            robot.run_for(0.3)
            assert robot.newest_shows()

            # A headset reaches the runtime; the server cannot start the robot app's trackers.
            openxr.handles = NO_HANDLES
            openxr.reached = True
            with pytest.raises(RuntimeError) as raised:
                _step_for(session, 10.0)
            assert type(raised.value) is RuntimeError
            assert str(raised.value) == START_FAILURE

            assert _wait_for(
                lambda: not session.publish_robot_state("arm", _reading(0))
            )
            provider = session.get_provider_status("openxr/runtime")
            assert provider.status == ProviderState.FAILED
            # Robot state ended with the session; the last reading stays readable as stale.
            assert _wait_for(lambda: not reader.connected)
            assert reader.server_reachable
            assert reader.description() == DESCRIPTION
            assert reader.joints("arm") is None
            assert reader.joints("arm", allow_stale=True) is not None

        # The server goes on without that robot app.
        assert server.running
        assert [text for text in logs() if text.endswith(": " + START_FAILURE)]
    assert server.stop() == 0


def test_a_server_that_stops_on_an_openxr_failure_ends_the_session_in_step(
    server, openxr, logs
):
    with RobotState.connect(robot=ROBOT) as reader:
        with _early(server) as session:
            robot = _Robot(session, reader)
            assert robot.reads()
            robot.run_for(0.3)

            # The server's next try to open its OpenXR session fails for another reason.
            openxr.error = OTHER_OPENXR_ERROR
            with pytest.raises(RuntimeError) as raised:
                _step_for(session, 10.0)
            assert type(raised.value) is RuntimeError
            assert str(raised.value) == OPEN_FAILURE
            assert _wait_for(
                lambda: not session.publish_robot_state("arm", _reading(0))
            )

        assert server.wait() == 1
        # The server is gone, and its reader socket with it.
        assert not reader.server_reachable
        assert reader.joints("arm", allow_stale=True) is None
    assert not os.path.exists(default_socket_path())
    assert f"{OPEN_FAILURE}; stopping the Teleop Session Server" in logs()
