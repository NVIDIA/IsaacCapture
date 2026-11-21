# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The server's loop in process: its OpenXR session and the robot app connections it serves.

Scripted stand-ins replace the listener, RemoteSessionHost (one per robot app), the OpenXR session
and CloudXR; a virtual clock replaces time.monotonic() and time.sleep(), and refusals run inline.
One test checks CloudXR through the server's own class, with a scripted runtime behind it.
"""

import json
import logging
import math
import os
import signal
import sys
from types import SimpleNamespace

import pytest

import isaaccapture
from isaaccapture import oxr
from isaaccapture.deviceio_trackers import HandTracker, HeadTracker
from isaaccapture.teleop_session_server import __main__ as server_main
from isaaccapture.teleop_session_server import _cloudxr, _common, _session

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the Teleop Session Server runs on Linux only"
)

NO_HEADSET_ERROR = (
    "Failed to get OpenXR system: XR_ERROR_FORM_FACTOR_UNAVAILABLE (-35). The OpenXR runtime "
    "is up but no headset is connected to it"
)
CONNECTED = oxr.OpenXRHeadsetState.CONNECTED
DISCONNECTED = oxr.OpenXRHeadsetState.DISCONNECTED
TICK_RATE_HZ = 100.0
TICK_S = 1.0 / TICK_RATE_HZ
BUSY = _common.BUSY_MESSAGE
SHUTDOWN = _common.SHUTDOWN_MESSAGE
WAITING = _session.WAITING_FOR_HEADSET
CONNECTED_ENTRY = {"state": "AVAILABLE", "headset_state": "CONNECTED", "reason": "NONE"}
# The STATUS headset entries while no trackers run: before READY, and from READY on.
NO_ENTRY = {"state": "", "headset_state": "", "reason": ""}
NOT_TRACKED_ENTRY = {
    "state": "AVAILABLE",
    "headset_state": "DISCONNECTED",
    "reason": "TRACKING_UNAVAILABLE",
}
# What the CloudXR stand-in's health check raises.
STOPPED = "The CloudXR runtime serving /run has stopped"
KEEPS_SERVING = (
    "Robot apps are served without headset data until this is fixed; the Teleop Session "
    "Server keeps checking and needs no restart"
)
START_HINT = ". Start CloudXR with: python -m isaaccapture.cloudxr.service start"
STAYS_CONNECTED = (
    "It stays connected without tracker data; the server starts the trackers again "
    "by itself"
)
OPEN_FAILED = "The Teleop Session Server could not open an OpenXR session: "
START_FAILED = "The Teleop Session Server could not start the trackers: "
UPDATE_FAILED = "The Teleop Session Server could not update the trackers: "
SESSION_LOST = (
    "The OpenXR session on the Teleop Session Server failed (SESSION_LOST): "
    "XR_ERROR_SESSION_LOST"
)
# Virtual seconds after which a scripted run stops, so a regression fails instead of hanging.
MAX_RUN_S = 600.0


def _between(start, end):
    return lambda now: start <= now < end


def _warning(reason, start_hint=False):
    """The server's warning about a headset side that cannot be used."""
    return f"{reason}. {KEEPS_SERVING}{START_HINT if start_hint else ''}"


def _hello(trackers=("HeadTracker", "HandTracker"), **fields):
    hello = {
        "protocol": 1,
        "isaaccapture_version": isaaccapture.__version__,
        "app_name": "Robot",
        "tick_rate_hz": TICK_RATE_HZ,
        "trackers": [
            {"name": str(i), "type": type_name, "params": {}, "vendor": None}
            for i, type_name in enumerate(trackers)
        ],
        "topics": ["0/head"],
    }
    hello.update(fields)
    return json.dumps(hello)


HELLO = _hello()


class _Clock:
    """Stands in for the time module: sleep() advances monotonic() at once."""

    def __init__(self, world):
        self.world = world
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        # A floor, so the loop's waits always move the clock on.
        self.now += max(seconds, 1e-6)
        self.world.check()


class _Listener:
    """RemoteSessionListener stand-in: accept() returns the scripted connections when they arrive."""

    def __init__(self, world):
        self.world = world
        self.incoming = []
        self.closed = False
        self.accepts = 0

    def accept(self, timeout_ms):
        assert not self.closed
        self.accepts += 1
        clock = self.world.clock
        due = clock.now + timeout_ms / 1000
        if self.incoming and self.incoming[0][0] <= due:
            at, fd, peer = self.incoming.pop(0)
            clock.now = max(clock.now, at)
            return fd, peer
        clock.sleep(timeout_ms / 1000)
        return None

    def close(self):
        self.closed = True


class _Host:
    """RemoteSessionHost stand-in: a scripted robot app that leaves at ``leave_at``.

    Its trackers must be ended (stop_session() or close()) while the OpenXR session they were
    started on is open; anything else is recorded in ``world.violations``.
    """

    def __init__(self, world, name, hello, on_tick):
        self.world = world
        self.name = name
        self.hello = hello
        self.on_tick = on_tick
        self.hello_error = None
        # The error text of each start_session() call in turn; None, or none left: it works.
        self.start_errors = []
        # The error text of a tick at clock time ``now``; None: it works.
        self.tick_error = lambda now: None
        # Raised by the next stop_session(), which then leaves the trackers running.
        self.stop_error = None
        # Raised by send_error() at clock time ``now``; None: it is sent.
        self.send_error_fails = lambda now: None
        self.leave_at = math.inf
        self.adopted_at = None
        self.started = None
        self.started_at = None
        # The clock time of every start_session() call, failed ones included.
        self.start_tries = []
        # Which OpenXR session the trackers run on, counted from 1; 0 on placeholder handles;
        # None while no trackers run.
        self.session = None
        self.ticks = 0
        self.tick_times = []
        # (kind, payload, clock time)
        self.sent = []
        self.close_reason = ""
        self._alive = True

    @property
    def alive(self):
        if self._alive and self.world.clock.now >= self.leave_at:
            self.leave()
        return self._alive

    def leave(self):
        self._alive = False
        self.close_reason = "receive failed: connection closed"

    def wait_hello(self, timeout_ms):
        if self.hello_error is not None:
            raise self.hello_error
        if not self.alive:
            raise RuntimeError(f"Remote session lost: {self.close_reason}")
        if self.hello is not None:
            return self.hello
        self.world.clock.sleep(timeout_ms / 1000)
        return None

    def _record(self, kind, payload):
        self.sent.append((kind, payload, self.world.clock.now))

    def send_hello_ack(self, text):
        self._record("hello_ack", json.loads(text))

    def send_status(self, text):
        self._record("status", json.loads(text))

    def send_ready(self, text):
        self._record("ready", json.loads(text))

    def send_error(self, text):
        failure = self.send_error_fails(self.world.clock.now)
        if failure is not None:
            raise failure
        self._record("error", text)

    def start_session(self, trackers, handles, vendor_config, topics):
        world = self.world
        self.start_tries.append(world.clock.now)
        world.expect(self.session is None, f"{self.name}: started while trackers run")
        if handles.instance == server_main.PLACEHOLDER_HANDLES[0]:
            session = 0
        else:
            session = world.openxr.opened
            world.expect(
                world.openxr.is_open and handles.space == session,
                f"{self.name}: started on an OpenXR session that is not the open one",
            )
        error = self.start_errors.pop(0) if self.start_errors else None
        if error is not None:
            raise RuntimeError(error)
        self.session = session
        self.started = (trackers, handles, topics)
        self.started_at = world.clock.now

    def tick(self):
        now = self.world.clock.now
        self.world.expect(
            self.session is not None, f"{self.name}: tick without trackers"
        )
        error = self.tick_error(now)
        if error is not None:
            raise RuntimeError(error)
        self.ticks += 1
        self.tick_times.append(now)
        if self.on_tick is not None:
            self.on_tick(self)

    def stop_session(self):
        self.world.events.append(f"{self.name}.stop")
        if self.stop_error is not None:
            error, self.stop_error = self.stop_error, None
            raise error
        self._end_trackers()

    def close(self):
        self._end_trackers()
        self.world.events.append(f"{self.name}.close")
        self._alive = False

    def _end_trackers(self):
        """The trackers call into the OpenXR session they were started on as they end."""
        openxr = self.world.openxr
        if self.session:
            self.world.expect(
                openxr.is_open and openxr.opened == self.session,
                f"{self.name}: trackers ended after their OpenXR session closed",
            )
        self.session = None

    def kinds(self):
        return [kind for kind, _, _ in self.sent]

    def messages(self, kind):
        return [payload for k, payload, _ in self.sent if k == kind]

    def sent_at(self, kind):
        return [at for k, _, at in self.sent if k == kind]

    def statuses(self, start=-math.inf, end=math.inf):
        """The STATUS payloads sent at clock times from ``start`` up to ``end``."""
        return [
            payload
            for kind, payload, at in self.sent
            if kind == "status" and start <= at < end
        ]

    def ticks_between(self, start, end):
        return len([at for at in self.tick_times if start <= at < end])

    def stages(self):
        return [
            (kind, payload["stage"], payload["message"])
            for kind, payload, _ in self.sent
            if kind in ("status", "ready")
        ]


class _OpenXR:
    """oxr.OpenXRSession stand-in: ``failures`` construction errors, then a session whose headset
    follows ``headset(now)``; its provider fails once ``failed(now)``.

    One session is open at a time; ``opened`` and ``closed`` count them. ``open_error(now)`` is
    a construction error text by clock time; ``handles_error``, ``snapshot_error`` and
    ``close_error`` give the exception that call raises at ``now``. None: the call works.
    """

    def __init__(self, world):
        self.world = world
        self.failures = []
        self.open_error = lambda now: None
        self.handles_error = lambda now: None
        self.snapshot_error = lambda now: None
        self.close_error = lambda now: None
        self.headset = lambda now: CONNECTED
        self.failed = lambda now: False
        # (clock time, app name, extensions, wait_for_system)
        self.attempts = []
        self.opened = 0
        self.closed = 0

    @property
    def is_open(self):
        return self.opened > self.closed

    def __call__(self, app_name, extensions, wait_for_system=True):
        now = self.world.clock.now
        self.attempts.append((now, app_name, list(extensions), wait_for_system))
        self.world.expect(not self.is_open, "a second OpenXR session while one is open")
        if self.failures:
            raise RuntimeError(self.failures.pop(0))
        error = self.open_error(now)
        if error is not None:
            raise RuntimeError(error)
        self.opened += 1
        self.world.events.append("openxr.open")
        return self

    def get_handles(self):
        error = self.handles_error(self.world.clock.now)
        if error is not None:
            raise error
        # The space tells the sessions apart.
        return oxr.OpenXRSessionHandles(7, 7, self.opened, 7)

    def get_provider_snapshot(self):
        now = self.world.clock.now
        self.world.expect(self.is_open, "a snapshot of a closed OpenXR session")
        error = self.snapshot_error(now)
        if error is not None:
            raise error
        if self.failed(now):
            return SimpleNamespace(
                state=oxr.OpenXRProviderState.FAILED,
                headset_state=DISCONNECTED,
                reason=oxr.OpenXRProviderReason.SESSION_LOST,
                error="XR_ERROR_SESSION_LOST",
            )
        state = self.headset(now)
        return SimpleNamespace(
            state=oxr.OpenXRProviderState.AVAILABLE,
            headset_state=state,
            reason=oxr.OpenXRProviderReason.NONE
            if state == CONNECTED
            else oxr.OpenXRProviderReason.TRACKING_UNAVAILABLE,
            error="",
        )

    def close(self):
        self.world.expect(self.is_open, "an OpenXR session closed twice")
        self.closed += 1
        self.world.events.append("openxr.close")
        error = self.close_error(self.world.clock.now)
        if error is not None:
            raise error


class _Launcher:
    """CloudXR stand-in whose health check fails from check ``fail_at`` on, and while
    ``down(now)``; ``error(now)`` is any other exception the check raises at ``now``."""

    def __init__(self, world):
        self.world = world
        self.checks = []
        self.fail_at = math.inf
        self.down = lambda now: False
        self.error = lambda now: None

    def health_check(self):
        now = self.world.clock.now
        self.checks.append(now)
        error = self.error(now)
        if error is not None:
            raise error
        if len(self.checks) >= self.fail_at or self.down(now):
            raise RuntimeError(STOPPED)


class _InlineThread:
    """threading.Thread stand-in that runs its target in start(), on the virtual clock."""

    def __init__(self, target, args=(), name=None, daemon=None):
        self._target = target
        self._args = args

    def start(self):
        self._target(*self._args)

    def is_alive(self):
        return False

    def join(self):
        pass


class _World:
    """The server's surroundings; ``stop_at`` is when SIGTERM arrives."""

    def __init__(self, monkeypatch):
        self.clock = _Clock(self)
        self.events = []
        self.stop = server_main._StopRequest()
        self.stop_at = MAX_RUN_S
        self.listener = _Listener(self)
        self.launcher = _Launcher(self)
        self.openxr = _OpenXR(self)
        self.hosts = {}
        self.at_stop = []
        # (clock time, robot app name, text)
        self.refused = []
        # What the stand-ins saw that a real OpenXR runtime does not survive.
        self.violations = []
        self.server = None
        monkeypatch.setattr(_session, "time", self.clock)
        monkeypatch.setattr(server_main, "time", self.clock)
        monkeypatch.setattr(oxr, "OpenXRSession", self.openxr)
        monkeypatch.setattr(
            server_main, "RemoteSessionHost", SimpleNamespace(adopt=self._adopt)
        )
        monkeypatch.setattr(
            server_main, "RemoteSessionListener", SimpleNamespace(reject=self._reject)
        )
        monkeypatch.setattr(
            server_main, "threading", SimpleNamespace(Thread=_InlineThread)
        )

    def connect(self, at, hello=HELLO, on_tick=None):
        """A robot app that connects at clock time ``at``; ``at=None``: as SIGTERM arrives.

        With ``hello=None`` it sends no HELLO.
        """
        fd = 100 + len(self.hosts)
        host = _Host(self, f"robot{len(self.hosts) + 1}", hello, on_tick)
        self.hosts[fd] = host
        if at is None:
            self.at_stop.append(fd)
        else:
            self.listener.incoming.append((at, fd, f"{host.name}:1"))
            self.listener.incoming.sort()
        return host

    def check(self):
        if self.stop.signal is None and self.clock.now >= self.stop_at:
            self.stop.signal = signal.SIGTERM
            for fd in self.at_stop:
                self.listener.incoming.insert(
                    0, (self.clock.now, fd, f"{self.hosts[fd].name}:1")
                )

    def run(self, placeholder=False, headset=True):
        """Run the server to its stop signal; ``headset=False`` is --no-headset."""
        listening = server_main._Listening(self.listener, False, "tcp://test")
        self.server = server_main._Server(
            [listening],
            self.launcher if headset else None,
            self.stop,
            placeholder=placeholder,
            headset=headset,
        )
        code = self.server.run()
        assert self.clock.now < MAX_RUN_S, "the scripted run did not stop"
        assert self.violations == []
        return code

    def expect(self, holds, violation):
        """Record ``violation`` unless ``holds``: the server guards the calls it is raised in."""
        if not holds:
            self.violations.append(f"{violation} (at {self.clock.now:.3f} s)")
            raise AssertionError(violation)

    def refusals(self):
        return [(name, text) for _, name, text in self.refused]

    def _adopt(self, fd, websocket):
        host = self.hosts[fd]
        host.adopted_at = self.clock.now
        return host

    def _reject(self, fd, websocket, text):
        self.refused.append((self.clock.now, self.hosts[fd].name, text))


@pytest.fixture
def world(monkeypatch):
    return _World(monkeypatch)


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.INFO, logger="isaaccapture.teleop_session_server")
    return caplog


def _logged(logs, level):
    """The texts logged at exactly ``level``."""
    return [record.getMessage() for record in logs.records if record.levelno == level]


def _deduplicated(items):
    """``items`` without the repeats that follow one another."""
    return [item for i, item in enumerate(items) if i == 0 or item != items[i - 1]]


def _leave_ticks_after_ready(ticks):
    def on_tick(host):
        ready = host.sent_at("ready")
        if ready and len([t for t in host.tick_times if t >= ready[0]]) > ticks:
            host.leave()

    return on_tick


# ============================================================================
# The OpenXR session
# ============================================================================


def test_opens_openxr_before_any_robot_app_and_retries_while_no_headset(world, logs):
    world.openxr.failures = [NO_HEADSET_ERROR] * 2
    world.stop_at = 5.0

    assert world.run() == 0

    extensions = _session.served_extensions()
    assert "XR_BD_body_tracking" in extensions
    assert [attempt[1:] for attempt in world.openxr.attempts] == [
        ("TeleopSessionServer", extensions, False)
    ] * 3
    times = [attempt[0] for attempt in world.openxr.attempts]
    assert times == pytest.approx([0.0, 1.0, 2.0], abs=1e-3)
    assert world.events == ["openxr.open", "openxr.close"]
    # Waiting is logged once however many attempts it takes.
    assert (
        logs.messages.count("No headset is connected to CloudXR; waiting for one") == 1
    )


def test_one_openxr_session_serves_robot_app_after_robot_app(world):
    first = world.connect(1.0, on_tick=_leave_ticks_after_ready(3))
    second = world.connect(5.0, on_tick=_leave_ticks_after_ready(3))
    world.stop_at = 10.0

    assert world.run() == 0

    assert len(world.openxr.attempts) == 1
    for robot in (first, second):
        assert robot.kinds().count("ready") == 1
        _, handles, _ = robot.started
        assert (handles.instance, handles.session) == (7, 7)
    assert world.events == [
        "openxr.open",
        "robot1.close",
        "robot2.close",
        "openxr.close",
    ]


def test_a_robot_app_waits_for_the_headset_to_connect_to_cloudxr(world):
    world.openxr.failures = [NO_HEADSET_ERROR] * 3
    world.openxr.headset = lambda now: CONNECTED if now >= 4.0 else DISCONNECTED
    robot = world.connect(0.5, on_tick=_leave_ticks_after_ready(3))
    world.stop_at = 10.0

    assert world.run() == 0

    assert robot.kinds()[0] == "hello_ack"
    assert robot.messages("hello_ack")[0]["server_version"] == isaaccapture.__version__
    stages = robot.stages()
    # Reported in this order, connecting_openxr once however long it lasts.
    assert stages[:2] == [
        ("status", "connecting_openxr", WAITING),
        ("status", "waiting_for_headset", WAITING),
    ]
    assert ("status", "waiting_for_headset", "headset connected") in stages
    # The OpenXR session opens on its fourth attempt, at 3 s; the trackers start then.
    assert robot.started_at == pytest.approx(3.0, abs=0.2)
    trackers, _, topics = robot.started
    assert [(type(t), name) for t, name in trackers] == [
        (HeadTracker, "0"),
        (HandTracker, "1"),
    ]
    assert topics == ["0/head"]
    [ready] = robot.messages("ready")
    assert ready["headset"] == CONNECTED_ENTRY
    [ready_at] = robot.sent_at("ready")
    assert 4.0 + _session.READY_AFTER_CONNECTED_S <= ready_at


def test_ready_needs_the_headset_connected_without_a_break(world):
    world.openxr.headset = lambda now: DISCONNECTED if 2.0 <= now < 2.3 else CONNECTED
    robot = world.connect(1.0, on_tick=_leave_ticks_after_ready(1))
    world.stop_at = 6.0

    assert world.run() == 0

    [ready_at] = robot.sent_at("ready")
    back = min(t for t in robot.tick_times if t >= 2.3)
    connected_for = ready_at - back
    # The specified minimum: connected continuously for at least 1.1 s.
    assert connected_for >= 1.1
    assert connected_for < _session.READY_AFTER_CONNECTED_S + 2 * TICK_S


def test_a_robot_app_leaving_while_no_headset_leaves_the_server_waiting(world):
    world.openxr.failures = [NO_HEADSET_ERROR] * 100
    robot = world.connect(0.5)
    robot.leave_at = 2.5
    world.stop_at = 6.0

    assert world.run() == 0

    assert robot.started is None
    assert "error" not in robot.kinds()
    assert world.events == ["robot1.close"]
    times = [attempt[0] for attempt in world.openxr.attempts]
    assert times[-1] >= 5.0
    assert all(b - a == pytest.approx(1.0, abs=1e-3) for a, b in zip(times, times[1:]))


def test_running_status_is_periodic_and_follows_headset_changes(world):
    world.openxr.headset = lambda now: DISCONNECTED if 5.0 <= now < 5.5 else CONNECTED
    robot = world.connect(0.5)
    robot.leave_at = 9.0
    world.stop_at = 10.0

    assert world.run() == 0

    running = [
        (payload["headset"]["headset_state"], at)
        for kind, payload, at in robot.sent
        if kind in ("ready", "status") and payload["stage"] in ("ready", "running")
    ]
    assert running[0][0] == "CONNECTED"
    # A change goes out on the tick that sees it.
    changes = [
        at for (now, at), (before, _) in zip(running[1:], running) if now != before
    ]
    assert changes == [
        pytest.approx(5.0, abs=TICK_S + 1e-3),
        pytest.approx(5.5, abs=TICK_S + 1e-3),
    ]
    # Otherwise once per STATUS_PERIOD_S.
    period = _session.STATUS_PERIOD_S
    for (now, at), (before, before_at) in zip(running[1:], running):
        if now == before:
            assert period <= at - before_at < period + 2 * TICK_S


def test_placeholder_mode_is_ready_at_once_without_openxr(world):
    robot = world.connect(
        0.5,
        hello=_hello(trackers=["KeyboardTracker"]),
        on_tick=_leave_ticks_after_ready(2),
    )
    world.stop_at = 2.0

    assert world.run(placeholder=True) == 0

    assert world.openxr.attempts == []
    assert robot.kinds()[:2] == ["hello_ack", "ready"]
    assert robot.messages("ready")[0]["headset"] == _session.UNTRACKED_HEADSET
    _, handles, _ = robot.started
    assert (handles.instance, handles.session) == (1, 1)
    assert world.events == ["robot1.close"]


# ============================================================================
# A headset side that cannot be used: the server keeps serving
# ============================================================================


def test_no_openxr_call_while_cloudxr_is_down(world, logs):
    world.launcher.down = lambda now: True
    robot = world.connect(1.0)
    world.stop_at = 60.0

    assert world.run() == 0

    assert world.openxr.attempts == []
    assert robot.kinds() == ["hello_ack", "status", "error"]
    assert robot.statuses() == [
        {"stage": "connecting_openxr", "message": STOPPED, "headset": NO_ENTRY}
    ]
    assert robot.messages("error") == [SHUTDOWN]
    assert _logged(logs, logging.WARNING) == [_warning(STOPPED, start_hint=True)]
    checks = world.launcher.checks
    assert len(checks) == pytest.approx(60, abs=1)
    assert all(
        b - a == pytest.approx(1.0, abs=TICK_S) for a, b in zip(checks, checks[1:])
    )
    assert world.events == ["robot1.close"]
    # The loop waits between its checks.
    assert world.listener.accepts < 60 * 25


def test_a_cloudxr_that_is_started_later_is_attached_and_served_without_a_restart(
    world, logs, monkeypatch
):
    """The server's own CloudXR class in the loop: nothing could be started at the start."""
    environment = {
        "XR_RUNTIME_JSON": "/x/openxr_cloudxr.json",
        "NV_CXR_RUNTIME_DIR": "/x/run",
    }
    for key in environment:
        monkeypatch.setenv(key, "before")
    read_at = []

    def is_runtime_live(run_dir):
        # A person starts CloudXR at 10 s, stops it at 20 s and starts it again at 30 s.
        return 10.0 <= world.clock.now < 20.0 or world.clock.now >= 30.0

    def read_exported_env(path):
        read_at.append(world.clock.now)
        return dict(environment)

    monkeypatch.setattr(_cloudxr, "is_runtime_live", is_runtime_live)
    monkeypatch.setattr(_cloudxr, "read_exported_env", read_exported_env)
    missing = "CloudXR is not available: no CloudXR SDK"
    stopped = "The CloudXR runtime serving /x/run has stopped"
    world.launcher = _cloudxr.CloudXR(None, "/x/run", missing)
    robot = world.connect(1.0)
    world.stop_at = 40.0

    assert world.run() == 0

    # Its environment file is read each time it is seen again, within a second.
    assert read_at == pytest.approx([10.0, 30.0], abs=2 * TICK_S)
    assert {key: os.environ[key] for key in environment} == environment
    times = [attempt[0] for attempt in world.openxr.attempts]
    assert times == read_at
    assert world.events == [
        "openxr.open",
        "robot1.stop",
        "openxr.close",
        "openxr.open",
        "robot1.close",
        "openxr.close",
    ]
    # The command names the folder this server looks at, which is not the service's default.
    folder = " --cloudxr-install-dir /x"
    assert _logged(logs, logging.WARNING) == [
        _warning(missing, start_hint=True) + folder,
        _warning(stopped, start_hint=True) + folder,
    ]
    returned = [
        "Attached to the CloudXR runtime serving /x/run",
        "CloudXR is available again",
        "Opened the OpenXR session",
    ]
    said = [text for text in logs.messages if text in returned]
    assert said == returned * 2
    assert robot.statuses(0.0, 10.0) == [
        {"stage": "connecting_openxr", "message": missing, "headset": NO_ENTRY}
    ]
    [ready_at] = robot.sent_at("ready")
    assert ready_at == pytest.approx(10.0 + _session.READY_AFTER_CONNECTED_S, abs=0.1)
    outage = robot.statuses(20.0 + 2 * TICK_S, 30.0)
    assert outage and all(
        status == {"stage": "running", "message": stopped, "headset": NOT_TRACKED_ENTRY}
        for status in outage
    )
    assert robot.ticks_between(30.5, 40.0) > 500
    assert robot.messages("error") == [SHUTDOWN]


def test_other_openxr_errors_are_retried_with_a_growing_pause(world, logs):
    world.openxr.failures = [NO_HEADSET_ERROR]
    world.openxr.open_error = lambda now: "Failed to create OpenXR instance: -51"
    robot = world.connect(0.5)
    world.stop_at = 300.0

    assert world.run() == 0

    times = [attempt[0] for attempt in world.openxr.attempts]
    # The try after "no headset" comes OPENXR_RETRY_S later; the others wait longer each time.
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert gaps == pytest.approx(
        [1.0, 5.0, 10.0, 20.0, 40.0, 60.0, 60.0, 60.0], abs=0.01
    )
    reason = OPEN_FAILED + "Failed to create OpenXR instance: -51"
    assert _logged(logs, logging.WARNING) == [_warning(reason)]
    assert robot.statuses() == [
        {"stage": "connecting_openxr", "message": WAITING, "headset": NO_ENTRY},
        {"stage": "connecting_openxr", "message": reason, "headset": NO_ENTRY},
    ]
    assert "ready" not in robot.kinds()
    assert robot.started is None
    assert robot.messages("error") == [SHUTDOWN]
    assert world.events == ["robot1.close"]


@pytest.mark.parametrize("with_robot", [True, False], ids=["robot", "idle"])
def test_a_failed_openxr_session_is_replaced_and_the_robot_app_kept(
    world, logs, with_robot
):
    world.openxr.failed = _between(3.0, 3.5)
    robot = world.connect(1.0) if with_robot else None
    world.stop_at = 20.0

    assert world.run() == 0

    assert _logged(logs, logging.WARNING) == [_warning(SESSION_LOST)]
    assert logs.messages.count("Opened the OpenXR session") == 2
    times = [attempt[0] for attempt in world.openxr.attempts]
    # Seen by the tick, or by the health check a robot app does not need.
    assert times == pytest.approx([0.0, 3.0 + _common.RETRY_S], abs=2 * TICK_S)
    if robot is None:
        assert world.events == [
            "openxr.open",
            "openxr.close",
            "openxr.open",
            "openxr.close",
        ]
        return
    # The trackers read through the OpenXR session, so they stop first.
    assert world.events == [
        "openxr.open",
        "robot1.stop",
        "openxr.close",
        "openxr.open",
        "robot1.close",
        "openxr.close",
    ]
    assert robot.kinds().count("ready") == 1
    assert robot.messages("error") == [SHUTDOWN]
    lost_at, back_at = times[1] - _common.RETRY_S, times[1]
    assert robot.ticks_between(lost_at + TICK_S, back_at) == 0
    during = robot.statuses(lost_at + TICK_S, back_at)
    assert during and all(
        status
        == {"stage": "running", "message": SESSION_LOST, "headset": NOT_TRACKED_ENTRY}
        for status in during
    )
    after = robot.statuses(back_at + 2 * TICK_S)
    assert after and all(
        status == {"stage": "running", "message": "", "headset": CONNECTED_ENTRY}
        for status in after
    )
    assert robot.ticks_between(back_at, 20.0) > 1000
    assert len(robot.start_tries) == 2


def test_cloudxr_stopping_and_returning_keeps_the_robot_app(world, logs):
    world.launcher.down = _between(10.0, 25.0)
    robot = world.connect(1.0)
    world.stop_at = 40.0

    assert world.run() == 0

    # The trackers stop before the OpenXR session they read through closes.
    assert world.events == [
        "openxr.open",
        "robot1.stop",
        "openxr.close",
        "openxr.open",
        "robot1.close",
        "openxr.close",
    ]
    assert robot.kinds().count("hello_ack") == 1
    assert robot.kinds().count("ready") == 1
    # Nothing ends the connection before the stop signal.
    assert robot.messages("error") == [SHUTDOWN]
    assert robot.sent[-1][0] == "error"
    checks = world.launcher.checks
    assert all(
        b - a == pytest.approx(1.0, abs=TICK_S) for a, b in zip(checks, checks[1:])
    )
    # No tick and no OpenXR call while CloudXR is down; a new session as soon as it is back.
    assert robot.ticks_between(10.0 + 2 * TICK_S, 25.0) == 0
    times = [attempt[0] for attempt in world.openxr.attempts]
    assert times == pytest.approx([0.0, 25.0], abs=2 * TICK_S)
    assert robot.start_tries == pytest.approx([1.0, 25.0], abs=0.2)
    assert robot.ticks_between(25.5, 40.0) > 1000

    outage = {"stage": "running", "message": STOPPED, "headset": NOT_TRACKED_ENTRY}
    sent_at = [
        at for kind, payload, at in robot.sent if kind == "status" and payload == outage
    ]
    # Sent as soon as the trackers stop, and about once a second while it lasts.
    assert sent_at[0] == pytest.approx(10.0, abs=2 * TICK_S)
    assert robot.statuses(sent_at[0], 25.0) == [outage] * len(sent_at)
    period = _session.STATUS_PERIOD_S
    assert all(
        period <= b - a <= period + _session.WAIT_SLICE_S + 1e-6
        for a, b in zip(sent_at, sent_at[1:])
    )
    assert len(sent_at) >= 13
    after = robot.statuses(25.5)
    assert after and all(
        status == {"stage": "running", "message": "", "headset": CONNECTED_ENTRY}
        for status in after
    )

    assert _logged(logs, logging.WARNING) == [_warning(STOPPED, start_hint=True)]
    assert not [r for r in logs.records if r.levelno >= logging.ERROR]
    assert logs.messages.count("CloudXR is available again") == 1
    assert logs.messages.count("Opened the OpenXR session") == 2
    assert logs.messages.count("Ready; streaming to the robot app at robot1:1") == 1


@pytest.mark.parametrize("with_robot", [True, False], ids=["robot", "idle"])
def test_a_cloudxr_that_stops_again_soon_is_warned_about_each_time(
    world, logs, with_robot
):
    # The second stop is 5 s after the first and the third 9 s after the second: each within
    # the time in which a failure of the OpenXR session counts as the same one repeating.
    outages = [(10.0, 12.0), (15.0, 17.0), (24.0, 26.0)]
    world.launcher.down = lambda now: any(start <= now < end for start, end in outages)
    robot = world.connect(1.0) if with_robot else None
    world.stop_at = 35.0

    assert world.run() == 0

    stopped = _warning(STOPPED, start_hint=True)
    back = "CloudXR is available again"
    # Every stop is the log's last word about CloudXR until CloudXR returns.
    assert [text for text in logs.messages if text in (stopped, back)] == [
        stopped,
        back,
    ] * 3
    assert _logged(logs, logging.WARNING) == [stopped] * 3
    assert not [r for r in logs.records if r.levelno >= logging.ERROR]
    # A new OpenXR session as soon as CloudXR is back, whatever the retry delay.
    times = [attempt[0] for attempt in world.openxr.attempts]
    assert times == pytest.approx([0.0, 12.0, 17.0, 26.0], abs=2 * TICK_S)
    assert world.events.count("openxr.close") == 4
    if robot is not None:
        assert world.events.count("robot1.stop") == 3
        assert robot.kinds().count("ready") == 1
        assert robot.ticks_between(26.5, 35.0) > 500
        assert robot.messages("error") == [SHUTDOWN]


def test_before_ready_an_outage_reads_connecting_openxr_and_ready_waits_for_a_steady_headset(
    world,
):
    # CloudXR stops before the headset has stayed connected long enough for READY.
    world.launcher.down = _between(1.8, 8.0)
    robot = world.connect(1.0)
    world.stop_at = 20.0

    assert world.run() == 0

    assert _deduplicated(robot.stages())[:5] == [
        ("status", "waiting_for_headset", "headset connected"),
        ("status", "connecting_openxr", STOPPED),
        ("status", "waiting_for_headset", "headset connected"),
        ("ready", "ready", ""),
        ("status", "running", ""),
    ]
    outage = [s for s in robot.statuses() if s["message"] == STOPPED]
    assert outage == [
        {"stage": "connecting_openxr", "message": STOPPED, "headset": NO_ENTRY}
    ]
    [ready_at] = robot.sent_at("ready")
    back_at = robot.start_tries[1]
    assert back_at == pytest.approx(8.0, abs=0.2)
    # Counted on the new OpenXR session, not from before the outage.
    waited_s = ready_at - back_at
    assert waited_s >= _session.READY_AFTER_CONNECTED_S
    assert waited_s < _session.READY_AFTER_CONNECTED_S + 2 * TICK_S


def test_a_robot_app_that_leaves_during_an_outage_frees_the_server(world):
    world.launcher.down = _between(5.0, 15.0)
    first = world.connect(1.0)
    first.leave_at = 8.0
    second = world.connect(9.0)
    world.stop_at = 25.0

    assert world.run() == 0

    assert "error" not in first.kinds()
    assert world.refused == []
    assert second.adopted_at == pytest.approx(9.0, abs=0.2)
    assert second.statuses()[0] == {
        "stage": "connecting_openxr",
        "message": STOPPED,
        "headset": NO_ENTRY,
    }
    assert second.kinds().count("ready") == 1
    assert second.sent_at("ready")[0] >= 15.0 + _session.READY_AFTER_CONNECTED_S
    assert world.events == [
        "openxr.open",
        "robot1.stop",
        "openxr.close",
        "robot1.close",
        "openxr.open",
        "robot2.close",
        "openxr.close",
    ]


def test_a_failure_that_repeats_is_logged_once_and_retried_less_often(world, logs):
    robot = world.connect(1.0)
    robot.tick_error = lambda now: "schema mismatch" if 10.0 <= now < 40.0 else None
    world.stop_at = 70.0

    assert world.run() == 0

    reason = UPDATE_FAILED + "schema mismatch"
    # A new OpenXR session each time, since a failed tick ends the old trackers for good.
    times = [attempt[0] for attempt in world.openxr.attempts]
    assert times == pytest.approx([0.0, 15.0, 25.0, 45.0], abs=0.1)
    assert world.events.count("robot1.stop") == 3
    assert world.events.count("openxr.close") == 4
    assert _logged(logs, logging.WARNING) == [_warning(reason)]
    assert not [r for r in logs.records if r.levelno >= logging.ERROR]
    during = robot.statuses(10.1, 45.0)
    assert during and all(
        status == {"stage": "running", "message": reason, "headset": NOT_TRACKED_ENTRY}
        for status in during
    )
    assert robot.ticks_between(10.0, 45.0) == 0
    assert robot.ticks_between(45.0, 70.0) > 1000
    assert robot.kinds().count("ready") == 1
    assert robot.messages("error") == [SHUTDOWN]


def test_a_tick_failure_names_cloudxr_when_cloudxr_stopped(world, logs):
    # The tick fails between two health checks, which are a second apart.
    world.launcher.down = _between(10.4, 20.0)
    robot = world.connect(1.0)
    robot.tick_error = lambda now: (
        "[HeadTracker] xrLocateSpace failed: -13" if 10.4 <= now < 20.0 else None
    )
    world.stop_at = 30.0

    assert world.run() == 0

    assert _logged(logs, logging.WARNING) == [_warning(STOPPED, start_hint=True)]
    assert not [r for r in logs.records if r.levelno >= logging.ERROR]
    assert world.events[:3] == ["openxr.open", "robot1.stop", "openxr.close"]
    assert robot.ticks_between(10.4, 20.0) == 0
    # CloudXR was asked at once, not at the next periodic check.
    last_before = max(at for at in robot.tick_times if at < 10.4)
    asked_at = min(at for at in world.launcher.checks if at > last_before)
    assert asked_at == pytest.approx(10.4, abs=2 * TICK_S)
    during = robot.statuses(10.5, 20.0)
    assert during and all(
        status == {"stage": "running", "message": STOPPED, "headset": NOT_TRACKED_ENTRY}
        for status in during
    )
    assert robot.kinds().count("ready") == 1
    assert robot.ticks_between(21.0, 30.0) > 500
    assert robot.messages("error") == [SHUTDOWN]


@pytest.mark.parametrize("call", ["snapshot", "close", "health check", "handles"])
def test_a_raising_headset_call_does_not_stop_the_server(world, logs, call):
    robot = world.connect(1.0)
    if call == "snapshot":
        world.openxr.snapshot_error = lambda now: (
            ValueError("snapshot blew up") if 10.0 <= now < 10.5 else None
        )
        warnings = [
            _warning(
                "The OpenXR session on the Teleop Session Server failed: snapshot blew up"
            )
        ]
    elif call == "close":
        world.openxr.failed = _between(10.0, 10.5)
        world.openxr.close_error = lambda now: (
            RuntimeError("the runtime is gone") if now < 12.0 else None
        )
        warnings = [
            "Closing the OpenXR session failed: the runtime is gone",
            _warning(SESSION_LOST),
        ]
    elif call == "health check":
        world.launcher.error = lambda now: (
            OSError("probe failed") if 10.0 <= now < 12.0 else None
        )
        warnings = [_warning("probe failed", start_hint=True)]
    else:
        world.openxr.handles_error = lambda now: (
            TypeError("no handles") if now < 1.0 else None
        )
        # The session that gave no handles is closed again.
        warnings = [_warning(OPEN_FAILED + "no handles")]
    world.stop_at = 30.0

    assert world.run() == 0

    assert _logged(logs, logging.WARNING) == warnings
    assert not [r for r in logs.records if r.levelno >= logging.ERROR]
    # One OpenXR session was given up for a new one.
    assert world.events.count("openxr.open") == 2
    assert world.events.count("openxr.close") == 2
    assert robot.kinds().count("ready") == 1
    assert robot.messages("error") == [SHUTDOWN]
    assert robot.ticks_between(20.0, 30.0) > 500


@pytest.mark.parametrize("seen_by", ["the health check", "a tick"])
def test_trackers_that_cannot_be_stopped_end_that_connection_before_openxr_closes(
    world, logs, seen_by
):
    if seen_by == "the health check":
        world.launcher.down = _between(10.0, 60.0)
    else:
        # Between two health checks, so a tick finds the failed OpenXR session.
        world.openxr.failed = _between(10.005, 10.5)
    robot = world.connect(1.0)
    robot.stop_error = RuntimeError("stop failed")
    later = world.connect(17.0)
    world.stop_at = 25.0

    assert world.run() == 0

    # Closing the connection ends its trackers, before the OpenXR session closes.
    assert world.events[:4] == [
        "openxr.open",
        "robot1.stop",
        "robot1.close",
        "openxr.close",
    ]
    assert world.events.count("robot1.close") == 1
    assert robot.sent[-1][:2] == (
        "error",
        "The Teleop Session Server failed: RuntimeError: stop failed",
    )
    [record] = [r for r in logs.records if r.levelno >= logging.ERROR]
    assert record.getMessage() == (
        "Robot app at robot1:1: the trackers could not be stopped"
    )
    assert record.exc_info is not None
    # The server serves the next robot app.
    assert later.adopted_at == pytest.approx(17.0, abs=0.2)
    assert later.kinds()[0] == "hello_ack"
    if seen_by == "a tick":
        assert "ready" in later.kinds()
        assert world.events[4:] == ["openxr.open", "robot2.close", "openxr.close"]
    else:
        assert world.events[4:] == ["robot2.close"]


def test_trackers_that_cannot_be_stopped_on_a_dead_link_do_not_stop_the_server(
    world, logs
):
    world.launcher.down = _between(10.0, 60.0)
    robot = world.connect(1.0)
    robot.stop_error = RuntimeError("stop failed")
    # The ERROR about it cannot be sent either.
    robot.send_error_fails = lambda now: OSError("the link is gone")
    world.stop_at = 15.0

    assert world.run() == 0

    assert world.events == [
        "openxr.open",
        "robot1.stop",
        "robot1.close",
        "openxr.close",
    ]
    assert "error" not in robot.kinds()
    errors = [r.getMessage() for r in logs.records if r.levelno >= logging.ERROR]
    assert set(errors) == {"Robot app at robot1:1: the trackers could not be stopped"}
    assert _logged(logs, logging.WARNING) == [_warning(STOPPED, start_hint=True)]


# ============================================================================
# The hidden test option, and a server without a headset
# ============================================================================


def test_placeholder_mode_keeps_its_cloudxr_check_and_its_trackers_when_cloudxr_stops(
    world, logs
):
    world.launcher.down = _between(3.0, 6.0)
    robot = world.connect(1.0, hello=_hello(trackers=["KeyboardTracker"]))
    world.stop_at = 10.0

    assert world.run(placeholder=True) == 0

    checks = world.launcher.checks
    assert len(checks) == pytest.approx(10, abs=1)
    assert all(
        b - a == pytest.approx(1.0, abs=TICK_S) for a, b in zip(checks, checks[1:])
    )
    assert world.openxr.attempts == []
    assert robot.kinds()[:2] == ["hello_ack", "ready"]
    # Its trackers read through no OpenXR session, so they tick on.
    assert robot.ticks_between(3.0, 6.0) == pytest.approx(3.0 * TICK_RATE_HZ, abs=3)
    assert robot.messages("error") == [SHUTDOWN]
    assert _logged(logs, logging.WARNING) == [
        f"{STOPPED}; --placeholder-openxr serves on without CloudXR"
    ]
    assert world.events == ["robot1.close"]


def test_placeholder_mode_starts_its_trackers_again_after_a_tick_failure(world, logs):
    robot = world.connect(1.0, hello=_hello(trackers=["KeyboardTracker"]))
    robot.tick_error = lambda now: "keyboard gone" if 3.0 <= now < 3.5 else None
    world.stop_at = 12.0

    assert world.run(placeholder=True) == 0

    reason = UPDATE_FAILED + "keyboard gone"
    assert [r.getMessage() for r in logs.records if r.levelno >= logging.ERROR] == [
        f"Robot app at robot1:1: {reason}. {STAYS_CONNECTED}"
    ]
    assert world.events == ["robot1.stop", "robot1.close"]
    assert world.openxr.attempts == []
    assert robot.start_tries == pytest.approx([1.0, 3.0 + _common.RETRY_S], abs=0.2)
    during = robot.statuses(3.0 + TICK_S, 8.0)
    assert during and all(
        status == {"stage": "running", "message": reason, "headset": NOT_TRACKED_ENTRY}
        for status in during
    )
    assert robot.ticks_between(3.0, 8.0) == 0
    assert robot.ticks_between(8.2, 12.0) > 300
    assert robot.kinds().count("ready") == 1
    assert robot.messages("error") == [SHUTDOWN]
    assert "The trackers of the robot app at robot1:1 run again" in logs.messages


def test_no_headset_mode_never_calls_openxr_or_cloudxr(world, logs):
    robot = world.connect(1.0)
    robot.leave_at = 15.0
    refused = world.connect(20.0, hello=_hello(trackers=["KeyboardTracker"]))
    world.stop_at = 40.0

    assert world.run(headset=False) == 0

    assert world.openxr.attempts == []
    assert world.launcher.checks == []
    assert robot.kinds() == ["hello_ack", "status"]
    assert robot.statuses() == [
        {
            "stage": "connecting_openxr",
            "message": "this Teleop Session Server runs without a headset (--no-headset)",
            "headset": NO_ENTRY,
        }
    ]
    assert robot.started is None
    # HELLO is checked as with a headset.
    refusal = (
        "KeyboardTracker (tracker '0') is not supported over a remote session: it would "
        "read the Teleop Session Server's keyboard"
    )
    assert refused.sent == [("error", refusal, refused.sent[0][2])]
    assert _logged(logs, logging.WARNING) == [
        f"Refused the robot app at robot2:1: {refusal}"
    ]
    assert world.events == ["robot1.close", "robot2.close"]
    # The loop waits between its checks.
    assert world.listener.accepts < 40 * 25


# ============================================================================
# Stopping
# ============================================================================


def test_a_stop_signal_tells_every_robot_app_and_closes_openxr_last(world, logs):
    robot = world.connect(0.5)
    world.connect(2.0)
    world.connect(None)
    world.stop_at = 3.0

    assert world.run() == 0

    assert robot.sent[-1][:2] == ("error", SHUTDOWN)
    # The one waiting for robot1, then the one in the listen backlog.
    assert world.refusals() == [("robot2", SHUTDOWN), ("robot3", SHUTDOWN)]
    assert world.listener.closed
    assert world.events == ["openxr.open", "robot1.close", "openxr.close"]
    assert "Received SIGTERM; stopping the Teleop Session Server" in logs.messages


def test_a_failing_accept_at_stop_still_closes_the_robot_app_before_openxr(
    world, monkeypatch
):
    robot = world.connect(0.5)
    world.stop_at = 2.0
    accept = world.listener.accept

    def accept_failing_at_stop(timeout_ms):
        if world.stop.signal is not None:
            raise RuntimeError("accept failed: Too many open files")
        return accept(timeout_ms)

    monkeypatch.setattr(world.listener, "accept", accept_failing_at_stop)

    with pytest.raises(RuntimeError, match="Too many open files"):
        world.run()

    assert robot.sent[-1][:2] == ("error", SHUTDOWN)
    assert world.events == ["openxr.open", "robot1.close", "openxr.close"]


# ============================================================================
# Trackers that fail: the robot app stays connected
# ============================================================================


def test_a_tick_failure_replaces_the_openxr_session_and_keeps_the_robot_app(
    world, logs
):
    def fail_at_third(host):
        if host.ticks == 3:
            raise RuntimeError("schema mismatch")

    robot = world.connect(0.5, on_tick=fail_at_third)
    world.stop_at = 12.0

    assert world.run() == 0

    reason = UPDATE_FAILED + "schema mismatch"
    assert _logged(logs, logging.WARNING) == [_warning(reason)]
    assert not [r for r in logs.records if r.levelno >= logging.ERROR]
    # A tracker update that fails is final for its trackers and their OpenXR session.
    assert world.events == [
        "openxr.open",
        "robot1.stop",
        "openxr.close",
        "openxr.open",
        "robot1.close",
        "openxr.close",
    ]
    failed_at = robot.tick_times[2]
    times = [attempt[0] for attempt in world.openxr.attempts]
    assert times == pytest.approx([0.0, failed_at + _common.RETRY_S], abs=2 * TICK_S)
    assert robot.ticks_between(failed_at + TICK_S / 2, times[1]) == 0
    # It failed before READY, so the robot app reads a server that is not ready yet.
    assert robot.statuses(failed_at, times[1]) == [
        {"stage": "connecting_openxr", "message": reason, "headset": NO_ENTRY}
    ]
    [ready_at] = robot.sent_at("ready")
    assert ready_at >= times[1] + _session.READY_AFTER_CONNECTED_S
    assert robot.messages("error") == [SHUTDOWN]
    assert "The trackers of the robot app at robot1:1 run again" in logs.messages


def test_a_tracker_start_failure_keeps_the_robot_app_and_is_retried(world, logs):
    # A headset that arrives late, with a tracker the server cannot start on it.
    world.openxr.failures = [NO_HEADSET_ERROR] * 2
    robot = world.connect(0.5)
    robot.start_errors = ["no live vendor 'body.x'"] * 3
    world.stop_at = 60.0

    assert world.run() == 0

    reason = START_FAILED + "no live vendor 'body.x'"
    # Said once however often the start fails.
    assert [r.getMessage() for r in logs.records if r.levelno >= logging.ERROR] == [
        f"Robot app at robot1:1: {reason}. {STAYS_CONNECTED}"
    ]
    assert _logged(logs, logging.WARNING) == []
    # The OpenXR session is kept; only the trackers are started again.
    assert world.events == ["openxr.open", "robot1.close", "openxr.close"]
    assert len(world.openxr.attempts) == 3
    gaps = [b - a for a, b in zip(robot.start_tries, robot.start_tries[1:])]
    assert robot.start_tries[0] == pytest.approx(2.0, abs=0.2)
    assert gaps == pytest.approx([5.0, 10.0, 20.0], abs=2 * _session.WAIT_SLICE_S)
    assert robot.statuses(0.0, robot.start_tries[-1]) == [
        {"stage": "connecting_openxr", "message": WAITING, "headset": NO_ENTRY},
        {"stage": "connecting_openxr", "message": reason, "headset": NO_ENTRY},
    ]
    assert robot.ticks_between(0.0, robot.start_tries[-1]) == 0
    [ready_at] = robot.sent_at("ready")
    assert ready_at >= robot.start_tries[-1] + _session.READY_AFTER_CONNECTED_S
    assert robot.messages("error") == [SHUTDOWN]
    assert "The trackers of the robot app at robot1:1 run again" in logs.messages


# ============================================================================
# Robot apps that are refused or dropped
# ============================================================================


def test_a_hello_that_requests_plugins_is_refused(world):
    plugin = {"plugin_name": "p", "plugin_root_id": "p", "plugin_args": []}
    robot = world.connect(0.5, hello=_hello(plugins=[plugin]))
    world.stop_at = 2.0

    assert world.run() == 0

    assert robot.kinds() == ["error"]
    assert robot.sent[0][1].startswith("The robot app requested plugins ['p'], but ")


def test_a_hello_that_is_not_utf8_is_refused(world, logs):
    robot = world.connect(0.5)
    # What the binding raises when it decodes the HELLO payload.
    robot.hello_error = UnicodeDecodeError(
        "utf-8", b'{"app_name": "\xff"}', 14, 15, "invalid start byte"
    )
    world.stop_at = 2.0

    assert world.run() == 0

    assert [payload for _, payload, _ in robot.sent] == [
        "HELLO is not valid UTF-8: 'utf-8' codec can't decode byte 0xff in "
        "position 14: invalid start byte"
    ]
    assert not [r for r in logs.records if r.levelno >= logging.ERROR]


def test_an_unexpected_failure_is_sent_as_error_and_the_server_keeps_serving(
    world, logs
):
    first = world.connect(0.5)
    first.hello_error = OSError("bad file descriptor")
    second = world.connect(1.0, on_tick=_leave_ticks_after_ready(1))
    world.stop_at = 4.0

    assert world.run() == 0

    assert [payload for _, payload, _ in first.sent] == [
        "The Teleop Session Server failed: OSError: bad file descriptor"
    ]
    [record] = [r for r in logs.records if r.levelno >= logging.ERROR]
    assert record.getMessage() == "Serving the robot app at robot1:1 failed"
    assert record.exc_info is not None
    assert "ready" in second.kinds()


def test_a_robot_app_that_sends_no_hello_is_dropped(world, logs):
    silent = world.connect(0.5, hello=None)
    later = world.connect(8.0, on_tick=_leave_ticks_after_ready(1))
    world.stop_at = 12.0

    assert world.run() == 0

    assert silent.sent == []
    assert "Robot app at robot1:1 sent no HELLO within 5 s" in logs.messages
    assert world.events[:2] == ["openxr.open", "robot1.close"]
    assert "ready" in later.kinds()


def test_a_robot_app_that_leaves_before_hello_frees_the_server(world, logs):
    gone = world.connect(0.5, hello=None)
    gone.leave_at = 1.0
    later = world.connect(2.0, on_tick=_leave_ticks_after_ready(1))
    world.stop_at = 5.0

    assert world.run() == 0

    assert gone.sent == []
    assert any(
        m.startswith("Robot app at robot1:1 left before HELLO: ") for m in logs.messages
    )
    assert "ready" in later.kinds()


# ============================================================================
# One robot app at a time
# ============================================================================


def test_a_robot_app_connecting_while_one_is_served_waits_for_it_to_leave(world, logs):
    first = world.connect(0.5)
    first.leave_at = 4.0
    second = world.connect(2.0, on_tick=_leave_ticks_after_ready(1))
    world.stop_at = 8.0

    assert world.run() == 0

    assert world.refused == []
    assert second.adopted_at == pytest.approx(4.0, abs=2 * TICK_S)
    assert "ready" in second.kinds()
    # The first kept ticking while the second waited.
    waited = [t for t in first.tick_times if 2.0 <= t < 4.0]
    assert len(waited) == pytest.approx(2.0 * TICK_RATE_HZ, abs=2)
    assert (
        "Robot app at robot2:1 waits up to 4 s for the one at robot1:1 to leave"
        in logs.messages
    )


def test_a_robot_app_that_left_between_its_ticks_frees_the_server_at_the_grace(world):
    # At 1 Hz robot1 ticks at 0.5, 1.5, ...: it leaves at 5.7, before its tick at 6.5.
    first = world.connect(0.5, hello=_hello(tick_rate_hz=1.0))
    first.leave_at = 5.7
    second = world.connect(2.0, on_tick=_leave_ticks_after_ready(1))
    world.stop_at = 10.0

    assert world.run() == 0

    assert world.refused == []
    assert second.adopted_at == pytest.approx(
        2.0 + server_main.PENDING_GRACE_S, abs=TICK_S
    )
    assert "ready" in second.kinds()


def test_a_robot_app_still_waiting_after_the_grace_is_refused_busy(world, logs):
    first = world.connect(0.5)
    second = world.connect(2.0)
    third = world.connect(3.0)
    world.stop_at = 8.0

    assert world.run() == 0

    # The third at once, as one already waits; the second once its grace ran out.
    assert world.refusals() == [("robot3", BUSY), ("robot2", BUSY)]
    times = [at for at, _, _ in world.refused]
    assert times == [
        pytest.approx(3.0, abs=TICK_S),
        pytest.approx(2.0 + server_main.PENDING_GRACE_S, abs=TICK_S),
    ]
    assert second.adopted_at is None and third.adopted_at is None
    # The first ticked throughout.
    gaps = [b - a for a, b in zip(first.tick_times, first.tick_times[1:])]
    assert max(gaps) < 2 * TICK_S
    assert first.tick_times[-1] == pytest.approx(8.0, abs=2 * TICK_S)
    refused = "Refused the robot app at robot{}:1: the one at robot1:1 is connected"
    assert refused.format(2) in logs.messages
    assert refused.format(3) in logs.messages
