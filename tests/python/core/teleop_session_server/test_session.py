# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The server's loop in process: its OpenXR session and the robot app connections it serves.

Scripted stand-ins replace the listener, RemoteSessionHost (one per robot app), the OpenXR session
and CloudXR; a virtual clock replaces time.monotonic() and time.sleep(), and refusals run inline.
"""

import json
import logging
import math
import signal
import sys
from types import SimpleNamespace

import pytest

import isaaccapture
from isaaccapture import oxr
from isaaccapture.deviceio_trackers import HandTracker, HeadTracker
from isaaccapture.teleop_session_server import __main__ as server_main
from isaaccapture.teleop_session_server import _common, _session

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
# Virtual seconds after which a scripted run stops, so a regression fails instead of hanging.
MAX_RUN_S = 600.0


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

    def accept(self, timeout_ms):
        assert not self.closed
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
    """RemoteSessionHost stand-in: a scripted robot app that leaves at ``leave_at``."""

    def __init__(self, world, name, hello, on_tick):
        self.world = world
        self.name = name
        self.hello = hello
        self.on_tick = on_tick
        self.hello_error = None
        self.start_error = None
        self.leave_at = math.inf
        self.adopted_at = None
        self.started = None
        self.started_at = None
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
        self._record("error", text)

    def start_session(self, trackers, handles, vendor_config, topics):
        if self.start_error is not None:
            raise self.start_error
        self.started = (trackers, handles, topics)
        self.started_at = self.world.clock.now

    def tick(self):
        self.ticks += 1
        self.tick_times.append(self.world.clock.now)
        if self.on_tick is not None:
            self.on_tick(self)

    def close(self):
        self.world.events.append(f"{self.name}.close")
        self._alive = False

    def kinds(self):
        return [kind for kind, _, _ in self.sent]

    def messages(self, kind):
        return [payload for k, payload, _ in self.sent if k == kind]

    def sent_at(self, kind):
        return [at for k, _, at in self.sent if k == kind]

    def stages(self):
        return [
            (kind, payload["stage"], payload["message"])
            for kind, payload, _ in self.sent
            if kind in ("status", "ready")
        ]


class _OpenXR:
    """oxr.OpenXRSession stand-in: ``failures`` construction errors, then a session whose headset
    follows ``headset(now)``; its provider fails once ``failed(now)``."""

    def __init__(self, world):
        self.world = world
        self.failures = []
        self.headset = lambda now: CONNECTED
        self.failed = lambda now: False
        # (clock time, app name, extensions, wait_for_system)
        self.attempts = []

    def __call__(self, app_name, extensions, wait_for_system=True):
        now = self.world.clock.now
        self.attempts.append((now, app_name, list(extensions), wait_for_system))
        if self.failures:
            raise RuntimeError(self.failures.pop(0))
        self.world.events.append("openxr.open")
        return self

    def get_handles(self):
        return oxr.OpenXRSessionHandles(7, 7, 7, 7)

    def get_provider_snapshot(self):
        now = self.world.clock.now
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
        self.world.events.append("openxr.close")


class _Launcher:
    """CloudXR stand-in whose health check fails from check ``fail_at`` on."""

    def __init__(self, world):
        self.world = world
        self.checks = []
        self.fail_at = math.inf

    def health_check(self):
        self.checks.append(self.world.clock.now)
        if len(self.checks) >= self.fail_at:
            raise RuntimeError("The CloudXR runtime serving /run has stopped")


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

    def run(self, placeholder=False):
        listening = server_main._Listening(self.listener, False, "tcp://test")
        server = server_main._Server(
            [listening], self.launcher, self.stop, placeholder=placeholder
        )
        code = server.run()
        assert self.clock.now < MAX_RUN_S, "the scripted run did not stop"
        return code

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
# Failures that stop the server
# ============================================================================


def test_other_openxr_errors_stop_the_server(world, logs):
    world.openxr.failures = [NO_HEADSET_ERROR, "Failed to create OpenXR instance: -51"]
    robot = world.connect(0.5)
    world.connect(0.7)

    assert world.run() == 1

    text = (
        "The Teleop Session Server could not open an OpenXR session: "
        "Failed to create OpenXR instance: -51"
    )
    assert robot.sent[-1][:2] == ("error", text)
    assert world.refusals() == [("robot2", SHUTDOWN)]
    assert world.events == ["robot1.close"]
    assert f"{text}; stopping the Teleop Session Server" in logs.messages


@pytest.mark.parametrize("with_robot", [True, False], ids=["robot", "idle"])
def test_a_failed_openxr_session_stops_the_server(world, with_robot):
    world.openxr.failed = lambda now: now >= 3.0
    robot = world.connect(1.0) if with_robot else None

    assert world.run() == 1

    assert world.clock.now == pytest.approx(3.0, abs=1.0)
    if robot is not None:
        assert robot.sent[-1][:2] == (
            "error",
            "The OpenXR session on the Teleop Session Server failed (SESSION_LOST): "
            "XR_ERROR_SESSION_LOST",
        )
        # The trackers read through the OpenXR session, so they close first.
        assert world.events == ["openxr.open", "robot1.close", "openxr.close"]
    else:
        assert world.events == ["openxr.open", "openxr.close"]


def test_cloudxr_stopping_stops_the_server(world, logs):
    world.launcher.fail_at = 4
    robot = world.connect(0.5)
    world.connect(1.0)

    assert world.run() == 1

    assert robot.sent[-1][:2] == ("error", SHUTDOWN)
    assert world.refusals() == [("robot2", SHUTDOWN)]
    checks = world.launcher.checks
    assert len(checks) == 4
    assert all(
        b - a == pytest.approx(1.0, abs=TICK_S) for a, b in zip(checks, checks[1:])
    )
    assert world.events == ["openxr.open", "robot1.close", "openxr.close"]
    assert (
        "The CloudXR runtime serving /run has stopped; stopping the Teleop Session Server"
        in logs.messages
    )


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
# Failures that end one robot app's connection
# ============================================================================


def test_a_tick_failure_drops_the_robot_app_and_the_server_keeps_serving(world):
    def fail_at_third(host):
        if host.ticks == 3:
            raise RuntimeError("schema mismatch")

    first = world.connect(0.5, on_tick=fail_at_third)
    second = world.connect(3.0, on_tick=_leave_ticks_after_ready(1))
    world.stop_at = 6.0

    assert world.run() == 0

    assert first.sent[-1][:2] == (
        "error",
        "The Teleop Session Server could not update the trackers: schema mismatch",
    )
    assert "ready" in second.kinds()
    assert len(world.openxr.attempts) == 1
    assert world.events == [
        "openxr.open",
        "robot1.close",
        "robot2.close",
        "openxr.close",
    ]


def test_a_tracker_start_failure_drops_the_robot_app(world):
    robot = world.connect(0.5)
    robot.start_error = RuntimeError("no live vendor 'body.x'")
    world.stop_at = 3.0

    assert world.run() == 0

    assert robot.sent[-1][:2] == (
        "error",
        "The Teleop Session Server could not start the trackers: no live vendor 'body.x'",
    )
    assert world.events == ["openxr.open", "robot1.close", "openxr.close"]


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
