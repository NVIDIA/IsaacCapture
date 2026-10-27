# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The session process's OpenXR path, which --placeholder-openxr skips, run in process.

A scripted host stands in for RemoteSessionHost, scripted OpenXR sessions and plugins for the
native ones, and a virtual clock for the session module's time.monotonic() and time.sleep().
"""

import json
import logging
import sys
from types import SimpleNamespace

import pytest

import isaaccapture
import isaaccapture.plugin_manager as pm
from isaaccapture import oxr
from isaaccapture.deviceio_trackers import HandTracker, HeadTracker
from isaaccapture.teleop_session_server import _session

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the Teleop Session Server runs on Linux only"
)

NO_HEADSET_ERROR = (
    "Failed to get OpenXR system: XR_ERROR_FORM_FACTOR_UNAVAILABLE (-35). The OpenXR runtime "
    "is up but no headset is connected to it"
)
CONNECTED = oxr.OpenXRHeadsetState.CONNECTED
DISCONNECTED = oxr.OpenXRHeadsetState.DISCONNECTED
TICK_RATE_HZ = 240.0
# A scripted robot app leaves after this many ticks or OpenXR attempts, so a regression fails
# instead of hanging.
MAX_STEPS = 5000


class _Clock:
    """Stands in for the time module in _session: sleep() advances monotonic() at once."""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _hello(plugins=()):
    return json.dumps(
        {
            "protocol": 1,
            "isaaccapture_version": isaaccapture.__version__,
            "app_name": "Robot",
            "tick_rate_hz": TICK_RATE_HZ,
            "trackers": [
                {"name": "0", "type": "HeadTracker", "params": {}, "vendor": None},
                {"name": "1", "type": "HandTracker", "params": {}, "vendor": None},
            ],
            "topics": ["0/head"],
            "plugins": [
                {
                    "plugin_name": name,
                    "plugin_root_id": name,
                    "plugin_args": [],
                    "required": True,
                }
                for name in plugins
            ],
        }
    )


class _Host:
    """RemoteSessionHost stand-in; `on_tick(host)` scripts the robot app."""

    def __init__(self, events, hello, on_tick=None):
        self.events = events
        self.hello = hello
        self.on_tick = on_tick
        self.clock = None
        self.sent = []
        # (tick count, clock time) when each message in `sent` went out.
        self.sent_at = []
        self.alive = True
        self.close_reason = ""
        self.ticks = 0
        self.ready_tick = None
        self.started = None

    def wait_hello(self, timeout_ms):
        return self.hello

    def _record(self, kind, payload):
        self.sent.append((kind, payload))
        self.sent_at.append((self.ticks, self.clock.now))

    def send_hello_ack(self, text):
        self._record("hello_ack", json.loads(text))

    def send_status(self, text):
        self._record("status", json.loads(text))

    def send_ready(self, text):
        self.ready_tick = self.ticks
        self._record("ready", json.loads(text))

    def send_error(self, text):
        self._record("error", text)

    def start_session(self, trackers, handles, vendor_config, topics):
        self.started = (trackers, handles, topics)

    def tick(self):
        self.ticks += 1
        if self.ticks >= MAX_STEPS:
            self.leave()
        if self.on_tick is not None:
            self.on_tick(self)

    def close(self):
        self.events.append("host.close")
        self.alive = False
        self.close_reason = self.close_reason or "closed"

    def leave(self):
        self.alive = False
        self.close_reason = "receive failed: connection closed"

    def stages(self):
        return [
            (kind, message["stage"], message["message"])
            for kind, message in self.sent
            if kind in ("status", "ready")
        ]


class _OpenXR:
    """Scripted oxr.OpenXRSession: `failures` construction errors, then `headset(tick)` states."""

    def __init__(
        self,
        events,
        failures=(),
        headset=lambda tick: CONNECTED,
        failed=None,
        on_attempt=None,
    ):
        self.events = events
        self.failures = list(failures)
        self.headset = headset
        self.failed = failed
        self.on_attempt = on_attempt
        self.attempts = []
        self.host = None

    def __call__(self, app_name, extensions, wait_for_system=True):
        self.attempts.append((app_name, list(extensions), wait_for_system))
        if len(self.attempts) >= MAX_STEPS:
            self.host.leave()
        if self.on_attempt is not None:
            self.on_attempt(len(self.attempts))
        if self.failures:
            raise RuntimeError(self.failures.pop(0))
        return self

    def __enter__(self):
        self.events.append("openxr.enter")
        return self

    def __exit__(self, *exc):
        self.events.append("openxr.exit")

    def get_handles(self):
        return oxr.OpenXRSessionHandles(7, 7, 7, 7)

    def get_provider_snapshot(self):
        if self.failed is not None and self.failed(self.host.ticks):
            return SimpleNamespace(
                state=oxr.OpenXRProviderState.FAILED,
                headset_state=DISCONNECTED,
                reason=oxr.OpenXRProviderReason.SESSION_LOST,
                error="XR_ERROR_SESSION_LOST",
            )
        state = self.headset(self.host.ticks)
        return SimpleNamespace(
            state=oxr.OpenXRProviderState.AVAILABLE,
            headset_state=state,
            reason=oxr.OpenXRProviderReason.NONE
            if state == CONNECTED
            else oxr.OpenXRProviderReason.TRACKING_UNAVAILABLE,
            error="",
        )


class _Plugin:
    def __init__(self, events, name, state=lambda: pm.ProcessState.RUNNING):
        self.events = events
        self.name = name
        self.state = state

    def get_process_snapshot(self):
        return SimpleNamespace(
            state=self.state(),
            reason=pm.ProcessReason.NONE,
            pid=100,
            exit_code=None,
            term_signal=None,
            error_code=None,
            error="",
        )

    def stop(self):
        self.events.append(f"plugin.stop {self.name}")


class _PluginManager:
    """Discovers `names`; starting one in `broken` raises PluginManager's startup error.

    A plugin's shutdown timeout is its entry in `shutdown_timeouts`, else the 2 s default.
    """

    def __init__(
        self,
        events,
        names=(),
        broken=(),
        state=lambda: pm.ProcessState.RUNNING,
        shutdown_timeouts=None,
    ):
        self.events = events
        self.names = list(names)
        self.broken = set(broken)
        self.state = state
        self.shutdown_timeouts = shutdown_timeouts or {}

    def __call__(self, search_paths):
        return self

    def get_plugin_names(self):
        return list(self.names)

    def get_plugin_info(self, name):
        timeout = self.shutdown_timeouts.get(name, 2.0)
        return SimpleNamespace(shutdown_timeout_seconds=timeout)

    def start(self, name, root_id, args):
        if name in self.broken:
            raise RuntimeError("Plugin process exited immediately: exit code 1")
        self.events.append(f"plugin.start {name}")
        return _Plugin(self.events, name, self.state)


@pytest.fixture
def scripted(monkeypatch):
    """Installs the virtual clock and the scripted OpenXR session, extensions and plugins."""
    clock = _Clock()
    monkeypatch.setattr(_session, "time", clock)
    extensions = SimpleNamespace(
        get_required_extensions=lambda trackers, vendor_config: ["XR_EXT_hand_tracking"]
    )
    monkeypatch.setattr(_session, "DeviceIOSession", extensions)

    def install(host, openxr=None, plugins=None, serving_pipe=None):
        host.clock = clock
        if openxr is not None:
            openxr.host = host
            monkeypatch.setattr(oxr, "OpenXRSession", openxr)
        if plugins is not None:
            monkeypatch.setattr(pm, "PluginManager", plugins)
        return _session._Session(host, "robot:1", False, [], serving_pipe)

    return install


def _leave_ticks_after_ready(ticks):
    def on_tick(host):
        if host.ready_tick is not None and host.ticks >= host.ready_tick + ticks:
            host.leave()

    return on_tick


def test_waits_for_a_headset_then_streams(scripted):
    events = []
    host = _Host(events, _hello(plugins=["p"]), on_tick=_leave_ticks_after_ready(3))
    openxr = _OpenXR(
        events,
        failures=[NO_HEADSET_ERROR, NO_HEADSET_ERROR],
        headset=lambda tick: CONNECTED if tick > 3 else DISCONNECTED,
    )
    session = scripted(host, openxr, _PluginManager(events, ["p"]))

    assert session.serve() == 0

    assert openxr.attempts == [("Robot", ["XR_EXT_hand_tracking"], False)] * 3
    assert host.sent[0][0] == "hello_ack"
    assert host.sent[0][1]["server_version"] == isaaccapture.__version__
    stages = host.stages()
    # Waiting is reported once however many attempts it takes.
    assert stages[:4] == [
        ("status", "connecting_openxr", "opening an OpenXR session"),
        ("status", "connecting_openxr", _session.WAITING_FOR_HEADSET),
        ("status", "starting_plugins", "starting p"),
        ("status", "waiting_for_headset", _session.WAITING_FOR_HEADSET),
    ]
    assert ("status", "waiting_for_headset", "headset connected") in stages
    ready = [message for kind, message in host.sent if kind == "ready"]
    assert len(ready) == 1
    assert ready[0]["headset"] == {
        "state": "AVAILABLE",
        "headset_state": "CONNECTED",
        "reason": "NONE",
    }
    assert [entry["state"] for entry in ready[0]["plugins"]] == ["RUNNING"]

    trackers, handles, topics = host.started
    assert [(type(t), name) for t, name in trackers] == [
        (HeadTracker, "0"),
        (HandTracker, "1"),
    ]
    assert (handles.instance, handles.session) == (7, 7)
    assert topics == ["0/head"]
    # Plugins stop first; the trackers close before the OpenXR session they read.
    assert events == [
        "openxr.enter",
        "plugin.start p",
        "plugin.stop p",
        "host.close",
        "openxr.exit",
    ]


def test_ready_needs_the_headset_connected_without_a_break(scripted):
    events = []
    tick_times = {}

    def on_tick(host):
        tick_times[host.ticks] = host.clock.now
        if host.ready_tick is not None:
            host.leave()

    # Connected, dropped at tick 4, back from tick 5.
    host = _Host(events, _hello(), on_tick=on_tick)
    openxr = _OpenXR(
        events, headset=lambda tick: DISCONNECTED if tick == 4 else CONNECTED
    )
    session = scripted(host, openxr)

    assert session.serve() == 0
    [(ready_tick, ready_time)] = [
        at for (kind, _), at in zip(host.sent, host.sent_at) if kind == "ready"
    ]
    assert ready_tick > 5
    connected_for = ready_time - tick_times[5]
    # The specified minimum: connected continuously for at least 1.1 s.
    assert connected_for >= 1.1
    assert connected_for < _session.READY_AFTER_CONNECTED_S + 2 / TICK_RATE_HZ


def test_robot_app_leaving_while_no_headset_ends_the_session(scripted):
    events = []
    host = _Host(events, _hello())

    def leave_at_third(attempt):
        if attempt == 3:
            host.leave()

    openxr = _OpenXR(events, failures=[NO_HEADSET_ERROR] * 5, on_attempt=leave_at_third)
    session = scripted(host, openxr)

    assert session.serve() == 0
    assert len(openxr.attempts) == 3
    # One attempt per OPENXR_RETRY_S.
    assert host.clock.now == pytest.approx(_Clock().now + 3 * _session.OPENXR_RETRY_S)
    assert host.started is None
    assert not any(kind == "error" for kind, _ in host.sent)


def test_other_openxr_errors_end_the_session(scripted):
    events = []
    host = _Host(events, _hello())
    openxr = _OpenXR(events, failures=["Failed to create OpenXR instance: -51"])
    session = scripted(host, openxr)

    assert session.serve() == 1
    assert len(openxr.attempts) == 1
    assert host.sent[-1] == (
        "error",
        "The Teleop Session Server could not open an OpenXR session: "
        "Failed to create OpenXR instance: -51",
    )


def test_a_failed_openxr_provider_ends_the_session(scripted):
    events = []
    host = _Host(events, _hello())
    openxr = _OpenXR(events, failed=lambda tick: tick >= 3)
    session = scripted(host, openxr)

    assert session.serve() == 1
    assert host.sent[-1] == (
        "error",
        "The OpenXR session on the Teleop Session Server failed (SESSION_LOST): "
        "XR_ERROR_SESSION_LOST",
    )
    assert events[-2:] == ["host.close", "openxr.exit"]


def test_a_plugin_that_cannot_start_ends_the_session(scripted):
    events = []
    host = _Host(events, _hello(plugins=["good", "bad"]))
    plugins = _PluginManager(events, ["good", "bad"], broken=["bad"])
    session = scripted(host, _OpenXR(events), plugins)

    assert session.serve() == 1
    assert host.sent[-1] == (
        "error",
        "Plugin 'bad' could not start on the Teleop Session Server: "
        "Plugin process exited immediately: exit code 1",
    )
    assert events == [
        "openxr.enter",
        "plugin.start good",
        "plugin.stop good",
        "host.close",
        "openxr.exit",
    ]


class _BrokenPluginManager:
    """Fails where the session first uses its PluginManager: creating it, or listing plugins."""

    def __init__(self, fail_in):
        self.fail_in = fail_in

    def __call__(self, search_paths):
        if self.fail_in == "create":
            raise OSError("no plugin directory today")
        return self

    def get_plugin_names(self):
        raise RuntimeError("plugin directory vanished")


@pytest.mark.parametrize(
    "fail_in, error",
    [
        ("create", "OSError: no plugin directory today"),
        ("list", "RuntimeError: plugin directory vanished"),
    ],
)
def test_an_unexpected_failure_before_hello_ack_is_sent_as_error(
    scripted, caplog, fail_in, error
):
    events = []
    host = _Host(events, _hello())
    session = scripted(host, _OpenXR(events), _BrokenPluginManager(fail_in))

    with caplog.at_level(logging.ERROR, logger="isaaccapture.teleop_session_server"):
        assert session.serve() == 1
    assert host.sent == [("error", f"The Teleop Session Server failed: {error}")]
    [record] = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert record.getMessage() == "Session for the robot app at robot:1 failed"
    assert record.exc_info is not None
    assert events == []


class _FailingHelloHost(_Host):
    """Its wait_hello raises `error`."""

    def __init__(self, error):
        super().__init__([], None)
        self.error = error

    def wait_hello(self, timeout_ms):
        raise self.error


def test_a_hello_that_is_not_utf8_is_refused(scripted, caplog):
    # What the binding raises when it decodes the HELLO payload.
    error = UnicodeDecodeError(
        "utf-8", b'{"app_name": "\xff"}', 14, 15, "invalid start byte"
    )
    host = _FailingHelloHost(error)
    session = scripted(host)

    with caplog.at_level(logging.WARNING, logger="isaaccapture.teleop_session_server"):
        assert session.serve() == 1
    assert host.sent == [
        (
            "error",
            "HELLO is not valid UTF-8: 'utf-8' codec can't decode byte 0xff in "
            "position 14: invalid start byte",
        )
    ]
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_an_unexpected_failure_while_waiting_for_hello_is_sent_as_error(
    scripted, caplog
):
    host = _FailingHelloHost(OSError("bad file descriptor"))
    session = scripted(host)

    with caplog.at_level(logging.ERROR, logger="isaaccapture.teleop_session_server"):
        assert session.serve() == 1
    assert host.sent == [
        ("error", "The Teleop Session Server failed: OSError: bad file descriptor")
    ]
    [record] = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert record.exc_info is not None


class _ServingPipe:
    """_ServingPipe stand-in that records what the session reports."""

    def __init__(self, events):
        self.events = events

    def mark_serving(self, teardown_s):
        self.events.append(f"serving {teardown_s:g}")

    def close(self):
        self.events.append("serving.close")


def test_reports_its_teardown_time_once_serving(scripted):
    """The plugins stop one after another, so their shutdown timeouts add up."""
    events = []
    host = _Host(
        events, _hello(plugins=["p", "q"]), on_tick=_leave_ticks_after_ready(1)
    )
    plugins = _PluginManager(events, ["p", "q"], shutdown_timeouts={"p": 30.0})
    session = scripted(host, _OpenXR(events), plugins, _ServingPipe(events))

    assert session.serve() == 0
    teardown_s = _session.TEARDOWN_MARGIN_S + 30.0 + 2.0
    # Before the teardown starts, so the main process waits for all of it.
    assert events == [
        f"serving {teardown_s:g}",
        "openxr.enter",
        "plugin.start p",
        "plugin.start q",
        "serving.close",
        "plugin.stop q",
        "plugin.stop p",
        "host.close",
        "openxr.exit",
    ]


def test_sigterm_tells_the_robot_app_before_teardown(scripted):
    events = []

    def on_tick(host):
        if host.ticks == 3:
            raise _session._Shutdown(0)

    host = _Host(events, _hello(plugins=["p"]), on_tick=on_tick)
    session = scripted(host, _OpenXR(events), _PluginManager(events, ["p"]))

    with pytest.raises(_session._Shutdown):
        session.serve()
    assert host.sent[-1] == ("error", _session.SHUTDOWN_MESSAGE)
    assert events[-3:] == ["plugin.stop p", "host.close", "openxr.exit"]


def test_a_tick_failure_ends_the_session(scripted):
    events = []

    def on_tick(host):
        raise RuntimeError("schema mismatch")

    host = _Host(events, _hello(), on_tick=on_tick)
    session = scripted(host, _OpenXR(events))

    assert session.serve() == 1
    assert host.sent[-1] == (
        "error",
        "The Teleop Session Server could not update the trackers: schema mismatch",
    )


def test_running_status_is_periodic_and_follows_changes(scripted):
    events = []
    state = {"plugin": pm.ProcessState.RUNNING}
    exit_after, leave_after = 300, 700  # ticks after READY

    def on_tick(host):
        if host.ready_tick is None:
            return
        if host.ticks == host.ready_tick + exit_after:
            state["plugin"] = pm.ProcessState.EXITED
        if host.ticks == host.ready_tick + leave_after:
            host.leave()

    plugins = _PluginManager(events, ["p"], state=lambda: state["plugin"])
    host = _Host(events, _hello(plugins=["p"]), on_tick=on_tick)
    session = scripted(host, _OpenXR(events), plugins)

    assert session.serve() == 0
    running = [
        (message["plugins"][0]["state"], tick, at)
        for (kind, message), (tick, at) in zip(host.sent, host.sent_at)
        if kind in ("ready", "status") and message["stage"] in ("ready", "running")
    ]
    assert running[0][0] == "RUNNING"
    # A change goes out on the tick that sees it.
    changes = [
        tick
        for (now, tick, _), (before, _, _) in zip(running[1:], running)
        if now != before
    ]
    assert changes == [host.ready_tick + exit_after]
    # Otherwise once per STATUS_PERIOD_S.
    period = _session.STATUS_PERIOD_S
    for (now, _, at), (before, _, before_at) in zip(running[1:], running):
        if now == before:
            assert period <= at - before_at < period + 2 / TICK_RATE_HZ
    # READY, a period later, the change, a period after that; then the robot app leaves.
    assert len(running) == 4
