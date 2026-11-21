# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TeleopSession in SessionMode.LIVE with wait_for_headset=False and no headset at the OpenXR runtime.

Entry returns, every step runs without tracker data, and the first step() after a headset has
reached the runtime starts the trackers and the plugins. The oxr.OpenXRSession constructor,
DeviceIOSession.run and the plugin manager are stand-ins; the session without tracker data, the
robot state host and its reader are real. No OpenXR runtime and no headset is used.

The background thread is held in each of its phases by an event of the stand-in constructor or
by the pause before its next attempt, not by a sleep of the test.
"""

import logging
import os
import random
import re
import signal
import sys
import threading
import time
from contextlib import ExitStack, contextmanager, suppress
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from test_local_robot_state import TAKEN
from test_remote_mode import _VALUE, _messages, _Source, _step_until, _wait_for
from test_remote_no_headset import (
    EXTENSIONS,
    NO_HEADSET_AT_OPENXR,
    reader_threads,
    robot_state_files,
)
from test_remote_robot_state import DESCRIPTION

from isaaccapture import deviceio
from isaaccapture.deviceio import McapRecordingConfig
from isaaccapture.deviceio_trackers import (
    HapticCommandPushTracker,
    HeadTracker,
    MessageChannelStatus,
    MessageChannelTracker,
    TactileTracker,
)
from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    ControllersSource,
    HandsSource,
    HeadSource,
    IDeviceIOSink,
    MessageChannelConnectionStatus,
    TactileSource,
    message_channel_config,
)
from isaaccapture.retargeting_engine.interface import ExecutionState, OutputCombiner
from isaaccapture.retargeting_engine.interface.tensor_group import TensorGroup
from isaaccapture.robot_state import (
    RobotState,
    RobotStateLocalService,
    RobotStateStore,
    default_socket_path,
)
from isaaccapture.robot_state._host import RobotStateHost
from isaaccapture.schema import MessageChannelMessages, MessageChannelMessagesTracked
from isaaccapture.teleop_session_manager import (
    DeviceState,
    PluginConfig,
    ProviderState,
    RetargetingExecutionConfig,
    StatusReason,
    TeleopSession,
    TeleopSessionConfig,
    _headset_wait,
    teleop_session,
)
from isaaccapture.teleop_session_manager.async_retarget_runner import (
    AsyncRetargetRunner,
    AsyncRetargetWorkerError,
)
from isaaccapture.teleop_session_manager.status_monitor import StatusMonitor

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="robot state is served over a Unix socket"
)

MODES = ["sync", "pipelined"]
# The thread that calls step(), the one that waits for a headset, and the pipelined worker.
APP = "MainThread"
BACKGROUND = "isaaccapture-headset-wait"
WORKER = "IsaacTeleopAsyncRetarget"
# The pause before each background attempt in these tests, and one that no test outlasts.
RETRY_S = 0.01
LONG_S = 60.0
WAIT_S = 10.0
# What the stand-in pipeline returns for a step without tracker data.
NO_DATA = {"n": 0}
# The library's periods, read before any test shortens them.
PERIODS = (_headset_wait.RETRY_S, _headset_wait.CLOSE_JOIN_S)

WAITING = "No headset has reached the OpenXR runtime yet; waiting for one"
ENTRY_LINE = (
    "No headset has reached the OpenXR runtime; the session starts without one. Its trackers "
    "report no data and its plugins are not started until one does"
)
OPENED_LINE = "A headset has reached the OpenXR runtime; opened the OpenXR session"
STARTING_LINE = (
    "A headset has reached the OpenXR runtime; starting {count} tracker(s) and the plugins "
    "now. This step() returns late, and every Python thread of this program waits until it does"
)
STARTED_LINE = re.compile(
    r"^Started the trackers and (\d+) plugin\(s\) in \d+ ms; "
    r"the session now reads the headset$"
)
STILL_INSIDE = (
    "The thread that waits for a headset is still inside the OpenXR runtime after {limit:g} s; "
    "it ends, and closes a session it still opens, when the runtime answers"
)
OPEN_FAILED = (
    "The OpenXR session could not be opened while waiting for a headset: {error}"
)
TRACKERS_NOT_STARTED = (
    "A headset reached the OpenXR runtime, but the trackers could not be started: "
    "{error}"
)
PLUGIN_NOT_STARTED = (
    "A headset reached the OpenXR runtime, but a plugin could not be started: {error}"
)
NEEDS_ONE = (
    "No headset has reached the OpenXR runtime since it started, and this session cannot start "
    "without one (wait_for_headset=False): {reason}. Wake a headset once, or set "
    "wait_for_headset=True to wait for one."
)
NO_RECORDING_FOLDER = (
    "The folder of the recording file does not exist or cannot be written: {folder}"
)
RUNTIME_STOPPED = "Failed to create OpenXR instance: -51"

RUNTIME_UP = (ProviderState.AVAILABLE, StatusReason.NONE, "")
NOT_STARTED = (ProviderState.INITIALIZING, StatusReason.PROVIDER_INITIALIZING, "")
HEADSET_WAITING = (DeviceState.DISCONNECTED, StatusReason.NO_CURRENT_DATA, WAITING)
HEADSET_TRACKED = (DeviceState.CONNECTED, StatusReason.OPENXR_SESSION_READY, "")
# The calls of one step of a session that reads the headset.
STEP_CALLS = [("deviceio.update", APP), ("openxr.snapshot", APP)]


def _me():
    return threading.current_thread().name


def _threads(name):
    return [thread for thread in threading.enumerate() if thread.name == name]


def _join(name):
    """Wait for every thread called ``name`` to end; whether none is left."""
    for thread in _threads(name):
        thread.join(WAIT_S)
    return _threads(name) == []


def _once(error):
    """A call that raises ``error`` the first time and does nothing after."""
    pending = [error]

    def call(*_args):
        if pending:
            raise pending.pop()

    return call


def _always(error):
    def call(*_args):
        raise error

    return call


# ============================================================================
# Stand-ins
# ============================================================================


class _OpenXR:
    """Stands in for an oxr.OpenXRSession; knows the thread that made it and the one that closed it."""

    def __init__(self, journal):
        self.handles = object()
        self.made_on = _me()
        self.closed_on = None
        self.tracked = True
        self._journal = journal

    def __enter__(self):
        self._journal.append(("openxr.enter", _me()))
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        if self.closed_on is None:
            self.closed_on = _me()
            self._journal.append(("openxr.close", self.closed_on))

    def get_handles(self):
        self._journal.append(("openxr.handles", _me()))
        return self.handles

    def get_provider_snapshot(self):
        self._journal.append(("openxr.snapshot", _me()))
        return SimpleNamespace(
            state="AVAILABLE",
            headset_state="CONNECTED" if self.tracked else "DISCONNECTED",
            reason="NONE" if self.tracked else "TRACKING_UNAVAILABLE",
            result_code=None,
            error="",
        )


class Runtime:
    """Stands in for the oxr.OpenXRSession constructor of a runtime that no headset has reached.

    With ``headset`` set an attempt returns a session; with ``error`` it raises that. ``gate``
    holds an attempt of the background thread inside the constructor: ``inside`` tells that one
    is there, ``left`` that one has returned. ``then_retry_s`` becomes the pause before the
    background thread's next attempt.
    """

    def __init__(self, journal):
        self.headset = threading.Event()
        self.gate = None
        self.inside = threading.Event()
        self.left = threading.Event()
        self.error = None
        self.then_retry_s = None
        self.calls = []
        self.sessions = []
        self._journal = journal

    def __call__(self, *args, **kwargs):
        thread = _me()
        self.calls.append((thread, args, kwargs))
        if thread != BACKGROUND:
            return self._attempt()
        self.inside.set()
        try:
            if self.gate is not None:
                assert self.gate.wait(LONG_S), "the test left an attempt held"
            return self._attempt()
        finally:
            if self.then_retry_s is not None:
                _headset_wait.RETRY_S = self.then_retry_s
            self.left.set()

    def _attempt(self):
        if self.error is not None:
            raise self.error
        if not self.headset.is_set():
            raise RuntimeError(NO_HEADSET_AT_OPENXR)
        session = _OpenXR(self._journal)
        self.sessions.append(session)
        return session


class _DeviceIO:
    """Stands in for the DeviceIOSession the trackers run in."""

    def __init__(self, journal):
        self._journal = journal

    def __enter__(self):
        self._journal.append(("deviceio.enter", _me()))
        return self

    def __exit__(self, *exc):
        self._journal.append(("deviceio.close", _me()))
        return False

    def update(self):
        self._journal.append(("deviceio.update", _me()))


class _NoData:
    """Enters the real session without tracker data and records its close."""

    def __init__(self, session, journal):
        self._session = session
        self._journal = journal

    def __enter__(self):
        return self._session.__enter__()

    def __exit__(self, *exc):
        self._journal.append(("no_data.close", _me()))
        return self._session.__exit__(*exc)


class _PluginContext:
    def __init__(self, name, journal):
        self._name = name
        self._journal = journal
        self.stopped = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stopped = True
        self._journal.append(("plugin.stop", _me(), self._name))
        return False

    def check_health(self):
        pass

    def get_process_snapshot(self):
        return SimpleNamespace(
            state="RUNNING",
            reason="NONE",
            pid=123,
            exit_code=None,
            term_signal=None,
            error_code=None,
            error="",
        )


class _Plugins:
    """Stands in for plugin_manager.PluginManager: "gamepad" is monitored, "legacy" is not.

    ``in_start`` maps a plugin's name to a call made inside its start.
    """

    def __init__(self, journal):
        self._journal = journal
        self.in_start = {}
        self.contexts = []

    def get_plugin_names(self):
        return ["gamepad", "legacy"]

    def get_plugin_info(self, name):
        device = SimpleNamespace(path=f"/{name}", type=name, description=name)
        return SimpleNamespace(name=name, devices=(device,))

    def start(self, name, root_id, args=None):
        self._journal.append(("plugin.start", _me(), name))
        self.in_start.get(name, lambda: None)()
        context = _PluginContext(name, self._journal)
        self.contexts.append(context)
        return context


class _Pipeline:
    """Counts its inputs; knows the thread of each step."""

    def __init__(self):
        self.threads = []
        self.error = None
        self.raised = threading.Event()
        self.delay_s = 0.0
        self.gate = None
        self.inside = threading.Event()
        self._journal = None

    def hold(self, journal):
        """Hold a step inside until ``gate`` is set; ``journal`` gets the moment it leaves."""
        self.gate = threading.Event()
        self._journal = journal

    def get_leaf_nodes(self):
        return []

    def execute_pipeline(self, inputs, context=None):
        self.threads.append(_me())
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.gate is not None and not self.gate.is_set():
            self.inside.set()
            assert self.gate.wait(LONG_S), "the test left a step held"
            self._journal.append(("pipeline.left", _me()))
        if self.error is not None:
            self.raised.set()
            raise self.error
        return {"n": len(inputs)}


class _Haptics(IDeviceIOSink):
    """A sink that keeps what it computes and the sessions it is flushed to."""

    def __init__(self, tracker):
        self._tracker = tracker
        self.computed = []
        self.flushed_to = []
        super().__init__("haptics")

    def input_spec(self):
        return {"in": _VALUE}

    def get_tracker(self):
        return self._tracker

    def flush_to_device(self, deviceio_session):
        self.flushed_to.append(deviceio_session)

    def _compute_fn(self, inputs, outputs, context):
        self.computed.append(inputs["in"][0])


class Rig:
    """The stand-ins of one test and, in ``journal``, every call they saw: (what, thread, ...)."""

    def __init__(self, runtime_dir):
        self.runtime_dir = runtime_dir
        self.retry_s = RETRY_S
        self.journal = []
        self.runtimes = []
        self.entered = []
        self.waits = []
        self.no_data = []
        self.live_run = None
        self._reset()

    def _reset(self):
        del self.journal[:]
        self.runtime = Runtime(self.journal)
        self.runtimes.append(self.runtime)
        self.live = _DeviceIO(self.journal)
        self.plugins = _Plugins(self.journal)
        self.closing = threading.Event()
        self.in_run = None
        self.fail_run = None

    def construct(self, *args, **kwargs):
        return self.runtime(*args, **kwargs)

    def run(self, *args, **kwargs):
        self.journal.append(("deviceio.run", _me(), bool(_threads(WORKER))))
        if self.in_run is not None:
            self.in_run()
        if self.fail_run is not None:
            raise self.fail_run
        return self.live

    def enter(self, mode="sync", **kwargs):
        """An entered session; the rig leaves it if the test does not."""
        session = TeleopSession(_config(mode, **kwargs))
        self.entered.append(session)
        session.__enter__()
        return session

    def leave_while_held(self, leave, pause_s=0.05):
        """Run ``leave`` and open the gate only once it waits for the background thread.

        Whether ``leave`` is still waiting ``pause_s`` after it begins to.
        """
        returned = threading.Event()
        waited = []

        def open_gate():
            if self.closing.wait(WAIT_S):
                waited.append(not returned.wait(pause_s))
            self.runtime.gate.set()

        helper = threading.Thread(target=open_gate, name="test-gate")
        helper.start()
        try:
            leave()
        finally:
            returned.set()
            helper.join(WAIT_S)
        return waited == [True]

    def check_nothing_left(self):
        """No background thread, no worker thread, every OpenXR session closed."""
        names = [thread.name for thread in threading.enumerate()]
        assert [name for name in names if name in (BACKGROUND, WORKER)] == []
        for runtime in self.runtimes:
            assert None not in [session.closed_on for session in runtime.sessions]

    def next_round(self):
        """Check that the round left nothing, then start with new stand-ins."""
        self.check_nothing_left()
        del self.runtimes[:], self.entered[:], self.waits[:], self.no_data[:]
        self.live_run.reset_mock()
        _headset_wait.RETRY_S = self.retry_s
        self._reset()

    def release(self):
        """End what a failed test left, while the constructor is still the stand-in."""
        for runtime in self.runtimes:
            if runtime.gate is not None:
                runtime.gate.set()
        for session in self.entered:
            gate = getattr(session.pipeline, "gate", None)
            if gate is not None:
                gate.set()
            if session._in_context:
                with suppress(Exception):
                    session.__exit__(None, None, None)
        for wait in self.waits:
            wait.close()


@pytest.fixture(autouse=True)
def rig(monkeypatch, tmp_path_factory):
    """Replace the OpenXR runtime, DeviceIO and the plugins; every test ends with nothing left."""
    runtime_dir = tmp_path_factory.mktemp("rt")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))
    assert default_socket_path().startswith(str(runtime_dir))
    rig = Rig(runtime_dir)
    create = deviceio.NoDataSession.create
    start = _headset_wait.HeadsetWait.start
    close = _headset_wait.HeadsetWait.close

    def recorded_create(*args, **kwargs):
        session = create(*args, **kwargs)
        rig.no_data.append(session)
        return _NoData(session, rig.journal)

    def recorded_start(wait):
        rig.waits.append(wait)
        start(wait)

    def recorded_close(wait):
        rig.closing.set()
        close(wait)

    monkeypatch.setattr(_headset_wait, "RETRY_S", RETRY_S)
    monkeypatch.setattr(_headset_wait.HeadsetWait, "start", recorded_start)
    monkeypatch.setattr(_headset_wait.HeadsetWait, "close", recorded_close)
    with (
        patch("isaaccapture.oxr.OpenXRSession", new=rig.construct),
        patch(
            "isaaccapture.deviceio.DeviceIOSession.run", side_effect=rig.run
        ) as rig.live_run,
        patch(
            "isaaccapture.deviceio.DeviceIOSession.get_required_extensions",
            return_value=list(EXTENSIONS),
        ),
        patch(
            "isaaccapture.deviceio.NoDataSession.create", side_effect=recorded_create
        ),
        patch(
            "isaaccapture.plugin_manager.PluginManager",
            side_effect=lambda paths: rig.plugins,
        ),
        patch("isaaccapture.deviceio_trackers.PluginDeviceStatusTracker"),
    ):
        try:
            yield rig
            rig.check_nothing_left()
        finally:
            rig.release()


@pytest.fixture
def fine_switching():
    """Let the threads take turns far more often than a step lasts, for the race tests."""
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-4)
    yield
    sys.setswitchinterval(interval)


# ============================================================================
# Helpers
# ============================================================================


def _base_config(mode="sync", **kwargs):
    kwargs.setdefault("pipeline", _Pipeline())
    return TeleopSessionConfig(
        app_name="LocalApp",
        verbose=False,
        retargeting_execution=RetargetingExecutionConfig(mode=mode),
        **kwargs,
    )


def _config(mode="sync", **kwargs):
    kwargs.setdefault("wait_for_headset", False)
    return _base_config(mode, **kwargs)


def _plugin(name, folder, **kwargs):
    return PluginConfig(
        plugin_name=name, plugin_root_id=name, search_paths=[folder], **kwargs
    )


def _provider(session, provider_id="openxr/runtime"):
    status = session.get_provider_status(provider_id)
    return status.status, status.reason, status.error


def _device(session, device_id="openxr/headset"):
    status = session.get_device_status(device_id)
    return status.status, status.reason, status.error


def _no_data_steps(session, mode):
    """Steps without tracker data; in pipelined mode until the worker has run one."""
    for _ in range(3):
        assert session.step() == NO_DATA
    if mode == "pipelined":
        assert _step_until(session, lambda: WORKER in session.pipeline.threads)
        assert not session.last_step_info.ran_synchronously
    else:
        assert session.last_step_info.ran_synchronously


def _opened(session):
    """Wait until the background thread holds an OpenXR session for the next step()."""
    wait = session._headset_wait
    assert _wait_for(lambda: wait.poll() is not None, WAIT_S)


def _hand_over(rig, session):
    """Let a headset reach the runtime and make the step that starts the trackers."""
    rig.runtime.headset.set()
    _opened(session)
    return session.step()


def _raising_step(session, error_type=RuntimeError):
    with pytest.raises(error_type) as raised:
        session.step()
    return raised.value


@contextmanager
def _leaving(session):
    """Leave an entered session at the end of the block, as its ``with`` block does."""
    with ExitStack() as stack:
        stack.push(session)
        yield session


def _interrupt_the_body(session):
    """Ctrl-C in the body of the ``with`` block of an entered session."""
    with pytest.raises(KeyboardInterrupt), _leaving(session):
        raise KeyboardInterrupt


def _named(journal, *names):
    return [entry for entry in journal if entry[0] in names]


def _run_arguments(rig):
    """The arguments of the one DeviceIOSession.run call, without the objects made per entry."""
    ((args, kwargs),) = rig.live_run.call_args_list
    trackers, handles, mcap_config, vendor_config = args
    return (
        list(trackers),
        handles is rig.runtime.sessions[0].handles,
        mcap_config.filename,
        [name for _tracker, name in mcap_config.get_tracker_names()],
        type(vendor_config).__name__,
        kwargs,
    )


PHASES = [
    "before the first attempt",
    "between attempts",
    "inside an attempt",
    "opened",
]


def _enter_in_phase(rig, monkeypatch, mode, phase, **kwargs):
    """An entered session that has stepped, its background thread held in ``phase``."""
    runtime = rig.runtime
    if phase == "before the first attempt":
        monkeypatch.setattr(_headset_wait, "RETRY_S", LONG_S)
    elif phase == "between attempts":
        runtime.then_retry_s = LONG_S
    elif phase == "inside an attempt":
        runtime.gate = threading.Event()
    session = rig.enter(mode, **kwargs)
    _no_data_steps(session, mode)
    if phase == "before the first attempt":
        assert len(runtime.calls) == 1
    elif phase == "between attempts":
        assert runtime.left.wait(WAIT_S)
    elif phase == "inside an attempt":
        assert runtime.inside.wait(WAIT_S)
    elif phase == "opened":
        runtime.headset.set()
        _opened(session)
    return session


def _channel(capacity=4):
    """A message channel's source and sink, and a pipeline that returns its status."""
    source, sink = message_channel_config(
        "chat", bytes(range(16)), "chat", outbound_queue_capacity=capacity
    )
    # A pipelined session copies its outputs, and a message batch cannot be copied.
    return source, sink, OutputCombiner({"status": source.output("status")})


def _send(sink, payload):
    group = TensorGroup(sink.input_spec()["messages_tracked"])
    group[0] = MessageChannelMessagesTracked([MessageChannelMessages(payload)])
    sink.compute({"messages_tracked": group}, {})


@pytest.fixture
def connected_channel(rig):
    """Once the trackers run, every message channel is connected; yields what was sent."""
    sent = []
    real = {
        name: getattr(MessageChannelTracker, name)
        for name in ("get_status", "get_messages", "send_message")
    }

    def route(name, answer):
        def accessor(tracker, session, *args):
            if session is rig.live:
                return answer(*args)
            return real[name](tracker, session, *args)

        return patch.object(MessageChannelTracker, name, accessor)

    with (
        route("get_status", lambda: MessageChannelStatus.CONNECTED),
        route("get_messages", MessageChannelMessagesTracked),
        route("send_message", lambda message: sent.append(bytes(message.payload))),
    ):
        yield sent


class _CountingTactileSource(TactileSource):
    """A TactileSource of a robot app that records what it is polled with."""

    def __init__(self, *args, **kwargs):
        self.polled_with = []
        super().__init__(*args, **kwargs)

    def poll_tracker(self, deviceio_session):
        self.polled_with.append(deviceio_session)
        return super().poll_tracker(deviceio_session)


# ============================================================================
# While no headset has reached the runtime
# ============================================================================


class TestWaiting:
    def test_the_texts_are_the_librarys(self):
        assert _headset_wait.THREAD_NAME == BACKGROUND
        assert _headset_wait.WAITING == WAITING
        assert _headset_wait.OPEN_FAILED == OPEN_FAILED
        assert NO_HEADSET_AT_OPENXR.startswith(_headset_wait.NO_HEADSET_ERROR)
        assert PERIODS == (1.0, 5.0)
        assert teleop_session._TRACKERS_NOT_STARTED == TRACKERS_NOT_STARTED
        assert teleop_session._PLUGIN_NOT_STARTED == PLUGIN_NOT_STARTED
        assert teleop_session._NO_HEADSET_NEEDS_ONE == NEEDS_ONE
        assert teleop_session._NO_RECORDING_FOLDER == NO_RECORDING_FOLDER

    @pytest.mark.parametrize("mode", MODES)
    def test_entry_returns_and_waits(self, mode, rig, tmp_path, caplog):
        config = _config(
            mode,
            trackers=[HeadTracker()],
            plugins=[_plugin("legacy", tmp_path), _plugin("gamepad", tmp_path)],
        )
        with (
            caplog.at_level(logging.INFO, logger="isaaccapture"),
            TeleopSession(config) as session,
        ):
            attempt = (("LocalApp", EXTENSIONS), {"wait_for_system": False})
            assert rig.runtime.calls[0] == (APP, *attempt)
            assert session.oxr_session is None
            assert type(session.deviceio_session) is deviceio.NoDataSession
            assert [session.deviceio_session] == rig.no_data
            assert (session.plugin_managers, session.plugin_contexts) == ([], [])
            assert [thread.daemon for thread in _threads(BACKGROUND)] == [True]

            for _ in range(2):
                assert _provider(session) == RUNTIME_UP
                assert _device(session) == HEADSET_WAITING
                assert _provider(session, "plugin/gamepad") == NOT_STARTED
                assert _device(session, "gamepad/gamepad") == (
                    DeviceState.INITIALIZING,
                    StatusReason.PROVIDER_INITIALIZING,
                    "",
                )
                _no_data_steps(session, mode)
                assert session.last_step_info.remote_tick_age_s is None
                events = session.last_context.execution_events
                assert events.execution_state == ExecutionState.RUNNING

            threads = session.pipeline.threads
            assert threads[0] == APP
            assert set(threads) == ({APP} if mode == "sync" else {APP, WORKER})
            # One attempt of the entry, every later one of the background thread.
            assert rig.runtime.left.wait(WAIT_S)
            calls = list(rig.runtime.calls)
            assert {thread for thread, _, _ in calls[1:]} == {BACKGROUND}
            assert all(call[1:] == attempt for call in calls)
            # Nothing of the trackers and the plugins is started.
            assert rig.journal == []
            rig.live_run.assert_not_called()
        assert _messages(caplog, logging.INFO).count(ENTRY_LINE) == 1
        assert _messages(caplog, logging.WARNING) == []
        assert rig.journal == [("no_data.close", APP)]

    def test_every_built_in_source_gives_no_data(self):
        head = HeadSource(name="head")
        hands = HandsSource(name="hands")
        controllers = ControllersSource(name="controllers")
        pipeline = OutputCombiner(
            {
                "head": head.output("head"),
                "hand_left": hands.output(HandsSource.LEFT),
                "hand_right": hands.output(HandsSource.RIGHT),
                "controller_left": controllers.output(ControllersSource.LEFT),
                "controller_right": controllers.output(ControllersSource.RIGHT),
            }
        )
        sides = ("openxr/controller/left", "openxr/controller/right")
        with TeleopSession(_config(pipeline=pipeline)) as session:
            result = session.step()
            assert len(result) == 5
            assert all(group.is_none for group in result.values())
            for side in sides:
                assert _device(session, side) == (
                    DeviceState.INITIALIZING,
                    StatusReason.PROVIDER_INITIALIZING,
                    "",
                )
            # The status monitor reports a controller without samples after one second.
            time.sleep(1.05)
            result = session.step()
            assert all(group.is_none for group in result.values())
            for side in sides:
                status, _reason, error = _device(session, side)
                assert status == DeviceState.DISCONNECTED
                assert error == "controller is inactive for at least 1 second"
            assert _device(session) == HEADSET_WAITING

    @pytest.mark.parametrize("mode", MODES)
    def test_a_sink_runs_and_is_not_flushed(self, mode, rig):
        head = _Source("head", HeadTracker())
        sink = _Haptics(HapticCommandPushTracker("c", "t"))
        config = _config(
            mode,
            pipeline=OutputCombiner({"value": head.output("value")}),
            sinks=[sink.connect({"in": head.output("value")})],
        )
        with TeleopSession(config) as session:
            for _ in range(3):
                assert session.step()["value"][0] == 0.0
            assert sink.computed
            assert mode == "pipelined" or len(sink.computed) == 3
            assert sink.flushed_to == []

            # The step that starts the trackers flushes to the session they run in.
            _hand_over(rig, session)
            assert sink.flushed_to == [rig.live]
            ((trackers, *_), _kwargs) = rig.live_run.call_args
            assert sink.get_tracker() in trackers

    @pytest.mark.parametrize("mode", MODES)
    def test_robot_state_is_read_from_the_first_step(self, mode, rig):
        config = _config(mode, robot_description=DESCRIPTION)
        with TeleopSession(config) as session, RobotState.connect() as robot:
            # Before any step.
            assert session.robot_state_active
            assert session.publish_robot_state("arm", [0.25, -0.5, 1.0])
            assert robot.connected
            assert robot.description() == DESCRIPTION
            assert robot.joints("arm").positions == (0.25, -0.5, 1.0)

            _no_data_steps(session, mode)
            assert session.publish_robot_state("arm", [0.5, 0.0, 0.0])
            assert robot.joints("arm").positions == (0.5, 0.0, 0.0)

            _hand_over(rig, session)
            assert session.oxr_session is rig.runtime.sessions[0]
            assert session.robot_state_active
            assert robot.connected
            assert session.publish_robot_state("arm", [0.75, 0.25, 1.0])
            assert robot.joints("arm").positions == (0.75, 0.25, 1.0)

    def test_the_second_robot_is_refused_before_any_attempt(self, rig):
        other = RobotStateLocalService(RobotStateStore(), default_socket_path())
        other.start()
        try:
            session = TeleopSession(_config(robot_description=DESCRIPTION))
            with pytest.raises(RuntimeError) as raised:
                session.__enter__()
            assert str(raised.value) == TAKEN.format(path=default_socket_path())
            assert (rig.runtime.calls, rig.no_data, rig.waits) == ([], [], [])
            assert _threads(BACKGROUND) == []
            assert not session._in_context
            assert other.serving
        finally:
            other.close()


# ============================================================================
# The step that starts the trackers
# ============================================================================


class TestHandOver:
    @pytest.mark.parametrize("mode", MODES)
    def test_the_next_step_starts_trackers_and_plugins(self, mode, rig, tmp_path):
        tracker = HeadTracker()
        kwargs = {
            "trackers": [tracker],
            "plugins": [_plugin("legacy", tmp_path), _plugin("gamepad", tmp_path)],
            "mcap_config": McapRecordingConfig(
                str(tmp_path / "session.mcap"), [(tracker, "head")]
            ),
        }
        # What an entry that waits for the headset passes to DeviceIOSession.run.
        rig.runtime.headset.set()
        with TeleopSession(_config(mode, wait_for_headset=True, **kwargs)):
            default = _run_arguments(rig)
        rig.next_round()

        with TeleopSession(_config(mode, **kwargs)) as session:
            _no_data_steps(session, mode)
            workers = _threads(WORKER)
            steps = session.frame_count
            rig.runtime.headset.set()
            _opened(session)
            # Opened, not taken: nothing is started before the next step().
            assert rig.journal == []
            assert session.oxr_session is None

            session.step()
            (openxr,) = rig.runtime.sessions
            assert openxr.made_on == BACKGROUND
            assert session.oxr_session is openxr
            assert session.deviceio_session is rig.live
            assert rig.journal == [
                ("openxr.handles", APP),
                ("deviceio.run", APP, False),
                ("deviceio.enter", APP),
                ("plugin.start", APP, "legacy"),
                ("plugin.start", APP, "gamepad"),
                *STEP_CALLS,
            ]
            assert _run_arguments(rig) == default
            assert len(session.plugin_contexts) == 2
            assert session.last_step_info.ran_synchronously
            assert session.frame_count == steps + 1
            assert _device(session) == HEADSET_TRACKED
            assert _provider(session) == RUNTIME_UP
            assert _provider(session, "plugin/gamepad")[:2] == RUNTIME_UP[:2]
            # The background thread keeps the OpenXR session until exit.
            assert [thread.is_alive() for thread in _threads(BACKGROUND)] == [True]
            assert openxr.closed_on is None

            mark = len(rig.journal)
            if mode == "sync":
                session.step()
                assert rig.journal[mark:] == STEP_CALLS
            else:
                update = ("deviceio.update", WORKER)
                assert _step_until(session, lambda: update in rig.journal)
                assert not session.last_step_info.ran_synchronously
                # The first step and the one that started the trackers ran on this thread.
                assert session.pipeline.threads.count(APP) == 2
                (worker,) = _threads(WORKER)
                assert worker not in workers
                assert not any(thread.is_alive() for thread in workers)
        assert rig.live_run.call_count == 1
        assert len(rig.runtime.sessions) == 1

    def test_a_reference_kept_from_entry_still_gives_no_data(self, rig):
        tracker = HeadTracker()
        with TeleopSession(_config(trackers=[tracker])) as session:
            session.step()
            kept = session.deviceio_session
            assert tracker.get_head(kept) is None
            _hand_over(rig, session)
            assert session.deviceio_session is not kept
            assert tracker.get_head(kept) is None
            session.step()
            assert tracker.get_head(kept) is None

    def test_one_line_is_logged_before_the_late_step_and_one_after(
        self, rig, tmp_path, caplog
    ):
        def info():
            return _messages(caplog, logging.INFO)

        before = []
        rig.in_run = lambda: before.append(info())
        config = _config(
            trackers=[HeadTracker()], plugins=[_plugin("gamepad", tmp_path)]
        )
        with (
            caplog.at_level(logging.INFO, logger="isaaccapture"),
            TeleopSession(config) as session,
        ):
            session.step()
            rig.runtime.headset.set()
            assert _wait_for(lambda: OPENED_LINE in info(), WAIT_S)
            session.step()
            session.step()
        ((trackers, *_), _kwargs) = rig.live_run.call_args
        starting = STARTING_LINE.format(count=len(trackers))
        assert len(trackers) == 2
        # Inside DeviceIOSession.run the first line is out and the second is not.
        (seen,) = before
        assert seen.count(starting) == 1
        assert [text for text in seen if STARTED_LINE.match(text)] == []
        lines = info()
        started = [text for text in lines if STARTED_LINE.match(text)]
        assert len(started) == 1
        assert STARTED_LINE.match(started[0]).group(1) == "1"
        assert lines.count(starting) == 1
        assert lines.index(starting) < lines.index(started[0])
        assert lines.count(OPENED_LINE) == 1
        assert lines.count(ENTRY_LINE) == 1
        assert _messages(caplog, logging.WARNING) == []

    def test_a_headset_that_leaves_later_changes_nothing(self, rig):
        with TeleopSession(_config()) as session:
            session.step()
            _hand_over(rig, session)
            openxr = session.oxr_session
            attempts = len(rig.runtime.calls)

            openxr.tracked = False
            rig.runtime.headset.clear()
            for _ in range(5):
                session.step()
                assert _device(session) == (
                    DeviceState.DISCONNECTED,
                    StatusReason.NO_CURRENT_DATA,
                    "",
                )
                assert _provider(session) == RUNTIME_UP
            openxr.tracked = True
            session.step()
            assert _device(session) == HEADSET_TRACKED

            assert session.oxr_session is openxr
            assert session.deviceio_session is rig.live
            assert len(rig.runtime.calls) == attempts
            assert rig.live_run.call_count == 1
            assert [thread.is_alive() for thread in _threads(BACKGROUND)] == [True]

    def test_one_session_enters_twice(self, rig):
        session = TeleopSession(_config())
        for entry in (1, 2):
            rig.runtime.headset.clear()
            with session:
                assert session.oxr_session is None
                assert session.step() == NO_DATA
                assert _device(session) == HEADSET_WAITING
                _hand_over(rig, session)
                assert session.oxr_session is rig.runtime.sessions[-1]
                assert _device(session) == HEADSET_TRACKED
                assert rig.live_run.call_count == entry
            assert session.oxr_session is None
            assert session._headset_wait is None
            assert _threads(BACKGROUND) == []
        closed_on = [openxr.closed_on for openxr in rig.runtime.sessions]
        assert closed_on == [BACKGROUND, BACKGROUND]

    def test_many_rounds_of_arrival_and_exit(self, rig, fine_switching):
        chance = random.Random(7)
        rig.retry_s = 0.002
        outcomes = {"handed over": 0, "opened, not taken": 0, "waiting": 0}
        for index in range(300):
            rig.next_round()
            pipeline = _Pipeline()
            pipeline.delay_s = chance.choice((0.0, 0.0, 0.001))
            # The robot app's loop: as fast as it can, or with a pause between steps.
            pause_s = chance.choice((0.0, 0.0005, 0.002))
            session = rig.enter(MODES[index % 2], pipeline=pipeline)
            arrival = threading.Timer(
                chance.uniform(0.0, 0.01), rig.runtime.headset.set
            )
            arrival.start()
            end = time.monotonic() + chance.uniform(0.0, 0.02)
            while time.monotonic() < end:
                session.step()
                time.sleep(pause_s)
            handed_over = session.oxr_session is not None
            session.__exit__(None, None, None)
            arrival.join()

            assert _threads(BACKGROUND) == _threads(WORKER) == []
            sessions = rig.runtime.sessions
            assert len(sessions) <= 1
            assert [openxr.closed_on for openxr in sessions] == [BACKGROUND] * len(
                sessions
            )
            closes = [what for what, *_ in rig.journal if what.endswith(".close")]
            if handed_over:
                # Once, on this thread, with no worker alive.
                assert _named(rig.journal, "deviceio.run") == [
                    ("deviceio.run", APP, False)
                ]
                assert closes == ["deviceio.close", "openxr.close", "no_data.close"]
                outcomes["handed over"] += 1
            else:
                assert _named(rig.journal, "deviceio.run") == []
                assert closes == ["openxr.close"] * len(sessions) + ["no_data.close"]
                outcomes["opened, not taken" if sessions else "waiting"] += 1
        assert all(outcomes.values()), outcomes


# ============================================================================
# Exit in each phase of the background thread
# ============================================================================


def _leave_inside_an_attempt(rig, mode, prepare):
    """Leave a session while its background thread is inside an attempt that ``prepare`` scripts."""
    rig.runtime.gate = threading.Event()
    session = rig.enter(mode)
    _no_data_steps(session, mode)
    assert rig.runtime.inside.wait(WAIT_S)
    prepare(rig.runtime)
    # Exit waits for the attempt and returns once it has.
    assert rig.leave_while_held(lambda: session.__exit__(None, None, None))
    assert rig.runtime.left.is_set()
    assert _threads(BACKGROUND) == []
    assert not session._in_context
    rig.live_run.assert_not_called()


class TestExit:
    @pytest.mark.parametrize("mode", MODES)
    def test_exit_before_the_first_background_attempt(
        self, mode, rig, monkeypatch, caplog
    ):
        session = _enter_in_phase(rig, monkeypatch, mode, PHASES[0])
        with caplog.at_level(logging.WARNING, logger="isaaccapture"):
            start = time.monotonic()
            session.__exit__(None, None, None)
            took = time.monotonic() - start
        assert took < 0.5
        assert _threads(BACKGROUND) == []
        assert len(rig.runtime.calls) == 1
        assert _messages(caplog, logging.WARNING) == []
        assert rig.journal == [("no_data.close", APP)]

    @pytest.mark.parametrize("mode", MODES)
    def test_exit_between_attempts(self, mode, rig, monkeypatch, caplog):
        session = _enter_in_phase(rig, monkeypatch, mode, PHASES[1])
        with caplog.at_level(logging.WARNING, logger="isaaccapture"):
            start = time.monotonic()
            session.__exit__(None, None, None)
            took = time.monotonic() - start
        assert took < 0.5
        assert _threads(BACKGROUND) == []
        assert [thread for thread, _, _ in rig.runtime.calls] == [APP, BACKGROUND]
        assert _messages(caplog, logging.WARNING) == []

    @pytest.mark.parametrize("mode", MODES)
    def test_exit_inside_an_attempt_that_finds_no_headset(self, mode, rig):
        _leave_inside_an_attempt(rig, mode, lambda runtime: None)
        assert rig.runtime.sessions == []
        assert rig.journal == [("no_data.close", APP)]

    @pytest.mark.parametrize("mode", MODES)
    def test_exit_inside_an_attempt_that_succeeds(self, mode, rig):
        _leave_inside_an_attempt(rig, mode, lambda runtime: runtime.headset.set())
        (openxr,) = rig.runtime.sessions
        assert (openxr.made_on, openxr.closed_on) == (BACKGROUND, BACKGROUND)
        assert rig.journal == [("openxr.close", BACKGROUND), ("no_data.close", APP)]

    @pytest.mark.parametrize("mode", MODES)
    def test_exit_inside_an_attempt_that_fails(self, mode, rig, caplog):
        error = RuntimeError(RUNTIME_STOPPED)
        with caplog.at_level(logging.WARNING, logger="isaaccapture"):
            _leave_inside_an_attempt(
                rig, mode, lambda runtime: setattr(runtime, "error", error)
            )
        assert rig.runtime.sessions == []
        assert _messages(caplog, logging.ERROR) == [OPEN_FAILED.format(error=error)]
        assert _messages(caplog, logging.WARNING) == []

    @pytest.mark.parametrize("mode", MODES)
    def test_exit_past_the_join_limit_warns_and_returns(
        self, mode, rig, monkeypatch, caplog
    ):
        monkeypatch.setattr(_headset_wait, "CLOSE_JOIN_S", 0.2)
        session = _enter_in_phase(rig, monkeypatch, mode, PHASES[2])
        rig.runtime.headset.set()
        with caplog.at_level(logging.WARNING, logger="isaaccapture"):
            start = time.monotonic()
            session.__exit__(None, None, None)
            took = time.monotonic() - start
        assert took >= 0.19
        assert _messages(caplog, logging.WARNING) == [STILL_INSIDE.format(limit=0.2)]
        assert not session._in_context
        assert session.oxr_session is None
        assert rig.journal == [("no_data.close", APP)]
        assert [thread.is_alive() for thread in _threads(BACKGROUND)] == [True]

        # When the runtime answers, the thread closes the session it got and ends.
        rig.runtime.gate.set()
        assert _join(BACKGROUND)
        (openxr,) = rig.runtime.sessions
        assert (openxr.made_on, openxr.closed_on) == (BACKGROUND, BACKGROUND)
        rig.live_run.assert_not_called()

    @pytest.mark.parametrize("mode", MODES)
    def test_exit_after_the_open_before_the_next_step(self, mode, rig, monkeypatch):
        session = _enter_in_phase(rig, monkeypatch, mode, PHASES[3])
        wait = session._headset_wait
        (openxr,) = rig.runtime.sessions
        assert wait.poll() is openxr
        session.__exit__(None, None, None)
        assert openxr.closed_on == BACKGROUND
        # A closed wait hands out no session, and closes once.
        assert wait.poll() is None
        wait.close()
        assert rig.journal == [("openxr.close", BACKGROUND), ("no_data.close", APP)]
        assert _threads(BACKGROUND) == []
        rig.live_run.assert_not_called()

    @pytest.mark.parametrize("mode", MODES)
    def test_exit_after_the_hand_over_closes_in_order(
        self, mode, rig, monkeypatch, tmp_path
    ):
        close = RobotStateHost.close

        def recorded_close(host):
            rig.journal.append(("host.close", _me()))
            close(host)

        monkeypatch.setattr(RobotStateHost, "close", recorded_close)
        session = rig.enter(
            mode,
            plugins=[_plugin("legacy", tmp_path), _plugin("gamepad", tmp_path)],
            robot_description=DESCRIPTION,
        )
        _no_data_steps(session, mode)
        _hand_over(rig, session)
        if mode == "pipelined":
            update = ("deviceio.update", WORKER)
            assert _step_until(session, lambda: update in rig.journal)
        mark = len(rig.journal)
        session.__exit__(None, None, None)
        # A worker step under way ends first.
        closing = [entry for entry in rig.journal[mark:] if entry[1] != WORKER]
        assert closing == [
            ("plugin.stop", APP, "gamepad"),
            ("plugin.stop", APP, "legacy"),
            ("deviceio.close", APP),
            ("openxr.close", BACKGROUND),
            ("no_data.close", APP),
            ("host.close", APP),
        ]
        assert reader_threads() == []

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("phase", PHASES)
    def test_ctrl_c_while_waiting(self, phase, mode, rig, monkeypatch):
        session = _enter_in_phase(rig, monkeypatch, mode, phase)
        if phase == "inside an attempt":
            assert rig.leave_while_held(lambda: _interrupt_the_body(session))
        else:
            _interrupt_the_body(session)
        assert not session._in_context
        assert session._headset_wait is None
        assert _threads(BACKGROUND) == _threads(WORKER) == []
        closed_on = [openxr.closed_on for openxr in rig.runtime.sessions]
        assert closed_on == ([BACKGROUND] if phase == "opened" else [])
        rig.live_run.assert_not_called()

    @pytest.mark.parametrize("mode", MODES)
    def test_ctrl_c_during_the_join_still_unwinds(self, mode, rig, monkeypatch):
        session = _enter_in_phase(
            rig, monkeypatch, mode, PHASES[2], robot_description=DESCRIPTION
        )
        rig.runtime.headset.set()
        interrupted = session._headset_wait._thread
        interrupted.join = _once(KeyboardInterrupt())
        with pytest.raises(KeyboardInterrupt):
            session.__exit__(None, None, None)
        del interrupted.join
        # The rest of the exit is done.
        assert not session._in_context
        assert session._headset_wait is None
        assert rig.journal == [("no_data.close", APP)]
        assert not session.robot_state_active
        assert robot_state_files(rig.runtime_dir) == ["robot_state.sock.lock"]
        assert reader_threads() == []
        assert _threads(WORKER) == []
        assert interrupted.is_alive()

        # The thread ends by itself and closes the session it gets.
        rig.runtime.gate.set()
        assert _join(BACKGROUND)
        (openxr,) = rig.runtime.sessions
        assert (openxr.made_on, openxr.closed_on) == (BACKGROUND, BACKGROUND)

    def test_a_failed_entry_after_the_thread_started_rolls_back(self, rig, monkeypatch):
        def failing(monitor, *args, **kwargs):
            raise RuntimeError("refresh failed")

        session = TeleopSession(_config(robot_description=DESCRIPTION))
        with monkeypatch.context() as patched:
            patched.setattr(StatusMonitor, "refresh", failing)
            with pytest.raises(RuntimeError, match=r"^refresh failed$"):
                session.__enter__()
        # The entry started the thread, and its rollback ended it.
        assert len(rig.waits) == 1
        assert _threads(BACKGROUND) == []
        assert rig.journal == [("no_data.close", APP)]
        assert not session._in_context
        assert (session._headset_wait, session._live_start) == (None, None)
        assert session.oxr_session is None
        assert not session.robot_state_active
        assert robot_state_files(rig.runtime_dir) == ["robot_state.sock.lock"]
        assert reader_threads() == []

        with session:
            assert session.step() == NO_DATA
            assert session.robot_state_active
            assert len(_threads(BACKGROUND)) == 1

    def test_many_rounds_of_exit_inside_an_attempt(self, rig, fine_switching):
        """The held attempt returns at a moment around exit: no headset, a headset, another error."""
        chance = random.Random(11)
        rig.retry_s = 0.002
        error = RuntimeError(RUNTIME_STOPPED)
        outcomes = {"handed over": 0, "raised by step()": 0, "left first": 0}
        for index in range(240):
            rig.next_round()
            answer = ("no headset", "a headset", "another error")[index // 2 % 3]
            rig.runtime.gate = threading.Event()
            session = rig.enter(MODES[index % 2])
            session.step()
            assert rig.runtime.inside.wait(WAIT_S)
            if answer == "a headset":
                rig.runtime.headset.set()
            elif answer == "another error":
                rig.runtime.error = error
            opening = threading.Timer(chance.uniform(0.0, 0.01), rig.runtime.gate.set)
            opening.start()
            end = time.monotonic() + chance.uniform(0.0, 0.01)
            raised = None
            while raised is None and time.monotonic() < end:
                try:
                    session.step()
                except RuntimeError as caught:
                    raised = caught
            handed_over = session.oxr_session is not None
            session.__exit__(None, None, None)
            opening.join()

            assert _threads(BACKGROUND) == _threads(WORKER) == []
            sessions = rig.runtime.sessions
            assert len(sessions) == (answer == "a headset")
            assert [openxr.closed_on for openxr in sessions] == [BACKGROUND] * len(
                sessions
            )
            runs = _named(rig.journal, "deviceio.run")
            assert runs == ([("deviceio.run", APP, False)] if handed_over else [])
            assert rig.journal[-1] == ("no_data.close", APP)
            if raised is not None:
                assert answer == "another error"
                assert str(raised) == OPEN_FAILED.format(error=error)
                assert raised.__cause__ is error
                outcomes["raised by step()"] += 1
            else:
                outcomes["handed over" if handed_over else "left first"] += 1
        assert all(outcomes.values()), outcomes


# ============================================================================
# Failures after entry
# ============================================================================

# How long the worker's step in progress goes on after a Ctrl-C in the step that waits for it.
HELD_S = 0.2
# Where a Ctrl-C arrives inside the step that starts the trackers.
INTERRUPTED = [
    ("the tracker start", "sync"),
    ("the tracker start", "pipelined"),
    ("a plugin start", "sync"),
    ("a plugin start", "pipelined"),
    ("the last statement", "sync"),
    ("the last statement", "pipelined"),
    ("the worker stop", "pipelined"),
]


def _interrupt_the_hand_over(rig, monkeypatch, session, where):
    """Make the next start of the trackers raise KeyboardInterrupt once, inside ``where``."""
    interrupt = _once(KeyboardInterrupt())
    if where == "the tracker start":
        rig.in_run = interrupt
    elif where == "a plugin start":
        rig.plugins.in_start["gamepad"] = interrupt
    elif where == "the last statement":
        push = session._exit_stack.push

        def pushed(stack):
            push(stack)
            interrupt()

        session._exit_stack.push = pushed
    else:
        stop = AsyncRetargetRunner.stop

        def stopped(runner, timeout_s=None):
            try:
                interrupt()
            except KeyboardInterrupt:
                # Inside the join: the worker is told to stop and is not waited for.
                stop(runner, 0.0)
                raise
            return stop(runner, timeout_s)

        monkeypatch.setattr(AsyncRetargetRunner, "stop", stopped)


def _hold_the_worker(rig, session):
    """Keep the pipelined worker inside a step until the pipeline's gate opens."""
    session.pipeline.hold(rig.journal)
    assert _step_until(session, session.pipeline.inside.is_set)


@contextmanager
def _ctrl_c_inside_the_wait_for_the_worker(pipeline):
    """In the block, the step() that waits for the held worker gets Ctrl-C.

    A real signal: it arrives inside Thread.join(), which no stand-in for the join shows.
    The worker's step ends ``HELD_S`` after it.
    """
    interrupted = threading.Event()

    def on_alarm(_signum, frame):
        while frame is not None and not interrupted.is_set():
            if frame.f_code is threading.Thread.join.__code__:
                signal.setitimer(signal.ITIMER_REAL, 0)
                interrupted.set()
                raise KeyboardInterrupt
            frame = frame.f_back

    def end_the_step():
        interrupted.wait(WAIT_S)
        time.sleep(HELD_S)
        pipeline.gate.set()

    helper = threading.Thread(target=end_the_step, name="test-gate")
    handler = signal.signal(signal.SIGALRM, on_alarm)
    helper.start()
    # Again and again until one arrives inside the join.
    signal.setitimer(signal.ITIMER_REAL, 0.02, 0.02)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, handler)
        interrupted.set()
        helper.join(WAIT_S)


def _everything_started_is_stopped(rig):
    journal = rig.journal
    entered = len(_named(journal, "deviceio.enter"))
    return (
        entered == len(_named(journal, "deviceio.close"))
        and all(context.stopped for context in rig.plugins.contexts)
        and len(rig.plugins.contexts) == len(_named(journal, "plugin.stop"))
    )


class TestFailure:
    @pytest.mark.parametrize("mode", MODES)
    def test_another_openxr_error_while_waiting_is_raised_by_every_step(
        self, mode, rig, monkeypatch, tmp_path, caplog
    ):
        error = RuntimeError(RUNTIME_STOPPED)
        text = OPEN_FAILED.format(error=error)
        session = _enter_in_phase(
            rig, monkeypatch, mode, PHASES[2], plugins=[_plugin("gamepad", tmp_path)]
        )
        with (
            caplog.at_level(logging.WARNING, logger="isaaccapture"),
            _leaving(session),
        ):
            rig.runtime.error = error
            # The attempt has not returned: steps go on.
            assert session.step() == NO_DATA
            assert _device(session) == HEADSET_WAITING
            rig.runtime.gate.set()
            # The background thread ends with the failed attempt.
            assert _join(BACKGROUND)
            attempts = len(rig.runtime.calls)

            raised = [_raising_step(session) for _ in range(3)]
            assert [type(caught) for caught in raised] == [RuntimeError] * 3
            assert {str(caught) for caught in raised} == {text}
            assert all(caught.__cause__ is error for caught in raised)
            assert len({id(caught) for caught in raised}) == 3

            status = session.get_status()
            assert len(status.providers) == 2
            for provider in status.providers:
                assert (provider.status, provider.reason, provider.error) == (
                    ProviderState.FAILED,
                    StatusReason.RUNTIME_UPDATE_FAILED,
                    text,
                )
            assert {device.status for device in status.devices} == {
                DeviceState.UNAVAILABLE
            }
            assert session.oxr_session is None
            assert len(rig.runtime.calls) == attempts
            rig.live_run.assert_not_called()
        assert _messages(caplog, logging.ERROR) == [text]
        assert _messages(caplog, logging.WARNING) == []
        assert rig.journal == [("no_data.close", APP)]

    def test_the_cause_does_not_grow_over_1000_failing_steps(self, rig):
        def depth(error):
            count, traceback = 0, error.__traceback__
            while traceback is not None:
                count, traceback = count + 1, traceback.tb_next
            return count

        error = RuntimeError(RUNTIME_STOPPED)
        with TeleopSession(_config()) as session:
            rig.runtime.error = error
            assert _join(BACKGROUND)
            first = _raising_step(session)
            depths = (depth(error), depth(first))
            for _ in range(1000):
                later = _raising_step(session)
            assert later is not first
            assert later.__cause__ is error
            assert (depth(error), depth(later)) == depths

    @pytest.mark.parametrize("mode", MODES)
    def test_trackers_that_cannot_start(self, mode, rig, monkeypatch, tmp_path):
        error = ValueError("HandTracker: XR_EXT_hand_tracking is not supported")
        text = TRACKERS_NOT_STARTED.format(error=error)
        rig.fail_run = error
        session = _enter_in_phase(
            rig, monkeypatch, mode, PHASES[3], plugins=[_plugin("gamepad", tmp_path)]
        )
        no_data = session.deviceio_session
        with _leaving(session):
            raised = [_raising_step(session) for _ in range(2)]
            assert [type(caught) for caught in raised] == [RuntimeError] * 2
            assert {str(caught) for caught in raised} == {text}
            assert all(caught.__cause__ is error for caught in raised)
            assert raised[0] is not raised[1]
            # No second try.
            assert rig.journal == [
                ("openxr.handles", APP),
                ("deviceio.run", APP, False),
            ]
            assert session.oxr_session is None
            assert session.deviceio_session is no_data
            assert (session.plugin_managers, session.plugin_contexts) == ([], [])
            for provider in session.get_status().providers:
                assert (provider.status, provider.reason, provider.error) == (
                    ProviderState.FAILED,
                    StatusReason.RUNTIME_UPDATE_FAILED,
                    text,
                )
            # The OpenXR session stays open, idle, until exit.
            (openxr,) = rig.runtime.sessions
            assert openxr.closed_on is None
            assert [thread.is_alive() for thread in _threads(BACKGROUND)] == [True]
        assert openxr.closed_on == BACKGROUND
        assert rig.journal[2:] == [("openxr.close", BACKGROUND), ("no_data.close", APP)]

    @pytest.mark.parametrize("mode", MODES)
    def test_a_plugin_that_cannot_start(self, mode, rig, monkeypatch, tmp_path):
        error = RuntimeError("Plugin process exited immediately")
        text = PLUGIN_NOT_STARTED.format(error=error)
        rig.plugins.in_start["legacy"] = _always(error)
        session = _enter_in_phase(
            rig,
            monkeypatch,
            mode,
            PHASES[3],
            plugins=[_plugin("gamepad", tmp_path), _plugin("legacy", tmp_path)],
        )
        no_data = session.deviceio_session
        with _leaving(session):
            raised = [_raising_step(session) for _ in range(2)]
            assert [type(caught) for caught in raised] == [RuntimeError] * 2
            assert {str(caught) for caught in raised} == {text}
            assert all(caught.__cause__ is error for caught in raised)
            # The plugin that had started is stopped and the DeviceIO session is closed.
            assert rig.journal == [
                ("openxr.handles", APP),
                ("deviceio.run", APP, False),
                ("deviceio.enter", APP),
                ("plugin.start", APP, "gamepad"),
                ("plugin.start", APP, "legacy"),
                ("plugin.stop", APP, "gamepad"),
                ("deviceio.close", APP),
            ]
            assert (session.plugin_managers, session.plugin_contexts) == ([], [])
            monitored = session._status_monitor._plugin_runtimes
            assert [runtime.context for runtime in monitored] == [None]
            assert session.oxr_session is None
            assert session.deviceio_session is no_data
            for provider in session.get_status().providers:
                assert (provider.status, provider.reason, provider.error) == (
                    ProviderState.FAILED,
                    StatusReason.RUNTIME_UPDATE_FAILED,
                    text,
                )
            (openxr,) = rig.runtime.sessions
            assert openxr.closed_on is None
        assert openxr.closed_on == BACKGROUND

    @pytest.mark.parametrize("failure", ["the open", "the trackers", "a plugin"])
    def test_publishing_goes_on_after_a_failed_hand_over(self, failure, rig, tmp_path):
        config = _config(
            robot_description=DESCRIPTION, plugins=[_plugin("gamepad", tmp_path)]
        )
        with TeleopSession(config) as session, RobotState.connect() as robot:
            session.step()
            assert session.publish_robot_state("arm", [0.25, 0.0, 0.0])
            if failure == "the open":
                rig.runtime.error = RuntimeError(RUNTIME_STOPPED)
                assert _join(BACKGROUND)
            else:
                if failure == "the trackers":
                    rig.fail_run = RuntimeError("XR_ERROR_EXTENSION_NOT_PRESENT")
                else:
                    rig.plugins.in_start["gamepad"] = _always(RuntimeError("exited"))
                rig.runtime.headset.set()
                _opened(session)
            for _ in range(2):
                _raising_step(session)
                assert session.robot_state_active
                assert session.publish_robot_state("arm", [0.5, 0.25, 1.0])
                arm = robot.joints("arm")
                assert (arm.positions, arm.fresh) == ((0.5, 0.25, 1.0), True)
                assert robot.connected
        assert reader_threads() == []

    @pytest.mark.parametrize(("where", "mode"), INTERRUPTED)
    def test_ctrl_c_in_the_hand_over_is_raised_once_and_tried_again(
        self, where, mode, rig, monkeypatch, tmp_path
    ):
        session = _enter_in_phase(
            rig,
            monkeypatch,
            mode,
            PHASES[3],
            plugins=[_plugin("legacy", tmp_path), _plugin("gamepad", tmp_path)],
        )
        no_data = session.deviceio_session
        (openxr,) = rig.runtime.sessions
        _interrupt_the_hand_over(rig, monkeypatch, session, where)
        with _leaving(session):
            _raising_step(session, KeyboardInterrupt)
            # Rolled back, and nothing is kept of the failure.
            assert _everything_started_is_stopped(rig)
            assert session.oxr_session is None
            assert session.deviceio_session is no_data
            assert (session.plugin_managers, session.plugin_contexts) == ([], [])
            assert _device(session) == HEADSET_WAITING
            assert _provider(session) == RUNTIME_UP
            assert _provider(session, "plugin/gamepad") == NOT_STARTED
            assert session._headset_wait.poll() is openxr
            assert openxr.closed_on is None

            mark = len(rig.journal)
            session.step()
            assert rig.journal[mark:] == [
                ("openxr.handles", APP),
                ("deviceio.run", APP, False),
                ("deviceio.enter", APP),
                ("plugin.start", APP, "legacy"),
                ("plugin.start", APP, "gamepad"),
                *STEP_CALLS,
            ]
            assert session.oxr_session is openxr
            assert session.deviceio_session is rig.live
            assert _device(session) == HEADSET_TRACKED
            assert _provider(session, "plugin/gamepad")[:2] == RUNTIME_UP[:2]
            updates = len(_named(rig.journal, "deviceio.update"))
            assert _step_until(
                session,
                lambda: len(_named(rig.journal, "deviceio.update")) >= updates + 3,
            )
            mark = len(rig.journal)
        closing = [entry for entry in rig.journal[mark:] if entry[1] != WORKER]
        assert closing == [
            ("plugin.stop", APP, "gamepad"),
            ("plugin.stop", APP, "legacy"),
            ("deviceio.close", APP),
            ("openxr.close", BACKGROUND),
            ("no_data.close", APP),
        ]
        assert _everything_started_is_stopped(rig)
        assert rig.runtime.sessions == [openxr]

    @pytest.mark.parametrize(("where", "mode"), INTERRUPTED)
    def test_ctrl_c_in_the_hand_over_ends_a_robot_app_that_leaves(
        self, where, mode, rig, monkeypatch, tmp_path
    ):
        session = _enter_in_phase(
            rig,
            monkeypatch,
            mode,
            PHASES[3],
            plugins=[_plugin("legacy", tmp_path), _plugin("gamepad", tmp_path)],
        )
        _interrupt_the_hand_over(rig, monkeypatch, session, where)
        with pytest.raises(KeyboardInterrupt), _leaving(session):
            session.step()
        assert not session._in_context
        assert _threads(BACKGROUND) == _threads(WORKER) == []
        assert _everything_started_is_stopped(rig)
        (openxr,) = rig.runtime.sessions
        assert openxr.closed_on == BACKGROUND
        assert rig.journal[-2:] == [
            ("openxr.close", BACKGROUND),
            ("no_data.close", APP),
        ]
        assert rig.live_run.call_count == (where != "the worker stop")

    def test_ctrl_c_inside_the_wait_for_the_worker_is_tried_again(self, rig, tmp_path):
        pipeline = _Pipeline()
        session = rig.enter(
            "pipelined", pipeline=pipeline, plugins=[_plugin("gamepad", tmp_path)]
        )
        _no_data_steps(session, "pipelined")
        _hold_the_worker(rig, session)
        rig.runtime.headset.set()
        _opened(session)
        with _leaving(session), _ctrl_c_inside_the_wait_for_the_worker(pipeline):
            _raising_step(session, KeyboardInterrupt)
            # Nothing is started beside the worker, and nothing is kept of the interrupt.
            rig.live_run.assert_not_called()
            assert session.oxr_session is None
            assert _device(session) == HEADSET_WAITING

            session.step()
            # The trackers and the plugin start only after the worker's step has ended.
            assert [entry[:2] for entry in rig.journal] == [
                ("pipeline.left", WORKER),
                ("openxr.handles", APP),
                ("deviceio.run", APP),
                ("deviceio.enter", APP),
                ("plugin.start", APP),
                *STEP_CALLS,
            ]
            assert session.last_step_info.ran_synchronously
            assert session.deviceio_session is rig.live
            assert _device(session) == HEADSET_TRACKED
            assert pipeline.threads.count(APP) == 2
        assert _everything_started_is_stopped(rig)

    def test_ctrl_c_inside_the_wait_for_the_worker_ends_a_robot_app_that_leaves(
        self, rig
    ):
        pipeline = _Pipeline()
        session = rig.enter("pipelined", pipeline=pipeline)
        _no_data_steps(session, "pipelined")
        _hold_the_worker(rig, session)
        rig.runtime.headset.set()
        _opened(session)
        runner = session._async_runner
        with _ctrl_c_inside_the_wait_for_the_worker(pipeline):
            with pytest.raises(KeyboardInterrupt), _leaving(session):
                session.step()
            # Exit returns once the worker's step has ended, on a session still open.
            assert rig.journal == [
                ("pipeline.left", WORKER),
                ("openxr.close", BACKGROUND),
                ("no_data.close", APP),
            ]
        assert not session._in_context
        runner.raise_if_failed()
        assert _wait_for(lambda: _threads(WORKER) == [], WAIT_S)
        rig.live_run.assert_not_called()

    def test_a_failed_worker_is_reported_and_nothing_is_handed_over(self, rig):
        pipeline = _Pipeline()
        boom = ValueError("the pipeline broke once")
        with TeleopSession(_config("pipelined", pipeline=pipeline)) as session:
            _no_data_steps(session, "pipelined")
            # The worker fails in a step, and the headset arrives before step() has told.
            pipeline.error = boom
            raised = []
            with suppress(AsyncRetargetWorkerError):
                session.step()
            assert pipeline.raised.wait(WAIT_S)
            pipeline.error = None
            rig.runtime.headset.set()
            _opened(session)

            while len(raised) < 3:
                raised.append(_raising_step(session, AsyncRetargetWorkerError))
            assert all(caught.__cause__ is boom for caught in raised)
            assert session.last_step_info.worker_exception is raised[-1]
            assert rig.journal == []
            assert session.oxr_session is None
            assert session._headset_wait is not None
            (openxr,) = rig.runtime.sessions
            assert openxr.closed_on is None
        assert openxr.closed_on == BACKGROUND
        rig.live_run.assert_not_called()


# ============================================================================
# Sessions that cannot start without a headset
# ============================================================================


def _rolled_back(rig, session):
    """Nothing of a refused entry is left, and it started no thread."""
    assert not session._in_context
    assert (session._headset_wait, session._live_start) == (None, None)
    assert session.oxr_session is None
    assert rig.waits == []
    assert _threads(BACKGROUND) == []
    assert [thread for thread, _, _ in rig.runtime.calls] == [APP]
    rig.live_run.assert_not_called()
    assert not session.robot_state_active
    assert robot_state_files(rig.runtime_dir) == ["robot_state.sock.lock"]
    assert reader_threads() == []


class TestRefusedAtEntry:
    def test_a_refusal_of_the_no_data_session_keeps_its_text(self, rig):
        refusal = ValueError("NoDataSession: tracker 'HeadTracker' is listed twice")
        session = TeleopSession(
            _config(trackers=[HeadTracker()], robot_description=DESCRIPTION)
        )
        with (
            patch.object(deviceio.NoDataSession, "create", side_effect=refusal),
            pytest.raises(RuntimeError) as raised,
        ):
            session.__enter__()
        assert type(raised.value) is RuntimeError
        assert raised.value.__cause__ is refusal
        assert str(raised.value) == NEEDS_ONE.format(reason=refusal)
        _rolled_back(rig, session)

    def test_a_missing_required_plugin(self, rig, tmp_path):
        missing = _plugin("nowhere", tmp_path / "missing", required=True)
        session = TeleopSession(
            _config(plugins=[missing], robot_description=DESCRIPTION)
        )
        with pytest.raises(RuntimeError) as raised:
            session.__enter__()
        assert str(raised.value).startswith(
            "Required plugin 'nowhere' has no existing search directories. "
        )
        _rolled_back(rig, session)

        # The text of an entry with a headset at the runtime.
        rig.runtime.headset.set()
        with pytest.raises(RuntimeError) as with_a_headset:
            session.__enter__()
        assert str(with_a_headset.value) == str(raised.value)

    @pytest.mark.parametrize("folder_is", ["missing", "read-only"])
    def test_a_recording_folder_that_cannot_be_written(self, folder_is, rig, tmp_path):
        if folder_is == "read-only" and os.getuid() == 0:
            pytest.skip("root writes to a folder whatever its mode")
        folder = tmp_path / "recordings"
        if folder_is == "read-only":
            folder.mkdir(mode=0o500)
        recording = McapRecordingConfig(str(folder / "session.mcap"), [])
        session = TeleopSession(
            _config(mcap_config=recording, robot_description=DESCRIPTION)
        )
        with pytest.raises(RuntimeError) as raised:
            session.__enter__()
        assert type(raised.value) is RuntimeError
        assert str(raised.value) == NO_RECORDING_FOLDER.format(folder=folder)
        _rolled_back(rig, session)

        # With a folder that can be written the same session enters and waits.
        folder.mkdir(exist_ok=True)
        folder.chmod(0o700)
        with session:
            assert session.step() == NO_DATA
            assert len(_threads(BACKGROUND)) == 1


# ============================================================================
# A message channel and a tracker with metadata
# ============================================================================


class TestEveryTrackerKind:
    """A message channel and a tracker with metadata wait for the headset like any other tracker."""

    @pytest.mark.parametrize(
        "make",
        [
            lambda: MessageChannelTracker(bytes(16), "channel"),
            lambda: TactileTracker("t"),
        ],
        ids=["MessageChannelTracker", "TactileTracker"],
    )
    def test_both_tracker_kinds_enter_and_serve_robot_state(self, make, rig):
        config = _config(trackers=[make()], robot_description=DESCRIPTION)
        with TeleopSession(config) as session:
            assert session.oxr_session is None
            assert type(session.deviceio_session) is deviceio.NoDataSession
            assert session.robot_state_active
            assert len(_threads(BACKGROUND)) == 1
            assert session.step() == NO_DATA

    @pytest.mark.parametrize("mode", MODES)
    def test_a_message_channel_has_no_messages_and_is_disconnected(self, mode, rig):
        source, _sink, pipeline = _channel()
        tracker = source.get_tracker()
        with TeleopSession(_config(mode, pipeline=pipeline)) as session:
            for _ in range(3):
                status = session.step()["status"][0]
                assert status == MessageChannelConnectionStatus.DISCONNECTED
            waiting = session.deviceio_session
            assert tracker.get_status(waiting) == MessageChannelStatus.DISCONNECTED
            assert tracker.get_messages(waiting).data == []
            assert _device(session) == HEADSET_WAITING
            assert [d.id for d in session.get_status().devices] == ["openxr/headset"]
            rig.live_run.assert_not_called()

    def test_a_direct_send_is_refused_while_waiting(self, rig):
        tracker = MessageChannelTracker(bytes(16), "channel")
        with TeleopSession(_config(trackers=[tracker])) as session:
            session.step()
            with pytest.raises(RuntimeError) as raised:
                tracker.send_message(
                    session.deviceio_session, MessageChannelMessages(b"x")
                )
            assert str(raised.value) == (
                "NoDataSession: MessageChannelTracker::send_message: channel is not open"
            )
            assert tracker.get_messages(session.deviceio_session).data == []
            # The refusal is not a failure of the session.
            session.step()
            assert _provider(session) == RUNTIME_UP

    @pytest.mark.parametrize("mode", MODES)
    def test_a_message_sent_while_waiting_goes_out_once_the_channel_is_connected(
        self, mode, rig, connected_channel
    ):
        source, sink, pipeline = _channel(capacity=4)
        with TeleopSession(_config(mode, pipeline=pipeline)) as session:
            for index in range(6):
                _send(sink, b"m%d" % index)
                session.step()
            assert connected_channel == []
            # A queue of four: the two oldest of six are dropped, as while a channel connects.
            assert len(source._outbound_queue) == 4

            _hand_over(rig, session)
            assert _step_until(
                session, lambda: connected_channel == [b"m2", b"m3", b"m4", b"m5"]
            )
            status = session.step()["status"][0]
            assert status == MessageChannelConnectionStatus.CONNECTED
            assert len(source._outbound_queue) == 0

    @pytest.mark.parametrize("mode", MODES)
    def test_a_tracker_with_metadata_has_no_data_and_no_metadata(self, mode, rig):
        tactile = TactileSource(
            "tactile", "tactile/right", num_segments=2, num_taxels=4
        )
        pipeline = OutputCombiner(
            {
                "forces": tactile.output(TactileSource.FORCES),
                "layout": tactile.output(TactileSource.LAYOUT),
            }
        )
        tracker = tactile.get_tracker()
        with TeleopSession(_config(mode, pipeline=pipeline)) as session:
            for _ in range(3):
                result = session.step()
                assert result["forces"].is_none and result["layout"].is_none
            assert tracker.get_data(session.deviceio_session) is None
            assert tracker.get_metadata(session.deviceio_session) is None
            assert _device(session) == HEADSET_WAITING

    @pytest.mark.parametrize("mode", MODES)
    def test_a_tracker_the_robot_app_reads_itself_has_no_data(self, mode, rig):
        tactile = TactileTracker("tactile/right")
        channel = MessageChannelTracker(bytes(16), "channel")
        with TeleopSession(_config(mode, trackers=[tactile, channel])) as session:
            for _ in range(3):
                session.step()
                waiting = session.deviceio_session
                assert tactile.get_data(waiting) is None
                assert tactile.get_metadata(waiting) is None
                assert channel.get_status(waiting) == MessageChannelStatus.DISCONNECTED
                assert channel.get_messages(waiting).data == []
            assert _device(session) == HEADSET_WAITING

    def test_a_source_of_the_robot_app_is_polled_in_every_step(self, rig):
        source = _CountingTactileSource("glove", "tactile/left", 2, 4)
        pipeline = OutputCombiner({"forces": source.output(TactileSource.FORCES)})
        real = {
            name: getattr(TactileTracker, name) for name in ("get_data", "get_metadata")
        }

        def route(name):
            # The stand-in for the live session is not a C++ session: answer for it here.
            def accessor(tracker, session):
                return None if session is rig.live else real[name](tracker, session)

            return patch.object(TactileTracker, name, accessor)

        with (
            route("get_data"),
            route("get_metadata"),
            TeleopSession(_config(pipeline=pipeline)) as session,
        ):
            waiting = session.deviceio_session
            for _ in range(3):
                assert session.step()["forces"].is_none
            assert source.polled_with == [waiting] * 3
            _hand_over(rig, session)
            session.step()
            assert source.polled_with[3:] == [rig.live] * 2

    def test_a_session_entered_twice_starts_disconnected(self, rig, connected_channel):
        source, _sink, pipeline = _channel()
        session = TeleopSession(_config(pipeline=pipeline))
        rig.runtime.headset.set()
        with session:
            assert (
                session.step()["status"][0] == MessageChannelConnectionStatus.CONNECTED
            )
        assert source._last_status == MessageChannelConnectionStatus.CONNECTED

        rig.next_round()
        with session:
            assert len(rig.no_data) == 1
            assert (
                session.step()["status"][0]
                == MessageChannelConnectionStatus.DISCONNECTED
            )
            assert source._last_drained_messages_tracked.data == []

    def test_the_hand_over_starts_both_kinds_as_an_entry_with_a_headset_does(self, rig):
        trackers = [
            HeadTracker(),
            MessageChannelTracker(bytes(16), "channel"),
            TactileTracker("t"),
        ]
        config = _config(trackers=trackers, robot_description=DESCRIPTION)
        with TeleopSession(config) as session:
            assert session.robot_state_active
            assert session.publish_robot_state("arm", [0.25, -0.5, 1.0])
            session.step()
            rig.live_run.assert_not_called()
            _hand_over(rig, session)
            assert session.deviceio_session is rig.live
            assert session.robot_state_active
            waited = rig.live_run.call_args

        rig.next_round()
        rig.runtime.headset.set()
        with TeleopSession(config) as session:
            assert rig.no_data == []
            started = rig.live_run.call_args
        (waited_trackers, _, *waited_rest), waited_kwargs = waited
        (started_trackers, _, *started_rest), started_kwargs = started
        assert list(waited_trackers) == list(started_trackers) == trackers
        assert waited_rest[0] is started_rest[0] is None
        assert type(waited_rest[1]) is type(started_rest[1])
        assert waited_kwargs == started_kwargs == {}

    def test_with_a_headset_no_no_data_session_is_made(self, rig):
        source, _sink, pipeline = _channel()
        tactile = TactileTracker("t")
        for wait in (True, False):
            rig.next_round()
            rig.runtime.headset.set()
            config = _base_config(
                pipeline=pipeline, trackers=[tactile], wait_for_headset=wait
            )
            with TeleopSession(config) as session:
                assert rig.no_data == []
                assert session.deviceio_session is rig.live
                assert _threads(BACKGROUND) == []
                assert list(rig.live_run.call_args.args[0]) == [
                    source.get_tracker(),
                    tactile,
                ]


# ============================================================================
# Entries that do not wait in the background
# ============================================================================


def _program(rig, caplog, mode, **kwargs):
    """Enter with a headset at the runtime, step three times and leave; the calls that makes."""
    rig.next_round()
    rig.runtime.headset.set()
    caplog.clear()
    with (
        caplog.at_level(logging.DEBUG, logger="isaaccapture"),
        TeleopSession(_base_config(mode, **kwargs)) as session,
    ):
        for _ in range(3):
            session.step()
        assert (session._headset_wait, session._live_start) == (None, None)
        assert _threads(BACKGROUND) == []
        assert session.oxr_session is rig.runtime.sessions[0]
        assert session.deviceio_session is rig.live
        assert _device(session) == HEADSET_TRACKED
    (openxr,) = rig.runtime.sessions
    return SimpleNamespace(
        constructed=list(rig.runtime.calls),
        # In pipelined mode the worker runs a number of steps that differs from run to run.
        calls=[entry for entry in rig.journal if entry[1] != WORKER],
        run=_run_arguments(rig),
        log=[
            (record.name, record.levelno, record.getMessage())
            for record in caplog.records
            if record.threadName != WORKER
        ],
        openxr_threads=(openxr.made_on, openxr.closed_on),
        background=(len(rig.waits), len(rig.no_data)),
    )


class TestUnchanged:
    @pytest.fixture
    def kwargs(self, tmp_path):
        tracker = HeadTracker()
        return {
            "trackers": [tracker],
            "plugins": [_plugin("legacy", tmp_path), _plugin("gamepad", tmp_path)],
            "mcap_config": McapRecordingConfig(
                str(tmp_path / "session.mcap"), [(tracker, "head")]
            ),
        }

    @pytest.mark.parametrize("mode", MODES)
    def test_the_default_makes_no_thread_and_no_no_data_session(
        self, mode, rig, caplog, monkeypatch, kwargs
    ):
        refused = MagicMock(
            side_effect=AssertionError("the entry made a background wait")
        )
        monkeypatch.setattr(_headset_wait, "HeadsetWait", refused)
        absent = _program(rig, caplog, mode, **kwargs)
        true = _program(rig, caplog, mode, wait_for_headset=True, **kwargs)
        assert vars(absent) == vars(true)

        # Two positional arguments, on the thread that enters.
        assert true.constructed == [(APP, ("LocalApp", EXTENSIONS), {})]
        steps_here = 3 if mode == "sync" else 1
        assert true.calls == [
            ("openxr.enter", APP),
            ("openxr.handles", APP),
            ("deviceio.run", APP, False),
            ("deviceio.enter", APP),
            ("plugin.start", APP, "legacy"),
            ("plugin.start", APP, "gamepad"),
            *STEP_CALLS * steps_here,
            ("plugin.stop", APP, "gamepad"),
            ("plugin.stop", APP, "legacy"),
            ("deviceio.close", APP),
            ("openxr.close", APP),
        ]
        assert true.openxr_threads == (APP, APP)
        assert true.background == (0, 0)
        refused.assert_not_called()
        assert [text for _, _, text in true.log if "headset" in text.lower()] == []

    @pytest.mark.parametrize("mode", MODES)
    def test_false_with_a_headset_at_the_runtime_starts_at_entry(
        self, mode, rig, caplog, kwargs
    ):
        true = _program(rig, caplog, mode, wait_for_headset=True, **kwargs)
        false = _program(rig, caplog, mode, wait_for_headset=False, **kwargs)
        # One attempt, which does not wait; everything after it is the default's.
        assert false.constructed == [
            (APP, ("LocalApp", EXTENSIONS), {"wait_for_system": False})
        ]
        assert true.constructed == [(APP, ("LocalApp", EXTENSIONS), {})]
        del false.constructed, true.constructed
        assert vars(false) == vars(true)
        assert false.background == (0, 0)
        assert false.openxr_threads == (APP, APP)
