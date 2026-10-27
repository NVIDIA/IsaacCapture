# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TeleopSession in SessionMode.REMOTE.

Most tests replace RemoteSession with a fake. TestLoopbackServer runs a Teleop Session Server in
miniature (RemoteSessionListener + RemoteSessionHost) on a thread, so HELLO, STATUS, READY and
ERROR cross a real connection.
"""

import json
import logging
import os
import signal
import socket
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import isaaccapture
from isaaccapture import oxr
from isaaccapture.deviceio_session import (
    RemotePeerClosedError,
    RemoteSessionHost,
    RemoteSessionListener,
    TrackerVendor,
    VendorConfig,
)
from isaaccapture.deviceio_trackers import (
    ControllerTracker,
    FullBodyTracker,
    HandTracker,
    HapticCommandPushTracker,
    HeadTracker,
    JointSe3PoseTracker,
    KeyboardTracker,
    MessageChannelTracker,
    PluginDeviceStatusTracker,
    TactileTracker,
    TensorPushTracker,
)
from isaaccapture.plugin_manager import PluginCrashException
from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    HeadSource,
    IDeviceIOSink,
    IDeviceIOSource,
    KeyboardSource,
    MessageChannelSource,
    TactileSource,
)
from isaaccapture.retargeting_engine.interface import (
    ComputeContext,
    OutputCombiner,
    TensorGroup,
    TensorGroupType,
)
from isaaccapture.retargeting_engine.tensor_types import FloatType
from isaaccapture.teleop_session_manager import (
    DeviceState,
    PluginConfig,
    ProviderState,
    RetargetingExecutionConfig,
    SessionMode,
    StatusReason,
    TeleopSession,
    TeleopSessionConfig,
)
from isaaccapture.teleop_session_manager.async_retarget_runner import (
    AsyncRetargetRunner,
    StepRequest,
)

ENDPOINT = "tcp://server.example:47600"
CLOSED_UNANSWERED = (
    "Remote session ended before HELLO_ACK: receive failed: connection closed"
)
_VALUE = TensorGroupType("value", [FloatType("value")])


def _value(value: float) -> TensorGroup:
    group = TensorGroup(_VALUE)
    group[0] = float(value)
    return group


class _Source(IDeviceIOSource):
    """A source over a real tracker that never reads the session."""

    def __init__(self, name, tracker, vendor=None):
        self._tracker = tracker
        super().__init__(name, vendor=vendor)

    def get_tracker(self):
        return self._tracker

    def poll_tracker(self, deviceio_session):
        return {"value": _value(0.0)}

    def input_spec(self):
        return {"value": _VALUE}

    def output_spec(self):
        return {"value": _VALUE}

    def _compute_fn(self, inputs, outputs, context):
        outputs["value"][0] = inputs["value"][0]


class _Sink(IDeviceIOSink):
    def __init__(self, name, tracker):
        self._tracker = tracker
        super().__init__(name)

    def input_spec(self):
        return {"in": _VALUE}

    def get_tracker(self):
        return self._tracker

    def flush_to_device(self, deviceio_session):
        pass

    def _compute_fn(self, inputs, outputs, context):
        pass


class _Pipeline:
    """Returns the remote tick age `session` reported for the update this step made."""

    def __init__(self, leaf_nodes=(), session=None):
        self._leaf_nodes = list(leaf_nodes)
        self.session = session

    def get_leaf_nodes(self):
        return self._leaf_nodes

    def execute_pipeline(self, inputs, context=None):
        age_ns = None if self.session is None else self.session.tick_age_ns
        return {"age": _value(-1.0 if age_ns is None else age_ns / 1e9)}


def _plugin_entry(name, root_id, state="RUNNING", **fields):
    entry = {
        "plugin_name": name,
        "plugin_root_id": root_id,
        "state": state,
        "reason": "NONE",
        "pid": 4321,
        "exit_code": None,
        "term_signal": None,
        "error_code": None,
        "error": "",
    }
    entry.update(fields)
    return entry


CONNECTED = {"state": "AVAILABLE", "headset_state": "CONNECTED", "reason": "NONE"}
TAKEN_OFF = {
    "state": "AVAILABLE",
    "headset_state": "DISCONNECTED",
    "reason": "TRACKING_UNAVAILABLE",
}
# What the server sends before it has an OpenXR session.
NO_HEADSET = {"state": "", "headset_state": "", "reason": ""}


def _status(stage, message="", plugins=(), headset=CONNECTED):
    return {
        "stage": stage,
        "message": message,
        "plugins": list(plugins),
        "headset": headset,
    }


class FakeRemoteSession:
    """Stands in for deviceio.RemoteSession: no network, scripted answers."""

    def __init__(self):
        self.events = []
        self.trackers = None
        self.max_tick_age_ns = None
        self.topics = ["0/head"]
        self.endpoint = None
        self.hello = None
        self.start_timeouts = []
        self.create_error = None
        self.start_error = None
        self.start_errors = []  # raised by the next start() calls, one each, before start_error
        self.hello_ack_error = None
        self.hello_ack_errors = []  # raised by the next wait_hello_ack() calls, one each
        self.hello_acked = True
        self.hello_ack_timeouts = []
        self.ack = {"session_id": "fake", "server_version": isaaccapture.__version__}
        self.ready = True
        self.ready_error = None
        self.wait_ready_timeouts = []
        self.status = _status("ready")
        self.update_error = None
        self.tick_age_ns = None
        self.next_age_ns = None  # each update() publishes this, then it grows by 1 ms

    def create(self, trackers, max_tick_age_ns):
        self.events.append("create")
        if self.create_error is not None:
            raise self.create_error
        self.trackers = trackers
        self.max_tick_age_ns = max_tick_age_ns
        return self

    def __enter__(self):
        self.events.append("enter")
        return self

    def __exit__(self, *exc):
        self.events.append("exit")
        return False

    def subscribed_topics(self):
        return list(self.topics)

    def start(self, endpoint, hello, connect_timeout_ms):
        self.events.append("start")
        self.endpoint = endpoint
        self.hello = json.loads(hello)
        self.start_timeouts.append(connect_timeout_ms)
        if self.start_errors:
            raise self.start_errors.pop(0)
        if self.start_error is not None:
            raise self.start_error

    def wait_hello_ack(self, timeout_ms):
        self.events.append("hello_ack")
        self.hello_ack_timeouts.append(timeout_ms)
        if self.hello_ack_errors:
            raise self.hello_ack_errors.pop(0)
        if self.hello_ack_error is not None:
            raise self.hello_ack_error
        return self.hello_acked

    def hello_ack(self):
        return json.dumps(self.ack)

    def wait_ready(self, timeout_ms):
        self.wait_ready_timeouts.append(timeout_ms)
        if self.ready_error is not None:
            raise self.ready_error
        ready = self.ready() if callable(self.ready) else self.ready
        if not ready:
            time.sleep(min(timeout_ms, 10) / 1000)
        return ready

    def latest_status(self):
        return None if self.status is None else json.dumps(self.status)

    def update(self):
        self.events.append("update")
        if self.update_error is not None:
            raise self.update_error
        if self.next_age_ns is not None:
            self.tick_age_ns = self.next_age_ns
            self.next_age_ns += 1_000_000


@contextmanager
def mock_remote_dependencies(fake=None):
    """Patch RemoteSession.create with `fake` and every local backend with a mock."""
    fake = fake or FakeRemoteSession()
    with (
        patch(
            "isaaccapture.deviceio.RemoteSession.create", side_effect=fake.create
        ) as create,
        patch("isaaccapture.oxr.OpenXRSession") as oxr_cls,
        patch("isaaccapture.deviceio.DeviceIOSession.run") as live_run,
        patch(
            "isaaccapture.deviceio.DeviceIOSession.get_required_extensions"
        ) as get_extensions,
        patch("isaaccapture.deviceio.ReplaySession.run") as replay_run,
        patch("isaaccapture.plugin_manager.PluginManager") as plugin_manager,
    ):
        yield SimpleNamespace(
            session=fake,
            create=create,
            oxr_cls=oxr_cls,
            live_run=live_run,
            get_extensions=get_extensions,
            replay_run=replay_run,
            plugin_manager=plugin_manager,
        )


def remote_config(pipeline=None, **kwargs):
    kwargs.setdefault("remote_endpoint", ENDPOINT)
    return TeleopSessionConfig(
        app_name="RemoteApp",
        pipeline=pipeline if pipeline is not None else _Pipeline(),
        mode=SessionMode.REMOTE,
        verbose=False,
        **kwargs,
    )


def _plugin(name="manus_hand_plugin", root_id="manus", **kwargs):
    return PluginConfig(
        plugin_name=name, plugin_root_id=root_id, search_paths=[], **kwargs
    )


# ============================================================================
# Configuration
# ============================================================================


class TestRemoteConfigValidation:
    def test_defaults(self):
        config = TeleopSessionConfig(app_name="a", pipeline=_Pipeline())
        assert config.remote_endpoint is None
        assert config.remote_tick_rate_hz == 90.0
        assert config.remote_max_tick_age_s == 0.25
        assert config.remote_connect_timeout_s == 60.0

    def test_remote_mode_accepts_an_endpoint(self):
        config = remote_config(remote_connect_timeout_s=None)
        assert config.mode == SessionMode.REMOTE
        assert config.remote_endpoint == ENDPOINT

    def test_mode_string_is_coerced(self):
        config = remote_config()
        config.mode = "remote"
        config._validate_session_mode()
        assert config.mode is SessionMode.REMOTE

    @pytest.mark.parametrize("endpoint", [None, ""])
    def test_remote_mode_requires_an_endpoint(self, endpoint):
        with pytest.raises(ValueError, match="remote_endpoint is required"):
            remote_config(remote_endpoint=endpoint)

    @pytest.mark.parametrize("mode", [SessionMode.LIVE, SessionMode.REPLAY])
    def test_endpoint_requires_remote_mode(self, mode):
        with pytest.raises(ValueError, match="remote_endpoint requires"):
            TeleopSessionConfig(
                app_name="a",
                pipeline=_Pipeline(),
                mode=mode,
                mcap_config=MagicMock() if mode == SessionMode.REPLAY else None,
                remote_endpoint=ENDPOINT,
            )

    @pytest.mark.parametrize(
        ("field", "match"),
        [
            ("oxr_handles", "oxr_handles cannot be used"),
            ("mcap_config", "mcap_config is not supported"),
            ("joint_publisher", "joint_publisher requires a live session; REMOTE"),
            ("oxr_session_factory", "oxr_session_factory requires LIVE"),
        ],
    )
    def test_remote_mode_rejects_local_openxr_and_recording(self, field, match):
        with pytest.raises(ValueError, match=match):
            remote_config(**{field: MagicMock()})

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("remote_tick_rate_hz", 0),
            ("remote_tick_rate_hz", -90.0),
            ("remote_tick_rate_hz", float("nan")),
            ("remote_tick_rate_hz", float("inf")),
            ("remote_tick_rate_hz", "90"),
            ("remote_tick_rate_hz", True),
            ("remote_max_tick_age_s", 0.0),
            ("remote_connect_timeout_s", 0),
            ("remote_connect_timeout_s", -1.0),
        ],
    )
    def test_numbers_must_be_positive(self, field, value):
        with pytest.raises(ValueError, match=field):
            remote_config(**{field: value})

    def test_entry_revalidates_a_mutated_config(self):
        config = TeleopSessionConfig(app_name="a", pipeline=_Pipeline())
        session = TeleopSession(config)
        config.mode = SessionMode.REMOTE
        with mock_remote_dependencies() as mocks:
            with pytest.raises(ValueError, match="remote_endpoint is required"):
                session.__enter__()
            mocks.create.assert_not_called()
            mocks.oxr_cls.assert_not_called()


# ============================================================================
# Inputs a remote session cannot serve
# ============================================================================


class TestRemoteRejectsUnsupportedInputs:
    def test_keyboard_source(self):
        config = remote_config(_Pipeline([KeyboardSource(name="keys")]))
        with pytest.raises(ValueError, match="KeyboardSource 'keys' is not supported"):
            TeleopSession(config)

    def test_message_channel_source(self):
        tracker = MessageChannelTracker(b"0123456789abcdef")
        source = MessageChannelSource("chat", tracker, deque())
        with pytest.raises(ValueError, match="MessageChannelSource 'chat'"):
            TeleopSession(remote_config(_Pipeline([source])))

    def test_tactile_source(self):
        source = TactileSource("touch", "tactile_right", num_segments=1, num_taxels=2)
        with pytest.raises(ValueError, match="TactileSource 'touch' is not supported"):
            TeleopSession(remote_config(_Pipeline([source])))

    @pytest.mark.parametrize(
        "make_tracker",
        [
            KeyboardTracker,
            lambda: MessageChannelTracker(b"0123456789abcdef"),
            lambda: TactileTracker("t"),
        ],
        ids=["keyboard", "message_channel", "tactile"],
    )
    def test_config_trackers(self, make_tracker):
        config = remote_config(trackers=[make_tracker()])
        with pytest.raises(ValueError, match=r"not supported in SessionMode\.REMOTE"):
            TeleopSession(config)

    def test_rejection_reruns_on_entry(self):
        config = TeleopSessionConfig(
            app_name="a", pipeline=_Pipeline([KeyboardSource(name="keys")])
        )
        session = TeleopSession(config)
        config.mode = SessionMode.REMOTE
        config.remote_endpoint = ENDPOINT
        with mock_remote_dependencies() as mocks:
            with pytest.raises(ValueError, match="KeyboardSource 'keys'"):
                session.__enter__()
            mocks.create.assert_not_called()

    @pytest.mark.parametrize(
        "tracker",
        [TensorPushTracker("c", "t"), HapticCommandPushTracker("c", "t")],
        ids=["tensor_push", "haptic_push"],
    )
    def test_unsupported_tracker_fails_before_any_connection(self, tracker):
        config = remote_config(trackers=[tracker])
        with mock_remote_dependencies() as mocks:
            session = TeleopSession(config)
            with pytest.raises(
                ValueError,
                match=r"TeleopSessionConfig.trackers\[0\]\) is not supported over a remote",
            ):
                session.__enter__()
            mocks.create.assert_not_called()


# ============================================================================
# Entering a remote session
# ============================================================================


class TestRemoteSessionEnter:
    def test_uses_no_local_openxr_deviceio_or_plugins(self):
        config = remote_config(
            _Pipeline([_Source("head", HeadTracker())]), plugins=[_plugin()]
        )
        with mock_remote_dependencies() as mocks:
            with TeleopSession(config) as session:
                session.step()
                assert session.oxr_session is None
                assert session.deviceio_session is mocks.session
                assert session.plugin_managers == []
            for local in (
                mocks.oxr_cls,
                mocks.live_run,
                mocks.get_extensions,
                mocks.replay_run,
                mocks.plugin_manager,
            ):
                local.assert_not_called()
        assert mocks.session.events[:4] == ["create", "enter", "start", "hello_ack"]
        # One update after READY checks the declared schemas, then one per step.
        assert mocks.session.events[4:] == ["update", "update", "exit"]

    def test_hello_describes_trackers_topics_plugins_and_vendors(self, caplog):
        head = _Source("head", HeadTracker())
        hands = _Source("hands", HandTracker())
        body = _Source(
            "body",
            FullBodyTracker(),
            vendor=TrackerVendor("body.pico-xr", {"mode": "full"}),
        )
        controller_tracker = ControllerTracker()
        controllers = _Source("controllers", controller_tracker)
        gloves = JointSe3PoseTracker("manus_sensors_left", 2048)
        pipeline = _Pipeline([head, hands, body, controllers])
        config = remote_config(
            pipeline,
            sinks=[
                _Sink("rumble", controller_tracker),  # shared with a source: kept
                _Sink("haptics", HapticCommandPushTracker("c", "t")),  # sink only
            ],
            trackers=[gloves, head.get_tracker(), PluginDeviceStatusTracker("x/s")],
            plugins=[
                _plugin(plugin_args=["--side=left", 3], required=True),
                _plugin("spacemouse", "mouse", enabled=False),
                _plugin("gamepad", "pad"),
            ],
            remote_tick_rate_hz=120,
            remote_max_tick_age_s=0.1,
        )
        with mock_remote_dependencies() as mocks:
            fake = mocks.session
            fake.topics = ["0/head", "4/joint_se3_pose_tracked"]
            with caplog.at_level(logging.WARNING), TeleopSession(config):
                pass

        assert [name for _, name in fake.trackers] == ["0", "1", "2", "3", "4"]
        assert [tracker for tracker, _ in fake.trackers] == [
            head.get_tracker(),
            hands.get_tracker(),
            body.get_tracker(),
            controller_tracker,
            gloves,
        ]
        assert fake.max_tick_age_ns == 100_000_000
        assert fake.endpoint == ENDPOINT
        # One TCP connect attempt is capped at 2 s; the HELLO_ACK wait gets the rest of 60 s.
        assert fake.start_timeouts == [2000]
        assert 59_000 <= fake.hello_ack_timeouts[0] <= 60_000
        assert fake.hello == {
            "protocol": 1,
            "isaaccapture_version": isaaccapture.__version__,
            "app_name": "RemoteApp",
            "tick_rate_hz": 120.0,
            "trackers": [
                {"name": "0", "type": "HeadTracker", "params": {}, "vendor": None},
                {"name": "1", "type": "HandTracker", "params": {}, "vendor": None},
                {
                    "name": "2",
                    "type": "FullBodyTracker",
                    "params": {},
                    "vendor": {"id": "body.pico-xr", "params": {"mode": "full"}},
                },
                {
                    "name": "3",
                    "type": "ControllerTracker",
                    "params": {},
                    "vendor": None,
                },
                {
                    "name": "4",
                    "type": "JointSe3PoseTracker",
                    "params": {
                        "collection_id": "manus_sensors_left",
                        "max_flatbuffer_size": 2048,
                    },
                    "vendor": None,
                },
            ],
            "topics": ["0/head", "4/joint_se3_pose_tracked"],
            "plugins": [
                {
                    "plugin_name": "manus_hand_plugin",
                    "plugin_root_id": "manus",
                    "plugin_args": ["--side=left", "3"],
                    "required": True,
                },
                {
                    "plugin_name": "gamepad",
                    "plugin_root_id": "pad",
                    "plugin_args": [],
                    "required": False,
                },
            ],
        }
        assert (
            "sink data, e.g. haptics, is dropped in remote mode until it is supported"
            in caplog.text.lower()
        )

    def test_polls_wait_ready_until_ready(self):
        polls = iter([False, False, True])
        with mock_remote_dependencies() as mocks:
            mocks.session.ready = lambda: next(polls)
            with TeleopSession(remote_config(remote_connect_timeout_s=None)):
                pass
        assert mocks.session.start_timeouts == [2000]
        assert mocks.session.hello_ack_timeouts == [-1]
        assert mocks.session.wait_ready_timeouts == [100, 100, 100]

    @pytest.mark.parametrize(
        ("status", "stage"),
        [
            (None, "no status from the server yet"),
            (_status("connecting_openxr"), "connecting_openxr"),
            (
                _status("starting_plugins", "starting manus_hand_plugin"),
                r"starting_plugins \(starting manus_hand_plugin\)",
            ),
            (_status("waiting_for_headset"), "waiting_for_headset"),
        ],
    )
    def test_timeout_names_the_last_stage(self, status, stage):
        with mock_remote_dependencies() as mocks:
            mocks.session.ready = False
            mocks.session.status = status
            session = TeleopSession(remote_config(remote_connect_timeout_s=0.2))
            start = time.monotonic()
            with pytest.raises(
                TimeoutError,
                match=rf"^Teleop Session Server at {ENDPOINT} was not ready after 0.2 s: {stage}$",
            ):
                session.__enter__()
            assert time.monotonic() - start < 2.0
        assert mocks.session.events[-1] == "exit"
        assert max(mocks.session.wait_ready_timeouts) <= 100

    def test_no_hello_ack_by_the_deadline_is_server_did_not_answer(self):
        with mock_remote_dependencies() as mocks:
            mocks.session.hello_acked = False
            session = TeleopSession(remote_config())
            with pytest.raises(
                TimeoutError, match=r"was not ready after 60 s: server did not answer$"
            ):
                session.__enter__()
        assert mocks.session.events == ["create", "enter", "start", "hello_ack", "exit"]

    def test_a_refused_connection_is_retried_until_the_deadline(self, caplog):
        refused = RuntimeError(
            "cannot connect to server.example:47600: connect: Connection refused"
        )
        with mock_remote_dependencies() as mocks:
            mocks.session.start_error = refused
            session = TeleopSession(remote_config(remote_connect_timeout_s=1.2))
            start = time.monotonic()
            with (
                caplog.at_level(logging.INFO),
                pytest.raises(
                    TimeoutError,
                    match=r"was not ready after 1.2 s: server did not answer \(cannot "
                    r"connect to server.example:47600: connect: Connection refused\)$",
                ) as raised,
            ):
                session.__enter__()
            elapsed = time.monotonic() - start
        assert raised.value.__cause__ is refused
        assert 1.1 < elapsed < 2.0
        # About every 0.5 s, none so close to the deadline that it could only time out.
        assert set(mocks.session.events[2:-1]) == {"start"}
        assert 3 <= len(mocks.session.events[2:-1]) <= 5
        assert 1100 <= mocks.session.start_timeouts[0] <= 1200
        assert min(mocks.session.start_timeouts) >= 100
        assert mocks.session.events[-1] == "exit"
        waiting = [
            r.getMessage()
            for r in caplog.records
            if r.getMessage().startswith("Waiting for the Teleop Session Server")
        ]
        assert waiting == [
            f"Waiting for the Teleop Session Server at {ENDPOINT} ({refused})"
        ]

    def test_connects_once_the_server_accepts(self):
        refused = RuntimeError("cannot connect to server.example:47600: timed out")
        with mock_remote_dependencies() as mocks:
            mocks.session.start_errors = [refused, refused]
            start = time.monotonic()
            with TeleopSession(remote_config()):
                pass
        assert 0.9 < time.monotonic() - start < 2.0
        assert mocks.session.events[:6] == [
            "create",
            "enter",
            "start",
            "start",
            "start",
            "hello_ack",
        ]

    def test_a_server_that_closes_unanswered_is_retried_until_it_answers(self, caplog):
        closed = RemotePeerClosedError(CLOSED_UNANSWERED)
        with mock_remote_dependencies() as mocks:
            fake = mocks.session
            fake.hello_ack_errors = [closed, closed]
            start = time.monotonic()
            with caplog.at_level(logging.INFO), TeleopSession(remote_config()):
                pass
        # Two closed attempts, about 0.5 s apart from their starts.
        assert 0.9 < time.monotonic() - start < 2.0
        assert fake.events[2:8] == ["start", "hello_ack"] * 3
        waiting = [
            r.getMessage()
            for r in caplog.records
            if r.getMessage().startswith("Waiting for the Teleop Session Server")
        ]
        assert waiting == [
            f"Waiting for the Teleop Session Server at {ENDPOINT} ({CLOSED_UNANSWERED})"
        ]

    def test_refused_and_closed_attempts_log_waiting_once(self, caplog):
        refused = RuntimeError("cannot connect to server.example:47600: timed out")
        with mock_remote_dependencies() as mocks:
            fake = mocks.session
            fake.start_errors = [refused]
            fake.hello_ack_errors = [RemotePeerClosedError(CLOSED_UNANSWERED)]
            with caplog.at_level(logging.INFO), TeleopSession(remote_config()):
                pass
        assert fake.events[2:6] == ["start", "start", "hello_ack", "start"]
        waiting = [
            r.getMessage()
            for r in caplog.records
            if r.getMessage().startswith("Waiting for the Teleop Session Server")
        ]
        assert waiting == [
            f"Waiting for the Teleop Session Server at {ENDPOINT} ({refused})"
        ]

    def test_a_server_that_keeps_closing_is_retried_until_the_deadline(self):
        closed = RemotePeerClosedError(CLOSED_UNANSWERED)
        with mock_remote_dependencies() as mocks:
            fake = mocks.session
            fake.hello_ack_error = closed
            session = TeleopSession(remote_config(remote_connect_timeout_s=1.2))
            start = time.monotonic()
            with pytest.raises(
                TimeoutError,
                match=r"was not ready after 1.2 s: server did not answer \(Remote session "
                r"ended before HELLO_ACK: receive failed: connection closed\)$",
            ) as raised:
                session.__enter__()
            elapsed = time.monotonic() - start
        assert raised.value.__cause__ is closed
        assert 1.1 < elapsed < 2.0
        attempts = fake.events[2:-1]
        assert attempts == ["start", "hello_ack"] * (len(attempts) // 2)
        assert 3 <= len(attempts) // 2 <= 5
        assert fake.events[-1] == "exit"

    @pytest.mark.parametrize(
        ("stage", "text"),
        [
            ("start", "WebSocket upgrade to server.example:47601/: send failed"),
            (
                "hello_ack",
                "WebSocket upgrade to server.example:47601/: connection closed before "
                "the answer",
            ),
        ],
    )
    def test_a_websocket_server_that_closes_unanswered_is_retried(self, stage, text):
        with mock_remote_dependencies() as mocks:
            fake = mocks.session
            getattr(fake, f"{stage}_errors").append(RemotePeerClosedError(text))
            with TeleopSession(remote_config()):
                pass
        assert fake.events.count("start") == 2

    @pytest.mark.parametrize(
        "text",
        [
            "busy: another robot app is connected to this Teleop Session Server",
            "the Teleop Session Server is shutting down",
            # A server's text that looks like a connect error is still the server's text.
            "cannot connect to the glove: no device",
        ],
    )
    def test_a_server_refusal_is_raised_at_once(self, text):
        with mock_remote_dependencies() as mocks:
            mocks.session.hello_ack_error = RuntimeError(text)
            session = TeleopSession(remote_config())
            with pytest.raises(RuntimeError) as raised:
                session.__enter__()
        assert type(raised.value) is RuntimeError
        assert str(raised.value) == text
        assert mocks.session.events == ["create", "enter", "start", "hello_ack", "exit"]

    @pytest.mark.parametrize("stage", ["start", "hello_ack"])
    def test_a_failed_websocket_upgrade_is_server_did_not_answer(self, stage):
        error = RuntimeError(
            "WebSocket upgrade to server.example:47601/ refused: HTTP/1.1 404 Not Found"
        )
        with mock_remote_dependencies() as mocks:
            setattr(mocks.session, f"{stage}_error", error)
            session = TeleopSession(remote_config())
            with pytest.raises(
                TimeoutError,
                match=r"was not ready: server did not answer \(WebSocket upgrade to "
                r"server.example:47601/ refused: HTTP/1.1 404 Not Found\)$",
            ) as raised:
                session.__enter__()
        assert raised.value.__cause__ is error
        assert mocks.session.events.count("start") == 1

    def test_other_connect_errors_are_raised_as_they_are(self):
        error = RuntimeError(
            "cannot resolve 'server.example': Name or service not known"
        )
        with mock_remote_dependencies() as mocks:
            mocks.session.start_error = error
            session = TeleopSession(remote_config())
            with pytest.raises(RuntimeError) as raised:
                session.__enter__()
        assert raised.value is error
        assert mocks.session.events == ["create", "enter", "start", "exit"]

    @pytest.mark.parametrize("stage", ["hello_ack", "wait_ready"])
    def test_server_error_text_is_raised_verbatim(self, stage):
        text = "Required plugin 'manus_hand_plugin' was not discovered on the server"
        with mock_remote_dependencies() as mocks:
            if stage == "hello_ack":
                mocks.session.hello_ack_error = RuntimeError(text)
            else:
                mocks.session.ready_error = RuntimeError(text)
            session = TeleopSession(remote_config())
            with pytest.raises(RuntimeError) as raised:
                session.__enter__()
        assert type(raised.value) is RuntimeError
        assert str(raised.value) == text

    @pytest.mark.parametrize(
        ("failure", "events"),
        [
            ("create", ["create"]),
            ("start", ["create", "enter", "start", "exit"]),
            ("hello_ack", ["create", "enter", "start", "hello_ack", "exit"]),
            ("ready", ["create", "enter", "start", "hello_ack", "exit"]),
            ("update", ["create", "enter", "start", "hello_ack", "update", "exit"]),
        ],
    )
    def test_failed_entry_rolls_back_the_remote_session_once(self, failure, events):
        with mock_remote_dependencies() as mocks:
            fake = mocks.session
            setattr(fake, f"{failure}_error", RuntimeError(f"{failure} failed"))
            session = TeleopSession(remote_config())
            with pytest.raises(RuntimeError, match=f"{failure} failed"):
                session.__enter__()
            assert fake.events == events
            assert session.deviceio_session is None
            assert session._remote_session is None
            assert not session._in_context

            fake.events.clear()
            setattr(fake, f"{failure}_error", None)
            with session:
                pass
            assert fake.events == [
                "create",
                "enter",
                "start",
                "hello_ack",
                "update",
                "exit",
            ]


# ============================================================================
# Steps
# ============================================================================


class TestRemoteStep:
    def test_sync_step_reports_the_remote_tick_age(self):
        with mock_remote_dependencies() as mocks:
            fake = mocks.session
            fake.next_age_ns = 7_000_000
            pipeline = _Pipeline([_Source("head", HeadTracker())], session=fake)
            with TeleopSession(remote_config(pipeline)) as session:
                for _ in range(3):
                    result = session.step()
                    info = session.last_step_info
                    assert info.ran_synchronously
                    assert info.remote_tick_age_s == pytest.approx(result["age"][0])
                assert info.remote_tick_age_s == pytest.approx(0.010)

                fake.next_age_ns = None
                fake.tick_age_ns = None  # stale: update() published no samples
                session.step()
                assert session.last_step_info.remote_tick_age_s is None

                fake.tick_age_ns = -2_000_000  # clock estimate slightly ahead
                session.step()
                assert session.last_step_info.remote_tick_age_s == 0.0

    def test_pipelined_step_reports_the_tick_age_of_the_returned_frame(self):
        with mock_remote_dependencies() as mocks:
            fake = mocks.session
            fake.next_age_ns = 1_000_000
            pipeline = _Pipeline([_Source("head", HeadTracker())], session=fake)
            config = remote_config(
                pipeline,
                retargeting_execution=RetargetingExecutionConfig(mode="pipelined"),
            )
            with TeleopSession(config) as session:
                worker_frames = 0
                for _ in range(20):
                    result = session.step()
                    info = session.last_step_info
                    assert info.remote_tick_age_s == pytest.approx(result["age"][0])
                    worker_frames += not info.ran_synchronously
                    time.sleep(0.002)
                assert worker_frames > 0
                assert info.remote_tick_age_s > 0.001

    def test_update_failure_marks_the_session_failed(self):
        with mock_remote_dependencies() as mocks:
            fake = mocks.session
            fake.status = _status("ready", plugins=[_plugin_entry("gamepad", "pad")])
            with TeleopSession(remote_config(plugins=[_plugin("gamepad", "pad")])) as s:
                s.step()
                provider = s.get_provider_status("plugin/pad")
                assert provider.status == ProviderState.AVAILABLE

                fake.update_error = RuntimeError("Remote session lost: no messages")
                with pytest.raises(RuntimeError, match="Remote session lost"):
                    s.step()
                provider = s.get_provider_status("plugin/pad")
                assert provider.status == ProviderState.FAILED
                assert provider.reason == StatusReason.RUNTIME_UPDATE_FAILED


class TestAsyncRunnerTickAge:
    def _frame(self, step_fn):
        runner = AsyncRetargetRunner(step_fn, RetargetingExecutionConfig("pipelined"))
        runner.start()
        try:
            runner.submit(
                StepRequest(
                    frame_id=1,
                    external_inputs=None,
                    graph_time=None,
                    execution_events=None,
                    submitted_time_s=time.monotonic(),
                )
            )
            return runner.wait_for_frame(1, timeout_s=5.0)
        finally:
            runner.stop()

    def test_executor_may_return_the_age_as_a_third_item(self):
        frame = self._frame(lambda request: ({}, ComputeContext(), 0.05))
        assert frame.remote_tick_age_s == 0.05

    def test_two_item_executors_still_work(self):
        frame = self._frame(lambda request: ({}, ComputeContext()))
        assert frame.remote_tick_age_s is None


# ============================================================================
# Plugins started by the server
# ============================================================================


class TestRemotePlugins:
    def test_failure_surfaces_on_the_next_60_frame_health_check(self):
        running = _status(
            "ready", plugins=[_plugin_entry("manus_hand_plugin", "manus")]
        )
        exited = _status(
            "running",
            plugins=[
                _plugin_entry(
                    "manus_hand_plugin",
                    "manus",
                    state="EXITED",
                    reason="NONZERO_EXIT",
                    exit_code=3,
                    error="Plugin process unexpectedly exited with code 3",
                )
            ],
        )
        with mock_remote_dependencies() as mocks:
            fake = mocks.session
            fake.status = running
            with TeleopSession(remote_config(plugins=[_plugin()])) as session:
                assert len(session.plugin_contexts) == 1
                assert [p.id for p in session.get_status().providers] == [
                    "plugin/manus",
                    "openxr/runtime",
                ]
                assert [d.id for d in session.get_status().devices] == [
                    "openxr/headset"
                ]
                session.step()  # frame 0 checks health

                fake.status = exited
                for _ in range(59):
                    session.step()
                # The status monitor already shows the failure; the health check raises it.
                provider = session.get_provider_status("plugin/manus")
                assert provider.status == ProviderState.FAILED
                assert provider.reason == StatusReason.PROCESS_EXITED
                assert provider.exit_code == 3
                with pytest.raises(
                    PluginCrashException,
                    match=r"^Plugin 'manus_hand_plugin' on the Teleop Session Server: "
                    "Plugin process unexpectedly exited with code 3$",
                ):
                    session.step()

    def test_a_clean_exit_is_a_failure(self):
        with mock_remote_dependencies() as mocks:
            mocks.session.status = _status(
                "running",
                plugins=[
                    _plugin_entry(
                        "gamepad",
                        "pad",
                        state="EXITED",
                        reason="CLEAN_EXIT",
                        exit_code=0,
                    )
                ],
            )
            with TeleopSession(remote_config(plugins=[_plugin("gamepad", "pad")])) as s:
                with pytest.raises(
                    PluginCrashException, match=r"EXITED \(CLEAN_EXIT\)"
                ):
                    s.step()

    def test_only_reported_plugins_get_proxies(self):
        with mock_remote_dependencies() as mocks:
            mocks.session.status = _status("ready")
            with TeleopSession(remote_config(plugins=[_plugin()])) as session:
                assert session.plugin_contexts == []
                assert [p.id for p in session.get_status().providers] == [
                    "openxr/runtime"
                ]
                session.step()


# ============================================================================
# The server's headset
# ============================================================================


def _headset(session):
    return session.get_device_status("openxr/headset")


class TestRemoteHeadsetStatus:
    def test_the_headset_follows_the_server(self):
        with mock_remote_dependencies() as mocks:
            fake = mocks.session
            with TeleopSession(remote_config()) as session:
                assert [p.id for p in session.get_status().providers] == [
                    "openxr/runtime"
                ]
                assert _headset(session).status == DeviceState.CONNECTED
                session.step()
                provider = session.get_provider_status("openxr/runtime")
                assert provider.status == ProviderState.AVAILABLE
                headset = _headset(session)
                assert (headset.status, headset.reason) == (
                    DeviceState.CONNECTED,
                    StatusReason.OPENXR_SESSION_READY,
                )

                fake.status = _status("running", headset=TAKEN_OFF)
                session.step()
                headset = _headset(session)
                assert (headset.status, headset.reason) == (
                    DeviceState.DISCONNECTED,
                    StatusReason.NO_CURRENT_DATA,
                )
                assert "headset DISCONNECTED (TRACKING_UNAVAILABLE)" in headset.error
                provider = session.get_provider_status("openxr/runtime")
                assert provider.status == ProviderState.AVAILABLE

                fake.status = _status("running")
                session.step()
                assert _headset(session).status == DeviceState.CONNECTED

    def test_the_headset_follows_the_server_before_the_first_step(self):
        with mock_remote_dependencies() as mocks:
            mocks.session.status = _status("ready", headset=TAKEN_OFF)
            with TeleopSession(remote_config()) as session:
                headset = _headset(session)
                assert (headset.status, headset.reason) == (
                    DeviceState.DISCONNECTED,
                    StatusReason.NO_CURRENT_DATA,
                )
                assert "headset DISCONNECTED (TRACKING_UNAVAILABLE)" in headset.error
                provider = session.get_provider_status("openxr/runtime")
                assert provider.status == ProviderState.AVAILABLE

    @pytest.mark.parametrize(
        "status",
        [
            _status("running", headset=NO_HEADSET),
            {"stage": "running", "message": "", "plugins": []},
            _status(
                "running",
                headset={"state": "AVAILABLE", "headset_state": "ON", "reason": "NONE"},
            ),
            _status("running", headset="CONNECTED"),
        ],
        ids=["empty", "missing", "unknown_name", "not_an_object"],
    )
    def test_a_malformed_headset_entry_is_a_poll_error(self, status):
        with mock_remote_dependencies() as mocks:
            with TeleopSession(remote_config()) as session:
                mocks.session.status = status
                session.step()
                provider = session.get_provider_status("openxr/runtime")
                assert (provider.status, provider.reason) == (
                    ProviderState.FAILED,
                    StatusReason.OPENXR_POLL_ERROR,
                )
                assert "no valid headset entry" in provider.error
                headset = _headset(session)
                assert (headset.status, headset.reason) == (
                    DeviceState.UNAVAILABLE,
                    StatusReason.OPENXR_POLL_ERROR,
                )

    def test_a_lost_server_session_fails_the_provider(self):
        lost = {
            "state": "FAILED",
            "headset_state": "CONNECTED",
            "reason": "SESSION_LOST",
        }
        with mock_remote_dependencies() as mocks:
            with TeleopSession(remote_config()) as session:
                mocks.session.status = _status("running", headset=lost)
                session.step()
                provider = session.get_provider_status("openxr/runtime")
                assert (provider.status, provider.reason) == (
                    ProviderState.FAILED,
                    StatusReason.OPENXR_SESSION_LOST,
                )
                assert _headset(session).status == DeviceState.UNAVAILABLE


# ============================================================================
# Controllers
# ============================================================================

CONTROLLER_IDS = ("openxr/controller/left", "openxr/controller/right")


class _Controllers:
    """A ControllerTracker as TeleopSession sees it, with samples the test sets."""

    def __init__(self):
        self.samples = {"left": None, "right": None}

    def get_name(self):
        return "ControllerTracker"

    def get_left_controller(self, session):
        return self.samples["left"]

    def get_right_controller(self, session):
        return self.samples["right"]


def _tracked_controller():
    pose = SimpleNamespace(is_valid=True)
    return SimpleNamespace(aim_pose=pose, grip_pose=pose)


def _controller(session, side):
    return session.get_device_status(f"openxr/controller/{side}")


class TestRemoteControllerStatus:
    def test_controller_devices_follow_the_controller_samples(self):
        controllers = _Controllers()
        pipeline = _Pipeline([_Source("controllers", controllers)])
        config = remote_config(pipeline, plugins=[_plugin()])
        with mock_remote_dependencies() as mocks:
            mocks.session.status = _status(
                "ready", plugins=[_plugin_entry("manus_hand_plugin", "manus")]
            )
            with TeleopSession(config) as session:
                assert [d.id for d in session.get_status().devices] == [
                    "openxr/headset",
                    *CONTROLLER_IDS,
                ]
                for side in ("left", "right"):
                    device = _controller(session, side)
                    assert (device.status, device.reason, device.error) == (
                        DeviceState.UNKNOWN,
                        StatusReason.NO_CURRENT_DATA,
                        "waiting for the first controller tracking sample",
                    )
                    assert device.provider_id == "openxr/runtime"

                controllers.samples = {
                    "left": _tracked_controller(),
                    "right": _tracked_controller(),
                }
                session.step()
                for side in ("left", "right"):
                    device = _controller(session, side)
                    assert (device.status, device.reason) == (
                        DeviceState.CONNECTED,
                        StatusReason.NONE,
                    )

                # No sample for longer than the tracking-loss delay: inactive, as in live mode.
                controllers.samples["left"] = None
                assert _step_until(
                    session,
                    lambda: (
                        _controller(session, "left").status == DeviceState.DISCONNECTED
                    ),
                )
                left = _controller(session, "left")
                assert left.reason == StatusReason.NO_CURRENT_DATA
                assert left.error == "controller is inactive for at least 1 second"
                assert _controller(session, "right").status == DeviceState.CONNECTED

    def test_no_controller_devices_without_a_controller_source(self):
        with mock_remote_dependencies():
            with TeleopSession(remote_config()) as session:
                assert [d.id for d in session.get_status().devices] == [
                    "openxr/headset"
                ]
                assert _controller(session, "left") is None


# ============================================================================
# Versions and sinks
# ============================================================================


def _messages(caplog, level):
    return [r.getMessage() for r in caplog.records if r.levelno == level]


class TestRemoteServerVersion:
    def test_the_server_version_is_logged(self, caplog):
        with (
            mock_remote_dependencies(),
            caplog.at_level(logging.INFO, logger="isaaccapture"),
            TeleopSession(remote_config()),
        ):
            pass
        assert (
            f"Teleop Session Server at {ENDPOINT} runs Isaac Capture "
            f"{isaaccapture.__version__}"
        ) in _messages(caplog, logging.INFO)
        assert not [m for m in _messages(caplog, logging.WARNING) if "version" in m]

    @pytest.mark.parametrize(
        ("ack", "shown"),
        [
            ({"session_id": "s", "server_version": "0.0.1"}, "0.0.1"),
            ({"session_id": "s"}, "unknown"),
        ],
        ids=["other", "missing"],
    )
    def test_another_server_version_is_a_warning(self, caplog, ack, shown):
        with mock_remote_dependencies() as mocks:
            mocks.session.ack = ack
            with (
                caplog.at_level(logging.INFO, logger="isaaccapture"),
                TeleopSession(remote_config()),
            ):
                pass
        assert (
            f"Teleop Session Server at {ENDPOINT} runs Isaac Capture {shown}"
            in _messages(caplog, logging.INFO)
        )
        assert _messages(caplog, logging.WARNING) == [
            f"Teleop Session Server at {ENDPOINT} runs Isaac Capture {shown}, this "
            f"robot app {isaaccapture.__version__}; run the same version on both "
            "machines"
        ]


class _FlushFailingSink(IDeviceIOSink):
    """A sink whose flush raises, as one writing to a tracker the RemoteSession lacks would."""

    def __init__(self, name, tracker):
        self._tracker = tracker
        self.computed = []
        self.flushes = 0
        super().__init__(name)

    def input_spec(self):
        return {"in": _VALUE}

    def get_tracker(self):
        return self._tracker

    def flush_to_device(self, deviceio_session):
        self.flushes += 1
        raise RuntimeError("the RemoteSession has no HapticCommandPushTracker")

    def _compute_fn(self, inputs, outputs, context):
        self.computed.append(inputs["in"][0])


class TestRemoteSinks:
    @pytest.mark.parametrize("mode", ["sync", "pipelined"])
    def test_sinks_run_but_are_not_flushed(self, mode):
        head = _Source("head", HeadTracker())
        sink = _FlushFailingSink("haptics", HapticCommandPushTracker("c", "t"))
        config = remote_config(
            OutputCombiner({"value": head.output("value")}),
            sinks=[sink.connect({"in": head.output("value")})],
            retargeting_execution=RetargetingExecutionConfig(mode=mode),
        )
        with mock_remote_dependencies(), TeleopSession(config) as session:
            for _ in range(5):
                assert session.step()["value"][0] == 0.0
        assert sink.flushes == 0
        assert sink.computed
        assert mode == "pipelined" or len(sink.computed) == 5


# ============================================================================
# A Teleop Session Server in miniature, over loopback
# ============================================================================

WAIT_MS = 5000
HELLO_ACK = {"session_id": "loopback", "server_version": "0"}
PLACEHOLDER_HANDLES = (1, 1, 1, 1)


def _wait_for(predicate, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.005)
    return True


class _MiniServer:
    """Accepts one robot on a thread, checks its HELLO is JSON, then runs `script(server, host)`.

    With `adopt_delay_s` it leaves the accepted connection unanswered that long, as the Teleop
    Session Server does while it hands over from the previous session.
    """

    def __init__(self, script, *, port=0, websocket=False, adopt_delay_s=0.0):
        self.listener = RemoteSessionListener("127.0.0.1", port)
        scheme = "ws" if websocket else "tcp"
        self.endpoint = f"{scheme}://127.0.0.1:{self.listener.port}"
        self.hello = None
        self.error = None
        self.stop_event = threading.Event()
        self.fail_plugin = threading.Event()
        self.take_off_headset = threading.Event()
        self._script = script
        self._websocket = websocket
        self._adopt_delay_s = adopt_delay_s
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            accepted = self.listener.accept(WAIT_MS)
            assert accepted is not None, "no robot connected"
            time.sleep(self._adopt_delay_s)
            with RemoteSessionHost.adopt(accepted[0], self._websocket) as host:
                hello = host.wait_hello(WAIT_MS)
                assert hello is not None, "no HELLO"
                self.hello = json.loads(hello)
                self._script(self, host)
        except Exception as e:  # reported by stop()
            self.error = e

    def idle(self, host):
        while not self.stop_event.is_set() and host.alive:
            time.sleep(0.005)

    def stop(self):
        self.stop_event.set()
        self._thread.join(timeout=10)
        assert not self._thread.is_alive()
        if self.error is not None:
            raise self.error


def _serve_until_stopped(server, host):
    """HELLO_ACK, the startup stages, READY, then a tick every 5 ms."""
    plugins = [_plugin_entry("manus_hand_plugin", "manus")]
    host.send_hello_ack(json.dumps(HELLO_ACK))
    host.send_status(json.dumps(_status("connecting_openxr")))
    host.send_status(json.dumps(_status("starting_plugins", plugins=plugins)))
    host.send_status(json.dumps(_status("waiting_for_headset", plugins=plugins)))
    # The keyboard needs no OpenXR. The robot reads a HeadTracker, so the topic filter drops
    # every keyboard record and each SAMPLES is empty, which still carries its tick time.
    host.start_session(
        [(KeyboardTracker(), "0")],
        oxr.OpenXRSessionHandles(*PLACEHOLDER_HANDLES),
        VendorConfig(),
        server.hello["topics"],
    )
    host.send_ready(json.dumps(_status("ready", plugins=plugins)))
    failed = False
    taken_off = False
    while not server.stop_event.is_set() and host.alive:
        if server.take_off_headset.is_set() and not taken_off:
            taken_off = True
            host.send_status(
                json.dumps(_status("running", plugins=plugins, headset=TAKEN_OFF))
            )
        if server.fail_plugin.is_set() and not failed:
            failed = True
            crashed = _plugin_entry(
                "manus_hand_plugin",
                "manus",
                state="EXITED",
                reason="NONZERO_EXIT",
                exit_code=3,
                error="Plugin process unexpectedly exited with code 3",
            )
            host.send_status(json.dumps(_status("running", plugins=[crashed])))
        host.tick()
        time.sleep(0.005)


def _loopback_config(server, **kwargs):
    kwargs.setdefault("remote_connect_timeout_s", 10.0)
    return TeleopSessionConfig(
        app_name="RemoteLoopback",
        pipeline=_Pipeline([HeadSource(name="head")]),
        mode=SessionMode.REMOTE,
        remote_endpoint=server.endpoint,
        **kwargs,
    )


def _step_until(session, predicate, timeout_s=5.0):
    return _wait_for(lambda: (session.step(), predicate())[1], timeout_s)


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@contextmanager
def _closing_listener(reply=b""):
    """Closes every connection once the robot sent something, after sending `reply`.

    Yields (port, accepted peers).
    """
    stop = threading.Event()
    accepted = []
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(4)
    listener.settimeout(0.05)

    def serve():
        while not stop.is_set():
            try:
                connection, peer = listener.accept()
            except TimeoutError:
                continue
            accepted.append(peer)
            with connection:
                connection.settimeout(WAIT_MS / 1000)
                try:
                    connection.recv(4096)
                    if reply:
                        connection.sendall(reply)
                        connection.shutdown(socket.SHUT_WR)
                        # Until the robot closes, so that no reset discards the reply.
                        while connection.recv(4096):
                            pass
                except OSError:
                    pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield listener.getsockname()[1], accepted
    finally:
        stop.set()
        thread.join(timeout=10)
        listener.close()


def _endpoint_config(endpoint, timeout_s):
    return TeleopSessionConfig(
        app_name="RemoteLoopback",
        pipeline=_Pipeline([HeadSource(name="head")]),
        mode=SessionMode.REMOTE,
        remote_endpoint=endpoint,
        remote_connect_timeout_s=timeout_s,
    )


@contextmanager
def _sigint_after(delay_s):
    """Sends this process SIGINT after `delay_s`; yields a list that gets the send time."""
    sent = []

    def interrupt():
        sent.append(time.monotonic())
        os.kill(os.getpid(), signal.SIGINT)

    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    timer = threading.Timer(delay_s, interrupt)
    timer.start()
    try:
        yield sent
    finally:
        timer.cancel()
        timer.join()
        signal.signal(signal.SIGINT, previous)


@pytest.mark.skipif(
    sys.platform == "win32", reason="remote sessions need POSIX sockets"
)
class TestLoopbackServer:
    def test_ready_session_streams_samples_and_reports_plugin_failure(self):
        server = _MiniServer(_serve_until_stopped)
        config = _loopback_config(
            server,
            remote_tick_rate_hz=120.0,
            plugins=[_plugin(plugin_args=["--side=left"], required=True)],
        )
        try:
            with TeleopSession(config) as session:
                assert server.hello == {
                    "protocol": 1,
                    "isaaccapture_version": isaaccapture.__version__,
                    "app_name": "RemoteLoopback",
                    "tick_rate_hz": 120.0,
                    "trackers": [
                        {
                            "name": "0",
                            "type": "HeadTracker",
                            "params": {},
                            "vendor": None,
                        }
                    ],
                    "topics": ["0/head"],
                    "plugins": [
                        {
                            "plugin_name": "manus_hand_plugin",
                            "plugin_root_id": "manus",
                            "plugin_args": ["--side=left"],
                            "required": True,
                        }
                    ],
                }
                assert _step_until(
                    session,
                    lambda: session.last_step_info.remote_tick_age_s is not None,
                )
                assert 0.0 <= session.last_step_info.remote_tick_age_s < 0.25
                provider = session.get_provider_status("plugin/manus")
                assert provider.status == ProviderState.AVAILABLE

                server.fail_plugin.set()
                with pytest.raises(PluginCrashException, match="exited with code 3"):
                    _step_until(session, lambda: False, timeout_s=10.0)
        finally:
            server.stop()

    def test_server_close_fails_the_next_step(self):
        server = _MiniServer(_serve_until_stopped)
        try:
            config = _loopback_config(server, plugins=[_plugin()])
            with TeleopSession(config) as session:
                session.step()
                server.stop()
                with pytest.raises(RuntimeError, match="Remote session lost"):
                    _step_until(session, lambda: False, timeout_s=10.0)
                provider = session.get_provider_status("plugin/manus")
                assert provider.status == ProviderState.FAILED
                assert provider.reason == StatusReason.RUNTIME_UPDATE_FAILED
        finally:
            server.stop()

    @pytest.mark.parametrize(
        "answer_hello", [True, False], ids=["after_ack", "at_hello"]
    )
    def test_server_error_text_is_raised_verbatim(self, answer_hello):
        text = "Required plugin 'manus_hand_plugin' was not discovered in /opt/plugins"

        def refuse(server, host):
            if answer_hello:
                host.send_hello_ack(json.dumps(HELLO_ACK))
                host.send_status(json.dumps(_status("starting_plugins")))
            host.send_error(text)

        server = _MiniServer(refuse)
        try:
            session = TeleopSession(_loopback_config(server, plugins=[_plugin()]))
            with pytest.raises(RuntimeError) as raised:
                session.__enter__()
            assert str(raised.value) == text
        finally:
            server.stop()

    def test_timeout_names_the_stage_the_server_is_stuck_at(self):
        def stall(server, host):
            host.send_hello_ack(json.dumps(HELLO_ACK))
            host.send_status(
                json.dumps(_status("waiting_for_headset", "put on the headset"))
            )
            server.idle(host)

        server = _MiniServer(stall)
        try:
            session = TeleopSession(
                _loopback_config(server, remote_connect_timeout_s=0.5)
            )
            with pytest.raises(
                TimeoutError,
                match=r"was not ready after 0.5 s: waiting_for_headset "
                r"\(put on the headset\)$",
            ):
                session.__enter__()
        finally:
            server.stop()

    def test_the_headset_status_comes_from_the_server(self):
        server = _MiniServer(_serve_until_stopped)
        try:
            with TeleopSession(_loopback_config(server)) as session:
                session.step()
                assert _headset(session).status == DeviceState.CONNECTED
                server.take_off_headset.set()
                assert _step_until(
                    session,
                    lambda: _headset(session).status == DeviceState.DISCONNECTED,
                )
                provider = session.get_provider_status("openxr/runtime")
                assert provider.status == ProviderState.AVAILABLE
        finally:
            server.stop()

    def test_silent_server_did_not_answer(self):
        server = _MiniServer(lambda server, host: server.idle(host))
        try:
            session = TeleopSession(
                _loopback_config(server, remote_connect_timeout_s=0.5)
            )
            with pytest.raises(
                TimeoutError, match=r"was not ready after 0.5 s: server did not answer$"
            ):
                session.__enter__()
            assert server.hello["topics"] == ["0/head"]
        finally:
            server.stop()

    @pytest.mark.parametrize("scheme", ["tcp", "ws"])
    def test_a_listener_that_never_answers_did_not_answer(self, scheme):
        """The kernel completes the TCP handshake; nothing ever reads or writes."""
        with socket.socket() as silent:
            silent.bind(("127.0.0.1", 0))
            silent.listen(1)
            config = _endpoint_config(
                f"{scheme}://127.0.0.1:{silent.getsockname()[1]}", 2.0
            )
            start = time.monotonic()
            with pytest.raises(
                TimeoutError, match=r"was not ready after 2 s: server did not answer$"
            ):
                TeleopSession(config).__enter__()
            assert 1.9 < time.monotonic() - start < 5.0

    def test_a_refused_connection_is_retried_until_the_deadline(self):
        port = _free_port()
        start = time.monotonic()
        with pytest.raises(
            TimeoutError,
            match=rf"was not ready after 1 s: server did not answer \(cannot connect to "
            rf"127.0.0.1:{port}: connect: Connection refused\)$",
        ):
            TeleopSession(_endpoint_config(f"127.0.0.1:{port}", 1.0)).__enter__()
        assert 0.9 < time.monotonic() - start < 3.0

    def test_a_server_that_starts_late_is_served(self, caplog):
        port = _free_port()
        servers = []
        starter = threading.Timer(
            1.5,
            lambda: servers.append(_MiniServer(_serve_until_stopped, port=port)),
        )
        config = _endpoint_config(f"tcp://127.0.0.1:{port}", 10.0)
        starter.start()
        try:
            start = time.monotonic()
            with (
                caplog.at_level(logging.INFO, logger="isaaccapture"),
                TeleopSession(config) as session,
            ):
                assert time.monotonic() - start >= 1.4
                assert _step_until(
                    session,
                    lambda: session.last_step_info.remote_tick_age_s is not None,
                )
        finally:
            starter.join()
            for server in servers:
                server.stop()
        waiting = [
            m
            for m in _messages(caplog, logging.INFO)
            if m.startswith("Waiting for the Teleop Session Server")
        ]
        assert len(waiting) == 1
        assert f"(cannot connect to 127.0.0.1:{port}" in waiting[0]

    @pytest.mark.parametrize("websocket", [False, True], ids=["tcp", "ws"])
    def test_a_server_that_answers_late_is_served(self, websocket):
        """Past the link's 3 s peer timeout: it applies from the server's first message."""
        server = _MiniServer(
            _serve_until_stopped, websocket=websocket, adopt_delay_s=5.0
        )
        try:
            start = time.monotonic()
            with TeleopSession(_loopback_config(server)) as session:
                assert time.monotonic() - start >= 4.9
                assert _step_until(
                    session,
                    lambda: session.last_step_info.remote_tick_age_s is not None,
                )
        finally:
            server.stop()

    @pytest.mark.parametrize("scheme", ["tcp", "ws"])
    def test_ctrl_c_while_waiting_for_the_server_to_answer(self, scheme):
        with socket.socket() as silent:
            silent.bind(("127.0.0.1", 0))
            silent.listen(1)
            port = silent.getsockname()[1]
            session = TeleopSession(
                _endpoint_config(f"{scheme}://127.0.0.1:{port}", 30.0)
            )
            with _sigint_after(0.5) as sent, pytest.raises(KeyboardInterrupt):
                session.__enter__()
            assert time.monotonic() - sent[0] < 1.0
            assert not session._in_context
            # The rollback closed the connection.
            connection, _ = silent.accept()
            with connection:
                connection.settimeout(5.0)
                while connection.recv(4096):
                    pass

    def test_ctrl_c_while_waiting_for_ready(self):
        def stall(server, host):
            host.send_hello_ack(json.dumps(HELLO_ACK))
            host.send_status(json.dumps(_status("waiting_for_headset")))
            server.idle(host)

        server = _MiniServer(stall)
        try:
            session = TeleopSession(
                _loopback_config(server, remote_connect_timeout_s=30.0)
            )
            with _sigint_after(0.5) as sent, pytest.raises(KeyboardInterrupt):
                session.__enter__()
            assert time.monotonic() - sent[0] < 1.0
            assert not session._in_context
            # The server sees the robot app leave.
            assert _wait_for(lambda: not server._thread.is_alive())
        finally:
            server.stop()

    def test_ctrl_c_while_retrying(self):
        config = _endpoint_config(f"127.0.0.1:{_free_port()}", 30.0)
        with _sigint_after(0.7) as sent, pytest.raises(KeyboardInterrupt):
            TeleopSession(config).__enter__()
        assert time.monotonic() - sent[0] < 1.0

    @pytest.mark.parametrize("websocket", [False, True], ids=["tcp", "ws"])
    def test_a_server_that_dies_with_the_robot_in_its_backlog_is_retried(
        self, websocket, caplog
    ):
        """The backlog's connections are reset; a server started on the port serves."""
        backlog = socket.socket()
        backlog.bind(("127.0.0.1", 0))
        backlog.listen(4)
        port = backlog.getsockname()[1]
        servers = []

        def restart():
            backlog.close()
            servers.append(
                _MiniServer(_serve_until_stopped, port=port, websocket=websocket)
            )

        scheme = "ws" if websocket else "tcp"
        restarter = threading.Timer(1.0, restart)
        restarter.start()
        try:
            with (
                caplog.at_level(logging.INFO, logger="isaaccapture"),
                TeleopSession(
                    _endpoint_config(f"{scheme}://127.0.0.1:{port}", 10.0)
                ) as session,
            ):
                assert _step_until(
                    session,
                    lambda: session.last_step_info.remote_tick_age_s is not None,
                )
        finally:
            restarter.join()
            backlog.close()
            for server in servers:
                server.stop()
        waiting = [
            m
            for m in _messages(caplog, logging.INFO)
            if m.startswith("Waiting for the Teleop Session Server")
        ]
        closed = (
            f"WebSocket upgrade to 127.0.0.1:{port}/: connection closed before the answer"
            if websocket
            else CLOSED_UNANSWERED
        )
        assert waiting == [
            f"Waiting for the Teleop Session Server at {scheme}://127.0.0.1:{port} "
            f"({closed})"
        ]

    @pytest.mark.parametrize("websocket", [False, True], ids=["tcp", "ws"])
    def test_a_server_that_keeps_closing_did_not_answer(self, websocket):
        scheme = "ws" if websocket else "tcp"
        with _closing_listener() as (port, accepted):
            start = time.monotonic()
            closed = (
                rf"WebSocket upgrade to 127.0.0.1:{port}/: connection closed before "
                r"the answer"
                if websocket
                else r"Remote session ended before HELLO_ACK: receive failed: "
                r"connection closed"
            )
            with pytest.raises(
                TimeoutError,
                match=rf"was not ready after 2 s: server did not answer \({closed}\)$",
            ) as raised:
                TeleopSession(
                    _endpoint_config(f"{scheme}://127.0.0.1:{port}", 2.0)
                ).__enter__()
            assert 1.9 < time.monotonic() - start < 4.0
        assert isinstance(raised.value.__cause__, RemotePeerClosedError)
        # About every 0.5 s.
        assert 3 <= len(accepted) <= 6

    @pytest.mark.parametrize("websocket", [False, True], ids=["tcp", "ws"])
    def test_a_server_that_sends_part_of_a_message_is_raised_at_once(self, websocket):
        """Four bytes and a close: not a server that closed before sending anything."""
        scheme = "ws" if websocket else "tcp"
        with _closing_listener(reply=b"ERR\n") as (port, accepted):
            if websocket:
                error_type = TimeoutError
                error = (
                    rf"server did not answer \(WebSocket upgrade to 127\.0\.0\.1:{port}/: "
                    r"no valid HTTP response\)$"
                )
            else:
                error_type = RuntimeError
                error = (
                    r"^Remote session ended before HELLO_ACK: receive failed: "
                    r"incomplete message$"
                )
            start = time.monotonic()
            with pytest.raises(error_type, match=error) as raised:
                TeleopSession(
                    _endpoint_config(f"{scheme}://127.0.0.1:{port}", 10.0)
                ).__enter__()
            assert time.monotonic() - start < 2.0
        assert not isinstance(raised.value, RemotePeerClosedError)
        assert not isinstance(raised.value.__cause__, RemotePeerClosedError)
        assert len(accepted) == 1

    @pytest.mark.parametrize("websocket", [False, True], ids=["tcp", "ws"])
    def test_a_busy_server_is_raised_at_once(self, websocket):
        text = "busy: another robot app is connected to this Teleop Session Server"
        listener = RemoteSessionListener("127.0.0.1", 0)

        def refuse():
            accepted = listener.accept(WAIT_MS)
            if accepted is not None:
                RemoteSessionListener.reject(accepted[0], websocket, text)

        refuser = threading.Thread(target=refuse)
        refuser.start()
        scheme = "ws" if websocket else "tcp"
        session = TeleopSession(
            _endpoint_config(f"{scheme}://127.0.0.1:{listener.port}", 10.0)
        )
        start = time.monotonic()
        try:
            with pytest.raises(RuntimeError) as raised:
                session.__enter__()
        finally:
            refuser.join(timeout=10)
            listener.close()
        assert time.monotonic() - start < 3.0
        assert type(raised.value) is RuntimeError
        assert str(raised.value) == text

    def test_ctrl_c_while_retrying_a_server_that_keeps_closing(self):
        with _closing_listener() as (port, _accepted):
            config = _endpoint_config(f"tcp://127.0.0.1:{port}", 30.0)
            with _sigint_after(0.7) as sent, pytest.raises(KeyboardInterrupt):
                TeleopSession(config).__enter__()
            assert time.monotonic() - sent[0] < 1.0
