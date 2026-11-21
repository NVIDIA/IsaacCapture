# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TeleopSession with wait_for_headset=False: entry that does not wait for a tracked headset.

In SessionMode.REMOTE the first tests replace RemoteSession with a fake; TestLoopback runs a server
end in miniature (RemoteSessionListener + RemoteSessionHost, a RobotStateStore attached) on a
thread, with the robot app's real trackers. In LIVE mode OpenXR and DeviceIO are replaced: no
OpenXR runtime and no headset is used.
"""

import dataclasses
import json
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
from test_remote_mode import (
    CONNECTED,
    ENDPOINT,
    HELLO_ACK,
    NO_HEADSET,
    PLACEHOLDER_HANDLES,
    TAKEN_OFF,
    WAITING_FOR_HEADSET,
    FakeRemoteSession,
    _headset,
    _messages,
    _MiniServer,
    _Pipeline,
    _Source,
    _status,
    _step_until,
    _wait_for,
    mock_remote_dependencies,
    remote_config,
)
from test_remote_robot_state import AGREED, DESCRIPTION, _ack, _Session

from isaaccapture import oxr
from isaaccapture.deviceio_session import VendorConfig
from isaaccapture.deviceio_trackers import HeadTracker, KeyboardTracker
from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    ControllersSource,
    HandsSource,
    HeadSource,
)
from isaaccapture.retargeting_engine.interface import (
    ExecutionState,
    OptionalType,
    OutputCombiner,
)
from isaaccapture.retargeting_engine.tensor_types import ControllerInputIndex
from isaaccapture.robot_state import (
    RobotStateStore,
    default_socket_path,
    records,
)
from isaaccapture.teleop_session_manager import (
    DefaultTeleopStateManager,
    DeviceState,
    ProviderState,
    RetargetingExecutionConfig,
    SessionMode,
    StatusReason,
    TeleopSession,
    TeleopSessionConfig,
    _remote,
    create_bool_selector,
    teleop_session,
)
from isaaccapture.teleop_session_manager.teleop_state_manager_types import bool_signal
from isaaccapture.teleop_session_server import _session as server_session

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="remote sessions need POSIX sockets"
)

IS_READY = f"Teleop Session Server at {ENDPOINT} is ready"
NOT_WAITING = (
    f"Teleop Session Server at {ENDPOINT} answered; not waiting for a tracked headset"
)
CONNECTING = _status("connecting_openxr", WAITING_FOR_HEADSET, headset=NO_HEADSET)
UNKNOWN_NAMES = {"state": "AVAILABLE", "headset_state": "ON", "reason": "NONE"}
# How oxr.OpenXRSession(wait_for_system=False) fails with no headset at the runtime (oxr_session.cpp).
NO_HEADSET_AT_OPENXR = (
    "Failed to get OpenXR system: XR_ERROR_FORM_FACTOR_UNAVAILABLE (-35). The OpenXR runtime "
    "is up but no headset is connected to it, and this session asked for wait_for_system=false "
    "rather than waiting for one. Connect the headset, or check NV_DEVICE_PROFILE matches it. "
    "See docs/source/references/cloudxr.rst, Troubleshooting."
)
NO_HEADSET_AT_RUNTIME = (
    "No headset has reached the OpenXR runtime since it started, and a LIVE session cannot "
    "start without one yet (wait_for_headset=False). Wake a headset once, set "
    "wait_for_headset=True to wait for one, or run a Teleop Session Server on this machine and "
    "use SessionMode.REMOTE."
)
EXTENSIONS = ["XR_EXT_test"]
READER_THREAD = "isaaccapture-robot-state-readers"


@pytest.fixture(autouse=True)
def runtime_dir(tmp_path_factory, monkeypatch):
    """A session that hosts robot state serves its reader socket under this folder."""
    path = tmp_path_factory.mktemp("rt")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(path))
    assert default_socket_path().startswith(str(path))
    return path


class _KeepsHello:
    """Keeps the HELLO as it was sent."""

    hello_text = None

    def start(self, endpoint, hello, connect_timeout_ms):
        self.hello_text = hello
        super().start(endpoint, hello, connect_timeout_ms)


class _Fake(_KeepsHello, FakeRemoteSession):
    pass


class _RobotFake(_KeepsHello, _Session):
    """The fake RemoteSession with the robot-state calls, and a STATUS the test sets."""

    def __init__(self, ack=None):
        super().__init__(_ack(**AGREED) if ack is None else ack)
        self.status = None

    def latest_status(self):
        return None if self.status is None else json.dumps(self.status)


def _no_headset(fake=None):
    """A fake whose server has answered HELLO and reports neither READY nor a STATUS."""
    fake = fake or _Fake()
    fake.ready = False
    fake.status = None
    return fake


def _early(pipeline=None, **kwargs):
    return remote_config(pipeline, wait_for_headset=False, **kwargs)


def _info(caplog):
    return _messages(caplog, logging.INFO)


# ============================================================================
# Configuration
# ============================================================================


class TestConfig:
    def test_entry_waits_for_a_headset_by_default(self):
        config = TeleopSessionConfig(app_name="a", pipeline=_Pipeline())
        assert config.wait_for_headset is True
        assert remote_config().wait_for_headset is True

    def test_the_last_fields_keep_their_order(self):
        """Positional construction of the fields before them keeps its meaning."""
        names = [field.name for field in dataclasses.fields(TeleopSessionConfig)]
        assert names[-3:] == [
            "remote_connect_timeout_s",
            "robot_description",
            "wait_for_headset",
        ]

    @pytest.mark.parametrize("value", [None, 0, 1, 0.0, "no", "False", []], ids=repr)
    def test_anything_but_true_or_false_is_refused(self, value):
        with pytest.raises(ValueError) as raised:
            remote_config(wait_for_headset=value)
        assert str(raised.value) == (
            f"wait_for_headset must be True or False, got {value!r}"
        )

    def test_entry_refuses_a_value_set_afterwards(self):
        config = remote_config()
        session = TeleopSession(config)
        config.wait_for_headset = None
        with mock_remote_dependencies() as mocks:
            with pytest.raises(ValueError, match="wait_for_headset must be True or"):
                session.__enter__()
            assert mocks.session.events == []

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"mode": SessionMode.REMOTE, "remote_endpoint": ENDPOINT},
            {"mode": SessionMode.LIVE},
            {"mode": SessionMode.REPLAY, "mcap_config": MagicMock()},
            {"mode": SessionMode.LIVE, "joint_publisher": MagicMock()},
        ],
        ids=["remote", "live", "replay", "joint_publisher"],
    )
    def test_false_is_accepted_in_every_mode(self, kwargs):
        config = TeleopSessionConfig(
            app_name="a", pipeline=_Pipeline(), wait_for_headset=False, **kwargs
        )
        assert config.wait_for_headset is False


# ============================================================================
# Early entry
# ============================================================================


class TestEarlyEntry:
    def test_entry_returns_once_the_server_has_answered(self):
        fake = _no_headset()
        with mock_remote_dependencies(fake):
            with TeleopSession(_early()) as session:
                assert fake.events == [
                    "create",
                    "enter",
                    "start",
                    "hello_ack",
                    "update",
                ]
                assert fake.wait_ready_timeouts == []
                assert session.deviceio_session is fake
                assert session._awaiting_ready
            assert fake.events[-1] == "exit"
            assert fake.wait_ready_timeouts == []
            # Left before READY: nothing of the wait is kept for the next entry.
            assert not session._awaiting_ready
            assert session._remote_status is None

    def test_the_default_waits_for_ready(self):
        fake = _Fake()
        with mock_remote_dependencies(fake), TeleopSession(remote_config()):
            assert fake.events == ["create", "enter", "start", "hello_ack", "update"]
            assert len(fake.wait_ready_timeouts) == 1

    @pytest.mark.parametrize("description", [None, DESCRIPTION], ids=["plain", "robot"])
    def test_the_hello_is_the_defaults_byte_for_byte(self, description):
        texts = []
        for wait in (True, False):
            fake = _Fake() if description is None else _RobotFake()
            fake.status = _status("ready")
            config = remote_config(robot_description=description, wait_for_headset=wait)
            with mock_remote_dependencies(fake), TeleopSession(config):
                pass
            texts.append(fake.hello_text)
        assert texts[0] is not None
        assert texts[0] == texts[1]
        assert "wait_for_headset" not in texts[1]
        assert ("robot_state" in json.loads(texts[1])) == (description is not None)

    def test_no_answer_by_the_deadline_is_a_timeout(self):
        fake = _no_headset()
        fake.hello_acked = False
        session = TeleopSession(_early(remote_connect_timeout_s=0.2))
        with mock_remote_dependencies(fake):
            with pytest.raises(TimeoutError) as raised:
                session.__enter__()
        assert str(raised.value) == (
            f"Teleop Session Server at {ENDPOINT} was not ready after 0.2 s: "
            "server did not answer"
        )
        assert fake.events == ["create", "enter", "start", "hello_ack", "exit"]
        assert fake.wait_ready_timeouts == []
        assert not session._in_context

    def test_the_servers_error_at_hello_is_raised_as_it_is(self):
        fake = _no_headset()
        fake.hello_ack_error = RuntimeError("Tracker name '0' is used twice")
        session = TeleopSession(_early())
        with mock_remote_dependencies(fake):
            with pytest.raises(RuntimeError) as raised:
                session.__enter__()
        assert raised.value is fake.hello_ack_error

    def test_a_failed_first_update_rolls_the_entry_back(self):
        fake = _no_headset()
        fake.update_error = RuntimeError("update failed")
        session = TeleopSession(_early())
        with mock_remote_dependencies(fake):
            with pytest.raises(RuntimeError, match="update failed"):
                session.__enter__()
            assert fake.events[-2:] == ["update", "exit"]
            assert not session._in_context
            assert not session._awaiting_ready
            assert session._remote_status is None

    def test_entry_says_that_it_does_not_wait(self, caplog):
        fake = _no_headset()
        with (
            mock_remote_dependencies(fake),
            caplog.at_level(logging.INFO, logger="isaaccapture"),
            TeleopSession(_early()),
        ):
            pass
        assert _info(caplog).count(NOT_WAITING) == 1
        assert IS_READY not in _info(caplog)
        assert _messages(caplog, logging.WARNING) == []


# ============================================================================
# The headset's status before the server has an OpenXR session
# ============================================================================


class _Latest:
    """Stands in for _remote.RemoteStatus."""

    def __init__(self, status):
        self.status = status

    def latest(self):
        return self.status


def _snapshot(status, **kwargs):
    return _remote.RemoteOpenXRSession(
        _Latest(status), **kwargs
    ).get_provider_snapshot()


BEFORE_OPENXR = [
    (None, "Teleop Session Server: no status from the server yet"),
    (
        CONNECTING,
        f"Teleop Session Server: connecting_openxr ({WAITING_FOR_HEADSET})",
    ),
    (
        _status("connecting_openxr", headset=NO_HEADSET),
        "Teleop Session Server: connecting_openxr",
    ),
]
MALFORMED = [
    _status("running", headset=NO_HEADSET),
    _status("ready", headset=NO_HEADSET),
    _status("waiting_for_headset", WAITING_FOR_HEADSET, headset=NO_HEADSET),
    _status("connecting_openxr", headset=UNKNOWN_NAMES),
    {"stage": "connecting_openxr", "message": ""},
    _status("connecting_openxr", headset="CONNECTED"),
    {},
]
MALFORMED_IDS = [
    "empty_running",
    "empty_ready",
    "empty_waiting",
    "unknown_names",
    "missing",
    "not_an_object",
    "empty_status",
]


class TestStatusSnapshot:
    def test_the_empty_entry_is_the_one_the_server_sends(self):
        assert _remote.NO_HEADSET_ENTRY == server_session._NO_HEADSET == NO_HEADSET
        sent = json.loads(
            server_session.status_json(_remote.CONNECTING_OPENXR, WAITING_FOR_HEADSET)
        )
        assert sent == CONNECTING

    @pytest.mark.parametrize(("status", "error"), BEFORE_OPENXR, ids=repr)
    def test_before_the_servers_openxr_session_the_headset_is_not_connected(
        self, status, error
    ):
        snapshot = _snapshot(status, no_headset_before_openxr=True)
        assert snapshot.state == oxr.OpenXRProviderState.AVAILABLE
        assert snapshot.headset_state == oxr.OpenXRHeadsetState.DISCONNECTED
        assert snapshot.reason == oxr.OpenXRProviderReason.TRACKING_UNAVAILABLE
        assert snapshot.result_code is None
        assert snapshot.error == error

    @pytest.mark.parametrize("flag", [{}, {"no_headset_before_openxr": False}])
    @pytest.mark.parametrize(("status", "_error"), BEFORE_OPENXR, ids=repr)
    def test_without_the_flag_that_time_is_a_malformed_status(
        self, status, _error, flag
    ):
        with pytest.raises(ValueError, match="no valid headset entry"):
            _snapshot(status, **flag)

    @pytest.mark.parametrize("flag", [True, False])
    @pytest.mark.parametrize("status", MALFORMED, ids=MALFORMED_IDS)
    def test_any_other_malformed_entry_stays_an_error(self, status, flag):
        with pytest.raises(ValueError, match="no valid headset entry"):
            _snapshot(status, no_headset_before_openxr=flag)

    @pytest.mark.parametrize("flag", [True, False])
    @pytest.mark.parametrize(
        "stage", ["connecting_openxr", "waiting_for_headset", "ready", "running"]
    )
    def test_a_tracked_entry_reads_the_same_with_and_without_the_flag(
        self, stage, flag
    ):
        tracked = _snapshot(_status(stage), no_headset_before_openxr=flag)
        assert tracked.headset_state == oxr.OpenXRHeadsetState.CONNECTED
        assert tracked.error == ""
        taken_off = _snapshot(
            _status(stage, headset=TAKEN_OFF), no_headset_before_openxr=flag
        )
        assert taken_off.headset_state == oxr.OpenXRHeadsetState.DISCONNECTED
        assert taken_off.error == (
            "Teleop Session Server reports OpenXR AVAILABLE, headset DISCONNECTED "
            "(TRACKING_UNAVAILABLE)"
        )

    @pytest.mark.parametrize(
        ("status", "reported"),
        [
            (None, False),
            ({}, False),
            ({"message": "ready"}, False),
            (CONNECTING, False),
            (_status("waiting_for_headset", "headset connected"), False),
            (_status("ready"), True),
            (_status("running"), True),
            (_status("running", headset=TAKEN_OFF), True),
        ],
        ids=repr,
    )
    def test_ready_is_reported_by_its_stage_and_every_later_one(self, status, reported):
        assert _remote.ready_reported(status) is reported


class TestStatus:
    def test_the_headset_is_disconnected_until_the_server_tracks_one(self):
        fake = _no_headset()
        with mock_remote_dependencies(fake), TeleopSession(_early()) as session:
            for status, error in BEFORE_OPENXR:
                fake.status = status
                if status is not None:
                    session.step()
                headset = _headset(session)
                assert (headset.status, headset.reason, headset.error) == (
                    DeviceState.DISCONNECTED,
                    StatusReason.NO_CURRENT_DATA,
                    error,
                )
                provider = session.get_provider_status("openxr/runtime")
                assert (provider.status, provider.reason, provider.error) == (
                    ProviderState.AVAILABLE,
                    StatusReason.NONE,
                    "",
                )

            fake.status = _status(
                "waiting_for_headset", WAITING_FOR_HEADSET, headset=TAKEN_OFF
            )
            session.step()
            headset = _headset(session)
            assert headset.status == DeviceState.DISCONNECTED
            assert "headset DISCONNECTED (TRACKING_UNAVAILABLE)" in headset.error

            for stage in ("waiting_for_headset", "ready", "running"):
                fake.status = _status(stage, headset=CONNECTED)
                session.step()
                headset = _headset(session)
                assert (headset.status, headset.reason) == (
                    DeviceState.CONNECTED,
                    StatusReason.OPENXR_SESSION_READY,
                )

    @pytest.mark.parametrize("status", MALFORMED, ids=MALFORMED_IDS)
    def test_a_malformed_headset_entry_is_a_poll_error(self, status):
        fake = _no_headset()
        with mock_remote_dependencies(fake), TeleopSession(_early()) as session:
            fake.status = status
            session.step()
            provider = session.get_provider_status("openxr/runtime")
            assert (provider.status, provider.reason) == (
                ProviderState.FAILED,
                StatusReason.OPENXR_POLL_ERROR,
            )
            assert "no valid headset entry" in provider.error
            assert _headset(session).status == DeviceState.UNAVAILABLE

    def test_with_the_default_that_time_is_a_poll_error(self):
        with mock_remote_dependencies() as mocks:
            with TeleopSession(remote_config()) as session:
                mocks.session.status = CONNECTING
                session.step()
                provider = session.get_provider_status("openxr/runtime")
                assert (provider.status, provider.reason) == (
                    ProviderState.FAILED,
                    StatusReason.OPENXR_POLL_ERROR,
                )


# ============================================================================
# Steps before READY
# ============================================================================


class TestNoDataStep:
    @pytest.mark.parametrize("mode", ["sync", "pipelined"])
    def test_a_step_runs_the_pipeline_without_a_tick(self, mode):
        fake = _no_headset()
        pipeline = _Pipeline([_Source("head", HeadTracker())], session=fake)
        config = _early(
            pipeline, retargeting_execution=RetargetingExecutionConfig(mode=mode)
        )
        with mock_remote_dependencies(fake), TeleopSession(config) as session:
            assert fake.events.count("update") == 1
            for _ in range(10):
                result = session.step()
                assert result["age"][0] == -1.0
                assert session.last_step_info.remote_tick_age_s is None
                time.sleep(0.002)
            if mode == "sync":
                assert fake.events.count("update") == 11
            else:
                assert _wait_for(lambda: fake.events.count("update") > 2)
            assert fake.wait_ready_timeouts == []

    def test_the_tick_age_appears_with_the_first_samples(self):
        fake = _no_headset()
        pipeline = _Pipeline([_Source("head", HeadTracker())], session=fake)
        with mock_remote_dependencies(fake), TeleopSession(_early(pipeline)) as session:
            session.step()
            assert session.last_step_info.remote_tick_age_s is None
            fake.status = _status("ready")
            fake.next_age_ns = 7_000_000
            session.step()
            assert session.last_step_info.remote_tick_age_s == pytest.approx(0.007)


class TestReadyLine:
    def test_step_logs_it_once_when_the_stage_turns_ready(self, caplog):
        fake = _no_headset()
        with (
            mock_remote_dependencies(fake),
            caplog.at_level(logging.INFO, logger="isaaccapture"),
            TeleopSession(_early()) as session,
        ):
            for status in (
                None,
                CONNECTING,
                _status("waiting_for_headset", WAITING_FOR_HEADSET, headset=TAKEN_OFF),
                _status("waiting_for_headset", "headset connected"),
            ):
                fake.status = status
                session.step()
                assert IS_READY not in _info(caplog)
            fake.status = _status("ready")
            session.step()
            assert _info(caplog).count(IS_READY) == 1
            for status in (_status("running"), _status("running", headset=TAKEN_OFF)):
                fake.status = status
                session.step()
                session.step()
            assert _info(caplog).count(IS_READY) == 1
        # Nothing else is logged by step() while it waits.
        assert _info(caplog).count(NOT_WAITING) == 1
        assert [m for m in _info(caplog) if WAITING_FOR_HEADSET in m] == []

    def test_step_logs_it_once_when_the_stage_jumps_to_running(self, caplog):
        fake = _no_headset()
        with (
            mock_remote_dependencies(fake),
            caplog.at_level(logging.INFO, logger="isaaccapture"),
            TeleopSession(_early()) as session,
        ):
            session.step()
            assert IS_READY not in _info(caplog)
            fake.status = _status("running")
            for _ in range(3):
                session.step()
            assert _info(caplog).count(IS_READY) == 1

    def test_a_second_entry_logs_it_once_more(self, caplog):
        fake = _no_headset()
        session = TeleopSession(_early())
        with (
            mock_remote_dependencies(fake),
            caplog.at_level(logging.INFO, logger="isaaccapture"),
        ):
            for entry in (1, 2):
                fake.status = None
                with session:
                    session.step()
                    assert _info(caplog).count(IS_READY) == entry - 1
                    fake.status = _status("ready")
                    session.step()
                    session.step()
                    assert _info(caplog).count(IS_READY) == entry
                assert not session._awaiting_ready

    def test_a_pipelined_step_logs_it_once(self, caplog):
        fake = _no_headset()
        config = _early(
            retargeting_execution=RetargetingExecutionConfig(mode="pipelined")
        )
        with (
            mock_remote_dependencies(fake),
            caplog.at_level(logging.INFO, logger="isaaccapture"),
            TeleopSession(config) as session,
        ):
            session.step()
            fake.status = _status("ready")
            assert _step_until(session, lambda: IS_READY in _info(caplog))
            for _ in range(5):
                session.step()
                time.sleep(0.002)
        assert _info(caplog).count(IS_READY) == 1

    @pytest.mark.parametrize("description", [None, DESCRIPTION], ids=["plain", "robot"])
    def test_with_the_default_entry_logs_it(self, description, caplog):
        fake = _Fake() if description is None else _RobotFake()
        fake.status = _status("ready")
        config = remote_config(robot_description=description)
        with (
            mock_remote_dependencies(fake),
            caplog.at_level(logging.DEBUG, logger="isaaccapture"),
            TeleopSession(config) as session,
        ):
            assert _info(caplog).count(IS_READY) == 1
            for _ in range(3):
                session.step()
        assert _info(caplog).count(IS_READY) == 1
        assert NOT_WAITING not in _info(caplog)
        # An app that does not opt in is told nothing about the field.
        texts = [record.getMessage() for record in caplog.records]
        assert [text for text in texts if "wait_for_headset" in text] == []
        assert [text for text in texts if "tracked headset" in text] == []


# ============================================================================
# What the pipeline is told while nobody is there
# ============================================================================

STOPPED = (ExecutionState.STOPPED, False)
BUTTONS = (
    DefaultTeleopStateManager.INPUT_KILL,
    DefaultTeleopStateManager.INPUT_RUN_TOGGLE,
    DefaultTeleopStateManager.INPUT_RESET,
)
RUN = DefaultTeleopStateManager.INPUT_RUN_TOGGLE


def _execution(session):
    """The execution state and the reset pulse the last step gave the pipeline."""
    events = session.last_context.execution_events
    return events.execution_state, events.reset


class _Buttons(_Source):
    """The operator's buttons as the test holds them; None is a step without tracker data."""

    def __init__(self):
        self.held = None
        super().__init__("buttons", HeadTracker())

    def output_spec(self):
        return {name: OptionalType(bool_signal(name)) for name in BUTTONS}

    def _compute_fn(self, inputs, outputs, context):
        for name in BUTTONS:
            if self.held is None:
                outputs[name].set_none()
            else:
                outputs[name][0] = name in self.held


class TestStartWithoutOperator:
    def test_without_a_control_pipeline_every_step_runs(self):
        # The session holds no output back: what moves the robot is the robot app's choice.
        fake = _no_headset()
        with mock_remote_dependencies(fake), TeleopSession(_early()) as session:
            for status in (None, CONNECTING):
                fake.status = status
                session.step()
                assert _execution(session) == (ExecutionState.RUNNING, False)

    def test_a_state_manager_stays_stopped_until_the_operator_starts_it(self):
        buttons = _Buttons()
        manager = DefaultTeleopStateManager(name="teleop_manager")
        control = manager.connect({name: buttons.output(name) for name in BUTTONS})
        fake = _no_headset()
        config = _early(teleop_control_pipeline=control)
        with mock_remote_dependencies(fake), TeleopSession(config) as session:

            def step(held):
                buttons.held = held
                session.step()
                return _execution(session)

            # Neither the server's answer, nor READY, nor a tracked headset starts it.
            for status in (None, CONNECTING, _status("ready"), _status("running")):
                fake.status = status
                assert step(None) == STOPPED
            # Nor does tracker data with no button held.
            assert step(()) == STOPPED
            assert step((RUN,)) == (ExecutionState.PAUSED, False)
            assert step(()) == (ExecutionState.PAUSED, False)
            assert step((RUN,)) == (ExecutionState.RUNNING, False)
            # Tracker data that goes away stops it in that step.
            assert step(None) == (ExecutionState.STOPPED, True)
            assert step(None) == STOPPED


# ============================================================================
# Robot state before READY
# ============================================================================


class TestRobotStateBeforeReady:
    def test_entry_declares_the_series_and_sends_the_description(self):
        fake = _no_headset(_RobotFake())
        config = _early(robot_description=DESCRIPTION)
        with mock_remote_dependencies(fake), TeleopSession(config) as session:
            assert fake.events == [
                "create",
                "set_robot_state",
                "start",
                "start_robot_state",
                "publish robot_description",
                "update",
            ]
            assert fake.series == records.declarations(DESCRIPTION)
            ((topic, data, _),) = fake.published
            assert topic == records.DESCRIPTION_TOPIC
            assert records.decode_description(data) == DESCRIPTION

            # Before READY, and before any step.
            assert session.robot_state_active
            assert session.publish_robot_state("arm", [0.25, -0.5, 1.0])
            assert session.publish_robot_record("wrist_force", [1.0, 2.0, True])
            session.step()
            assert session._awaiting_ready
            assert session.publish_robot_state("arm", [0.5, 0.0, 0.0], time_ns=1234)
        assert [series for series, _, _ in fake.published[1:]] == [
            "arm",
            "wrist_force",
            "arm",
        ]
        assert "ready" not in fake.events
        assert not session.robot_state_active

    def test_a_server_that_does_not_take_robot_state_gives_early_entry(self):
        fake = _no_headset(_RobotFake(_ack()))
        config = _early(robot_description=DESCRIPTION)
        with mock_remote_dependencies(fake), TeleopSession(config) as session:
            assert not session.robot_state_active
            assert not session.publish_robot_state("arm", [0.0, 0.0, 0.0])
            session.step()
        assert fake.events == [
            "create",
            "set_robot_state",
            "start",
            "update",
            "update",
            "exit",
        ]


# ============================================================================
# A server end in miniature, over loopback, with the robot app's real trackers
# ============================================================================

START_FAILURE = (
    "The Teleop Session Server could not start the trackers: "
    "XR_ERROR_EXTENSION_NOT_PRESENT"
)
SILENCE_S = 3.5


def _late_headset(*, store=None, answer=None, failure=None):
    """A script that answers HELLO, reports `connecting_openxr` and stays silent until `go`.

    Then it starts its session, ticks and sends READY; with `failure` it sends that ERROR and
    closes instead. Returns (script, go).
    """
    go = threading.Event()

    def script(server, host):
        if store is not None:
            host.attach_robot_state(store)
        host.send_hello_ack(json.dumps({**HELLO_ACK, **(answer or {})}))
        host.send_status(json.dumps(CONNECTING))
        while not go.is_set():
            if server.stop_event.is_set() or not host.alive:
                return
            time.sleep(0.005)
        if failure is not None:
            host.send_error(failure)
            return
        # The keyboard needs no OpenXR; the robot app's topic filter drops its records, so
        # each SAMPLES is empty and still carries its tick time.
        host.start_session(
            [(KeyboardTracker(), "0")],
            oxr.OpenXRSessionHandles(*PLACEHOLDER_HANDLES),
            VendorConfig(),
            server.hello["topics"],
        )
        host.send_status(
            json.dumps(
                _status("waiting_for_headset", "headset connected", headset=CONNECTED)
            )
        )
        host.tick()
        host.send_ready(json.dumps(_status("ready")))
        while not server.stop_event.is_set() and host.alive:
            host.tick()
            time.sleep(0.005)

    return script, go


def _state_manager(controllers):
    """The teleop_control_pipeline of the control state machine page: left controller buttons."""
    left = controllers.output(ControllersSource.LEFT)

    def button(name, index):
        selector = create_bool_selector(
            left, name=name, selector_fn=lambda selected: selected[index]
        )
        return selector.output("value")

    manager = DefaultTeleopStateManager(name="teleop_manager")
    return manager.connect(
        {
            manager.INPUT_KILL: button("kill", ControllerInputIndex.SECONDARY_CLICK),
            manager.INPUT_RUN_TOGGLE: button(
                "run_toggle", ControllerInputIndex.PRIMARY_CLICK
            ),
            manager.INPUT_RESET: button("reset", ControllerInputIndex.THUMBSTICK_CLICK),
        }
    )


def _real_sources_config(server, *, state_manager=False, **kwargs):
    head = HeadSource(name="head")
    hands = HandsSource(name="hands")
    controllers = ControllersSource(name="controllers")
    if state_manager:
        kwargs["teleop_control_pipeline"] = _state_manager(controllers)
    pipeline = OutputCombiner(
        {
            "head": head.output("head"),
            "hand_left": hands.output(HandsSource.LEFT),
            "hand_right": hands.output(HandsSource.RIGHT),
            "controller_left": controllers.output(ControllersSource.LEFT),
            "controller_right": controllers.output(ControllersSource.RIGHT),
        }
    )
    return TeleopSessionConfig(
        app_name="NoHeadsetLoopback",
        pipeline=pipeline,
        mode=SessionMode.REMOTE,
        remote_endpoint=server.endpoint,
        remote_connect_timeout_s=10.0,
        wait_for_headset=False,
        verbose=False,
        **kwargs,
    )


def _all_absent(result):
    return len(result) == 5 and all(group.is_none for group in result.values())


def _reading(count):
    """A reading of the arm that tells `count` apart from its neighbours, exact in a float."""
    return [(count % 64) / 64.0, 0.0, 1.0]


def _arm(store):
    record = store.latest("arm")
    if record.data is None:
        return None
    return [joint.position for joint in records.decode_joints(record.data)]


class TestLoopback:
    def test_the_robot_app_steps_and_publishes_until_a_headset_is_tracked(self, caplog):
        store = RobotStateStore()
        script, go = _late_headset(store=store, answer=AGREED)
        server = _MiniServer(script)
        config = _real_sources_config(server, robot_description=DESCRIPTION)
        ready = f"Teleop Session Server at {server.endpoint} is ready"
        try:
            with (
                caplog.at_level(logging.INFO, logger="isaaccapture"),
                TeleopSession(config) as session,
            ):
                assert server.hello["robot_state"] == {"version": 1}
                assert [entry["type"] for entry in server.hello["trackers"]] == [
                    "HeadTracker",
                    "HandTracker",
                    "ControllerTracker",
                ]
                assert session.robot_state_active
                assert _headset(session).status == DeviceState.DISCONNECTED

                # The server sends nothing for longer than a link without heartbeats lives.
                steps = 0
                end = time.monotonic() + SILENCE_S
                while time.monotonic() < end:
                    result = session.step()
                    assert _all_absent(result)
                    assert session.last_step_info.remote_tick_age_s is None
                    steps += 1
                    assert session.publish_robot_state("arm", _reading(steps))
                    time.sleep(0.01)
                assert steps > 20
                headset = _headset(session)
                assert (headset.status, headset.reason, headset.error) == (
                    DeviceState.DISCONNECTED,
                    StatusReason.NO_CURRENT_DATA,
                    f"Teleop Session Server: connecting_openxr ({WAITING_FOR_HEADSET})",
                )
                provider = session.get_provider_status("openxr/runtime")
                assert provider.status == ProviderState.AVAILABLE
                assert _wait_for(lambda: _arm(store) == _reading(steps))
                assert store.latest("arm").fresh
                assert store.connected
                assert store.dropped_records == 0
                assert ready not in _info(caplog)

                go.set()
                assert _step_until(
                    session,
                    lambda: session.last_step_info.remote_tick_age_s is not None,
                )
                assert 0.0 <= session.last_step_info.remote_tick_age_s < 0.25
                assert _step_until(
                    session, lambda: _headset(session).status == DeviceState.CONNECTED
                )
                assert _step_until(session, lambda: ready in _info(caplog))
                for _ in range(20):
                    session.step()
                    time.sleep(0.005)
                assert _info(caplog).count(ready) == 1
                assert session.robot_state_active
                assert session.publish_robot_state("arm", [-0.5, 0.25, 0.0])
                assert _wait_for(lambda: _arm(store) == [-0.5, 0.25, 0.0])
        finally:
            server.stop()
        assert not store.connected

    def test_a_server_built_before_robot_state_gives_early_entry(self, caplog):
        store = RobotStateStore()
        script, go = _late_headset(store=store)
        server = _MiniServer(script)
        config = _real_sources_config(server, robot_description=DESCRIPTION)
        try:
            with (
                caplog.at_level(logging.WARNING, logger="isaaccapture"),
                TeleopSession(config) as session,
            ):
                assert server.hello["robot_state"] == {"version": 1}
                assert not session.robot_state_active
                for _ in range(20):
                    assert _all_absent(session.step())
                    assert not session.publish_robot_state("arm", [0.0, 0.0, 0.0])
                    time.sleep(0.005)
                assert store.series() == []
                go.set()
                assert _step_until(
                    session, lambda: _headset(session).status == DeviceState.CONNECTED
                )
                assert not session.publish_robot_state("arm", [0.0, 0.0, 0.0])
            assert (
                f"The Teleop Session Server at {server.endpoint} does not take robot "
                "state; none is sent"
            ) in _messages(caplog, logging.WARNING)
        finally:
            server.stop()

    def test_a_state_manager_keeps_the_session_stopped_with_nobody_there(self):
        script, go = _late_headset()
        server = _MiniServer(script)
        config = _real_sources_config(server, state_manager=True)
        try:
            with TeleopSession(config) as session:
                # One ControllerTracker serves the pipeline and the state manager.
                assert [entry["type"] for entry in server.hello["trackers"]] == [
                    "HeadTracker",
                    "HandTracker",
                    "ControllerTracker",
                ]
                for _ in range(20):
                    assert _all_absent(session.step())
                    assert _execution(session) == STOPPED
                    time.sleep(0.005)

                # A tracked headset and READY are not an operator input.
                go.set()
                assert _step_until(session, lambda: not session._awaiting_ready)
                assert _step_until(
                    session, lambda: _headset(session).status == DeviceState.CONNECTED
                )
                for _ in range(20):
                    assert _all_absent(session.step())
                    assert session.last_step_info.remote_tick_age_s is not None
                    assert _execution(session) == STOPPED
                    time.sleep(0.005)
        finally:
            server.stop()

    def test_a_tracker_failure_on_the_server_ends_the_session_in_step(self):
        store = RobotStateStore()
        script, go = _late_headset(store=store, answer=AGREED, failure=START_FAILURE)
        server = _MiniServer(script)
        config = _real_sources_config(server, robot_description=DESCRIPTION)
        try:
            with TeleopSession(config) as session:
                for reading in range(5):
                    session.step()
                    assert session.publish_robot_state("arm", _reading(reading))
                assert _wait_for(lambda: _arm(store) == _reading(4))
                assert store.connected

                go.set()
                with pytest.raises(RuntimeError) as raised:
                    _step_until(session, lambda: False, timeout_s=10.0)
                assert type(raised.value) is RuntimeError
                assert str(raised.value) == START_FAILURE
                assert _wait_for(
                    lambda: not session.publish_robot_state("arm", _reading(9))
                )
                assert _wait_for(lambda: not store.connected)
                provider = session.get_provider_status("openxr/runtime")
                assert provider.status == ProviderState.FAILED
                # What was read before the failure stays readable.
                assert _arm(store) == _reading(4)
        finally:
            server.stop()


# ============================================================================
# LIVE and REPLAY
# ============================================================================


class _OpenXR:
    """Stands in for an entered oxr.OpenXRSession with a tracked headset."""

    def __init__(self):
        self.handles = object()
        self.exited = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.exited = True
        return False

    def get_handles(self):
        return self.handles

    def get_provider_snapshot(self):
        return SimpleNamespace(
            state="AVAILABLE",
            headset_state="CONNECTED",
            reason="NONE",
            result_code=None,
            error="",
        )


@contextmanager
def local_backends():
    """Replace OpenXR, DeviceIO, replay and the robot twin; yields the mocks."""
    openxr = _OpenXR()
    with (
        patch("isaaccapture.oxr.OpenXRSession", return_value=openxr) as oxr_cls,
        patch("isaaccapture.deviceio.DeviceIOSession.run") as live_run,
        patch(
            "isaaccapture.deviceio.DeviceIOSession.get_required_extensions",
            return_value=list(EXTENSIONS),
        ),
        patch("isaaccapture.deviceio.ReplaySession.run") as replay_run,
        patch("isaaccapture.deviceio.McapReplayConfig") as replay_config,
        patch.object(
            TeleopSession, "_start_twin", return_value=PLACEHOLDER_HANDLES
        ) as start_twin,
    ):
        yield SimpleNamespace(
            openxr=openxr,
            oxr_cls=oxr_cls,
            live_run=live_run,
            replay_run=replay_run,
            replay_config=replay_config,
            start_twin=start_twin,
        )


def local_config(**kwargs):
    kwargs.setdefault("pipeline", _Pipeline())
    return TeleopSessionConfig(app_name="LocalApp", verbose=False, **kwargs)


def robot_state_files(runtime_dir):
    """What robot state left under the runtime folder, by name."""
    folder = runtime_dir / "isaaccapture"
    if not folder.is_dir():
        return []
    return sorted(name for name in os.listdir(folder) if "robot_state" in name)


def reader_threads():
    return [t for t in threading.enumerate() if t.name == READER_THREAD]


class TestLive:
    def test_false_makes_one_attempt_that_does_not_wait(self):
        with local_backends() as mocks:
            with TeleopSession(local_config(wait_for_headset=False)) as session:
                assert mocks.oxr_cls.call_args_list == [
                    call("LocalApp", EXTENSIONS, wait_for_system=False)
                ]
                assert session.oxr_session is mocks.openxr
                assert mocks.live_run.call_count == 1
                assert mocks.live_run.call_args.args[1] is mocks.openxr.handles
                session.step()
                assert _headset(session).status == DeviceState.CONNECTED
            assert mocks.openxr.exited
            assert mocks.oxr_cls.call_count == 1

    def test_the_default_keeps_its_two_arguments(self):
        with local_backends() as mocks:
            with TeleopSession(local_config()) as session:
                assert mocks.oxr_cls.call_args_list == [call("LocalApp", EXTENSIONS)]
                assert session.oxr_session is mocks.openxr

    @pytest.mark.parametrize("description", [None, DESCRIPTION], ids=["plain", "robot"])
    def test_no_headset_at_the_runtime_is_a_clear_error(
        self, description, runtime_dir, caplog
    ):
        assert teleop_session._NO_HEADSET_AT_RUNTIME == NO_HEADSET_AT_RUNTIME
        assert NO_HEADSET_AT_OPENXR.startswith(teleop_session._NO_HEADSET_ERROR)
        error = RuntimeError(NO_HEADSET_AT_OPENXR)
        config = local_config(wait_for_headset=False, robot_description=description)
        session = TeleopSession(config)
        with local_backends() as mocks, caplog.at_level(logging.WARNING):
            mocks.oxr_cls.side_effect = error
            start = time.monotonic()
            with pytest.raises(RuntimeError) as raised:
                session.__enter__()
            assert time.monotonic() - start < 5.0
            assert type(raised.value) is RuntimeError
            assert str(raised.value) == NO_HEADSET_AT_RUNTIME
            assert raised.value.__cause__ is error
            assert mocks.oxr_cls.call_args_list == [
                call("LocalApp", EXTENSIONS, wait_for_system=False)
            ]
            mocks.live_run.assert_not_called()

            # Rolled back: nothing of the entry is left.
            assert not session._in_context
            assert session.oxr_session is None
            assert not session.robot_state_active
            assert not os.path.exists(default_socket_path())
            assert robot_state_files(runtime_dir) == (
                [] if description is None else ["robot_state.sock.lock"]
            )
            assert reader_threads() == []
            if description is not None:
                with pytest.raises(RuntimeError, match="TeleopSession is not active"):
                    session.publish_robot_state("arm", [0.0, 0.0, 0.0])

            # Once a headset has reached the runtime the same session enters.
            mocks.oxr_cls.side_effect = None
            with session:
                assert session.robot_state_active == (description is not None)
        assert _messages(caplog, logging.WARNING) == []

    @pytest.mark.parametrize(
        "text",
        [
            "Failed to get OpenXR system: -2",
            "Failed to create OpenXR instance: -51",
            "Environment variable NV_CXR_RUNTIME_DIR is not set",
        ],
    )
    def test_any_other_openxr_error_passes_as_it_is(self, text):
        error = RuntimeError(text)
        session = TeleopSession(local_config(wait_for_headset=False))
        with local_backends() as mocks:
            mocks.oxr_cls.side_effect = error
            with pytest.raises(RuntimeError) as raised:
                session.__enter__()
        assert raised.value is error
        assert raised.value.__cause__ is None

    def test_with_the_default_the_no_headset_error_is_not_reworded(self):
        """The default constructor waits and never raises it; a stand-in that does is passed on."""
        error = RuntimeError(NO_HEADSET_AT_OPENXR)
        session = TeleopSession(local_config())
        with local_backends() as mocks:
            mocks.oxr_cls.side_effect = error
            with pytest.raises(RuntimeError) as raised:
                session.__enter__()
        assert raised.value is error


def _handle_values(handles):
    return (handles.instance, handles.session, handles.space, handles.proc_addr)


def _calls(mocks, factory):
    """Every call the entry made to a backend, without the objects made per entry."""
    live_runs = [
        (args[0], _handle_values(args[1]), args[2], type(args[3]).__name__, kwargs)
        for args, kwargs in mocks.live_run.call_args_list
    ]
    return {
        "openxr": mocks.oxr_cls.call_args_list,
        "live": live_runs,
        "replay": mocks.replay_run.call_args_list,
        "replay_config": mocks.replay_config.call_args_list,
        "twin": [args[1:] for args, _ in mocks.start_twin.call_args_list],
        "factory": factory.call_args_list,
    }


class TestNoEffect:
    @pytest.mark.parametrize(
        "case", ["oxr_handles", "oxr_session_factory", "joint_publisher", "replay"]
    )
    def test_false_changes_no_call(self, case):
        handles = oxr.OpenXRSessionHandles(*PLACEHOLDER_HANDLES)
        mcap_config = MagicMock()
        mcap_config.filename = "session.mcap"
        made = {}
        for wait in (True, False):
            factory = MagicMock(return_value=handles)
            kwargs = {
                "oxr_handles": {"oxr_handles": handles},
                "oxr_session_factory": {"oxr_session_factory": factory},
                "joint_publisher": {"joint_publisher": MagicMock()},
                "replay": {"mode": SessionMode.REPLAY, "mcap_config": mcap_config},
            }[case]
            config = local_config(wait_for_headset=wait, **kwargs)
            with local_backends() as mocks:
                mocks.replay_config.return_value = "replay config"
                with TeleopSession(config) as session:
                    session.step()
                made[wait] = _calls(mocks, factory)
        assert made[True] == made[False]
        assert made[False]["openxr"] == []
        assert len(made[False]["replay" if case == "replay" else "live"]) == 1
        assert len(made[False]["twin"]) == (case == "joint_publisher")
        assert len(made[False]["factory"]) == (case == "oxr_session_factory")
