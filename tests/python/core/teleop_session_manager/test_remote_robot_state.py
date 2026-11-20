# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Robot state sent by TeleopSession in SessionMode.REMOTE.

The first tests replace RemoteSession with a fake. TestLoopback runs a server end in miniature
(RemoteSessionListener + RemoteSessionHost + RobotStateStore) on a thread, so HELLO, HELLO_ACK and
the robot-state link messages cross a real connection: a robot app and a server built with and
without robot state, in each pairing.
"""

import dataclasses
import json
import logging
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

import isaaccapture
from isaaccapture.deviceio_session import (
    RemotePeerClosedError,
    RemoteSessionHost,
    RemoteSessionListener,
)
from isaaccapture.robot_state import (
    RobotDescriptionError,
    RobotStateStore,
    parse_robot_description,
    records,
)
from isaaccapture.teleop_session_manager import (
    SessionMode,
    TeleopSession,
    TeleopSessionConfig,
    _remote,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="remote sessions need POSIX sockets"
)

ENDPOINT = "tcp://server.example:47600"
WAIT_MS = 5000
NOT_TAKEN = (
    f"The Teleop Session Server at {ENDPOINT} does not take robot state; none is sent"
)
DESCRIPTION_TOML = """
robot = "test_arm"

[shapes.force]
fields = ["fx", "fy", { name = "contact", kind = "flag" }]

[series.arm]
shape = "joints"
rate_hz = 20.0
joints = [
  { name = "joint1",  kind = "revolute", min = -2.0, max = 2.0 },
  { name = "joint2",  kind = "revolute", min = -1.0, max = 3.0 },
  { name = "gripper", kind = "gripper",  min = 0.0,  max = 1.0 },
]

[series.status]
shape = "status"
readings = ["motors_on", "fault"]

[series.wrist_force]
shape = "force"
"""
DESCRIPTION = parse_robot_description(DESCRIPTION_TOML)
SERIES_NAMES = ["robot_description", "arm", "status", "wrist_force"]
CONNECTED = {"state": "AVAILABLE", "headset_state": "CONNECTED", "reason": "NONE"}
READY = {"stage": "ready", "message": "", "headset": CONNECTED}
AGREED = {"robot_state": {"version": 1}}
# Answers that do not agree to this robot app's robot state.
NOT_AGREED = [
    {},
    {"robot_state": {"version": 2}},
    {"robot_state": {"version": "1"}},
    {"robot_state": {"version": True}},
    {"robot_state": {"version": 1.0}},
    {"robot_state": {}},
    {"robot_state": True},
    {"robot_state": None},
    {"robot_state": [1]},
]


@pytest.fixture
def description_file(tmp_path):
    path = tmp_path / "test_arm.toml"
    path.write_text(DESCRIPTION_TOML)
    return path


class _Pipeline:
    def get_leaf_nodes(self):
        return []

    def execute_pipeline(self, inputs, context=None):
        return {}


def _config(**kwargs):
    kwargs.setdefault("remote_endpoint", ENDPOINT)
    return TeleopSessionConfig(
        app_name="RobotStateApp",
        pipeline=_Pipeline(),
        mode=SessionMode.REMOTE,
        verbose=False,
        **kwargs,
    )


def _ack(**fields):
    return {"session_id": "fake", "server_version": isaaccapture.__version__, **fields}


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


# ============================================================================
# Configuration, HELLO and HELLO_ACK
# ============================================================================


class TestConfig:
    def test_no_description_by_default(self):
        assert _config().robot_description is None

    def test_it_is_the_last_field(self):
        """Positional construction of the fields before it keeps its meaning."""
        names = [field.name for field in dataclasses.fields(TeleopSessionConfig)]
        assert names[-2:] == ["remote_connect_timeout_s", "robot_description"]

    def test_a_path_or_a_loaded_description_is_accepted(self, description_file):
        assert _config(robot_description=description_file).robot_description
        assert _config(robot_description=str(description_file)).robot_description
        assert _config(robot_description=DESCRIPTION).robot_description is DESCRIPTION

    @pytest.mark.parametrize("mode", [SessionMode.LIVE, SessionMode.REPLAY])
    def test_it_requires_remote_mode(self, mode):
        with pytest.raises(
            ValueError, match=r"^robot_description requires mode=SessionMode\.REMOTE$"
        ):
            TeleopSessionConfig(
                app_name="RobotStateApp",
                pipeline=_Pipeline(),
                mode=mode,
                mcap_config=MagicMock() if mode == SessionMode.REPLAY else None,
                robot_description=DESCRIPTION,
            )

    def test_entry_checks_a_mode_changed_afterwards(self):
        config = _config(robot_description=DESCRIPTION)
        session = TeleopSession(config)
        config.mode = SessionMode.LIVE
        config.remote_endpoint = None
        with pytest.raises(ValueError, match="robot_description requires"):
            session.__enter__()


HELLO_FIELDS = {
    "app_name": "App",
    "tick_rate_hz": 90,
    "trackers": [{"name": "0"}],
    "topics": ["0/head"],
}
# The HELLO of a robot app built before robot state, byte for byte.
HELLO_BEFORE = json.dumps(
    {
        "protocol": 1,
        "isaaccapture_version": isaaccapture.__version__,
        "app_name": "App",
        "tick_rate_hz": 90.0,
        "trackers": [{"name": "0"}],
        "topics": ["0/head"],
    }
)


class TestHello:
    def test_without_robot_state_it_is_unchanged(self):
        assert _remote.build_hello(**HELLO_FIELDS) == HELLO_BEFORE
        assert _remote.build_hello(**HELLO_FIELDS, robot_state=False) == HELLO_BEFORE

    def test_with_robot_state_it_gains_one_key(self):
        hello = json.loads(_remote.build_hello(**HELLO_FIELDS, robot_state=True))
        assert hello.pop("robot_state") == {"version": 1}
        assert hello == json.loads(HELLO_BEFORE)
        assert _remote.ROBOT_STATE_VERSION == 1

    def test_an_answer_with_this_version_agrees(self):
        assert _remote.robot_state_agreed(json.dumps(_ack(**AGREED)))

    @pytest.mark.parametrize("fields", NOT_AGREED, ids=repr)
    def test_any_other_answer_does_not(self, fields):
        assert not _remote.robot_state_agreed(json.dumps(_ack(**fields)))

    @pytest.mark.parametrize("text", ["", "not JSON", "[]", '"robot_state"'])
    def test_an_answer_that_is_no_object_does_not(self, text):
        assert not _remote.robot_state_agreed(text)


# ============================================================================
# A fake RemoteSession
# ============================================================================


class _SessionBefore:
    """deviceio.RemoteSession of a build without robot state: none of the robot-state calls."""

    def __init__(self, ack=None):
        self.events = []
        self.hello = None
        self.ack = _ack() if ack is None else ack
        self.start_errors = []
        self.tick_age_ns = None

    def create(self, trackers, max_tick_age_ns):
        self.events.append("create")
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.events.append("exit")
        return False

    def subscribed_topics(self):
        return []

    def start(self, endpoint, hello, connect_timeout_ms):
        self.events.append("start")
        self.hello = json.loads(hello)
        if self.start_errors:
            raise self.start_errors.pop(0)

    def wait_hello_ack(self, timeout_ms):
        return True

    def hello_ack(self):
        return json.dumps(self.ack)

    def wait_ready(self, timeout_ms):
        self.events.append("ready")
        return True

    def latest_status(self):
        return json.dumps(READY)

    def update(self):
        self.events.append("update")


class _Session(_SessionBefore):
    """With the robot-state calls, which it records."""

    def __init__(self, ack=None):
        super().__init__(ack)
        self.series = None
        self.robot_state_active = False
        self.link_up = True
        # (series, record bytes, sample time)
        self.published = []

    def set_robot_state(self, series):
        self.events.append("set_robot_state")
        self.series = list(series)

    def start_robot_state(self):
        self.events.append("start_robot_state")
        self.robot_state_active = True

    def publish_robot_record(self, series, data, sample_time_ns):
        if not self.robot_state_active or not self.link_up:
            return False
        self.events.append(f"publish {series}")
        self.published.append((series, data, sample_time_ns))
        return True


@contextmanager
def _fake(session):
    with (
        patch("isaaccapture.deviceio.RemoteSession.create", side_effect=session.create),
        patch("isaaccapture.oxr.OpenXRSession"),
        patch("isaaccapture.deviceio.DeviceIOSession.run"),
    ):
        yield session


class TestWithoutADescription:
    def test_a_remote_session_without_the_new_calls_still_enters(self):
        with _fake(_SessionBefore(_ack(**AGREED))) as fake:
            with TeleopSession(_config()) as session:
                session.step()
                assert not session.robot_state_active
        assert fake.events == ["create", "start", "ready", "update", "update", "exit"]
        assert "robot_state" not in fake.hello
        assert sorted(fake.hello) == [
            "app_name",
            "isaaccapture_version",
            "protocol",
            "tick_rate_hz",
            "topics",
            "trackers",
        ]

    def test_publishing_says_what_is_missing(self):
        session = TeleopSession(_config())
        for publish in (session.publish_robot_state, session.publish_robot_record):
            with pytest.raises(
                RuntimeError,
                match=r"^TeleopSessionConfig\.robot_description is not set$",
            ):
                publish("arm", [0.0, 0.0, 0.0])
        with _fake(_Session()), session:
            with pytest.raises(RuntimeError, match="robot_description is not set"):
                session.publish_robot_state("arm", [0.0, 0.0, 0.0])
        assert not session.robot_state_active

    def test_importing_the_library_does_not_load_the_robot_state_python_modules(self):
        code = (
            "import sys, isaaccapture, isaaccapture.teleop_session_manager\n"
            "import isaaccapture.deviceio_session\n"
            "loaded = sorted(m for m in sys.modules if m.startswith('isaaccapture.robot_state.'))\n"
            "print(loaded)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        # The store's native module comes with the session bindings; nothing written in Python does.
        assert result.stdout.strip().splitlines()[-1] == (
            "['isaaccapture.robot_state._robot_state']"
        )


class TestWithADescription:
    def test_entry_declares_the_series_and_sends_the_description(
        self, description_file
    ):
        with _fake(_Session(_ack(**AGREED))) as fake:
            with TeleopSession(_config(robot_description=description_file)) as session:
                assert session.robot_state_active
                session.step()
        assert fake.events == [
            "create",
            "set_robot_state",
            "start",
            "start_robot_state",
            "publish robot_description",
            "ready",
            "update",
            "update",
            "exit",
        ]
        assert fake.hello["robot_state"] == {"version": 1}
        assert [entry.name for entry in fake.series] == SERIES_NAMES
        assert fake.series == records.declarations(DESCRIPTION)
        ((topic, data, sample_time_ns),) = fake.published
        assert topic == records.DESCRIPTION_TOPIC
        assert records.decode_description(data) == DESCRIPTION
        assert abs(time.monotonic_ns() - sample_time_ns) < 60e9
        assert not session.robot_state_active

    def test_a_loaded_description_works_as_a_file_does(self):
        with _fake(_Session(_ack(**AGREED))) as fake:
            with TeleopSession(_config(robot_description=DESCRIPTION)) as session:
                assert session.robot_state_active
        assert fake.series == records.declarations(DESCRIPTION)

    def test_a_reading_of_each_shape_is_published(self):
        with _fake(_Session(_ack(**AGREED))) as fake:
            with TeleopSession(_config(robot_description=DESCRIPTION)) as session:
                fake.published.clear()
                before_ns = time.monotonic_ns()
                assert session.publish_robot_state("arm", [0.25, -0.5, 1.0])
                assert session.publish_robot_state(
                    "arm", [0.5, 0.0, 0.0], valid=[True, False, True], time_ns=1234
                )
                assert session.publish_robot_state(
                    "status",
                    [True, "none"],
                    faults=[("overtemperature", "joint2 is hot")],
                )
                assert session.publish_robot_state("wrist_force", [1.5, -2.0, True])
                assert session.publish_robot_record(
                    "wrist_force", {"fx": 0.5, "fy": 0.25, "contact": False}, time_ns=99
                )
                after_ns = time.monotonic_ns()
        assert [series for series, _, _ in fake.published] == [
            "arm",
            "arm",
            "status",
            "wrist_force",
            "wrist_force",
        ]
        first, second, status, force, record = fake.published
        assert [
            (j.name, j.position, j.valid) for j in records.decode_joints(first[1])
        ] == [
            ("joint1", 0.25, True),
            ("joint2", -0.5, True),
            ("gripper", 1.0, True),
        ]
        # Without time_ns the reading is taken as made now, on the monotonic clock.
        assert before_ns <= first[2] <= after_ns
        assert [j.valid for j in records.decode_joints(second[1])] == [
            True,
            False,
            True,
        ]
        assert second[2] == 1234
        readings, faults = records.decode_status(status[1])
        assert readings == {"motors_on": True, "fault": "none"}
        assert [(f.code, f.text) for f in faults] == [
            ("overtemperature", "joint2 is hot")
        ]
        assert records.decode_custom(force[1]) == {
            "fx": 1.5,
            "fy": -2.0,
            "contact": True,
        }
        assert records.decode_custom(record[1]) == {
            "fx": 0.5,
            "fy": 0.25,
            "contact": False,
        }
        assert record[2] == 99

    def test_a_reading_that_does_not_fit_raises_and_sends_nothing(self):
        with _fake(_Session(_ack(**AGREED))) as fake:
            with TeleopSession(_config(robot_description=DESCRIPTION)) as session:
                fake.published.clear()
                with pytest.raises(
                    ValueError, match=r"has no series 'leg' \(it has: arm"
                ):
                    session.publish_robot_state("leg", [0.0])
                with pytest.raises(ValueError, match="series 'arm' takes 3 values"):
                    session.publish_robot_state("arm", [0.0, 0.0])
                with pytest.raises(ValueError, match="faults does not apply"):
                    session.publish_robot_state("arm", [0.0, 0.0, 0.0], faults=[])
                with pytest.raises(ValueError, match="unknown fields"):
                    session.publish_robot_record(
                        "wrist_force", {"fx": 0, "fy": 0, "contact": True, "fz": 0}
                    )
        assert fake.published == []

    def test_publishing_returns_false_once_the_link_is_down(self):
        with _fake(_Session(_ack(**AGREED))) as fake:
            with TeleopSession(_config(robot_description=DESCRIPTION)) as session:
                assert session.publish_robot_state("arm", [0.0, 0.0, 0.0])
                fake.link_up = False
                assert not session.publish_robot_state("arm", [0.0, 0.0, 0.0])
                assert not session.publish_robot_record(
                    "wrist_force", [0.0, 0.0, False]
                )

    def test_publishing_outside_the_session_raises(self):
        session = TeleopSession(_config(robot_description=DESCRIPTION))
        assert not session.robot_state_active
        with pytest.raises(RuntimeError, match=r"^TeleopSession is not active$"):
            session.publish_robot_state("arm", [0.0, 0.0, 0.0])
        with _fake(_Session(_ack(**AGREED))), session:
            assert session.publish_robot_state("arm", [0.0, 0.0, 0.0])
        with pytest.raises(RuntimeError, match=r"^TeleopSession is not active$"):
            session.publish_robot_record("wrist_force", [0.0, 0.0, False])

    def test_publishing_from_another_thread(self):
        results = []
        with _fake(_Session(_ack(**AGREED))):
            with TeleopSession(_config(robot_description=DESCRIPTION)) as session:
                thread = threading.Thread(
                    target=lambda: results.append(
                        session.publish_robot_state("arm", [0.0, 0.0, 0.0])
                    )
                )
                thread.start()
                thread.join(timeout=10)
        assert results == [True]

    @pytest.mark.parametrize("fields", NOT_AGREED, ids=repr)
    def test_a_server_that_does_not_agree_gets_nothing(self, fields, caplog):
        caplog.set_level(logging.INFO)
        with _fake(_Session(_ack(**fields))) as fake:
            with TeleopSession(_config(robot_description=DESCRIPTION)) as session:
                assert not session.robot_state_active
                assert not session.publish_robot_state("arm", [0.0, 0.0, 0.0])
                assert not session.publish_robot_record(
                    "wrist_force", [0.0, 0.0, False]
                )
                # What does not fit the description is still refused.
                with pytest.raises(ValueError, match="takes 3 values"):
                    session.publish_robot_state("arm", [0.0])
                session.step()
        assert fake.events == [
            "create",
            "set_robot_state",
            "start",
            "ready",
            "update",
            "update",
            "exit",
        ]
        assert fake.hello["robot_state"] == {"version": 1}
        assert fake.published == []
        assert _warnings(caplog) == [NOT_TAKEN]

    def test_a_description_that_cannot_be_loaded_fails_before_any_connection(
        self, tmp_path
    ):
        broken = tmp_path / "broken.toml"
        broken.write_text('robot = "r"\n[series.arm]\nshape = "wheels"\n')
        for path, error in (
            (tmp_path / "missing.toml", FileNotFoundError),
            (broken, RobotDescriptionError),
        ):
            with _fake(_Session(_ack(**AGREED))) as fake:
                session = TeleopSession(_config(robot_description=path))
                with pytest.raises(error):
                    session.__enter__()
            assert fake.events == []
            assert not session.robot_state_active
            with pytest.raises(RuntimeError, match="TeleopSession is not active"):
                session.publish_robot_state("arm", [0.0, 0.0, 0.0])

    def test_a_connection_made_again_starts_robot_state_once(self):
        """After a server closed the connection unanswered, start() runs again on the session."""
        fake = _Session(_ack(**AGREED))
        fake.start_errors = [RemotePeerClosedError("closed unanswered")]
        with _fake(fake), patch.object(_remote, "CONNECT_RETRY_S", 0.0):
            with TeleopSession(_config(robot_description=DESCRIPTION)) as session:
                assert session.robot_state_active
        assert fake.events == [
            "create",
            "set_robot_state",
            "start",
            "start",
            "start_robot_state",
            "publish robot_description",
            "ready",
            "update",
            "exit",
        ]

    def test_a_session_entered_again_declares_again(self):
        first, second = _Session(_ack(**AGREED)), _Session(_ack())
        session = TeleopSession(_config(robot_description=DESCRIPTION))
        with _fake(first), session:
            assert session.robot_state_active
        with _fake(second), session:
            assert not session.robot_state_active
            assert not session.publish_robot_state("arm", [0.0, 0.0, 0.0])
        assert second.events[:3] == ["create", "set_robot_state", "start"]
        assert second.published == []


# ============================================================================
# Over a real connection
# ============================================================================


class _MiniServer:
    """Accepts one robot app on a thread, answers its HELLO, reports READY and idles.

    `answer` are the HELLO_ACK's extra fields. With `store`, the host attaches it before the
    answer, whatever the answer says: a store then shows every robot-state link message sent.
    """

    def __init__(self, *, answer=None, store=None):
        self.listener = RemoteSessionListener("127.0.0.1", 0)
        self.endpoint = f"tcp://127.0.0.1:{self.listener.port}"
        self.answer = answer or {}
        self.store = store
        self.hello_text = None
        self.error = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            accepted = self.listener.accept(WAIT_MS)
            assert accepted is not None, "no robot connected"
            with RemoteSessionHost.adopt(accepted[0], False) as host:
                self.hello_text = host.wait_hello(WAIT_MS)
                assert self.hello_text is not None, "no HELLO"
                if self.store is not None:
                    host.attach_robot_state(self.store)
                host.send_hello_ack(json.dumps(_ack(**self.answer)))
                host.send_ready(json.dumps(READY))
                while not self._stop.is_set() and host.alive:
                    time.sleep(0.005)
        except Exception as e:  # reported by stop()
            self.error = e

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=10)
        assert not self._thread.is_alive()
        if self.error is not None:
            raise self.error


def _wait_for(predicate, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.005)
    return True


def _loopback_config(server, **kwargs):
    return TeleopSessionConfig(
        app_name="RobotStateLoopback",
        pipeline=_Pipeline(),
        mode=SessionMode.REMOTE,
        remote_endpoint=server.endpoint,
        remote_connect_timeout_s=10.0,
        verbose=False,
        **kwargs,
    )


class TestLoopback:
    def test_new_robot_app_and_new_server(self, description_file):
        store = RobotStateStore()
        server = _MiniServer(answer=AGREED, store=store)
        try:
            config = _loopback_config(server, robot_description=description_file)
            with TeleopSession(config) as session:
                assert json.loads(server.hello_text)["robot_state"] == {"version": 1}
                assert session.robot_state_active
                # Declared and described at HELLO_ACK, before any reading is published.
                assert _wait_for(
                    lambda: store.latest("robot_description").data is not None
                )
                assert [info.name for info in store.series()] == SERIES_NAMES
                described = records.decode_description(
                    store.latest("robot_description").data
                )
                assert described == DESCRIPTION
                assert store.latest("robot_description").fresh

                assert session.publish_robot_state("arm", [0.25, -0.5, 1.0])
                assert session.publish_robot_state(
                    "status", [True, False], faults=[("overtemperature", "hot")]
                )
                assert session.publish_robot_record(
                    "wrist_force", {"fx": 1.0, "fy": 2.0, "contact": True}
                )
                assert _wait_for(
                    lambda: all(
                        store.latest(name).data is not None for name in SERIES_NAMES
                    )
                )
                arm = store.latest("arm")
                assert [j.position for j in records.decode_joints(arm.data)] == [
                    0.25,
                    -0.5,
                    1.0,
                ]
                assert arm.fresh
                assert arm.age_ns is not None
                assert 0 <= arm.age_ns < 2_000_000_000
                readings, faults = records.decode_status(store.latest("status").data)
                assert readings == {"motors_on": True, "fault": False}
                assert [f.code for f in faults] == ["overtemperature"]
                assert records.decode_custom(store.latest("wrist_force").data) == {
                    "fx": 1.0,
                    "fy": 2.0,
                    "contact": True,
                }
                session.step()
                assert store.connected
                assert store.dropped_records == 0
            assert not session.robot_state_active
        finally:
            server.stop()
        assert not store.connected

    @pytest.mark.parametrize(
        "answer", [{}, {"robot_state": {"version": 2}}], ids=["old server", "refused"]
    )
    def test_new_robot_app_and_a_server_that_does_not_agree(self, answer, caplog):
        """The HELLO offers robot state; without the agreement no robot-state link message leaves."""
        caplog.set_level(logging.WARNING)
        store = RobotStateStore()
        server = _MiniServer(answer=answer, store=store)
        try:
            config = _loopback_config(server, robot_description=DESCRIPTION)
            with TeleopSession(config) as session:
                assert json.loads(server.hello_text)["robot_state"] == {"version": 1}
                assert not session.robot_state_active
                assert not session.publish_robot_state("arm", [0.25, -0.5, 1.0])
                assert not session.publish_robot_record("wrist_force", [1.0, 2.0, True])
                for _ in range(20):
                    session.step()
                    time.sleep(0.005)
                assert store.series() == []
                assert store.dropped_records == 0
            assert _warnings(caplog) == [
                f"The Teleop Session Server at {server.endpoint} does not take robot "
                "state; none is sent"
            ]
        finally:
            server.stop()

    def test_old_robot_app_and_new_server(self):
        """Without a description the HELLO is the one of a robot app built before robot state."""
        store = RobotStateStore()
        server = _MiniServer(answer=AGREED, store=store)
        try:
            with TeleopSession(_loopback_config(server)) as session:
                assert server.hello_text == json.dumps(
                    {
                        "protocol": 1,
                        "isaaccapture_version": isaaccapture.__version__,
                        "app_name": "RobotStateLoopback",
                        "tick_rate_hz": 90.0,
                        "trackers": [],
                        "topics": [],
                    }
                )
                assert not session.robot_state_active
                for _ in range(20):
                    session.step()
                    time.sleep(0.005)
                assert store.series() == []
                assert store.dropped_records == 0
        finally:
            server.stop()
