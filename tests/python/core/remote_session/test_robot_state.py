# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Robot state through the session bindings: a RemoteSession publishes, a RemoteSessionHost stores.

Both ends run in one process over loopback, the server on a thread. No tracker runs.
"""

import json
import os
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from isaaccapture.deviceio_session import (
    RemotePeerClosedError,
    RemoteSession,
    RemoteSessionHost,
    RemoteSessionListener,
)
from isaaccapture.robot_state import RobotStateStore

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="remote sessions need POSIX sockets"
)

WAIT_MS = 5000
HELLO = json.dumps({"protocol": 1, "topics": [], "robot_state": {"version": 1}})
HELLO_ACK = {"session_id": "test", "server_version": "0"}
MAX_RECORD_BYTES = 1 << 20
TRANSPORTS = pytest.mark.parametrize("websocket", [False, True], ids=["tcp", "ws"])


def _wait_for(predicate, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.002)
    return True


def _series(name, in_order=False):
    """What set_robot_state() reads: any object with these attributes."""
    return SimpleNamespace(
        name=name,
        shape="test",
        schema_name=f"test.{name}",
        schema_encoding="jsonschema",
        schema_data=b"{}",
        message_encoding="json",
        stale_after_ns=500_000_000,
        in_order=in_order,
        rate_hz=20.0,
    )


SERIES = [_series("robot_description", in_order=True), _series("arm")]


class _Server:
    """Accepts one robot on a thread and answers its HELLO.

    With `store` the host attaches it before HELLO_ACK and the answer takes robot state, as the
    Teleop Session Server does; without, the answer is the one of a server built before robot state.
    """

    def __init__(self, store=None, websocket=False):
        self.listener = RemoteSessionListener("127.0.0.1", 0)
        scheme = "ws" if websocket else "tcp"
        self.endpoint = f"{scheme}://127.0.0.1:{self.listener.port}"
        self.store = store
        self.websocket = websocket
        self.host = None
        self.hello = None
        self.error = None
        self.served = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            accepted = self.listener.accept(WAIT_MS)
            assert accepted is not None, "no robot connected"
            with RemoteSessionHost.adopt(accepted[0], self.websocket) as host:
                self.host = host
                self.hello = json.loads(host.wait_hello(WAIT_MS))
                ack = dict(HELLO_ACK)
                if self.store is not None:
                    host.attach_robot_state(self.store)
                    ack["robot_state"] = {"version": 1}
                host.send_hello_ack(json.dumps(ack))
                self.served.set()
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


def _text(store, name):
    data = store.latest(name).data
    return None if data is None else data.decode()


def _now_ns():
    return time.monotonic_ns()


# ============================================================================
# Robot end to server end
# ============================================================================


@TRANSPORTS
def test_robot_state_reaches_the_store(websocket):
    store = RobotStateStore()
    server = _Server(store, websocket)
    with RemoteSession.create([]) as session:
        session.set_robot_state(SERIES)
        session.connect(server.endpoint, HELLO, WAIT_MS)
        assert json.loads(session.hello_ack())["robot_state"] == {"version": 1}
        assert store.connected
        assert not session.robot_state_active

        session.start_robot_state()
        assert session.robot_state_active
        assert _wait_for(lambda: len(store.series()) == 2)
        description, arm = store.series()
        assert (description.name, description.in_order) == ("robot_description", True)
        assert (arm.name, arm.shape, arm.in_order) == ("arm", "test", False)
        assert (arm.schema_name, arm.schema_encoding) == ("test.arm", "jsonschema")
        assert (arm.schema_data, arm.message_encoding) == (b"{}", "json")
        assert (arm.stale_after_ns, arm.rate_hz) == (500_000_000, 20.0)

        sample_time_ns = _now_ns()
        assert session.publish_robot_record(
            "robot_description", b"what", sample_time_ns
        )
        assert session.publish_robot_record("arm", b"joints 1", sample_time_ns)
        assert _wait_for(lambda: _text(store, "arm") == "joints 1")
        assert _text(store, "robot_description") == "what"
        record = store.latest("arm")
        assert record.sequence == 1
        assert record.sample_time_ns == sample_time_ns
        assert record.fresh
        # Both ends are on this machine's clock, so the age is that of the reading.
        assert record.age_ns is not None
        assert 0 <= record.age_ns < 2_000_000_000

        assert session.publish_robot_record("arm", b"joints 2", _now_ns())
        assert _wait_for(lambda: _text(store, "arm") == "joints 2")
        assert store.latest("arm").sequence == 2
        assert store.dropped_records == 0
    server.stop()
    # The host closed: the robot app left, what it sent stays.
    assert not store.connected
    assert _text(store, "arm") == "joints 2"


def test_the_series_of_a_robot_description_are_declared():
    from isaaccapture.robot_state import parse_robot_description, records

    description = parse_robot_description(
        'robot = "one_joint"\n'
        "[series.arm]\n"
        'shape = "joints"\n'
        'joints = [{ name = "j1", kind = "revolute", min = -1.0, max = 1.0 }]\n'
    )
    store = RobotStateStore()
    server = _Server(store)
    with RemoteSession.create([]) as session:
        session.set_robot_state(records.declarations(description))
        session.connect(server.endpoint, HELLO, WAIT_MS)
        session.start_robot_state()
        now_ns = _now_ns()
        assert session.publish_robot_record(
            records.DESCRIPTION_TOPIC,
            records.encode_description(description, now_ns),
            now_ns,
        )
        assert session.publish_robot_record(
            "arm",
            records.encode_reading(description, "arm", [0.25], time_ns=now_ns),
            now_ns,
        )
        assert _wait_for(lambda: store.latest("arm").data is not None)
        assert [info.name for info in store.series()] == ["robot_description", "arm"]
        assert [info.shape for info in store.series()] == ["description", "joints"]
        assert (
            records.decode_description(store.latest("robot_description").data)
            == description
        )
        (joint,) = records.decode_joints(store.latest("arm").data)
        assert (joint.name, joint.position, joint.valid) == ("j1", 0.25, True)
    server.stop()


def test_nothing_is_sent_to_a_server_that_did_not_agree():
    """A server built before robot state: its HELLO_ACK lacks the key and the app starts nothing."""
    server = _Server()
    with RemoteSession.create([]) as session:
        session.set_robot_state(SERIES)
        session.connect(server.endpoint, HELLO, WAIT_MS)
        assert json.loads(session.hello_ack()) == HELLO_ACK
        assert not session.robot_state_active
        assert not session.publish_robot_record("arm", b"joints", _now_ns())
        # A store attached now would take whatever the robot sent: it sends nothing.
        store = RobotStateStore()
        server.host.attach_robot_state(store)
        assert not session.publish_robot_record("arm", b"joints", _now_ns())
        time.sleep(0.1)
        assert store.series() == []
        assert store.dropped_records == 0
        assert server.host.alive
    server.stop()


def test_a_host_without_a_store_drops_robot_state():
    server = _Server()
    with RemoteSession.create([]) as session:
        session.set_robot_state(SERIES)
        session.connect(server.endpoint, HELLO, WAIT_MS)
        session.start_robot_state()
        assert session.publish_robot_record("arm", b"joints 1", _now_ns())
        store = RobotStateStore()
        # The SETUP went by unread, so the next record is on a channel the store does not know.
        time.sleep(0.1)
        server.host.attach_robot_state(store)
        assert session.publish_robot_record("arm", b"joints 2", _now_ns())
        assert _wait_for(lambda: store.dropped_records == 1)
        assert store.series() == []
        assert server.host.alive
        session.update()
    server.stop()


def test_publishing_returns_false_once_the_link_is_down():
    store = RobotStateStore()
    server = _Server(store)
    with RemoteSession.create([]) as session:
        session.set_robot_state(SERIES)
        session.connect(server.endpoint, HELLO, WAIT_MS)
        session.start_robot_state()
        assert session.publish_robot_record("arm", b"joints", _now_ns())
        assert _wait_for(lambda: _text(store, "arm") == "joints")
        server.stop()
        assert _wait_for(
            lambda: not session.publish_robot_record("arm", b"late", _now_ns())
        )
        with pytest.raises(RuntimeError, match="Remote session lost"):
            session.update()
    assert not session.robot_state_active


# ============================================================================
# The calls
# ============================================================================


def test_before_a_connection_nothing_is_active():
    with RemoteSession.create([]) as session:
        assert not session.robot_state_active
        assert not session.publish_robot_record("arm", b"joints", _now_ns())
        session.set_robot_state(SERIES)
        assert not session.robot_state_active
        assert not session.publish_robot_record("arm", b"joints", _now_ns())
        with pytest.raises(
            RuntimeError,
            match=r"^RemoteSession is not connected; call connect\(\) first$",
        ):
            session.start_robot_state()


def test_a_series_that_cannot_be_declared_is_refused_and_the_others_kept():
    store = RobotStateStore()
    server = _Server(store)
    with RemoteSession.create([]) as session:
        session.set_robot_state([_series("arm")])
        with pytest.raises(ValueError, match="'leg'"):
            session.set_robot_state([_series("leg"), _series("leg")])
        with pytest.raises(ValueError, match="RobotStatePublisher"):
            session.set_robot_state([_series("")])
        with pytest.raises(AttributeError):
            session.set_robot_state([SimpleNamespace(name="leg")])
        with pytest.raises(TypeError):
            session.set_robot_state(None)
        session.connect(server.endpoint, HELLO, WAIT_MS)
        session.start_robot_state()
        assert _wait_for(lambda: len(store.series()) == 1)
        assert store.series()[0].name == "arm"
    server.stop()


def test_the_series_cannot_change_once_robot_state_started():
    store = RobotStateStore()
    server = _Server(store)
    with RemoteSession.create([]) as session:
        session.set_robot_state([_series("arm")])
        session.connect(server.endpoint, HELLO, WAIT_MS)
        # Still allowed on a connection that has not started robot state.
        session.set_robot_state(SERIES)
        session.start_robot_state()
        session.start_robot_state()
        with pytest.raises(RuntimeError, match="after start_robot_state"):
            session.set_robot_state([_series("leg")])
        assert _wait_for(lambda: len(store.series()) == 2)
        assert [info.name for info in store.series()] == ["robot_description", "arm"]
    server.stop()
    with pytest.raises(RuntimeError, match="closed"):
        session.set_robot_state(SERIES)
    with pytest.raises(RuntimeError, match="closed"):
        session.start_robot_state()
    assert not session.robot_state_active
    assert not session.publish_robot_record("arm", b"joints", _now_ns())


def test_a_record_the_publisher_refuses_raises():
    store = RobotStateStore()
    server = _Server(store)
    with RemoteSession.create([]) as session:
        session.set_robot_state(SERIES)
        session.connect(server.endpoint, HELLO, WAIT_MS)
        session.start_robot_state()
        with pytest.raises(ValueError, match="'leg'"):
            session.publish_robot_record("leg", b"x", _now_ns())
        with pytest.raises(ValueError, match="arm"):
            session.publish_robot_record("arm", bytes(MAX_RECORD_BYTES + 1), _now_ns())
        with pytest.raises(TypeError):
            session.publish_robot_record("arm", "text", _now_ns())
        assert session.publish_robot_record("arm", bytes(MAX_RECORD_BYTES), _now_ns())
        assert _wait_for(lambda: store.latest("arm").data is not None)
        assert len(store.latest("arm").data) == MAX_RECORD_BYTES
        session.update()
    server.stop()


# ============================================================================
# A session that connects more than once
# ============================================================================


def test_a_start_after_a_refused_connection_still_sends_the_series():
    closed = RemoteSessionListener("127.0.0.1", 0)
    nobody = f"tcp://127.0.0.1:{closed.port}"
    closed.close()
    store = RobotStateStore()
    server = _Server(store)
    with RemoteSession.create([]) as session:
        session.set_robot_state(SERIES)
        with pytest.raises(RuntimeError, match=r"^cannot connect to "):
            session.start(nobody, HELLO, 1000)
        assert not session.robot_state_active
        session.start(server.endpoint, HELLO, WAIT_MS)
        assert session.wait_hello_ack(WAIT_MS)
        session.start_robot_state()
        assert _wait_for(lambda: len(store.series()) == 2)
    server.stop()


def test_a_start_after_a_server_closed_unanswered_still_sends_the_series():
    """What TeleopSession does after a RemotePeerClosedError: start() again on the same session."""
    dying = RemoteSessionListener("127.0.0.1", 0)

    def close_unanswered():
        accepted = dying.accept(WAIT_MS)
        if accepted is not None:
            os.close(accepted[0])

    closer = threading.Thread(target=close_unanswered, daemon=True)
    closer.start()
    store = RobotStateStore()
    server = _Server(store)
    with RemoteSession.create([]) as session:
        session.set_robot_state(SERIES)
        session.start(f"tcp://127.0.0.1:{dying.port}", HELLO, WAIT_MS)
        with pytest.raises(RemotePeerClosedError):
            session.wait_hello_ack(WAIT_MS)
        closer.join(timeout=10)
        dying.close()
        with pytest.raises(RuntimeError, match="not connected"):
            session.start_robot_state()

        session.start(server.endpoint, HELLO, WAIT_MS)
        assert session.wait_hello_ack(WAIT_MS)
        session.start_robot_state()
        assert session.publish_robot_record("arm", b"joints", _now_ns())
        assert _wait_for(lambda: _text(store, "arm") == "joints")
        assert [info.name for info in store.series()] == ["robot_description", "arm"]
    server.stop()


# ============================================================================
# The host
# ============================================================================


def test_the_host_ends_the_store_session_once_per_attach():
    store = RobotStateStore()
    server = _Server(store)
    with RemoteSession.create([]) as session:
        session.connect(server.endpoint, HELLO, WAIT_MS)
        assert server.served.wait(5)
        host = server.host
        assert (store.connected, store.session_count) == (True, 1)

        host.attach_robot_state(None)
        assert (store.connected, store.session_count) == (False, 1)
        host.attach_robot_state(store)
        assert (store.connected, store.session_count) == (True, 2)
        other = RobotStateStore()
        host.attach_robot_state(other)
        assert not store.connected
        assert other.connected

        host.close()
        assert not other.connected
        # The next robot app's session is not ended by the host of the one before.
        other.begin_session()
        host.close()
        host.attach_robot_state(other)
        assert (other.connected, other.session_count) == (True, 2)
    server.stop()
    assert other.connected


def test_attach_takes_only_a_store_or_none():
    server = _Server()
    with RemoteSession.create([]) as session:
        session.connect(server.endpoint, HELLO, WAIT_MS)
        assert server.served.wait(5)
        with pytest.raises(TypeError):
            server.host.attach_robot_state("store")
        with pytest.raises(TypeError):
            server.host.attach_robot_state()
    server.stop()
