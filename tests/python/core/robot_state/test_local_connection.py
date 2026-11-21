# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reading robot state from another program: `RobotState.connect()` and the local service of
the program that serves it.

Every socket is a Unix socket in a temporary directory of the test.
"""

import json
import logging
import os
import select
import socket
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import repo_paths
from test_reader import put, rover, six_joint_arm, start_robot

from isaaccapture.robot_state import (
    RobotState,
    RobotStateLocalService,
    RobotStateStore,
    _local,
    default_socket_path,
    records,
)
from isaaccapture.robot_state._local import ANSWER, MAGIC, REQUEST

pytestmark = pytest.mark.skipif(
    not _local._HAS_UNIX_SOCKETS, reason="the reader socket is a Unix domain socket"
)

ZEROS = [0.0, 0.0, 0.0]
READER_LOGGER = "isaaccapture.robot_state.reader"


def wait_for(condition, timeout_s=20.0):
    """The first true value of `condition()`; fails when none comes in time."""
    deadline = time.monotonic() + timeout_s
    while True:
        value = condition()
        if value:
            return value
        assert time.monotonic() < deadline, "the awaited state did not come"
        time.sleep(0.01)


@pytest.fixture
def socket_path(tmp_path_factory):
    # Shorter than `tmp_path`, which holds the test's name: a Unix socket path is limited.
    return str(tmp_path_factory.mktemp("rs") / "rs.sock")


@pytest.fixture
def store():
    return RobotStateStore()


@pytest.fixture
def serve():
    """Start services that are closed when the test ends."""
    services = []

    def start(store, path):
        service = RobotStateLocalService(store, path)
        services.append(service)
        service.start()
        return service

    yield start
    for service in services:
        service.close()


@pytest.fixture
def connect():
    """Make readers that are closed when the test ends."""
    readers = []

    def make(path, **options):
        reader = RobotState.connect(path, **options)
        readers.append(reader)
        return reader

    yield make
    for reader in readers:
        reader.close()


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.INFO, logger="isaaccapture.robot_state")
    return caplog


def messages(caplog, level, logger=None):
    return [
        entry.getMessage()
        for entry in caplog.records
        if entry.levelno == level and logger in (None, entry.name)
    ]


def raw_connection(path):
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(10.0)
    connection.connect(path)
    return connection


def receive(connection, count):
    data = b""
    while len(data) < count:
        chunk = connection.recv(count - len(data))
        assert chunk, f"the connection closed after {len(data)} of {count} bytes"
        data += chunk
    return data


def ask(connection, known_setup_generation=0):
    """One request on a raw connection; the snapshot of its answer."""
    connection.sendall(REQUEST.pack(MAGIC, 1, 1, 0, known_setup_generation))
    magic, version, op, zero, length = ANSWER.unpack(receive(connection, ANSWER.size))
    assert (magic, version, op, zero) == (MAGIC, 1, 2, 0)
    return receive(connection, length)


class FakeServer:
    """A server whose answers the test writes: `answer(request)` gives the bytes to send, or
    None to stay silent."""

    def __init__(self, path, answer):
        self._answer = answer
        self._stop = threading.Event()
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(path)
        self._listener.listen(8)
        self._listener.settimeout(0.05)
        self.accepted = 0
        self._threads = [threading.Thread(target=self._accept, daemon=True)]
        self._threads[0].start()

    def _accept(self):
        while not self._stop.is_set():
            try:
                connection, _ = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            self.accepted += 1
            thread = threading.Thread(
                target=self._talk, args=(connection,), daemon=True
            )
            self._threads.append(thread)
            thread.start()

    def _talk(self, connection):
        with connection:
            connection.settimeout(0.05)
            while not self._stop.is_set():
                try:
                    request = connection.recv(REQUEST.size)
                except TimeoutError:
                    continue
                except OSError:
                    return
                if not request:
                    return
                reply = self._answer(request)
                if reply is None:
                    self._stop.wait()
                    return
                connection.sendall(reply)

    def close(self):
        self._stop.set()
        self._listener.close()
        for thread in self._threads:
            thread.join(10.0)


@pytest.fixture
def fake_server():
    servers = []

    def start(path, answer):
        server = FakeServer(path, answer)
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.close()


def put_arm(store, description, lift):
    put(store, description, "arm", [lift, 0.02, 0.0], velocities=ZEROS)


# ---------------------------------------------------------------------------------------------
# Readers and servers that come and go
# ---------------------------------------------------------------------------------------------


def test_a_reader_that_starts_before_the_server_sees_it_once_it_serves(
    socket_path, store, serve, connect, logs
):
    reader = connect(socket_path)
    assert reader.server_reachable is False
    assert reader.connected is False
    assert reader.description() is None
    assert reader.series() == {}
    assert reader.joints("arm") is None
    assert reader.gripper("jaw", allow_stale=True) is None
    assert reader.status() is None
    assert reader.record("sonar") is None
    assert reader.refresh() is False

    # The server runs and a robot app sends, but the server has not opened the socket yet.
    description = rover()
    start_robot(store, description)
    put_arm(store, description, 0.25)
    assert reader.server_reachable is False
    assert reader.joints("arm") is None

    serve(store, socket_path)
    assert reader.refresh() is True
    assert reader.server_reachable is True
    assert reader.connected is True
    assert reader.description() == description
    arm = reader.joints("arm")
    assert arm["lift"].position == pytest.approx(0.25)
    assert 0.0 <= arm.age_s < 30.0
    assert reader.gripper("jaw").opening == pytest.approx(0.5, abs=1e-5)

    # One line per change of what the reader can reach and one per robot, however often it asks.
    assert messages(logs, logging.INFO, READER_LOGGER) == [
        f"Nothing serves robot state at {socket_path} "
        "([Errno 2] No such file or directory)",
        f"Reading robot state at {socket_path}",
        f"Robot state at {socket_path}: robot 'rover'",
    ]
    assert messages(logs, logging.WARNING) == []


def test_a_server_that_stops_takes_everything_it_served_with_it(
    socket_path, store, serve, connect, logs
):
    description = rover()
    start_robot(store, description)
    put_arm(store, description, 0.25)
    put(store, description, "health", [True, 12.0, "idle"])
    service = serve(store, socket_path)
    reader = connect(socket_path)
    assert reader.joints("arm") is not None
    assert reader.status("health")["mode"] == "idle"

    service.close()
    assert not os.path.exists(socket_path)
    assert reader.refresh() is False
    assert reader.server_reachable is False
    assert reader.connected is False
    assert reader.description() is None
    assert reader.series() == {}
    # Nothing of the lost server is readable, not even as a stale reading.
    assert reader.joints("arm", allow_stale=True) is None
    assert reader.gripper("jaw", allow_stale=True) is None
    assert reader.status("health", allow_stale=True) is None
    assert reader.record("sonar", allow_stale=True) is None

    # The server runs again, with no robot app yet.
    again = RobotStateStore()
    serve(again, socket_path)
    assert reader.refresh() is True
    assert reader.server_reachable is True
    assert reader.connected is False
    assert reader.description() is None

    second = six_joint_arm()
    start_robot(again, second)
    put(again, second, "status", [True, ""])
    assert reader.connected is True
    assert reader.description().robot == "rebot_b601"
    assert reader.status(allow_stale=True)["motors_on"] is True

    infos = messages(logs, logging.INFO, READER_LOGGER)
    assert [text.split(" at ")[0] for text in infos] == [
        "Reading robot state",
        "Robot state",
        "Nothing serves robot state",
        "Reading robot state",
        "Robot state",
    ]
    assert infos[1].endswith(": robot 'rover'")
    assert infos[4].endswith(": robot 'rebot_b601'")


def test_a_server_that_restarts_between_two_reads_is_read_at_once(
    socket_path, store, serve, connect
):
    description = rover()
    start_robot(store, description)
    put_arm(store, description, 0.25)
    service = serve(store, socket_path)
    reader = connect(socket_path)
    assert reader.joints("arm") is not None

    service.close()
    serve(store, socket_path)
    # The kept connection is gone; the same refresh makes a new one.
    assert reader.refresh() is True
    assert reader.server_reachable is True
    assert reader.joints("arm")["lift"].position == pytest.approx(0.25)


def test_two_readers_at_once_read_the_same(socket_path, store, serve, connect):
    description = rover()
    start_robot(store, description)
    put_arm(store, description, 0.25)
    serve(store, socket_path)
    one = connect(socket_path)
    other = connect(socket_path)

    for lift in (0.05, 0.1, 0.15):
        put_arm(store, description, lift)
        first, second = one.joints("arm"), other.joints("arm")
        assert first["lift"].position == pytest.approx(lift)
        assert first.joints == second.joints
        assert first.sequence == second.sequence
        assert first.sample_time_ns == second.sample_time_ns
    assert one.description() == other.description() == description

    one.close()
    assert one.server_reachable is False
    assert one.joints("arm", allow_stale=True) is None
    put_arm(store, description, 0.3)
    assert other.joints("arm")["lift"].position == pytest.approx(0.3)


def test_a_late_reader_still_gets_the_description(socket_path, store, serve, connect):
    description = rover()
    start_robot(store, description)
    serve(store, socket_path)
    early = connect(socket_path)
    assert early.description() == description
    # The robot app has been sending for a while; its description went out once, at the start.
    for count in range(50):
        put_arm(store, description, count / 1000)
        put(store, description, "health", [True, 12.0, "idle"])

    late = connect(socket_path)
    assert late.description() == description
    assert late.series()["arm"].joint_names == ["lift", "jaw", "thumb"]
    assert late.joints("arm").sequence == 50
    assert late.status("health")["volts"] == pytest.approx(12.0)
    assert late.gripper("pinch") is not None


def test_a_robot_app_that_leaves_and_the_next_one_are_seen_by_the_same_reader(
    socket_path, store, serve, connect
):
    first = rover()
    start_robot(store, first)
    put_arm(store, first, 0.25)
    put(store, first, "mast", [3.0])
    serve(store, socket_path)
    reader = connect(socket_path)
    assert reader.joints("mast")["turn"].position == pytest.approx(3.0)

    store.end_session()
    assert reader.server_reachable is True
    assert reader.connected is False
    assert reader.joints("arm") is None
    assert reader.joints("mast") is None
    stale = reader.joints("mast", allow_stale=True)
    assert stale["turn"].position == pytest.approx(3.0)
    assert stale.fresh is False
    assert reader.description() == first

    second = six_joint_arm()
    start_robot(store, second)
    assert reader.connected is True
    assert reader.description() == second
    assert reader.joints("arm") is None
    with pytest.raises(KeyError, match="has no series 'mast'"):
        reader.joints("mast")
    put(store, second, "arm", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0])
    arm = reader.joints("arm", allow_stale=True)
    assert arm.names[-1] == "gripper"
    assert arm.sequence == 1


def test_without_auto_refresh_a_reader_keeps_one_snapshot_until_it_refreshes(
    socket_path, store, serve, connect
):
    serve(store, socket_path)
    reader = connect(socket_path, auto_refresh=False)
    assert reader.server_reachable is True

    description = rover()
    start_robot(store, description)
    put_arm(store, description, 0.25)
    assert reader.connected is False
    assert reader.description() is None
    assert reader.refresh() is True
    assert reader.connected is True
    first = reader.joints("arm")
    assert first["lift"].position == pytest.approx(0.25)

    put_arm(store, description, 0.3)
    store.end_session()
    again = reader.joints("arm")
    assert again.joints == first.joints
    assert again.sequence == first.sequence
    assert again.age_s > first.age_s  # the same reading, older
    assert reader.connected is True
    assert reader.refresh() is True
    assert reader.connected is False
    assert reader.joints("arm", allow_stale=True)["lift"].position == pytest.approx(0.3)


def test_a_snapshot_that_is_not_refreshed_goes_stale(
    socket_path, store, serve, connect
):
    description = rover()
    start_robot(store, description)
    serve(store, socket_path)
    reader = connect(socket_path, auto_refresh=False)
    # "quick" goes stale after half a second; this reading already is two seconds old.
    put(store, description, "quick", [1.5], age_s=2.0)
    put(store, description, "mast", [3.0], age_s=2.0)
    reader.refresh()
    assert reader.joints("quick") is None
    quick = reader.joints("quick", allow_stale=True)
    assert quick.age_s >= 2.0
    assert quick.fresh is False
    time.sleep(0.05)
    assert reader.joints("quick", allow_stale=True).age_s >= quick.age_s + 0.05
    assert reader.joints("mast").fresh is True


def test_a_kept_reader_fetches_the_declarations_once(
    socket_path, store, serve, connect
):
    asked = []

    class Recording:
        def export_snapshot(self, known_setup_generation):
            asked.append(known_setup_generation)
            return store.export_snapshot(known_setup_generation)

    description = rover()
    start_robot(store, description)
    serve(Recording(), socket_path)
    reader = connect(socket_path)
    for lift in (0.1, 0.2, 0.3):
        put_arm(store, description, lift)
        assert reader.joints("arm")["lift"].position == pytest.approx(lift)
    known = store.setup_generation
    assert asked[0] == 0
    assert asked[1:] == [known] * (len(asked) - 1)
    assert len(asked) >= 4

    # The next robot app declares its series anew: the reader takes them and asks with the new number.
    second = six_joint_arm()
    start_robot(store, second)
    assert store.setup_generation > known
    del asked[:]
    assert reader.description() == second
    assert reader.description() == second
    assert asked == [known, store.setup_generation]


def test_one_thread_serves_every_reader(socket_path, store, serve, connect):
    def serving():
        return [
            thread
            for thread in threading.enumerate()
            if thread.name == "isaaccapture-robot-state-readers"
        ]

    assert serving() == []
    service = serve(store, socket_path)
    readers = [connect(socket_path) for _ in range(3)]
    assert all(reader.server_reachable for reader in readers)
    (thread,) = serving()
    assert thread.daemon is True
    service.start()  # a second start changes nothing
    assert serving() == [thread]
    service.close()
    assert serving() == []
    assert not thread.is_alive()


def test_a_large_robot_state_reaches_a_reader_while_another_does_not_take_its_answer(
    socket_path, store, serve, connect
):
    store.begin_session()
    schemas = {f"series_{index}": bytes([index]) * 700_000 for index in range(6)}
    for name, data in schemas.items():
        store.declare_series(name, "blob", f"vendor.{name}", "vendor", data, "vendor")
        store.put_record(name, name.encode() * 1000, 1)
    serve(store, socket_path)

    # A reader that asks and does not read: its answer does not fit the socket's buffer.
    stalled = raw_connection(socket_path)
    stalled.sendall(REQUEST.pack(MAGIC, 1, 1, 0, 0))
    time.sleep(0.1)

    reader = connect(socket_path)
    assert reader.server_reachable is True
    for name, data in schemas.items():
        assert reader._store.series_info(name).schema_data == data
        assert reader._store.latest(name).data == name.encode() * 1000

    magic, _version, op, _zero, length = ANSWER.unpack(receive(stalled, ANSWER.size))
    assert (magic, op) == (MAGIC, 2)
    assert length > 4_000_000
    snapshot = receive(stalled, length)
    stalled.close()
    copy = RobotStateStore()
    copy.import_snapshot(snapshot)
    assert [series.name for series in copy.series()] == list(schemas)


# ---------------------------------------------------------------------------------------------
# A reader that expects one robot
# ---------------------------------------------------------------------------------------------


def test_a_reader_that_expects_another_robot_reads_nothing(
    socket_path, store, serve, connect, logs
):
    description = rover()
    start_robot(store, description)
    put_arm(store, description, 0.25)
    put(store, description, "mast", [3.0])
    put(store, description, "health", [True, 12.0, "idle"])
    put(store, description, "sonar", [0.5, "wall"])
    serve(store, socket_path)

    reader = connect(socket_path, robot="rebot_b601")
    for _ in range(3):
        assert reader.refresh() is True
        assert reader.server_reachable is True
        assert reader.connected is False
        assert reader.description() is None
        assert reader.series() == {}
        for allow_stale in (False, True):
            assert reader.joints("arm", allow_stale=allow_stale) is None
            assert reader.joints("mast", allow_stale=allow_stale) is None
            assert reader.gripper("jaw", allow_stale=allow_stale) is None
            assert reader.status("health", allow_stale=allow_stale) is None
            assert reader.record("sonar", allow_stale=allow_stale) is None
    # A reader without the argument reads the robot that is served.
    assert connect(socket_path).joints("arm")["lift"].position == pytest.approx(0.25)
    assert messages(logs, logging.WARNING) == [
        f"Robot state at {socket_path} is of robot 'rover', not 'rebot_b601'; "
        "nothing is read"
    ]

    # The robot app leaves and the expected robot's app connects: the same reader reads it.
    second = six_joint_arm()
    start_robot(store, second)
    put(store, second, "arm", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0])
    put(store, second, "status", [True, ""])
    assert reader.connected is True
    assert reader.description() == second
    assert list(reader.series()) == list(second.series)
    assert reader.joints("arm")["joint2"].position == pytest.approx(0.1)
    assert reader.gripper("gripper") is not None
    assert reader.status()["motors_on"] is True
    assert reader.record("arm").sequence == 1
    assert len(messages(logs, logging.WARNING)) == 1

    # The other robot again: nothing is read, and it is said again.
    start_robot(store, description)
    put_arm(store, description, 0.3)
    assert reader.server_reachable is True
    assert reader.connected is False
    assert reader.description() is None
    assert reader.joints("arm", allow_stale=True) is None
    assert len(messages(logs, logging.WARNING)) == 2


def test_a_reader_that_expects_the_robot_served_reads_it(
    socket_path, store, serve, connect, logs
):
    description = rover()
    start_robot(store, description, send_description=False)
    serve(store, socket_path)
    reader = connect(socket_path, robot="rover")
    # Which robot it is, is not known before its description came.
    assert reader.server_reachable is True
    assert reader.connected is False
    assert reader.description() is None

    start_robot(store, description)
    put_arm(store, description, 0.25)
    assert reader.connected is True
    assert reader.description() == description
    assert reader.joints("arm")["lift"].position == pytest.approx(0.25)
    assert messages(logs, logging.WARNING) == []


def test_one_wrong_robot_is_reported_once_however_often_it_returns(
    socket_path, store, serve, connect, logs
):
    description = rover()
    serve(store, socket_path)
    reader = connect(socket_path, robot="rebot_b601")
    for _ in range(3):
        start_robot(store, description)
        assert reader.connected is False
        assert reader.description() is None
        store.end_session()
        assert reader.description() is None
    warnings = messages(logs, logging.WARNING)
    assert len(warnings) == 1
    assert "'rover'" in warnings[0] and "'rebot_b601'" in warnings[0]


def test_the_robot_that_is_read_is_logged_once_per_description(
    socket_path, store, serve, connect, logs
):
    def named():
        return [
            text
            for text in messages(logs, logging.INFO, READER_LOGGER)
            if text.startswith("Robot state at ")
        ]

    first = rover()
    start_robot(store, first)
    service = serve(store, socket_path)
    reader = connect(socket_path)
    assert named() == []  # nothing asked for the description yet
    for lift in (0.1, 0.2, 0.3):
        put_arm(store, first, lift)
        assert reader.description() == first
        assert reader.joints("arm") is not None
    assert named() == [f"Robot state at {socket_path}: robot 'rover'"]

    # The next robot app's description is another one, also of the same robot.
    start_robot(store, first)
    assert reader.description() == first
    start_robot(store, six_joint_arm())
    assert reader.description().robot == "rebot_b601"
    assert reader.description().robot == "rebot_b601"
    assert named() == [
        f"Robot state at {socket_path}: robot 'rover'",
        f"Robot state at {socket_path}: robot 'rover'",
        f"Robot state at {socket_path}: robot 'rebot_b601'",
    ]

    # Read again after the program that serves it was away.
    service.close()
    assert reader.description() is None
    serve(store, socket_path)
    assert reader.description().robot == "rebot_b601"
    assert len(named()) == 4


def test_a_reader_of_a_store_in_this_program_names_no_robot(store, logs):
    description = rover()
    start_robot(store, description)
    reader = RobotState.from_store(store)
    assert reader.description() == description
    assert reader.connected is True
    assert messages(logs, logging.INFO, READER_LOGGER) == []
    with pytest.raises(TypeError):
        RobotState.from_store(store, robot="rover")


# ---------------------------------------------------------------------------------------------
# Closing
# ---------------------------------------------------------------------------------------------


def test_a_closed_reader_reads_nothing_and_does_not_come_back(
    socket_path, store, serve
):
    description = rover()
    start_robot(store, description)
    put_arm(store, description, 0.25)
    serve(store, socket_path)

    with RobotState.connect(socket_path) as reader:
        assert reader.joints("arm") is not None
    assert reader.refresh() is False
    assert reader.server_reachable is False
    assert reader.connected is False
    assert reader.description() is None
    assert reader.joints("arm", allow_stale=True) is None
    reader.close()
    assert reader.refresh() is False


def test_a_service_is_closed_once_and_not_started_again(socket_path, store):
    never_started = RobotStateLocalService(store, socket_path)
    assert never_started.path == socket_path
    never_started.close()
    never_started.close()
    assert not os.path.exists(socket_path)
    with pytest.raises(RuntimeError, match="RobotStateLocalService is closed"):
        never_started.start()

    service = RobotStateLocalService(store, Path(socket_path))
    assert service.path == socket_path
    service.start()
    assert stat.S_ISSOCK(os.stat(socket_path).st_mode)
    service.close()
    service.close()
    assert not os.path.exists(socket_path)
    with pytest.raises(RuntimeError):
        service.start()


def test_closing_a_service_leaves_the_socket_of_its_successor(
    socket_path, store, serve, connect
):
    first = serve(store, socket_path)
    # The path is taken over by another program while the first still runs: somebody removed
    # the socket file and the lock file beside it.
    os.unlink(socket_path)
    os.unlink(socket_path + _local.LOCK_SUFFIX)
    serve(store, socket_path)
    first.close()
    assert os.path.exists(socket_path)
    assert connect(socket_path).server_reachable is True


# ---------------------------------------------------------------------------------------------
# Who may connect, and where
# ---------------------------------------------------------------------------------------------


def test_the_socket_and_a_directory_made_for_it_are_closed_to_other_users(
    tmp_path_factory, store, serve, connect
):
    base = tmp_path_factory.mktemp("rs")
    path = str(base / "made" / "deeper" / "rs.sock")
    serve(store, path)
    assert stat.S_IMODE(os.stat(base / "made").st_mode) == 0o700
    assert stat.S_IMODE(os.stat(base / "made" / "deeper").st_mode) == 0o700
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert connect(path).server_reachable is True


@pytest.mark.parametrize("mode", [0o750, 0o705, 0o777])
def test_a_directory_open_to_group_or_others_is_refused(tmp_path_factory, store, mode):
    directory = tmp_path_factory.mktemp("rs") / "open"
    directory.mkdir()
    directory.chmod(mode)
    service = RobotStateLocalService(store, directory / "rs.sock")
    with pytest.raises(OSError, match="is open to group or others"):
        service.start()
    assert list(directory.iterdir()) == []
    service.close()


def test_a_directory_of_another_user_is_refused(socket_path, store, monkeypatch):
    service = RobotStateLocalService(store, socket_path)
    uid = os.getuid()
    with monkeypatch.context() as patched:
        patched.setattr(_local.os, "getuid", lambda: uid + 1)
        with pytest.raises(OSError, match="belongs to another user"):
            service.start()
    assert not os.path.exists(socket_path)
    service.close()


def test_a_reader_of_another_user_is_closed(socket_path, store, serve, connect, logs):
    start_robot(store, rover())
    service = serve(store, socket_path)
    # The service takes every peer for another user's.
    service._uid += 1
    reader = connect(socket_path)
    assert reader.server_reachable is False
    assert reader.description() is None
    assert reader.refresh() is False
    warnings = messages(logs, logging.WARNING)
    assert len(warnings) == 1
    assert "reader of another user" in warnings[0]

    service._uid -= 1
    assert reader.refresh() is True
    assert reader.description().robot == "rover"


def test_a_path_another_server_serves_is_refused(socket_path, store, serve, connect):
    start_robot(store, rover())
    serve(store, socket_path)
    second = RobotStateLocalService(RobotStateStore(), socket_path)
    with pytest.raises(OSError) as raised:
        second.start()
    assert str(raised.value) == f"another program serves robot state at {socket_path}"
    second.close()
    # The first one is untouched.
    assert os.path.exists(socket_path)
    assert connect(socket_path).description().robot == "rover"


def test_a_socket_file_nobody_serves_is_replaced(socket_path, store, serve, connect):
    left_behind = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    left_behind.bind(socket_path)
    left_behind.close()
    assert os.path.exists(socket_path)

    reader = connect(socket_path)
    assert reader.server_reachable is False
    serve(store, socket_path)
    assert reader.refresh() is True


def test_a_file_that_is_not_a_socket_is_left_alone(socket_path, store):
    Path(socket_path).write_text("notes")
    service = RobotStateLocalService(store, socket_path)
    with pytest.raises(OSError, match="exists and is not a socket"):
        service.start()
    assert Path(socket_path).read_text() == "notes"
    service.close()
    assert Path(socket_path).read_text() == "notes"


def test_a_socket_path_above_the_limit(tmp_path_factory, store, logs):
    base = tmp_path_factory.mktemp("rs")
    path = str(base / ("d" * 100) / "rs.sock")
    assert len(os.fsencode(path)) > 103

    service = RobotStateLocalService(store, path)
    with pytest.raises(OSError) as raised:
        service.start()
    assert path in str(raised.value)
    assert "over the 103 a Unix domain socket allows" in str(raised.value)
    assert list(base.iterdir()) == []
    service.close()

    # A reader takes such a path as a server that is not running, and says so once.
    with RobotState.connect(path) as reader:
        assert reader.server_reachable is False
        assert reader.refresh() is False
        assert reader.description() is None
        assert reader.joints("arm") is None
    warnings = messages(logs, logging.WARNING)
    assert len(warnings) == 1
    assert "over the 103 a Unix domain socket allows" in warnings[0]


def test_the_default_socket_is_in_the_user_s_runtime_directory(tmp_path, monkeypatch):
    fallback = f"/tmp/isaaccapture-{os.getuid()}/robot_state.sock"
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    assert default_socket_path() == str(tmp_path / "isaaccapture" / "robot_state.sock")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "missing"))
    assert default_socket_path() == fallback
    monkeypatch.setenv("XDG_RUNTIME_DIR", "")
    assert default_socket_path() == fallback
    monkeypatch.delenv("XDG_RUNTIME_DIR")
    assert default_socket_path() == fallback
    # Asking for the path creates nothing.
    assert list(tmp_path.iterdir()) == []


def test_a_server_and_a_reader_meet_at_the_default_socket(
    tmp_path_factory, store, monkeypatch
):
    runtime = tmp_path_factory.mktemp("run")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    start_robot(store, rover())
    service = RobotStateLocalService(store)
    assert service.path == str(runtime / "isaaccapture" / "robot_state.sock")
    service.start()
    try:
        assert stat.S_IMODE(os.stat(runtime / "isaaccapture").st_mode) == 0o700
        with RobotState.connect() as reader:
            assert reader.server_reachable is True
            assert reader.description().robot == "rover"
    finally:
        service.close()


# ---------------------------------------------------------------------------------------------
# The exchange itself
# ---------------------------------------------------------------------------------------------


def test_a_request_is_answered_with_a_snapshot_of_the_store(socket_path, store, serve):
    description = rover()
    start_robot(store, description)
    put_arm(store, description, 0.25)
    serve(store, socket_path)

    with raw_connection(socket_path) as connection:
        copy = RobotStateStore()
        copy.import_snapshot(ask(connection))
        assert [series.name for series in copy.series()] == [
            series.name for series in store.series()
        ]
        assert copy.connected is True
        assert copy.session_count == store.session_count
        assert copy.setup_generation == store.setup_generation
        assert copy.latest("arm").data == store.latest("arm").data

        # With the number it holds, the declarations are left out of the answer.
        full = len(ask(connection, 0))
        short = ask(connection, store.setup_generation)
        assert len(short) < full
        copy.import_snapshot(short)
        assert len(copy.series()) == len(store.series())

        # Two requests sent together, and one sent in two parts, are each answered.
        request = REQUEST.pack(MAGIC, 1, 1, 0, 0)
        connection.sendall(request + request)
        for _ in range(2):
            header = ANSWER.unpack(receive(connection, ANSWER.size))
            assert len(receive(connection, header[4])) == full
        connection.sendall(request[:7])
        time.sleep(0.05)
        connection.sendall(request[7:])
        header = ANSWER.unpack(receive(connection, ANSWER.size))
        assert len(receive(connection, header[4])) == full


@pytest.mark.parametrize(
    "request_bytes",
    [
        REQUEST.pack(b"HTTP", 1, 1, 0, 0),
        REQUEST.pack(MAGIC, 2, 1, 0, 0),
        REQUEST.pack(MAGIC, 0, 1, 0, 0),
        REQUEST.pack(MAGIC, 1, 2, 0, 0),
        REQUEST.pack(MAGIC, 1, 0, 0, 0),
        b"GET / HTTP/1.1\r\n\r\n",
    ],
    ids=["magic", "newer-version", "version-0", "op-2", "op-0", "http"],
)
def test_a_request_in_another_format_closes_that_connection_only(
    socket_path, store, serve, connect, request_bytes
):
    start_robot(store, rover())
    serve(store, socket_path)
    reader = connect(socket_path)
    assert reader.server_reachable is True

    with raw_connection(socket_path) as connection:
        connection.sendall(request_bytes)
        assert connection.recv(64) == b""
    assert reader.refresh() is True
    assert reader.description().robot == "rover"


def test_readers_above_the_limit_are_closed(socket_path, store, serve, connect, logs):
    start_robot(store, rover())
    serve(store, socket_path)
    held = [raw_connection(socket_path) for _ in range(_local.MAX_CLIENTS)]
    try:
        for connection in held:
            assert ask(connection)
        with raw_connection(socket_path) as one_more:
            assert one_more.recv(64) == b""
        reader = connect(socket_path)
        assert reader.server_reachable is False
        warnings = messages(logs, logging.WARNING)
        assert len(warnings) == 1
        assert f"{_local.MAX_CLIENTS} are connected already" in warnings[0]
        # The readers that were there are still served.
        assert ask(held[0])

        held.pop().close()
        wait_for(reader.refresh)
        assert reader.description().robot == "rover"
    finally:
        for connection in held:
            connection.close()


def test_a_snapshot_the_reader_cannot_take_is_reported_once(
    socket_path, fake_server, connect, logs
):
    def answer(_request):
        junk = b"not a snapshot of this version"
        return ANSWER.pack(MAGIC, 1, 2, 0, len(junk)) + junk

    server = fake_server(socket_path, answer)
    reader = connect(socket_path)
    assert reader.server_reachable is False
    assert reader.connected is False
    assert reader.description() is None
    for _ in range(3):
        assert reader.refresh() is False
    wait_for(lambda: server.accepted >= 4)  # every refresh tried again

    warnings = messages(logs, logging.WARNING)
    assert len(warnings) == 1
    assert "runs another version of the isaaccapture package" in warnings[0]
    assert socket_path in warnings[0]


@pytest.mark.parametrize(
    "header",
    [
        ANSWER.pack(MAGIC, 2, 2, 0, 0),
        ANSWER.pack(MAGIC, 1, 3, 0, 0),
        ANSWER.pack(b"ICRX", 1, 2, 0, 0),
    ],
    ids=["version", "op", "magic"],
)
def test_an_answer_in_another_format_is_reported_once(
    socket_path, fake_server, connect, logs, header
):
    fake_server(socket_path, lambda _request: header)
    reader = connect(socket_path)
    assert reader.server_reachable is False
    assert reader.refresh() is False
    assert reader.description() is None
    warnings = messages(logs, logging.WARNING)
    assert len(warnings) == 1
    assert "runs another version of the isaaccapture package" in warnings[0]


def test_a_reader_asks_for_the_declarations_again_when_its_own_do_not_fit(
    socket_path, store, fake_server, connect
):
    start_robot(store, rover())
    asked = []

    def answer(request):
        known = REQUEST.unpack(request)[4]
        asked.append(known)
        # Answers without declarations whatever the reader holds; only 0 gets them.
        other = RobotStateStore()
        if known:
            while other.setup_generation <= known:
                other.begin_session()
            snapshot = other.export_snapshot(other.setup_generation)
        else:
            snapshot = store.export_snapshot(0)
        return ANSWER.pack(MAGIC, 1, 2, 0, len(snapshot)) + snapshot

    fake_server(socket_path, answer)
    reader = connect(socket_path, auto_refresh=False)
    assert reader.description().robot == "rover"
    assert asked == [0]
    assert reader.refresh() is True
    assert asked == [0, store.setup_generation, 0]
    assert reader.description().robot == "rover"


def test_an_answer_above_the_limit_is_not_read(socket_path, fake_server, connect, logs):
    too_long = ANSWER.pack(MAGIC, 1, 2, 0, _local.MAX_ANSWER_BYTES + 1)
    fake_server(socket_path, lambda _request: too_long)
    reader = connect(socket_path)
    assert reader.server_reachable is False
    assert reader.refresh() is False
    assert messages(logs, logging.WARNING) == []
    assert "above the limit" in messages(logs, logging.INFO, READER_LOGGER)[0]


def test_a_server_that_does_not_answer_in_time_counts_as_not_running(
    socket_path, fake_server, connect, logs
):
    server = fake_server(socket_path, lambda _request: None)
    started = time.monotonic()
    reader = connect(socket_path, timeout_s=0.2)
    assert reader.server_reachable is False
    assert reader.refresh() is False
    elapsed = time.monotonic() - started
    assert 0.5 <= elapsed < 20.0  # three refreshes of 0.2 s each
    assert reader.connected is False
    assert reader.description() is None
    # Each refresh closed its connection and made a new one.
    wait_for(lambda: server.accepted == 5)
    infos = messages(logs, logging.INFO, READER_LOGGER)
    assert len(infos) == 1
    assert "no answer within 0.2 s" in infos[0]


def test_a_server_that_stops_answering_is_waited_for_once(
    socket_path, store, fake_server, connect
):
    start_robot(store, rover())
    answers = []

    def answer(_request):
        if answers:
            return None
        snapshot = store.export_snapshot(0)
        answers.append(snapshot)
        return ANSWER.pack(MAGIC, 1, 2, 0, len(snapshot)) + snapshot

    server = fake_server(socket_path, answer)
    reader = connect(socket_path, timeout_s=0.2, auto_refresh=False)
    assert reader.description().robot == "rover"
    assert reader.refresh() is False
    assert reader.description() is None
    # The wait ran out on the kept connection; no second connection was tried for that refresh.
    time.sleep(0.2)
    assert server.accepted == 1
    assert reader.refresh() is False
    wait_for(lambda: server.accepted == 2)


# ---------------------------------------------------------------------------------------------
# The server in a program of its own
# ---------------------------------------------------------------------------------------------

_PROGRAM = (
    "import sys; sys.path[:0] = sys.argv[1:3]; "
    "import test_local_connection; test_local_connection.serve_commands(sys.argv[3])"
)


def serve_commands(socket_path):
    """The server's side, run in a program of its own: one command per line of its input."""
    descriptions = {"rover": rover, "six_joint_arm": six_joint_arm}
    store = RobotStateStore()
    service = RobotStateLocalService(store, socket_path)
    description = None
    for line in sys.stdin:
        command = json.loads(line)
        do = command.pop("do")
        if do == "start":
            service.start()
        elif do == "robot":
            description = descriptions[command["name"]]()
            start_robot(store, description)
        elif do == "put":
            put(store, description, command.pop("series"), **command)
        elif do == "leave":
            store.end_session()
        elif do == "close":
            service.close()
        print(f"done {do}", flush=True)


class ServerProgram:
    """A process that runs `serve_commands()`."""

    def __init__(self, socket_path):
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _PROGRAM,
                str(Path(__file__).resolve().parent),
                str(Path(repo_paths.__file__).resolve().parent),
                socket_path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )

    def do(self, do, **command):
        self._process.stdin.write(json.dumps({"do": do, **command}) + "\n")
        self._process.stdin.flush()
        while True:
            ready, _, _ = select.select([self._process.stdout], [], [], 60.0)
            assert ready, f"the server program did not answer '{do}'"
            line = self._process.stdout.readline()
            assert line, f"the server program ended (exit code {self._process.poll()})"
            if line.strip() == f"done {do}":
                return

    def kill(self):
        self._process.kill()
        self._process.wait(30.0)
        self._process.stdin.close()
        self._process.stdout.close()


@pytest.fixture
def server_program():
    programs = []

    def start(path):
        program = ServerProgram(path)
        programs.append(program)
        return program

    yield start
    for program in programs:
        program.kill()


def test_a_reader_and_a_server_in_two_programs(socket_path, server_program, connect):
    reader = connect(socket_path)
    assert reader.server_reachable is False

    # The server program runs; it opens the socket only once it is told to.
    program = server_program(socket_path)
    program.do("robot", name="rover")
    assert reader.refresh() is False
    program.do("start")
    assert reader.refresh() is True
    assert reader.connected is True
    assert reader.description() == rover()
    assert reader.joints("arm") is None

    program.do("put", series="arm", values=[0.25, 0.02, 0.0], velocities=ZEROS)
    arm = reader.joints("arm")
    assert arm["lift"].position == pytest.approx(0.25)
    # Both programs read the same clock of this machine.
    assert 0.0 <= arm.age_s < 30.0
    assert arm.clock_two_way is True
    assert reader.gripper("jaw").opening == pytest.approx(0.5, abs=1e-5)

    # A second reader, late: the description went out before it connected.
    late = connect(socket_path)
    assert late.description() == rover()
    assert late.joints("arm").sequence == 1
    program.do("put", series="health", values=[True, 12.5, "driving"])
    program.do("put", series="sonar", values={"near": 0.75, "label": "wall"})
    for one in (reader, late):
        assert one.status("health")["volts"] == pytest.approx(12.5)
        assert one.record("sonar").values == {"near": 0.75, "label": "wall"}

    # The robot app leaves; the next one is another robot.
    program.do("leave")
    for one in (reader, late):
        assert one.connected is False
        assert one.joints("arm") is None
        assert one.joints("arm", allow_stale=True)["lift"].position == pytest.approx(
            0.25
        )
    program.do("robot", name="six_joint_arm")
    program.do("put", series="status", values=[True, "overheated"])
    for one in (reader, late):
        assert one.connected is True
        assert one.description() == six_joint_arm()
        assert one.status(allow_stale=True)["fault"] == "overheated"

    # The server program dies without closing anything.
    program.kill()
    assert os.path.exists(socket_path)
    for one in (reader, late):
        assert one.refresh() is False
        assert one.server_reachable is False
        assert one.connected is False
        assert one.description() is None
        assert one.status(allow_stale=True) is None

    # It is started again at the same path, where the dead one's socket file still lies.
    again = server_program(socket_path)
    again.do("start")
    for one in (reader, late):
        assert one.refresh() is True
        assert one.connected is False
        assert one.description() is None
    again.do("robot", name="rover")
    again.do("put", series="mast", values=[3.0])
    for one in (reader, late):
        assert one.description().robot == "rover"
        assert one.joints("mast")["turn"].position == pytest.approx(3.0)

    # It stops in order and takes its socket file with it.
    again.do("close")
    assert not os.path.exists(socket_path)
    for one in (reader, late):
        assert one.server_reachable is False
        assert one.joints("mast", allow_stale=True) is None


def test_records_of_a_reader_s_store_are_the_bytes_the_robot_sent(
    socket_path, store, serve, connect
):
    description = rover()
    start_robot(store, description)
    sample_time_ns = put(store, description, "sonar", [0.5, "wall"])
    serve(store, socket_path)
    reader = connect(socket_path)
    sonar = reader.record("sonar")
    assert sonar.data == records.encode_reading(
        description, "sonar", [0.5, "wall"], time_ns=sample_time_ns
    )
    assert sonar.sample_time_ns == sample_time_ns
