# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Robot state hosted in the robot app: `RobotStateHost` with a real store and a real socket.

Every socket is a Unix socket in a temporary directory of the test.
"""

import json
import logging
import os
import socket
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from test_reader import ROVER, rover, six_joint_arm, start_robot

import isaaccapture.robot_state as robot_state
from isaaccapture.robot_state import (
    RobotState,
    RobotStateLocalService,
    RobotStateStore,
    _host,
    _local,
    default_socket_path,
    parse_robot_description,
    records,
)
from isaaccapture.robot_state._host import RobotStateHost
from isaaccapture.robot_state._robot_state import DESCRIPTION_TOPIC

pytestmark = pytest.mark.skipif(
    not _local._HAS_UNIX_SOCKETS, reason="the reader socket is a Unix domain socket"
)

MS = 1_000_000
SECOND = 1_000 * MS
MIB = 1 << 20
HOST_LOGGER = "isaaccapture.robot_state._host"
OTHER = """
robot = "other"

[series.arm]
shape = "joints"
joints = [{ name = "joint1", kind = "revolute", min = -1.0, max = 1.0 }]
"""
# What a declaration and a declared series have in common.
DECLARED = (
    "name",
    "shape",
    "schema_name",
    "schema_encoding",
    "schema_data",
    "message_encoding",
    "stale_after_ns",
    "in_order",
    "rate_hz",
)


def declared(series):
    return tuple(getattr(series, name) for name in DECLARED)


def publish(host, description, series, values, *, time_ns=None, **reading):
    """One reading, handed to the host as `TeleopSession.publish_robot_state()` hands it over."""
    now_ns = time.monotonic_ns()
    sample_ns = now_ns if time_ns is None else time_ns
    data = records.encode_reading(
        description,
        series,
        values,
        time_ns=sample_ns,
        publish_time_ns=now_ns,
        **reading,
    )
    return host.publish_robot_record(series, data, sample_ns)


def other_store():
    """A store as another program's robot app leaves it."""
    store = RobotStateStore()
    start_robot(store, parse_robot_description(OTHER))
    return store


def refused(reason):
    """The warning of a host that cannot open the reader socket."""
    return (
        f"Reader programs cannot get robot state from this robot app: {reason}. Publish calls "
        "return False until the socket can be opened; trying again every 2 s"
    )


def served_by_another(path):
    return f"another program serves robot state at {path}"


def taken(path):
    """What a host raises that starts at a path another program serves."""
    return (
        f"Another program on this machine already serves a robot's state at {path}: a Teleop "
        "Session Server or another robot app. One robot per machine is served; stop that "
        "program, then start this robot app again."
    )


def not_a_socket(path):
    return f"{path} exists and is not a socket"


def lock_descriptors(path):
    """This program's open descriptors of `path`'s lock file; None where they cannot be listed."""
    if not os.path.isdir("/proc/self/fd"):
        return None
    lock_path = os.path.realpath(path + _local.LOCK_SUFFIX)
    found = []
    for name in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{name}")
        except OSError:
            continue
        if target == lock_path:
            found.append(int(name))
    return found


def host_warnings(caplog):
    return [
        entry.getMessage()
        for entry in caplog.records
        if entry.levelno >= logging.WARNING and entry.name == HOST_LOGGER
    ]


def wait_for(condition, timeout_s=20.0):
    """The first true value of `condition()`; fails when none comes in time."""
    deadline = time.monotonic() + timeout_s
    while True:
        value = condition()
        if value:
            return value
        assert time.monotonic() < deadline, "the awaited state did not come"
        time.sleep(0.01)


def serving_threads():
    return [
        thread for thread in threading.enumerate() if thread.name == _local.THREAD_NAME
    ]


def retry_threads():
    return [
        thread
        for thread in threading.enumerate()
        if thread.name == _host.RETRY_THREAD_NAME
    ]


def files_beside(path):
    return sorted(entry.name for entry in Path(path).parent.iterdir())


@pytest.fixture
def socket_path(tmp_path_factory):
    # Shorter than `tmp_path`, which holds the test's name: a Unix socket path is limited.
    return str(tmp_path_factory.mktemp("rs") / "rs.sock")


@pytest.fixture
def hosts():
    """Make hosts that are closed when the test ends."""
    made = []

    def make(path):
        host = RobotStateHost(path)
        made.append(host)
        return host

    yield make
    for host in made:
        host.close()


@pytest.fixture
def serve():
    """Start services of another owner; they are closed when the test ends."""
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

    def make(path=None, **options):
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


class Clock:
    """Stands in for the `time` module inside `_host`: the test moves its monotonic clock."""

    monotonic_ns = staticmethod(time.monotonic_ns)

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(_host, "time", clock)
    return clock


@pytest.fixture
def attempts(monkeypatch):
    """The paths of the services the host tried to start, in order."""
    tried = []

    class Counted(RobotStateLocalService):
        def start(self):
            tried.append(self.path)
            super().start()

    monkeypatch.setattr(_host, "RobotStateLocalService", Counted)
    return tried


@pytest.fixture
def finished(monkeypatch):
    """What each finished attempt of a host to open its socket returned, in order."""
    results = []
    attempt = RobotStateHost._open

    def counted(host, **options):
        served = attempt(host, **options)
        results.append(served)
        return served

    monkeypatch.setattr(RobotStateHost, "_open", counted)
    return results


@pytest.fixture
def retry(clock, finished):
    """Let the retry period pass and publish: the call is refused and has the socket tried on
    the host's thread. Returns what that attempt returned, once it has finished."""

    def after_the_period(host, description):
        clock.now += _host.RETRY_S
        count = len(finished)
        assert publish(host, description, "mast", [0.25]) is False
        wait_for(lambda: len(finished) > count)
        return finished[count]

    return after_the_period


class Asked(threading.Event):
    """Stands in for a host's request to its retry thread: counts how often it was made."""

    def __init__(self):
        super().__init__()
        self.count = 0

    def set(self):
        self.count += 1
        super().set()


# ---------------------------------------------------------------------------------------------
# A started host is read as a Teleop Session Server is
# ---------------------------------------------------------------------------------------------


def test_a_reader_sees_the_series_and_the_description_of_a_started_host(
    socket_path, hosts, connect
):
    description = rover()
    host = hosts(socket_path)
    assert host.robot_state_active is False
    host.start(description)
    assert host.robot_state_active is True

    reader = connect(socket_path)
    assert reader.server_reachable is True
    assert reader.connected is True
    assert reader.description() == description
    assert list(reader.series()) == list(description.series)
    # The declarations, in order: the description's own series, then the robot's.
    assert [declared(series) for series in reader._store.series()] == [
        declared(series) for series in records.declarations(description)
    ]
    held = reader._store.latest(DESCRIPTION_TOPIC)
    assert records.decode_description(held.data) == description
    assert held.fresh is True
    assert held.sequence == 1
    assert held.clock_two_way is True
    # No reading before the first publish.
    assert reader.joints("arm") is None
    assert reader.joints("arm", allow_stale=True) is None


def test_the_six_joint_arm_of_the_example_is_hosted(socket_path, hosts, connect):
    description = six_joint_arm()
    host = hosts(socket_path)
    host.start(description)
    reader = connect(socket_path, robot=description.robot)
    assert reader.connected is True
    assert reader.description() == description

    positions = [0.1, 1.204, 0.3, 0.4, 0.5, 0.6, 0.96]
    assert publish(host, description, "arm", positions) is True
    assert reader.joints("arm").positions == pytest.approx(positions)


def test_the_store_is_filled_before_the_socket_opens(socket_path, hosts, monkeypatch):
    description = rover()
    seen = []

    class Watched(RobotStateLocalService):
        def start(self):
            store = self._store
            seen.append(
                (
                    [series.name for series in store.series()],
                    store.connected,
                    store.latest(DESCRIPTION_TOPIC).data is not None,
                    os.path.exists(self.path),
                )
            )
            super().start()

    monkeypatch.setattr(_host, "RobotStateLocalService", Watched)
    hosts(socket_path).start(description)
    assert seen == [([DESCRIPTION_TOPIC, *description.series], True, True, False)]
    assert os.path.exists(socket_path)


def test_a_reading_reads_back_with_its_time_and_its_age(socket_path, hosts, connect):
    description = rover()
    host = hosts(socket_path)
    host.start(description)
    reader = connect(socket_path)

    taken_ns = time.monotonic_ns() - 250 * MS
    assert (
        publish(
            host,
            description,
            "arm",
            [0.25, 0.02, -0.5],
            velocities=[1.0, 0.0, -1.0],
            time_ns=taken_ns,
        )
        is True
    )
    published_ns = time.monotonic_ns()
    before_s = (published_ns - taken_ns) / 1e9
    arm = reader.joints("arm")
    after_s = (time.monotonic_ns() - taken_ns) / 1e9

    assert arm.names == ("lift", "jaw", "thumb")
    assert arm.positions == pytest.approx([0.25, 0.02, -0.5])
    assert [joint.velocity for joint in arm.joints] == [1.0, 0.0, -1.0]
    assert arm.sample_time_ns == taken_ns
    assert arm.sequence == 1
    assert arm.fresh is True
    assert arm.clock_two_way is True
    # The age is the time since the reading, not since the publish call.
    assert before_s - 0.001 <= arm.age_s <= after_s + 0.001
    assert 0.0 <= arm.receive_age_s <= arm.age_s - 0.249
    # The record keeps both times: of the reading and of the publish call.
    held = reader._store.latest("arm")
    assert held.sample_time_ns == taken_ns
    assert taken_ns + 250 * MS <= held.publish_time_ns <= published_ns

    # The thumb is at the gripper's open position.
    pinch = reader.gripper("pinch")
    assert pinch.opening == 1.0
    assert pinch.sample_time_ns == taken_ns

    assert (
        publish(
            host, description, "health", [True, 24.5, "idle"], faults=[("E1", "hot")]
        )
        is True
    )
    health = reader.status("health")
    assert dict(health) == {"ok": True, "volts": 24.5, "mode": "idle"}
    assert [(fault.code, fault.text) for fault in health.faults] == [("E1", "hot")]
    assert health.clock_two_way is True

    assert publish(host, description, "sonar", {"near": 0.75, "label": "wall"}) is True
    sonar = reader.record("sonar")
    assert sonar.values == {"near": 0.75, "label": "wall"}
    assert sonar.sequence == 1

    # Each publish is one record of its series.
    assert publish(host, description, "health", [False, 23.0, "fault"]) is True
    assert reader.status("health").sequence == 2
    assert reader.status("health")["ok"] is False
    assert reader.joints("arm").sequence == 1


def test_a_reading_stamped_in_the_future_ages_from_the_publish_call(
    socket_path, hosts, connect
):
    description = rover()
    host = hosts(socket_path)
    host.start(description)
    reader = connect(socket_path)

    called_ns = time.monotonic_ns()
    assert (
        publish(host, description, "mast", [0.5], time_ns=called_ns + 5 * SECOND)
        is True
    )
    mast = reader.joints("mast")
    after_s = (time.monotonic_ns() - called_ns) / 1e9
    assert mast.sample_time_ns == called_ns + 5 * SECOND
    assert mast.clock_two_way is True
    assert 0.0 <= mast.age_s <= after_s
    assert mast.age_s == mast.receive_age_s

    time.sleep(0.05)
    later = reader.joints("mast")
    assert later.age_s >= mast.age_s + 0.05
    assert later.age_s == later.receive_age_s


def test_a_reading_goes_stale_as_its_series_says(socket_path, hosts, connect):
    description = rover()
    limit_s = description.series["quick"].stale_after_s
    assert 0.0 < limit_s < 1.0
    host = hosts(socket_path)
    host.start(description)
    reader = connect(socket_path)

    published = time.monotonic()
    assert publish(host, description, "quick", [0.5]) is True
    assert publish(host, description, "mast", [0.25]) is True
    assert publish(host, description, "arm", [0.1, 0.02, 0.3], velocities=[0, 0, 0])
    assert reader.joints("quick").fresh is True

    wait_for(lambda: reader.joints("quick") is None)
    assert time.monotonic() - published >= limit_s
    stale = reader.joints("quick", allow_stale=True)
    assert stale.fresh is False
    assert stale.positions == (0.5,)
    assert stale.age_s >= limit_s
    # A series that never goes stale, and one that stays fresh for a minute.
    assert reader.joints("mast").fresh is True
    assert reader.joints("arm").fresh is True
    assert reader.connected is True

    # The next reading is fresh again.
    assert publish(host, description, "quick", [0.75]) is True
    assert reader.joints("quick").positions == (0.75,)


def test_the_default_path_is_the_one_a_reader_connects_to(
    tmp_path_factory, monkeypatch, connect
):
    runtime = tmp_path_factory.mktemp("rt")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    path = default_socket_path()
    assert path == str(runtime / "isaaccapture" / "robot_state.sock")

    description = rover()
    host = RobotStateHost()
    try:
        host.start(description)
        assert host.robot_state_active is True
        assert os.path.exists(path)
        assert connect().description() == description
    finally:
        host.close()
    assert not os.path.exists(path)


def test_the_host_is_not_among_the_package_s_names():
    assert "RobotStateHost" not in robot_state.__all__
    with pytest.raises(AttributeError):
        robot_state.RobotStateHost  # noqa: B018
    assert _host.RETRY_S == 2.0


# ---------------------------------------------------------------------------------------------
# A reader in another program, and what close() leaves
# ---------------------------------------------------------------------------------------------

_READER = """
import json, sys
from isaaccapture.robot_state import RobotState
with RobotState.connect(sys.argv[1], robot="rover") as robot:
    arm = robot.joints("arm")
    print(json.dumps({
        "reachable": robot.server_reachable,
        "connected": robot.connected,
        "robot": robot.description().robot,
        "series": list(robot.series()),
        "positions": list(arm.positions),
        "sequence": arm.sequence,
        "sample_time_ns": arm.sample_time_ns,
        "age_s": arm.age_s,
        "fresh": arm.fresh,
        "clock_two_way": arm.clock_two_way,
    }))
"""


def test_a_second_program_reads_the_same_robot(socket_path, hosts):
    description = rover()
    host = hosts(socket_path)
    host.start(description)
    taken_ns = time.monotonic_ns()
    assert publish(host, description, "arm", [0.5, 0.25, 0.75], velocities=[0, 0, 0])
    assert publish(
        host,
        description,
        "arm",
        [0.25, 0.5, 0.75],
        velocities=[0, 0, 0],
        time_ns=taken_ns,
    )

    done = subprocess.run(
        [sys.executable, "-c", _READER, socket_path],
        check=False,
        capture_output=True,
        text=True,
        timeout=120.0,
    )
    since_s = (time.monotonic_ns() - taken_ns) / 1e9
    assert done.returncode == 0, done.stderr
    read = json.loads(done.stdout)
    age_s = read.pop("age_s")
    assert read == {
        "reachable": True,
        "connected": True,
        "robot": "rover",
        "series": list(description.series),
        "positions": [0.25, 0.5, 0.75],
        "sequence": 2,
        "sample_time_ns": taken_ns,
        "fresh": True,
        "clock_two_way": True,
    }
    # Both programs read one clock of this machine.
    assert 0.0 <= age_s <= since_s + 0.001


def test_close_ends_the_robot_s_session_and_frees_the_path(socket_path, connect, serve):
    description = rover()
    host = RobotStateHost(socket_path)
    host.start(description)
    reader = connect(socket_path)
    assert publish(host, description, "mast", [0.5]) is True
    assert reader.joints("mast").positions == (0.5,)
    assert len(serving_threads()) == 1
    store = host._store

    host.close()
    assert host.robot_state_active is False
    assert store.connected is False
    assert reader.server_reachable is False
    assert reader.connected is False
    assert reader.description() is None
    assert reader.joints("mast", allow_stale=True) is None
    assert not os.path.exists(socket_path)
    assert files_beside(socket_path) == ["rs.sock.lock"]
    assert serving_threads() == []

    # A later publish stores nothing, and a second close changes nothing.
    assert publish(host, description, "mast", [0.75]) is False
    assert store.latest("mast").sequence == 1
    host.close()
    assert host.robot_state_active is False
    with pytest.raises(RuntimeError, match="RobotStateHost is closed"):
        host.start(description)

    # The path is free for the next owner, which the kept reader then reads.
    serve(other_store(), socket_path)
    assert reader.description().robot == "other"


def test_a_host_that_was_not_started_stores_and_serves_nothing(socket_path):
    description = rover()
    host = RobotStateHost(socket_path)
    assert host.robot_state_active is False
    assert publish(host, description, "mast", [0.5]) is False
    host.close()
    host.close()
    assert files_beside(socket_path) == []

    started = RobotStateHost(socket_path)
    try:
        started.start(description)
        with pytest.raises(RuntimeError, match="RobotStateHost is started"):
            started.start(description)
        assert started.robot_state_active is True
    finally:
        started.close()


def test_a_start_that_fails_part_way_is_closed_like_any_other(socket_path, connect):
    host = RobotStateHost(socket_path)
    # Not a description: start() fails before anything is stored or served.
    with pytest.raises(AttributeError):
        host.start(None)
    assert host.robot_state_active is False
    host.close()
    assert files_beside(socket_path) == []
    assert connect(socket_path).server_reachable is False


def test_publishing_from_several_threads(socket_path, hosts, connect):
    description = rover()
    host = hosts(socket_path)
    host.start(description)
    reader = connect(socket_path)
    results = []

    def run():
        results.append([publish(host, description, "mast", [0.5]) for _ in range(250)])

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60.0)
    assert [all(result) and len(result) for result in results] == [250] * 4
    assert reader.joints("mast").sequence == 1000


def test_closing_while_another_thread_publishes(socket_path, connect):
    description = rover()
    host = RobotStateHost(socket_path)
    host.start(description)
    results = []
    failures = []

    def run():
        try:
            while True:
                results.append(publish(host, description, "mast", [0.5]))
                if not results[-1]:
                    return
        except Exception as error:
            failures.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    wait_for(lambda: len(results) > 100)
    host.close()
    thread.join(60.0)
    assert not thread.is_alive()
    assert failures == []
    assert results[-1] is False
    assert all(results[:-1])
    assert not os.path.exists(socket_path)
    assert serving_threads() == []
    assert connect(socket_path).server_reachable is False


def test_a_host_closed_while_it_opens_the_socket_serves_nothing(
    socket_path, monkeypatch, connect
):
    description = rover()
    host = RobotStateHost(socket_path)

    class ClosedMeanwhile(RobotStateLocalService):
        def start(self):
            super().start()
            # As close() on another thread, before the host has taken this service.
            host.close()

    monkeypatch.setattr(_host, "RobotStateLocalService", ClosedMeanwhile)
    host.start(description)
    assert host.robot_state_active is False
    assert host._service is None
    assert not os.path.exists(socket_path)
    assert serving_threads() == []
    assert publish(host, description, "mast", [0.5]) is False
    assert connect(socket_path).server_reachable is False


# ---------------------------------------------------------------------------------------------
# The path is another program's, and a socket that cannot be opened yet
# ---------------------------------------------------------------------------------------------


def test_a_host_does_not_start_at_a_path_another_program_serves(
    socket_path, serve, connect, logs, attempts
):
    other = serve(other_store(), socket_path)
    kept = connect(socket_path)
    expecting = connect(socket_path, robot="rover")
    assert kept.description().robot == "other"
    held = lock_descriptors(socket_path)

    description = rover()
    host = RobotStateHost(socket_path)
    with pytest.raises(RuntimeError) as raised:
        host.start(description)
    assert str(raised.value) == taken(socket_path)
    cause = raised.value.__cause__
    assert type(cause) is _local.ReaderSocketTaken
    assert str(cause) == served_by_another(socket_path)
    assert attempts == [socket_path]

    # The host is closed and holds nothing: no thread, no lock, no session a reader could see.
    assert host._closed is True
    assert host.robot_state_active is False
    assert host._service is None
    assert host._retry_thread is None
    assert retry_threads() == []
    assert host._store.connected is False
    assert lock_descriptors(socket_path) == held
    assert publish(host, description, "mast", [0.5]) is False
    assert host._store.latest("mast").sequence == 0
    with pytest.raises(RuntimeError, match="RobotStateHost is closed"):
        host.start(description)
    host.close()
    assert attempts == [socket_path]
    # The exception is the whole report.
    assert host_warnings(logs) == []

    # The first owner and its readers are untouched.
    assert other.serving is True
    assert len(serving_threads()) == 1
    assert kept.refresh() is True
    assert kept.connected is True
    assert kept.description().robot == "other"
    assert expecting.server_reachable is True
    assert expecting.connected is False
    assert files_beside(socket_path) == ["rs.sock", "rs.sock.lock"]

    # The other program left: a host that starts then serves, and the kept readers read it.
    other.close()
    following = RobotStateHost(socket_path)
    try:
        following.start(description)
        assert following.robot_state_active is True
        assert publish(following, description, "mast", [0.75]) is True
        assert kept.description() == description
        assert expecting.connected is True
        assert expecting.joints("mast").positions == (0.75,)
    finally:
        following.close()
    assert host_warnings(logs) == []


def test_a_host_does_not_start_at_the_socket_of_a_program_without_a_lock(
    socket_path, hosts, logs
):
    """A program whose package has no lock file answers the look at its socket."""
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(socket_path)
    listener.listen(4)
    description = rover()
    try:
        host = RobotStateHost(socket_path)
        with pytest.raises(RuntimeError) as raised:
            host.start(description)
        assert str(raised.value) == taken(socket_path)
        assert type(raised.value.__cause__) is _local.ReaderSocketTaken
        assert host.robot_state_active is False
        assert retry_threads() == []
        assert lock_descriptors(socket_path) in (None, [])
        assert stat.S_ISSOCK(os.lstat(socket_path).st_mode)
    finally:
        listener.close()
    assert host_warnings(logs) == []
    # It ended and left its socket file: nobody is refused for that.
    following = hosts(socket_path)
    following.start(description)
    assert following.robot_state_active is True


def test_a_host_whose_socket_cannot_be_opened_serves_once_it_can(
    socket_path, hosts, connect, logs
):
    Path(socket_path).write_text("notes")
    expecting = connect(socket_path, robot="rover")

    description = rover()
    host = hosts(socket_path)
    host.start(description)
    assert host.robot_state_active is False
    assert publish(host, description, "mast", [0.5]) is False
    assert publish(host, description, "mast", [0.5]) is False
    # Nothing is stored for readers that cannot come.
    assert host._store.latest("mast").data is None
    assert host._store.latest("mast").sequence == 0
    assert host_warnings(logs) == [refused(not_a_socket(socket_path))]
    assert expecting.server_reachable is False
    assert Path(socket_path).read_text() == "notes"

    Path(socket_path).unlink()
    freed = time.monotonic()
    while not publish(host, description, "mast", [0.75]):
        assert time.monotonic() - freed < _host.RETRY_S + 1.0
        time.sleep(0.02)
    assert host.robot_state_active is True

    # Readers at the path read this robot, with its description.
    assert expecting.connected is True
    assert expecting.description() == description
    mast = expecting.joints("mast")
    assert mast.positions == (0.75,)
    assert mast.sequence == 1
    assert host_warnings(logs) == [refused(not_a_socket(socket_path))]


@pytest.mark.skipif(
    hasattr(os, "getuid") and os.getuid() == 0,
    reason="root connects to a socket file whatever its mode",
)
def test_a_host_starts_beside_a_socket_file_closed_to_this_user(
    socket_path, hosts, connect, logs
):
    """A socket file left behind that this user may not connect to: nobody is known to serve it."""
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(socket_path)
    listener.close()
    os.chmod(socket_path, 0)
    reason = f"{socket_path} cannot be connected to ([Errno 13] Permission denied)"

    description = rover()
    host = hosts(socket_path)
    host.start(description)
    assert host.robot_state_active is False
    assert publish(host, description, "mast", [0.5]) is False
    assert host_warnings(logs) == [refused(reason)]
    assert stat.S_ISSOCK(os.lstat(socket_path).st_mode)

    # It can be connected to again and nobody answers: the host replaces it and serves.
    os.chmod(socket_path, 0o600)
    freed = time.monotonic()
    while not publish(host, description, "mast", [0.75]):
        assert time.monotonic() - freed < _host.RETRY_S + 1.0
        time.sleep(0.02)
    assert host.robot_state_active is True
    assert connect(socket_path, robot="rover").joints("mast").positions == (0.75,)
    assert host_warnings(logs) == [refused(reason)]


def test_a_host_without_a_socket_tries_it_once_per_retry_period(
    socket_path, hosts, connect, logs, clock, attempts, finished
):
    Path(socket_path).write_text("notes")
    description = rover()
    host = hosts(socket_path)
    asked = host._retry_asked = Asked()
    started = clock.now
    host.start(description)
    assert attempts == [socket_path]
    assert finished == [False]

    def publishes(first_s, last_s):
        """Publish calls every 10 ms of the host's clock; what they returned."""
        returned = set()
        for step in range(round(first_s * 100), round(last_s * 100) + 1):
            clock.now = started + step / 100
            returned.add(publish(host, description, "mast", [0.5]))
        return returned

    # One publish call per period asks for an attempt, which the host's thread makes.
    assert publishes(0.0, 1.99) == {False}
    assert asked.count == 0
    assert publishes(2.0, 3.99) == {False}
    assert asked.count == 1
    wait_for(lambda: len(finished) == 2)
    assert publishes(4.0, 5.0) == {False}
    assert asked.count == 2
    wait_for(lambda: len(finished) == 3)
    assert attempts == [socket_path] * 3
    assert host.robot_state_active is False
    # One warning, however often the socket is tried.
    assert host_warnings(logs) == [refused(not_a_socket(socket_path))]

    # The file is gone: the next attempt opens the socket. The call that asked for it is
    # refused like the ones before.
    Path(socket_path).unlink()
    assert publishes(5.01, 5.99) == {False}
    assert asked.count == 2
    assert publishes(6.0, 6.0) == {False}
    assert asked.count == 3
    wait_for(lambda: len(finished) == 4)
    assert finished == [False, False, False, True]
    assert host.robot_state_active is True
    # A host that serves does not try again.
    assert publishes(6.01, 12.0) == {True}
    assert asked.count == 3
    assert attempts == [socket_path] * 4
    assert len(finished) == 4

    # Only the publish calls that returned True are stored.
    mast = connect(socket_path).joints("mast")
    assert mast.sequence == 600
    assert host_warnings(logs) == [refused(not_a_socket(socket_path))]


def test_the_socket_is_tried_again_off_the_publishing_thread(
    socket_path, hosts, connect, clock, monkeypatch
):
    entered = threading.Event()
    release = threading.Event()
    tried_on = []

    class Slow(RobotStateLocalService):
        def start(self):
            tried_on.append(threading.current_thread().name)
            if len(tried_on) > 1:
                entered.set()
                release.wait(60.0)
            super().start()

    monkeypatch.setattr(_host, "RobotStateLocalService", Slow)
    Path(socket_path).write_text("notes")
    description = rover()
    host = hosts(socket_path)
    # The first attempt is made by start() itself, which also starts the retry thread.
    host.start(description)
    assert tried_on == [threading.current_thread().name]
    assert len(retry_threads()) == 1
    Path(socket_path).unlink()

    # The publish call that asks returns while the attempt is under way, and so do later ones.
    clock.now += _host.RETRY_S
    assert publish(host, description, "mast", [0.25]) is False
    assert entered.wait(20.0)
    assert not release.is_set()
    clock.now += _host.RETRY_S
    assert publish(host, description, "mast", [0.25]) is False
    assert host.robot_state_active is False
    assert host._store.latest("mast").sequence == 0

    release.set()
    wait_for(lambda: host.robot_state_active)
    assert tried_on == [threading.current_thread().name, _host.RETRY_THREAD_NAME]
    assert publish(host, description, "mast", [0.75]) is True
    assert connect(socket_path).joints("mast").positions == (0.75,)
    # The retry thread ends once the socket is served.
    wait_for(lambda: retry_threads() == [])


def test_a_host_that_serves_from_the_start_has_no_retry_thread(socket_path, hosts):
    description = rover()
    host = hosts(socket_path)
    host.start(description)
    assert host.robot_state_active is True
    assert publish(host, description, "mast", [0.5]) is True
    assert host._retry_thread is None
    assert retry_threads() == []


def test_closing_a_host_without_a_socket_ends_its_retry_thread(socket_path, retry):
    Path(socket_path).write_text("notes")
    description = rover()
    host = RobotStateHost(socket_path)
    host.start(description)
    assert len(retry_threads()) == 1
    assert retry(host, description) is False

    host.close()
    assert retry_threads() == []
    assert publish(host, description, "mast", [0.5]) is False
    host.close()
    # What is at the path is untouched.
    assert Path(socket_path).read_text() == "notes"
    assert files_beside(socket_path) == ["rs.sock", "rs.sock.lock"]
    assert lock_descriptors(socket_path) in (None, [])


def test_closing_while_the_retry_thread_opens_the_socket_leaves_nothing(
    socket_path, serve, connect, clock, monkeypatch
):
    entered = threading.Event()
    release = threading.Event()

    class Slow(RobotStateLocalService):
        def start(self):
            super().start()
            if threading.current_thread().name == _host.RETRY_THREAD_NAME:
                entered.set()
                release.wait(60.0)

    monkeypatch.setattr(_host, "RobotStateLocalService", Slow)
    Path(socket_path).write_text("notes")
    description = rover()
    host = RobotStateHost(socket_path)
    host.start(description)
    Path(socket_path).unlink()
    clock.now += _host.RETRY_S
    assert publish(host, description, "mast", [0.25]) is False
    # The attempt has bound the socket and the host has not taken the service yet.
    assert entered.wait(20.0)
    assert os.path.exists(socket_path)
    assert host.robot_state_active is False

    closer = threading.Thread(target=host.close)
    closer.start()
    wait_for(lambda: host._closed)
    release.set()
    closer.join(20.0)
    assert not closer.is_alive()
    # close() waited for the attempt, which closed what it had opened.
    assert host._service is None
    assert not os.path.exists(socket_path)
    assert serving_threads() == []
    assert retry_threads() == []
    assert connect(socket_path).server_reachable is False
    # The path is free for the next owner.
    assert serve(other_store(), socket_path).serving is True


def test_a_path_another_program_takes_later_is_a_warning_each_time_it_is_lost(
    socket_path, hosts, serve, logs, retry
):
    description = rover()
    host = hosts(socket_path)
    host.start(description)
    assert host.robot_state_active is True
    assert host_warnings(logs) == []

    # The reader thread ends and another program takes the path before the host's next attempt.
    host._service.close()
    assert host.robot_state_active is False
    other = serve(other_store(), socket_path)
    assert retry(host, description) is False
    assert retry(host, description) is False
    assert host_warnings(logs) == [refused(served_by_another(socket_path))]
    # Only a host that starts is refused: this one stays open and its thread goes on trying.
    assert host._closed is False
    assert len(retry_threads()) == 1

    # The host serves again, then loses the path once more: that is reported once more.
    other.close()
    assert retry(host, description) is True
    assert publish(host, description, "mast", [0.5]) is True
    assert host.robot_state_active is True
    host._service.close()
    serve(other_store(), socket_path)
    assert retry(host, description) is False
    assert host_warnings(logs) == [refused(served_by_another(socket_path))] * 2


def test_a_host_whose_reader_thread_ended_serves_again(
    socket_path, hosts, connect, logs, retry
):
    description = rover()
    host = hosts(socket_path)
    host.start(description)
    reader = connect(socket_path)
    assert publish(host, description, "mast", [0.5]) is True
    assert reader.joints("mast").sequence == 1
    assert retry_threads() == []

    def explode(request):
        raise RuntimeError("the service exploded")

    dead = host._service
    dead._answer = explode
    # The next request ends the service's thread.
    assert reader.refresh() is False
    wait_for(lambda: serving_threads() == [])
    assert dead.serving is False
    assert host.robot_state_active is False
    assert not os.path.exists(socket_path)

    # Not before the retry period has passed since the last attempt.
    assert publish(host, description, "mast", [0.25]) is False
    assert retry_threads() == []
    assert retry(host, description) is True
    assert publish(host, description, "mast", [0.75]) is True
    assert host.robot_state_active is True
    assert host._service is not dead
    assert dead._closed is True
    wait_for(lambda: len(serving_threads()) == 1)

    # The store is the same: the reader sees the robot's session go on.
    mast = reader.joints("mast")
    assert mast.positions == (0.75,)
    assert mast.sequence == 2
    assert reader.description() == description
    assert host_warnings(logs) == []


_FORKING_HOST = """
import os, sys, time
from isaaccapture.robot_state import RobotState, _host, parse_robot_description, records

path = sys.argv[1]
description = parse_robot_description(sys.argv[2])
host = _host.RobotStateHost(path)
host.start(description)


def publish(position):
    now_ns = time.monotonic_ns()
    data = records.encode_reading(description, "mast", [position], time_ns=now_ns)
    return host.publish_robot_record("mast", data, now_ns)


assert publish(0.25) is True
pid = os.fork()
if pid == 0:
    code = 1
    try:
        # The child's publish has the socket tried at once.
        _host.RETRY_S = 0.0
        if host.robot_state_active:
            code = 2
        elif publish(0.5) is not False:
            code = 3
        else:
            host.close()
            code = 0
    finally:
        os._exit(code)
_, status = os.waitpid(pid, 0)
print("child", os.waitstatus_to_exitcode(status), flush=True)
print("parent", host.robot_state_active, publish(0.75), flush=True)
with RobotState.connect(path) as robot:
    mast = robot.joints("mast")
    print("read", robot.connected, mast.sequence, mast.positions[0], flush=True)
host.close()
print("left", os.path.exists(path), flush=True)
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_a_forked_child_of_a_hosting_program_takes_nothing_from_it(socket_path):
    done = subprocess.run(
        [sys.executable, "-c", _FORKING_HOST, socket_path, ROVER],
        check=False,
        capture_output=True,
        text=True,
        timeout=120.0,
    )
    assert done.returncode == 0, done.stderr
    # The child is not active and its publish is refused; the parent goes on serving.
    assert done.stdout.splitlines() == [
        "child 0",
        "parent True True",
        "read True 2 0.75",
        "left False",
    ]


# ---------------------------------------------------------------------------------------------
# What the host refuses, and sockets that cannot be opened
# ---------------------------------------------------------------------------------------------


def test_another_service_is_refused_at_a_path_the_host_serves(
    socket_path, hosts, connect
):
    description = rover()
    host = hosts(socket_path)
    host.start(description)
    reader = connect(socket_path)

    late = RobotStateLocalService(other_store(), socket_path)
    with pytest.raises(OSError) as refusal:
        late.start()
    assert str(refusal.value) == served_by_another(socket_path)
    assert late.serving is False
    late.close()

    # A second host in this program does not start either.
    second = hosts(socket_path)
    with pytest.raises(RuntimeError) as raised:
        second.start(parse_robot_description(OTHER))
    assert str(raised.value) == taken(socket_path)
    assert second.robot_state_active is False
    assert retry_threads() == []

    assert host.robot_state_active is True
    assert publish(host, description, "mast", [0.5]) is True
    assert reader.description() == description
    assert reader.joints("mast").positions == (0.5,)


def test_a_record_the_store_does_not_take_raises(socket_path, hosts, connect):
    description = rover()
    host = hosts(socket_path)
    host.start(description)
    now_ns = time.monotonic_ns()

    with pytest.raises(ValueError, match=r"^RobotStateStore: .*above the limit"):
        host.publish_robot_record("sonar", b"0" * (MIB + 1), now_ns)
    assert host._store.latest("sonar").data is None
    # The limit itself is taken.
    assert host.publish_robot_record("sonar", b"0" * MIB, now_ns) is True
    assert len(host._store.latest("sonar").data) == MIB

    with pytest.raises(ValueError, match=r"^RobotStateStore: .*is not declared"):
        host.publish_robot_record("wing", b"{}", now_ns)

    # The host goes on.
    assert host.robot_state_active is True
    assert publish(host, description, "mast", [0.5]) is True
    assert connect(socket_path).joints("mast").positions == (0.5,)


def test_a_socket_path_that_is_too_long_is_a_warning(
    tmp_path_factory, hosts, logs, retry
):
    base = tmp_path_factory.mktemp("rs")
    path = str(base / ("d" * 100) / "rs.sock")
    reason = (
        f"socket path is {len(os.fsencode(path))} bytes, over the 103 a Unix domain "
        f"socket allows ({path})"
    )
    description = rover()
    host = hosts(path)
    host.start(description)
    assert host.robot_state_active is False
    assert publish(host, description, "mast", [0.5]) is False
    assert retry(host, description) is False
    assert host._store.latest("mast").data is None
    assert host_warnings(logs) == [refused(reason)]
    assert list(base.iterdir()) == []
    host.close()
    assert list(base.iterdir()) == []


def test_a_reader_thread_that_cannot_start_is_a_warning(
    socket_path, hosts, connect, logs, clock, retry, monkeypatch
):
    class NoThread(threading.Thread):
        def start(self):
            raise RuntimeError("can't start new thread")

    description = rover()
    host = hosts(socket_path)
    with monkeypatch.context() as patched:
        # Neither the service's thread nor the host's retry thread can be started.
        patched.setattr(_local.threading, "Thread", NoThread)
        host.start(description)
        assert publish(host, description, "mast", [0.5]) is False
        clock.now += _host.RETRY_S
        assert publish(host, description, "mast", [0.5]) is False
    assert host.robot_state_active is False
    assert host._retry_thread is None
    assert host._store.latest("mast").data is None
    assert host_warnings(logs) == [refused("can't start new thread")]
    # Nothing of the failed attempts is left but the lock file, and that is not held.
    assert files_beside(socket_path) == ["rs.sock.lock"]

    assert retry(host, description) is True
    assert publish(host, description, "mast", [0.75]) is True
    assert host.robot_state_active is True
    reader = connect(socket_path)
    assert reader.description() == description
    assert reader.joints("mast").sequence == 1
    assert host_warnings(logs) == [refused("can't start new thread")]


def test_where_there_are_no_unix_sockets_the_host_stores_nothing(
    socket_path, hosts, logs, monkeypatch
):
    monkeypatch.setattr(_local, "_HAS_UNIX_SOCKETS", False)
    description = rover()
    host = hosts(socket_path)
    host.start(description)
    assert host.robot_state_active is False
    assert publish(host, description, "mast", [0.5]) is False
    assert host._store.latest("mast").data is None
    assert host_warnings(logs) == [refused("this platform has no Unix domain sockets")]
    assert files_beside(socket_path) == []
