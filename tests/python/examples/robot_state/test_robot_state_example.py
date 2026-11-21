# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The robot state example end to end, over loopback.

A Teleop Session Server subprocess (--no-headset, a port the system picks, its reader socket
under a temporary folder), the example's stand-in robot through a remote teleop session, and
a reader of isaaccapture.robot_state.
"""

import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

from isaaccapture.robot_state import RobotState, load_robot_description, records
from isaaccapture_examples.robot_state import stand_in_robot

needs_server = pytest.mark.skipif(
    sys.platform == "win32", reason="the Teleop Session Server runs on Linux only"
)

BANNER = re.compile(r"^Teleop Session Server listening on (tcp://\S+)$", re.MULTILINE)
MARKER = "ISAACCAPTURE_TEST_EXAMPLE_MARKER"
SWEEP = range(stand_in_robot.SWEEP_RECORDS)
# Positions, velocities and efforts travel as 32-bit floats.
FLOAT32 = {"rel": 1e-6, "abs": 1e-6}
# The folder `python -m isaaccapture_examples.robot_state` is importable from.
EXAMPLE_PYTHON = Path(stand_in_robot.__file__).resolve().parents[2]


def _wait_for(predicate, timeout_s=10.0):
    """The first truthy value of predicate(), or its last value after timeout_s."""
    deadline = time.monotonic() + timeout_s
    while True:
        value = predicate()
        if value or time.monotonic() > deadline:
            return value
        time.sleep(0.01)


def _description(name):
    return load_robot_description(stand_in_robot.description_path(name))


# ============================================================================
# Processes
# ============================================================================


def _child_environment(folder):
    """An environment whose process leads its own logging, with its log files under `folder`."""
    env = dict(os.environ)
    env.pop("ISAACCAPTURE_LOG_SOCKET", None)
    env["ISAACCAPTURE_LOG_DIR"] = str(folder / "logs")
    return env


class _Server:
    """A Teleop Session Server subprocess that takes robot state."""

    def __init__(self, folder):
        self.socket_path = folder / "rs.sock"
        self.log_path = folder / "server.log"
        self._output_path = folder / "server.out"
        command = [
            sys.executable,
            "-m",
            "isaaccapture.teleop_session_server",
            "--no-headset",
            "--listen",
            "tcp://127.0.0.1:0",
            "--robot-state-socket",
            str(self.socket_path),
        ]
        with open(self._output_path, "wb") as output, open(self.log_path, "wb") as log:
            self.proc = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=log,
                env=_child_environment(folder),
                start_new_session=True,
            )
        self.killed = False
        self.endpoint = _wait_for(self._banner, 60.0)
        if not self.endpoint:
            self.proc.kill()
            self.proc.wait()
            pytest.fail(f"the server printed no listening banner:\n{self.log()}")

    def _banner(self):
        match = BANNER.search(self._output_path.read_text(errors="replace"))
        return match.group(1) if match else None

    def log(self):
        return self.log_path.read_text(errors="replace")

    def kill(self):
        """End the server the way a crash does: it closes and removes nothing."""
        self.killed = True
        self.proc.kill()
        self.proc.wait()

    def close(self):
        if self.killed:
            return
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
        try:
            code = self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
            pytest.fail(f"the server did not stop:\n{self.log()}")
        assert code == 0, self.log()


@pytest.fixture
def server(tmp_path_factory):
    # A short folder name: a Unix socket path holds about a hundred bytes.
    server = _Server(tmp_path_factory.mktemp("rs"))
    yield server
    server.close()


@contextmanager
def _stand_in_robot(server, name, *, loses_link=False):
    """The example's stand-in robot, publishing in a thread of this process until the block ends.

    Yields what became of it: "published" once it stopped, "error" when its session raised.
    With `loses_link` the block is expected to end the server, so an error is no failure.
    """
    stop = threading.Event()
    outcome = {}

    def publish():
        try:
            outcome["published"] = stand_in_robot.run(
                server.endpoint,
                stand_in_robot.description_path(name),
                stop=stop,
                connect_timeout_s=30.0,
            )
        # Kept for the assertions after the block.
        except Exception as error:
            outcome["error"] = error

    thread = threading.Thread(target=publish, daemon=True)
    thread.start()
    try:
        yield outcome
    except AssertionError as error:
        error.add_note(f"stand-in robot: {outcome}\nserver log:\n{server.log()}")
        raise
    finally:
        stop.set()
        thread.join(timeout=30.0)
    assert not thread.is_alive(), "the stand-in robot did not stop"
    if loses_link:
        return
    assert "error" not in outcome, f"{outcome['error']!r}\n{server.log()}"
    assert outcome["published"] > 0


def _running(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return text[text.rindex(")") + 2 :].split()[0] != "Z"


def _marked(marker):
    """Running processes whose environment carries `marker`."""
    entry = f"{MARKER}={marker}".encode()
    found = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            environ = Path(f"/proc/{name}/environ").read_bytes()
        except OSError:
            continue
        if entry in environ.split(b"\0") and _running(int(name)):
            found.append(int(name))
    return found


# ============================================================================
# What the reader must show
# ============================================================================


def _described(robot, name):
    what = robot.description()
    return what if what is not None and what.robot == name else None


def _aged(reading):
    """`reading` when it says an age between 0 and 1 s, else None."""
    if reading is None or reading.age_s is None:
        return None
    return reading if 0.0 <= reading.age_s <= 1.0 else None


def _check_joints(robot, name, series):
    arm = _wait_for(lambda: _aged(robot.joints(name)))
    assert arm is not None, repr(robot.joints(name, allow_stale=True))
    assert arm.series == name
    assert arm.fresh
    assert list(arm.names) == series.joint_names
    assert list(arm) == series.joint_names
    assert [arm[joint].position for joint in arm.names] == list(arm.positions)
    assert all(arm[joint].valid for joint in arm.names)
    records = [
        record
        for record in SWEEP
        if list(arm.positions)
        == pytest.approx(stand_in_robot.joint_positions(series, record), **FLOAT32)
    ]
    assert records, f"{name}: {arm.positions} is no record the stand-in robot published"
    for joint, (low, high) in series.limits.items():
        assert low <= arm[joint].position <= high
    for declared, values, published in (
        (
            series.has_velocity,
            [arm[joint].velocity for joint in arm.names],
            stand_in_robot.joint_velocities,
        ),
        (
            series.has_effort,
            [arm[joint].effort for joint in arm.names],
            stand_in_robot.joint_efforts,
        ),
    ):
        if declared:
            assert any(
                values == pytest.approx(published(series, record), **FLOAT32)
                for record in records
            )
        else:
            assert values == [None] * len(values)


def _with_fault(status):
    return status if status is not None and status.faults else None


def _check_status(robot, description, name, series):
    status = _wait_for(lambda: _aged(robot.status(name)))
    assert status is not None, repr(robot.status(name, allow_stale=True))
    assert list(status) == series.value_names
    assert status["motors_on"] is True
    if "battery" in status:
        assert status["battery"] == pytest.approx(24.0, abs=0.5)
    faulty = _wait_for(lambda: _with_fault(robot.status(name)))
    assert faulty is not None, "the stand-in robot's fault did not arrive"
    assert list(faulty.faults) == stand_in_robot.status_faults(
        description, stand_in_robot.FAULT_RECORDS
    )
    assert faulty["fault"] == stand_in_robot.FAULT_CODE
    fault = faulty.faults[0]
    assert (fault.code, fault.severity) == ("overtemperature", "warning")


def _check_custom(robot, name, series):
    reading = _wait_for(lambda: _aged(robot.record(name)))
    assert reading is not None, repr(robot.record(name, allow_stale=True))
    assert reading.message_encoding == "json"
    assert sorted(reading.values) == sorted(series.value_names)
    assert any(
        reading.values == pytest.approx(stand_in_robot.custom_record(series, record))
        for record in SWEEP
    ), f"{name}: {reading.values} is no record the stand-in robot published"


def _check_gripper(robot, description, name, gripper):
    reading = _wait_for(lambda: robot.gripper(name))
    assert reading is not None
    assert reading.name == name
    assert reading.valid
    series = description.series[gripper.series]
    index = series.joint_names.index(gripper.joints[0])
    assert any(
        reading.position
        == pytest.approx(
            stand_in_robot.joint_positions(series, record)[index], **FLOAT32
        )
        for record in SWEEP
    )
    opening = (reading.position - gripper.closed) / (gripper.open - gripper.closed)
    # The stand-in robot keeps every joint inside the middle 80 % of its range.
    assert 0.0 < reading.opening < 1.0
    assert reading.opening == pytest.approx(opening, abs=1e-6)


def _check_robot(robot, name):
    """The reader shows the robot of the bundled description `name`; returns that description."""
    expected = _description(name)
    what = _wait_for(lambda: _described(robot, expected.robot))
    assert what is not None, f"read {robot.description()!r}"
    assert robot.server_reachable
    assert robot.connected
    assert list(what.series) == list(expected.series)
    for series_name, series in expected.series.items():
        assert what.series[series_name].shape_name == series.shape_name
        assert what.series[series_name].joint_names == series.joint_names
        assert what.series[series_name].value_names == series.value_names
        assert what.series[series_name].limits == series.limits
    assert what.grippers == expected.grippers
    assert what == expected
    assert robot.series() == expected.series

    for series_name, series in expected.series.items():
        if series.shape == "joints":
            _check_joints(robot, series_name, series)
        elif series.shape == "status":
            _check_status(robot, expected, series_name, series)
        else:
            _check_custom(robot, series_name, series)
    for gripper_name, gripper in expected.grippers.items():
        _check_gripper(robot, expected, gripper_name, gripper)
    return expected


def _check_robot_left(robot, description):
    """The robot app left: nothing reads as current; its description and last records stay."""
    assert _wait_for(lambda: not robot.connected, 5.0), "connected stayed True"
    assert robot.server_reachable
    for name in description.series:
        assert robot.record(name) is None
        assert robot.record(name, allow_stale=True) is not None
    for name in description.grippers:
        assert robot.gripper(name) is None
    assert robot.description() == description


# ============================================================================
# Tests
# ============================================================================


def test_the_bundled_descriptions_cover_every_shape():
    arm = _description("six_joint_arm")
    assert arm.robot == "rebot_b601"
    assert [series.shape for series in arm.series.values()] == ["joints", "status"]
    assert list(arm.grippers) == ["gripper"]

    two_arm = _description("two_arm_robot")
    assert two_arm.robot == "two_arm_example"
    shapes = {series.shape for series in two_arm.series.values()}
    assert shapes == {"joints", "status", "custom"}
    assert list(two_arm.grippers) == ["l_gripper", "r_gripper"]
    assert any(series.has_effort for series in two_arm.series.values())


class _EncodingSession:
    """The two publishing calls of TeleopSession, encoding as the library does; keeps the bytes."""

    def __init__(self, description):
        self._description = description
        self.records = {}

    def publish_robot_state(self, series, values, **arguments):
        self.records[series] = records.encode_reading(
            self._description, series, values, time_ns=time.monotonic_ns(), **arguments
        )
        return True

    def publish_robot_record(self, series, record):
        return self.publish_robot_state(series, record)


@pytest.mark.parametrize("name", ["six_joint_arm", "two_arm_robot"])
def test_the_stand_in_robot_publishes_records_that_fit_its_description(name):
    description = _description(name)
    session = _EncodingSession(description)
    for record in (0, stand_in_robot.FAULT_RECORDS, stand_in_robot.SWEEP_RECORDS - 1):
        assert stand_in_robot.publish_all(session, description, record)
        assert list(session.records) == list(description.series)
        for series_name, series in description.series.items():
            data = session.records[series_name]
            if series.shape == "joints":
                joints = records.decode_joints(data)
                assert [joint.name for joint in joints] == series.joint_names
                assert [joint.position for joint in joints] == pytest.approx(
                    stand_in_robot.joint_positions(series, record), **FLOAT32
                )
            elif series.shape == "status":
                readings, faults = records.decode_status(data)
                assert list(readings) == series.value_names
                assert bool(faults) == stand_in_robot.fault_active(record)
            else:
                assert records.decode_custom(data) == stand_in_robot.custom_record(
                    series, record
                )


@needs_server
@pytest.mark.parametrize(
    "first, second",
    [("six_joint_arm", "two_arm_robot"), ("two_arm_robot", "six_joint_arm")],
)
def test_a_reader_sees_what_each_stand_in_robot_publishes(server, first, second):
    with RobotState.connect(socket_path=server.socket_path) as robot:
        # The server opens its reader socket when the first robot app agreed robot state.
        assert not robot.server_reachable
        assert not robot.connected
        assert robot.description() is None
        assert robot.series() == {}

        with _stand_in_robot(server, first):
            description = _check_robot(robot, first)
        _check_robot_left(robot, description)

        gone = next(
            name
            for name in description.series
            if name not in _description(second).series
        )
        with _stand_in_robot(server, second):
            _check_robot(robot, second)
            # The same reader shows the other robot: a series of the first one is unknown.
            with pytest.raises(KeyError):
                robot.record(gone)


@needs_server
@pytest.mark.parametrize("stopped", ["shut_down", "killed"])
def test_a_reader_follows_a_server_that_stops_and_runs_again(tmp_path_factory, stopped):
    folder = tmp_path_factory.mktemp("rs")
    server = _Server(folder)
    try:
        with RobotState.connect(socket_path=server.socket_path) as robot:
            with _stand_in_robot(server, "six_joint_arm", loses_link=True) as outcome:
                _check_robot(robot, "six_joint_arm")
                if stopped == "killed":
                    server.kill()
                else:
                    server.close()
                # While no server answers nothing of the robot is readable, its last records included.
                assert _wait_for(lambda: not robot.server_reachable, 5.0)
                assert not robot.connected
                assert robot.description() is None
                assert robot.series() == {}
                # The robot app's session ends with the server.
                assert _wait_for(lambda: "error" in outcome), "the session went on"

            # Same reader socket path; a killed server left its socket file behind.
            server = _Server(folder)
            # The server opens its reader socket when a robot app agreed robot state.
            assert not robot.server_reachable
            with _stand_in_robot(server, "two_arm_robot"):
                description = _check_robot(robot, "two_arm_robot")
            _check_robot_left(robot, description)
    finally:
        server.close()


@needs_server
def test_the_example_runs_for_a_person(tmp_path):
    expected = _description("two_arm_robot")
    marker = uuid.uuid4().hex
    env = _child_environment(tmp_path)
    env[MARKER] = marker
    env["PYTHONPATH"] = os.pathsep.join(
        [str(EXAMPLE_PYTHON), *filter(None, [env.get("PYTHONPATH")])]
    )
    command = [
        sys.executable,
        "-m",
        "isaaccapture_examples.robot_state",
        "--description",
        "two_arm_robot",
        "--duration-s",
        "8",
        "--period-s",
        "0.5",
    ]
    done = subprocess.run(
        command,
        check=False,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    output = f"{done.stdout}\n{done.stderr}"
    assert done.returncode == 0, output
    assert f"robot {expected.robot}\n" in done.stdout, output
    for name, series in expected.series.items():
        assert re.search(rf"^  {name} \({series.shape_name}\), ", done.stdout, re.M), (
            output
        )
        for value in (*series.joint_names, *series.value_names):
            assert re.search(rf"^    {value} ", done.stdout, re.M), output
    for name in expected.grippers:
        assert re.search(rf"^  gripper {name}: \d+ % open$", done.stdout, re.M), output
    assert _wait_for(lambda: not _marked(marker)), "the example left processes running"
