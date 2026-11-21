# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reading robot state in the server's program: `RobotState.from_store()` on a store filled by hand."""

import logging
import math
import threading
from collections.abc import Mapping

import pytest
from repo_paths import repo_root

from isaaccapture.robot_state import (
    Fault,
    GripperReading,
    JointReading,
    JointsReading,
    Reading,
    RecordReading,
    RobotState,
    RobotStateLocalService,
    RobotStateStore,
    StatusReading,
    _local,
    load_robot_description,
    parse_robot_description,
    records,
)
from isaaccapture.robot_state._robot_state import DESCRIPTION_TOPIC, monotonic_now_ns

DESCRIPTIONS = (
    repo_root()
    / "examples"
    / "robot_state"
    / "python"
    / "isaaccapture_examples"
    / "robot_state"
    / "descriptions"
)

MS = 1_000_000
SECOND = 1_000 * MS

# Series that stay fresh for a minute, one that never goes stale, a gripper of two joints with
# its open position below its closed one, and a shape of the robot's own.
ROVER = """
robot = "rover"

[shapes.range]
fields = ["near", { name = "label", kind = "text" }]

[series.arm]
shape = "joints"
stale_after_s = 60.0
has_velocity = true
joints = [
  { name = "lift", kind = "prismatic", min = 0.0, max = 0.4 },
  { name = "jaw", kind = "gripper", min = 0.0, max = 0.04, unit = "m", open = 0.03, closed = 0.01 },
  { name = "thumb", kind = "revolute", min = -1.0, max = 1.0 },
]

[series.mast]
shape = "joints"
stale_after_s = 0
joints = [{ name = "turn", kind = "continuous" }]

[series.quick]
shape = "joints"
joints = [{ name = "wheel", kind = "continuous" }]

[series.health]
shape = "status"
stale_after_s = 60.0
readings = ["ok", { name = "volts", kind = "number", unit = "V" }, { name = "mode", kind = "text" }]

[series.sonar]
shape = "range"
stale_after_s = 60.0

[grippers.pinch]
series = "arm"
joints = ["thumb", "jaw"]
open = -0.5
closed = 0.5
"""


def six_joint_arm():
    return load_robot_description(DESCRIPTIONS / "six_joint_arm.toml")


def two_arm_robot():
    return load_robot_description(DESCRIPTIONS / "two_arm_robot.toml")


def rover():
    return parse_robot_description(ROVER)


def start_robot(store, description, *, send_description=True):
    """What a robot app that agreed robot state leaves in the server's store."""
    store.begin_session()
    for series in records.declarations(description):
        store.declare_series(
            series.name,
            series.shape,
            series.schema_name,
            series.schema_encoding,
            series.schema_data,
            series.message_encoding,
            series.stale_after_ns,
            series.in_order,
            series.rate_hz,
        )
    if send_description:
        now = monotonic_now_ns()
        store.put_record(
            DESCRIPTION_TOPIC,
            records.encode_description(description, now),
            now,
            robot_minus_server_ns=0,
            clock_two_way=True,
        )


def put(
    store,
    description,
    series,
    values,
    *,
    age_s=0.0,
    robot_minus_server_ns=0,
    clock_two_way=True,
    **reading,
):
    """One reading taken `age_s` ago; returns its time on the robot's clock."""
    sample_time_ns = monotonic_now_ns() - round(age_s * SECOND)
    if robot_minus_server_ns is not None:
        sample_time_ns += robot_minus_server_ns
    data = records.encode_reading(
        description, series, values, time_ns=sample_time_ns, **reading
    )
    store.put_record(
        series,
        data,
        sample_time_ns,
        robot_minus_server_ns=robot_minus_server_ns,
        clock_two_way=clock_two_way,
    )
    return sample_time_ns


class StoreFilledMeanwhile:
    """A store that `then()` changes right after one read of `series`, as another thread may."""

    def __init__(self, store, series, then):
        self._store = store
        self._series = series
        self._then = [then]

    def __getattr__(self, name):
        return getattr(self._store, name)

    def latest(self, name, *arguments):
        record = self._store.latest(name, *arguments)
        if name == self._series and self._then:
            self._then.pop()()
        return record


@pytest.fixture
def store():
    return RobotStateStore()


@pytest.fixture
def robot(store):
    return RobotState.from_store(store)


@pytest.fixture
def warnings(caplog):
    caplog.set_level(logging.INFO, logger="isaaccapture.robot_state")
    return caplog


def warning_texts(caplog):
    return [
        entry.getMessage()
        for entry in caplog.records
        if entry.levelno == logging.WARNING
    ]


# ---------------------------------------------------------------------------------------------
# The calls of the plan document, on the six-joint arm
# ---------------------------------------------------------------------------------------------


def test_a_reader_that_knows_nothing_about_the_robot_reads_the_six_joint_arm(
    store, robot
):
    description = six_joint_arm()
    positions = [0.1, 1.204, 0.3, 0.4, 0.5, 0.6, 0.96]
    start_robot(store, description)

    what = robot.description()
    assert what.robot == "rebot_b601"
    assert list(what.series) == ["arm", "status"]
    assert what.series["arm"].joint_names == [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "gripper",
    ]
    assert what.series["arm"].limits["joint2"] == (-0.05, 3.14)

    # Each reading is put right before it is read: this arm's go stale after half a second.
    put(store, description, "arm", positions, age_s=0.03)
    arm = robot.joints("arm")
    assert arm["joint2"].position == pytest.approx(1.204, abs=1e-6)
    assert arm["gripper"].position == pytest.approx(0.96, abs=1e-6)
    assert 0.03 <= arm.age_s <= 0.5
    put(store, description, "arm", positions)
    assert robot.gripper("gripper").opening == pytest.approx(0.96, abs=1e-6)
    put(store, description, "status", [True, ""])
    status = robot.status()
    assert status["fault"] == ""
    assert status["motors_on"] is True


def test_the_description_read_back_equals_the_one_sent(store, robot):
    for description in (six_joint_arm(), two_arm_robot(), rover()):
        start_robot(store, description)
        assert robot.description() == description
        assert robot.series() == dict(description.series)
        assert list(robot.series()) == list(description.series)


def test_a_two_arm_robot_differs_only_in_its_description(store, robot):
    description = two_arm_robot()
    start_robot(store, description)
    efforts = [1.0, 2.0, 3.0, 4.0, 5.0]
    put(
        store,
        description,
        "left_arm",
        [0.1, 0.2, 0.3, 0.4, 0.04],
        efforts=efforts,
    )
    put(
        store,
        description,
        "right_arm",
        [-0.1, -0.2, -0.3, -0.4, 0.08],
        efforts=efforts,
    )
    put(store, description, "waist", [0.5])
    put(store, description, "left_wrist_force", [1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    put(
        store,
        description,
        "status",
        [True, "", 24.5],
        faults=[("overtemperature", "The elbow is hot", "l_elbow", "warning")],
    )

    # Freshness is not the subject here: the readings are taken whatever their age.
    every_joint = {
        name: robot.joints(name, allow_stale=True)
        for name, series in robot.series().items()
        if series.shape == "joints"
    }
    assert list(every_joint) == ["left_arm", "right_arm", "waist"]
    assert every_joint["left_arm"].names == (
        "l_shoulder_pan",
        "l_shoulder_lift",
        "l_elbow",
        "l_wrist_roll",
        "l_gripper",
    )
    assert every_joint["right_arm"]["r_elbow"].position == pytest.approx(-0.3)
    assert every_joint["left_arm"]["l_elbow"].effort == pytest.approx(3.0)
    assert every_joint["left_arm"]["l_elbow"].velocity is None
    assert every_joint["waist"].positions == pytest.approx((0.5,))

    left = robot.gripper("l_gripper", allow_stale=True)
    right = robot.gripper("r_gripper", allow_stale=True)
    assert left.opening == pytest.approx(0.5, abs=1e-6)
    assert right.opening == pytest.approx(1.0, abs=1e-6)
    assert right.series == "right_arm"

    force = robot.record("left_wrist_force", allow_stale=True)
    assert force.values == {
        "fx": 1.0,
        "fy": 2.0,
        "fz": 3.0,
        "tx": 0.1,
        "ty": 0.2,
        "tz": 0.3,
    }
    assert robot.record("right_wrist_force", allow_stale=True) is None

    status = robot.status(allow_stale=True)
    assert status["battery"] == pytest.approx(24.5)
    assert status.faults == (
        Fault("overtemperature", "The elbow is hot", "l_elbow", "warning"),
    )


# ---------------------------------------------------------------------------------------------
# A store read in the server's program
# ---------------------------------------------------------------------------------------------


def test_a_store_in_this_program_is_always_reachable(store, robot):
    assert robot.server_reachable is True
    assert robot.refresh() is True
    assert robot.connected is False
    store.begin_session()
    assert robot.connected is True
    store.end_session()
    assert robot.connected is False
    assert robot.server_reachable is True


def test_without_a_description_every_read_is_none(store, robot):
    assert robot.description() is None
    assert robot.series() == {}
    assert robot.joints("arm") is None
    assert robot.gripper("gripper") is None
    assert robot.status() is None
    assert robot.record("anything", allow_stale=True) is None

    # The series are declared and a reading is there, but the description has not arrived.
    description = rover()
    start_robot(store, description, send_description=False)
    put(store, description, "arm", [0.1, 0.02, 0.0], velocities=[0.0, 0.0, 0.0])
    assert robot.description() is None
    assert robot.joints("arm", allow_stale=True) is None


def test_a_series_without_a_record_reads_as_none(store, robot):
    start_robot(store, rover())
    assert robot.joints("arm") is None
    assert robot.joints("arm", allow_stale=True) is None
    assert robot.gripper("jaw") is None
    assert robot.status("health") is None
    assert robot.record("sonar") is None


def test_an_unknown_series_is_a_key_error_that_names_the_known_ones(store, robot):
    start_robot(store, rover())
    for read in (robot.joints, robot.status, robot.record):
        with pytest.raises(KeyError) as raised:
            read("leg")
        assert raised.value.args[0] == (
            "robot 'rover' has no series 'leg' (it has: arm, mast, quick, health, sonar)"
        )
    # The default status series is "status"; this robot calls its own "health".
    with pytest.raises(KeyError, match="has no series 'status'"):
        robot.status()
    # The description's own series is not one of the robot's.
    with pytest.raises(KeyError):
        robot.record(DESCRIPTION_TOPIC)


def test_an_unknown_gripper_is_a_key_error_that_names_the_known_ones(store, robot):
    start_robot(store, rover())
    with pytest.raises(KeyError) as raised:
        robot.gripper("hand")
    assert raised.value.args[0] == (
        "robot 'rover' has no gripper 'hand' (it has: jaw, pinch)"
    )

    start_robot(
        store,
        parse_robot_description(
            'robot = "bare"\n[series.s]\nshape = "status"\nreadings = ["a"]\n'
        ),
    )
    with pytest.raises(KeyError, match=r"has no gripper 'jaw' \(it has: none\)"):
        robot.gripper("jaw")


def test_a_series_of_another_shape_is_a_value_error(store, robot):
    description = rover()
    start_robot(store, description)
    put(store, description, "health", [True, 12.0, "idle"])
    with pytest.raises(
        ValueError, match="series 'health' has shape 'status', not 'joints'"
    ):
        robot.joints("health")
    with pytest.raises(
        ValueError, match="series 'arm' has shape 'joints', not 'status'"
    ):
        robot.status("arm")
    with pytest.raises(
        ValueError, match="series 'sonar' has shape 'range', not 'joints'"
    ):
        robot.joints("sonar")


# ---------------------------------------------------------------------------------------------
# Joints, grippers, status, a shape of the robot's own
# ---------------------------------------------------------------------------------------------


def test_joints_are_read_by_name_in_the_record_s_order(store, robot):
    description = rover()
    start_robot(store, description)
    put(store, description, "arm", [0.25, 0.02, -0.5], velocities=[0.1, 0.0, -0.2])

    arm = robot.joints("arm")
    assert isinstance(arm, JointsReading)
    assert isinstance(arm, Reading)
    assert isinstance(arm, Mapping)
    assert arm.series == "arm"
    assert len(arm) == 3
    assert list(arm) == ["lift", "jaw", "thumb"]
    assert arm.names == ("lift", "jaw", "thumb")
    assert arm.positions == pytest.approx((0.25, 0.02, -0.5))
    assert "jaw" in arm
    assert "elbow" not in arm
    assert arm.get("elbow") is None
    with pytest.raises(KeyError):
        arm["elbow"]
    lift = arm["lift"]
    assert isinstance(lift, JointReading)
    assert (lift.name, lift.valid, lift.effort) == ("lift", True, None)
    assert lift.position == pytest.approx(0.25)
    assert lift.velocity == pytest.approx(0.1)
    assert [joint.name for joint in arm.joints] == ["lift", "jaw", "thumb"]
    assert [joint.name for joint in arm.values()] == ["lift", "jaw", "thumb"]
    with pytest.raises(AttributeError):
        arm.series = "other"


def test_a_joint_the_robot_could_not_read_is_not_valid(store, robot):
    description = rover()
    start_robot(store, description)
    put(
        store,
        description,
        "arm",
        [0.25, math.nan, -0.5],
        velocities=[0.1, math.nan, -0.2],
        valid=[True, False, True],
    )
    arm = robot.joints("arm")
    assert arm["jaw"].valid is False
    assert arm["lift"].valid is True
    # Not the 0.0 its record carries, which would read as a position.
    assert math.isnan(arm["jaw"].position)
    assert math.isnan(arm["jaw"].velocity)
    assert arm["jaw"].effort is None
    assert math.isnan(arm.positions[1])
    assert (arm.positions[0], arm.positions[2]) == pytest.approx((0.25, -0.5))
    assert arm["lift"].velocity == pytest.approx(0.1)
    assert arm.names == ("lift", "jaw", "thumb")

    jaw = robot.gripper("jaw")
    assert jaw.valid is False
    assert math.isnan(jaw.position)
    assert math.isnan(jaw.opening)
    # The gripper of two joints measures its opening on the first, which was read.
    assert robot.gripper("pinch").valid is True


def test_a_gripper_opening_runs_from_closed_to_open_and_is_clamped(store, robot):
    description = rover()
    start_robot(store, description)
    zeros = [0.0, 0.0, 0.0]

    # jaw: closed at 0.01, open at 0.03. pinch: thumb closed at 0.5, open at -0.5.
    for jaw_position, thumb_position, jaw_opening, pinch_opening in (
        (0.02, 0.0, 0.5, 0.5),
        (0.03, -0.5, 1.0, 1.0),
        (0.01, 0.5, 0.0, 0.0),
        (0.04, -1.0, 1.0, 1.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.015, 0.25, 0.25, 0.25),
    ):
        put(
            store,
            description,
            "arm",
            [0.1, jaw_position, thumb_position],
            velocities=zeros,
        )
        jaw = robot.gripper("jaw")
        assert isinstance(jaw, GripperReading)
        assert (jaw.name, jaw.series, jaw.valid) == ("jaw", "arm", True)
        assert jaw.position == pytest.approx(jaw_position)
        assert jaw.opening == pytest.approx(jaw_opening, abs=1e-5)
        pinch = robot.gripper("pinch")
        assert pinch.position == pytest.approx(thumb_position)
        assert pinch.opening == pytest.approx(pinch_opening, abs=1e-5)
        assert pinch.sequence == robot.joints("arm").sequence


def test_status_readings_are_read_by_name_with_the_faults(store, robot):
    description = rover()
    start_robot(store, description)
    put(
        store,
        description,
        "health",
        [False, None, "parked"],
        faults=[("stall", "The lift stalled", "lift"), Fault("low", "Battery low")],
    )

    health = robot.status("health")
    assert isinstance(health, StatusReading)
    assert isinstance(health, Mapping)
    assert dict(health) == {"ok": False, "volts": None, "mode": "parked"}
    assert list(health) == ["ok", "volts", "mode"]
    assert len(health) == 3
    assert health["mode"] == "parked"
    assert health["volts"] is None
    assert "ok" in health
    assert health.get("speed", 0.0) == 0.0
    assert health.faults == (
        Fault("stall", "The lift stalled", "lift", "error"),
        Fault("low", "Battery low", "", "error"),
    )

    put(store, description, "health", [True, 12.5, "driving"])
    health = robot.status("health")
    assert health["volts"] == pytest.approx(12.5)
    assert health.faults == ()


def test_record_gives_each_series_in_its_own_kind_of_reading(store, robot):
    description = rover()
    start_robot(store, description)
    put(store, description, "arm", [0.25, 0.02, -0.5], velocities=[0.0, 0.0, 0.0])
    put(store, description, "health", [True, 12.0, "idle"])
    put(store, description, "sonar", {"near": 0.75, "label": "wall"})

    assert isinstance(robot.record("arm"), JointsReading)
    assert robot.record("arm")["lift"].position == pytest.approx(0.25)
    assert isinstance(robot.record("health"), StatusReading)
    assert robot.record("health")["mode"] == "idle"

    sonar = robot.record("sonar")
    assert isinstance(sonar, RecordReading)
    assert sonar.series == "sonar"
    assert sonar.shape == "range"
    assert sonar.schema_name == "rover.range"
    assert sonar.message_encoding == "json"
    assert sonar.values == {"near": 0.75, "label": "wall"}
    assert sonar.data == b'{"near":0.75,"label":"wall"}'
    assert sonar.fresh is True
    assert sonar.sequence == 1


def test_a_record_in_an_encoding_the_library_does_not_know_is_handed_out_as_bytes(
    store, robot
):
    description = rover()
    start_robot(store, description)
    # A robot app declares the series itself, with a schema and encoding this build never saw.
    store.declare_series(
        "sonar", "range", "vendor.Range", "protobuf", b"\x0a\x05Range", "protobuf", 0
    )
    store.put_record("sonar", b"\x08\x96\x01", monotonic_now_ns())

    sonar = robot.record("sonar")
    assert isinstance(sonar, RecordReading)
    assert sonar.values is None
    assert sonar.data == b"\x08\x96\x01"
    assert (sonar.shape, sonar.schema_name, sonar.message_encoding) == (
        "range",
        "vendor.Range",
        "protobuf",
    )


def test_a_record_that_cannot_be_decoded_reads_as_none_with_one_warning(
    store, robot, warnings
):
    description = rover()
    start_robot(store, description)
    now = monotonic_now_ns()
    store.put_record("arm", b"not a flatbuffer", now)
    store.put_record("health", b"\x00\x01", now)
    store.put_record("sonar", b"{not json", now)

    for _ in range(3):
        assert robot.joints("arm") is None
        assert robot.gripper("jaw") is None
        assert robot.status("health") is None
        sonar = robot.record("sonar")
        assert sonar.values is None
        assert sonar.data == b"{not json"
    texts = warning_texts(warnings)
    assert len(texts) == 3
    assert sum("series 'arm'" in text for text in texts) == 1
    assert sum("series 'health'" in text for text in texts) == 1
    assert sum("series 'sonar'" in text for text in texts) == 1

    # A good record after it is read again.
    put(store, description, "arm", [0.25, 0.02, -0.5], velocities=[0.0, 0.0, 0.0])
    assert robot.joints("arm")["lift"].position == pytest.approx(0.25)


def test_a_series_declared_with_another_shape_than_its_description_is_refused(
    store, robot
):
    description = rover()
    start_robot(store, description)
    declared = {series.name: series for series in records.declarations(description)}
    health = declared["health"]
    store.declare_series(
        "arm",
        health.shape,
        health.schema_name,
        health.schema_encoding,
        health.schema_data,
        health.message_encoding,
        0,
    )
    put_time = monotonic_now_ns()
    store.put_record(
        "arm",
        records.encode_reading(
            description, "health", [True, 1.0, "x"], time_ns=put_time
        ),
        put_time,
    )
    with pytest.raises(ValueError, match="series 'arm' does not carry joints records"):
        robot.joints("arm")
    with pytest.raises(ValueError, match="series 'arm' does not carry joints records"):
        robot.gripper("jaw")
    assert isinstance(robot.record("arm"), StatusReading)


# ---------------------------------------------------------------------------------------------
# Age and freshness
# ---------------------------------------------------------------------------------------------


def test_a_reading_says_how_old_it_is(store, robot):
    description = rover()
    start_robot(store, description)
    offset = 5_000 * SECOND
    sample_time_ns = put(
        store,
        description,
        "arm",
        [0.25, 0.02, -0.5],
        velocities=[0.0, 0.0, 0.0],
        age_s=2.0,
        robot_minus_server_ns=offset,
    )

    arm = robot.joints("arm")
    # The time of the reading is on the robot's clock; the age is on this machine's.
    assert arm.sample_time_ns == sample_time_ns
    assert arm.sample_time_ns > monotonic_now_ns() + offset - 60 * SECOND
    assert 2.0 <= arm.age_s < 30.0
    assert 0.0 <= arm.receive_age_s < arm.age_s - 1.9
    assert arm.clock_two_way is True
    assert arm.fresh is True
    assert arm.sequence == 1

    put(
        store,
        description,
        "arm",
        [0.25, 0.02, -0.5],
        velocities=[0.0, 0.0, 0.0],
        clock_two_way=False,
    )
    arm = robot.joints("arm")
    assert arm.clock_two_way is False
    assert arm.sequence == 2
    assert 0.0 <= arm.age_s < 30.0


def test_without_a_clock_estimate_the_age_is_unknown_and_freshness_follows_arrival(
    store, robot
):
    description = rover()
    start_robot(store, description)
    # The reading's own time says nothing without an estimate of the robot's clock: by its
    # number it is 3 s old, and "quick" goes stale after half a second.
    put(store, description, "quick", [1.0], age_s=3.0, robot_minus_server_ns=None)
    quick = robot.joints("quick")
    assert quick.age_s is None
    assert 0.0 <= quick.receive_age_s <= 0.5
    assert quick.fresh is True
    assert quick.clock_two_way is False

    data = records.encode_reading(description, "quick", [1.0], time_ns=0)
    store.put_record("quick", data, 0, receive_time_ns=monotonic_now_ns() - 2 * SECOND)
    assert robot.joints("quick") is None
    quick = robot.joints("quick", allow_stale=True)
    assert quick.age_s is None
    assert quick.receive_age_s >= 2.0
    assert quick.fresh is False


def test_a_stale_record_reads_as_none_unless_allowed(store, robot):
    description = rover()
    start_robot(store, description)
    # "quick" goes stale after the default half second.
    put(store, description, "quick", [1.5], age_s=2.0)

    assert robot.joints("quick") is None
    assert robot.record("quick") is None
    quick = robot.joints("quick", allow_stale=True)
    assert quick["wheel"].position == pytest.approx(1.5)
    assert quick.fresh is False
    assert quick.age_s >= 2.0
    assert robot.record("quick", allow_stale=True).fresh is False

    put(store, description, "quick", [2.5])
    quick = robot.joints("quick")
    assert quick["wheel"].position == pytest.approx(2.5)
    assert quick.fresh is True


def test_each_series_goes_stale_by_its_own_rule(store, robot):
    description = rover()
    start_robot(store, description)
    put(store, description, "quick", [1.0], age_s=5.0)
    put(
        store,
        description,
        "arm",
        [0.25, 0.02, -0.5],
        velocities=[0.0, 0.0, 0.0],
        age_s=5.0,
    )
    put(store, description, "health", [True, 12.0, "idle"], age_s=5.0)
    put(store, description, "sonar", [0.5, "wall"], age_s=5.0)
    # The robot's clock runs ahead, so a reading of a day ago still has a positive time.
    put(
        store,
        description,
        "mast",
        [3.0],
        age_s=86_400.0,
        robot_minus_server_ns=200_000 * SECOND,
    )

    assert robot.joints("quick") is None  # half a second
    assert robot.joints("arm").fresh is True  # a minute
    assert robot.gripper("jaw").fresh is True
    assert robot.status("health").fresh is True
    assert robot.record("sonar").fresh is True
    mast = robot.joints("mast")  # never stale
    assert mast.fresh is True
    assert mast.age_s >= 86_400.0

    put(
        store,
        description,
        "arm",
        [0.25, 0.02, -0.5],
        velocities=[0.0, 0.0, 0.0],
        age_s=61.0,
    )
    put(store, description, "health", [True, 12.0, "idle"], age_s=61.0)
    put(store, description, "sonar", [0.5, "wall"], age_s=61.0)
    assert robot.joints("arm") is None
    assert robot.gripper("jaw") is None
    assert robot.gripper("jaw", allow_stale=True).fresh is False
    assert robot.status("health") is None
    assert robot.status("health", allow_stale=True)["mode"] == "idle"
    assert robot.record("sonar") is None
    assert robot.record("sonar", allow_stale=True).values == {
        "near": 0.5,
        "label": "wall",
    }


def test_the_description_never_goes_stale(store, robot):
    description = rover()
    start_robot(store, description, send_description=False)
    # Written a day ago by the robot's clock, which runs ahead of this machine's.
    ahead = 200_000 * SECOND
    long_ago = monotonic_now_ns() + ahead - 86_400 * SECOND
    store.put_record(
        DESCRIPTION_TOPIC,
        records.encode_description(description, long_ago),
        long_ago,
        robot_minus_server_ns=ahead,
    )
    assert store.latest(DESCRIPTION_TOPIC).age_ns >= 86_400 * SECOND
    assert robot.description() == description


# ---------------------------------------------------------------------------------------------
# Robot apps that come and go
# ---------------------------------------------------------------------------------------------


def test_after_the_robot_app_left_reads_are_none_unless_allowed(store, robot):
    description = rover()
    start_robot(store, description)
    put(store, description, "arm", [0.25, 0.02, -0.5], velocities=[0.0, 0.0, 0.0])
    put(store, description, "mast", [3.0])
    put(store, description, "health", [True, 12.0, "idle"])
    put(store, description, "sonar", [0.5, "wall"])
    assert robot.joints("mast").fresh is True

    store.end_session()
    assert robot.connected is False
    # The record of "mast" never goes stale by its age, but its robot app is gone.
    assert robot.joints("mast") is None
    assert robot.joints("arm") is None
    assert robot.gripper("jaw") is None
    assert robot.status("health") is None
    assert robot.record("sonar") is None

    mast = robot.joints("mast", allow_stale=True)
    assert mast["turn"].position == pytest.approx(3.0)
    assert mast.fresh is False
    assert robot.joints("arm", allow_stale=True)["lift"].position == pytest.approx(0.25)
    assert robot.gripper("jaw", allow_stale=True).opening == pytest.approx(
        0.5, abs=1e-5
    )
    assert robot.status("health", allow_stale=True)["ok"] is True
    assert robot.record("sonar", allow_stale=True).fresh is False

    # What the robot is stays readable until the next one connects.
    assert robot.description() == description
    assert list(robot.series()) == ["arm", "mast", "quick", "health", "sonar"]
    with pytest.raises(KeyError):
        robot.joints("leg")


def test_the_next_robot_app_replaces_everything_of_the_one_before(store, robot):
    first = rover()
    start_robot(store, first)
    put(store, first, "arm", [0.25, 0.02, -0.5], velocities=[0.0, 0.0, 0.0])
    assert robot.description().robot == "rover"
    store.end_session()

    second = six_joint_arm()
    start_robot(store, second, send_description=False)
    # Connected, but its description has not arrived: nothing of the first robot is shown.
    assert robot.connected is True
    assert robot.description() is None
    assert robot.series() == {}
    assert robot.joints("arm", allow_stale=True) is None

    now = monotonic_now_ns()
    store.put_record(DESCRIPTION_TOPIC, records.encode_description(second, now), now)
    assert robot.description().robot == "rebot_b601"
    assert robot.joints("arm") is None
    put(store, second, "arm", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0])
    arm = robot.joints("arm")
    assert arm.names[-1] == "gripper"
    assert arm.sequence == 1
    with pytest.raises(KeyError, match="has no gripper 'jaw'"):
        robot.gripper("jaw")
    assert robot.gripper("gripper").opening == pytest.approx(1.0)


@pytest.mark.parametrize("allow_stale", [False, True])
def test_a_record_read_while_the_next_robot_app_connects_is_not_handed_out(
    store, allow_stale
):
    description = six_joint_arm()
    start_robot(store, description)
    put(store, description, "arm", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    def next_robot_app():
        store.end_session()
        start_robot(store, description)

    # The record is the first robot app's; `connected` and the declaration of "arm" are read
    # after the next one connected.
    robot = RobotState.from_store(StoreFilledMeanwhile(store, "arm", next_robot_app))
    assert robot.joints("arm", allow_stale=allow_stale) is None
    assert store.session_count == 2
    assert robot.connected is True

    put(store, description, "arm", [0.9, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    arm = robot.joints("arm", allow_stale=allow_stale)
    assert arm["joint1"].position == pytest.approx(0.9)
    assert (arm.fresh, arm.sequence) == (True, 1)


def test_a_record_is_not_read_by_the_description_of_the_robot_app_before(store):
    first = rover()
    start_robot(store, first)
    put(store, first, "arm", [0.25, 0.02, -0.5], velocities=[0.0, 0.0, 0.0])
    assert RobotState.from_store(store).gripper("jaw").opening == pytest.approx(
        0.5, abs=1e-5
    )
    # Another robot whose series "arm" also has a gripper "jaw", with other limits.
    second = parse_robot_description(
        """
robot = "other"

[series.arm]
shape = "joints"
stale_after_s = 60.0
joints = [{ name = "jaw", kind = "gripper", min = 0.0, max = 1.0 }]
"""
    )

    def next_robot_app():
        store.end_session()
        start_robot(store, second)
        put(store, second, "arm", [0.02])

    # The description is the first robot app's; the record is the next one's.
    robot = RobotState.from_store(
        StoreFilledMeanwhile(store, DESCRIPTION_TOPIC, next_robot_app)
    )
    assert robot.gripper("jaw") is None
    assert robot.description().robot == "other"
    assert robot.gripper("jaw").opening == pytest.approx(0.02)


def test_the_description_is_decoded_once_per_record(store, robot, monkeypatch):
    decoded = []
    decode = records.decode_description

    def counting(data):
        decoded.append(len(data))
        return decode(data)

    monkeypatch.setattr(records, "decode_description", counting)

    first = rover()
    start_robot(store, first)
    what = robot.description()
    put(store, first, "mast", [3.0])
    assert robot.description() is what
    assert robot.series()["arm"] is what.series["arm"]
    robot.joints("mast")
    robot.gripper("jaw")
    assert len(decoded) == 1

    # The next robot session: its description record has the same number as the first one's.
    second = six_joint_arm()
    start_robot(store, second)
    assert robot.description().robot == "rebot_b601"
    assert robot.description().robot == "rebot_b601"
    assert len(decoded) == 2

    # The same robot app sends its description again.
    now = monotonic_now_ns()
    store.put_record(DESCRIPTION_TOPIC, records.encode_description(second, now), now)
    assert robot.description() == second
    assert robot.description() == second
    assert len(decoded) == 3


def test_a_description_that_cannot_be_decoded_reads_as_none_with_one_warning(
    store, robot, warnings
):
    description = rover()
    start_robot(store, description, send_description=False)
    store.put_record(DESCRIPTION_TOPIC, b"junk, not a description", monotonic_now_ns())
    put(store, description, "mast", [3.0])

    for _ in range(3):
        assert robot.description() is None
        assert robot.series() == {}
        assert robot.joints("mast") is None
    texts = warning_texts(warnings)
    assert len(texts) == 1
    assert "description cannot be read" in texts[0]

    now = monotonic_now_ns()
    store.put_record(
        DESCRIPTION_TOPIC, records.encode_description(description, now), now
    )
    assert robot.description() == description
    assert robot.joints("mast")["turn"].position == pytest.approx(3.0)


# ---------------------------------------------------------------------------------------------
# The reader object
# ---------------------------------------------------------------------------------------------


def test_closing_a_reader_leaves_the_server_s_store_alone(store):
    description = rover()
    start_robot(store, description)
    put(store, description, "mast", [3.0])
    with RobotState.from_store(store) as robot:
        assert robot.joints("mast") is not None
    robot.close()
    assert store.connected is True
    assert store.latest("mast").data is not None
    assert robot.refresh() is True
    assert robot.joints("mast")["turn"].position == pytest.approx(3.0)


def test_two_readers_of_one_store_see_the_same(store):
    description = rover()
    start_robot(store, description)
    put(store, description, "mast", [3.0])
    one = RobotState.from_store(store)
    other = RobotState.from_store(store)
    assert one.description() == other.description()
    assert one.joints("mast").positions == other.joints("mast").positions
    assert one.joints("mast").sequence == other.joints("mast").sequence == 1


def test_readings_compare_by_value_and_can_be_kept_in_a_set(store, robot):
    description = rover()
    start_robot(store, description)
    put(store, description, "mast", [3.0])
    put(store, description, "health", [True, 12.0, "idle"])
    put(store, description, "sonar", [0.5, "wall"])

    mast = robot.joints("mast")
    again = JointsReading(
        series=mast.series,
        sequence=mast.sequence,
        sample_time_ns=mast.sample_time_ns,
        age_s=mast.age_s,
        receive_age_s=mast.receive_age_s,
        fresh=mast.fresh,
        clock_two_way=mast.clock_two_way,
        joints=mast.joints,
    )
    assert again == mast
    assert hash(again) == hash(mast)
    assert again != robot.joints("mast", allow_stale=True)  # read later: older
    assert len({mast, robot.status("health"), robot.record("sonar")}) == 3
    assert "JointReading(name='turn'" in repr(mast)

    # A joint that was not read holds nan and still compares equal to itself.
    put(store, description, "mast", [0.0], valid=[False])
    unread = robot.joints("mast")
    assert math.isnan(unread["turn"].position)
    assert unread == JointsReading(
        series=unread.series,
        sequence=unread.sequence,
        sample_time_ns=unread.sample_time_ns,
        age_s=unread.age_s,
        receive_age_s=unread.receive_age_s,
        fresh=unread.fresh,
        clock_two_way=unread.clock_two_way,
        joints=unread.joints,
    )
    assert unread.joints == robot.joints("mast").joints
    assert hash(unread.joints) == hash(robot.joints("mast").joints)


def test_a_reader_is_read_from_several_threads_while_the_store_changes(store, robot):
    description = rover()
    start_robot(store, description)
    put(store, description, "mast", [0.0])
    failures = []
    stop = threading.Event()

    def read():
        try:
            last = 0
            while not stop.is_set():
                mast = robot.joints("mast")
                assert mast.sequence >= last
                assert mast["turn"].position == pytest.approx(mast.sequence - 1)
                last = mast.sequence
                assert robot.description().robot == "rover"
                assert robot.gripper("jaw") is None
        except BaseException as error:
            failures.append(error)

    threads = [threading.Thread(target=read) for _ in range(4)]
    for thread in threads:
        thread.start()
    for count in range(1, 300):
        put(store, description, "mast", [float(count)])
    stop.set()
    for thread in threads:
        thread.join(30.0)
    assert failures == []
    assert robot.joints("mast").sequence == 300


# ---------------------------------------------------------------------------------------------
# A platform without Unix sockets
# ---------------------------------------------------------------------------------------------


def test_without_unix_sockets_nothing_is_served_and_a_reader_sees_no_server(
    store, tmp_path, monkeypatch, warnings
):
    monkeypatch.setattr(_local, "_HAS_UNIX_SOCKETS", False)
    path = tmp_path / "rs.sock"
    start_robot(store, rover())

    service = RobotStateLocalService(store, path)
    with pytest.raises(OSError, match="no Unix domain sockets"):
        service.start()
    service.close()
    assert not path.exists()

    with RobotState.connect(path) as reader:
        assert reader.server_reachable is False
        assert reader.connected is False
        assert reader.description() is None
        assert reader.refresh() is False
    assert warning_texts(warnings) == [
        "Robot state cannot be read: this platform has no Unix domain sockets"
    ]
    # In the server's program the store is read without a socket.
    assert RobotState.from_store(store).description().robot == "rover"
