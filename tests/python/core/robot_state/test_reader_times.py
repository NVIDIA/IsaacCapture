# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The publish time and the receive time of a reading: `Reading.publish_time_ns` and
`Reading.receive_time_ns`."""

import pytest
from test_reader import SECOND, put, rover, start_robot

from isaaccapture.robot_state import (
    GripperReading,
    JointsReading,
    RobotState,
    RobotStateLocalService,
    RobotStateStore,
    _local,
    records,
)
from isaaccapture.robot_state._robot_state import monotonic_now_ns

# The robot machine's clock is this far ahead of the reader's in these tests.
ROBOT_AHEAD_NS = 5_000 * SECOND


@pytest.fixture
def store():
    return RobotStateStore()


@pytest.fixture
def robot(store):
    return RobotState.from_store(store)


def put_timed(
    store, description, series, values, *, sample_ago_ns, publish_ago_ns, **reading
):
    """One record taken and published that long ago, received at this moment.

    Returns (publish time on the robot's clock, receive time on this machine's).
    """
    received_ns = monotonic_now_ns()
    robot_now_ns = received_ns + ROBOT_AHEAD_NS
    sample_ns = robot_now_ns - sample_ago_ns
    store.put_record(
        series,
        records.encode_reading(
            description, series, values, time_ns=sample_ns, **reading
        ),
        sample_ns,
        publish_time_ns=robot_now_ns - publish_ago_ns,
        receive_time_ns=received_ns,
        robot_minus_server_ns=ROBOT_AHEAD_NS,
        clock_two_way=True,
    )
    return robot_now_ns - publish_ago_ns, received_ns


def test_a_reading_carries_the_publish_time_and_the_receive_time(store, robot):
    description = rover()
    start_robot(store, description)
    published_ns, received_ns = put_timed(
        store,
        description,
        "mast",
        [3.0],
        sample_ago_ns=40_000_000,
        publish_ago_ns=10_000_000,
    )

    mast = robot.joints("mast")
    assert mast.publish_time_ns == published_ns
    assert mast.receive_time_ns == received_ns
    # The publish call comes after the reading, on the robot machine's clock.
    assert mast.publish_time_ns - mast.sample_time_ns == 30_000_000
    # The arrival is on this machine's clock: the receive age counts from it.
    before_ns = monotonic_now_ns()
    again = robot.joints("mast")
    after_ns = monotonic_now_ns()
    receive_age_ns = round(again.receive_age_s * 1e9)
    assert before_ns <= again.receive_time_ns + receive_age_ns <= after_ns


def test_every_kind_of_reading_carries_the_two_times(store, robot):
    description = rover()
    start_robot(store, description)
    records_put = {
        "arm": ([0.25, 0.02, -0.5], {"velocities": [0.0, 0.0, 0.0]}),
        "health": ([True, 12.0, "idle"], {}),
        "sonar": ([0.5, "wall"], {}),
    }
    stamps = {
        series: put_timed(
            store,
            description,
            series,
            values,
            sample_ago_ns=20_000_000 + index,
            publish_ago_ns=5_000_000 + index,
            **reading,
        )
        for index, (series, (values, reading)) in enumerate(records_put.items())
    }

    readings = {
        "arm": robot.joints("arm"),
        "health": robot.status("health"),
        "sonar": robot.record("sonar"),
    }
    for series, reading in readings.items():
        assert (reading.publish_time_ns, reading.receive_time_ns) == stamps[series]
    # A gripper reading is made from its joints record and has that record's times.
    pinch = robot.gripper("pinch")
    assert isinstance(pinch, GripperReading)
    assert (pinch.publish_time_ns, pinch.receive_time_ns) == stamps["arm"]


def test_a_record_put_without_a_publish_time_has_its_sample_time(store, robot):
    description = rover()
    start_robot(store, description)
    before_ns = monotonic_now_ns()
    sample_ns = put(store, description, "mast", [3.0], age_s=2.0)
    after_ns = monotonic_now_ns()

    mast = robot.joints("mast")
    assert mast.publish_time_ns == sample_ns
    assert before_ns <= mast.receive_time_ns <= after_ns


def test_a_stale_reading_keeps_its_two_times(store, robot):
    description = rover()
    start_robot(store, description)
    published_ns, received_ns = put_timed(
        store,
        description,
        "quick",
        [1.0],
        sample_ago_ns=3 * SECOND,
        publish_ago_ns=3 * SECOND,
    )

    assert robot.joints("quick") is None
    quick = robot.joints("quick", allow_stale=True)
    assert quick.fresh is False
    assert (quick.publish_time_ns, quick.receive_time_ns) == (published_ns, received_ns)


def test_a_reading_built_without_the_two_times_equals_the_readers(store, robot):
    description = rover()
    start_robot(store, description)
    put_timed(
        store,
        description,
        "mast",
        [3.0],
        sample_ago_ns=40_000_000,
        publish_ago_ns=10_000_000,
    )

    mast = robot.joints("mast")
    by_hand = JointsReading(
        series=mast.series,
        sequence=mast.sequence,
        sample_time_ns=mast.sample_time_ns,
        age_s=mast.age_s,
        receive_age_s=mast.receive_age_s,
        fresh=mast.fresh,
        clock_two_way=mast.clock_two_way,
        joints=mast.joints,
    )
    assert (by_hand.publish_time_ns, by_hand.receive_time_ns) == (0, 0)
    assert by_hand == mast and hash(by_hand) == hash(mast)
    # The two times are given by name only.
    with pytest.raises(TypeError):
        JointsReading("mast", 1, 0, None, 0.0, True, True, 5, 6, mast.joints)
    timed = JointsReading(
        "mast", 1, 0, None, 0.0, True, True, mast.joints, publish_time_ns=5
    )
    assert (timed.publish_time_ns, timed.receive_time_ns) == (5, 0)


@pytest.mark.skipif(
    not _local._HAS_UNIX_SOCKETS, reason="the reader socket is a Unix domain socket"
)
def test_a_reader_of_another_program_gets_the_servers_two_times(
    store, tmp_path_factory
):
    description = rover()
    start_robot(store, description)
    published_ns, received_ns = put_timed(
        store,
        description,
        "mast",
        [3.0],
        sample_ago_ns=40_000_000,
        publish_ago_ns=10_000_000,
    )
    # Shorter than `tmp_path`, which holds the test's name: a Unix socket path is limited.
    path = str(tmp_path_factory.mktemp("rs") / "rs.sock")
    service = RobotStateLocalService(store, path)
    service.start()
    try:
        with RobotState.connect(path, timeout_s=5.0) as reader:
            mast = reader.joints("mast")
            in_server = RobotState.from_store(store).joints("mast")
    finally:
        service.close()

    assert mast is not None
    assert (mast.publish_time_ns, mast.receive_time_ns) == (published_ns, received_ns)
    assert (in_server.publish_time_ns, in_server.receive_time_ns) == (
        published_ns,
        received_ns,
    )
