# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bindings of the robot state store, fed with payloads laid out by hand as the link carries them."""

import struct
import threading
import time

import pytest

from isaaccapture.robot_state import _robot_state
from isaaccapture.robot_state._robot_state import (
    DESCRIPTION_TOPIC,
    RobotRecord,
    RobotSeriesInfo,
    RobotStateStore,
    monotonic_now_ns,
)

MS = 1_000_000
SECOND = 1_000 * MS
CLOCK_VALID = 1
CLOCK_TWO_WAY = 2

DESCRIPTION, ARM, STATUS = 1, 2, 3


def _text(value: str | bytes) -> bytes:
    data = value.encode() if isinstance(value, str) else value
    return struct.pack("<I", len(data)) + data


def _record(opcode: int, body: bytes) -> bytes:
    """An MCAP record: opcode, body length, body."""
    return struct.pack("<BQ", opcode, len(body)) + body


def schema_record(schema_id: int, name: str, encoding: str, data: bytes) -> bytes:
    return _record(
        0x03, struct.pack("<H", schema_id) + _text(name) + _text(encoding) + _text(data)
    )


def channel_record(
    channel_id: int, schema_id: int, topic: str, encoding: str, metadata: dict[str, str]
) -> bytes:
    pairs = b"".join(_text(key) + _text(metadata[key]) for key in sorted(metadata))
    body = struct.pack("<HH", channel_id, schema_id) + _text(topic) + _text(encoding)
    return _record(0x04, body + _text(pairs))


def message_record(
    channel_id: int, sample_time_ns: int, data: bytes, sequence: int = 1
) -> bytes:
    header = struct.pack(
        "<HIQQ", channel_id, sequence, sample_time_ns, sample_time_ns + 1
    )
    return _record(0x05, header + data)


def samples_payload(
    records: bytes,
    *,
    publish_seq: int = 1,
    robot_minus_server_ns: int = 0,
    flags: int = CLOCK_VALID | CLOCK_TWO_WAY,
) -> bytes:
    """The 32-byte prefix of ROBOT_STATE_SAMPLES and ROBOT_STATE_RECORDS, then the records."""
    return (
        struct.pack("<QqqII", publish_seq, 0, robot_minus_server_ns, flags, 0) + records
    )


def robot_setup() -> bytes:
    """A description, an arm that goes stale after 100 ms and a status after 2 s."""
    return b"".join(
        [
            schema_record(
                DESCRIPTION, "core.RobotDescriptionRecord", "flatbuffer", b"d"
            ),
            channel_record(
                DESCRIPTION,
                DESCRIPTION,
                DESCRIPTION_TOPIC,
                "flatbuffer",
                {"shape": "description", "delivery": "in_order", "stale_after_ns": "0"},
            ),
            schema_record(
                ARM, "core.JointStateOutputRecord", "flatbuffer", b"\x00j\xff"
            ),
            channel_record(
                ARM,
                ARM,
                "arm",
                "flatbuffer",
                {
                    "shape": "joints",
                    "delivery": "latest",
                    "stale_after_ns": str(100 * MS),
                    "rate_hz": "20",
                },
            ),
            schema_record(STATUS, "core.RobotStatusRecord", "flatbuffer", b"s"),
            channel_record(
                STATUS,
                STATUS,
                "status",
                "flatbuffer",
                {"shape": "status", "stale_after_ns": str(2 * SECOND)},
            ),
        ]
    )


@pytest.fixture
def store() -> RobotStateStore:
    """A store in a robot session with the three series declared."""
    filled = RobotStateStore()
    filled.begin_session()
    filled.add_setup(robot_setup())
    return filled


def test_module_names_and_clock():
    assert DESCRIPTION_TOPIC == "robot_description"
    assert sorted(name for name in dir(_robot_state) if not name.startswith("_")) == [
        "DESCRIPTION_TOPIC",
        "RobotRecord",
        "RobotSeriesInfo",
        "RobotStateStore",
        "monotonic_now_ns",
    ]
    # The clock of ages is the one time.monotonic_ns() reads.
    before = time.monotonic_ns()
    now = monotonic_now_ns()
    assert before <= now <= time.monotonic_ns()


def test_the_package_hands_out_the_same_classes():
    import isaaccapture.robot_state as robot_state

    assert robot_state.RobotStateStore is RobotStateStore
    assert robot_state.RobotSeriesInfo is RobotSeriesInfo
    assert robot_state.RobotRecord is RobotRecord


def test_a_new_store_is_empty():
    empty = RobotStateStore()
    assert empty.connected is False
    assert empty.session_count == 0
    assert empty.generation == 0
    assert empty.setup_generation == 1
    assert empty.dropped_records == 0
    assert empty.series() == []
    assert empty.series_info("arm") is None

    none = empty.latest("arm")
    assert none.data is None
    assert none.fresh is False
    assert none.age_ns is None
    assert none.sequence == 0


def test_properties_are_read_only():
    empty = RobotStateStore()
    for name in (
        "connected",
        "session_count",
        "generation",
        "setup_generation",
        "dropped_records",
    ):
        with pytest.raises(AttributeError):
            setattr(empty, name, 1)
    with pytest.raises(TypeError):
        RobotSeriesInfo()
    with pytest.raises(TypeError):
        RobotRecord()


def test_setup_declares_series_in_order_with_their_metadata(store):
    assert [info.name for info in store.series()] == [
        DESCRIPTION_TOPIC,
        "arm",
        "status",
    ]

    arm = store.series_info("arm")
    assert isinstance(arm, RobotSeriesInfo)
    assert arm.name == "arm"
    assert arm.shape == "joints"
    assert arm.schema_name == "core.JointStateOutputRecord"
    assert arm.schema_encoding == "flatbuffer"
    assert arm.schema_data == b"\x00j\xff"
    assert isinstance(arm.schema_data, bytes)
    assert arm.message_encoding == "flatbuffer"
    assert arm.stale_after_ns == 100 * MS
    assert arm.in_order is False
    assert arm.rate_hz == 20.0
    assert repr(arm) == "RobotSeriesInfo(name='arm', shape='joints')"
    with pytest.raises(AttributeError):
        arm.name = "leg"

    description = store.series_info(DESCRIPTION_TOPIC)
    assert description.shape == "description"
    assert description.in_order is True
    assert description.stale_after_ns == 0
    assert description.rate_hz == 0.0


def test_samples_give_the_newest_record_with_its_times(store):
    robot_ahead = 1_000 * SECOND
    sample_here = 50 * SECOND
    store.add_samples(
        samples_payload(
            message_record(ARM, sample_here + robot_ahead, b"\x00angles\xff"),
            robot_minus_server_ns=robot_ahead,
        ),
        receive_time_ns=sample_here + 3 * MS,
    )

    record = store.latest("arm", sample_here + 30 * MS)
    assert isinstance(record, RobotRecord)
    assert record.data == b"\x00angles\xff"
    assert isinstance(record.data, bytes)
    assert record.sequence == 1
    assert record.sample_time_ns == sample_here + robot_ahead
    assert record.publish_time_ns == sample_here + robot_ahead + 1
    assert record.receive_time_ns == sample_here + 3 * MS
    assert record.age_ns == 30 * MS
    assert record.receive_age_ns == 27 * MS
    assert record.clock_two_way is True
    assert record.fresh is True
    assert repr(record) == "RobotRecord(sequence=1, bytes=8, fresh=True)"
    with pytest.raises(AttributeError):
        record.fresh = False

    # Past the series' own limit the record is still handed out, as not fresh.
    stale = store.latest("arm", now_ns=sample_here + 101 * MS)
    assert stale.data == b"\x00angles\xff"
    assert stale.fresh is False
    assert stale.age_ns == 101 * MS


def test_each_series_has_its_own_limit_and_the_description_none(store):
    store.add_samples(
        samples_payload(message_record(DESCRIPTION, 10 * SECOND, b"what")),
        10 * SECOND,
    )
    store.add_samples(
        samples_payload(
            message_record(ARM, 10 * SECOND, b"angles")
            + message_record(STATUS, 10 * SECOND, b"ok")
        ),
        10 * SECOND,
    )
    later = 11 * SECOND
    assert store.latest("arm", later).fresh is False
    assert store.latest("status", later).fresh is True
    assert store.latest("status", 13 * SECOND).fresh is False
    assert store.latest(DESCRIPTION_TOPIC, 10_000 * SECOND).fresh is True
    assert store.latest(DESCRIPTION_TOPIC, 10_000 * SECOND).data == b"what"


def test_without_a_clock_estimate_there_is_no_age(store):
    store.add_samples(
        samples_payload(message_record(ARM, 7, b"angles"), flags=0), 50 * SECOND
    )
    record = store.latest("arm", 50 * SECOND + 100 * MS)
    assert record.age_ns is None
    assert record.clock_two_way is False
    assert record.receive_age_ns == 100 * MS
    assert record.fresh is True
    assert store.latest("arm", 50 * SECOND + 101 * MS).fresh is False

    # A one-way estimate gives an age and says so.
    store.add_samples(
        samples_payload(
            message_record(ARM, 60 * SECOND, b"angles"),
            publish_seq=2,
            flags=CLOCK_VALID,
        ),
        60 * SECOND,
    )
    record = store.latest("arm", 60 * SECOND + 5 * MS)
    assert record.age_ns == 5 * MS
    assert record.clock_two_way is False


@pytest.mark.parametrize("through", ["link message payload", "put_record"])
def test_a_time_stamp_ahead_of_the_clock_does_not_keep_a_record_fresh(store, through):
    # A robot app that stamps a reading with time.time_ns() instead of time.monotonic_ns().
    wall_clock = 1_700_000_000 * SECOND
    if through == "put_record":
        store.put_record(
            "arm",
            b"angles",
            wall_clock,
            receive_time_ns=50 * SECOND,
            robot_minus_server_ns=0,
            clock_two_way=True,
        )
    else:
        store.add_samples(
            samples_payload(message_record(ARM, wall_clock, b"angles")), 50 * SECOND
        )

    early = store.latest("arm", 50 * SECOND + 30 * MS)
    assert early.age_ns == 30 * MS
    assert early.receive_age_ns == 30 * MS
    assert early.fresh is True

    # The age is never less than the time since the record arrived.
    late = store.latest("arm", 55 * SECOND)
    assert late.data == b"angles"
    assert late.age_ns == 5 * SECOND
    assert late.receive_age_ns == 5 * SECOND
    assert late.fresh is False


def test_receive_time_and_now_default_to_this_machines_clock(store):
    before = monotonic_now_ns()
    store.add_samples(samples_payload(message_record(ARM, before, b"angles")))
    record = store.latest("arm")
    after = monotonic_now_ns()
    assert before <= record.receive_time_ns <= after
    assert 0 <= record.age_ns <= after - before
    assert 0 <= record.receive_age_ns <= after - before
    assert record.fresh is True


def test_sessions(store):
    store.add_samples(samples_payload(message_record(DESCRIPTION, 1, b"an arm")), 1)
    assert store.connected is True
    assert store.session_count == 1
    setup_generation = store.setup_generation
    assert setup_generation > 2

    store.end_session()
    assert store.connected is False
    assert len(store.series()) == 3
    assert store.latest(DESCRIPTION_TOPIC, 1).data == b"an arm"
    generation = store.generation
    store.end_session()
    assert store.generation == generation

    store.begin_session()
    assert store.connected is True
    assert store.session_count == 2
    assert store.setup_generation > setup_generation
    assert store.series() == []
    assert store.latest(DESCRIPTION_TOPIC, 1).data is None


def test_undeclared_channels_and_older_payloads_are_dropped_and_counted(store):
    store.add_samples(
        samples_payload(
            message_record(99, 5, b"nobody's") + message_record(ARM, 5, b"fifth"),
            publish_seq=5,
        ),
        5,
    )
    assert store.dropped_records == 1
    store.add_samples(
        samples_payload(message_record(ARM, 4, b"fourth"), publish_seq=4), 6
    )
    assert store.dropped_records == 2
    assert store.latest("arm", 6).data == b"fifth"
    assert store.latest("arm", 6).sequence == 1


@pytest.mark.parametrize(
    ("call", "payload"),
    [
        ("add_setup", robot_setup()[:-1]),
        ("add_setup", message_record(ARM, 5, b"angles")),
        ("add_setup", channel_record(9, 77, "lost", "flatbuffer", {})),
        ("add_samples", b"\x00" * 31),
        ("add_samples", samples_payload(message_record(ARM, 5, b"angles")[:-1])),
        ("add_samples", samples_payload(schema_record(9, "s", "flatbuffer", b""))),
        ("import_snapshot", b""),
        ("import_snapshot", b"not a snapshot at all"),
        ("import_snapshot", robot_setup()),
    ],
    ids=[
        "setup cut short",
        "setup with a Message",
        "channel without its schema",
        "samples shorter than the prefix",
        "samples cut short",
        "samples with a Schema",
        "empty snapshot",
        "text as a snapshot",
        "setup as a snapshot",
    ],
)
def test_malformed_bytes_raise_and_change_nothing(store, call, payload):
    store.add_samples(samples_payload(message_record(ARM, 5, b"angles")), 5)
    generation = store.generation

    with pytest.raises(RuntimeError, match="RobotStateStore"):
        getattr(store, call)(payload)

    assert store.generation == generation
    assert store.connected is True
    assert [info.name for info in store.series()] == [
        DESCRIPTION_TOPIC,
        "arm",
        "status",
    ]
    assert store.latest("arm", 5).data == b"angles"
    assert store.dropped_records == 0


def test_byte_arguments_must_be_bytes(store):
    with pytest.raises(TypeError):
        store.add_setup("text")
    with pytest.raises(TypeError):
        store.add_samples(None)
    with pytest.raises(TypeError):
        store.import_snapshot(7)
    with pytest.raises(TypeError):
        store.put_record("arm", "text", 0)


def test_declare_series_and_put_record():
    direct = RobotStateStore()
    direct.begin_session()
    direct.declare_series("plain", "", "", "", b"", "json")
    direct.declare_series(
        name="arm",
        shape="joints",
        schema_name="core.JointStateOutputRecord",
        schema_encoding="flatbuffer",
        schema_data=b"\x01\x02",
        message_encoding="flatbuffer",
        stale_after_ns=100 * MS,
        in_order=False,
        rate_hz=20.5,
    )
    assert [info.name for info in direct.series()] == ["plain", "arm"]
    plain = direct.series_info("plain")
    assert (plain.stale_after_ns, plain.in_order, plain.rate_hz) == (0, False, 0.0)
    arm = direct.series_info("arm")
    assert (arm.schema_data, arm.stale_after_ns, arm.rate_hz) == (
        b"\x01\x02",
        100 * MS,
        20.5,
    )

    # Every time given: the robot's clock is 990 s ahead.
    direct.put_record(
        "arm",
        b"angles",
        1_000 * SECOND,
        publish_time_ns=1_000 * SECOND + 2,
        receive_time_ns=10 * SECOND + 1,
        robot_minus_server_ns=990 * SECOND,
        clock_two_way=True,
    )
    record = direct.latest("arm", 10 * SECOND + 20 * MS)
    assert record.data == b"angles"
    assert record.sequence == 1
    assert record.publish_time_ns == 1_000 * SECOND + 2
    assert record.receive_time_ns == 10 * SECOND + 1
    assert record.age_ns == 20 * MS
    assert record.clock_two_way is True
    assert record.fresh is True

    # Defaults: published when sampled, received now, no clock estimate.
    before = monotonic_now_ns()
    direct.put_record("arm", b"angles 2", 123)
    record = direct.latest("arm")
    assert record.sequence == 2
    assert record.sample_time_ns == 123
    assert record.publish_time_ns == 123
    assert before <= record.receive_time_ns <= monotonic_now_ns()
    assert record.age_ns is None
    assert record.clock_two_way is False
    assert record.fresh is True

    # A one-way estimate.
    direct.put_record(
        "arm", b"angles 3", 50, robot_minus_server_ns=0, receive_time_ns=50
    )
    record = direct.latest("arm", 60)
    assert record.age_ns == 10
    assert record.clock_two_way is False

    with pytest.raises(ValueError, match="'leg' is not declared"):
        direct.put_record("leg", b"", 0)
    with pytest.raises(ValueError, match="needs a name"):
        direct.declare_series("", "", "", "", b"", "")
    with pytest.raises(ValueError, match="zero or above"):
        direct.declare_series("x", "", "", "", b"", "", stale_after_ns=-1)
    with pytest.raises(TypeError):
        direct.put_record("arm", b"", 0, 5)
    assert direct.latest("arm").sequence == 3


def test_snapshot_round_trip(store):
    robot_ahead = 1_000 * SECOND
    store.add_samples(
        samples_payload(
            message_record(DESCRIPTION, robot_ahead + 5, b"what"),
            robot_minus_server_ns=robot_ahead,
        ),
        6,
    )
    store.add_samples(
        samples_payload(
            message_record(ARM, robot_ahead + 7, b"\x00angles"),
            robot_minus_server_ns=robot_ahead,
            flags=CLOCK_VALID,
        ),
        8,
    )

    snapshot = store.export_snapshot()
    assert isinstance(snapshot, bytes)
    reader = RobotStateStore()
    reader.import_snapshot(snapshot)

    for name in ("connected", "session_count", "generation", "setup_generation"):
        assert getattr(reader, name) == getattr(store, name), name
    assert [info.name for info in reader.series()] == [
        DESCRIPTION_TOPIC,
        "arm",
        "status",
    ]
    for name in (DESCRIPTION_TOPIC, "arm", "status"):
        ours, theirs = reader.series_info(name), store.series_info(name)
        for field in (
            "shape",
            "schema_name",
            "schema_encoding",
            "schema_data",
            "message_encoding",
            "stale_after_ns",
            "in_order",
            "rate_hz",
        ):
            assert getattr(ours, field) == getattr(theirs, field), (name, field)
        ours, theirs = reader.latest(name, 20), store.latest(name, 20)
        for field in (
            "data",
            "sequence",
            "sample_time_ns",
            "publish_time_ns",
            "receive_time_ns",
            "age_ns",
            "receive_age_ns",
            "clock_two_way",
            "fresh",
        ):
            assert getattr(ours, field) == getattr(theirs, field), (name, field)
    assert reader.latest("arm", 20).age_ns == 13
    assert reader.latest("status", 20).data is None


def test_snapshot_without_declarations(store):
    store.add_samples(samples_payload(message_record(ARM, 5, b"angles")), 5)
    full = store.export_snapshot()
    brief = store.export_snapshot(store.setup_generation)
    assert len(brief) < len(full)
    assert store.export_snapshot(0) == full
    assert store.export_snapshot(known_setup_generation=store.setup_generation) == brief

    # A store that does not have these declarations refuses it; one that has them takes it.
    reader = RobotStateStore()
    with pytest.raises(RuntimeError, match="setup generation"):
        reader.import_snapshot(brief)
    assert reader.series() == []
    reader.import_snapshot(full)
    store.add_samples(
        samples_payload(message_record(STATUS, 6, b"ok"), publish_seq=2), 6
    )
    reader.import_snapshot(store.export_snapshot(reader.setup_generation))
    assert reader.latest("status", 6).data == b"ok"
    assert reader.series_info("arm").schema_name == "core.JointStateOutputRecord"
    assert reader.generation == store.generation


def test_many_threads_read_while_one_writes(store):
    """Every arm record holds its own number as text and as its sample time, so a torn read shows."""
    records = 2_000
    failures: list[str] = []
    done = threading.Event()

    def read(use_known_setup: bool) -> None:
        copy = RobotStateStore()
        last = 0
        while not done.is_set():
            arm = store.latest("arm", 0)
            if arm.data is not None:
                if arm.data != str(arm.sample_time_ns).encode() or arm.sequence < last:
                    failures.append(f"torn read: {arm!r}")
                last = arm.sequence
            if len(store.series()) != 3 or not store.connected:
                failures.append("series changed")
            copy.import_snapshot(
                store.export_snapshot(copy.setup_generation if use_known_setup else 0)
            )
            copied = copy.latest("arm", 0)
            if (
                copied.data is not None
                and copied.data != str(copied.sample_time_ns).encode()
            ):
                failures.append(f"torn snapshot: {copied!r}")

    readers = [
        threading.Thread(target=read, args=(index % 2 == 0,)) for index in range(4)
    ]
    for reader in readers:
        reader.start()
    try:
        for number in range(1, records + 1):
            store.add_samples(
                samples_payload(
                    message_record(ARM, number, str(number).encode()),
                    publish_seq=number,
                ),
                number,
            )
    finally:
        done.set()
        for reader in readers:
            reader.join(timeout=30)

    assert failures == []
    assert not any(reader.is_alive() for reader in readers)
    newest = store.latest("arm", 0)
    assert newest.data == str(records).encode()
    assert newest.sequence == records
    assert store.dropped_records == 0
