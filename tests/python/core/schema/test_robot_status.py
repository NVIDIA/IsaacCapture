# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the RobotStatus types in isaaccapture.schema.

- RobotStatusReading: one named reading, a number, a flag or a text
- RobotFault: one fault with code, text, source and severity
- RobotStatus: readings and faults
- RobotStatusRecord: record wrapper carrying DeviceDataTimestamp
"""

import pytest

from isaaccapture.schema import (
    DeviceDataTimestamp,
    JointStateOutputRecord,
    RobotFault,
    RobotFaultSeverity,
    RobotReadingKind,
    RobotStatus,
    RobotStatusReading,
    RobotStatusRecord,
)


def _status() -> RobotStatus:
    return RobotStatus(
        readings=[
            RobotStatusReading(name="motors_on", kind=RobotReadingKind.FLAG, flag=True),
            RobotStatusReading(
                name="battery", kind=RobotReadingKind.NUMBER, number=24.5
            ),
            RobotStatusReading(name="mode", kind=RobotReadingKind.TEXT, text="hold"),
            RobotStatusReading(name="temperature", valid=False),
        ],
        faults=[
            RobotFault(
                code="overtemperature",
                text="joint3 is too hot",
                source="joint3",
                severity=RobotFaultSeverity.FATAL,
            )
        ],
    )


class TestRobotStatusReading:
    def test_defaults(self):
        reading = RobotStatusReading()

        assert reading.name == ""
        assert reading.kind == RobotReadingKind.NUMBER
        assert reading.number == 0.0
        assert reading.flag is False
        assert reading.text == ""
        assert reading.valid is True

    def test_every_field_reads_back(self):
        reading = RobotStatusReading(
            name="battery",
            kind=RobotReadingKind.TEXT,
            number=1.5,
            flag=True,
            text="low",
            valid=False,
        )

        assert reading.name == "battery"
        assert reading.kind == RobotReadingKind.TEXT
        assert reading.number == 1.5
        assert reading.flag is True
        assert reading.text == "low"
        assert reading.valid is False

    def test_is_read_only(self):
        reading = RobotStatusReading(name="battery")

        with pytest.raises(AttributeError):
            reading.name = "other"

    def test_repr(self):
        assert "battery" in repr(RobotStatusReading(name="battery"))


class TestRobotFault:
    def test_defaults(self):
        fault = RobotFault()

        assert fault.code == ""
        assert fault.text == ""
        assert fault.source == ""
        assert fault.severity == RobotFaultSeverity.ERROR

    def test_every_field_reads_back(self):
        fault = RobotFault(
            "stall", "motor stalled", "joint2", RobotFaultSeverity.WARNING
        )

        assert fault.code == "stall"
        assert fault.text == "motor stalled"
        assert fault.source == "joint2"
        assert fault.severity == RobotFaultSeverity.WARNING
        assert "stall" in repr(fault)

    def test_severity_values(self):
        assert int(RobotFaultSeverity.WARNING) == 0
        assert int(RobotFaultSeverity.ERROR) == 1
        assert int(RobotFaultSeverity.FATAL) == 2
        assert int(RobotReadingKind.NUMBER) == 0
        assert int(RobotReadingKind.FLAG) == 1
        assert int(RobotReadingKind.TEXT) == 2


class TestRobotStatus:
    def test_empty(self):
        status = RobotStatus()

        assert status.readings == []
        assert status.faults == []

    def test_readings_keep_their_order(self):
        status = _status()

        assert [reading.name for reading in status.readings] == [
            "motors_on",
            "battery",
            "mode",
            "temperature",
        ]
        assert status.readings[0].flag is True
        assert status.readings[1].number == 24.5
        assert status.readings[2].text == "hold"
        assert status.readings[3].valid is False

    def test_faults_read_back(self):
        fault = _status().faults[0]

        assert fault.code == "overtemperature"
        assert fault.text == "joint3 is too hot"
        assert fault.source == "joint3"
        assert fault.severity == RobotFaultSeverity.FATAL

    def test_repr_counts(self):
        assert repr(_status()) == "RobotStatus(readings=4, faults=1)"


class TestRobotStatusRecord:
    def test_record_carries_status_and_timestamp(self):
        record = RobotStatusRecord(_status(), DeviceDataTimestamp(30, 20, 0))

        assert len(record.data.readings) == 4
        assert record.timestamp.available_time_local_common_clock == 30
        assert record.timestamp.sample_time_local_common_clock == 20
        assert record.timestamp.sample_time_raw_device_clock == 0

    def test_bytes_round_trip(self):
        data = RobotStatusRecord(_status(), DeviceDataTimestamp(30, 20, 0)).to_bytes()
        record = RobotStatusRecord.from_bytes(data)

        assert record.to_bytes() == data
        assert record.data.readings[1].name == "battery"
        assert record.data.readings[1].number == 24.5
        assert record.data.faults[0].code == "overtemperature"
        assert record.timestamp.sample_time_local_common_clock == 20

    def test_nested_table_round_trips_on_its_own(self):
        record = RobotStatusRecord(_status(), DeviceDataTimestamp(30, 20, 0))
        status = RobotStatus.from_bytes(record.data.to_bytes())

        assert [reading.name for reading in status.readings][:2] == [
            "motors_on",
            "battery",
        ]

    @pytest.mark.parametrize(
        "junk", [b"", b"\x00", b"not a flatbuffer", b"\xff" * 64, b"\x00" * 64]
    )
    def test_from_bytes_refuses_junk(self, junk):
        with pytest.raises(ValueError, match="not a valid RobotStatusRecord buffer"):
            RobotStatusRecord.from_bytes(junk)

    def test_from_bytes_refuses_a_truncated_record(self):
        data = RobotStatusRecord(_status(), DeviceDataTimestamp(30, 20, 0)).to_bytes()

        with pytest.raises(ValueError, match="not a valid RobotStatusRecord buffer"):
            RobotStatusRecord.from_bytes(data[: len(data) // 2])

    def test_from_bytes_takes_bytes_only(self):
        with pytest.raises(TypeError):
            RobotStatusRecord.from_bytes("text")

    def test_from_bytes_names_the_class_it_was_called_on(self):
        with pytest.raises(
            ValueError, match="not a valid JointStateOutputRecord buffer"
        ):
            JointStateOutputRecord.from_bytes(b"junk")
