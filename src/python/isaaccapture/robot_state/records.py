# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Robot-state records: numbers of a reading into record bytes, and record bytes back.

Joints travel as `JointStateOutputRecord`, status as `RobotStatusRecord`, the description as
`RobotDescriptionRecord`, and a shape the robot defines as one JSON object per record.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

from isaaccapture import schema

from .description import (
    DEFAULT_STALE_AFTER_S,
    DESCRIPTION_TOPIC,
    GripperDescription,
    JointDescription,
    RobotDescription,
    SeriesDescription,
    ValueDescription,
    default_joint_unit,
)

__all__ = [
    "DEFAULT_STALE_AFTER_S",
    "DESCRIPTION_TOPIC",
    "Fault",
    "JointValue",
    "SeriesDeclaration",
    "declarations",
    "decode_custom",
    "decode_description",
    "decode_joints",
    "decode_status",
    "encode_description",
    "encode_reading",
]

_FLOAT32_MAX = 3.4028234663852886e38

_JOINT_KINDS = {
    "revolute": schema.RobotJointKind.REVOLUTE,
    "prismatic": schema.RobotJointKind.PRISMATIC,
    "continuous": schema.RobotJointKind.CONTINUOUS,
    "gripper": schema.RobotJointKind.GRIPPER,
}
_VALUE_KINDS = {
    "any": schema.RobotValueKind.ANY,
    "number": schema.RobotValueKind.NUMBER,
    "flag": schema.RobotValueKind.FLAG,
    "text": schema.RobotValueKind.TEXT,
    "numbers": schema.RobotValueKind.NUMBERS,
}
_SERIES_SHAPES = {
    "joints": schema.RobotSeriesShape.JOINTS,
    "status": schema.RobotSeriesShape.STATUS,
    "custom": schema.RobotSeriesShape.CUSTOM,
}
_SEVERITIES = {
    "warning": schema.RobotFaultSeverity.WARNING,
    "error": schema.RobotFaultSeverity.ERROR,
    "fatal": schema.RobotFaultSeverity.FATAL,
}
_JOINT_KIND_NAMES = {value: name for name, value in _JOINT_KINDS.items()}
_VALUE_KIND_NAMES = {value: name for name, value in _VALUE_KINDS.items()}
_SERIES_SHAPE_NAMES = {value: name for name, value in _SERIES_SHAPES.items()}
_SEVERITY_NAMES = {value: name for name, value in _SEVERITIES.items()}

_JSON_TYPES = {
    "number": {"type": "number"},
    "flag": {"type": "boolean"},
    "text": {"type": "string"},
    "numbers": {"type": "array", "items": {"type": "number"}},
    "any": {},
}


@dataclass(frozen=True)
class SeriesDeclaration:
    """What the two ends of a remote teleop session declare for one data series."""

    name: str
    shape: str
    schema_name: str
    schema_encoding: str
    schema_data: bytes
    message_encoding: str
    stale_after_ns: int
    in_order: bool
    rate_hz: float


class JointValue(NamedTuple):
    """One joint of a joints record. `velocity` and `effort` are None when the series has none."""

    name: str
    position: float
    velocity: float | None
    effort: float | None
    valid: bool


class Fault(NamedTuple):
    """One fault a robot reports. `severity` is "warning", "error" or "fatal"."""

    code: str
    text: str
    source: str = ""
    severity: str = "error"


def declarations(description: RobotDescription) -> list[SeriesDeclaration]:
    """The description's own series first, then the robot's series in file order."""
    name, data = schema.robot_state_record_schema("description")
    declared = [
        SeriesDeclaration(
            name=DESCRIPTION_TOPIC,
            shape="description",
            schema_name=name,
            schema_encoding="flatbuffer",
            schema_data=data,
            message_encoding="flatbuffer",
            stale_after_ns=0,
            in_order=True,
            rate_hz=0.0,
        )
    ]
    for series in description.series.values():
        if series.shape == "custom":
            schema_name = f"{description.robot}.{series.shape_name}"
            encoding = ("jsonschema", "json")
            data = _json_schema(schema_name, series.values)
        else:
            schema_name, data = schema.robot_state_record_schema(series.shape)
            encoding = ("flatbuffer", "flatbuffer")
        declared.append(
            SeriesDeclaration(
                name=series.name,
                shape=series.shape_name,
                schema_name=schema_name,
                schema_encoding=encoding[0],
                schema_data=data,
                message_encoding=encoding[1],
                stale_after_ns=round(series.stale_after_s * 1e9),
                in_order=False,
                rate_hz=series.rate_hz or 0.0,
            )
        )
    return declared


def _json_schema(title: str, fields: Iterable[ValueDescription]) -> bytes:
    properties = {}
    for field in fields:
        properties[field.name] = dict(_JSON_TYPES[field.kind])
        if field.unit:
            properties[field.name]["unit"] = field.unit
    document = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": title,
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    return json.dumps(document, separators=(",", ":")).encode("utf-8")


def _text_pairs(attributes: Mapping[str, str]) -> list:
    return [schema.RobotTextPair(key, value) for key, value in attributes.items()]


def encode_description(description: RobotDescription, time_ns: int) -> bytes:
    """The description as `RobotDescriptionRecord` bytes, stamped `time_ns`."""
    series = [
        schema.RobotSeriesDescription(
            name=entry.name,
            shape=_SERIES_SHAPES[entry.shape],
            shape_name=entry.shape_name,
            rate_hz=entry.rate_hz or 0.0,
            stale_after_s=entry.stale_after_s,
            joints=[
                schema.RobotJointDescription(
                    name=joint.name,
                    kind=_JOINT_KINDS[joint.kind],
                    has_limits=joint.min is not None and joint.max is not None,
                    min_position=joint.min or 0.0,
                    max_position=joint.max or 0.0,
                    max_velocity=joint.max_velocity or 0.0,
                    max_effort=joint.max_effort or 0.0,
                    unit=joint.unit,
                )
                for joint in entry.joints
            ],
            has_velocity=entry.has_velocity,
            has_effort=entry.has_effort,
            values=[
                schema.RobotValueDescription(
                    value.name, _VALUE_KINDS[value.kind], value.unit
                )
                for value in entry.values
            ],
            attributes=_text_pairs(entry.attributes),
        )
        for entry in description.series.values()
    ]
    grippers = [
        schema.RobotGripperDescription(
            name=gripper.name,
            series=gripper.series,
            joints=list(gripper.joints),
            open_position=gripper.open,
            closed_position=gripper.closed,
        )
        for gripper in description.grippers.values()
    ]
    data = schema.RobotDescription(
        robot=description.robot,
        series=series,
        grippers=grippers,
        attributes=_text_pairs(description.attributes),
    )
    timestamp = schema.DeviceDataTimestamp(time_ns, time_ns, 0)
    return schema.RobotDescriptionRecord(data, timestamp).to_bytes()


def decode_description(data: bytes) -> RobotDescription:
    """The description of `RobotDescriptionRecord` bytes; ValueError on a bad buffer."""
    encoded = schema.RobotDescriptionRecord.from_bytes(data).data
    if encoded is None:
        raise ValueError("a RobotDescriptionRecord without a description")
    try:
        return _description_of(encoded)
    except KeyError as error:
        raise ValueError(
            f"a RobotDescriptionRecord with a kind this build does not know: {error}"
        ) from None


def _description_of(encoded) -> RobotDescription:
    series = {}
    for entry in encoded.series:
        joints = []
        for joint in entry.joints:
            kind = _JOINT_KIND_NAMES[joint.kind]
            joints.append(
                JointDescription(
                    name=joint.name,
                    kind=kind,
                    min=joint.min_position if joint.has_limits else None,
                    max=joint.max_position if joint.has_limits else None,
                    max_velocity=joint.max_velocity or None,
                    max_effort=joint.max_effort or None,
                    unit=joint.unit or default_joint_unit(kind),
                )
            )
        series[entry.name] = SeriesDescription(
            name=entry.name,
            shape=_SERIES_SHAPE_NAMES[entry.shape],
            shape_name=entry.shape_name,
            joints=tuple(joints),
            values=tuple(
                ValueDescription(value.name, _VALUE_KIND_NAMES[value.kind], value.unit)
                for value in entry.values
            ),
            has_velocity=entry.has_velocity,
            has_effort=entry.has_effort,
            rate_hz=entry.rate_hz or None,
            stale_after_s=entry.stale_after_s,
            attributes={pair.key: pair.value for pair in entry.attributes},
        )
    # Grippers come from the record's list only; none is derived from a joint's kind here.
    grippers = {
        gripper.name: GripperDescription(
            name=gripper.name,
            series=gripper.series,
            joints=tuple(gripper.joints),
            open=gripper.open_position,
            closed=gripper.closed_position,
        )
        for gripper in encoded.grippers
    }
    return RobotDescription(
        robot=encoded.robot,
        series=series,
        grippers=grippers,
        attributes={pair.key: pair.value for pair in encoded.attributes},
    )


def encode_reading(
    description: RobotDescription,
    series: str,
    values,
    *,
    velocities=None,
    efforts=None,
    valid=None,
    faults=None,
    time_ns: int,
    publish_time_ns: int | None = None,
) -> bytes:
    """One reading of `series` as record bytes; ValueError when it does not fit the description.

    `values` are in the description's order (or a mapping by name; a single value for a series
    of one). `time_ns` is the time of the reading and `publish_time_ns` the time it was handed
    over (default: the same), both on this machine's monotonic clock.
    """
    entry = description.series.get(series)
    if entry is None:
        known = ", ".join(description.series)
        raise ValueError(
            f"robot '{description.robot}' has no series '{series}' (it has: {known})"
        )
    given = {
        "velocities": velocities,
        "efforts": efforts,
        "valid": valid,
        "faults": faults,
    }
    allowed = {"joints": ("velocities", "efforts", "valid"), "status": ("faults",)}
    for name, argument in given.items():
        if argument is not None and name not in allowed.get(entry.shape, ()):
            raise ValueError(
                f"series '{series}' has shape '{entry.shape_name}': {name} does not apply"
            )
    available_ns = time_ns if publish_time_ns is None else publish_time_ns
    timestamp = schema.DeviceDataTimestamp(available_ns, time_ns, 0)
    if entry.shape == "joints":
        return _encode_joints(
            description.robot, entry, values, velocities, efforts, valid, timestamp
        )
    if entry.shape == "status":
        return _encode_status(entry, values, faults, timestamp)
    return _encode_custom(entry, values)


def _ordered_values(series: str, names: list[str], values) -> list:
    """`values` as a list in the order of `names`: a mapping by name, or a sequence in that order.

    A single value that is not a sequence stands for a sequence of one.
    """
    if isinstance(values, Mapping):
        lacking = [name for name in names if name not in values]
        if lacking:
            raise ValueError(f"series '{series}': record lacks {', '.join(lacking)}")
        unknown = [str(name) for name in values if name not in names]
        if unknown:
            raise ValueError(
                f"series '{series}': record has unknown fields {', '.join(unknown)}"
            )
        return [values[name] for name in names]
    if isinstance(values, (str, bytes)) or not _is_iterable(values):
        ordered = [values]
    else:
        ordered = list(values)
    if len(ordered) != len(names):
        raise ValueError(
            f"series '{series}' takes {len(names)} values ({', '.join(names)}), got {len(ordered)}"
        )
    return ordered


def _parallel(series: str, argument: str, values, count: int) -> list:
    ordered = list(values) if _is_iterable(values) else [values]
    if len(ordered) != count:
        raise ValueError(
            f"series '{series}': {argument} takes {count} values, got {len(ordered)}"
        )
    return ordered


def _is_flag(value: Any) -> bool:
    """True for a bool, numpy's included (a numpy comparison gives one)."""
    kind = type(value)
    return isinstance(value, bool) or (
        kind.__module__ == "numpy" and kind.__name__ in ("bool", "bool_")
    )


def _is_number(value: Any) -> bool:
    if _is_flag(value) or isinstance(value, (str, bytes)) or value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _float32(series: str, name: str, value: Any, valid: bool) -> float:
    """`value` for a 32-bit float field; a reading that is not valid travels as 0."""
    if not valid:
        return 0.0
    if not _is_number(value) or abs(float(value)) > _FLOAT32_MAX:
        raise ValueError(f"series '{series}': '{name}' is not a finite number")
    return float(value)


def _encode_joints(
    robot: str,
    entry: SeriesDescription,
    values,
    velocities,
    efforts,
    valid,
    timestamp,
) -> bytes:
    series = entry.name
    names = entry.joint_names
    positions = _ordered_values(series, names, values)
    optional = {}
    for argument, flag, given in (
        ("velocities", "has_velocity", velocities),
        ("efforts", "has_effort", efforts),
    ):
        declared = getattr(entry, flag)
        if given is not None and not declared:
            raise ValueError(f"series '{series}' does not declare {flag}")
        if given is None and declared:
            raise ValueError(
                f"series '{series}' declares {flag}: {argument} are required"
            )
        optional[argument] = (
            _parallel(series, argument, given, len(names))
            if declared
            else [0.0] * len(names)
        )
    valids = (
        [True] * len(names)
        if valid is None
        else [bool(flag) for flag in _parallel(series, "valid", valid, len(names))]
    )
    joints = [
        schema.JointState(
            name=name,
            position=_float32(series, name, position, ok),
            velocity=_float32(series, name, velocity, ok),
            effort=_float32(series, name, effort, ok),
            valid=ok,
        )
        for name, position, velocity, effort, ok in zip(
            names, positions, optional["velocities"], optional["efforts"], valids
        )
    ]
    output = schema.JointStateOutput(
        joints=joints,
        device_id=robot,
        has_velocity=entry.has_velocity,
        has_effort=entry.has_effort,
    )
    return schema.JointStateOutputRecord(output, timestamp).to_bytes()


def _is_convertible(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def _is_iterable(value: Any) -> bool:
    try:
        iter(value)
    except TypeError:
        return False
    return True


def _kind_of(value: Any) -> str | None:
    """The value kind a Python value travels as; None when it is none of them."""
    if _is_flag(value):
        return "flag"
    if isinstance(value, str):
        return "text"
    if isinstance(value, (bytes, Mapping)):
        return None
    if _is_convertible(value):
        return "number"
    return "numbers" if _is_iterable(value) else None


_READING_KINDS = {
    "number": schema.RobotReadingKind.NUMBER,
    "flag": schema.RobotReadingKind.FLAG,
    "text": schema.RobotReadingKind.TEXT,
}


def _encode_status(entry: SeriesDescription, values, faults, timestamp) -> bytes:
    series = entry.name
    readings = []
    for declared, value in zip(
        entry.values, _ordered_values(series, entry.value_names, values)
    ):
        name = declared.name
        if value is None:
            kind = _READING_KINDS.get(declared.kind, schema.RobotReadingKind.NUMBER)
            readings.append(
                schema.RobotStatusReading(name=name, kind=kind, valid=False)
            )
            continue
        kind = _kind_of(value)
        if kind not in _READING_KINDS or declared.kind not in ("any", kind):
            raise ValueError(
                f"series '{series}': reading '{name}' is declared {declared.kind}, "
                f"got {type(value).__name__}"
            )
        if kind == "flag":
            content = {"flag": bool(value)}
        elif kind == "text":
            content = {"text": value}
        elif _is_number(value):
            content = {"number": float(value)}
        else:
            raise ValueError(f"series '{series}': '{name}' is not a finite number")
        readings.append(
            schema.RobotStatusReading(name=name, kind=_READING_KINDS[kind], **content)
        )
    status = schema.RobotStatus(
        readings=readings,
        faults=[_encode_fault(series, fault) for fault in faults or ()],
    )
    return schema.RobotStatusRecord(status, timestamp).to_bytes()


def _encode_fault(series: str, fault: Any):
    parts = (
        ()
        if isinstance(fault, (str, bytes)) or not _is_iterable(fault)
        else tuple(fault)
    )
    if not 2 <= len(parts) <= 4 or not all(isinstance(part, str) for part in parts):
        raise ValueError(
            f"series '{series}': a fault is (code, text), (code, text, source) "
            "or (code, text, source, severity)"
        )
    code, text, source, severity = parts + ("", "error")[len(parts) - 2 :]
    if severity not in _SEVERITIES:
        raise ValueError(
            f"series '{series}': fault severity '{severity}' is not one of "
            f"{', '.join(_SEVERITIES)}"
        )
    return schema.RobotFault(
        code=code, text=text, source=source, severity=_SEVERITIES[severity]
    )


def _encode_custom(entry: SeriesDescription, values) -> bytes:
    series = entry.name
    record = {}
    for declared, value in zip(
        entry.values, _ordered_values(series, entry.value_names, values)
    ):
        name = declared.name
        if declared.kind not in ("any", _kind_of(value)):
            raise ValueError(
                f"series '{series}': field '{name}' is declared {declared.kind}, "
                f"got {type(value).__name__}"
            )
        if _is_flag(value):
            value = bool(value)
        elif declared.kind == "number":
            if not _is_number(value):
                raise ValueError(f"series '{series}': '{name}' is not a finite number")
            value = float(value)
        elif declared.kind == "numbers":
            value = list(value)
            if not all(_is_number(item) for item in value):
                raise ValueError(f"series '{series}': '{name}' is not a finite number")
            value = [float(item) for item in value]
        record[name] = value
    try:
        text = json.dumps(record, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"series '{series}': record is not JSON: {error}") from None
    return text.encode("utf-8")


def decode_joints(data: bytes) -> list[JointValue]:
    """The joints of `JointStateOutputRecord` bytes, in the record's order; ValueError on a bad buffer."""
    output = schema.JointStateOutputRecord.from_bytes(data).data
    if output is None:
        return []
    return [
        JointValue(
            name=joint.name,
            position=joint.position,
            velocity=joint.velocity if output.has_velocity else None,
            effort=joint.effort if output.has_effort else None,
            valid=joint.valid,
        )
        for joint in output.joints
    ]


def decode_status(
    data: bytes,
) -> tuple[dict[str, float | bool | str | None], list[Fault]]:
    """Readings by name and the faults of `RobotStatusRecord` bytes; ValueError on a bad buffer.

    A reading the robot could not read is None.
    """
    status = schema.RobotStatusRecord.from_bytes(data).data
    if status is None:
        return {}, []
    readings: dict[str, float | bool | str | None] = {}
    for reading in status.readings:
        if not reading.valid:
            readings[reading.name] = None
        elif reading.kind == schema.RobotReadingKind.FLAG:
            readings[reading.name] = reading.flag
        elif reading.kind == schema.RobotReadingKind.TEXT:
            readings[reading.name] = reading.text
        else:
            readings[reading.name] = reading.number
    faults = [
        Fault(
            fault.code,
            fault.text,
            fault.source,
            _SEVERITY_NAMES.get(fault.severity, "error"),
        )
        for fault in status.faults
    ]
    return readings, faults


def decode_custom(data: bytes) -> dict:
    """The JSON object of a record in a shape the robot defines; ValueError when it is not one."""
    try:
        record = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"not a JSON record: {error}") from None
    if not isinstance(record, dict):
        raise ValueError("not a JSON record: the record is not an object")
    return record
