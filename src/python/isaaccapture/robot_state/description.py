# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Robot description: what a robot is, read from a TOML file.

A description names the robot's data series (joints, status, or a shape the robot defines),
the joints of each joints series with kind and limits, and the grippers. The order of joints,
readings and fields in the file is the order of the values in a record.
"""

from __future__ import annotations

import math
import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

DESCRIPTION_TOPIC = "robot_description"
DEFAULT_STALE_AFTER_S = 0.5

MAX_SERIES = 64
MAX_JOINTS = 256
MAX_VALUES = 256
MAX_ATTRIBUTE_CHARS = 256

JOINT_KINDS = ("revolute", "prismatic", "continuous", "gripper")
READING_KINDS = ("any", "number", "flag", "text")
FIELD_KINDS = ("number", "flag", "text", "numbers", "any")

_ROBOT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")
_TOP_KEYS = ("robot", "series", "shapes", "grippers", "attributes")
_SERIES_KEYS = ("shape", "rate_hz", "stale_after_s", "attributes")
_SHAPE_KEYS = {
    "joints": ("joints", "has_velocity", "has_effort"),
    "status": ("readings",),
}
_JOINT_KEYS = (
    "name",
    "kind",
    "min",
    "max",
    "max_velocity",
    "max_effort",
    "unit",
    "open",
    "closed",
)
_VALUE_KEYS = ("name", "kind", "unit")
_GRIPPER_KEYS = ("series", "joints", "open", "closed")
_KIND_UNITS = {"revolute": "rad", "continuous": "rad", "prismatic": "m"}


class RobotDescriptionError(ValueError):
    """A robot description that breaks a rule; the text names the source and the rule."""


@dataclass(frozen=True)
class JointDescription:
    """One joint of a joints series. `min` and `max` are None for a joint without limits."""

    name: str
    kind: str
    min: float | None
    max: float | None
    max_velocity: float | None
    max_effort: float | None
    unit: str


@dataclass(frozen=True)
class GripperDescription:
    """A gripper: joints of one joints series; the first joint measures the opening."""

    name: str
    series: str
    joints: tuple[str, ...]
    open: float
    closed: float


@dataclass(frozen=True)
class ValueDescription:
    """One reading of a status series, or one field of a shape the robot defines."""

    name: str
    kind: str
    unit: str


@dataclass(frozen=True)
class SeriesDescription:
    """One data series. `shape` is "joints", "status" or "custom"; `shape_name` names a custom shape."""

    name: str
    shape: str
    shape_name: str
    joints: tuple[JointDescription, ...]
    values: tuple[ValueDescription, ...]
    has_velocity: bool
    has_effort: bool
    rate_hz: float | None
    stale_after_s: float
    attributes: Mapping[str, str]

    @property
    def joint_names(self) -> list[str]:
        return [joint.name for joint in self.joints]

    @property
    def value_names(self) -> list[str]:
        return [value.name for value in self.values]

    @property
    def limits(self) -> dict[str, tuple[float, float]]:
        """Position limits of the joints that have them, by joint name."""
        return {
            joint.name: (joint.min, joint.max)
            for joint in self.joints
            if joint.min is not None and joint.max is not None
        }


@dataclass(frozen=True)
class RobotDescription:
    """What a robot is: its data series, in file order, and its grippers."""

    robot: str
    series: Mapping[str, SeriesDescription]
    grippers: Mapping[str, GripperDescription]
    attributes: Mapping[str, str]


def default_joint_unit(kind: str) -> str:
    """Unit of a joint position when the description states none."""
    return _KIND_UNITS.get(kind, "")


def load_robot_description(path: str | os.PathLike) -> RobotDescription:
    """Read and check a robot description file. A file that cannot be read raises its OSError."""
    source = os.fspath(path)
    with open(source, "rb") as file:
        raw = file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RobotDescriptionError(f"{source}: not valid TOML: {error}") from None
    return parse_robot_description(text, source=str(source))


def parse_robot_description(text: str, *, source: str = "<string>") -> RobotDescription:
    """Check the text of a robot description file and return the description."""
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise RobotDescriptionError(f"{source}: not valid TOML: {error}") from None
    try:
        return _Parser(document).parse()
    except _Refused as refused:
        raise RobotDescriptionError(f"{source}: {refused}") from None


class _Refused(Exception):
    """A broken rule, without the source."""


def _fail(where: str, reason: str):
    raise _Refused(f"{where}: {reason}" if where else reason)


def _check_name(what: str, name: str) -> None:
    if not _NAME.fullmatch(name):
        _fail(
            "",
            f"{what} name '{name}' must be 1 to 64 letters, digits or '_', starting with a letter",
        )


def _table(where: str, key: str, value: Any) -> dict:
    if not isinstance(value, dict):
        _fail(where, f"{key} must be a table")
    return value


def _number(where: str, key: str, value: Any) -> float:
    # bool is an int in Python; a description that says `min = true` is a mistake.
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        _fail(where, f"{key} must be a finite number")
    return float(value)


def _positive(where: str, key: str, table: dict) -> float | None:
    if key not in table:
        return None
    value = _number(where, key, table[key])
    if value <= 0.0:
        _fail(where, f"{key} must be above zero")
    return value


def _flag(where: str, key: str, table: dict) -> bool:
    value = table.get(key, False)
    if not isinstance(value, bool):
        _fail(where, f"{key} must be true or false")
    return value


def _text(where: str, key: str, value: Any) -> str:
    if not isinstance(value, str):
        _fail(where, f"{key} must be text")
    return value


def _attributes(where: str, table: dict) -> dict[str, str]:
    attributes = _table(where, "attributes", table.get("attributes", {}))
    for key, value in attributes.items():
        _text(where, f"attributes.{key}", value)
        if len(value) > MAX_ATTRIBUTE_CHARS:
            _fail(
                where,
                f"more than {MAX_ATTRIBUTE_CHARS} characters in attribute '{key}'",
            )
    return dict(attributes)


def _non_empty_list(where: str, key: str, table: dict, limit: int) -> list:
    entries = table.get(key)
    if not isinstance(entries, list) or not entries:
        _fail(where, f"{key} must be a non-empty list")
    if len(entries) > limit:
        _fail(where, f"more than {limit} {key}")
    return entries


def _unknown_keys(where: str, table: dict, known: tuple[str, ...]) -> None:
    for key in table:
        if key not in known:
            _fail(where, f"unknown key '{key}'")


def _values(
    where: str,
    key: str,
    what: str,
    table: dict,
    kinds: tuple[str, ...],
    default_kind: str,
) -> tuple[ValueDescription, ...]:
    """The readings of a status series or the fields of a shape."""
    values: list[ValueDescription] = []
    seen: set[str] = set()
    for index, entry in enumerate(_non_empty_list(where, key, table, MAX_VALUES)):
        if isinstance(entry, str):
            entry = {"name": entry}
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            _fail(where, f"{key}[{index}] must be a name or a table with a name")
        name = entry["name"]
        _check_name(what, name)
        if name in seen:
            _fail(where, f"{what} '{name}' is listed twice")
        seen.add(name)
        entry_where = f"{where}: {what} '{name}'"
        _unknown_keys(entry_where, entry, _VALUE_KEYS)
        kind = entry.get("kind", default_kind)
        if kind not in kinds:
            _fail(entry_where, f"kind '{kind}' is not one of {', '.join(kinds)}")
        unit = _text(entry_where, "unit", entry.get("unit", ""))
        values.append(ValueDescription(name=name, kind=kind, unit=unit))
    return tuple(values)


class _Parser:
    def __init__(self, document: dict) -> None:
        self._document = document
        self._shapes: dict[str, tuple[ValueDescription, ...]] = {}
        self._grippers: dict[str, GripperDescription] = {}

    def parse(self) -> RobotDescription:
        document = self._document
        for key in document:
            if key not in _TOP_KEYS:
                _fail("", f"unknown key '{key}' (expected {', '.join(_TOP_KEYS)})")
        robot = document.get("robot")
        if not isinstance(robot, str) or not _ROBOT_NAME.fullmatch(robot):
            _fail(
                "",
                "robot must be a name of 1 to 64 letters, digits, '_', '.' or '-'",
            )
        attributes = _attributes("", document)

        for name, table in _table("", "shapes", document.get("shapes", {})).items():
            self._shapes[name] = self._shape(name, table)

        series_tables = _table("", "series", document.get("series", {}))
        if not series_tables:
            _fail("", "series must declare at least one data series")
        if len(series_tables) > MAX_SERIES:
            _fail("", f"more than {MAX_SERIES} series")
        series = {
            name: self._series(name, table) for name, table in series_tables.items()
        }

        grippers = _table("", "grippers", document.get("grippers", {}))
        for name, table in grippers.items():
            self._gripper_table(name, table, series)

        return RobotDescription(
            robot=robot, series=series, grippers=self._grippers, attributes=attributes
        )

    def _shape(self, name: str, table: Any) -> tuple[ValueDescription, ...]:
        _check_name("shape", name)
        if name in _SHAPE_KEYS:
            _fail("", f"shape name '{name}' is reserved")
        where = f"shape '{name}'"
        _unknown_keys(where, _table("", where, table), ("fields",))
        return _values(where, "fields", "field", table, FIELD_KINDS, "number")

    def _series(self, name: str, table: Any) -> SeriesDescription:
        _check_name("series", name)
        if name == DESCRIPTION_TOPIC:
            _fail("", f"series name '{DESCRIPTION_TOPIC}' is reserved")
        where = f"series '{name}'"
        _table("", where, table)
        if "shape" not in table:
            _fail(
                where, "shape is required (joints, status, or a shape under [shapes])"
            )
        shape_name = _text(where, "shape", table["shape"])
        if shape_name not in _SHAPE_KEYS and shape_name not in self._shapes:
            _fail(where, f"unknown shape '{shape_name}'")

        own_keys = _SHAPE_KEYS.get(shape_name, ())
        for key in table:
            if key in _SERIES_KEYS or key in own_keys:
                continue
            if any(key in keys for keys in _SHAPE_KEYS.values()):
                _fail(where, f"'{key}' does not apply to shape '{shape_name}'")
            _fail(where, f"unknown key '{key}'")

        rate_hz = _positive(where, "rate_hz", table)
        stale_after_s = _number(
            where, "stale_after_s", table.get("stale_after_s", DEFAULT_STALE_AFTER_S)
        )
        if stale_after_s < 0.0:
            _fail(where, "stale_after_s must be zero or above")

        joints: tuple[JointDescription, ...] = ()
        values: tuple[ValueDescription, ...] = ()
        if shape_name == "joints":
            joints = self._joints(name, where, table)
        elif shape_name == "status":
            values = _values(where, "readings", "reading", table, READING_KINDS, "any")
        else:
            values = self._shapes[shape_name]

        return SeriesDescription(
            name=name,
            shape=shape_name if shape_name in _SHAPE_KEYS else "custom",
            shape_name=shape_name,
            joints=joints,
            values=values,
            has_velocity=_flag(where, "has_velocity", table),
            has_effort=_flag(where, "has_effort", table),
            rate_hz=rate_hz,
            stale_after_s=stale_after_s,
            attributes=_attributes(where, table),
        )

    def _joints(
        self, series: str, where: str, table: dict
    ) -> tuple[JointDescription, ...]:
        joints: list[JointDescription] = []
        seen: set[str] = set()
        entries = _non_empty_list(where, "joints", table, MAX_JOINTS)
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                _fail(where, f"joints[{index}] must be a table with a name")
            name = entry["name"]
            _check_name("joint", name)
            if name in seen:
                _fail(where, f"joint '{name}' is listed twice")
            seen.add(name)
            joints.append(self._joint(series, f"{where}: joint '{name}'", entry))
        return tuple(joints)

    def _joint(self, series: str, where: str, entry: dict) -> JointDescription:
        name = entry["name"]
        _unknown_keys(where, entry, _JOINT_KEYS)
        kind = entry.get("kind", "revolute")
        if kind not in JOINT_KINDS:
            _fail(where, f"kind '{kind}' is not one of {', '.join(JOINT_KINDS)}")

        low = _number(where, "min", entry["min"]) if "min" in entry else None
        high = _number(where, "max", entry["max"]) if "max" in entry else None
        if kind == "continuous":
            if low is not None or high is not None:
                _fail(where, "min and max do not apply to kind 'continuous'")
        elif low is None or high is None:
            _fail(where, f"min and max are required for kind '{kind}'")
        elif low >= high:
            _fail(where, f"min {low:g} must be below max {high:g}")

        max_velocity = _positive(where, "max_velocity", entry)
        max_effort = _positive(where, "max_effort", entry)
        unit = _text(where, "unit", entry.get("unit", ""))

        if kind != "gripper":
            if "open" in entry or "closed" in entry:
                _fail(where, "open and closed apply to kind 'gripper' only")
        else:
            # The joint declares a gripper of its own name: closed at min, open at max unless stated.
            open_position = _number(where, "open", entry.get("open", high))
            closed_position = _number(where, "closed", entry.get("closed", low))
            self._add_gripper(
                where,
                name,
                series,
                (name,),
                open_position,
                closed_position,
                "rename the joint, or give it another kind and declare its gripper "
                "under [grippers]",
            )

        return JointDescription(
            name=name,
            kind=kind,
            min=low,
            max=high,
            max_velocity=max_velocity,
            max_effort=max_effort,
            unit=unit or default_joint_unit(kind),
        )

    def _gripper_table(
        self, name: str, table: Any, series: dict[str, SeriesDescription]
    ) -> None:
        _check_name("gripper", name)
        where = f"gripper '{name}'"
        _unknown_keys(where, _table("", where, table), _GRIPPER_KEYS)
        series_name = _text(where, "series", table.get("series"))
        target = series.get(series_name)
        if target is None or target.shape != "joints":
            _fail(where, f"series '{series_name}' is not a joints series")
        joints = _non_empty_list(where, "joints", table, MAX_JOINTS)
        for index, joint in enumerate(joints):
            if not isinstance(joint, str):
                _fail(where, f"joints[{index}] must be a name")
            if joint not in target.joint_names:
                _fail(where, f"joint '{joint}' is not in series '{series_name}'")
            if joint in joints[:index]:
                _fail(where, f"joint '{joint}' is listed twice")
        open_position = _number(where, "open", table.get("open"))
        closed_position = _number(where, "closed", table.get("closed"))
        self._add_gripper(
            where,
            name,
            series_name,
            tuple(joints),
            open_position,
            closed_position,
            "give this gripper another name",
        )

    def _add_gripper(
        self,
        where: str,
        name: str,
        series: str,
        joints: tuple[str, ...],
        open_position: float,
        closed_position: float,
        remedy: str,
    ) -> None:
        # Only a joint of kind gripper declares one before this: [grippers] comes after the series.
        first = self._grippers.get(name)
        if first is not None:
            _fail(
                where,
                f"a gripper named '{name}' is already declared by joint '{name}' of series "
                f"'{first.series}'; gripper names are unique across the robot: {remedy}",
            )
        if open_position == closed_position:
            _fail(where, "open and closed must differ")
        self._grippers[name] = GripperDescription(
            name=name,
            series=series,
            joints=joints,
            open=open_position,
            closed=closed_position,
        )
