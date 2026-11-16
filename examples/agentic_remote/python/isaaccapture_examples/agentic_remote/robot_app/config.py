# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Robot app settings: frozen dataclasses, checked when built, loaded from YAML.

Every list becomes a tuple, so a loaded configuration cannot be changed in place.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Sequence
from dataclasses import MISSING, dataclass, field, fields
from numbers import Integral, Real
from typing import Any

import yaml

# The assistance wait is never shorter than this; see section 10 of
# docs/design/claude_agentic_remote_execution.md.
MIN_ASSISTANCE_S = 90.0
# The robot connection is never dialed more often than once in this many s.
MIN_RETRY_S = 0.2

_LIMIT_SET_NAMES = ("agent", "human", "recovery")

# The time law of a return to rest, as rest.py plans it: the peak speed of a
# minimum-jerk move over its mean speed, and the share of rest.recovery_speed
# that the planned peak takes.
_PEAK_OVER_MEAN = 1.875
_SPEED_SHARE = 0.9

_Check = Callable[[str, object], object]


class ConfigError(ValueError):
    """A setting is missing, unknown, of the wrong kind or out of range."""


def _real(where: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigError(f"{where} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{where} must be finite, got {value!r}")
    return number


def _positive(where: str, value: object) -> float:
    number = _real(where, value)
    if number <= 0.0:
        raise ConfigError(f"{where} must be above 0, got {number!r}")
    return number


def _not_negative(where: str, value: object) -> float:
    number = _real(where, value)
    if number < 0.0:
        raise ConfigError(f"{where} must not be negative, got {number!r}")
    return number


def _fraction(where: str, value: object) -> float:
    number = _real(where, value)
    if not 0.0 < number < 1.0:
        raise ConfigError(f"{where} must be between 0 and 1, got {number!r}")
    return number


def _at_least(minimum: int) -> _Check:
    def check(where: str, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ConfigError(f"{where} must be a whole number, got {value!r}")
        if value < minimum:
            raise ConfigError(f"{where} must be at least {minimum}, got {value!r}")
        return int(value)

    return check


def _flag(where: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{where} must be true or false, got {value!r}")
    return value


def _text(where: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where} must be a non-empty string, got {value!r}")
    return value


def _name_or_empty(where: str, value: object) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise ConfigError(
            f"{where} must be a string with no space around it, empty for "
            f"none, got {value!r}"
        )
    return value


def _items(where: str, value: object) -> list[object]:
    if hasattr(value, "tolist"):  # a numpy array
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value:
        raise ConfigError(f"{where} must be a non-empty list, got {value!r}")
    return list(value)


def _reals(where: str, value: object) -> tuple[float, ...]:
    items = _items(where, value)
    return tuple(_real(f"{where}[{i}]", item) for i, item in enumerate(items))


def _names(where: str, value: object) -> tuple[str, ...]:
    items = _items(where, value)
    names = tuple(_text(f"{where}[{i}]", item) for i, item in enumerate(items))
    if len(set(names)) != len(names):
        raise ConfigError(f"{where} must not repeat a name, got {list(names)!r}")
    return names


def _optional(check: _Check) -> _Check:
    return lambda where, value: None if value is None else check(where, value)


def _instance(kind: type) -> _Check:
    def check(where: str, value: object) -> object:
        if not isinstance(value, kind):
            raise ConfigError(f"{where} must be a {kind.__name__}, got {value!r}")
        return value

    return check


def _normalise(obj: Any, block: str, **checks: _Check) -> None:
    """Replace every field of a frozen dataclass by its checked value."""
    if set(checks) != {f.name for f in fields(obj)}:
        raise TypeError(f"{type(obj).__name__} must name one check per field")
    for name, check in checks.items():
        where = f"{block}.{name}" if block else name
        object.__setattr__(obj, name, check(where, getattr(obj, name)))


def _check_box(
    where_min: str, low: tuple[float, ...], where_max: str, high: tuple[float, ...]
) -> None:
    if len(low) != len(high):
        raise ConfigError(
            f"{where_min} has {len(low)} values but {where_max} has {len(high)}"
        )
    for i, (a, b) in enumerate(zip(low, high)):
        if not a < b:
            raise ConfigError(
                f"{where_min}[{i}] must be below {where_max}[{i}], got {a!r} and {b!r}"
            )


@dataclass(frozen=True, kw_only=True)
class LimitSet:
    """The limits one control owner's checker applies."""

    name: str  # "agent", "human" or "recovery"
    version: str
    joint_min: tuple[float, ...]  # rad
    joint_max: tuple[float, ...]  # rad
    max_joint_speed: float  # rad/s
    max_step_rad: float | None = None  # per-command clamp; the human set only
    envelope_min: tuple[float, ...] | None = None  # the agent's joint box, rad
    envelope_max: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        block = f"limits.{_text('a limit set name', self.name)}"
        _normalise(
            self,
            block,
            name=_text,
            version=_text,
            joint_min=_reals,
            joint_max=_reals,
            max_joint_speed=_positive,
            max_step_rad=_optional(_positive),
            envelope_min=_optional(_reals),
            envelope_max=_optional(_reals),
        )
        _check_box(
            f"{block}.joint_min", self.joint_min, f"{block}.joint_max", self.joint_max
        )
        if self.envelope_min is None and self.envelope_max is None:
            return
        if self.envelope_min is None or self.envelope_max is None:
            raise ConfigError(
                f"{block}.envelope_min and {block}.envelope_max must be given together"
            )
        _check_box(
            f"{block}.envelope_min",
            self.envelope_min,
            f"{block}.envelope_max",
            self.envelope_max,
        )
        if len(self.envelope_min) != len(self.joint_min):
            raise ConfigError(
                f"{block}.envelope_min has {len(self.envelope_min)} values but "
                f"{block}.joint_min has {len(self.joint_min)}"
            )


@dataclass(frozen=True, kw_only=True)
class Limits:
    """The three limit sets; the control owner selects one."""

    agent: LimitSet
    human: LimitSet
    recovery: LimitSet

    def __post_init__(self) -> None:
        for slot in _LIMIT_SET_NAMES:
            limit_set = _instance(LimitSet)(f"limits.{slot}", getattr(self, slot))
            if limit_set.name != slot:
                raise ConfigError(
                    f"limits.{slot} holds the limit set named {limit_set.name!r}"
                )
            if len(limit_set.joint_min) != len(self.agent.joint_min):
                raise ConfigError(
                    f"limits.{slot} has {len(limit_set.joint_min)} joints but "
                    f"limits.agent has {len(self.agent.joint_min)}"
                )
        if self.human.max_step_rad is None:
            raise ConfigError("limits.human.max_step_rad is required")
        for slot in ("agent", "recovery"):
            if getattr(self, slot).max_step_rad is not None:
                raise ConfigError(
                    f"limits.{slot}.max_step_rad must not be set: "
                    "only human targets are clamped"
                )
        for slot in ("human", "recovery"):
            if getattr(self, slot).envelope_min is not None:
                raise ConfigError(
                    f"limits.{slot} must not have an envelope: only the agent has one"
                )
        # A return to rest must be able to start wherever a person left the arm.
        boxes = zip(
            self.recovery.joint_min,
            self.human.joint_min,
            self.human.joint_max,
            self.recovery.joint_max,
        )
        for i, (rec_min, hum_min, hum_max, rec_max) in enumerate(boxes):
            if hum_min < rec_min or hum_max > rec_max:
                raise ConfigError(
                    "the limits.recovery joint box must contain the limits.human "
                    f"joint box: joint index {i} has human {hum_min!r}..{hum_max!r} "
                    f"and recovery {rec_min!r}..{rec_max!r}"
                )


@dataclass(frozen=True, kw_only=True)
class ExecutorConfig:
    """Pacing of a move: one setpoint per period."""

    period_s: float = 0.02
    max_setpoints: int = 3000
    small_delay_s: float = 0.005  # a setpoint later than this is a missed deadline
    large_delay_s: float = 0.1  # a setpoint later than this ends the move in a hold
    settle_tol_rad: float = 0.03
    settle_timeout_s: float = 2.0
    reads_per_step: int = 2
    max_start_tol_rad: float = 0.05  # the widest start tolerance a move may ask for

    def __post_init__(self) -> None:
        _normalise(
            self,
            "executor",
            period_s=_positive,
            max_setpoints=_at_least(1),
            small_delay_s=_not_negative,
            large_delay_s=_positive,
            settle_tol_rad=_positive,
            settle_timeout_s=_positive,
            reads_per_step=_at_least(1),
            max_start_tol_rad=_positive,
        )
        if self.large_delay_s <= self.small_delay_s:
            raise ConfigError(
                "executor.large_delay_s must be above executor.small_delay_s, got "
                f"{self.large_delay_s!r} and {self.small_delay_s!r}"
            )


@dataclass(frozen=True, kw_only=True)
class PermitsConfig:
    evidence_age_limit_s: float = 3.0  # oldest camera evidence a permit may rest on
    required_for_agent: bool = True
    first_wait_s: float = 0.5  # longest a move waits for its first covering permit

    def __post_init__(self) -> None:
        _normalise(
            self,
            "permits",
            evidence_age_limit_s=_positive,
            required_for_agent=_flag,
            first_wait_s=_not_negative,
        )


@dataclass(frozen=True, kw_only=True)
class WaitsConfig:
    assistance_s: float = 120.0
    # Seconds before the end at which both terminals ring the bell.
    bell_at_s: tuple[float, ...] = (30.0, 10.0, 5.0, 4.0, 3.0, 2.0, 1.0)

    def __post_init__(self) -> None:
        _normalise(self, "waits", assistance_s=_real, bell_at_s=_reals)
        if self.assistance_s < MIN_ASSISTANCE_S:
            raise ConfigError(
                f"waits.assistance_s must be at least {MIN_ASSISTANCE_S:g} s, "
                f"got {self.assistance_s!r}"
            )
        marks = self.bell_at_s
        if any(later >= earlier for earlier, later in zip(marks, marks[1:])):
            raise ConfigError(
                f"waits.bell_at_s must run from the largest mark down, "
                f"got {list(marks)!r}"
            )
        if marks[0] > self.assistance_s or marks[-1] <= 0.0:
            raise ConfigError(
                "waits.bell_at_s must lie above 0 and within waits.assistance_s, "
                f"got {list(marks)!r}"
            )


@dataclass(frozen=True, kw_only=True)
class RestConfig:
    """Return to rest and torque release."""

    rest_q: tuple[float, ...]  # rad
    rest_tol_rad: float = 0.05  # at rest for a release and at the end of a return
    # At rest when the app connects: how far from rest_q the arm may lie with
    # its motors off. Never below rest_tol_rad.
    start_rest_tol_rad: float = 0.05
    agree_tol_rad: float = 0.02  # two reads of a joint must agree this closely
    read_gap_s: float = 0.2
    recovery_speed: float = 0.1  # rad/s
    min_duration_s: float = 2.0
    region_min: tuple[float, ...]  # the tested recovery region, a joint box, rad
    region_max: tuple[float, ...]
    auto_return: bool = True  # the timed return when the assistance wait runs out
    table_clearance_m: float = 0.02

    def __post_init__(self) -> None:
        _normalise(
            self,
            "rest",
            rest_q=_reals,
            rest_tol_rad=_positive,
            start_rest_tol_rad=_positive,
            agree_tol_rad=_positive,
            read_gap_s=_positive,
            recovery_speed=_positive,
            min_duration_s=_positive,
            region_min=_reals,
            region_max=_reals,
            auto_return=_flag,
            table_clearance_m=_not_negative,
        )
        if self.start_rest_tol_rad < self.rest_tol_rad:
            raise ConfigError(
                "rest.start_rest_tol_rad must not be below rest.rest_tol_rad, got "
                f"{self.start_rest_tol_rad!r} and {self.rest_tol_rad!r}"
            )
        _check_box(
            "rest.region_min", self.region_min, "rest.region_max", self.region_max
        )
        if len(self.rest_q) != len(self.region_min):
            raise ConfigError(
                f"rest.rest_q has {len(self.rest_q)} values but rest.region_min has "
                f"{len(self.region_min)}"
            )
        for i, q in enumerate(self.rest_q):
            if not self.region_min[i] <= q <= self.region_max[i]:
                raise ConfigError(
                    f"rest.rest_q[{i}] must lie inside rest.region_min..region_max, "
                    f"got {q!r}"
                )


@dataclass(frozen=True, kw_only=True)
class GripperConfig:
    """Jaw rules; openings run from 0 closed to 1 open."""

    closed_frac: float = 0.04  # a reading at or below this counts as closed
    # Jaws proven empty: how far a reading may lie from the last jaw target. The
    # design's 0.1 rad as a fraction of the gripper's travel, or less.
    target_tol: float = 0.03
    # A command has reached its target, and a close has not ended short of it.
    reach_tol: float = 0.03
    # A reading this close to 1 after a command to full open is a full open.
    full_open_tol: float = 0.03
    agree_tol: float = 0.02
    reads: int = 3
    read_gap_s: float = 0.05
    open_step: float = 0.05
    reference_at_start: bool = True

    def __post_init__(self) -> None:
        _normalise(
            self,
            "gripper",
            closed_frac=_fraction,
            target_tol=_fraction,
            reach_tol=_fraction,
            full_open_tol=_fraction,
            agree_tol=_fraction,
            reads=_at_least(2),
            read_gap_s=_positive,
            open_step=_fraction,
            reference_at_start=_flag,
        )


@dataclass(frozen=True, kw_only=True)
class GraspConfig:
    """How large the grasp region of a MOVE may be; a larger one is refused."""

    max_radius_m: float = 0.3
    max_depth_m: float = 0.06  # how far below the table plane a region may reach

    def __post_init__(self) -> None:
        _normalise(self, "grasp", max_radius_m=_positive, max_depth_m=_not_negative)

    def problem(
        self,
        region: Any,
        table_z: float = 0.0,
        box: tuple[Sequence[float], Sequence[float]] | None = None,
    ) -> str | None:
        """Why a grasp region is not taken, None when it is.

        table_z is the height of the table plane; box is (min, max) of the
        workspace box, which the region's centre must lie in.
        """
        try:
            x, y = (float(value) for value in region.center_xy)
            radius, z_min = float(region.radius_m), float(region.z_min)
            z_max = None if region.z_max is None else float(region.z_max)
        except (TypeError, ValueError):
            return "the grasp region holds a value that is not a number"
        numbers = (x, y, radius, z_min, z_min if z_max is None else z_max)
        if not all(math.isfinite(value) for value in numbers):
            return "the grasp region holds a value that is not a number"
        if radius <= 0.0:
            return "the grasp region needs a radius above 0"
        if radius > self.max_radius_m:
            return (
                f"the grasp region's radius is {radius:g} m, the limit is "
                f"{self.max_radius_m:g} m"
            )
        depth = table_z - z_min
        if depth > self.max_depth_m + 1e-9:
            return (
                f"the grasp region reaches {depth:g} m below the table, the limit "
                f"is {self.max_depth_m:g} m"
            )
        if z_max is not None and z_max < z_min:
            return (
                f"the grasp region ends at z = {z_max:g} m, below its lowest "
                f"z = {z_min:g} m"
            )
        if box is not None:
            low, high = box
            if not (low[0] <= x <= high[0] and low[1] <= y <= high[1]):
                return (
                    f"the grasp region's centre ({x:g}, {y:g}) m is outside the "
                    "workspace box"
                )
        return None


@dataclass(frozen=True, kw_only=True)
class StateConfig:
    publish_period_s: float = 0.05
    recent_window_s: float = 0.5
    history_s: float = 6.0
    idle_read_period_s: float = 0.02

    def __post_init__(self) -> None:
        _normalise(
            self,
            "state",
            publish_period_s=_positive,
            recent_window_s=_positive,
            history_s=_positive,
            idle_read_period_s=_positive,
        )
        if self.recent_window_s > self.history_s:
            raise ConfigError(
                "state.recent_window_s must not exceed state.history_s, got "
                f"{self.recent_window_s!r} and {self.history_s!r}"
            )


@dataclass(frozen=True, kw_only=True)
class HumanConfig:
    view_age_limit_s: float = 0.5  # oldest live view a person may drive with
    sample_age_limit_s: float = 0.25  # oldest controller sample that counts as fresh

    def __post_init__(self) -> None:
        _normalise(
            self, "human", view_age_limit_s=_positive, sample_age_limit_s=_positive
        )


@dataclass(frozen=True, kw_only=True)
class ConnectionConfig:
    # From a failed dial, or a lost robot connection, to the next dial.
    retry_s: float = 1.0

    def __post_init__(self) -> None:
        _normalise(self, "connection", retry_s=_real)
        if self.retry_s < MIN_RETRY_S:
            raise ConfigError(
                f"connection.retry_s must be at least {MIN_RETRY_S:g}, "
                f"got {self.retry_s!r}"
            )


@dataclass(frozen=True, kw_only=True)
class JawsConfig:
    # Whether "reference" may be typed at the Thor terminal: the jaws then close
    # to their stop on a person's word that they are empty.
    allow_reference: bool = False

    def __post_init__(self) -> None:
        _normalise(self, "jaws", allow_reference=_flag)


@dataclass(frozen=True, kw_only=True)
class RobotAppConfig:
    robot_id: str
    profile_id: str
    n_joints: int
    joint_names: tuple[str, ...]
    limits: Limits
    rest: RestConfig
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    permits: PermitsConfig = field(default_factory=PermitsConfig)
    waits: WaitsConfig = field(default_factory=WaitsConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)
    grasp: GraspConfig = field(default_factory=GraspConfig)
    state: StateConfig = field(default_factory=StateConfig)
    human: HumanConfig = field(default_factory=HumanConfig)
    connection: ConnectionConfig = field(default_factory=ConnectionConfig)
    jaws: JawsConfig = field(default_factory=JawsConfig)
    # The name of the camera calibration in use, sent in WELCOME; "" for none.
    camera_calibration_id: str = ""

    def __post_init__(self) -> None:
        _normalise(
            self,
            "",
            robot_id=_text,
            profile_id=_text,
            n_joints=_at_least(1),
            joint_names=_names,
            limits=_instance(Limits),
            rest=_instance(RestConfig),
            executor=_instance(ExecutorConfig),
            permits=_instance(PermitsConfig),
            waits=_instance(WaitsConfig),
            gripper=_instance(GripperConfig),
            grasp=_instance(GraspConfig),
            state=_instance(StateConfig),
            human=_instance(HumanConfig),
            connection=_instance(ConnectionConfig),
            jaws=_instance(JawsConfig),
            camera_calibration_id=_name_or_empty,
        )
        # Each block has already checked that its own lists agree in length.
        per_joint: dict[str, tuple[object, ...]] = {
            "joint_names": self.joint_names,
            "limits.agent.joint_min": self.limits.agent.joint_min,
            "rest.rest_q": self.rest.rest_q,
        }
        for where, values in per_joint.items():
            if len(values) != self.n_joints:
                raise ConfigError(
                    f"{where} has {len(values)} values but n_joints is {self.n_joints}"
                )
        recovery = self.limits.recovery
        for i, q in enumerate(self.rest.rest_q):
            if not recovery.joint_min[i] <= q <= recovery.joint_max[i]:
                raise ConfigError(
                    f"rest.rest_q[{i}] must lie inside the limits.recovery joint box, "
                    f"got {q!r}"
                )
        # The recovery checker would refuse a faster return to rest every time.
        if self.rest.recovery_speed > recovery.max_joint_speed:
            raise ConfigError(
                "rest.recovery_speed must not exceed limits.recovery.max_joint_speed, "
                f"got {self.rest.recovery_speed!r} and {recovery.max_joint_speed!r}"
            )

    @classmethod
    def default(cls) -> RobotAppConfig:
        """Demo values for the simulated arm, equal to configs/robot_app.yaml."""
        n_joints = 6

        def every_joint(value: float) -> tuple[float, ...]:
            return (value,) * n_joints

        return cls(
            robot_id="sim-arm",
            profile_id="sim-6dof",
            n_joints=n_joints,
            joint_names=tuple(f"joint{i}" for i in range(1, n_joints + 1)),
            limits=Limits(
                agent=LimitSet(
                    name="agent",
                    version="sim-1",
                    joint_min=every_joint(-2.0),
                    joint_max=every_joint(2.0),
                    max_joint_speed=0.4,
                    envelope_min=every_joint(-1.5),
                    envelope_max=every_joint(1.5),
                ),
                human=LimitSet(
                    name="human",
                    version="sim-1",
                    joint_min=every_joint(-3.0),
                    joint_max=every_joint(3.0),
                    max_joint_speed=1.5,
                    max_step_rad=0.1,
                ),
                recovery=LimitSet(
                    name="recovery",
                    version="sim-1",
                    joint_min=every_joint(-3.0),
                    joint_max=every_joint(3.0),
                    max_joint_speed=0.1,
                ),
            ),
            rest=RestConfig(
                rest_q=every_joint(0.0),
                region_min=every_joint(-2.5),
                region_max=every_joint(2.5),
            ),
        )


def longest_return_setpoints(config: RobotAppConfig) -> int:
    """Setpoints of the longest return to rest: from the far end of the rest region."""
    rest = config.rest
    travel = _longest_return_travel(rest)
    by_speed = _PEAK_OVER_MEAN * travel / (_SPEED_SHARE * rest.recovery_speed)
    duration = max(rest.min_duration_s, by_speed)
    return max(1, math.ceil(duration / config.executor.period_s - 1e-9))


def _longest_return_travel(rest: RestConfig) -> float:
    """The largest joint travel to rest from inside the rest region, in rad."""
    box = zip(rest.rest_q, rest.region_min, rest.region_max)
    return max(max(q - low, high - q) for q, low, high in box)


def _check_return_fits(config: RobotAppConfig) -> None:
    """A return to rest from anywhere in the rest region must fit in one move."""
    needed = longest_return_setpoints(config)
    if needed > config.executor.max_setpoints:
        raise ConfigError(
            f"executor.max_setpoints must be at least {needed}, got "
            f"{config.executor.max_setpoints}: a return to rest from the far end "
            "of rest.region_min..region_max (largest joint travel "
            f"{_longest_return_travel(config.rest):g} rad at rest.recovery_speed "
            f"{config.rest.recovery_speed:g} rad/s) would be refused for its length"
        )


class _StrictLoader(yaml.SafeLoader):
    """A safe loader that refuses a mapping with a repeated key."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        seen: set[object] = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=True)
            try:
                repeated = key in seen
                seen.add(key)
            except TypeError:  # unhashable; the base class reports it
                continue
            if repeated:
                line = key_node.start_mark.line + 1
                raise ConfigError(f"line {line}: key {key!r} is given twice")
        return super().construct_mapping(node, deep=deep)


def _mapping(
    where: str, raw: object, kind: type, skip: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Check the keys of a YAML mapping against the fields of its dataclass."""
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must be a mapping of settings, got {raw!r}")
    known = [f for f in fields(kind) if f.name not in skip]
    names = [f.name for f in known]
    unknown = sorted(str(key) for key in raw if key not in names)
    if unknown:
        raise ConfigError(
            f"{where} has unknown key(s) {', '.join(unknown)}; "
            f"known keys: {', '.join(names)}"
        )
    missing = [
        f.name
        for f in known
        if f.name not in raw and f.default is MISSING and f.default_factory is MISSING
    ]
    if missing:
        raise ConfigError(f"{where} lacks required key(s) {', '.join(missing)}")
    return raw


_BLOCKS: dict[str, type] = {
    "rest": RestConfig,
    "executor": ExecutorConfig,
    "permits": PermitsConfig,
    "waits": WaitsConfig,
    "gripper": GripperConfig,
    "grasp": GraspConfig,
    "state": StateConfig,
    "human": HumanConfig,
    "connection": ConnectionConfig,
    "jaws": JawsConfig,
}


def _from_mapping(raw: object) -> RobotAppConfig:
    top = dict(_mapping("the top level", raw, RobotAppConfig))
    sets = _mapping("limits", top["limits"], Limits)
    top["limits"] = Limits(
        **{
            name: LimitSet(
                name=name,
                **_mapping(f"limits.{name}", sets[name], LimitSet, skip=("name",)),
            )
            for name in _LIMIT_SET_NAMES
        }
    )
    for name, kind in _BLOCKS.items():
        if name in top:
            top[name] = kind(**_mapping(name, top[name], kind))
    return RobotAppConfig(**top)


def load_config(path: str | os.PathLike[str]) -> RobotAppConfig:
    """Load and check a robot app YAML file; raises ConfigError naming the file.

    A loaded file must also leave room for the longest return to rest in
    executor.max_setpoints; a configuration built in code is not held to that.
    """
    try:
        with open(path, encoding="utf-8") as stream:
            raw = yaml.load(stream, Loader=_StrictLoader)
        config = _from_mapping(raw)
        _check_return_fits(config)
        return config
    except yaml.YAMLError as exc:
        raise ConfigError(f"{os.fspath(path)}: not valid YAML: {exc}") from exc
    except ConfigError as exc:
        raise ConfigError(f"{os.fspath(path)}: {exc}") from None
