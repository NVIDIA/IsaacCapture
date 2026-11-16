# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""APAI behind the robot app's ports: checkers on SafetyHarness, the arm on a backend.

The only module that imports apai. It never calls a backend's stop() or
hard_estop(), and it reads a hardware backend's joints only with the calls that
fail instead of returning a cached pose. Wiring: load_apai_profile(), then
config_for_profile(), build_apai_checkers() and build_apai_arm().
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import math
import os
import threading
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from .clock import Clock, FakeClock, SystemClock
from .config import GraspConfig, LimitSet, RobotAppConfig
from .interfaces import (
    ArmFault,
    ArmPort,
    CheckerSet,
    ContactStop,
    FeedbackLost,
    GripperOutcome,
    JointReading,
    ReadFailed,
    Refusal,
    TipState,
)
from apai.config import Cfg, load_demo_config
from apai.control import arm_hold
from apai.control.arm_base import ContactFeedback, make_arm
from apai.control.kinematics import Kinematics
from apai.safety.contact import ContactMonitor, ContactMonitorConfig, format_summary
from apai.safety.harness import SafetyHarness, SafetyLimits
from apai.types import (
    ContactDetected,
    GripperFault,
    SafetyViolation,
    pose_to_transform,
)

if TYPE_CHECKING:
    from ..protocol.messages import GraspRegion
    from .state import JointHistory

log = logging.getLogger("isaaccapture.examples.agentic_remote.apai_backend")

JAW_MODES = ("plain", "close_until_contact")
ARM_TYPES = ("mock", "rebot_rs")  # the APAI backends ApaiArm is written for

_EPS = 1e-9
# The harness keeps the text of every refusal; the checker caps the list at this.
_VIOLATIONS_KEPT = 256
_JAW_POLL_S = 0.15
_JAW_GRACE_S = 0.3  # a jaw that is still starting to move is not stalled
_JAW_TIMEOUT_S = 4.0
_JAW_STALL = 0.005  # fraction of travel between two polls
# How often a jaw command is given to a driver that could not read the jaws and
# sent nothing, and the wait before each further try.
_JAW_SEND_TRIES = 3
_JAW_RETRY_S = 0.05
# How APAI's gripper guard ends the fault it raises before any jaw command.
_NOTHING_SENT = "no gripper command sent"
# Part of the fault the guard raises when its stop probe ends a reference.
_REFERENCE_STOPPED = "referencing interrupted by a stop"
# "Jaws proven empty": a reading lies within this of the last jaw target, rad.
JAW_TARGET_TOL_RAD = 0.1
# The reBot RS driver's jaw positions when the profile names none, rad.
_RS_OPEN_POS, _RS_CLOSED_POS = 6.2, 0.0
_MOCK_REFERENCE_TOL = 0.05  # fraction of travel from closed
# How the harness's pose vet begins its refusal of a pose inside a contact spot.
_CONTACT_ZONE_REFUSAL = "TCP would enter the contact zone"
# The radius of the column check_low_pose() opens around the gripper tip, m.
_TIP_COLUMN_M = 1e-6
# How ContactFeedback.stopped() begins its note of a hold at the measured pose,
# and of a hold whose target the driver did not take.
_HELD_MEASURED = "measured pose ["
_HOLD_NOT_SENT = "last command (hold command failed"
# How the name of an operator's confirmation file in APAI's hold folder ends.
_CONFIRM_SUFFIX = ".confirm.json"
# APAI keeps time on the machine's monotonic clock, whatever clock the app runs on.
_MACHINE = SystemClock()


def _refusal(exc: SafetyViolation) -> Refusal:
    """APAI's violation as the robot app's refusal; a contact keeps its own kind."""
    # ContactDetected and MotionHalted both subclass SafetyViolation.
    if isinstance(exc, ContactDetected):
        return ContactStop(str(exc))
    return Refusal(str(exc))


def _joint_of(reading: JointReading) -> int:
    return reading.joint


def _first_outside(q: np.ndarray, low: np.ndarray, high: np.ndarray) -> int | None:
    outside = np.flatnonzero((q < low - _EPS) | (q > high + _EPS))
    return int(outside[0]) if outside.size else None


def _in_region(point: np.ndarray, region: GraspRegion) -> bool:
    x, y = region.center_xy
    if math.hypot(point[0] - x, point[1] - y) > region.radius_m:
        return False
    if point[2] < region.z_min:
        return False
    return region.z_max is None or point[2] <= region.z_max


def _low_tip(
    tip: np.ndarray,
    clearance_m: float,
    limits: SafetyLimits,
    region: GraspRegion | None,
) -> str | None:
    """Why the gripper tip is low: under the table clearance or inside region."""
    floor = limits.table_z + clearance_m
    if tip[2] < floor:
        return (
            f"the gripper tip is at z={tip[2]:.3f} m, below the table "
            f"clearance {floor:.3f} m"
        )
    if region is not None and _in_region(tip, region):
        return "the gripper tip is inside the last grasp region"
    return None


def _merge(base: dict[str, Any], extra: Mapping[str, Any]) -> None:
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value


def _plain(block: Any) -> dict[str, Any]:
    """A config block, APAI's Cfg or a mapping or None, as a plain dict."""
    if isinstance(block, Cfg):
        return block.as_dict()
    return dict(block or {})


@dataclass(frozen=True)
class ApaiProfile:
    """One APAI arm profile and the kinematics built from it."""

    apai_cfg: Cfg  # APAI's configuration as this arm sees it (safety, ...)
    arm_cfg: Cfg  # the arm profile itself (type, model, gripper, ...)
    kinematics: Kinematics


def load_apai_profile(
    arm: str = "mock", overrides: Mapping[str, Any] | None = None
) -> ApaiProfile:
    """Load an APAI arm profile with no camera, no model and no obstacle map.

    arm names a file in APAI's configs/arms, or is a path without ".yaml".
    overrides is merged over the arm's view of APAI's configuration.
    """
    for name in ("APAI_ROBOT", "APAI_BASE"):
        if os.environ.get(name):
            raise ValueError(f"unset {name}: it selects another APAI configuration")
    view = load_demo_config(camera="mock", arm=arm, llm="mock").arms[0]["resolved"]
    if overrides:
        _merge(view, overrides)
    arm_cfg = Cfg(view["arm"])
    kinematics = Kinematics(
        model_path=arm_cfg.model,
        ee_frame=arm_cfg.get("ee_frame", "gripper_end"),
        n_controlled=int(arm_cfg.get("n_joints", 6)),
        joint_signs=arm_cfg.get("joint_signs"),
        ik_task_weights=arm_cfg.get("ik_task_weights"),
    )
    return ApaiProfile(apai_cfg=Cfg(view), arm_cfg=arm_cfg, kinematics=kinematics)


def config_for_profile(config: RobotAppConfig, profile: ApaiProfile) -> RobotAppConfig:
    """The configuration with the two tolerances an APAI profile decides.

    rest.start_rest_tol_rad becomes the profile's rest check tolerance
    (rest_check_tol_rad, APAI's default without one), never less than
    rest.rest_tol_rad. gripper.target_tol becomes at most JAW_TARGET_TOL_RAD
    of a real arm's jaw travel.
    """
    tolerance = max(arm_hold.rest_tolerance(profile.arm_cfg), config.rest.rest_tol_rad)
    rest = dataclasses.replace(config.rest, start_rest_tol_rad=tolerance)
    gripper = config.gripper
    travel = jaw_travel_rad(profile)
    if travel is not None:
        target_tol = min(gripper.target_tol, JAW_TARGET_TOL_RAD / travel)
        gripper = dataclasses.replace(gripper, target_tol=target_tol)
    return dataclasses.replace(config, rest=rest, gripper=gripper)


def jaw_travel_rad(profile: ApaiProfile) -> float | None:
    """A real arm's jaw travel from closed to open, rad; None for the mock arm."""
    if profile.arm_cfg.get("type") == "mock":
        return None  # its jaw position is a fraction, not an angle
    gripper = _plain(profile.arm_cfg.get("gripper"))
    open_pos = float(gripper.get("open_pos", _RS_OPEN_POS))
    closed_pos = float(gripper.get("closed_pos", _RS_CLOSED_POS))
    return abs(open_pos - closed_pos) or None


def unresolved_hold(profile: ApaiProfile) -> dict[str, Any] | None:
    """APAI's hold marker for the motor bus this profile's arm drives, or None.

    A hold marker is the file APAI writes when a run leaves the arm powered
    and holding. None also for a profile whose arm is never held, such as the
    mock arm. It reads one file and sends nothing.
    """
    key = hold_bus(profile)
    return None if key is None else arm_hold.read(key)


def hold_bus(profile: ApaiProfile) -> str | None:
    """The motor bus APAI names this profile's arm by in its hold markers.

    None for an arm that is never held, such as the mock arm.
    """
    return arm_hold.hold_key(profile.arm_cfg.get("type"), profile.arm_cfg)


def confirmation_on_file() -> str | None:
    """The path of an operator's confirmation file in APAI's hold folder, or None.

    Any "*.confirm.json" there counts, whatever arm it names, however old it
    is and whether or not it can be read: APAI's driver claims one by its file
    name and would then connect an arm it does not measure at rest. A folder
    that cannot be listed counts too, and is the path returned: see
    hold_folder_unlistable(). It only lists.
    """
    folder = arm_hold.hold_dir()
    names, unlistable = _hold_folder_names()
    if unlistable is not None:
        return str(folder)
    for name in names:
        if name.endswith(_CONFIRM_SUFFIX):
            return str(folder / name)
    return None


def hold_folder_unlistable() -> str | None:
    """Why APAI's hold folder cannot be listed; None when it can, or is not there.

    A confirmation may lie in such a folder, and APAI's driver would still
    find it by its name. It only lists.
    """
    return _hold_folder_names()[1]


def _hold_folder_names() -> tuple[list[str], str | None]:
    """The names in APAI's hold folder, and why it cannot be listed, if so."""
    try:
        return sorted(os.listdir(arm_hold.hold_dir())), None
    except (FileNotFoundError, NotADirectoryError):
        return [], None  # no folder: nothing is on file
    except OSError as error:
        return [], str(error)


def speed_cap(profile: ApaiProfile) -> float:
    """The APAI profile's joint speed cap (safety.max_joint_vel), rad/s."""
    return float(SafetyLimits.from_config(profile.apai_cfg.safety).max_joint_vel)


def _contact_config(profile: ApaiProfile) -> ContactMonitorConfig:
    block = profile.apai_cfg.safety.get("contact_monitor")
    return ContactMonitorConfig.from_config(block) or ContactMonitorConfig()


def build_apai_checkers(
    config: RobotAppConfig,
    profile: ApaiProfile,
    clock: Clock,
    *,
    paced: bool | None = None,
) -> CheckerSet:
    """Three checkers, one per control owner, each on its own SafetyHarness.

    Speed comes from the owner's limit set, which may not exceed the profile's
    speed cap (ValueError); the rest of the harness limits come from the APAI
    profile. No harness gets an obstacle map. paced is passed to ApaiChecker.
    """
    kinematics = profile.kinematics
    if kinematics.n != config.n_joints:
        raise ValueError(
            f"the APAI profile has {kinematics.n} joints but n_joints is "
            f"{config.n_joints}"
        )
    base = SafetyLimits.from_config(profile.apai_cfg.safety)
    for limit_set in (config.limits.agent, config.limits.human, config.limits.recovery):
        # The profile's cap is the hardware speed bound; nothing is clamped to it.
        if limit_set.max_joint_speed > base.max_joint_vel + _EPS:
            raise ValueError(
                f"limits.{limit_set.name}.max_joint_speed "
                f"{limit_set.max_joint_speed:g} rad/s is above the APAI profile's "
                f"speed cap of {base.max_joint_vel:g} rad/s (safety.max_joint_vel)"
            )
    contact = _contact_config(profile)
    if not contact.enforcing:
        log.info(
            "the APAI profile's contact monitor is off or only measures; "
            "the robot app runs it enforcing"
        )
    # The robot app never moves the arm without a contact watch that can stop it.
    contact = dataclasses.replace(
        contact,
        enabled=True,
        mode="enforce",
        min_reads_per_cycle=config.executor.reads_per_step,
    )
    pose = profile.arm_cfg.get("base_pose")
    base_pose = None if pose is None else pose_to_transform(pose)

    def checker(limit_set: LimitSet, **changes: float) -> ApaiChecker:
        limits = dataclasses.replace(
            base, max_joint_vel=limit_set.max_joint_speed, contact=contact, **changes
        )
        harness = SafetyHarness(
            limits,
            kinematics=kinematics,
            occupancy=None,
            base_pose=base_pose,
            arm_name=limit_set.name,
        )
        return ApaiChecker(
            limit_set,
            harness,
            clock,
            period_s=config.executor.period_s,
            table_clearance_m=config.rest.table_clearance_m,
            paced=paced,
            grasp=config.grasp,
        )

    return CheckerSet(
        agent=checker(config.limits.agent),
        human=checker(config.limits.human),
        # Nothing can feed the recovery watchdog, and the rest pose lies on joint
        # stops, inside the joint margin the other two keep.
        recovery=checker(config.limits.recovery, watchdog_s=math.inf, joint_margin=0.0),
    )


def build_apai_arm(
    config: RobotAppConfig, profile: ApaiProfile, clock: Clock
) -> ApaiArm:
    """The arm port on the backend the profile names; nothing is connected yet.

    The mock arm is placed at the rest pose. For any other arm the
    configuration must be the one config_for_profile() gives, and the profile
    must name its park pose.
    """
    kind = profile.arm_cfg.get("type")
    if kind not in ARM_TYPES:
        raise ValueError(f"arm type {kind!r} is not one of {', '.join(ARM_TYPES)}")
    park = profile.arm_cfg.get("park_q")
    rest = config.rest.rest_q
    if park is None and kind != "mock":
        # APAI's driver would measure rest against all zeros, whatever rest_q is.
        raise ValueError(
            "the APAI profile names no park pose (park_q): a profile that drives "
            "hardware must name it, and rest.rest_q must be that pose"
        )
    if park is not None and (
        len(park) != len(rest) or not np.allclose(park, rest, atol=1e-6)
    ):
        raise ValueError(
            f"rest.rest_q {list(rest)} differs from the APAI profile's park_q "
            f"{list(park)}"
        )
    arm_cfg = profile.arm_cfg
    if kind == "mock":
        # A real arm must start at rest; the mock arm starts at its home_q.
        arm_cfg = Cfg({**arm_cfg.as_dict(), "home_q": list(rest)})
    else:
        # A real arm lies at its unpowered rest when the app connects to it.
        at_start = config_for_profile(config, profile).rest.start_rest_tol_rad
        if config.rest.start_rest_tol_rad != at_start:
            raise ValueError(
                f"rest.start_rest_tol_rad {config.rest.start_rest_tol_rad:g} differs "
                f"from the APAI profile's rest check tolerance {at_start:g}: use "
                "config_for_profile()"
            )
        travel, target_tol = jaw_travel_rad(profile), config.gripper.target_tol
        if travel is not None and target_tol * travel > JAW_TARGET_TOL_RAD + _EPS:
            raise ValueError(
                f"gripper.target_tol {target_tol:g} is {target_tol * travel:.3g} rad "
                f"of the APAI profile's jaw travel of {travel:g} rad, over the "
                f'{JAW_TARGET_TOL_RAD:g} rad of "jaws proven empty": use '
                "config_for_profile()"
            )
    return ApaiArm(
        make_arm(arm_cfg, kinematics=profile.kinematics),
        config,
        clock,
        gripper_cfg=arm_cfg.get("gripper"),
        read_timeout_s=_contact_config(profile).read_timeout_s,
        # The motor library swallows a failed disable; the mock arm has none.
        release_reports=kind == "mock",
    )


class ApaiChecker:
    """One control owner's checker: its limit set on an unmodified SafetyHarness.

    The checker itself applies the limit set's joint box and the envelope. A joint
    outside the box may only move back toward it, as with the harness's own
    limits; the envelope cannot be re-entered at all. The rest is the harness's.
    """

    def __init__(
        self,
        limit_set: LimitSet,
        harness: SafetyHarness,
        clock: Clock,
        *,
        period_s: float,
        table_clearance_m: float,
        paced: bool | None = None,
        grasp: GraspConfig | None = None,
    ) -> None:
        """paced False counts a move's time in steps, for a clock that is not real.

        None picks it from the clock: in steps on a FakeClock, else real pacing.
        grasp bounds the grasp region of a move; None gives the default bounds.
        """
        self._grasp = GraspConfig() if grasp is None else grasp
        kinematics = harness.kin
        if kinematics is None:
            raise ValueError("the harness has no kinematics")
        contact = harness.limits.contact
        if contact is None or not contact.enforcing:
            raise ValueError("the harness's contact monitor is off or only measures")
        self.name = limit_set.name
        self.version = limit_set.version
        self._low = np.array(limit_set.joint_min, dtype=float)
        self._high = np.array(limit_set.joint_max, dtype=float)
        self._n = len(self._low)
        if len(kinematics.joint_limits[0]) != self._n:
            raise ValueError(
                f"limits.{self.name} has {self._n} joints but the arm model has "
                f"{len(kinematics.joint_limits[0])}"
            )
        self._envelope: tuple[np.ndarray, np.ndarray] | None = None
        if limit_set.envelope_min is not None and limit_set.envelope_max is not None:
            self._envelope = (
                np.array(limit_set.envelope_min, dtype=float),
                np.array(limit_set.envelope_max, dtype=float),
            )
        self._harness = harness
        self._kinematics = kinematics
        self._clock = clock
        self._period_s = float(period_s)
        if not (math.isfinite(self._period_s) and self._period_s > 0.0):
            raise ValueError(f"period_s must be above 0, got {period_s!r}")
        self._table_clearance_m = float(table_clearance_m)
        self._paced = not isinstance(clock, FakeClock) if paced is None else bool(paced)
        self._region_lock = threading.Lock()
        self._column = contextlib.ExitStack()  # holds an open grasp column

    @property
    def harness(self) -> SafetyHarness:
        return self._harness

    # Checker

    def feed_watchdog(self, t_mono: float | None = None) -> None:
        if t_mono is None:
            self._harness.heartbeat()
            return
        # The harness ignores evidence stamped in the future.
        age = self._clock.monotonic() - float(t_mono)
        self._harness.heartbeat(at=_MACHINE.monotonic() - age)

    def halt_count(self) -> int:
        return int(self._harness._halt_generation)

    def halt(self, reason: str) -> int:
        self._harness.halt(reason)
        self._trim_violations()
        return self.halt_count()

    def latch(self, reason: str) -> None:
        self._harness.estop(reason)
        self._trim_violations()

    def reset_latch(self) -> None:
        self._harness.reset_estop()

    def latched(self) -> bool:
        return bool(self._harness.estopped)

    def begin_motion(self, halt_count: int) -> None:
        try:
            self._harness.begin_motion(halt_generation=int(halt_count))
        except SafetyViolation as exc:
            self._trim_violations()
            raise _refusal(exc) from exc

    def end_motion(self) -> None:
        self._harness.end_motion()

    def check_setpoint(
        self, q_prev: Sequence[float], q_next: Sequence[float], dt: float
    ) -> None:
        start, target = self._pose(q_prev), self._pose(q_next)
        step_s = self._step_time(dt)
        self._refuse(self._box_problem(target, start))
        try:
            self._harness.approve(start, target, step_s)
        except SafetyViolation as exc:
            self._trim_violations()
            raise _refusal(exc) from exc

    def check_pose(self, q: Sequence[float]) -> None:
        self._check_pose(q, low_tip=False)

    def check_low_pose(self, q: Sequence[float]) -> None:
        self._check_pose(q, low_tip=True)

    def _check_pose(self, q: Sequence[float], *, low_tip: bool) -> None:
        pose = self._pose(q)
        if self._harness.estopped:
            raise Refusal("e-stop latched")
        self._refuse(self._box_problem(pose, pose))
        # The rest pose lies on joint stops, inside the harness's joint margin, where
        # its pose vet refuses. The limit set decides the joints; the harness vets
        # the rest at the nearest pose its margin accepts.
        inside = self._kinematics.clamp(pose, self._harness.limits.joint_margin)
        if low_tip:
            # The pose vet takes a column in which the table rules are off. This
            # one holds the gripper tip and nothing else, at any height.
            tip = self._kinematics.fk(inside)[:3, 3]
            reason = self._harness.vet_pose(inside, tip[:2], _TIP_COLUMN_M, -math.inf)
        else:
            reason = self._harness.vet_pose(inside)
        if reason is not None and reason.startswith(_CONTACT_ZONE_REFUSAL):
            # The pose vet takes a pose for a target. The arm is already here,
            # and leaving a contact spot is allowed: the step vet knows that
            # rule and runs the gates the pose vet had not reached.
            reason = self._harness.vet_step(inside, inside, self._period_s)
        self._refuse(reason)

    def check_path(
        self, q_start: Sequence[float], setpoints: Sequence[Sequence[float]], dt: float
    ) -> None:
        if len(setpoints) == 0:
            raise Refusal("the path has no setpoints")
        step_s = self._step_time(dt)
        start = self._pose(q_start)
        for index, row in enumerate(setpoints):
            try:
                target = self._pose(row)
                self._refuse(self._box_problem(target, start))
                self._refuse(self._harness.vet_step(start, target, step_s))
            except Refusal as exc:
                raise Refusal(f"setpoint {index}: {exc.reason}") from None
            start = target

    def check_grasp_region(self, region: GraspRegion) -> None:
        # The harness leaves the size of a region to its caller. Inside one
        # only its workspace box is left to keep the tip above the table.
        limits = self._harness.limits
        box = (limits.workspace_min, limits.workspace_max)
        self._refuse(self._grasp.problem(region, limits.table_z, box))

    def set_grasp_region(self, region: GraspRegion | None) -> None:
        try:
            with self._region_lock:
                self._column.close()
                self._harness.clear_grasp_exemption()
                if region is not None:
                    self._open_region(region)
        except SafetyViolation as exc:
            raise _refusal(exc) from exc

    def note_contact(self, q: Sequence[float]) -> None:
        try:
            pose = self._pose(q)
        except Refusal as exc:
            log.warning("no contact spot kept for %s: %s", self.name, exc.reason)
            return
        # The app reports the contact of whichever owner's move met it.
        label = "an earlier move"
        monitor = ContactMonitor(self._harness.limits.contact, label=label)
        monitor.tripped = {
            "joint": "unknown",
            "motion": label,
            "measured_q": pose.tolist(),
        }
        self._harness._note_contact(monitor, f"contact during {label}")
        self._trim_violations()

    def clear_contact_spots(self) -> None:
        self._harness.clear_contact_zones("cleared on request")

    def tip_blocked(self, q: Sequence[float], region: GraspRegion | None) -> str | None:
        try:
            tip = self._kinematics.fk(self._pose(q))[:3, 3]
        except Refusal as exc:
            return exc.reason
        return _low_tip(tip, self._table_clearance_m, self._harness.limits, region)

    def tip_state(
        self, q: Sequence[float], region: GraspRegion | None
    ) -> TipState | None:
        tip = self._kinematics.fk(self._pose(q))[:3, 3]
        # Low is where a return to rest is refused or the harness's pose vet
        # refuses the tip: under the higher of the two table clearances.
        limits = self._harness.limits
        clearance = max(self._table_clearance_m, float(limits.table_clearance))
        return TipState(float(tip[2]), _low_tip(tip, clearance, limits, region))

    def contact_watch(
        self, arm: ArmPort, q_start: Sequence[float], history: JointHistory | None
    ) -> ApaiContactWatch:
        monitor = self._harness.contact_monitor(f"{self.name} move")
        try:
            return ApaiContactWatch(
                arm,
                self._pose(q_start),
                history,
                monitor,
                self._clock,
                period_s=self._period_s,
                paced=self._paced,
            )
        except SafetyViolation as exc:
            raise _refusal(exc) from exc

    # Internals

    def _pose(self, q: Sequence[float]) -> np.ndarray:
        try:
            pose = np.array(q, dtype=float)
        except (TypeError, ValueError):
            raise Refusal("a joint pose must be a list of numbers") from None
        if pose.shape != (self._n,):
            raise Refusal(f"a joint pose needs {self._n} values, got {pose.shape}")
        if not np.all(np.isfinite(pose)):
            raise Refusal("a joint pose holds a value that is not finite")
        return pose

    @staticmethod
    def _step_time(dt: float) -> float:
        # The harness skips its speed check for a step time of 0 or less.
        if isinstance(dt, bool) or not isinstance(dt, (int, float)):
            raise Refusal(f"the step time must be a number, got {dt!r}")
        if not (math.isfinite(dt) and dt > 0.0):
            raise Refusal(f"the step time must be above 0, got {dt!r}")
        return float(dt)

    @staticmethod
    def _refuse(reason: str | None) -> None:
        if reason is not None:
            raise Refusal(reason)

    def _box_problem(self, target: np.ndarray, start: np.ndarray) -> str | None:
        """Why a step breaks the limit set's joint box or the envelope, else None."""
        # A joint that sagged or was left outside the box must be able to come back.
        returning = ((target < self._low) & (target > start)) | (
            (target > self._high) & (target < start)
        )
        joint = _first_outside(
            np.where(returning, self._low, target), self._low, self._high
        )
        if joint is not None:
            return (
                f"joint {joint + 1} target {target[joint]:.3f} rad is outside the "
                f"{self.name} joint limits "
                f"{self._low[joint]:.3f}..{self._high[joint]:.3f}"
            )
        if self._envelope is None:
            return None
        low, high = self._envelope
        for what, pose in (("the step starts", start), ("the target is", target)):
            joint = _first_outside(pose, low, high)
            if joint is not None:
                return (
                    f"{what} outside the {self.name} envelope, which cannot be "
                    f"re-entered: joint {joint + 1} at {pose[joint]:.3f} rad, "
                    f"envelope {low[joint]:.3f}..{high[joint]:.3f}"
                )
        return None

    def _open_region(self, region: GraspRegion) -> None:
        self.check_grasp_region(region)
        center = np.array(region.center_xy, dtype=float)
        if region.z_max is None:
            self._harness.allow_grasp_descent(center, region.radius_m, region.z_min)
            return
        # A region with a ceiling is APAI's target column, open until the next call.
        cylinder = (center, region.radius_m, region.z_min)
        column = self._harness.target_column_exemption(cylinder, region.z_max)
        if not self._column.enter_context(column):
            raise Refusal("the harness did not open the grasp region")

    def _trim_violations(self) -> None:
        kept = self._harness.violations
        if len(kept) > _VIOLATIONS_KEPT:
            del kept[: len(kept) - _VIOLATIONS_KEPT // 2]


class ApaiContactWatch:
    """One move's joint reads and contact check, run by APAI's ContactFeedback.

    Paced, stream time is the machine clock's and each step reads as many joints
    as fit before its deadline. Unpaced, stream time is counted in steps and
    every step reads every joint.
    """

    def __init__(
        self,
        arm: ArmPort,
        q_start: np.ndarray,
        history: JointHistory | None,
        monitor: ContactMonitor,
        clock: Clock,
        *,
        period_s: float,
        paced: bool,
    ) -> None:
        self._arm = arm
        self._history = history
        self._monitor = monitor
        self._clock = clock
        self._readings: list[JointReading] = []
        self._steps = 0
        self._finished = False
        self._contact: ContactDetected | None = None  # what ended the move
        # ContactFeedback asks its arm for a per-joint reader; the watch is that arm.
        self._feedback = ContactFeedback(
            self, monitor, q_start, dt=period_s, paced=paced
        )

    def contact_joint_reader(self) -> Any:
        return self._read

    def after_send(
        self, index: int, q: Sequence[float], deadline_mono: float
    ) -> list[JointReading]:
        self._steps += 1
        if self._steps == 1:
            # Stream time starts with the first setpoint, so a wait between building
            # the watch and that setpoint is not lost feedback.
            self._feedback.start(_MACHINE.monotonic())
        self._readings = []
        deadline = _MACHINE.monotonic() + (deadline_mono - self._clock.monotonic())
        try:
            self._feedback.after_send(
                self._steps, np.array(q, dtype=float), deadline=deadline
            )
        except ContactDetected as exc:
            self._contact = exc
            raise _refusal(exc) from exc
        except SafetyViolation as exc:
            # The monitor notes lost feedback before it raises a plain violation.
            if self._monitor.feedback_lost:
                raise FeedbackLost(str(exc)) from exc
            raise _refusal(exc) from exc
        finally:
            self._keep_readings()
        return list(self._readings)

    def hold(self) -> None:
        # APAI's own hold after a contact: one fresh read of every joint, sent
        # as the target only if it agrees with the monitor's newest samples.
        contact = self._contact or ContactDetected("contact")
        self._monitor.hold = None
        self._feedback.stopped(contact)
        held = self._monitor.hold or ""
        if held.startswith(_HOLD_NOT_SENT):
            raise ArmFault(held)
        if not held.startswith(_HELD_MEASURED):
            raise ReadFailed(held or "the hold was not attempted")

    # What ContactFeedback.stopped() asks of its arm.

    def read_positions_for_monitor(self) -> list[float]:
        readings = self._arm.read_all()
        if self._history is not None:
            for reading in readings:
                self._history.add(reading)
        return [reading.value for reading in sorted(readings, key=_joint_of)]

    def send_joint_target(self, q: Sequence[float]) -> None:
        self._arm.send_target([float(value) for value in q])

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        summary = self._monitor.summary()
        ended_early = (
            summary["tripped"] or summary["feedback_lost"] or summary["would_trip"]
        )
        level = logging.INFO if ended_early else logging.DEBUG
        log.log(level, "%s", format_summary(summary))

    def _read(self, joint: int, _timeout_s: float) -> float:
        # A ReadFailed raised here is what ContactFeedback counts as a failed read.
        reading = self._arm.read_joint(joint)
        if not reading.valid:
            raise ReadFailed(f"joint index {joint} gave a reading that is not valid")
        self._readings.append(reading)
        return reading.value

    def _keep_readings(self) -> None:
        # The monitor takes a reading too far from the joint's last one for a failed
        # read; such a reading is kept as not valid.
        accepted = self._monitor.latest
        self._readings = [
            reading
            if accepted[reading.joint] == reading.value
            else dataclasses.replace(reading, valid=False)
            for reading in self._readings
        ]
        if self._history is not None:
            for reading in self._readings:
                self._history.add(reading)


class _JawsUnread(Exception):
    """The driver could not read the jaws and sent them nothing."""


class _GuardJaws:
    """The jaws of a reBot RS backend, through its gripper guard (jaw angles, rad)."""

    def __init__(self, backend: Any, guard: Any, gripper: Mapping[str, Any]) -> None:
        if not callable(getattr(guard, "_stopped", None)):
            raise ValueError("the gripper guard has no stop probe to end a ramp with")
        self._backend = backend
        self._guard = guard
        self._hold_kp = float(gripper.get("hold_kp", 1.0))  # the driver's default
        self.closed = float(guard.closed_pos)
        self.open = float(guard.open_pos)

    def referenced(self) -> bool:
        return bool(self._guard.referenced)

    def reference(self, cancel: threading.Event | None = None) -> None:
        """The guard's slow close to the stop; it ends at its next pass on cancel.

        Ended by cancel, the jaws stay unreferenced and are held where they
        are, at the hold stiffness.
        """
        guard = self._guard
        if cancel is None:
            guard.reference()
            return
        if cancel.is_set():
            return  # nothing was sent: there is nothing to hold
        driver_stopped = guard._stopped
        # As in send(): the guard asks this before each command of its ramp.
        guard._stopped = lambda: cancel.is_set() or driver_stopped()
        try:
            guard.reference()
        except GripperFault as exc:
            if not (cancel.is_set() and _REFERENCE_STOPPED in str(exc)):
                raise
            # The ramp's last command still presses toward closed.
            guard.hold_raw(float(guard._read_raw()), self._hold_kp)
        finally:
            guard._stopped = driver_stopped

    def read(self) -> float | None:
        return self._guard.position()

    def target(self) -> float | None:
        return self._guard._last_target

    def send(self, jaw: float, effort: float, cancel: threading.Event) -> None:
        """One jaw command; the guard's opening ramp ends at its next pass on cancel.

        Raises _JawsUnread when the guard could not read the jaws and sent nothing.
        """
        guard = self._guard
        driver_stopped = guard._stopped
        # The guard asks this before each ramp command. The driver's own stop
        # flag is never set: later jaw commands would vanish without an error.
        guard._stopped = lambda: cancel.is_set() or driver_stopped()
        try:
            self._backend.set_gripper(jaw, effort)
        except GripperFault as exc:
            if _NOTHING_SENT in str(exc):
                raise _JawsUnread(str(exc)) from exc
            raise
        finally:
            guard._stopped = driver_stopped

    def hold_light(self, jaw: float | None = None) -> None:
        """Hold at the last jaw target, or at `jaw`, at the hold stiffness."""
        # hold_raw never references and never ramps, so it cannot block a STOP.
        offset = self._guard.offset
        if jaw is None:
            jaw = self._guard._last_target
        if jaw is not None and offset is not None:
            self._guard.hold_raw(jaw + offset, self._hold_kp)


class _MockJaws:
    """The jaws of a backend without a gripper guard: APAI's mock arm."""

    def __init__(self, backend: Any, gripper: Mapping[str, Any]) -> None:
        self._backend = backend
        self.closed = float(gripper.get("closed_pos", 1.0))
        self.open = float(gripper.get("open_pos", 0.0))
        # The stiffness defaults are the reBot RS driver's.
        kp = float(gripper.get("kp", 6.0))
        hold_kp = float(gripper.get("hold_kp", 1.0))
        self._hold_effort = min(1.0, hold_kp / kp) if kp > 0.0 else 1.0
        self._referenced = False
        self._target: float | None = None

    def referenced(self) -> bool:
        return self._referenced

    def reference(self, cancel: threading.Event | None = None) -> None:
        if cancel is not None and cancel.is_set():
            return
        # Like the guard's reference: close, and accept only the closed stop.
        self._backend.set_gripper(self.closed, 1.0)
        short = abs(self._measure() - self.closed)
        if short > _MOCK_REFERENCE_TOL * abs(self.open - self.closed):
            raise RuntimeError("the jaws stopped short of the closed stop")
        self._target = self.closed
        self._referenced = True

    def read(self) -> float | None:
        return self._measure() if self._referenced else None

    def target(self) -> float | None:
        return self._target

    def send(self, jaw: float, effort: float, cancel: threading.Event) -> None:
        self._backend.set_gripper(jaw, effort)
        self._target = jaw

    def hold_light(self, jaw: float | None = None) -> None:
        if jaw is not None:
            self._target = jaw
        if self._target is not None:
            self._backend.set_gripper(self._target, self._hold_effort)

    def _measure(self) -> float:
        return float(self._backend.get_state().gripper_pos)


@contextlib.contextmanager
def _no_confirmation_claim() -> Iterator[None]:
    """Inside, APAI finds no operator's confirmation to claim.

    With one, APAI's driver connects an arm it does not measure at rest, and
    that connect cuts torque. APAI's hold gate looks claim_confirmation up in
    arm_hold at each call. The swap is the process's: for the one connect,
    which runs before the robot app's threads start.
    """
    claim = arm_hold.claim_confirmation
    arm_hold.claim_confirmation = lambda key: None
    try:
        yield
    finally:
        arm_hold.claim_confirmation = claim


class ApaiArm:
    """An APAI arm backend as the robot app's arm port.

    release_torque is the only path to the backend's disconnect, which cuts
    motor torque. connect() never goes ahead on an operator's confirmation.
    A disconnect that raised may have switched some motors off: torque_on()
    stays true, the driver is sent no further target, and release_torque may
    be called again.
    """

    def __init__(
        self,
        backend: Any,
        config: RobotAppConfig,
        clock: Clock,
        *,
        gripper_cfg: Any = None,
        read_timeout_s: float = 0.05,
        release_reports: bool = False,
    ) -> None:
        """gripper_cfg is the APAI profile's gripper block, a Cfg or a mapping.

        release_reports says the backend's disconnect cannot fail unnoticed;
        without it a release counts as sent, never as confirmed.
        """
        self._release_reports = bool(release_reports)
        if int(backend.n_joints) != config.n_joints:
            raise ValueError(
                f"the backend has {backend.n_joints} joints but n_joints is "
                f"{config.n_joints}"
            )
        self.n_joints = config.n_joints
        self.joint_names = list(config.joint_names)
        self._backend = backend
        self._clock = clock
        self._read_timeout_s = float(read_timeout_s)
        # None for a backend that reads only its full state, such as the mock arm.
        self._reader = backend.contact_joint_reader()
        gripper = _plain(gripper_cfg)
        guard = getattr(backend, "_grip_guard", None)
        self._jaws: _GuardJaws | _MockJaws = (
            _MockJaws(backend, gripper)
            if guard is None
            else _GuardJaws(backend, guard, gripper)
        )
        if self._jaws.open == self._jaws.closed:
            raise ValueError("the gripper's open and closed positions are equal")
        self._reach_tol = config.gripper.reach_tol
        self._connected = False
        self._released = False  # the backend's disconnect returned
        self._release_failed = False  # its last call raised
        self._last_target: tuple[list[float], float] | None = None

    @property
    def backend(self) -> Any:
        return self._backend

    # ArmPort: arm

    def connect(self) -> None:
        if self._connected:
            raise RuntimeError("the arm is connected once and never again")
        with _no_confirmation_claim():
            self._backend.connect()
        self._connected = True

    def torque_on(self) -> bool:
        return self._connected and not self._released

    def send_target(self, q: Sequence[float]) -> None:
        target = np.array(q, dtype=float)
        if target.shape != (self.n_joints,) or not np.all(np.isfinite(target)):
            raise ValueError(f"a target needs {self.n_joints} finite values, got {q!r}")
        self._require_torque()
        try:
            self._backend.send_joint_target(target)
        except Exception as exc:
            raise ArmFault(f"the driver did not take a joint target: {exc}") from exc
        self._last_target = (target.tolist(), self._clock.monotonic())

    def last_target(self) -> tuple[list[float], float] | None:
        sent = self._last_target
        return None if sent is None else (list(sent[0]), sent[1])

    def read_joint(self, joint: int) -> JointReading:
        if not 0 <= joint < self.n_joints:
            raise ValueError(f"no joint index {joint!r} in {self.n_joints}")
        t_start = self._clock.monotonic()
        try:
            if self._reader is None:
                value = float(self._backend.read_positions_for_monitor()[joint])
            else:
                value = float(self._reader(joint, self._read_timeout_s))
        except Exception as exc:
            raise ReadFailed(f"{self.joint_names[joint]}: {exc}") from exc
        t_end = self._clock.monotonic()
        if not math.isfinite(value):
            raise ReadFailed(f"{self.joint_names[joint]}: read {value!r}")
        return JointReading(joint, value, t_start, t_end)

    def read_all(self) -> list[JointReading]:
        t_start = self._clock.monotonic()
        try:
            pose = np.array(self._backend.read_positions_for_monitor(), dtype=float)
        except Exception as exc:
            raise ReadFailed(f"the joints could not be read: {exc}") from exc
        t_end = self._clock.monotonic()
        if pose.shape != (self.n_joints,) or not np.all(np.isfinite(pose)):
            raise ReadFailed(f"the joints read as {pose.tolist()}")
        return [
            JointReading(joint, float(value), t_start, t_end)
            for joint, value in enumerate(pose)
        ]

    def hold_measured(self) -> None:
        # Not the hold after a contact: that one is the move's contact watch's,
        # which sends only a reading it has checked against its own samples.
        self.send_target([reading.value for reading in self.read_all()])

    def release_torque(self) -> None:
        try:
            self._backend.disconnect()
        except Exception as exc:
            self._release_failed = True
            raise ArmFault(f"the driver's disconnect failed: {exc}") from exc
        self._released = True
        self._release_failed = False

    def release_confirmed(self) -> bool:
        return self._released and self._release_reports

    def faults(self) -> list[str]:
        # APAI's drivers have no fault read. An error a driver raises becomes
        # an ArmFault from the call that met it, but the reBot motor library
        # raises nothing when a command reaches no motor: a lost setpoint, light
        # jaw hold or disable shows only in later readings, if at all.
        return []

    def temperatures(self) -> list[float] | None:
        return None

    # ArmPort: jaws; openings run from 0 closed to 1 open

    def gripper_referenced(self) -> bool:
        return self._jaws.referenced()

    def gripper_reference(self, cancel: threading.Event | None = None) -> None:
        self._require_torque()
        try:
            self._jaws.reference(cancel)
        except Exception as exc:
            raise ArmFault(f"the jaws could not be referenced: {exc}") from exc

    def gripper_read(self) -> float | None:
        try:
            jaw = self._jaws.read()
        except Exception as exc:
            log.debug("jaw read failed: %s", exc)
            return None
        return None if jaw is None or not math.isfinite(jaw) else self._opening(jaw)

    def gripper_last_target(self) -> float | None:
        jaw = self._jaws.target()
        return None if jaw is None else self._opening(jaw)

    def gripper_command(
        self, opening: float, max_effort: float, mode: str, cancel: threading.Event
    ) -> GripperOutcome:
        refusal = self._jaw_refusal(opening, max_effort, mode)
        if refusal is not None:
            measured = self.gripper_read() if self.torque_on() else None
            return GripperOutcome("refused", measured, refusal)
        if cancel.is_set():
            return GripperOutcome("cancelled", self.gripper_read())
        for attempt in range(1, _JAW_SEND_TRIES + 1):
            try:
                # On hardware an opening blocks here through APAI's guarded ramp,
                # which ends at its next pass once cancel is set.
                self._jaws.send(self._jaw(opening), max_effort, cancel)
                break
            except _JawsUnread as exc:
                # Nothing was sent: one missed jaw answer is not a fault.
                if attempt == _JAW_SEND_TRIES or not self._jaws.referenced():
                    detail = "the jaws could not be read, so no jaw command was sent"
                    return GripperOutcome("refused", None, f"{detail}: {exc}")
                self._clock.sleep(_JAW_RETRY_S)
                if cancel.is_set():
                    return GripperOutcome("cancelled", self.gripper_read())
            except Exception as exc:
                raise ArmFault(f"the jaw command failed: {exc}") from exc
        return self._await_jaws(opening, cancel)

    def gripper_light_hold(self) -> None:
        self._hold_light(None)

    # Internals

    def _no_command(self) -> str | None:
        """Why the driver is sent no target and no jaw command; None when it is."""
        if self._release_failed:
            return "a torque release failed: the motors may be on or off"
        if not self.torque_on():
            return "the arm is not connected or its motors are off"
        return None

    def _require_torque(self) -> None:
        reason = self._no_command()
        if reason is not None:
            raise ArmFault(reason)

    def _opening(self, jaw: float) -> float:
        opening = (jaw - self._jaws.closed) / (self._jaws.open - self._jaws.closed)
        return opening + 0.0  # 0.0 for a closed jaw, never -0.0

    def _jaw(self, opening: float) -> float:
        return self._jaws.closed + opening * (self._jaws.open - self._jaws.closed)

    def _hold_light(self, opening: float | None) -> None:
        """The light hold at the jaw target in force, or at this opening."""
        if self._no_command() is not None:
            return
        try:
            self._jaws.hold_light(None if opening is None else self._jaw(opening))
        except Exception as exc:
            raise ArmFault(f"the light jaw hold could not be sent: {exc}") from exc

    def _cancelled(self, measured: float | None) -> GripperOutcome:
        """End a cancelled command in a light hold that opens the jaws no further."""
        target = self.gripper_last_target()
        opening = target is not None and measured is not None and measured < target
        # An opening is held where the jaws are; a close keeps its target.
        self._hold_light(measured if opening else None)
        return GripperOutcome("cancelled", measured)

    def _jaw_refusal(self, opening: float, max_effort: float, mode: str) -> str | None:
        reason = self._no_command()
        if reason is not None:
            return reason
        if not self._jaws.referenced():
            return "the jaws are not referenced"
        for name, value in (("opening", opening), ("max_effort", max_effort)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return f"{name} must be a number, got {value!r}"
            if not 0.0 <= value <= 1.0:
                return f"{name} must be between 0 and 1, got {value!r}"
        if mode not in JAW_MODES:
            return f"unknown mode {mode!r}"
        return None

    def _await_jaws(self, opening: float, cancel: threading.Event) -> GripperOutcome:
        """Poll the jaws until they reach the opening, stall, time out or are cancelled.

        A close that ends short of its target is left in a light hold. The wait
        is counted in polls, so a fake clock advanced elsewhere cannot time it out.
        """
        grace = math.ceil(_JAW_GRACE_S / _JAW_POLL_S - _EPS)
        limit = math.ceil(_JAW_TIMEOUT_S / _JAW_POLL_S - _EPS)
        before: float | None = None
        for poll in range(limit + 1):
            if poll:
                self._clock.sleep(_JAW_POLL_S)
            measured = self.gripper_read()
            if cancel.is_set():
                return self._cancelled(measured)
            if measured is None:
                continue
            if abs(measured - opening) <= self._reach_tol:
                return GripperOutcome("reached", measured)
            if poll >= grace:
                if before is not None and abs(measured - before) < _JAW_STALL:
                    if measured > opening:
                        self.gripper_light_hold()
                    return GripperOutcome("stalled", measured)
                before = measured
        measured = self.gripper_read()
        # Without a reading only a command to full open is known not to be a close.
        short = opening < 1.0 if measured is None else measured > opening
        if short:
            self.gripper_light_hold()
        return GripperOutcome("timeout", measured)
