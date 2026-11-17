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
import importlib.util
import io
import logging
import math
import os
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml
from .clock import Clock, FakeClock, SystemClock
from .config import GraspConfig, LimitSet, RobotAppConfig
from .gripper import (
    CANCELLED,
    CLOSE,
    CLOSE_TORQUE,
    CLOSE_UNTIL_CONTACT,
    GRIPPER_FAULT,
    OPEN,
    PLAIN,
    REACHED,
    REFUSED,
    SENT,
    STALLED,
    TIMEOUT,
    DriverOutcome,
)
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

# What ApaiArm.hold_evidence() says of the motors.
from .text import MOTORS_HOLDING, MOTORS_NOT_HOLDING, MOTORS_UNKNOWN
from apai.config import Cfg, load_demo_config, load_profile
from apai.control import arm_hold, rs_gripper
from apai.control.arm_base import ContactFeedback, make_arm
from apai.control.kinematics import Kinematics
from apai.control.robstride import (
    GRIP_GRACE_S,
    GRIP_POLL_S,
    GRIP_REACH_RAD,
    GRIP_STALL_RAD,
)
from apai.grasping.force import select_profile
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

# Each jaw mode is one routine of APAI's driver: PLAIN its set_gripper, a
# CLOSE_UNTIL_CONTACT with a scout opening its close_gripper_two_stage, OPEN
# its open_gripper, CLOSE_TORQUE its close_gripper_torque, and CLOSE its
# two-stage close with the grip profile of APAI's close_gripper skill.
JAW_MODES = (PLAIN, CLOSE_UNTIL_CONTACT, OPEN, CLOSE_TORQUE, CLOSE)
ARM_TYPES = ("mock", "rebot_rs")  # the APAI backends ApaiArm is written for
# The experiment launcher's files (thor_runtime.py), all in its configs
# folder: its settings, which name the arm profile it runs, and the arm
# profile it reads its joint speed from. And the APAI arm profile whose joint
# speed cap that speed may not exceed.
LAUNCHER_SETTINGS = "thor_runtime.yaml"
SPEED_SOURCE = "rebot_workstation"
SPEED_CEILING = "rebot_rs"
# The experiment's module that holds its motion bounds, and the label its
# launcher gives them.
BOUNDS_MODULE = "thor_calibration_rig"
BOUNDS_LABEL = "operation"
# The pause between two looks at the motors' fault reports, in s.
_FAULT_POLL_S = 0.5

_EPS = 1e-9
# The harness keeps the text of every refusal; the checker caps the list at this.
_VIOLATIONS_KEPT = 256
# A close until contact without a scout opening has no routine in APAI's
# driver: the jaws of a person's trigger, and the mock arm. Its wait ends when
# the jaws stop, and at the latest after this.
_CONTACT_WAIT_S = 4.0
_MOCK_STALL = 0.005  # the mock arm's jaws: fraction of travel between two polls
# How APAI's gripper guard ends the fault it raises before any jaw command.
_NOTHING_SENT = "no gripper command sent"
# Part of the fault the guard raises when its stop probe ends a reference.
_REFERENCE_STOPPED = "referencing interrupted by a stop"
# Parts of the guard's faults that leave the arm usable and the jaws held where
# they are: the outcome of the jaw command, not a fault-stop. Nothing was sent
# to jaws it could not read, an opening was blocked, and its own reference
# found something between the jaws.
_JAW_FAULTS = (_NOTHING_SENT, "blocked while opening", "gripper reference failed")
# "Jaws proven empty": a reading lies within this of the last jaw target, rad.
# It is how near its target APAI's driver takes a close stage as reached.
JAW_TARGET_TOL_RAD = GRIP_REACH_RAD
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
    # The folder of the experiment whose launcher runs this arm profile; None
    # for any other profile. See launcher_folder().
    launcher_dir: Path | None = None


def launcher_folder(arm: str) -> Path | None:
    """The experiment folder whose launcher runs this arm profile, else None.

    The experiment is the on-device agent this example was split from. Its
    launcher (thor_runtime.py) reads LAUNCHER_SETTINGS beside the arm profile
    and runs the profile that file names as arm_profile. A name in APAI's
    configs/arms, and a path no such file names, give None.
    """
    path = Path(arm)
    if len(path.parts) < 2:
        return None  # a name in APAI's configs/arms
    profile = path.resolve()
    root = profile.parent.parent
    try:
        named = yaml.safe_load((profile.parent / LAUNCHER_SETTINGS).read_text())
        runs = (root / str(named["arm_profile"])).resolve().with_suffix("")
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError):
        return None
    return root if runs == profile.with_suffix("") else None


def launcher_speed(folder: Path) -> float:
    """The joint speed the experiment's launcher runs its arm at, rad/s.

    Read as its load_joint_speed() reads its speed profile "reference", the
    one it runs without --speed-profile: safety.max_joint_vel of the arm
    profile SPEED_SOURCE in its configs folder, which may not exceed that of
    APAI's SPEED_CEILING profile. ValueError otherwise.
    """
    source = folder / "configs" / SPEED_SOURCE
    try:
        ceiling = float(
            load_profile("arms", SPEED_CEILING).overrides.safety.max_joint_vel
        )
        raw = load_profile("arms", str(source)).overrides.safety.max_joint_vel
        speed = float(raw)
    except Exception as exc:
        raise ValueError(
            f"the experiment launcher's joint speed cannot be read from "
            f"{source}.yaml: {type(exc).__name__}: {exc}"
        ) from exc
    if isinstance(raw, bool) or not math.isfinite(speed) or not 0 < speed <= ceiling:
        raise ValueError(
            f"the experiment launcher's joint speed in {source}.yaml "
            f"(overrides.safety.max_joint_vel) is {raw!r}: it must be above 0 and "
            f"at most {ceiling:g} rad/s, the cap of APAI's {SPEED_CEILING} profile"
        )
    return speed


def load_apai_profile(
    arm: str = "mock", overrides: Mapping[str, Any] | None = None
) -> ApaiProfile:
    """Load an APAI arm profile with no camera, no model and no obstacle map.

    arm names a file in APAI's configs/arms, or is a path without ".yaml".
    The arm profile the experiment's launcher runs gets the joint speed cap
    that launcher gives it (launcher_speed), as the launcher writes it over
    the profile's own. overrides is then merged over the arm's view of APAI's
    configuration.
    """
    for name in ("APAI_ROBOT", "APAI_BASE"):
        if os.environ.get(name):
            raise ValueError(f"unset {name}: it selects another APAI configuration")
    view = load_demo_config(camera="mock", arm=arm, llm="mock").arms[0]["resolved"]
    folder = launcher_folder(arm)
    if folder is not None:
        speed = launcher_speed(folder)
        log.info(
            "the joint speed cap of %s is %g rad/s, the speed of the experiment's "
            "launcher (%s/configs/%s.yaml)",
            arm,
            speed,
            folder,
            SPEED_SOURCE,
        )
        view["safety"]["max_joint_vel"] = speed
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
    return ApaiProfile(
        apai_cfg=Cfg(view), arm_cfg=arm_cfg, kinematics=kinematics, launcher_dir=folder
    )


def config_for_profile(config: RobotAppConfig, profile: ApaiProfile) -> RobotAppConfig:
    """The configuration with the tolerances an APAI profile decides.

    rest.start_rest_tol_rad becomes the profile's rest check tolerance
    (rest_check_tol_rad, APAI's default without one), never less than
    rest.rest_tol_rad. Of a real arm's jaw travel, gripper.target_tol becomes
    at most JAW_TARGET_TOL_RAD and gripper.full_open_tol how far short of its
    target APAI's guarded opening may end (rs_gripper.OPEN_SHORT_RAD).
    gripper.closed_frac becomes the opening below which APAI takes jaws as
    closed on nothing (grasp.air_grasp_frac), where the profile names it.
    """
    tolerance = max(arm_hold.rest_tolerance(profile.arm_cfg), config.rest.rest_tol_rad)
    rest = dataclasses.replace(config.rest, start_rest_tol_rad=tolerance)
    gripper = config.gripper
    travel = jaw_travel_rad(profile)
    if travel is not None:
        gripper = dataclasses.replace(
            gripper,
            target_tol=min(gripper.target_tol, JAW_TARGET_TOL_RAD / travel),
            full_open_tol=min(1.0, rs_gripper.OPEN_SHORT_RAD / travel),
        )
    closed = _plain(profile.apai_cfg.get("grasp")).get("air_grasp_frac")
    if closed is not None:
        gripper = dataclasses.replace(gripper, closed_frac=float(closed))
    config = dataclasses.replace(config, rest=rest, gripper=gripper)
    return _motion_for_profile(config, profile)


def _motion_for_profile(config: RobotAppConfig, profile: ApaiProfile) -> RobotAppConfig:
    """The configuration with the settle rule of the driver APAI builds.

    executor.settle_tol_rad and executor.settle_timeout_s become the driver's
    own settle tolerance and settle time (driver_settle).
    """
    settle = driver_settle(profile)
    if settle is None:
        return config
    executor = dataclasses.replace(
        config.executor, settle_tol_rad=settle[0], settle_timeout_s=settle[1]
    )
    return dataclasses.replace(config, executor=executor)


def driver_settle(profile: ApaiProfile) -> tuple[float, float] | None:
    """(settle tolerance in rad, settle time in s) of the profile's APAI driver.

    Read from a driver object APAI builds for the profile, which is not
    connected: building one opens no motor bus. None for an arm type ApaiArm
    is not written for.
    """
    if profile.arm_cfg.get("type") not in ARM_TYPES:
        return None
    driver = make_arm(profile.arm_cfg, kinematics=profile.kinematics)
    return float(driver.settle_tol), float(driver.settle_timeout_s)


def move_pause_limit(config: RobotAppConfig, profile: ApaiProfile) -> float:
    """The longest pause between two setpoints APAI's own stream survives, in s.

    APAI's stream ends a move when a joint has no good reading for the contact
    monitor's max_feedback_gap_s. A joint is read once in n_joints /
    reads_per_step periods, so a pause that long less one such round ends
    the move there. executor.large_delay_s when that is no more than
    executor.small_delay_s.
    """
    executor = config.executor
    round_s = math.ceil(config.n_joints / executor.reads_per_step) * executor.period_s
    pause = _contact_config(profile).max_feedback_gap_s - round_s
    return pause if pause > executor.small_delay_s else executor.large_delay_s


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


def human_speed_cap(profile: ApaiProfile) -> float:
    """The joint speed a human limit set may not exceed, rad/s.

    The profile's own cap. The arm profile the experiment's launcher runs has
    that launcher's speed for its cap, which is the agent's: there a person's
    bound is the cap of APAI's SPEED_CEILING profile, the bound the launcher
    holds its own speed to. ValueError when that profile cannot be read.
    """
    if profile.launcher_dir is None:
        return speed_cap(profile)
    try:
        raw = load_profile("arms", SPEED_CEILING).overrides.safety.max_joint_vel
        bound = float(raw)
    except Exception as exc:
        raise ValueError(
            f"the joint speed cap of APAI's {SPEED_CEILING} profile cannot be "
            f"read: {type(exc).__name__}: {exc}"
        ) from exc
    if isinstance(raw, bool) or not (math.isfinite(bound) and bound > 0.0):
        raise ValueError(
            f"the joint speed cap of APAI's {SPEED_CEILING} profile is {raw!r}"
        )
    return bound


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

    Speed comes from the owner's limit set. The agent's and the recovery set
    may not exceed the profile's speed cap, the human set not
    human_speed_cap() (ValueError); the rest of the harness limits come from
    the APAI profile. No harness gets an obstacle map. paced is passed to
    ApaiChecker.
    """
    kinematics = profile.kinematics
    if kinematics.n != config.n_joints:
        raise ValueError(
            f"the APAI profile has {kinematics.n} joints but n_joints is "
            f"{config.n_joints}"
        )
    base = SafetyLimits.from_config(profile.apai_cfg.safety)
    for limit_set in (config.limits.agent, config.limits.recovery):
        # Nothing is clamped to the profile's cap: a faster set is refused.
        if limit_set.max_joint_speed > base.max_joint_vel + _EPS:
            raise ValueError(
                f"limits.{limit_set.name}.max_joint_speed "
                f"{limit_set.max_joint_speed:g} rad/s is above the APAI profile's "
                f"speed cap of {base.max_joint_vel:g} rad/s (safety.max_joint_vel)"
            )
    human, bound = config.limits.human, human_speed_cap(profile)
    if human.max_joint_speed > bound + _EPS:
        whose = "the APAI profile's speed cap"
        if profile.launcher_dir is not None:
            whose = f"the speed cap of APAI's {SPEED_CEILING} profile"
        raise ValueError(
            f"limits.human.max_joint_speed {human.max_joint_speed:g} rad/s is "
            f"above {whose} of {bound:g} rad/s (safety.max_joint_vel)"
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

    def checker(
        limit_set: LimitSet,
        fold_onto: Sequence[float] | None = None,
        **changes: float,
    ) -> ApaiChecker:
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
            fold_onto=fold_onto,
        )

    # The agent's rest tool folds the arm onto the rest pose, which lies on
    # joint stops: its setpoints may enter the joint margin there, and only
    # there. Every other agent move keeps the margin.
    agent = checker(config.limits.agent, fold_onto=config.rest.rest_q)
    _restrict_as_launcher(agent.harness, profile)
    return CheckerSet(
        agent=agent,
        human=checker(config.limits.human),
        # Nothing can feed the recovery watchdog, and the rest pose lies on joint
        # stops, inside the joint margin the other two keep.
        recovery=checker(config.limits.recovery, watchdog_s=math.inf, joint_margin=0.0),
    )


def _restrict_as_launcher(harness: SafetyHarness, profile: ApaiProfile) -> None:
    """Give the agent's harness the motion bounds the experiment's launcher gives.

    The launcher calls restrict_motion() of its BOUNDS_MODULE on its one
    harness with the arm profile's motion_bounds: every pose of an agent move
    stays within a joint distance of a centre and the gripper tip within a
    radius of a point. The workstation checker has the same bounds, so this
    refuses nothing that checker passed. Nothing is added for a profile the
    launcher does not run, or without motion_bounds. The human and recovery
    harnesses keep their own limits: their paths are not the launcher's.
    """
    bounds = profile.arm_cfg.get("motion_bounds")
    if profile.launcher_dir is None or bounds is None:
        return
    source = profile.launcher_dir / f"{BOUNDS_MODULE}.py"
    spec = importlib.util.spec_from_file_location(
        f"_agentic_remote_{BOUNDS_MODULE}", source
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"the experiment's motion bounds cannot be loaded: {source}")
    module = importlib.util.module_from_spec(spec)
    said = io.StringIO()
    try:
        spec.loader.exec_module(module)
        # The routine prints one line; it goes to the log.
        with contextlib.redirect_stdout(said):
            module.restrict_motion(harness, _plain(bounds), label=BOUNDS_LABEL)
    except Exception as exc:
        raise ValueError(
            f"the experiment's motion bounds ({source}) could not be applied: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    log.info("%s", said.getvalue().strip() or "the experiment's motion bounds apply")


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
        jaws, own = config.gripper, config_for_profile(config, profile).gripper
        if (jaws.full_open_tol, jaws.closed_frac) != (
            own.full_open_tol,
            own.closed_frac,
        ):
            # "Open" and "closed" are what APAI's driver and skills take them as.
            raise ValueError(
                f"gripper.full_open_tol {jaws.full_open_tol:g} and "
                f"gripper.closed_frac {jaws.closed_frac:g} differ from APAI's "
                f"{own.full_open_tol:g} and {own.closed_frac:g}: use "
                "config_for_profile()"
            )
    backend = make_arm(arm_cfg, kinematics=profile.kinematics)
    executor = config.executor
    settle = (float(backend.settle_tol), float(backend.settle_timeout_s))
    ours = (executor.settle_tol_rad, executor.settle_timeout_s)
    if kind != "mock" and ours != settle:
        # A real arm settles as its own driver settles it.
        raise ValueError(
            f"executor.settle_tol_rad {executor.settle_tol_rad:g} and "
            f"executor.settle_timeout_s {executor.settle_timeout_s:g} differ from "
            f"the settle tolerance {settle[0]:g} rad and settle time {settle[1]:g} s "
            "of APAI's driver: use config_for_profile()"
        )
    contact = _contact_config(profile)
    return ApaiArm(
        backend,
        config,
        clock,
        gripper_cfg=arm_cfg.get("gripper"),
        read_timeout_s=contact.read_timeout_s,
        # A joint that holds stays inside the band the contact monitor gives
        # a joint that follows its command.
        hold_tol_rad=contact.tolerances(config.n_joints).tolist(),
        hold_samples=contact.debounce_samples,
        pause_limit_s=move_pause_limit(config, profile),
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
        fold_onto: Sequence[float] | None = None,
    ) -> None:
        """paced False counts a move's time in steps, for a clock that is not real.

        None picks it from the clock: in steps on a FakeClock, else real pacing.
        grasp bounds the grasp region of a move; None gives the default bounds.
        fold_onto is the rest pose a setpoint may fold onto inside the
        harness's joint margin (see _fold_margin); None: no setpoint may.
        """
        self._fold_onto = (
            None if fold_onto is None else np.array(fold_onto, dtype=float)
        )
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
        margin = self._fold_margin(start, target)
        try:
            if margin is None:
                self._harness.approve(start, target, step_s)
            else:
                self._harness.approve(start, target, step_s, joint_margin=margin)
        except SafetyViolation as exc:
            self._trim_violations()
            raise _refusal(exc) from exc

    def _fold_margin(self, start: np.ndarray, target: np.ndarray) -> float | None:
        """The joint margin a setpoint onto the rest pose is checked with, else None.

        The joint margin is the band the harness keeps clear of each joint
        stop. The last fold step of APAI's rest tool drives onto the rest
        pose with the margin at 0, as its harness allows for that step. 0.0
        for a setpoint that enters the margin only at joints whose rest value
        lies in it, goes no further than that value and moves no joint away
        from the rest pose. None, the harness's own margin, for every other
        setpoint. The check ahead has the same rule (agent/checkahead.py).
        """
        rest = self._fold_onto
        if rest is None or rest.shape != target.shape:
            return None
        limits = self._kinematics.joint_limits
        low, high = (np.asarray(side, dtype=float) for side in limits)
        margin = float(self._harness.limits.joint_margin)
        under = target < low + margin - _EPS
        over = target > high - margin + _EPS
        if not (under.any() or over.any()):
            return None
        onto = np.where(
            under,
            (rest < low + margin) & (target >= rest - _EPS),
            (rest > high - margin) & (target <= rest + _EPS),
        )
        nearer = np.abs(target - rest) <= np.abs(start - rest) + _EPS
        return 0.0 if onto[under | over].all() and nearer.all() else None

    def check_pose(self, q: Sequence[float]) -> None:
        self._check_pose(q, low_tip=False)

    def check_low_pose(self, q: Sequence[float]) -> None:
        self._check_pose(q, low_tip=True)

    def check_start_pose(self, q: Sequence[float]) -> None:
        pose = self._pose(q)
        if self._harness.estopped:
            raise Refusal("e-stop latched")
        self._refuse(self._box_problem(pose, pose))
        # The arm is already here. APAI checks no pose when a task starts: its
        # step check judges the first setpoint from where the arm is, and lets
        # a gripper tip under the table clearance rise and one outside the
        # workspace box come back. So a hold at this pose is asked of that
        # step check, at the nearest pose the harness's joint margin accepts,
        # as check_pose() does for the rest pose on its joint stops.
        inside = self._kinematics.clamp(pose, self._harness.limits.joint_margin)
        self._refuse(self._harness.vet_step(inside, inside, self._period_s))

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


class _ReferenceStopped(Exception):
    """A cancel ended the reference the guard began before a jaw command."""


class _GuardJaws:
    """The jaws of a reBot RS backend: its driver's own routines (jaw angles, rad).

    Every routine runs with the guard's stop probe also asking the command's
    cancel, so a STOP ends a ramp at its next pass and keeps the next stage
    of a close from being sent. The driver's own stop flag is never set:
    later jaw commands would vanish without an error.
    """

    # The guard references jaws that have no reference before a command: it
    # closes them slowly to their stop.
    references_itself = True

    def __init__(self, backend: Any, guard: Any, gripper: Mapping[str, Any]) -> None:
        if not callable(getattr(guard, "_stopped", None)):
            raise ValueError("the gripper guard has no stop probe to end a ramp with")
        self._backend = backend
        self._guard = guard
        self._hold_kp = float(gripper.get("hold_kp", 1.0))  # the driver's default
        self.closed = float(guard.closed_pos)
        self.open = float(guard.open_pos)
        self.travel_rad = abs(self.open - self.closed)

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
        with contextlib.suppress(_ReferenceStopped), self._cancellable(cancel):
            guard.reference()

    def read(self) -> float | None:
        return self._guard.position()

    def target(self) -> float | None:
        return self._guard._last_target

    @contextlib.contextmanager
    def _cancellable(self, cancel: threading.Event) -> Iterator[None]:
        """Inside, the guard's stop probe asks `cancel` too.

        Raises _ReferenceStopped when cancel ended a reference the guard
        began: the jaws stay unreferenced, held where they are.
        """
        guard = self._guard
        driver_stopped = guard._stopped
        guard._stopped = lambda: cancel.is_set() or driver_stopped()
        try:
            yield
        except GripperFault as exc:
            if not (cancel.is_set() and _REFERENCE_STOPPED in str(exc)):
                raise
            # The ramp's last command still presses toward closed.
            guard.hold_raw(float(guard._read_raw()), self._hold_kp)
            raise _ReferenceStopped(str(exc)) from exc
        finally:
            guard._stopped = driver_stopped

    def send(self, jaw: float, effort: float, cancel: threading.Event) -> None:
        """The driver's set_gripper: an opening blocks through its guarded ramp."""
        with self._cancellable(cancel):
            self._backend.set_gripper(jaw, effort)

    def open_jaws(self, effort: float, cancel: threading.Event) -> None:
        """The driver's open_gripper, at the effort the driver gives it."""
        with self._cancellable(cancel):
            self._backend.open_gripper()

    def close_two_stage(
        self, stage1: float, stage2: float, effort: float, cancel: threading.Event
    ) -> tuple[float | None, int | None]:
        """The driver's close_gripper_two_stage.

        Returns the jaw position it ended at, None when it could not be read,
        and how many stages the driver skipped.
        """
        with self._cancellable(cancel):
            end = self._backend.close_gripper_two_stage(
                width_frac_stage1=stage1, width_frac_stage2=stage2, effort=effort
            )
        skipped = getattr(self._backend, "close_stages_skipped", None)
        return end, skipped if type(skipped) is int else None

    def close_torque(
        self, effort: float | None, cancel: threading.Event
    ) -> tuple[bool, float | None]:
        """The driver's close_gripper_torque; effort None is the driver's own.

        Returns what it returned, and where it noted contact, None for none.
        """
        with self._cancellable(cancel):
            if effort is None:
                closed = self._backend.close_gripper_torque()
            else:
                closed = self._backend.close_gripper_torque(effort=effort)
        return bool(closed), getattr(self._backend, "_grip_contact_pos", None)

    def hold_light(self, jaw: float | None = None) -> None:
        """Hold at the last jaw target, or at `jaw`, at the hold stiffness."""
        # hold_raw never references and never ramps, so it cannot block a STOP.
        offset = self._guard.offset
        if jaw is None:
            jaw = self._guard._last_target
        if jaw is not None and offset is not None:
            self._guard.hold_raw(jaw + offset, self._hold_kp)


class _MockJaws:
    """The jaws of a backend without a gripper guard: APAI's mock arm.

    Its driver has one jaw routine, set_gripper: an opening is that command to
    the open position, and a close is one command followed by the port's wait.
    """

    references_itself = False
    travel_rad = None  # its jaw position is a fraction, not an angle

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

    def open_jaws(self, effort: float, cancel: threading.Event) -> None:
        self.send(self.open, effort, cancel)

    def hold_light(self, jaw: float | None = None) -> None:
        if jaw is not None:
            self._target = jaw
        if self._target is not None:
            self._backend.set_gripper(self._target, self._hold_effort)

    def _measure(self) -> float:
        return float(self._backend.get_state().gripper_pos)


class _HoldWatch:
    """Whether each joint stays at the target it was last told to hold.

    A motor that is switched off, or ignores its commands, lets its joint
    leave that target: the measured pose is the evidence, as for APAI's own
    connect. A joint is off its hold when `samples` readings in a row lie
    more than its tolerance from the target, each begun at least `after_s`
    after the target was sent: the time a move has to settle. A joint that
    never moves when its motor goes off, such as one that lies on its stop,
    is not found this way. Thread-safe.
    """

    def __init__(
        self,
        names: Sequence[str],
        tol_rad: Sequence[float],
        samples: int,
        after_s: float,
    ) -> None:
        if len(tol_rad) != len(names):
            raise ValueError(f"hold_tol_rad needs {len(names)} values, got {tol_rad!r}")
        if samples < 1:
            raise ValueError(f"hold_samples must be 1 or more, got {samples!r}")
        self._names = list(names)
        self._tol = [float(value) for value in tol_rad]
        self._samples = int(samples)
        self._after_s = float(after_s)
        self._lock = threading.Lock()
        self._off = [0] * len(names)  # readings in a row beyond the tolerance
        self._last: list[tuple[float, float] | None] = [None] * len(names)
        self._at: list[float | None] = [None] * len(names)  # newest one inside

    def target_sent(self) -> None:
        """A new target stands: what was read against the old one counts no more."""
        with self._lock:
            self._off = [0] * len(self._off)
            self._at = [None] * len(self._at)

    def note(
        self, reading: JointReading, target: tuple[list[float], float] | None
    ) -> None:
        """Take one joint reading; `target` is the target in force and its send time."""
        if target is None or not reading.valid:
            return
        q, sent_t = target
        if reading.t_start < sent_t + self._after_s:
            return  # the joint may still be on its way
        joint = reading.joint
        away = abs(reading.value - q[joint])
        with self._lock:
            if away > self._tol[joint]:
                self._off[joint] += 1
                self._last[joint] = (reading.value, away)
            else:
                self._off[joint] = 0
                self._at[joint] = reading.t_end

    def problem(self) -> str | None:
        """Which joint is off its hold, and by how much; None when none is."""
        with self._lock:
            for joint, count in enumerate(self._off):
                if count >= self._samples and self._last[joint] is not None:
                    value, away = self._last[joint]
                    return (
                        f"the motors do not hold the arm: {self._names[joint]} "
                        f"reads {value:.3f} rad, {away:.3f} rad from the target it "
                        f"was told to hold (limit {self._tol[joint]:g} rad): it is "
                        "pushed, blocked, or its motor is off"
                    )
        return None

    def evidence(self, target: tuple[list[float], float] | None) -> tuple[str, str]:
        """What the readings say of the hold: (state, why)."""
        problem = self.problem()
        if problem is not None:
            return MOTORS_NOT_HOLDING, problem
        if target is None:
            return MOTORS_UNKNOWN, "no joint target was sent since the connect"
        with self._lock:
            unread = [name for name, at in zip(self._names, self._at) if at is None]
        if unread:
            return MOTORS_UNKNOWN, (
                f"{', '.join(unread)} not read at the target in force yet"
            )
        return MOTORS_HOLDING, "every joint reads at the target it holds"


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
    From the start of that call the driver is sent no target and no jaw
    command. A disconnect that raised may have switched some motors off:
    torque_on() stays true and the driver is sent nothing further. Only a
    driver whose disconnect reports (release_reports) is released a second
    time: the reBot driver lets go of its motors when its disconnect raises,
    so its next disconnect returns without reaching one.
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
        hold_tol_rad: Sequence[float] | None = None,
        hold_samples: int = 2,
        pause_limit_s: float | None = None,
    ) -> None:
        """gripper_cfg is the APAI profile's gripper block, a Cfg or a mapping.

        release_reports says the backend's disconnect cannot fail unnoticed;
        without it a release counts as sent, never as confirmed. hold_tol_rad
        is, per joint, how far a reading may lie from the target the joint
        holds, and hold_samples how many readings in a row beyond it say that
        the joint does not hold; None watches no hold. pause_limit_s is
        move_pause_limit_s.
        """
        # How often the driver itself reads the joints before it gives up on a
        # move's start; the executor reads a start pose that often.
        self.start_read_tries = max(1, int(getattr(backend, "MAX_READ_FAILURES", 1)))
        # The longest a setpoint of a move may wait, for the loop or for a
        # permit; None leaves executor.large_delay_s.
        self.move_pause_limit_s = pause_limit_s
        self._hold = (
            None
            if hold_tol_rad is None
            else _HoldWatch(
                list(config.joint_names),
                hold_tol_rad,
                hold_samples,
                config.executor.settle_timeout_s,
            )
        )
        self._motor_faults: list[str] = []
        self._fault_baseline: dict[str, int] = {}
        self._fault_poll_t = -math.inf
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
        # For the wait of a close until contact without a scout opening: how
        # near its target the jaws have reached it, and how little they move
        # between two looks once they have stopped. APAI's own numbers on a
        # real arm, as fractions of its jaw travel.
        travel = self._jaws.travel_rad
        self._reach_tol = (
            config.gripper.reach_tol if travel is None else GRIP_REACH_RAD / travel
        )
        self._stall_tol = _MOCK_STALL if travel is None else GRIP_STALL_RAD / travel
        self._connected = False
        self._releasing = False  # the backend's disconnect is under way
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
        # What each motor reports as its fault right after the connect cleared
        # them: only a later change to a fault counts.
        self._fault_baseline = dict(self._fault_reports())
        # A connect leaves no joint target in force: nothing is held yet.
        self._last_target = None
        if self._hold is not None:
            self._hold.target_sent()

    def torque_on(self) -> bool:
        # Whether this program switched the motors on and has not released
        # them. It is not a reading of the motors: hold_evidence() is.
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
        if self._hold is not None:
            self._hold.target_sent()

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
        reading = JointReading(joint, value, t_start, t_end)
        self._note_readings([reading])
        return reading

    def read_all(self) -> list[JointReading]:
        t_start = self._clock.monotonic()
        try:
            pose = np.array(self._backend.read_positions_for_monitor(), dtype=float)
        except Exception as exc:
            raise ReadFailed(f"the joints could not be read: {exc}") from exc
        t_end = self._clock.monotonic()
        if pose.shape != (self.n_joints,) or not np.all(np.isfinite(pose)):
            raise ReadFailed(f"the joints read as {pose.tolist()}")
        readings = [
            JointReading(joint, float(value), t_start, t_end)
            for joint, value in enumerate(pose)
        ]
        self._note_readings(readings)
        return readings

    def hold_measured(self) -> None:
        # Not the hold after a contact: that one is the move's contact watch's,
        # which sends only a reading it has checked against its own samples.
        self.send_target([reading.value for reading in self.read_all()])

    def release_torque(self) -> None:
        if self._release_failed and not self._release_reports:
            # A return of this driver's next disconnect would prove nothing.
            raise ArmFault(
                "an earlier disconnect failed: the motors may be on or off, and "
                "this driver takes no second release"
            )
        self._releasing = True
        try:
            self._backend.disconnect()
        except Exception as exc:
            self._release_failed = True
            raise ArmFault(f"the driver's disconnect failed: {exc}") from exc
        else:
            self._released = True
            self._release_failed = False
        finally:
            # Last: a STOP's light jaw hold never finds a gap to go out in.
            self._releasing = False

    def release_confirmed(self) -> bool:
        return self._released and self._release_reports

    def faults(self) -> list[str]:
        """Why the motors do not hold the arm, as far as it can be read.

        The reBot motor bus has no read of a motor's enable state, and the
        motor library raises nothing when a command reaches no motor. What
        can be read: a joint that leaves the target it was told to hold
        (_HoldWatch), and a motor whose fault report turned to a fault since
        the connect: such a motor ignores its commands. Nothing is read here:
        the joint readings are those the app takes anyway.
        """
        if not self.torque_on() or self._release_failed or self._releasing:
            # Nothing holds, nothing is known, or the motors are being
            # switched off on purpose.
            return []
        found = self._poll_motor_faults()
        problem = None if self._hold is None else self._hold.problem()
        return found if problem is None else [problem, *found]

    def hold_evidence(self) -> tuple[str, str]:
        """(MOTORS_HOLDING, MOTORS_NOT_HOLDING or MOTORS_UNKNOWN, and why).

        What the joint readings and the motors' fault reports say about the
        hold. Holding is said only of readings: each joint's newest one lies
        at the target it holds. Unknown also for an arm no hold is watched on.
        """
        if not self.torque_on():
            return MOTORS_NOT_HOLDING, "the motors are off"
        if self._release_failed:
            return MOTORS_UNKNOWN, "a torque release failed"
        faults = self.faults()
        if faults:
            return MOTORS_NOT_HOLDING, "; ".join(faults)
        if self._hold is None:
            return MOTORS_UNKNOWN, "this arm's hold is not watched"
        return self._hold.evidence(self._last_target)

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

    # What the gripper worker asks before it gives a command of another mode
    # than "plain" and "close_until_contact".
    jaw_modes = JAW_MODES

    def gripper_command(
        self, opening: float, max_effort: float, mode: str, cancel: threading.Event
    ) -> GripperOutcome:
        """One routine of APAI's driver, by the mode; its end is the driver's.

        "plain" is set_gripper: an opening blocks through the guarded ramp and
        ends "reached" where the driver takes it as done; a close or a hold
        ends "sent" at once, at the stiffness of its effort. A fault of the
        driver's that leaves the arm usable ends "gripper_fault" with its
        text. No command is followed by a light hold but a cancelled one.
        """
        refusal = self._jaw_refusal(opening, max_effort, mode)
        if refusal is not None:
            measured = self.gripper_read() if self.torque_on() else None
            return GripperOutcome(REFUSED, measured, refusal)
        if cancel.is_set():
            return GripperOutcome(CANCELLED, self.gripper_read())
        jaws = self._jaws
        if mode == CLOSE:
            grip = select_profile("", "rigid")  # as APAI's close_gripper skill
            return self._two_stage(
                grip.close_frac_stage1, grip.close_frac_stage2, grip.effort, cancel
            )
        if mode == CLOSE_TORQUE:
            return self._close_torque(max_effort, cancel)
        if mode == OPEN:
            return self._opened(1.0, lambda: jaws.open_jaws(max_effort, cancel), cancel)
        target = self._jaw(opening)
        if mode == PLAIN:
            return self._opened(
                opening, lambda: jaws.send(target, max_effort, cancel), cancel
            )
        return self._contact_close(opening, max_effort, cancel)

    def gripper_close_two_stage(
        self,
        scout_opening: float,
        opening: float,
        max_effort: float,
        cancel: threading.Event,
    ) -> GripperOutcome:
        """APAI's two-stage close: the scout stage, then the seat stage to `opening`.

        The driver decides when a stage has ended and which it skips. It ends
        "reached" within the driver's reach of the seat stage, else "stalled":
        on an object, or at a stage's deadline. The jaws are left pushing at
        the seat stage with the close's stiffness.
        """
        refusal = self._jaw_refusal(opening, max_effort, CLOSE_UNTIL_CONTACT)
        refusal = refusal or _fraction_problem("scout_opening", scout_opening)
        if refusal is not None:
            measured = self.gripper_read() if self.torque_on() else None
            return GripperOutcome(REFUSED, measured, refusal)
        if cancel.is_set():
            return GripperOutcome(CANCELLED, self.gripper_read())
        return self._two_stage(1.0 - scout_opening, 1.0 - opening, max_effort, cancel)

    def gripper_close_torque(self, cancel: threading.Event) -> GripperOutcome:
        """APAI's close before a torque release: its close_gripper_torque as it is.

        The jaws close to their stop at the driver's own effort and are left
        in the light hold where they met it, or what they hold.
        """
        refusal = self._no_command()
        if refusal is not None:
            return GripperOutcome(REFUSED, None, refusal)
        if cancel.is_set():
            return GripperOutcome(CANCELLED, self.gripper_read())
        return self._close_torque(None, cancel)

    def gripper_light_hold(self) -> None:
        self._hold_light(None)

    # Internals

    def _note_readings(self, readings: Sequence[JointReading]) -> None:
        """Give the hold watch the joint readings that pass through the port."""
        if self._hold is not None and self.torque_on():
            for reading in readings:
                self._hold.note(reading, self._last_target)

    def _fault_reports(self) -> list[tuple[str, int]]:
        """(motor name, fault bits) of each motor the motor library reports for.

        The library keeps each RobStride motor's newest fault report; reading
        it sends nothing. Empty for a driver without such motors, and for a
        motor whose report cannot be read.
        """
        motors = getattr(getattr(self._backend, "_arm", None), "_motor_map", None)
        if not isinstance(motors, Mapping):
            return []
        reports: list[tuple[str, int]] = []
        for name, motor in sorted(motors.items()):
            read = getattr(motor, "robstride_get_fault_report", None)
            if not callable(read):
                continue
            try:
                fault, _warning = read()
            except Exception as exc:
                log.debug("no fault report of motor %s: %s", name, exc)
                continue
            reports.append((str(name), int(fault)))
        return reports

    def _poll_motor_faults(self) -> list[str]:
        """Motors whose fault report turned to a fault since the connect."""
        now = self._clock.monotonic()
        if now - self._fault_poll_t < _FAULT_POLL_S:
            return self._motor_faults
        self._fault_poll_t = now
        self._motor_faults = [
            f"motor {name} reports the fault 0x{fault:08x}: a motor with a fault "
            "ignores its commands, so it may not hold"
            for name, fault in self._fault_reports()
            if fault and fault != self._fault_baseline.get(name, 0)
        ]
        return self._motor_faults

    def _no_command(self) -> str | None:
        """Why the driver is sent no target and no jaw command; None when it is."""
        if self._releasing:
            return "a torque release is under way"
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
        return GripperOutcome(CANCELLED, measured)

    def _jaw_refusal(self, opening: float, max_effort: float, mode: str) -> str | None:
        reason = self._no_command()
        if reason is not None:
            return reason
        if not (self._jaws.referenced() or self._jaws.references_itself):
            return "the jaws are not referenced"
        for name, value in (("opening", opening), ("max_effort", max_effort)):
            problem = _fraction_problem(name, value)
            if problem is not None:
                return problem
        if mode not in JAW_MODES:
            return f"unknown mode {mode!r}"
        return None

    def _driver(
        self, routine: Callable[[], Any], cancel: threading.Event
    ) -> tuple[Any, GripperOutcome | None]:
        """Run one jaw routine of the driver: (what it returned, None).

        (None, the outcome) when it did not run to its end: a cancel ended
        the reference the driver began, or the driver raised a fault of the
        jaws that leaves the arm usable. Any other error is an ArmFault.
        """
        try:
            return routine(), None
        except _ReferenceStopped as stopped:
            return None, GripperOutcome(CANCELLED, None, str(stopped))
        except GripperFault as exc:
            text = str(exc)
            if any(part in text for part in _JAW_FAULTS):
                return None, GripperOutcome(GRIPPER_FAULT, self.gripper_read(), text)
            raise ArmFault(f"the jaw command failed: {exc}") from exc
        except Exception as exc:
            raise ArmFault(f"the jaw command failed: {exc}") from exc

    def _opened(
        self, opening: float, routine: Callable[[], Any], cancel: threading.Event
    ) -> GripperOutcome:
        """The driver's set_gripper or open_gripper, and how it ended.

        "reached" for a command to a wider opening than the jaw target in
        force, which the driver's guarded opening took to its end; "sent"
        for any other: the driver's close and hold return at once.
        """
        before = self.gripper_last_target()
        _, ended = self._driver(routine, cancel)
        if ended is not None:
            return ended
        measured = self.gripper_read()
        if cancel.is_set():
            return self._cancelled(measured)
        wider = before is None or opening > before + _EPS
        return GripperOutcome(REACHED if wider else SENT, measured)

    def _two_stage(
        self, stage1: float, stage2: float, effort: float, cancel: threading.Event
    ) -> GripperOutcome:
        """The driver's two-stage close to the close fractions stage1 and stage2."""
        jaws = self._jaws
        seat = 1.0 - stage2
        if not isinstance(jaws, _GuardJaws):
            # A driver without a two-stage close: its seat stage alone.
            return self._contact_close(seat, effort, cancel)
        returned, ended = self._driver(
            lambda: jaws.close_two_stage(stage1, stage2, effort, cancel), cancel
        )
        if ended is not None:
            return ended
        end, skipped = returned
        if cancel.is_set():
            return self._cancelled(self.gripper_read())
        measured = None if end is None else self._opening(float(end))
        reached = measured is not None and abs(measured - seat) < self._reach_tol
        return DriverOutcome(
            REACHED if reached else STALLED, measured, stages_skipped=skipped
        )

    def _close_torque(
        self, effort: float | None, cancel: threading.Event
    ) -> GripperOutcome:
        """The driver's close to the closed stop, which ends in its light hold."""
        jaws = self._jaws
        if not isinstance(jaws, _GuardJaws):
            # A driver without that routine: one close, then the light hold.
            ended = self._contact_close(0.0, 1.0 if effort is None else effort, cancel)
            if ended.status in (REACHED, STALLED):
                self._hold_light(None)
            return ended
        returned, ended = self._driver(
            lambda: jaws.close_torque(effort, cancel), cancel
        )
        if ended is not None:
            return ended
        closed, contact = returned
        measured = self.gripper_read()
        if cancel.is_set():
            return self._cancelled(measured)
        status = TIMEOUT if not closed else REACHED if contact is None else STALLED
        return DriverOutcome(status, measured, closed=closed)

    def _contact_close(
        self, opening: float, effort: float, cancel: threading.Event
    ) -> GripperOutcome:
        """One close to `opening`, then a wait until the jaws are there or stop.

        For a close that APAI's driver has no routine for. It ends "reached"
        or "stalled"; the jaws keep the close's stiffness.
        """
        jaws = self._jaws
        _, ended = self._driver(
            lambda: jaws.send(self._jaw(opening), effort, cancel), cancel
        )
        if ended is not None:
            return ended
        grace = math.ceil(GRIP_GRACE_S / GRIP_POLL_S - _EPS)
        limit = math.ceil(_CONTACT_WAIT_S / GRIP_POLL_S - _EPS)
        before: float | None = None
        measured: float | None = None
        # Counted in looks, so a fake clock advanced elsewhere cannot end it.
        for look in range(limit + 1):
            if look:
                self._clock.sleep(GRIP_POLL_S)
            measured = self.gripper_read()
            if cancel.is_set():
                return self._cancelled(measured)
            if measured is None:
                continue
            if abs(measured - opening) <= self._reach_tol:
                return GripperOutcome(REACHED, measured)
            if look >= grace:
                if before is not None and abs(measured - before) < self._stall_tol:
                    break
                before = measured
        return GripperOutcome(STALLED, measured)


def _fraction_problem(name: str, value: object) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return f"{name} must be a number, got {value!r}"
    if not 0.0 <= value <= 1.0:
        return f"{name} must be between 0 and 1, got {value!r}"
    return None
