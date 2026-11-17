# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The arm APAI drives on the workstation: a stand-in for its motor driver.

RobotLinkArm sends no setpoint. A move becomes one MOVE and then PERMITs: the
robot app on the Thor paces the setpoints and sends each one only while a
permit covers it. A permit covers a setpoint only after APAI's checker callback
cleared it on the obstacle grid whose camera-frame time the permit quotes (the
check ahead). The adapter sends MOVE, PERMIT, GRIPPER, CHECKER_STATE and STOP,
and, where APAI's rest tool ends, the OWNER_REQUEST motors_off: the robot app
switches the motors off for it only with the arm measured at rest.

APAI is imported inside functions, so the module imports without it.
"""

from __future__ import annotations

import inspect
import logging
import math
import re
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..protocol.messages import (
    JAW_CLOSE_TORQUE,
    JAW_CLOSE_UNTIL_CONTACT,
    JAW_OPEN,
    JAW_PLAIN,
    MOVE_ARMED,
    MOVE_SENDING,
    PROTOCOL_VERSION,
    REQUEST_MOTORS_OFF,
    GraspRegion,
    ProtocolError,
)
from .link_client import LOST, CommandTimeout, LinkLost, StaleState
from .setpoints import largest_step, sample_move

if TYPE_CHECKING:
    from ..protocol.messages import Result, RobotState
    from .checkahead import CheckAhead
    from .link_client import AgentLink, Pinned

_log = logging.getLogger("isaaccapture.examples.agentic_remote.arm_adapter")

# The profile type the launcher's arm factory answers to.
ARM_TYPE = "robot_link"
# The sender of a STOP made by stop() outside a skill: a person's STOP.
AGENT_TERMINAL = "agent_terminal"
# The sender of a STOP the adapter makes itself: the agent's own, not a
# person's (AGENT_SENDER in stop.py).
AGENT_SENDER = "agent"
# The two CHECKER_STATE actions (CLOSE_GRASP, CLEAR_CONTACT_SPOTS in app.py).
CLOSE_GRASP = "close_grasp"
CLEAR_CONTACT_SPOTS = "clear_contact_spots"
# In a contact RESULT's reason when no hold followed (HOLD_FAILED in executor.py).
HOLD_FAILED = "the hold at the measured pose failed"
# The contact monitor's trip text (apai.safety.contact.ContactMonitor._trip).
CONTACT_TEXT = re.compile(
    r"contact detected at joint (\d+): tracking error ([0-9.]+) rad for ([0-9.]+) ms"
)
# Jaw refusals that are the jaws' own fault, not a refusal of the command.
_JAWS_UNREFERENCED = "the jaws are not referenced"
_JAWS_UNREAD = "the jaws could not be read, so no jaw command was sent"
# A RESULT's detail "outcome" when the Thor's driver ended a jaw command with a
# fault of the jaws that leaves the arm usable; the RESULT's reason is the
# driver's own text (GRIPPER_FAULT in robot_app/gripper.py).
_GRIPPER_FAULT = "gripper_fault"
# How long the end of a move waits for a ROBOT_STATE that shows it, in s.
_REFLECT_S = 0.3
# The longest one wait for fresh joint readings sleeps, in s.
_FRESH_LOOK_S = 0.05
# How the robot app begins its refusal of a MOVE whose start pose the arm has
# left (executor.py, _read_start).
START_POSE = "start pose: "
# How long the permit loop waits for a RESULT beyond the Thor's own waits, in s.
_RESULT_MARGIN_S = 3.0
# How soon the pause of the permit loop sees a halt of the checker, in s.
_HALT_LOOK_S = 0.005


class RobotLinkError(RuntimeError):
    """A fault of the robot connection or of the robot app, not a refusal."""


class _NotYet(RuntimeError):
    """The newest ROBOT_STATE gives no usable pose; a newer one may."""


@dataclass(frozen=True)
class ArmSettings:
    """What the adapter needs of agent.yaml; the defaults are the first values."""

    # link
    expected_profile_id: str = ""  # "" accepts any profile id
    calibration_id: str = ""  # the workstation's camera calibration id
    allow_unnamed_calibration: bool = False
    connect_timeout_s: float = 10.0
    state_max_age_s: float = 0.5
    # No PERMIT is sent on a ROBOT_STATE older than this; under state_max_age_s.
    state_renew_age_s: float = 0.1
    # The workstation's joint range, compared with the WELCOME's for a warning.
    joint_min: tuple[float, ...] | None = None
    joint_max: tuple[float, ...] | None = None
    # thor: copies of the Thor's settings that are not in the WELCOME
    first_wait_s: float = 0.5
    settle_timeout_s: float = 2.0
    agent_max_joint_speed: float = 0.4
    arm_wait_s: float = 1.0
    # move
    start_tol_rad: float = 0.02
    fresh_wait_s: float = 0.6
    start_grid_age_s: float = 1.0
    start_grid_wait_s: float = 3.0
    expires_in_s: float = 2.0
    admit_wait_s: float = 1.0
    cancel_wait_s: float = 2.0
    # permits
    reach_setpoints: int = 25
    renew_s: float = 0.05
    life_s: float = 1.0
    # jaws
    fresh_s: float = 1.0
    result_timeout_s: float = 12.0
    # rest: how long the robot app has to switch the motors off at rest
    motors_off_timeout_s: float = 30.0

    def start_tol(self, period_s: float) -> float:
        """The start tolerance a MOVE quotes.

        Half of one period's travel at the Thor's agent speed limit: the arm
        may lie that far from the pose the move was sampled from, and the
        first setpoint's real step still stays inside the speed limit. The
        Thor checks the first setpoint from the move's start pose only for a
        tolerance this small (executor.py, _PLANNED_START_TOL_RAD).
        """
        return min(self.start_tol_rad, 0.5 * self.agent_max_joint_speed * period_s)


def _types() -> Any:
    import apai.types

    return apai.types


_arm_type_lock = threading.Lock()
_arm_type: type | None = None


# What RobotLinkArm keeps of APAI's LazyArm: its type, and the answers about
# a connect made on an operator's confirmation, which is never made here.
# Every other name of LazyArm is RobotLinkArm's own or ArmBase's.
LAZY_KEPT = frozenset(
    {
        "PASSES_WHILE_HOLDING",
        "holding_on_confirmation",
        "holding_untold",
        "told_to_let_go",
        "next_call",
        "let_go",
        "profile_type",
    }
)


def _arm_type_on_apai() -> type:
    """RobotLinkArm as a subclass of APAI's LazyArm; made on first use.

    A LazyArm is APAI's arm whose motors may be off between uses (standby):
    APAI's rest tool switches only such an arm's motors off, and its next
    motion switches them on again. Here the motors are the Thor's.
    """
    global _arm_type
    with _arm_type_lock:
        if _arm_type is None:
            from apai.control.arm_base import ArmBase
            from apai.control.lazy_arm import LazyArm

            class _OnArmBase(RobotLinkArm, LazyArm):
                __doc__ = RobotLinkArm.__doc__

            # LazyArm passes its calls on to a backend it makes. There is no
            # such backend: where RobotLinkArm has no method of its own,
            # ArmBase's is used.
            for name, value in vars(LazyArm).items():
                if name.startswith("__") or name in LAZY_KEPT:
                    continue
                if name in vars(RobotLinkArm):
                    continue
                if not (callable(value) or isinstance(value, property)):
                    continue
                if not hasattr(ArmBase, name):
                    raise RuntimeError(
                        f"APAI's LazyArm has {name}, which the robot link arm "
                        "does not know"
                    )
                setattr(_OnArmBase, name, getattr(ArmBase, name))
            _OnArmBase.__name__ = _OnArmBase.__qualname__ = "RobotLinkArm"
            _OnArmBase.__module__ = __name__
            _arm_type = _OnArmBase
        return _arm_type


# The setpoints of a motion from a start pose to a goal: sample(start, goal).
_Sample = Callable[[np.ndarray, np.ndarray], list[np.ndarray]]


class _FreshReader:
    """What prepare_stream reads the start pose from: fresh readings only."""

    def __init__(self, read: Callable[[], Any]) -> None:
        self.get_state = read


# What the adapter hands to the launcher's measurements: measure(kind, facts).
# The kinds: MEASURED_MOVE with a move's stretches when stream_to() ends,
# MEASURED_JAW_BEGIN when a GRIPPER is about to be sent and MEASURED_JAW when
# its wait has ended. The times in the facts are of this machine's monotonic
# clock, in s. It is called on the skill thread and must only queue.
Measure = Callable[[str, dict[str, Any]], None]
MEASURED_MOVE = "move"
MEASURED_JAW_BEGIN = "jaw_begin"
MEASURED_JAW = "jaw"
# The points of a move's permit loop that RobotLinkArm._mark() is called at:
# the first PERMIT has left, the Thor reports the first setpoint as sent, the
# Thor reports the last one as sent (what follows is the settling).
MARK_FIRST_PERMIT = "first_permit"
MARK_FIRST_SETPOINT = "first_setpoint"
MARK_ALL_SENT = "all_sent"
MARKS = (MARK_FIRST_PERMIT, MARK_FIRST_SETPOINT, MARK_ALL_SENT)


@contextmanager
def _timed(stats: dict[str, Any], name: str) -> Iterator[None]:
    """Add the seconds the block takes to stats[name], also when it raises."""
    began = time.monotonic()
    try:
        yield
    finally:
        stats[name] = stats.get(name, 0.0) + time.monotonic() - began


class RobotLinkArm:
    """APAI's arm on the robot link. RobotLinkArm(...) builds an ArmBase subclass.

    Entry points for the launcher and the task gate, which APAI never calls:
    bind_checker(), open_task(), close_task(), skill_scope() and
    check_welcome().
    """

    n_joints = 6
    motion_rate_hz = 50.0
    settle_tol = 0.03  # ArmBase's; the Thor settles its own moves
    # Teardown: APAI neither parks this arm nor cuts its torque. Only its
    # rest tool, inside a task, ends in disconnect() with the motors to go off.
    shutdown_park_required = False
    disconnect_preserves_drive_state = False
    disconnect_cuts_torque = False
    hold_key = None
    connect_cuts_torque = False
    connected_on_confirmation = False

    def __new__(cls, *args: Any, **kwargs: Any) -> RobotLinkArm:
        made = _arm_type_on_apai() if cls is RobotLinkArm else cls
        return object.__new__(made)

    def __init__(
        self,
        cfg: Any,
        kinematics: Any = None,
        *,
        link: AgentLink,
        settings: ArmSettings | None = None,
        joint_feed: Any = None,
        measure: Measure | None = None,
    ) -> None:
        self._cfg = cfg
        self.kin = kinematics
        self._link = link
        self._settings = settings or ArmSettings()
        self._joint_feed = joint_feed
        self._measure = measure
        self._measure_failed = False
        n_joints = _cfg_get(cfg, "n_joints")
        self.n_joints = 6 if n_joints is None else int(n_joints)
        jaws = _cfg_get(cfg, "gripper") or {}
        self._grip_open = float(_cfg_get(jaws, "open_pos", 0.0))
        self._grip_closed = float(_cfg_get(jaws, "closed_pos", 1.0))
        if self._grip_open == self._grip_closed:
            raise ValueError("the arm profile's jaws have no travel")
        # What APAI's LazyArm keeps, for the code of APAI that reads it.
        self._lock = threading.Lock()
        self._holding: str | None = None
        self._profile_type = ARM_TYPE
        self._hold_key = None
        self._attached = False  # connect() found a robot app this arm fits
        self._standby = False  # the robot app switched the motors off for us
        self._latched = False
        # How many stages the Thor's driver skipped in the last two-stage
        # close; None before the first one and when the driver gave no count.
        self.close_stages_skipped: int | None = None
        self._check: CheckAhead | None = None
        self._task = False
        self._skill = threading.local()
        self._grasp_sent = False  # a MOVE of this task carried a grasp region
        self._checker_ids: list[tuple[str, str]] = []  # (action, command id)
        self._ids_lock = threading.Lock()
        # What the last move measured: its stretches and what its permit loop
        # did. The same facts go to `measure` when stream_to() ends.
        self.last_move: dict[str, Any] = {}
        # The same of the last jaw command.
        self.last_jaw: dict[str, Any] = {}

    # What the launcher calls

    def bind_checker(self, check_ahead: CheckAhead) -> None:
        """Take the check ahead and hook two of its checker's state changes.

        After the checker's clear_grasp_exemption() and clear_contact_zones()
        the robot app is told the same with a CHECKER_STATE, inside a task.
        """
        self._check = check_ahead
        harness = check_ahead.harness
        for name, action in (
            ("clear_grasp_exemption", CLOSE_GRASP),
            ("clear_contact_zones", CLEAR_CONTACT_SPOTS),
        ):
            original = getattr(harness, name)
            if getattr(original, "robot_link_action", None) == action:
                original = original.robot_link_original
            setattr(harness, name, self._hooked(original, action))

    def open_task(self, pinned: Pinned | None = None) -> None:
        """The agent owns the arm with these numbers: motion may be sent."""
        if pinned is not None and self._link.pinned != pinned:
            self._link.pin(pinned)
        self._grasp_sent = False
        self._latched = False
        self._task = True

    def close_task(self) -> None:
        """End the task scope and drop the pinned numbers; sends nothing."""
        self._task = False
        self._log_checker_results()
        self._link.unpin()

    @contextmanager
    def skill_scope(self) -> Iterator[None]:
        """While open on this thread, stop() is a skill stop: no STOP is sent."""
        depth = getattr(self._skill, "depth", 0)
        self._skill.depth = depth + 1
        try:
            yield
        finally:
            self._skill.depth = depth

    # The six methods of ArmBase

    def connect(self) -> None:
        """Wait for the robot app and compare what it says of itself; sends nothing."""
        link, settings = self._link, self._settings
        try:
            welcome = link.wait_ready(settings.connect_timeout_s)
            state = link.state()
        except (LinkLost, StaleState, TimeoutError) as error:
            raise RobotLinkError(f"the robot app is not ready: {error}") from error
        problem = self.check_welcome(welcome)
        if problem is not None:
            raise RobotLinkError(problem)
        self._warn_joint_range(welcome)
        self._attached = True
        self._standby = not state.torque_on

    def disconnect(self) -> None:
        """Where APAI's rest tool ends: the robot app is asked for the motors off.

        Inside a task with the motors on, the robot app is asked to switch
        them off; it does so only with the arm measured at rest. RobotLinkError,
        with the motors left on, when it refuses or does not answer. Anywhere
        else, as when the program ends, nothing is sent: the robot app keeps
        holding the arm.
        """
        if not (self._task and self.connected):
            return
        timeout_s = self._settings.motors_off_timeout_s
        try:
            result = self._link.request_owner(
                REQUEST_MOTORS_OFF,
                reason="the agent's rest tool has folded the arm",
                quote=self._link.pinned,
                timeout_s=timeout_s,
            )
        except (LinkLost, StaleState, CommandTimeout) as error:
            raise RobotLinkError(
                f"the robot app did not answer the request for the motors off "
                f"({error}); the motors stay on unless the Thor terminal says "
                "otherwise"
            ) from error
        if result.status != "completed":
            raise RobotLinkError(f"the robot app kept the motors on: {result.reason}")
        self._standby = True

    @property
    def connected(self) -> bool:
        """Whether the Thor's motors are on for this arm; False in standby.

        Standby is APAI's word for an arm whose motors are off until its next
        motion. Without a ROBOT_STATE to say, the motors count as on.
        """
        if not self._attached or self._standby:
            return False
        try:
            return bool(self._link.state().torque_on)
        except (LinkLost, StaleState):
            return True

    @connected.setter
    def connected(self, value: bool) -> None:
        self._attached = bool(value)
        self._standby = False

    def motors_on(self) -> None:
        """The robot app switched the motors on again: standby has ended.

        In standby the obstacle grid has no arm pose to mask the arm out of
        the camera's depth with, so its refreshes fail and it refuses every
        check. As APAI's own arm when it wakes, this waits for the grid's
        next refresh with a pose, with APAI's routine and its bound
        (occupancy.ready_timeout_s): the first motion is neither refused for
        the standby nor left unchecked.
        """
        self._standby = False
        grid = getattr(getattr(self._check, "harness", None), "occupancy", None)
        if callable(getattr(grid, "wait_until_usable", None)):
            from apai.apps.demo import _await_occupancy_after_wake

            _await_occupancy_after_wake(grid, self._cfg)

    # APAI's LazyArm has a backend of its own behind these; this arm is its own.

    @property
    def _arm(self) -> Any:
        return self if self.connected else None

    def _ensure(self) -> Any:
        return self

    def _command(self) -> Any:
        return self

    def _woke(self, arm: Any) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    def get_state(self) -> Any:
        """The newest published pose; RuntimeError when none is usable.

        APAI's own driver reads the motors anew at each call and goes on over
        a missed answer. The readings here are the Thor's: a pose that is not
        usable yet is waited for, at most fresh_wait_s. A lost link is not
        waited for.
        """
        end = time.monotonic() + self._settings.fresh_wait_s
        while True:
            try:
                return self._state_now()
            except _NotYet:
                left = end - time.monotonic()
                if left <= 0:
                    raise
            self._link.wait_change(min(left, _FRESH_LOOK_S))

    def _state_now(self) -> Any:
        """The newest published pose; never blocks.

        Its time t is that of its oldest joint reading, and it is at most
        state_max_age_s before now. _NotYet when a newer ROBOT_STATE can
        mend it, RuntimeError for a lost link.
        """
        settings = self._settings
        limit = settings.state_max_age_s
        try:
            state, arrived = self._link.state_and_arrival(limit)
        except StaleState as error:
            raise _NotYet(f"robot link feedback lost: {error}") from error
        except LinkLost as error:
            raise RuntimeError(f"robot link feedback lost: {error}") from error
        q = _pose(state.joints, self.n_joints)
        if q is None:
            raise _NotYet(
                "robot link feedback lost: a joint has no valid reading in the "
                "newest ROBOT_STATE"
            )
        age = state.health.get("reading_age_s")
        if not _is_number(age) or age > limit:
            raise _NotYet(
                f"robot link feedback lost: the joint readings are {age!r} s old, "
                f"over {limit:g} s"
            )
        since = time.monotonic() - arrived
        if since + age > limit:
            # What reads the pose's time takes it for a reading made just now.
            raise _NotYet(
                f"robot link feedback lost: the joint readings are {age:.2f} s old "
                f"and their ROBOT_STATE arrived {since:.2f} s ago, over {limit:g} s "
                "in all"
            )
        position, valid = self._jaw_reading(state)
        return _types().RobotState(
            q=np.asarray(q, dtype=float),
            gripper_pos=position,
            gripper_valid=valid,
            t=arrived - float(age),
        )

    def send_joint_target(self, q: Any) -> None:
        raise RuntimeError(
            "the robot link takes whole moves, not single setpoints; nothing sent"
        )

    def stream_profile(self, *args: Any, **kwargs: Any) -> bool:
        """A motion planner's profile; APAI's reBot driver has no such motion.

        APAI calls it only on an arm with a motion planner bound, which the
        experiment's arm is not: there the call ends in an AttributeError of
        the driver, and here too.
        """
        raise AttributeError(
            "the robot link arm streams no planner profile, as APAI's reBot "
            "driver; nothing sent"
        )

    def set_gripper(self, pos: float, effort: float = 1.0) -> None:
        """APAI's set_gripper on the Thor: one plain GRIPPER.

        pos is a jaw position in the arm profile's units. As on APAI's driver
        an opening blocks until its guarded ramp has ended, and a close or a
        hold returns once it is sent, with the jaws still moving. GripperFault
        with the driver's own text when the driver refused the command or
        found the jaws blocked while they opened.
        """
        self._refuse_unless_task("jaw command")
        self._opened(self._opening(pos), effort, JAW_PLAIN)

    def open_gripper(self) -> None:
        """APAI's open_gripper on the Thor: the driver's own opening and effort."""
        self._refuse_unless_task("jaw command")
        self._opened(1.0, 1.0, JAW_OPEN)

    def _opened(self, opening: float, effort: float, mode: str) -> None:
        """One GRIPPER that may open the jaws; GripperFault when they were blocked."""
        before = self._jaw_before()
        result = self._jaw(opening, effort, mode)
        if self._jaw_ending(result) == "short":
            if before is None or opening > before + 1e-9:
                raise _types().GripperFault("the jaws were blocked while opening")

    def close_gripper_torque(
        self,
        effort: float = 0.6,
        stall_rad: float = 0.03,
        settle_tol: float = 0.1,
        timeout_s: float = 3.0,
        hold_kp: float | None = None,
    ) -> bool:
        """APAI's close_gripper_torque on the Thor; what the driver returned.

        The jaws close to their stop and are left in the light hold where
        they met it or what they hold. The Thor's driver runs it with its own
        stall_rad, settle_tol, timeout_s and hold_kp: other values than the
        driver's own are refused.
        """
        self._refuse_unless_task("jaw command")
        self._only_defaults(
            "close_gripper_torque",
            stall_rad=stall_rad,
            settle_tol=settle_tol,
            timeout_s=timeout_s,
            hold_kp=hold_kp,
        )
        result = self._jaw(0.0, effort, JAW_CLOSE_TORQUE)
        self._jaw_ending(result)
        detail = result.detail or {}
        return bool(detail.get("closed", detail.get("outcome") != "timeout"))

    def stop(self) -> None:
        """Latch the adapter's stop and end a running move; never raises.

        Inside skill_scope() on this thread nothing is sent: the agent keeps
        the arm. Anywhere else it is a person's STOP.
        """
        try:
            self._latched = True
            if getattr(self._skill, "depth", 0) > 0:
                return
            self._link.stop("stop at the agent program", AGENT_TERMINAL)
        except Exception:
            _log.exception("stop() failed")

    # Other calls of APAI

    def resume(self) -> None:
        raise RuntimeError(
            "resume is not available over the robot link: start a new task"
        )

    def body_mask_state(self) -> Any:
        """The pose for APAI's self mask: at the camera frame's capture time."""
        from apai.perception.robot_mask import BodyMaskUnavailable

        from .self_mask import mask_pose

        try:
            # Not waited for: a camera frame without a pose is left out.
            return mask_pose(self._joint_feed, self._state_now)
        except BodyMaskUnavailable:
            raise
        except RuntimeError as error:
            raise BodyMaskUnavailable(str(error)) from error

    def body_mask_q(self) -> Any:
        return self.body_mask_state().q

    def stream_path(
        self,
        waypoints: Any,
        duration_s: float,
        rate_hz: float = 50.0,
        approve: Any = None,
        settle_tol: float | None = None,
        settle_timeout_s: float | None = None,
        preflight: Any = None,
        before_stream: Any = None,
        contact_monitor: Any = None,
    ) -> bool:
        """A dense joint path as one MOVE, as APAI's own driver streams one.

        The setpoints are those of APAI's path_ticks along the waypoints from
        a fresh start pose; the arm does not stop at a waypoint. The rest is
        stream_to()'s.
        """
        violation = _types().SafetyViolation
        try:
            points = [np.asarray(w, dtype=float).reshape(-1) for w in waypoints]
        except (TypeError, ValueError):
            raise violation("a joint path needs joint poses; no motion sent") from None
        if not points:
            raise violation("empty joint path; no motion sent")
        for point in points:
            if point.shape != (self.n_joints,) or not np.isfinite(point).all():
                raise violation(
                    f"every pose of a joint path needs {self.n_joints} finite "
                    "joint values; no motion sent"
                )

        def sample(start: np.ndarray, goal: np.ndarray) -> list[np.ndarray]:
            from apai.control.arm_base import path_ticks

            return path_ticks(start, points, duration_s, rate_hz)[0]

        return self._stream(
            points[-1],
            sample,
            duration_s,
            rate_hz,
            approve,
            settle_tol,
            settle_timeout_s,
            preflight,
            before_stream,
            contact_monitor,
        )

    def close_gripper_two_stage(
        self,
        width_frac_stage1: float = 0.5,
        width_frac_stage2: float = 0.85,
        effort: float = 0.6,
        stall_vel: float = 0.05,
        timeout_s: float = 3.0,
    ) -> float | None:
        """APAI's close_gripper_two_stage on the Thor: one GRIPPER for both stages.

        The Thor's driver runs the scout stage and the seat stage, decides
        when each has ended and which it skips, and leaves the jaws pushing at
        the seat stage. Returns the jaw position it ended at in the arm
        profile's units, None when the driver could not read it;
        close_stages_skipped is the driver's count, None when it gave none.
        The driver runs it with its own stall_vel and timeout_s: other values
        than the driver's own are refused.
        """
        self._refuse_unless_task("jaw command")
        self._only_defaults(
            "close_gripper_two_stage", stall_vel=stall_vel, timeout_s=timeout_s
        )
        seat = min(1.0, max(0.0, 1.0 - float(width_frac_stage2)))
        scout = min(1.0, max(0.0, 1.0 - float(width_frac_stage1)))
        result = self._jaw(seat, effort, JAW_CLOSE_UNTIL_CONTACT, scout)
        self._jaw_ending(result)
        detail = result.detail or {}
        skipped = detail.get("close_stages_skipped")
        self.close_stages_skipped = skipped if type(skipped) is int else None
        measured = detail.get("opening")
        return self._position(measured) if _is_number(measured) else None

    def stream_to(
        self,
        q_target: Any,
        duration_s: float,
        rate_hz: float = 50.0,
        approve: Any = None,
        settle_tol: float | None = None,
        settle_timeout_s: float | None = None,
        preflight: Any = None,
        before_stream: Any = None,
        contact_monitor: Any = None,
    ) -> bool:
        """One MOVE, then PERMITs until its final RESULT; on the skill thread.

        True when the arm settled at the goal, False when it did not. A
        refusal before the robot app armed the move has called the checker
        callback for no setpoint. When the robot app finds the arm away from
        the start pose the move was sampled from, the move is sampled again
        from a new reading, as APAI's prepare_stream does for a route.
        """

        def sample(start: np.ndarray, goal: np.ndarray) -> list[np.ndarray]:
            from apai.control.motion_profile import min_jerk

            return sample_move(start, goal, duration_s, rate_hz, min_jerk)

        return self._stream(
            q_target,
            sample,
            duration_s,
            rate_hz,
            approve,
            settle_tol,
            settle_timeout_s,
            preflight,
            before_stream,
            contact_monitor,
        )

    def _stream(self, goal: Any, sample: _Sample, *args: Any) -> bool:
        """One streamed motion; what it measured goes to `measure` at its end."""
        stats: dict[str, Any] = {}  # what this move measures
        try:
            settled = self._stream_checked(stats, goal, sample, *args)
        except BaseException as error:
            self._note_move(stats, type(error).__name__)
            raise
        self._note_move(stats, "completed" if settled else "not_settled")
        return settled

    def _stream_checked(
        self,
        stats: dict[str, Any],
        q_target: Any,
        sample: _Sample,
        duration_s: float,
        rate_hz: float,
        approve: Any,
        settle_tol: float | None,
        settle_timeout_s: float | None,
        preflight: Any,
        before_stream: Any,
        contact_monitor: Any,
    ) -> bool:
        """_stream(); the stretches of the move go into stats as they end.

        sample(start pose, goal) gives the setpoints of the motion.
        """
        violation = _types().SafetyViolation
        settings = self._settings
        self._log_checker_results()
        welcome = self._link.welcome
        if approve is None:
            raise violation("a move needs the checker callback; no motion sent")
        if welcome is None or not self.connected:
            raise violation("the arm is not connected; no motion sent")
        period = float(welcome.setpoint_period_s)
        if not (_is_number(rate_hz) and math.isclose(rate_hz, 1.0 / period)):
            raise violation(
                f"the robot app sends {1.0 / period:g} setpoints a second, not "
                f"{rate_hz!r}; no motion sent"
            )
        if settle_tol is not None or settle_timeout_s is not None:
            raise violation(
                "the settle tolerance and time are the robot app's; no motion sent"
            )
        try:
            target = np.asarray(q_target, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            target = np.empty(0)
        if target.shape != (self.n_joints,) or not np.isfinite(target).all():
            raise violation(
                f"the goal needs {self.n_joints} finite joint values; no motion sent"
            )
        if not _is_number(duration_s) or duration_s < 0:
            raise violation("the duration must be 0 or more; no motion sent")
        self._refuse_unless_task("motion")
        check = self._check
        if check is None:
            raise violation("no check ahead is bound to the arm; no motion sent")

        # Read where they stand at this call: the experiment's launcher gives
        # the feedback budget its own length.
        from apai.control import arm_base

        def fresh() -> Any:
            stats["fresh_reads"] = stats.get("fresh_reads", 0) + 1
            with _timed(stats, "fresh_wait_s"):
                return self._fresh_state()

        tol = settings.start_tol(period)
        room = settings.agent_max_joint_speed * period - tol
        tries = int(arm_base.PREFLIGHT_MAX_CHECKS)
        give_up = time.monotonic() + float(arm_base.PREFLIGHT_REBIND_BUDGET_S)
        for attempt in range(1, tries + 1):
            start, target = arm_base.prepare_stream(
                _FreshReader(fresh),
                target,
                duration_s,
                preflight,
                before_stream,
            )
            q = sample(start, target)
            if len(q) > welcome.max_move_setpoints:
                raise violation(
                    f"the move has {len(q)} setpoints, the robot app takes at most "
                    f"{welcome.max_move_setpoints}; no motion sent"
                )
            with _timed(stats, "grid_wait_s"):
                check.wait_fresh_grid(
                    settings.start_grid_age_s, settings.start_grid_wait_s
                )
            with _timed(stats, "whole_check_s"):
                evidence = self._vet_start(check, start, q)
            if largest_step(start, q[0]) > room:
                raise violation(
                    f"the first setpoint is {largest_step(start, q[0]):.4f} rad "
                    f"from the start pose; the robot app's speed limit leaves "
                    f"{room:.4f} rad for it; no motion sent"
                )
            if contact_monitor is not None:
                # From here a TypeError must not make SafeArm send the move again.
                contact_monitor.started = True
            try:
                return self._move(
                    start, q, tol, evidence.wall_s, approve, contact_monitor, stats
                )
            except TypeError as error:
                raise RobotLinkError(
                    f"error in the arm adapter during a move: {error!r}"
                ) from error
            except violation as error:
                # The arm moved between this program's reading and the Thor's:
                # nothing was sent. APAI's prepare_stream takes a new reading
                # in this case; so does this.
                moved = getattr(error, "start_pose_moved", False)
                if not moved or attempt == tries or time.monotonic() >= give_up:
                    raise
                stats["start_retries"] = attempt
                _log.info("the move is sampled again from a new reading: %s", error)
        raise AssertionError("unreachable")

    def _vet_start(self, check: CheckAhead, start: np.ndarray, q: list[Any]) -> Any:
        """The check before a MOVE is sent: the setpoints its first PERMIT covers.

        APAI checks a plain move setpoint by setpoint as it sends it, and a
        route or a joint path whole in the preflight that has run by now. So
        only what the first PERMIT reaches is vetted here; each setpoint is
        checked again before a PERMIT covers it. Returns the grid's evidence.
        """
        reach = max(1, min(len(q), self._settings.reach_setpoints))
        try:
            return check.vet_whole(start, q[:reach])
        except Exception as error:
            if type(error).__name__ != "CheckRefused" or len(q) == reach:
                raise
            # Refused setpoints are named by their place in the whole move.
            raise type(error)(error.index, error.reason, len(q)) from None

    # A move

    def _move(
        self,
        start: np.ndarray,
        q: list[np.ndarray],
        tol: float,
        evidence_wall_s: float,
        approve: Any,
        monitor: Any,
        stats: dict[str, Any] | None = None,
    ) -> bool:
        link, settings = self._link, self._settings
        violation = _types().SafetyViolation
        move_id = link.new_id()
        region = self._grasp_region()
        began = time.monotonic()
        stats = {} if stats is None else stats
        stats.update(move_id=move_id, setpoints=len(q), permits=0)
        self.last_move = stats
        # The numbers every PERMIT of this move quotes, also after close_task().
        pinned = link.pinned
        if pinned is None:
            raise violation("no ownership is pinned; no motion sent")
        try:
            link.send_move(
                move_id, start, tol, q, region, evidence_wall_s, settings.expires_in_s
            )
        except LinkLost as error:
            self._latched = True
            raise RobotLinkError(str(error)) from error
        except (StaleState, ProtocolError, RuntimeError) as error:
            raise violation(
                f"the MOVE could not be sent: {error}; no motion sent"
            ) from None
        if region is not None:
            self._grasp_sent = True
        begun = getattr(self._check.harness, "motions_begun", None)
        try:
            self._await_armed(move_id, began)
            result, cause = self._permit_loop(move_id, pinned, start, q, approve, began)
        except BaseException:
            link.forget(move_id)
            raise
        if result is not None:
            link.wait_reflected(move_id, _REFLECT_S)
            detail = result.detail if isinstance(result.detail, dict) else {}
            stats.update(
                status=result.status,
                outcome=detail.get("outcome"),
                sent=detail.get("sent"),
            )
            approved = stats.get("approved", 0)
            if result.status != "completed" and detail.get("sent") == 0 and approved:
                self._uncount(approved, begun)
        link.forget(move_id)
        return self._move_ending(result, cause, len(q), monitor)

    def _await_armed(self, move_id: str, began: float) -> None:
        """Admission, then the wait until the robot app shows the move as armed.

        Raises when the move ended first: nothing was sent and the checker
        callback saw no setpoint.
        """
        link, settings = self._link, self._settings
        first = self._await_first(move_id)
        self.last_move["first_result_after_s"] = time.monotonic() - began
        if first.status != "accepted":
            raise self._ended_unarmed(first)
        end = time.monotonic() + settings.arm_wait_s
        while True:
            result = link.final_result(move_id)
            if result is not None:
                raise self._ended_unarmed(result)
            try:
                state = link.state()
            except LinkLost as error:
                self._latched = True
                raise RobotLinkError(str(error)) from error
            except StaleState:
                state = None
            if state is not None and state.move_id == move_id:
                if getattr(state, "move_stage", "") in (MOVE_ARMED, MOVE_SENDING):
                    self.last_move["armed_after_s"] = time.monotonic() - began
                    return
            left = end - time.monotonic()
            if left <= 0:
                raise RobotLinkError("the robot app did not arm the move")
            link.wait_change(min(left, settings.renew_s))

    def _await_first(self, move_id: str) -> Result:
        """The first RESULT of a MOVE: its admission, or its end."""
        settings = self._settings
        try:
            return self._link.wait_first(move_id, settings.admit_wait_s)
        except LinkLost as error:
            self._latched = True
            raise RobotLinkError(str(error)) from error
        except CommandTimeout:
            raise RobotLinkError(
                f"no RESULT for the MOVE within {settings.admit_wait_s:g} s; no "
                "permit was sent"
            ) from None

    def _ended_unarmed(self, result: Result) -> BaseException:
        """What a move's final RESULT means when it came before "armed"."""
        types = _types()
        reason = result.reason
        self.last_move["status"] = result.status
        if reason.startswith(LOST):
            self._latched = True
            return RobotLinkError(reason)
        if result.status == "cancelled":
            self._mirror()
            return types.MotionHalted(f"the robot app stopped the move: {reason}")
        refused = types.SafetyViolation(f"robot app refused the move: {reason}")
        # The Thor read the arm away from the pose the move starts at and sent
        # nothing: _stream_checked() samples the move again.
        refused.start_pose_moved = result.status == "refused" and reason.startswith(
            START_POSE
        )
        return refused

    def _uncount(self, approved: int, begun: Any) -> None:
        """Take a move the Thor sent nothing of out of the checker's counts.

        APAI counts a command when its check cleared it, just before it is
        sent: a move whose count did not rise was refused before any motion,
        and the skill stops nothing. The check ahead clears setpoints before
        the Thor sends one, so a move the Thor ended before its first
        setpoint is taken out again: `approved` cleared commands, and its
        place among the motions begun when that rose from `begun`.
        """
        harness = self._check.harness
        try:
            from apai.safety.harness import _APPROVED_LOCK

            with _APPROVED_LOCK:
                count = getattr(harness, "approved_commands", None)
                if type(count) is int:
                    harness.approved_commands = max(0, count - approved)
            now = getattr(harness, "motions_begun", None)
            if type(begun) is int and type(now) is int and now > begun:
                harness.motions_begun = now - 1
        except Exception:
            _log.warning("the checker's counts were not put back", exc_info=True)

    def _pause(self, move_id: str) -> None:
        """The pause between two passes of the permit loop, at most renew_s.

        It ends at the next ROBOT_STATE or RESULT, and within _HALT_LOOK_S of
        a halt or a stop of the workstation checker: APAI's own stream sees
        one at its next setpoint, so the cancel must not wait out the pause.
        """
        link = self._link
        harness = self._check.harness
        end = time.monotonic() + self._settings.renew_s
        try:
            seen: float | None = link.state_arrived_s()
        except StaleState:
            seen = None
        while True:
            left = end - time.monotonic()
            if left <= 0:
                return
            link.wait_change(min(left, _HALT_LOOK_S))
            if getattr(harness, "halted", None) is not None:
                return
            if getattr(harness, "estopped", False):
                return
            if link.final_result(move_id) is not None:
                return
            try:
                if link.state_arrived_s() != seen:
                    return
            except StaleState:
                continue

    def _permit_loop(
        self,
        move_id: str,
        pinned: Pinned,
        start: np.ndarray,
        q: list[np.ndarray],
        approve: Any,
        began: float,
    ) -> tuple[Result | None, BaseException | None]:
        """Send PERMITs until the move's final RESULT.

        Every PERMIT quotes `pinned`. Returns (the final RESULT, why the
        adapter cancelled). The RESULT is None when none came; the cause is
        None when the adapter did not cancel.
        """
        link, settings = self._link, self._settings
        stats = self.last_move
        count = len(q)
        reach = settings.reach_setpoints
        # APAI's stream has no time limit: a move lasts as long as its
        # setpoints go out. The Thor ends a move whose first permit does not
        # come and one that does not settle, so no RESULT is coming once no
        # further setpoint was reported for longer than both and a margin.
        quiet_s = settings.first_wait_s + settings.settle_timeout_s + _RESULT_MARGIN_S
        progress_at = time.monotonic()
        move_check = self._check.begin(approve, start, q)
        last_evidence: float | None = None  # of the last PERMIT sent
        sent = 0
        cause: BaseException | None = None
        try:
            while True:
                result = link.final_result(move_id)
                if result is not None:
                    stats["result_after_s"] = time.monotonic() - began
                    stats["approved"] = move_check.approved_through + 1
                    stats["sent_seen"] = sent
                    return result, None
                if time.monotonic() - progress_at >= quiet_s:
                    self._latched = True
                    cause = RobotLinkError("no RESULT for the move")
                    break
                cause = self._live_gate()
                if cause is not None:
                    break
                state, arrived = link.state_and_arrival(settings.state_max_age_s)
                # On an old state nothing is renewed: the Thor's last PERMIT
                # runs out unless a newer state arrives.
                fresh = time.monotonic() - arrived <= settings.state_renew_age_s
                if state.move_id == move_id and fresh:
                    started = time.monotonic()
                    if int(state.move_index) > sent:
                        sent = int(state.move_index)
                        progress_at = started
                    grant = move_check.step(sent, reach)
                    took = time.monotonic() - started
                    stats["passes"] = stats.get("passes", 0) + 1
                    stats["pass_total_s"] = stats.get("pass_total_s", 0.0) + took
                    stats["max_pass_s"] = max(stats.get("max_pass_s", 0.0), took)
                    if grant is None and move_check.refusal is not None:
                        cause = move_check.refusal
                        break
                    if grant is not None:
                        if (
                            last_evidence is not None
                            and grant.evidence_wall_s < last_evidence
                        ):
                            # The Thor would ignore it.
                            cause = _types().SafetyViolation(
                                "the obstacle grid's camera-frame time went back; "
                                "the permits were stopped"
                            )
                            break
                        link.send_permit(
                            move_id,
                            grant.through_index,
                            grant.evidence_wall_s,
                            settings.life_s,
                            pinned,
                        )
                        if last_evidence is None:
                            stats["first_permit_after_s"] = time.monotonic() - began
                            self._point(MARK_FIRST_PERMIT)
                        last_evidence = grant.evidence_wall_s
                        stats["permits"] += 1
                        stats["through"] = grant.through_index
                    # Behind the PERMIT of this pass: nothing here delays it.
                    self._note_progress(stats, sent, count, started - began)
                # A fresh state without the move: its RESULT is due.
                self._pause(move_id)
        except (LinkLost, StaleState) as error:
            self._latched = True
            cause = RobotLinkError(f"{error}; the permits were stopped")
        except BaseException as error:
            # APAI's SignalRequest (the first Ctrl+C) or an error of ours:
            # cancel first, then it goes on up.
            cause = error
        stats["approved"] = move_check.approved_through + 1
        stats["sent_seen"] = sent
        cancelled = time.monotonic()
        if last_evidence is not None:
            # The cancel: no setpoint index is at or below -1. It quotes the
            # evidence of the last PERMIT and the numbers the move began with.
            try:
                link.send_permit(move_id, -1, last_evidence, settings.life_s, pinned)
                stats["cancelled"] = True
            except Exception:
                # The cause of the cancel goes on up, not this.
                _log.exception("the cancel PERMIT for %s was not sent", move_id)
        # Without a PERMIT the Thor refuses the move when its first wait ends.
        try:
            result = link.wait_final(move_id, settings.cancel_wait_s)
            stats["cancel_to_result_s"] = time.monotonic() - cancelled
        except (LinkLost, CommandTimeout):
            result = None
        return result, cause

    # What a move measures. None of it raises, waits or writes: the facts go
    # to `measure` when stream_to() ends.

    def _mark(self, name: str) -> None:
        """A point of a move's permit loop, one of MARKS; nothing happens here.

        The launcher's timing wraps this method and shows each call as a span.
        """

    def _point(self, name: str) -> None:
        try:
            self._mark(name)
        except Exception:
            self._measure_trouble()

    def _note_progress(
        self, stats: dict[str, Any], sent: int, count: int, after_s: float
    ) -> None:
        """Note when the count of setpoints the Thor reports as sent rose.

        The first time it is 1 or more, the last time it rose, and the first
        time it is the whole move: a move that settles at once ends before a
        ROBOT_STATE shows that.
        """
        if sent > stats.get("sent_seen", 0):
            stats["sent_seen"] = sent
            stats["last_progress_after_s"] = after_s
        if sent >= 1 and "first_sent_after_s" not in stats:
            stats["first_sent_after_s"] = after_s
            self._point(MARK_FIRST_SETPOINT)
        if sent >= count and "all_sent_after_s" not in stats:
            stats["all_sent_after_s"] = after_s
            self._point(MARK_ALL_SENT)

    def _note_move(self, stats: dict[str, Any], ended: str) -> None:
        """Hand over what a move measured; nothing for a call that measured nothing."""
        if stats:
            stats["ended"] = ended
            self._note(MEASURED_MOVE, stats)

    def _note(self, kind: str, facts: dict[str, Any]) -> None:
        """Call `measure`; a failure there is logged once and costs nothing else."""
        if self._measure is None:
            return
        try:
            self._measure(kind, facts)
        except Exception:
            self._measure_trouble()

    def _measure_trouble(self) -> None:
        if not self._measure_failed:
            self._measure_failed = True
            _log.warning(
                "a measurement of the arm adapter failed; the run goes on without it",
                exc_info=True,
            )

    def _live_gate(self) -> BaseException | None:
        """Why the permits must stop now; None while they may go on."""
        types = _types()
        harness = self._check.harness
        if getattr(harness, "estopped", False):
            return types.SafetyViolation("e-stop latched")
        halted = getattr(harness, "halted", None)
        if halted is not None:
            return types.MotionHalted(f"halted: {halted}")
        if self._latched:
            return types.SafetyViolation("arm stop latched")
        lost = self._link.lost_reason()
        if lost is None:
            return None
        self._latched = True
        try:
            self._link.state()
        except (LinkLost, StaleState):
            return RobotLinkError(lost)
        return types.MotionHalted(f"the agent no longer owns the arm: {lost}")

    def _move_ending(
        self,
        result: Result | None,
        cause: BaseException | None,
        count: int,
        monitor: Any,
    ) -> bool:
        """What APAI gets for the final RESULT of a move (or for none)."""
        types = _types()
        if cause is not None and not isinstance(cause, Exception):
            raise cause
        if result is None:
            assert cause is not None
            raise cause
        reason, detail = result.reason, result.detail or {}
        outcome = detail.get("outcome")
        sent = detail.get("sent")
        sent = int(sent) if _is_number(sent) else 0
        if monitor is not None:
            monitor.commands = sent
            if result.status != "completed" and outcome != "contact":
                monitor.hold = "last command (stream aborted)"
        if reason.startswith(LOST):
            self._latched = True
            raise RobotLinkError(reason)
        if result.status == "failed" and outcome == "contact":
            raise self._contact(result, monitor)
        if reason.startswith(("fault: ", "robot app error")):
            self._latched = True
            raise RobotLinkError(reason)
        if result.status == "cancelled":
            self._mirror()
            raise types.MotionHalted(f"the robot app stopped the move: {reason}")
        if cause is not None:
            raise cause
        if result.status == "completed":
            return True
        if result.status == "failed" and outcome == "not_settled":
            return False
        if result.status == "failed" and outcome == "late":
            raise types.SafetyViolation(
                f"the robot app ended the move late at setpoint {sent} of {count}: "
                f"{reason}"
            )
        if result.status == "refused":
            if sent == 0:
                _log.warning(
                    "refused after the first permit with nothing sent: %s", reason
                )
            raise types.SafetyViolation(
                f"the robot app ended the move at setpoint {sent} of {count}: {reason}"
            )
        raise RobotLinkError(f"the move ended {result.status}: {reason}")

    def _contact(self, result: Result, monitor: Any) -> BaseException:
        """ContactDetected for a contact RESULT, with the monitor's record filled."""
        text = str((result.detail or {}).get("contact") or result.reason)
        end_q = None if result.end_q is None else [float(v) for v in result.end_q]
        event: dict[str, Any]
        found = CONTACT_TEXT.search(text)
        try:
            event = {
                "joint": int(found.group(1)),
                "error_rad": float(found.group(2)),
                "duration_ms": float(found.group(3)),
            }
        except (AttributeError, ValueError):
            event = {
                "joint": "unknown",
                "error_rad": 0.0,
                "duration_ms": 0.0,
                "parsed": False,
            }
        event["phase"] = "stream"
        event["motion"] = getattr(monitor, "label", "move")
        event["measured_q"] = end_q
        held = HOLD_FAILED not in result.reason
        if held:
            event["hold_q"] = end_q
        if monitor is not None:
            monitor.tripped = event
            monitor.hold = (
                "measured pose (reported by the robot app)"
                if held
                else "last command (the hold at the measured pose failed)"
            )
        if "contact detected" not in text:
            text = f"contact detected by the robot app: {text}"
        return _types().ContactDetected(text, event)

    def _fresh_state(self) -> Any:
        """A pose from readings taken after this call began; blocks for them."""
        link, settings = self._link, self._settings
        types = _types()
        try:
            mark = link.state(settings.state_max_age_s).clock_mono_s
            end = time.monotonic() + settings.fresh_wait_s
            while True:
                state = link.state(settings.state_max_age_s)
                q = _pose(state.joints, self.n_joints, newer_than=mark)
                if q is not None:
                    return types.RobotState(
                        q=np.asarray(q, dtype=float),
                        gripper_pos=0.0,
                        gripper_valid=False,
                        t=time.monotonic(),
                    )
                left = end - time.monotonic()
                if left <= 0:
                    break
                link.wait_change(min(left, _FRESH_LOOK_S))
        except (LinkLost, StaleState) as error:
            raise types.SafetyViolation(
                f"no fresh joint readings ({error}); no motion sent"
            ) from None
        raise types.SafetyViolation("no fresh joint readings; no motion sent")

    def _grasp_region(self) -> GraspRegion | None:
        """The workstation checker's open grasp region, for the MOVE."""
        exempt = getattr(self._check.harness, "_grasp_exempt", None)
        if exempt is None:
            return None
        center = exempt[0]
        return GraspRegion(
            center_xy=(float(center[0]), float(center[1])),
            radius_m=float(exempt[1]),
            z_min=float(exempt[2]),
            z_max=float(exempt[3]) if len(exempt) > 3 else None,
        )

    # Gates and the mirror of the Thor's state

    def _refuse_unless_task(self, what: str) -> None:
        """The refusals before anything is read or sent."""
        violation = _types().SafetyViolation
        if self._latched:
            raise violation(f"arm stop latched; no {what} sent")
        # APAI's jaw close does not pass its SafeArm, which looks at this latch.
        if self._check is not None and getattr(self._check.harness, "estopped", False):
            raise violation(f"e-stop latched; no {what} sent")
        if not self._task:
            raise violation(
                f"the agent does not own the arm; motion needs a confirmed task; "
                f"no {what} sent"
            )
        lost = self._link.lost_reason()
        if lost is not None:
            self._latched = True
            raise violation(f"the agent no longer owns the arm: {lost}; no {what} sent")

    def _mirror(self) -> None:
        """Latch the adapter's stop when the pinned ownership no longer stands."""
        if self._link.lost_reason() is not None:
            self._latched = True

    # Jaws

    def _opening(self, position: float) -> float:
        """A jaw position in profile units as an opening, 0 closed to 1 open."""
        span = self._grip_open - self._grip_closed
        return min(1.0, max(0.0, (float(position) - self._grip_closed) / span))

    def _position(self, opening: float) -> float:
        span = self._grip_open - self._grip_closed
        return self._grip_closed + float(opening) * span

    def _jaw_reading(self, state: RobotState) -> tuple[float, bool]:
        """(jaw position, whether it is known) of a ROBOT_STATE.

        Known only when the reading is valid, not older than the last arm
        setpoint and at most fresh_s old on the Thor's clock.
        """
        jaws = state.gripper
        known = (
            jaws.valid is True
            and _is_number(jaws.opening)
            and _is_number(jaws.t)
            and (state.target_t is None or jaws.t >= state.target_t)
            and state.clock_mono_s - jaws.t <= self._settings.fresh_s
        )
        return (self._position(jaws.opening), True) if known else (0.0, False)

    def _jaw_before(self) -> float | None:
        """The jaw opening before a command, else the last target; None if unknown."""
        try:
            state = self._link.state()
        except (LinkLost, StaleState):
            return None
        jaws = state.gripper
        if jaws.valid is True and _is_number(jaws.opening):
            return float(jaws.opening)
        return float(jaws.last_target) if _is_number(jaws.last_target) else None

    def _only_defaults(self, routine: str, **given: Any) -> None:
        """Refuse values the Thor's driver would not run this routine with.

        The GRIPPER carries the positions and the effort; the driver's other
        parameters stay the driver's own, which are this method's defaults.
        """
        defaults = inspect.signature(getattr(RobotLinkArm, routine)).parameters
        for name, value in given.items():
            if value != defaults[name].default:
                raise ValueError(
                    f"{routine}: {name}={value!r} is not sent over the robot "
                    f"link; the Thor's driver uses its own "
                    f"({defaults[name].default!r}); no jaw command sent"
                )

    def _jaw(
        self,
        opening: float,
        effort: float,
        mode: str,
        scout_opening: float | None = None,
    ) -> Result:
        """Send one GRIPPER and wait for its final RESULT.

        scout_opening is the scout stage of the driver's two-stage close.
        Without a RESULT in result_timeout_s the adapter gives up and sends a
        STOP, also inside skill_scope(): the robot app then starts nothing of
        the command, and holds the jaws lightly where a running one left them.
        No STOP is sent once a fresh ROBOT_STATE shows that the ownership has
        ended: the robot app refuses the command by its numbers then, and a
        STOP, which quotes none, would end a person's return to rest or drive.
        """
        timeout_s = self._settings.result_timeout_s
        effort = min(1.0, max(0.0, float(effort)))
        facts: dict[str, Any] = {"mode": mode, "opening": opening, "effort": effort}
        if scout_opening is not None:
            facts["scout_opening"] = scout_opening
        self.last_jaw = facts
        self._note(MEASURED_JAW_BEGIN, facts)
        began = time.monotonic()
        try:
            result = self._link.gripper(opening, effort, mode, timeout_s, scout_opening)
            facts["sent_to_result_s"] = time.monotonic() - began
            detail = result.detail if isinstance(result.detail, dict) else {}
            facts["status"] = result.status
            facts["outcome"] = detail.get("outcome")
            facts["result_opening"] = detail.get("opening")
            self._await_later_states()
            return result
        except LinkLost as error:
            self._latched = True
            facts["status"] = "link lost"
            raise RobotLinkError(str(error)) from error
        except CommandTimeout:
            self._latched = True
            facts["status"] = "no RESULT"
            told = f"no RESULT for the jaw command within {timeout_s:g} s"
            if self._ownership_ended():
                raise RobotLinkError(
                    f"{told}; the agent no longer owns the arm, so no STOP was sent"
                ) from None
            self._link.stop(
                f"the agent gave up on a jaw command: no RESULT within {timeout_s:g} s",
                AGENT_SENDER,
            )
            raise RobotLinkError(f"{told}; a STOP was sent") from None
        finally:
            facts["waited_s"] = time.monotonic() - began
            self._note(MEASURED_JAW, facts)

    def _ownership_ended(self) -> bool:
        """Whether a fresh ROBOT_STATE shows that the pinned ownership has ended.

        False when that cannot be told: without a pin, without a fresh state.
        """
        pinned = self._link.pinned
        if pinned is None:
            return False
        try:
            state = self._link.state(self._settings.state_max_age_s)
        except (LinkLost, StaleState):
            return False
        shown = (state.session_id, state.owner, state.owner_number, state.stop_number)
        return shown != (
            pinned.session_id,
            "agent",
            pinned.owner_number,
            pinned.stop_number,
        )

    def _await_later_states(self) -> None:
        """Wait for two ROBOT_STATEs newer than now, at most 0.3 s.

        The next get_state() then shows what the command just answered left
        behind, not the reading from before it.
        """
        link = self._link
        end = time.monotonic() + _REFLECT_S
        seen: set[float] = set()
        try:
            first = link.state_arrived_s()
            while len(seen) < 2:
                left = end - time.monotonic()
                if left <= 0:
                    return
                link.wait_change(left)
                link.state()  # a lost link ends the wait
                arrived = link.state_arrived_s()
                if arrived != first:
                    seen.add(arrived)
        except (LinkLost, StaleState):
            return

    def _jaw_ending(self, result: Result) -> str:
        """ "done", or "short" for a stall or a timeout; raises for the rest.

        GripperFault, as APAI's driver raises it and with the driver's own
        text, for a fault of the jaws that leaves the arm usable. "short" is
        what a driver that waits for the jaws itself reports, such as the
        simulated one.
        """
        types = _types()
        reason = result.reason
        if result.status == "completed":
            return "done"
        if reason.startswith((LOST, "fault: ")):
            self._latched = True
            raise RobotLinkError(reason)
        if result.status == "cancelled":
            self._mirror()
            raise types.MotionHalted(f"the robot app stopped the jaw command: {reason}")
        if result.status == "refused":
            if reason == _JAWS_UNREFERENCED or reason.startswith(_JAWS_UNREAD):
                raise types.GripperFault(reason)
            raise types.SafetyViolation(f"robot app refused the jaw command: {reason}")
        outcome = (result.detail or {}).get("outcome")
        if outcome == _GRIPPER_FAULT:
            raise types.GripperFault(reason or "the driver reported a gripper fault")
        if outcome in ("stalled", "timeout"):
            return "short"
        # A driver fault or an error of the jaw worker: the Thor has stopped.
        self._latched = True
        raise RobotLinkError(f"the jaw command failed: {reason}")

    # CHECKER_STATE

    def _hooked(self, original: Callable[..., Any], action: str) -> Callable[..., Any]:
        def hooked(*args: Any, **kwargs: Any) -> Any:
            out = original(*args, **kwargs)
            self._tell_checker_state(action)
            return out

        hooked.robot_link_action = action
        hooked.robot_link_original = original
        return hooked

    def _tell_checker_state(self, action: str) -> None:
        """Send a CHECKER_STATE inside a task; never raises, never waits."""
        try:
            if not self._task:
                return
            if action == CLOSE_GRASP:
                if not self._grasp_sent:
                    return
                self._grasp_sent = False
            command_id = self._link.checker_state(action)
            with self._ids_lock:
                self._checker_ids.append((action, command_id))
        except Exception as error:
            _log.warning("CHECKER_STATE %s was not sent: %s", action, error)

    def _log_checker_results(self) -> None:
        """Log the CHECKER_STATEs the robot app refused; forget the answered ones."""
        with self._ids_lock:
            waiting, self._checker_ids = self._checker_ids, []
        still: list[tuple[str, str]] = []
        for action, command_id in waiting:
            result = self._link.final_result(command_id)
            if result is None:
                still.append((action, command_id))
                continue
            if result.status != "completed":
                _log.warning(
                    "the robot app answered CHECKER_STATE %s with %s: %s",
                    action,
                    result.status,
                    result.reason,
                )
            self._link.forget(command_id)
        with self._ids_lock:
            self._checker_ids = still[-16:] + self._checker_ids

    # The comparisons of connect(), made again by the task gate for each task

    def check_welcome(self, welcome: Any) -> str | None:
        """The first thing of the WELCOME that does not fit this arm, or None."""
        settings = self._settings
        if welcome.protocol_version != PROTOCOL_VERSION:
            return (
                f"the robot app speaks protocol version {welcome.protocol_version}, "
                f"this program {PROTOCOL_VERSION}"
            )
        if len(welcome.joint_names) != self.n_joints:
            return (
                f"the robot app has {len(welcome.joint_names)} joints, the arm "
                f"profile {self.n_joints}"
            )
        if not math.isclose(welcome.setpoint_period_s, 1.0 / self.motion_rate_hz):
            return (
                f"the robot app's setpoint period is {welcome.setpoint_period_s:g} s, "
                f"the arm's rate is {self.motion_rate_hz:g} a second"
            )
        expected = settings.expected_profile_id
        if expected and welcome.profile_id != expected:
            return (
                f"the robot app's profile is {welcome.profile_id!r}, expected "
                f"{expected!r}"
            )
        theirs = welcome.camera_calibration_id
        if not theirs:
            if not settings.allow_unnamed_calibration:
                return "the robot app names no camera calibration"
        elif theirs != settings.calibration_id:
            return (
                f"the robot app's camera calibration is {theirs!r}, this "
                f"program's {settings.calibration_id!r}"
            )
        return None

    def _warn_joint_range(self, welcome: Any) -> None:
        low, high = self._settings.joint_min, self._settings.joint_max
        if low is None or high is None:
            return
        outside = [
            index
            for index in range(min(self.n_joints, len(low), len(high)))
            if low[index] < welcome.joint_min[index]
            or high[index] > welcome.joint_max[index]
        ]
        if outside:
            _log.warning(
                "the workstation's joint range reaches outside the robot app's on "
                "joint index %s: such a setpoint will be refused there",
                ", ".join(str(index) for index in outside),
            )


def arm_factory(
    link: AgentLink,
    settings: ArmSettings | None = None,
    joint_feed: Any = None,
    measure: Measure | None = None,
) -> Callable[..., Any]:
    """make_arm(cfg, kinematics=None) for APAI: RobotLinkArm for type robot_link.

    Any other type goes to APAI's own factory. The arms it built are in the
    returned function's .made.
    """
    made: list[Any] = []

    def make_arm(cfg: Any, kinematics: Any = None) -> Any:
        if _cfg_get(cfg, "type") != ARM_TYPE:
            from apai.control.arm_base import make_arm as apai_make_arm

            return apai_make_arm(cfg, kinematics)
        adapter = RobotLinkArm(
            cfg,
            kinematics,
            link=link,
            settings=settings,
            joint_feed=joint_feed,
            measure=measure,
        )
        made.append(adapter)
        return adapter

    make_arm.made = made
    return make_arm


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    """A value of an APAI Cfg or of a dict."""
    if cfg is None:
        return default
    get = getattr(cfg, "get", None)
    value = get(name) if callable(get) else getattr(cfg, name, None)
    return default if value is None else value


def _is_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def _pose(
    samples: Sequence[Any], n_joints: int, newer_than: float | None = None
) -> list[float] | None:
    """The pose of the newest reading per joint; None unless all are usable.

    With newer_than, a reading counts only when its read ended after that
    time of the Thor's monotonic clock.
    """
    pose: list[float | None] = [None] * n_joints
    for sample in samples:
        if not (0 <= sample.joint < n_joints and sample.valid):
            continue
        if not _is_number(sample.value):
            continue
        if newer_than is not None and not sample.t_end > newer_than:
            continue
        pose[sample.joint] = float(sample.value)
    return None if any(value is None for value in pose) else pose
