# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The arm APAI drives on the workstation: a stand-in for its motor driver.

RobotLinkArm sends no setpoint. A move becomes one MOVE and then PERMITs: the
robot app on the Thor paces the setpoints and sends each one only while a
permit covers it. A permit covers a setpoint only after APAI's checker callback
cleared it on the obstacle grid whose camera-frame time the permit quotes (the
check ahead). The adapter sends MOVE, PERMIT, GRIPPER, CHECKER_STATE and STOP,
and nothing that releases the motors.

APAI is imported inside functions, so the module imports without it.
"""

from __future__ import annotations

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
    MOVE_ARMED,
    MOVE_SENDING,
    PROTOCOL_VERSION,
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
# How long the end of a move waits for a ROBOT_STATE that shows it, in s.
_REFLECT_S = 0.3
# The longest one wait for fresh joint readings sleeps, in s.
_FRESH_LOOK_S = 0.05


class RobotLinkError(RuntimeError):
    """A fault of the robot connection or of the robot app, not a refusal."""


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

    def start_tol(self, period_s: float) -> float:
        """The start tolerance a MOVE quotes.

        The Thor checks the first setpoint from the pose it measured, so the
        tolerance leaves half of one period's travel at its speed limit.
        """
        return min(self.start_tol_rad, 0.5 * self.agent_max_joint_speed * period_s)


def _types() -> Any:
    import apai.types

    return apai.types


_arm_type_lock = threading.Lock()
_arm_type: type | None = None


def _arm_type_on_apai() -> type:
    """RobotLinkArm as a subclass of APAI's ArmBase; made on first use."""
    global _arm_type
    with _arm_type_lock:
        if _arm_type is None:
            from apai.control.arm_base import ArmBase

            class _OnArmBase(RobotLinkArm, ArmBase):
                __doc__ = RobotLinkArm.__doc__

            _OnArmBase.__name__ = _OnArmBase.__qualname__ = "RobotLinkArm"
            _OnArmBase.__module__ = __name__
            _arm_type = _OnArmBase
        return _arm_type


class _FreshReader:
    """What prepare_stream reads the start pose from: fresh readings only."""

    def __init__(self, read: Callable[[], Any]) -> None:
        self.get_state = read


class RobotLinkArm:
    """APAI's arm on the robot link. RobotLinkArm(...) builds an ArmBase subclass.

    Entry points for the launcher, which APAI never calls: bind_checker(),
    open_task(), close_task() and skill_scope().
    """

    n_joints = 6
    motion_rate_hz = 50.0
    # Teardown: APAI neither parks this arm nor cuts its torque.
    shutdown_park_required = False
    disconnect_preserves_drive_state = True
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
    ) -> None:
        self._cfg = cfg
        self.kin = kinematics
        self._link = link
        self._settings = settings or ArmSettings()
        self._joint_feed = joint_feed
        n_joints = _cfg_get(cfg, "n_joints")
        self.n_joints = 6 if n_joints is None else int(n_joints)
        jaws = _cfg_get(cfg, "gripper") or {}
        self._grip_open = float(_cfg_get(jaws, "open_pos", 0.0))
        self._grip_closed = float(_cfg_get(jaws, "closed_pos", 1.0))
        if self._grip_open == self._grip_closed:
            raise ValueError("the arm profile's jaws have no travel")
        self.connected = False
        self._stopped = False
        self.close_stages_skipped = 0
        self._check: CheckAhead | None = None
        self._task = False
        self._skill = threading.local()
        self._grasp_sent = False  # a MOVE of this task carried a grasp region
        self._checker_ids: list[tuple[str, str]] = []  # (action, command id)
        self._ids_lock = threading.Lock()
        # What the last move's permit loop did; for tests and the bench.
        self.last_move: dict[str, Any] = {}

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
        self._stopped = False
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
        problem = self._mismatch(welcome)
        if problem is None and not state.torque_on:
            problem = "the robot app reports the motors off; restart the robot app"
        if problem is not None:
            raise RobotLinkError(problem)
        self._warn_joint_range(welcome)
        self.connected = True

    def disconnect(self) -> None:
        """Nothing is sent: the robot app keeps holding the arm."""
        self.connected = False

    def get_state(self) -> Any:
        """The newest published pose; never blocks. RuntimeError when it is unusable."""
        settings = self._settings
        try:
            state, arrived = self._link.state_and_arrival(settings.state_max_age_s)
        except (LinkLost, StaleState) as error:
            raise RuntimeError(f"robot link feedback lost: {error}") from error
        q = _pose(state.joints, self.n_joints)
        if q is None:
            raise RuntimeError(
                "robot link feedback lost: a joint has no valid reading in the "
                "newest ROBOT_STATE"
            )
        age = state.health.get("reading_age_s")
        if not _is_number(age) or age > settings.state_max_age_s:
            raise RuntimeError(
                f"robot link feedback lost: the joint readings are {age!r} s old, "
                f"over {settings.state_max_age_s:g} s"
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

    def set_gripper(self, pos: float, effort: float = 1.0) -> None:
        """One plain GRIPPER; blocks until its final RESULT."""
        self._refuse_unless_task("jaw command")
        opening = self._opening(pos)
        before = self._jaw_before()
        result = self._jaw(opening, effort, "plain")
        if self._jaw_ending(result) == "short":
            if before is None or opening > before + 1e-9:
                raise _types().GripperFault("the jaws were blocked while opening")

    def stop(self) -> None:
        """Latch the adapter's stop and end a running move; never raises.

        Inside skill_scope() on this thread nothing is sent: the agent keeps
        the arm. Anywhere else it is a person's STOP.
        """
        try:
            self._stopped = True
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
            return mask_pose(self._joint_feed, self.get_state)
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
        raise _types().SafetyViolation(
            "joint paths are not supported over the robot link; no motion sent"
        )

    def close_gripper_two_stage(
        self,
        width_frac_stage1: float = 0.5,
        width_frac_stage2: float = 0.85,
        effort: float = 0.6,
        stall_vel: float = 0.05,
        timeout_s: float = 3.0,
    ) -> float | None:
        """The grasp close: one GRIPPER that closes until the jaws stall.

        Returns the measured end position in profile units, or None. A close
        never opens: a target that is not more closed than the last one sends
        nothing.
        """
        self._refuse_unless_task("jaw command")
        opening = min(1.0, max(0.0, 1.0 - float(width_frac_stage2)))
        last = self._last_jaw_target()
        if last is not None and opening >= last - 1e-9:
            self.close_stages_skipped = 2
            return None
        result = self._jaw(opening, effort, "close_until_contact")
        if self._jaw_ending(result) == "short":
            # A stall is "completed" in this mode; what is left is a timeout.
            raise _types().GripperFault(result.reason or "the jaw close timed out")
        self.close_stages_skipped = 0
        measured = result.detail.get("opening")
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
        callback for no setpoint.
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

        from apai.control.arm_base import prepare_stream
        from apai.control.motion_profile import min_jerk

        start, target = prepare_stream(
            _FreshReader(self._fresh_state),
            target,
            duration_s,
            preflight,
            before_stream,
        )
        q = sample_move(start, target, duration_s, rate_hz, min_jerk)
        if len(q) > welcome.max_move_setpoints:
            raise violation(
                f"the move has {len(q)} setpoints, the robot app takes at most "
                f"{welcome.max_move_setpoints}; no motion sent"
            )
        check.wait_fresh_grid(settings.start_grid_age_s, settings.start_grid_wait_s)
        evidence = check.vet_whole(start, q)
        tol = settings.start_tol(period)
        room = settings.agent_max_joint_speed * period - tol
        if largest_step(start, q[0]) > room:
            raise violation(
                f"the first setpoint is {largest_step(start, q[0]):.4f} rad from "
                f"the start pose; the robot app's speed limit leaves {room:.4f} "
                "rad for it; no motion sent"
            )
        if contact_monitor is not None:
            # From here a TypeError must not make SafeArm send the move again.
            contact_monitor.started = True
        try:
            return self._move(start, q, tol, evidence.wall_s, approve, contact_monitor)
        except TypeError as error:
            raise RobotLinkError(
                f"error in the arm adapter during a move: {error!r}"
            ) from error

    # A move

    def _move(
        self,
        start: np.ndarray,
        q: list[np.ndarray],
        tol: float,
        evidence_wall_s: float,
        approve: Any,
        monitor: Any,
    ) -> bool:
        link, settings = self._link, self._settings
        violation = _types().SafetyViolation
        move_id = link.new_id()
        region = self._grasp_region()
        began = time.monotonic()
        self.last_move = {"move_id": move_id, "setpoints": len(q), "permits": 0}
        # The numbers every PERMIT of this move quotes, also after close_task().
        pinned = link.pinned
        if pinned is None:
            raise violation("no ownership is pinned; no motion sent")
        try:
            link.send_move(
                move_id, start, tol, q, region, evidence_wall_s, settings.expires_in_s
            )
        except LinkLost as error:
            self._stopped = True
            raise RobotLinkError(str(error)) from error
        except (StaleState, ProtocolError, RuntimeError) as error:
            raise violation(
                f"the MOVE could not be sent: {error}; no motion sent"
            ) from None
        if region is not None:
            self._grasp_sent = True
        try:
            self._await_armed(move_id, began)
            result, cause = self._permit_loop(move_id, pinned, start, q, approve, began)
        except BaseException:
            link.forget(move_id)
            raise
        if result is not None:
            link.wait_reflected(move_id, _REFLECT_S)
        link.forget(move_id)
        return self._move_ending(result, cause, len(q), monitor)

    def _await_armed(self, move_id: str, began: float) -> None:
        """Admission, then the wait until the robot app shows the move as armed.

        Raises when the move ended first: nothing was sent and the checker
        callback saw no setpoint.
        """
        link, settings = self._link, self._settings
        try:
            first = link.wait_first(move_id, settings.admit_wait_s)
        except LinkLost as error:
            self._stopped = True
            raise RobotLinkError(str(error)) from error
        except CommandTimeout:
            raise RobotLinkError(
                f"no RESULT for the MOVE within {settings.admit_wait_s:g} s; no "
                "permit was sent"
            ) from None
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
                self._stopped = True
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

    def _ended_unarmed(self, result: Result) -> BaseException:
        """What a move's final RESULT means when it came before "armed"."""
        types = _types()
        reason = result.reason
        if reason.startswith(LOST):
            self._stopped = True
            return RobotLinkError(reason)
        if result.status == "cancelled":
            self._mirror()
            return types.MotionHalted(f"the robot app stopped the move: {reason}")
        return types.SafetyViolation(f"robot app refused the move: {reason}")

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
        deadline = time.monotonic() + (
            settings.first_wait_s
            + count * float(link.welcome.setpoint_period_s) * 1.1
            + settings.settle_timeout_s
            + 3.0
        )
        move_check = self._check.begin(approve, start, q)
        last_evidence: float | None = None  # of the last PERMIT sent
        sent = 0
        cause: BaseException | None = None
        try:
            while True:
                result = link.final_result(move_id)
                if result is not None:
                    stats["approved"] = move_check.approved_through + 1
                    stats["sent_seen"] = sent
                    return result, None
                if time.monotonic() >= deadline:
                    self._stopped = True
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
                    sent = max(sent, int(state.move_index))
                    started = time.monotonic()
                    grant = move_check.step(sent, reach)
                    took = time.monotonic() - started
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
                        last_evidence = grant.evidence_wall_s
                        stats["permits"] += 1
                        stats["through"] = grant.through_index
                # A fresh state without the move: its RESULT is due.
                link.wait_change(settings.renew_s)
        except (LinkLost, StaleState) as error:
            self._stopped = True
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

    def _live_gate(self) -> BaseException | None:
        """Why the permits must stop now; None while they may go on."""
        types = _types()
        harness = self._check.harness
        if getattr(harness, "estopped", False):
            return types.SafetyViolation("e-stop latched")
        halted = getattr(harness, "halted", None)
        if halted is not None:
            return types.MotionHalted(f"halted: {halted}")
        if self._stopped:
            return types.SafetyViolation("arm stop latched")
        lost = self._link.lost_reason()
        if lost is None:
            return None
        self._stopped = True
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
            self._stopped = True
            raise RobotLinkError(reason)
        if result.status == "failed" and outcome == "contact":
            raise self._contact(result, monitor)
        if reason.startswith(("fault: ", "robot app error")):
            self._stopped = True
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
        if self._stopped:
            raise violation(f"arm stop latched; no {what} sent")
        if not self._task:
            raise violation(
                f"the agent does not own the arm; motion needs a confirmed task; "
                f"no {what} sent"
            )
        lost = self._link.lost_reason()
        if lost is not None:
            self._stopped = True
            raise violation(f"the agent no longer owns the arm: {lost}; no {what} sent")

    def _mirror(self) -> None:
        """Latch the adapter's stop when the pinned ownership no longer stands."""
        if self._link.lost_reason() is not None:
            self._stopped = True

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

    def _last_jaw_target(self) -> float | None:
        try:
            last = self._link.state().gripper.last_target
        except (LinkLost, StaleState):
            return None
        return float(last) if _is_number(last) else None

    def _jaw(self, opening: float, effort: float, mode: str) -> Result:
        """Send one GRIPPER and wait for its final RESULT."""
        timeout_s = self._settings.result_timeout_s
        effort = min(1.0, max(0.0, float(effort)))
        try:
            result = self._link.gripper(opening, effort, mode, timeout_s)
            self._await_later_states()
            return result
        except LinkLost as error:
            self._stopped = True
            raise RobotLinkError(str(error)) from error
        except CommandTimeout:
            self._stopped = True
            raise RobotLinkError(
                f"no RESULT for the jaw command within {timeout_s:g} s"
            ) from None

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
        """ "done", or "short" for a stall or a timeout; raises for the rest."""
        types = _types()
        reason = result.reason
        if result.status == "completed":
            return "done"
        if reason.startswith((LOST, "fault: ")):
            self._stopped = True
            raise RobotLinkError(reason)
        if result.status == "cancelled":
            self._mirror()
            raise types.MotionHalted(f"the robot app stopped the jaw command: {reason}")
        if result.status == "refused":
            if reason == _JAWS_UNREFERENCED or reason.startswith(_JAWS_UNREAD):
                raise types.GripperFault(reason)
            raise types.SafetyViolation(f"robot app refused the jaw command: {reason}")
        if (result.detail or {}).get("outcome") in ("stalled", "timeout"):
            return "short"
        # A driver fault or an error of the jaw worker: the Thor has stopped.
        self._stopped = True
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

    # connect()'s comparisons

    def _mismatch(self, welcome: Any) -> str | None:
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
    link: AgentLink, settings: ArmSettings | None = None, joint_feed: Any = None
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
            cfg, kinematics, link=link, settings=settings, joint_feed=joint_feed
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
