# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The follower/leader handoff: one call before ``step()``, one after.

Everything in a clutched-preview frame loop that is not rendering, split out so the whole
engage sequence can be driven headlessly at frame rate -- the only way to exercise the
gate's pass-through conjunct, which a quasi-static drive passes by luck.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
from isaacteleop.retargeting_engine.interface import (
    ExecutionEvents,
    ExecutionState,
    TensorGroup,
    ValueInput,
)
from isaacteleop.retargeting_engine.interface.tensor_group_type import (
    OptionalType,
    TensorGroupType,
)
from isaacteleop.retargeting_engine.tensor_types import BoolType, ControllerInputIndex
from isaacteleop.retargeters.controller_pose import HandPose
from isaacteleop.retargeters.rate_limiter import EE_POSE_KEY
from isaacteleop.retargeters.SO101.clutch_retargeter import SO101ClutchRetargeter
from isaacteleop.retargeters.SO101.gripper_retargeter import GRIPPER_COMMAND_KEY

from . import frames
from .anchor import is_unit, yaw_of_axis
from .quaternion import MIN_QUAT_NORM
from .clutch_phase import ClutchPhase, PhaseMachine
from .engage_gate import KEY_ENGAGED, GateVerdict
from .harness import HarnessBand
from .quaternion import conjugate, multiply, rotate, to_matrix
from .ghost import GHOST_GROUP, ghost_bodies
from .so101_ghost import (
    GHOST_POINTING_AXIS,
    POS_HAND_FROM_GHOST,
    ghost_body_from_pose,
    grip_quat_from_ghost_body,
    pose_from_ghost_body,
)

LOG = logging.getLogger(__name__)

# One hand and no flag: the ghost is a right-handed gripper.
GHOST_HAND = ControllersSource.RIGHT

# Three pose channels, three jobs, none substitutes for another. HAND_POSE_KEY is Optional
# and is the only tracking-loss oracle; it drives the follower and the gate's rotation
# operand. COMMANDED_POSE_KEY is the reference the limiter band is measured against, and is
# required, so it can never signal loss. EE_POSE_KEY is the limiter's output.
HAND_POSE_KEY = "hand_pose"
COMMANDED_POSE_KEY = "commanded_ee_pose"

# The one hand frame, reaching the follower's drive, the gate's operand, the clutch's home
# and the ghost. Aim because only aim's -Z is a pointing ray; the cost is aim's
# device-specific ray origin. Everything relative to this frame must be re-derived when it
# changes. See README.md.
HAND_POSE = HandPose.AIM

# The B button. ControllerInput carries no field of that name: the OpenXR bindings put
# `/user/hand/right/input/b/click` on SECONDARY_CLICK
# (live_controller_tracker_impl.cpp:292). GHOST_HAND is the right controller.
_RESET_OFFSET_BUTTON = ControllerInputIndex.SECONDARY_CLICK

# The clutch's engage permission, the one external graph leaf this app feeds. TeleopSession
# validates every external leaf name is present in external_inputs on every step, so it is
# sent unconditionally. EngageGate is a plain object driven in after_step, not a graph node,
# so its permission reaches the clutch on the following step -- 14 ms at 72 Hz, against the
# gate's own 100 ms dwell -- and in exchange both operands come from the same frame.
ENGAGE_PERMITTED_LEAF = "engage_permitted"

# The clutch spells its own permission type inline, so this mirrors it rather than
# importing a name that does not exist.
PERMITTED_TYPE = OptionalType(
    TensorGroupType(
        SO101ClutchRetargeter.ENGAGE_PERMITTED_INPUT, [BoolType("permitted")]
    )
)


def permission(value: bool) -> TensorGroup:
    """One frame of the permission leaf. BoolType wants a real Python bool."""
    group = TensorGroup(PERMITTED_TYPE.inner_type)
    group[0] = bool(value)
    return group


def _pose(result, key: str) -> np.ndarray | None:
    """One of the pipeline's three 7-D pose channels, or None when it carries nothing.

    `is_none` is only one of the two spellings: it is hardcoded False on the required
    channels, and the limiter keeps a path that returns without writing
    (rate_limiter.py:424-427), so reading an unset tensor raises instead.
    """
    pose = result[key]
    if pose.is_none:
        return None
    try:
        tensor = pose[0]
    except ValueError:
        return None
    return np.asarray(np.from_dlpack(tensor), dtype=float)


# The wrist posture the gate will demand, in the operator's frame. _QUAT_HAND_FROM_GHOST
# cancels exactly through the engage handoff, so only the thumb direction carries it -- the
# tool direction reads 15.2 deg for every calibration swept, so it is a guard on Q_HOME.
# See README.md.
_OPERATOR_FORWARD = np.array((0.0, 0.0, -1.0))
# The -Z of whatever HAND_POSE names: the thumb axis on GRIP, the pointing ray on AIM. The
# angle only reads as "comfortable to hold" on the matching reading, so read the log's
# wording and not just the number.
_HAND_REPORT_AXIS = np.array((0.0, 0.0, -1.0))
# Warned on, never asserted: bounding the posture angle in a test would make
# EULER_HAND_FROM_GHOST_DEG untunable on a headset, the one place it can be tuned.
_POSTURE_LIMIT_DEG = 45.0

# Which axis of HAND_POSE the yaw drive reads: aim's -Z, the pointing ray. The choice only
# decides how much wrist roll and pitch leak into the arm's yaw. README.md tabulates the
# leak per candidate, measured on grip; aim's is unmeasured because the grip-to-aim
# transform is per-device. Re-measure on a headset before trusting it.
_HAND_FORWARD_AXIS = np.array((0.0, 0.0, -1.0))


def hand_facing_xr(q_hand_xyzw: np.ndarray) -> np.ndarray:
    """The XR yaw (wxyz) the operator's hand is facing, for the follower's base."""
    return yaw_of_axis(q_hand_xyzw, _HAND_FORWARD_AXIS)


def _log_hand_frames(result) -> bool:
    """Measure the device's grip-to-aim transform and print the calibration it implies.

    The transform is per-device, so one frame with both poses valid is what yields a
    replacement :data:`POS_HAND_FROM_GHOST`. Position only: porting the rotation would hand
    back the wrist pitch that solving it from Q_HOME removes. Never raises.
    """
    try:
        controller = result[GHOST_HAND]
        if controller.is_none or not (
            bool(controller[ControllerInputIndex.GRIP_IS_VALID])
            and bool(controller[ControllerInputIndex.AIM_IS_VALID])
        ):
            return False
        grip = np.asarray(
            controller[ControllerInputIndex.GRIP_ORIENTATION], dtype=float
        )
        aim = np.asarray(controller[ControllerInputIndex.AIM_ORIENTATION], dtype=float)
        grip_pos = np.asarray(
            controller[ControllerInputIndex.GRIP_POSITION], dtype=float
        )
        aim_pos = np.asarray(controller[ControllerInputIndex.AIM_POSITION], dtype=float)
        if min(np.linalg.norm(aim), np.linalg.norm(grip)) < MIN_QUAT_NORM:
            return False
    except (ValueError, IndexError, TypeError):
        return False

    # aim^-1 . grip, wxyz: what carries a direction from the GRIP frame into AIM's.
    inverse = conjugate(aim[[3, 0, 1, 2]])
    aim_from_grip = multiply(inverse, grip[[3, 0, 1, 2]])
    # The ghost offset lives in the hand's frame, so carrying it across costs both terms:
    # the origins' separation pulled back into the new frame, and the old offset turned by
    # the rotation above. Dropping the second leaves the ghost centimetres out.
    separation = rotate(grip_pos - aim_pos, inverse)
    turned = rotate(np.asarray(POS_HAND_FROM_GHOST, dtype=float), aim_from_grip)
    offset = separation + turned

    LOG.info(
        "hand frames: this device's aim pose sits %.0f deg and %.0f mm off its grip "
        "pose. HAND_POSE is %s, so for the ghost to sit where it did on GRIP, its "
        "position wants:",
        math.degrees(2.0 * math.acos(min(1.0, abs(float(aim_from_grip[0]))))),
        1000.0 * float(np.linalg.norm(grip_pos - aim_pos)),
        HAND_POSE.value.upper(),
    )
    LOG.info(
        "hand frames:   POS_HAND_FROM_GHOST = np.array((%.3f, %.3f, %.3f))", *offset
    )
    return True


def base_yaw_bias(arm) -> np.ndarray:
    """How far the base yaw must lead the hand for the jaw to face it (wxyz).

    Measured off the arm at startup, not authored: the jaw's offset from its base yaw
    follows from Q_HOME and upstream's chain. Both operands are yaws about +Y.
    """
    inverse = conjugate(arm.jaw_yaw_xr)
    bias = multiply(arm.base_yaw_xr, inverse)
    return bias


def log_grip_posture(arm) -> tuple[float, float]:
    """Invert the chain at Q_HOME and report the posture the gate will ask for.

    Both angles are un-yawed into the operator's frame, which is what lets this run before
    the anchor. Warns rather than raises: this app is the only place the calibration can
    be judged.
    """
    p_body, q_body = arm.gripper_pose_mj()
    ghost_axis = rotate(GHOST_POINTING_AXIS, q_body)
    # The operator's frame is the hand's yaw, which the base leads by base_yaw_bias.
    # Un-yawing by the base instead is wrong by 93 degrees once that bias is large.
    inverse_bias = conjugate(base_yaw_bias(arm))
    hand_yaw = multiply(arm.base_yaw_xr, inverse_bias)
    unyaw = conjugate(hand_yaw)

    def in_operator_frame(direction_xr):
        out = rotate(np.asarray(direction_xr, dtype=float), unyaw)
        return out

    tool = in_operator_frame(
        frames.xr_from_mj_pos(p_body + ghost_axis) - frames.xr_from_mj_pos(p_body)
    )
    hand_axis = in_operator_frame(
        pose_from_ghost_body(p_body, q_body, arm.hand_from_ghost)[:3, :3]
        @ _HAND_REPORT_AXIS
    )

    def ahead(direction):
        return math.degrees(
            math.acos(min(1.0, max(-1.0, float(direction @ _OPERATOR_FORWARD))))
        )

    LOG.info(
        "grip calib: at Q_HOME the tool points (%+.2f, %+.2f, %+.2f), %.0f deg off the "
        "operator's forward, and the gate will demand a hand whose %s axis is "
        "(%+.2f, %+.2f, %+.2f), %.0f deg off. XR axes, in the operator's frame. Only the "
        "SECOND depends on EULER_HAND_FROM_GHOST_DEG.",
        *tool,
        ahead(tool),
        "pointing" if HAND_POSE is HandPose.AIM else "thumb",
        *hand_axis,
        ahead(hand_axis),
    )
    # The hand axis only: the tool angle does not depend on the calibration, so a warning
    # on it would report a Q_HOME or mesh change under a misleading name.
    if ahead(hand_axis) > _POSTURE_LIMIT_DEG:
        LOG.warning(
            "grip calib: the gate will demand a hand held %.0f deg off neutral, past the "
            "%.0f deg that reads as a comfortable hold. Check EULER_HAND_FROM_GHOST_DEG "
            "-- it is the only constant this angle depends on.",
            ahead(hand_axis),
            _POSTURE_LIMIT_DEG,
        )
    return ahead(tool), ahead(hand_axis)


# The per-frame order is before_step(), step(), after_step(), render. The leaves
# before_step emits are one frame stale, deliberately: the gate needs a limiter band and an
# arm pose step N has not computed yet. That 14 ms costs nothing -- a denied latch stays
# OWED and fires on the first permitted frame.
class ClutchPreview:
    """The follower/leader handoff: one call before ``step()``, one after."""

    def __init__(
        self,
        twin,
        monitor,
        arm,
        clutch,
        gate,
        *,
        owns_clutch_home: bool = True,
        ghost_pose_key: str = EE_POSE_KEY,
    ) -> None:
        """Bind to one twin, one pair of tools, one clutch and the gate node.

        Measures the arm's yaw bias before ``anchor`` runs, so the reading is the arm's own
        and not a head's.

        Args:
            owns_clutch_home: Whether this preview supplies the clutch's home every
                non-engaged frame. Set it False when a real robot is on the other end of
                the clutch: the home must then come from that arm's measured EE.
            ghost_pose_key: Which channel places the leader ghost. Defaults to the
                limiter's output, which only coincides with the tool in the hand while the
                clutch runs in the operator's frame. With a real arm the clutch runs in the
                robot's frame and such a pose cannot be placed in the hand at all -- the
                rebase's translation is unknowable because an engage-relative clutch
                cancels it -- so those consumers pass :data:`HAND_POSE_KEY` and give up the
                ghost lagging the hand.
        """
        self._twin = twin
        self._monitor = monitor
        self._arm = arm
        self._clutch = clutch
        self._gate = gate
        self._owns_clutch_home = bool(owns_clutch_home)
        self._recentered = False
        self._ghost_pose_key = str(ghost_pose_key)
        # Named rather than discovered, so a renamed geom is an error.
        twin.declare_group(GHOST_GROUP, geoms=arm.ghost.geoms)
        self.phases = PhaseMachine()
        # The gate's own pre-first-step verdict, which reports that nothing has been
        # judged rather than reading as engageable.
        self.verdict = gate.verdict
        # The gate's extra conjunct, latched by after_step because before_step runs a
        # frame ahead of the limiter band it is read off.
        self._limiter_passing = False
        # Neither tool is drawn until the arm is anchored, so start both hidden rather
        # than relying on the first after_step to arrive.
        twin.publish(groups={GHOST_GROUP: False})
        # SO101GripperRetargeter's own released end, held until the first frame with a
        # usable hand pose refreshes it.
        self._closedness = 0.0
        # B is edge-triggered, so a held button resets the offset once.
        self._reset_held = False
        # The ghost body of the last usable hand pose, latched by after_step because
        # before_step runs a frame ahead of it. None until one arrives, and no latch can
        # be permitted before then: the gate reports `controller not tracked`.
        self._hand_body_mj: np.ndarray | None = None
        self._yaw_bias = base_yaw_bias(arm)
        # One reading is all it takes, and it needs a tracked controller, so it cannot
        # happen at construction.
        self._frames_logged = False
        LOG.info(
            "preview arm: the base leads the hand by %+.2f deg of yaw, measured off "
            "Q_HOME so the JAW faces where the controller does.",
            math.degrees(2.0 * math.atan2(self._yaw_bias[2], self._yaw_bias[0])),
        )

    def notify_reference_space_changed(self) -> None:
        """The runtime recentered; drop everything latched in the old reference space.

        Re-anchors the arm off the next head pose and forces one STOPPED frame, which
        re-arms the clutch's pending-latch sentinel. Without it a recentre mid-engagement
        steps the clutch's delta by the whole recenter transform, walking a real arm to the
        wrong pose at the clamp.
        """
        self._recentered = True
        self._arm.unanchor()

    def before_step(self, head: np.ndarray | None) -> tuple[dict, ExecutionEvents]:
        """Anchor the arm if it is not anchored, then build this frame's step() kwargs.

        Two rules on the home push: key it off the app's phase, never ``clutch.is_engaged``,
        which drops on four paths of which one is a real disengage; and take the position
        from the hand and the rotation from the gripper, since the gripper's own position
        would carry the preview's offset into the clutch's delta.
        """
        if head is not None and not self._arm.anchored:
            self._arm.anchor(head)
        if (
            self._owns_clutch_home
            and self.phases.phase is not ClutchPhase.ENGAGED
            and self._hand_body_mj is not None
        ):
            self._clutch.set_home_base_T_ee(
                pose_from_ghost_body(
                    self._hand_body_mj,
                    self._arm.gripper_pose_mj()[1],
                    self._arm.hand_from_ghost,
                )
            )
        # The reset pulse re-seeds the limiter's baseline on the first frame after a
        # disengage; without it the limiter rejects for ~30 frames (0.92 s). execution_state
        # is spelled out because ExecutionEvents defaults to UNKNOWN, which would make the
        # clutch silently never engage.
        reset = self.phases.reset_requested
        self.phases.reset_requested = False
        # One STOPPED frame re-arms the clutch's pending-latch sentinel, so the next
        # engage takes a fresh origin in the new reference space.
        recentered = self._recentered
        self._recentered = False
        # From the previous after_step, and False before the first one, so the clutch
        # cannot latch on a frame nothing has judged.
        return (
            {
                ENGAGE_PERMITTED_LEAF: {
                    ValueInput.VALUE: permission(self._gate.permitted)
                },
            },
            ExecutionEvents(
                reset=reset or recentered,
                execution_state=(
                    ExecutionState.STOPPED if recentered else ExecutionState.RUNNING
                ),
            ),
        )

    def after_step(self, result, dt: float) -> ClutchPhase:
        """Advance the phase, drive both tools, and re-evaluate the gate."""
        if not self._arm.anchored:
            # Nowhere to put the arm yet, so there is no reference and the gate blocks
            # on it. Draw neither tool. Returns before the phase machine, so a frame
            # with no placement is a frame that did not happen.
            self._show(follower_visible=False, ghost_visible=False)
            self._blocked(self._judge(None, None, dt))
            return self.phases.phase

        # HAND_POSE_KEY is the only channel that can report tracking loss.
        hand = _pose(result, HAND_POSE_KEY)
        commanded = _pose(result, COMMANDED_POSE_KEY)
        governed = _pose(result, EE_POSE_KEY)
        # Where the ghost goes, which is the limiter's output only while the clutch runs in
        # the operator's own frame. See ghost_pose_key.
        drawn = (
            governed
            if self._ghost_pose_key == EE_POSE_KEY
            else _pose(result, self._ghost_pose_key)
        )

        # ControllerPoseSource drops on the pose's IS_VALID; the clutch also disarms on a
        # non-finite pose and on a degenerate quaternion. Both folded in, so `hand` covers
        # the clutch's whole disarm set and the phase machine cannot read a bad frame as a
        # real disengage. `is_unit` is deliberately wider than the clutch's own norm floor:
        # a norm-0.5 quaternion clears that floor and then raises out of yaw_of_axis, on
        # the frame loop, ending the session.
        if hand is not None and (
            not np.all(np.isfinite(hand[:3])) or not is_unit(hand[3:7])
        ):
            hand = None

        # Latched, and refreshed only on a usable frame. SO101GripperRetargeter tests
        # `inp.is_none` alone, so it keeps articulating the trigger while GRIP_IS_VALID is
        # false, and a jaw swinging on a frozen body reads as "the gripper actuated".
        if hand is not None:
            self._closedness = float(result[GRIPPER_COMMAND_KEY][0])
            # Taken from the hand rather than read back off the arm, so the grip offset
            # cannot leak into an engagement the clutch composes as a delta.
            self._hand_body_mj = ghost_body_from_pose(hand, self._arm.hand_from_ghost)[
                0
            ]

        if not self._frames_logged:
            self._frames_logged = _log_hand_frames(result)

        # Before the phase advance, so a press takes effect on this frame rather
        # than the next.
        self._reset_offset(result)

        phase = self.phases.advance(
            is_engaged=self._clutch.is_engaged, hand_present=hand is not None, dt=dt
        )

        # Nothing moves the arm while ENGAGED -- it is frozen where it stood on the engage
        # frame, and the disengage edge lands DISENGAGED, so the drag resumes with no ramp.
        if phase is ClutchPhase.DISENGAGED and hand is not None:
            # The hand, not the governed pose, and raw XR rather than the ghost body:
            # preview_arm.py is free of the grip calibration and must stay that way.
            # The stick always adjusts position; face buttons do not remap it.
            stick_x, stick_y = self._stick(result)
            facing, base_yaw = self._yaws(hand[3:7])
            self._arm.drive(hand[:3], facing, base_yaw, stick_x, stick_y, dt)

        engaged = phase is ClutchPhase.ENGAGED
        self._show(follower_visible=not engaged, ghost_visible=engaged)

        self._limiter_passing = False
        if drawn is not None:
            # The body needs no tracking-loss gate: the clutch emits its held pose on
            # every disarm path. The jaw does, hence the latch above.
            self._twin.publish(
                bodies=ghost_bodies(
                    self._arm.ghost,
                    *ghost_body_from_pose(drawn, self._arm.hand_from_ghost),
                    self._closedness,
                )
            )
        if commanded is not None and governed is not None:
            # Classified on every governed frame, painted only while the ghost is the
            # tool on show: the band needs an unbroken baseline to tell a refused frame
            # from a clamped one.
            band = self._monitor.update(commanded, governed, paint=engaged)
            self._limiter_passing = band is HarnessBand.PASS_THROUGH

        # Last, so both operands are this frame's: the arm has already been dragged and the
        # band above is the one the gate's extra conjunct reads. The hand's raw XR
        # orientation, not the ghost body's -- the grip calibration rotates the tool in the
        # hand and would show up as a constant misalignment.
        self._blocked(self._judge(hand, self._arm.gripper_pose_mj()[1], dt))
        return phase

    def _judge(
        self, hand: np.ndarray | None, q_gripper_wxyz: np.ndarray | None, dt: float
    ) -> GateVerdict:
        """Re-evaluate the gate for this frame and return its verdict.

        The reference is the arm's own gripper orientation, which carries the hand's yaw put
        there by ``Follower.drive``, so the two cancel and the gate measures wrist pitch and
        roll. Both operands must go through ``grip_quat_from_ghost_body``: the gate's
        geodesic angle is meaningless across frames, and handing a scene-frame wxyz
        quaternion over raw is a 128-168 deg error. ``permits_engagement`` is the phase, not
        ``clutch.is_engaged``, which would deny the recovery latch after a brief dropout.
        """
        reference = None
        if q_gripper_wxyz is not None:
            q_xyzw = grip_quat_from_ghost_body(
                q_gripper_wxyz, self._arm.hand_from_ghost
            )
            reference = to_matrix(np.array([q_xyzw[3], *q_xyzw[:3]]))
        return self._gate.update(
            None if hand is None else hand[3:7],
            reference,
            engaged=self.phases.permits_engagement,
            app_ok=self._limiter_passing,
            dt=dt,
        )

    def _yaws(self, q_hand_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Controller facing and base heading (wxyz), separated by the measured bias."""
        facing = hand_facing_xr(q_hand_xyzw)
        return facing, multiply(facing, self._yaw_bias)

    def _stick(self, result) -> tuple[float, float]:
        """The right thumbstick's two raw axes, or a stick at rest.

        ``Follower.drive`` owns which way each one points.
        """
        controller = result[GHOST_HAND]
        if controller.is_none:
            # Absent before the tracker has a controller, and reading the group there
            # raises rather than reporting a stick at rest -- see `_reset_offset`.
            return 0.0, 0.0
        return (
            float(controller[ControllerInputIndex.THUMBSTICK_X]),
            float(controller[ControllerInputIndex.THUMBSTICK_Y]),
        )

    def _reset_offset(self, result) -> None:
        """B, on its rising edge: put the grip offset back to its authored value.

        Deliberately phase-free, and needs no pose: the offset is a constant in the
        anchor's frame.
        """
        # The controller group is Optional and absent before the tracker has one; reading it
        # there raises rather than returning a falsy button, which took the whole session
        # down at startup. Absent is "not pressed".
        controller = result[GHOST_HAND]
        pressed = not controller.is_none and bool(controller[_RESET_OFFSET_BUTTON])
        rising = pressed and not self._reset_held
        self._reset_held = pressed
        if not rising:
            return
        self._arm.reset_offset()
        LOG.info(
            "preview arm: grip offset reset to XR (%.2f, %.2f, %.2f).",
            *self._arm.grip_from_controller_xr,
        )

    def _show(self, *, follower_visible: bool, ghost_visible: bool) -> None:
        """The only place either tool's visibility is set. At most one is drawn."""
        self._arm.set_visible(follower_visible)
        self._twin.publish(groups={GHOST_GROUP: ghost_visible})

    def _blocked(self, verdict: GateVerdict) -> None:
        """Publish the gate's verdict: the arm's colour, and a log on transitions.

        Keyed on `verdict.keys`, never on the text it renders, and stored even on frames
        nothing is logged so a release is always heard.
        """
        previous_keys = self.verdict.keys
        self.verdict = verdict
        self._arm.set_engageable(verdict.ok)
        if verdict.keys == previous_keys or KEY_ENGAGED in verdict.keys:
            return
        LOG.info(
            "clutch: %s",
            "engageable"
            if verdict.ok
            else "blocked (" + "; ".join(verdict.blocked) + ")",
        )
