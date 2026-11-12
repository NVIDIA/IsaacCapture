# SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn the operator's body pose into G1 locomanipulation IK targets.

Each arm is rebuilt from the robot's shoulder with the robot's segment lengths and the operator's
segment directions. Do not import ``isaaclab`` here: the tests run this module without Isaac Sim,
a GPU or the device. The task, its config and the robot's measurements are in ``jmex_g1_tasks.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from isaaccapture.logging_config._core import logging_enabled
from isaaccapture.retargeters import (
    LocomotionFixedRootCmdRetargeter,
    LocomotionFixedRootCmdRetargeterConfig,
    TensorReorderer,
)
from isaaccapture.retargeting_engine.interface import (
    BaseRetargeter,
    ComputeContext,
    OptionalType,
    OutputCombiner,
    OutputSelector,
    RetargeterIO,
    RetargeterIOType,
    TensorGroupType,
)
from isaaccapture.retargeting_engine.tensor_types import (
    DLDataType,
    FullBodyInput,
    FullBodyInputIndex,
    NDArrayType,
)
from isaaccapture.schema import BodyJoint

from .plugin import DEFAULT_CHANNEL, DEFAULT_COLLECTION_ID

logger = logging.getLogger("isaaccapture.examples.jmex.jmex_g1_retargeting")


def info_or_print(log: logging.Logger, message: str, *args: object) -> None:
    """Log at info, or print under ``ISAACCAPTURE_LOGGING=off``, where nothing shows an info record."""
    if logging_enabled():
        log.info(message, *args)
    else:
        print(message % args if args else message)


#: The pipeline's output. Isaac Lab's teleop device looks for one literally named "action".
ACTION_OUTPUT = "action"

#: Position (3) plus orientation (4) per Pink frame task, in the order the action term reads.
POSITION_DIM = 3
ORIENTATION_DIM = 4
POSE_DIM = POSITION_DIM + ORIENTATION_DIM

#: Change of basis from the plugin's OpenXR axes (+X the operator's right, +Y up, -Z the way
#: they face; src/plugins/jmex/core/openxr_convention.cpp) to Isaac's (+X forward, +Y left,
#: +Z up). A proper rotation (determinant +1), which ``_hand_orientation`` relies on.
_OPENXR_TO_ISAAC = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)

#: The ``forearm_frame`` in the axes of the standard human skeleton's ``WRIST`` slot, per arm.
#: Columns: x out along the forearm, y the elbow's hinge, z = x cross y. In the slot's axes x is
#: -X on the left arm and +X on the right, and the hinge is -Y and +Y. Measured 2026-09-23: the
#: wrist-to-hand offset is rigid in these axes, 55.2 mm (sd 0.000 mm), along x to 0.00 deg. A
#: skeleton whose ``WRIST`` slot is laid out differently needs a new matrix, or the hands come
#: out a quarter turn off; nothing in the stream reports it.
_FOREARM_FRAME_IN_PUBLISHED_WRIST = {
    int(BodyJoint.LEFT_WRIST): np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ),
    int(BodyJoint.RIGHT_WRIST): np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ),
}

#: Below this, a segment carries no usable direction and the frame is treated as unusable.
_MIN_SEGMENT_LENGTH_M = 1e-4
#: Anchors and link lengths closer than this are one measurement, taken twice.
_SAME_MEASUREMENT_TOLERANCE_M = 1e-9

#: Frames in a row that may miss a slot before the model, not a bad frame, is blamed. One frame
#: is one environment step.
_INCOMPLETE_FRAMES_BEFORE_WARNING = 30
#: How often a condition that persists is repeated on the console.
_REPORT_EVERY_FRAMES = 300


def left_wrist_chain() -> tuple[int, ...]:
    """Body slots from the left shoulder out to the left wrist, anchor first."""
    return (
        int(BodyJoint.LEFT_SHOULDER),
        int(BodyJoint.LEFT_ELBOW),
        int(BodyJoint.LEFT_WRIST),
    )


def right_wrist_chain() -> tuple[int, ...]:
    """Body slots from the right shoulder out to the right wrist, anchor first."""
    return (
        int(BodyJoint.RIGHT_SHOULDER),
        int(BodyJoint.RIGHT_ELBOW),
        int(BodyJoint.RIGHT_WRIST),
    )


def forearm_frame(
    shoulder: Sequence[float],
    elbow: Sequence[float],
    wrist: Sequence[float],
    *,
    elbow_axis: Sequence[float],
) -> Rotation:
    """The frame a hand is turned relative to, built from an arm resting with its elbow bent.

    x runs elbow to wrist; y is ``elbow_axis`` made square to x and signed along (upper arm x
    forearm), whichever way the model signs its joint; z completes it. Raises ``ValueError`` if
    the forearm has no length, the elbow is straight, or the elbow's axis runs along the forearm.
    """
    shoulder, elbow, wrist = (
        np.asarray(point, dtype=np.float64) for point in (shoulder, elbow, wrist)
    )
    forearm = wrist - elbow
    if np.linalg.norm(forearm) < _MIN_SEGMENT_LENGTH_M:
        raise ValueError("the forearm has no length")
    distal = forearm / np.linalg.norm(forearm)
    bend = np.cross(elbow - shoulder, distal)
    if np.linalg.norm(bend) < _MIN_SEGMENT_LENGTH_M:
        raise ValueError("the elbow rests straight, so which way it bends is unknown")
    hinge = np.asarray(elbow_axis, dtype=np.float64)
    hinge = hinge - np.dot(hinge, distal) * distal
    if np.linalg.norm(hinge) < _MIN_SEGMENT_LENGTH_M:
        raise ValueError("the elbow's axis runs along the forearm")
    hinge = hinge / np.linalg.norm(hinge)
    if np.dot(hinge, bend) < 0.0:
        hinge = -hinge
    return Rotation.from_matrix(
        np.column_stack([distal, hinge, np.cross(distal, hinge)])
    )


@dataclass(frozen=True)
class PelvisPose:
    """The robot's pelvis, in the per-env world frame Isaac hands the action term.

    ``orientation_xyzw`` is scalar-last, as Isaac Lab 3.0 hands it over; a scalar-first one
    raises nothing and arrives rotated.
    """

    position: np.ndarray
    orientation_xyzw: np.ndarray

    @property
    def rotation(self) -> Rotation:
        """``orientation_xyzw`` as a ``Rotation``."""
        return Rotation.from_quat(np.asarray(self.orientation_xyzw, dtype=np.float64))

    def place(self, points_in_pelvis: np.ndarray) -> np.ndarray:
        """Re-express pelvis-frame points in the world frame."""
        return np.asarray(self.position, dtype=np.float64) + self.rotation.apply(
            points_in_pelvis
        )

    def compose(
        self, offset_in_pelvis: np.ndarray, orientation_in_pelvis: Rotation
    ) -> tuple[np.ndarray, np.ndarray]:
        """Re-express a pelvis-frame target in the world frame.

        The quaternion is the one of its two signs with w >= 0, so one orientation is always
        written the same way.
        """
        return (
            self.place(offset_in_pelvis),
            (self.rotation * orientation_in_pelvis).as_quat(canonical=True),
        )


def _unit_quaternion(value: Sequence[float], *, what: str) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (ORIENTATION_DIM,):
        raise ValueError(
            f"{what} must be four numbers (x, y, z, w), got {quaternion.shape}"
        )
    norm = float(np.linalg.norm(quaternion))
    # The action term builds its rotation matrix by dividing by the squared norm, so a zero
    # reaches the IK solver as NaN rather than as "unconstrained".
    if not np.isfinite(norm) or norm < _MIN_SEGMENT_LENGTH_M:
        raise ValueError(
            f"{what} must be a finite, non-zero quaternion, got {quaternion.tolist()}"
        )
    return quaternion / norm


@dataclass(frozen=True)
class FrameTargetCfg:
    """One Pink ``FrameTask``, and the chain of body slots whose last point is its target.

    The robot's numbers are measured off the live pinocchio model by the wiring module, in
    the pelvis frame at the robot's rest configuration.
    """

    name: str
    #: Body slots from the anchor out to this task's own (see ``arm_frame_targets``).
    chain: tuple[int, ...]
    #: The robot's own point that ``chain[0]`` stands for: its shoulder, for an arm.
    anchor_position_pelvis: tuple[float, float, float]
    #: Meters between consecutive frames of ``chain`` on the robot.
    robot_segment_lengths: tuple[float, ...]
    #: Where the task's frame rests: the target until a frame drives the chain.
    neutral_position_pelvis: tuple[float, float, float]
    #: How the task's frame rests. A task without a forearm frame never turns from it.
    neutral_orientation_pelvis_xyzw: tuple[float, float, float, float]
    #: The robot's ``forearm_frame`` in the task frame's own axes, which carries the
    #: operator's hand onto the task. Exactly the chains that end at a wrist have one.
    forearm_frame_in_task_xyzw: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if not self.chain:
            raise ValueError(f"{self.name}: a chain needs at least its anchor slot")
        if np.shape(self.anchor_position_pelvis) != (POSITION_DIM,):
            raise ValueError(
                f"{self.name}: the anchor must be three numbers (x, y, z), got "
                f"{np.shape(self.anchor_position_pelvis)}"
            )
        if len(self.robot_segment_lengths) != len(self.chain) - 1:
            raise ValueError(
                f"{self.name}: {len(self.chain)} slots need "
                f"{len(self.chain) - 1} link lengths, got {len(self.robot_segment_lengths)}"
            )
        if not all(
            np.isfinite(length) and length > 0.0
            for length in self.robot_segment_lengths
        ):
            raise ValueError(f"{self.name}: link lengths must be finite and positive")
        # Raises here, at construction, rather than emitting a NaN target later.
        _unit_quaternion(
            self.neutral_orientation_pelvis_xyzw,
            what=f"{self.name}: neutral orientation",
        )
        ends_at_wrist = self.chain[-1] in _FOREARM_FRAME_IN_PUBLISHED_WRIST
        if ends_at_wrist and self.forearm_frame_in_task_xyzw is None:
            raise ValueError(
                f"{self.name}: a chain that ends at a wrist needs a forearm frame to carry "
                "the operator's hand onto its task"
            )
        if not ends_at_wrist and self.forearm_frame_in_task_xyzw is not None:
            raise ValueError(
                f"{self.name}: only a chain that ends at a wrist takes a forearm frame; this "
                f"one ends at {BodyJoint(self.chain[-1]).name}"
            )
        if self.forearm_frame_in_task_xyzw is not None:
            _unit_quaternion(
                self.forearm_frame_in_task_xyzw, what=f"{self.name}: forearm frame"
            )

    @property
    def neutral_rotation(self) -> Rotation:
        """``neutral_orientation_pelvis_xyzw`` as a ``Rotation``."""
        return Rotation.from_quat(
            _unit_quaternion(
                self.neutral_orientation_pelvis_xyzw,
                what=f"{self.name}: neutral orientation",
            )
        )

    @property
    def forearm_frame_in_task(self) -> Rotation:
        """``forearm_frame_in_task_xyzw`` as a ``Rotation``; ``ValueError`` if None."""
        return Rotation.from_quat(
            _unit_quaternion(
                self.forearm_frame_in_task_xyzw,
                what=f"{self.name}: forearm frame",
            )
        )


def arm_frame_targets(
    chain: tuple[int, ...],
    *,
    anchor_position_pelvis: Sequence[float],
    robot_segment_lengths: Sequence[float],
    rest_poses_pelvis: Sequence[tuple[Sequence[float], Sequence[float]]],
    forearm_frame_in_task_xyzw: Sequence[float],
) -> tuple[FrameTargetCfg, ...]:
    """One task per slot of an arm's ``chain``, anchor first: the chain cut short at that slot.

    ``rest_poses_pelvis`` holds each slot's frame at rest as (position, xyzw). The task that
    ends the chain -- the wrist -- is the one that takes the forearm frame.
    """

    if len(rest_poses_pelvis) != len(chain):
        raise ValueError(
            f"{len(chain)} slots need {len(chain)} rest poses, got {len(rest_poses_pelvis)}"
        )

    def floats(values: Sequence[float]) -> tuple[float, ...]:
        return tuple(float(value) for value in values)

    return tuple(
        FrameTargetCfg(
            name=BodyJoint(chain[depth - 1]).name.lower(),
            chain=chain[:depth],
            anchor_position_pelvis=floats(anchor_position_pelvis),
            robot_segment_lengths=floats(robot_segment_lengths[: depth - 1]),
            neutral_position_pelvis=floats(position),
            neutral_orientation_pelvis_xyzw=floats(orientation),
            forearm_frame_in_task_xyzw=(
                floats(forearm_frame_in_task_xyzw) if depth == len(chain) else None
            ),
        )
        for depth, (position, orientation) in enumerate(rest_poses_pelvis, start=1)
    )


@dataclass(frozen=True)
class G1ActionLayout:
    """The frame tasks the action starts with, in the controller's order, and the waist.

    ``_leading_frame_tasks`` in ``jmex_g1_tasks.py`` enforces that the frame tasks come first.
    """

    frame_targets: tuple[FrameTargetCfg, ...]
    #: The point the robot's upper body turns about, in the pelvis frame at rest: where every
    #: anchor is turned through the operator's torso pose.
    waist_pivot_pelvis: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not self.frame_targets:
            raise ValueError("a layout needs at least one frame task")
        if np.shape(self.waist_pivot_pelvis) != (POSITION_DIM,):
            raise ValueError(
                "the waist pivot must be three numbers (x, y, z), got "
                f"{np.shape(self.waist_pivot_pelvis)}"
            )
        # Slots two chains share are the same points of one arm. Rebuilt from different
        # anchors or lengths, an elbow and a wrist target need not be reachable together.
        for target, other in combinations(self.frame_targets, 2):
            shared = _shared_prefix_length(target.chain, other.chain)
            if shared and not (
                _same_measurement(
                    target.anchor_position_pelvis, other.anchor_position_pelvis
                )
                and _same_measurement(
                    target.robot_segment_lengths[: shared - 1],
                    other.robot_segment_lengths[: shared - 1],
                )
            ):
                raise ValueError(
                    f"{target.name} and {other.name} share the arm out of "
                    f"{BodyJoint(target.chain[0]).name} but not its anchor and link lengths"
                )


@dataclass(frozen=True)
class JmexG1Settings:
    """Defaults for the j-mex G1 retargeting example."""

    #: The collection the plugin publishes the body pose on.
    collection_id: str = DEFAULT_COLLECTION_ID
    #: MOXI Player channel the plugin pairs on.
    channel: int = DEFAULT_CHANNEL
    plugin_auto_launch: bool = True
    status_period_s: float = 1.0


DEFAULT_JMEX_G1_SETTINGS = JmexG1Settings()


@dataclass(frozen=True)
class ReferenceOverlay:
    """What one frame draws over the robot, in the per-env world frame.

    ``skeleton`` is the body pose as it arrived, per slot, with the operator's heading taken
    out and its pelvis placed on the robot's -- neither rebuilt nor scaled. ``chains`` maps
    each frame task to its rebuilt chain, anchor first and the IK target last.
    """

    skeleton: dict[int, np.ndarray]
    chains: dict[str, np.ndarray]


@dataclass
class _Pairing:
    """What the current pairing has seen. Dropped whole when the body pose stops arriving."""

    operator_measured: bool = False
    incomplete_frames: int = 0


class JmexG1Retargeter(BaseRetargeter):
    """One frame of the operator's body pose in, every frame task's target pose out.

    The body pose is ``FullBodySource``'s ``full_body``, in the plugin's OpenXR axes. The poses
    are one flat array, (x, y, z, qx, qy, qz, qw) per task in the layout's order.
    """

    FULL_BODY = "full_body"
    FRAME_POSES = "frame_poses"

    def __init__(
        self,
        name: str = "jmex_g1",
        *,
        layout: G1ActionLayout,
        pelvis_pose_provider: Callable[[], PelvisPose],
        settings: JmexG1Settings = DEFAULT_JMEX_G1_SETTINGS,
        overlay_sink: Callable[[ReferenceOverlay], None] | None = None,
    ) -> None:
        """Wire the retargeter to a layout and a live pelvis.

        ``pelvis_pose_provider`` gives the robot pelvis's pose in the per-env world frame, read once
        per frame; targets are composed onto it. ``settings`` names the collection and channel in
        console messages. ``overlay_sink`` gets what to draw over the robot each frame.
        """
        self._layout = layout
        self._settings = settings
        self._pelvis_pose_provider = pelvis_pose_provider
        self._overlay_sink = overlay_sink
        # Held between frames so an unusable frame changes nothing rather than jumping.
        # Pelvis-frame, so a moving pelvis still carries the held target with it. Each is
        # the rebuilt chain, target last; before the first pairing it is the rest pose alone.
        self._held_chains: dict[str, np.ndarray] = {
            target.name: np.asarray([target.neutral_position_pelvis], dtype=np.float64)
            for target in layout.frame_targets
        }
        # Updated only together with the chain, so the two are always held as a pair.
        self._held_orientations: dict[str, Rotation] = {
            target.name: target.neutral_rotation for target in layout.frame_targets
        }
        self._rebuilt_once = False
        self._frame_count = 0
        self._unpaired_frames = 0
        self._last_status_s: float | None = None
        self._pairing: _Pairing | None = None
        super().__init__(name)

    def input_spec(self) -> RetargeterIOType:
        """Declare the body pose input, absent while nothing is paired."""
        return {self.FULL_BODY: OptionalType(FullBodyInput())}

    def output_spec(self) -> RetargeterIOType:
        """Declare the flat frame-pose output."""
        return {
            self.FRAME_POSES: TensorGroupType(
                self.FRAME_POSES,
                [
                    NDArrayType(
                        self.FRAME_POSES,
                        shape=(len(self._layout.frame_targets) * POSE_DIM,),
                        dtype=DLDataType.FLOAT,
                        dtype_bits=32,
                    )
                ],
            )
        }

    def _compute_fn(
        self,
        inputs: RetargeterIO,
        outputs: RetargeterIO,
        context: ComputeContext,
    ) -> None:
        """Reconstruct each frame task's target and pack the flat poses."""
        self._frame_count += 1
        full_body = inputs[self.FULL_BODY]
        skeleton: dict[int, np.ndarray] = {}

        if full_body.is_none:
            self._on_unpaired()
        else:
            pairing = self._announce_pairing()
            positions = (
                np.asarray(
                    full_body[FullBodyInputIndex.JOINT_POSITIONS], dtype=np.float64
                )
                @ _OPENXR_TO_ISAAC.T
            )
            orientations = full_body[FullBodyInputIndex.JOINT_ORIENTATIONS]
            valid = np.asarray(
                full_body[FullBodyInputIndex.JOINT_VALID], dtype=np.uint8
            )
            whole = self._is_whole_skeleton(pairing, valid)
            heading = _heading(positions, valid)
            # No heading, nothing to follow and no way to face the skeleton on the robot.
            if heading is not None:
                # From here on every direction is the operator's own: turning is taken out.
                positions = heading.inv().apply(positions)
                if whole and (torso := _torso_pose(positions)) is not None:
                    self._measure_operator(pairing, positions)
                    pivot = np.asarray(self._layout.waist_pivot_pelvis)
                    chains = [
                        _reconstruct(
                            target,
                            positions,
                            anchor=pivot
                            + torso.apply(
                                np.asarray(target.anchor_position_pelvis) - pivot
                            ),
                        )
                        for target in self._layout.frame_targets
                    ]
                    # All or none: an elbow that moved while its wrist held would no longer
                    # be a point of the same arm.
                    if all(chain is not None for chain in chains):
                        self._rebuilt_once = True
                        for target, chain in zip(self._layout.frame_targets, chains):
                            self._held_chains[target.name] = chain
                            self._held_orientations[target.name] = _hand_orientation(
                                target, orientations, heading
                            )
                skeleton = _received_skeleton(positions, valid)

        pelvis = self._pelvis_pose_provider()
        outputs[self.FRAME_POSES][0] = self._pack(pelvis)
        if self._overlay_sink is not None:
            self._overlay_sink(
                ReferenceOverlay(
                    skeleton={
                        slot: pelvis.place(point) for slot, point in skeleton.items()
                    },
                    chains={
                        name: pelvis.place(chain)
                        for name, chain in self._held_chains.items()
                    },
                )
            )
        self._report_status(context, pelvis)

    def _pack(self, pelvis: PelvisPose) -> np.ndarray:
        poses = np.empty((len(self._layout.frame_targets), POSE_DIM), dtype=np.float32)
        for row, target in zip(poses, self._layout.frame_targets):
            position, orientation_xyzw = pelvis.compose(
                self._held_chains[target.name][-1],
                self._held_orientations[target.name],
            )
            row[:POSITION_DIM] = position
            row[POSITION_DIM:] = orientation_xyzw
        return poses.ravel()

    def _measure_operator(self, pairing: _Pairing, positions: np.ndarray) -> None:
        """Log the streamed skeleton's segment lengths off the first usable frame of a pairing.

        For diagnosis only: the reconstruction uses the robot's lengths, not these.
        """
        if pairing.operator_measured:
            return
        measured: dict[str, tuple[float, ...]] = {}
        for target in self._layout.frame_targets:
            lengths = _segment_lengths(target.chain, positions)
            if lengths is None:
                # A segment of no length: wait for a frame where every one has a length.
                return
            measured[target.name] = lengths
        pairing.operator_measured = True
        for target in self._layout.frame_targets:
            if _extended_by_another(target, self._layout.frame_targets):
                continue  # The longer chain's log line already has these lengths.
            info_or_print(
                logger,
                "measured operator %s chain %s m; retargeting onto robot %s m",
                target.name,
                _fmt_lengths(measured[target.name]),
                _fmt_lengths(target.robot_segment_lengths),
            )

    def _announce_pairing(self) -> _Pairing:
        if self._pairing is not None:
            return self._pairing
        self._pairing = _Pairing()
        self._unpaired_frames = 0
        info_or_print(
            logger,
            "paired -- first body pose received on collection '%s'",
            self._settings.collection_id,
        )
        return self._pairing

    def _is_whole_skeleton(self, pairing: _Pairing, valid: np.ndarray) -> bool:
        """Does this frame carry every slot? Name the missing ones if that keeps not happening.

        A frame short of any slot is not followed, even where an arm's own slots all arrived.
        """
        if all(valid):
            pairing.incomplete_frames = 0
            return True
        pairing.incomplete_frames += 1
        if not _report_due(
            pairing.incomplete_frames, _INCOMPLETE_FRAMES_BEFORE_WARNING
        ):
            return False
        missing = [BodyJoint(slot).name for slot, ok in enumerate(valid) if not ok]
        logger.warning(
            "the body pose on collection '%s' has been missing %d of %d slots for %d frames "
            "(%s), so MOXI Player is not streaming the whole standard human skeleton. %s "
            "Check that the standard human skeleton is the model loaded in Player.",
            self._settings.collection_id,
            len(missing),
            len(valid),
            pairing.incomplete_frames,
            ", ".join(missing),
            self._holding(),
        )
        return False

    def _holding(self) -> str:
        """What the robot is doing while nothing drives it, for the console."""
        if self._rebuilt_once:
            return "The robot is holding the last target it was given."
        return "The robot is holding its own reset pose."

    def _on_unpaired(self) -> None:
        """No body pose. Say so, and forget what the last pairing saw.

        A Player disconnect does not land here: the plugin keeps the collection, so the tracker
        goes on handing out the last frame.
        """
        # Do not reset the held targets to rest here: the arms stay where they were last sent.
        self._pairing = None
        self._unpaired_frames += 1
        if not _report_due(self._unpaired_frames, 1):
            return
        logger.warning(
            "the jmex plugin is not paired -- no body pose on collection '%s' after %d/%d "
            "frames. %s Start MOXI Player with the standard human skeleton loaded, on channel "
            "%d; Player is the client and connects to the plugin's receiver, so nothing "
            "happens until it does.",
            self._settings.collection_id,
            self._unpaired_frames,
            self._frame_count,
            self._holding(),
            self._settings.channel,
        )

    def _report_status(self, context: ComputeContext, pelvis: PelvisPose) -> None:
        if self._pairing is None:
            return
        now_s = context.graph_time.real_time_ns * 1.0e-9
        if (
            self._last_status_s is not None
            and now_s - self._last_status_s < self._settings.status_period_s
        ):
            return
        self._last_status_s = now_s
        targets = " ".join(
            f"{target.name}={_fmt_vec(self._held_chains[target.name][-1])}"
            for target in self._layout.frame_targets
        )
        info_or_print(
            logger,
            "frame=%d pelvis=%s targets(pelvis frame) %s",
            self._frame_count,
            _fmt_vec(np.asarray(pelvis.position)),
            targets,
        )


def g1_locomanipulation_pipeline(
    frame_poses: OutputSelector, layout: G1ActionLayout, *, num_hand_joints: int
) -> OutputCombiner:
    """The G1 action: ``frame_poses``, then ``num_hand_joints`` fingers, then a standing command.

    That is the order the Pink IK action term and then the lower-body action term read. No input
    names a finger, so ``TensorReorderer`` leaves the fingers at zero.
    """
    standing = LocomotionFixedRootCmdRetargeter(
        LocomotionFixedRootCmdRetargeterConfig(), name="standing_command"
    )
    (command,) = standing.output_spec()["root_command"].types
    pose_names = [
        f"{target.name}_{index}"
        for target in layout.frame_targets
        for index in range(POSE_DIM)
    ]
    finger_names = [f"finger_{index}" for index in range(num_hand_joints)]
    command_names = [f"root_command_{index}" for index in range(command.shape[0])]
    reorderer = TensorReorderer(
        input_config={
            JmexG1Retargeter.FRAME_POSES: pose_names,
            "root_command": command_names,
        },
        output_order=pose_names + finger_names + command_names,
        name="g1_action",
        input_types={JmexG1Retargeter.FRAME_POSES: "array", "root_command": "array"},
    )
    info_or_print(
        logger,
        "the action is %d long -- %d for each of %s, then %d fingers and %d locomotion",
        len(pose_names) + len(finger_names) + len(command_names),
        POSE_DIM,
        ", ".join(target.name for target in layout.frame_targets),
        len(finger_names),
        len(command_names),
    )
    connected = reorderer.connect(
        {
            JmexG1Retargeter.FRAME_POSES: frame_poses,
            "root_command": standing.output("root_command"),
        }
    )
    return OutputCombiner({ACTION_OUTPUT: connected.output("output")})


def _report_due(frames: int, first: int) -> bool:
    """Say it on frame ``first`` of a condition, then every ``_REPORT_EVERY_FRAMES``."""
    return frames >= first and (frames - first) % _REPORT_EVERY_FRAMES == 0


def _heading(positions: np.ndarray, valid: np.ndarray) -> Rotation | None:
    """The way the operator faces, as a turn about Isaac's vertical from +X, or None.

    Read off the line between the hips, leveled, and not off the pelvis slot's orientation:
    the standard human skeleton's pelvis axes are unmeasured. None when a hip is missing or
    the two have no level distance between them, one above the other.
    """
    left, right = int(BodyJoint.LEFT_HIP), int(BodyJoint.RIGHT_HIP)
    if not (valid[left] and valid[right]):
        return None
    across = positions[left] - positions[right]
    if np.linalg.norm(across[:2]) < _MIN_SEGMENT_LENGTH_M:
        return None
    # Isaac's +Y is the left of whoever faces +X, so the operator faces a quarter turn short
    # of the way their left hip lies.
    return Rotation.from_euler("z", np.arctan2(across[1], across[0]) - np.pi / 2.0)


def _torso_pose(positions: np.ndarray) -> Rotation | None:
    """The operator's upper body, turned from upright and facing their heading, or None.

    ``positions`` have the heading taken out, so upright and facing ahead -- MOXI Player's
    reference pose -- is the identity. Read positions only: the spine slots' axes are
    unmeasured. None when the spine has no length or the shoulders have no line across it.
    """
    spine1, spine3 = int(BodyJoint.SPINE1), int(BodyJoint.SPINE3)
    left, right = int(BodyJoint.LEFT_SHOULDER), int(BodyJoint.RIGHT_SHOULDER)
    up = positions[spine3] - positions[spine1]
    if np.linalg.norm(up) < _MIN_SEGMENT_LENGTH_M:
        return None
    up = up / np.linalg.norm(up)
    across = positions[left] - positions[right]
    across = across - np.dot(across, up) * up
    if np.linalg.norm(across) < _MIN_SEGMENT_LENGTH_M:
        return None
    across = across / np.linalg.norm(across)
    # Isaac's axes: +X ahead, +Y to the left, +Z up.
    return Rotation.from_matrix(np.column_stack([np.cross(across, up), across, up]))


def _shared_prefix_length(chain: tuple[int, ...], other: tuple[int, ...]) -> int:
    """How many slots ``chain`` and ``other`` have in common before they part."""
    shared = 0
    for slot, other_slot in zip(chain, other):
        if slot != other_slot:
            break
        shared += 1
    return shared


def _same_measurement(values: Sequence[float], other: Sequence[float]) -> bool:
    return bool(
        np.allclose(values, other, rtol=0.0, atol=_SAME_MEASUREMENT_TOLERANCE_M)
    )


def _extended_by_another(
    target: FrameTargetCfg, targets: Sequence[FrameTargetCfg]
) -> bool:
    """Does another task's chain run on past the end of ``target``'s?"""
    return any(
        len(other.chain) > len(target.chain)
        and _shared_prefix_length(target.chain, other.chain) == len(target.chain)
        for other in targets
    )


def _segment_lengths(
    chain: tuple[int, ...], positions: np.ndarray
) -> tuple[float, ...] | None:
    """Meters between consecutive slots of ``chain``, or None if a segment has no length."""
    lengths = []
    for start, end in zip(chain, chain[1:]):
        length = float(np.linalg.norm(positions[end] - positions[start]))
        if length < _MIN_SEGMENT_LENGTH_M:
            return None
        lengths.append(length)
    return tuple(lengths)


def _received_skeleton(
    positions: np.ndarray, valid: np.ndarray
) -> dict[int, np.ndarray]:
    """Each usable slot as it arrived, relative to the operator's pelvis, in the pelvis frame.

    ``positions`` already have the operator's heading taken out; the caller draws nothing when
    there is none. Empty when the pelvis itself is unusable: it is what the skeleton is placed
    by.
    """
    pelvis = int(BodyJoint.PELVIS)
    if not valid[pelvis]:
        return {}
    return {
        slot: positions[slot] - positions[pelvis]
        for slot in range(len(positions))
        if valid[slot]
    }


def _reconstruct(
    target: FrameTargetCfg,
    positions: np.ndarray,
    *,
    anchor: np.ndarray,
) -> np.ndarray | None:
    """Walk ``target``'s chain out from ``anchor``: operator directions, robot lengths.

    ``anchor`` is already turned with the operator's torso; ``positions`` have the heading
    taken out, so their directions read as the robot's pelvis frame. Returns the chain's
    points in that frame, anchor first and target last, or None if a segment has no length.
    """
    points = [np.asarray(anchor, dtype=np.float64)]
    for index, (start, end) in enumerate(zip(target.chain, target.chain[1:])):
        segment = positions[end] - positions[start]
        length = float(np.linalg.norm(segment))
        if length < _MIN_SEGMENT_LENGTH_M:
            return None
        points.append(
            points[-1] + (segment / length) * target.robot_segment_lengths[index]
        )
    return np.asarray(points)


def _hand_orientation(
    target: FrameTargetCfg, orientations: np.ndarray, heading: Rotation
) -> Rotation:
    """The operator's hand, turned onto ``target``'s task frame, in the pelvis frame.

    ``heading`` is taken out, as from the positions. The ``WRIST`` slot is read whole. Do not
    build the orientation from the rebuilt forearm's direction plus a twist: a bent wrist
    takes the hand off the elbow-to-wrist line.
    """
    if target.forearm_frame_in_task_xyzw is None:
        # Uncosted, but not inert: Pink's frame error is an SE(3) log, whose position part an
        # orientation difference turns by up to half its angle. It steers the approach, never
        # where the task settles.
        return target.neutral_rotation
    wrist = target.chain[-1]
    published = Rotation.from_quat(
        np.asarray(orientations[wrist], dtype=np.float64)
    ).as_matrix()
    # The WRIST rotation in Isaac's axes is M R M^T, and the constant's columns move by M, so
    # the forearm frame in Isaac's world is (M R M^T)(M F) = M R F.
    forearm = _OPENXR_TO_ISAAC @ published @ _FOREARM_FRAME_IN_PUBLISHED_WRIST[wrist]
    return (
        heading.inv()
        * Rotation.from_matrix(forearm)
        * target.forearm_frame_in_task.inv()
    )


def _fmt_vec(vec: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:+.3f}" for value in np.asarray(vec).ravel()) + "]"


def _fmt_lengths(lengths: Sequence[float]) -> str:
    return "[" + ", ".join(f"{value:.3f}" for value in lengths) + "]"


__all__ = [
    "ACTION_OUTPUT",
    "DEFAULT_JMEX_G1_SETTINGS",
    "ORIENTATION_DIM",
    "POSE_DIM",
    "POSITION_DIM",
    "FrameTargetCfg",
    "G1ActionLayout",
    "JmexG1Retargeter",
    "JmexG1Settings",
    "PelvisPose",
    "ReferenceOverlay",
    "arm_frame_targets",
    "forearm_frame",
    "g1_locomanipulation_pipeline",
    "left_wrist_chain",
    "right_wrist_chain",
]
