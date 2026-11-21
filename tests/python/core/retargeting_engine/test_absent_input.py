# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the retargeters write while their input is absent.

A session entered with ``wait_for_headset=False`` runs its pipeline with every tracker input
absent until a headset is tracked. The robot state page lists what each retargeter writes then
(docs/source/teleop/robot_state.rst, "What a Pipeline Gives With No Tracker Data"); these tests
hold that list and the page's way to test for tracker data.
"""

import numpy as np
import pytest

import isaaccapture.retargeters as retargeters
from isaaccapture.retargeting_engine.deviceio_source_nodes import ControllersSource
from isaaccapture.retargeting_engine.interface import OutputCombiner
from isaaccapture.retargeting_engine.interface.execution_events import (
    ExecutionEvents,
    ExecutionState,
)
from isaaccapture.retargeting_engine.interface.retargeter_core_types import (
    ComputeContext,
)
from isaaccapture.retargeting_engine.interface.tensor_group import (
    OptionalTensorGroup,
    TensorGroup,
)
from isaaccapture.retargeting_engine.tensor_types import (
    ControllerInput,
    ControllerInputIndex,
)
from isaaccapture.schema import (
    ControllerInputState,
    ControllerPose,
    ControllerSnapshot,
    Point,
    Pose,
    Quaternion,
)
from isaaccapture.teleop_session_manager import DefaultTeleopStateManager

# Position, then the rotation as x, y, z, w.
ORIGIN_POSE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
CONTROLLER_POSITION = [0.4, 1.1, -0.3]
STEPS = 3


def _context(state: ExecutionState = ExecutionState.RUNNING) -> ComputeContext:
    """The context of a step; RUNNING is what a session without a control pipeline passes."""
    return ComputeContext(
        execution_events=ExecutionEvents(reset=False, execution_state=state)
    )


def _numbers(group, index: int = 0) -> list[float]:
    """One slot of an output group as plain numbers."""
    value = group[index]
    if isinstance(value, (bool, int, float)):
        return [float(value)]
    return [float(x) for x in np.asarray(np.from_dlpack(value)).ravel()]


def _all_numbers(group) -> list[float]:
    return [float(group[index]) for index in range(len(group))]


def _controller(*, squeeze: float = 0.0, trigger: float = 0.0) -> OptionalTensorGroup:
    """A tracked controller as a source node hands it to a retargeter."""
    group = OptionalTensorGroup(ControllerInput())
    position = np.array(CONTROLLER_POSITION, dtype=np.float32)
    rotation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    group[ControllerInputIndex.GRIP_POSITION] = position
    group[ControllerInputIndex.GRIP_ORIENTATION] = rotation
    group[ControllerInputIndex.GRIP_IS_VALID] = True
    group[ControllerInputIndex.AIM_POSITION] = position
    group[ControllerInputIndex.AIM_ORIENTATION] = rotation
    group[ControllerInputIndex.AIM_IS_VALID] = True
    for index in (
        ControllerInputIndex.PRIMARY_CLICK,
        ControllerInputIndex.SECONDARY_CLICK,
        ControllerInputIndex.THUMBSTICK_X,
        ControllerInputIndex.THUMBSTICK_Y,
        ControllerInputIndex.THUMBSTICK_CLICK,
        ControllerInputIndex.MENU_CLICK,
    ):
        group[index] = 0.0
    group[ControllerInputIndex.SQUEEZE_VALUE] = squeeze
    group[ControllerInputIndex.TRIGGER_VALUE] = trigger
    return group


def _home() -> np.ndarray:
    home = np.eye(4)
    home[:3, 3] = [0.2, 0.0, 0.15]
    return home


class TestPosesAndJointValues:
    """Retargeters that write a pose, joint values or a root command for absent input."""

    @pytest.mark.parametrize("input_device", ["controller_right", "hand_right"])
    def test_se3_abs_writes_the_origin_then_its_last_pose(self, input_device):
        retargeter = retargeters.Se3AbsRetargeter(
            retargeters.Se3RetargeterConfig(input_device=input_device), name="ee_pose"
        )
        for _ in range(STEPS):
            assert _numbers(retargeter({}, _context())["ee_pose"]) == ORIGIN_POSE
        stopped = retargeter({}, _context(ExecutionState.STOPPED))
        assert _numbers(stopped["ee_pose"]) == ORIGIN_POSE

    def test_se3_abs_jumps_to_the_controller_and_holds_that_pose(self):
        retargeter = retargeters.Se3AbsRetargeter(
            retargeters.Se3RetargeterConfig(input_device="controller_right"),
            name="ee_pose",
        )
        retargeter({}, _context())
        tracked = _numbers(
            retargeter({"controller_right": _controller()}, _context())["ee_pose"]
        )
        assert tracked[:3] == pytest.approx(CONTROLLER_POSITION)
        for _ in range(STEPS):
            assert _numbers(retargeter({}, _context())["ee_pose"]) == tracked

    def test_joint_state_writes_zero_for_every_joint(self):
        retargeter = retargeters.JointStateRetargeter(
            "joints",
            "joint",
            retargeters.JointStateRetargeterConfig(device_joints=["j1", "j2"]),
        )
        for _ in range(STEPS):
            targets = retargeter({}, _context())["joint_targets"]
            assert _all_numbers(targets) == [0.0, 0.0]

    def test_trihand_writes_zero_for_every_joint(self):
        names = ["thumb", "index", "middle"]
        one_hand = retargeters.TriHandMotionControllerRetargeter(
            retargeters.TriHandMotionControllerConfig(hand_joint_names=names),
            name="hand",
        )
        two_hands = retargeters.TriHandBiManualMotionControllerRetargeter(
            retargeters.TriHandMotionControllerConfig(
                hand_joint_names=names, controller_side="left"
            ),
            retargeters.TriHandMotionControllerConfig(
                hand_joint_names=names, controller_side="right"
            ),
            ["thumb", "index"],
            name="hands",
        )
        for _ in range(STEPS):
            assert _all_numbers(one_hand({}, _context())["hand_joints"]) == [0.0] * 3
            assert _all_numbers(two_hands({}, _context())["hand_joints"]) == [0.0] * 2

    def test_so101_clutch_writes_its_home_pose(self):
        home = _home()
        retargeter = retargeters.SO101ClutchRetargeter("clutch", home)
        expected = [*home[:3, 3], 0.0, 0.0, 0.0, 1.0]
        for _ in range(STEPS):
            pose = _numbers(retargeter({}, _context())["ee_pose"])
            assert pose == pytest.approx(expected)

    def test_root_commands_are_zero_velocity_at_the_configured_height(self):
        fixed_config = retargeters.LocomotionFixedRootCmdRetargeterConfig()
        stick_config = retargeters.LocomotionRootCmdRetargeterConfig()
        pedal_config = retargeters.FootPedalRootCmdRetargeterConfig()
        cases = [
            (
                retargeters.LocomotionFixedRootCmdRetargeter(fixed_config, name="a"),
                fixed_config.hip_height,
            ),
            (
                retargeters.LocomotionRootCmdRetargeter(stick_config, name="b"),
                stick_config.initial_hip_height,
            ),
            (
                retargeters.FootPedalRootCmdRetargeter(pedal_config, name="c"),
                pedal_config.max_squat_pos,
            ),
        ]
        for retargeter, hip_height in cases:
            for _ in range(STEPS):
                command = _numbers(retargeter({}, _context())["root_command"])
                assert command == pytest.approx([0.0, 0.0, 0.0, hip_height])


class TestGripperCommands:
    """Gripper retargeters: 1.0 is open for the standard ones, 0.0 for the SO-101 one."""

    def test_gripper_retargeter_opens_whenever_its_input_is_absent(self):
        retargeter = retargeters.GripperRetargeter(
            retargeters.GripperRetargeterConfig(hand_side="right"), name="gripper"
        )
        assert _numbers(retargeter({}, _context())["gripper_command"]) == [1.0]
        closed = retargeter({"controller_right": _controller(trigger=1.0)}, _context())
        assert _numbers(closed["gripper_command"]) == [-1.0]
        assert _numbers(retargeter({}, _context())["gripper_command"]) == [1.0]

    def test_so101_gripper_starts_open_and_holds_its_last_value(self):
        retargeter = retargeters.SO101GripperRetargeter("gripper")
        assert _numbers(retargeter({}, _context())["gripper_command"]) == [0.0]
        closed = _numbers(
            retargeter({"controller_right": _controller(trigger=1.0)}, _context())[
                "gripper_command"
            ]
        )
        assert closed == [1.0]
        assert _numbers(retargeter({}, _context())["gripper_command"]) == closed

    @pytest.mark.parametrize(
        "name",
        [
            "KeyboardGripperRetargeter",
            "SpaceMouseGripperRetargeter",
            "GamepadGripperRetargeter",
        ],
    )
    def test_toggle_grippers_start_open(self, name):
        retargeter = getattr(retargeters, name)(name="gripper")
        for _ in range(STEPS):
            assert _numbers(retargeter({}, _context())["gripper_command"]) == [1.0]


class TestZeroMotion:
    """Retargeters that write a motion, not a pose: no input is no motion."""

    @pytest.mark.parametrize(
        ("name", "output", "size"),
        [
            ("KeyboardToSe3RelRetargeter", "ee_delta", 6),
            ("GamepadToSe3RelRetargeter", "ee_delta", 6),
            ("SpaceMouseToSe3RelRetargeter", "ee_delta", 6),
            ("KeyboardToSe2Retargeter", "base_command", 3),
            ("GamepadToSe2Retargeter", "base_command", 3),
            ("SpaceMouseToSe2Retargeter", "base_command", 3),
        ],
    )
    def test_device_retargeters_write_zeros(self, name, output, size):
        config = getattr(retargeters, f"{name}Config")()
        retargeter = getattr(retargeters, name)(config, name="motion")
        for _ in range(STEPS):
            assert _numbers(retargeter({}, _context())[output]) == [0.0] * size

    def test_se3_rel_writes_a_zero_delta(self):
        retargeter = retargeters.Se3RelRetargeter(
            retargeters.Se3RetargeterConfig(input_device="controller_right"),
            name="ee_delta",
        )
        for _ in range(STEPS):
            assert _numbers(retargeter({}, _context())["ee_delta"]) == [0.0] * 6


class TestNoOutput:
    """Retargeters that write nothing for absent input."""

    def test_controller_pose_source_output_is_absent(self):
        retargeter = retargeters.ControllerPoseSource("pose")
        for _ in range(STEPS):
            assert retargeter({}, _context())["ee_pose"].is_none

    def test_jump_guard_output_is_absent_before_a_first_pose(self):
        retargeter = retargeters.EePoseJumpGuard("guard")
        for state in (ExecutionState.RUNNING, ExecutionState.STOPPED):
            for _ in range(STEPS):
                assert retargeter({}, _context(state))["ee_pose"].is_none

    def test_rate_limiters_leave_their_output_unset_before_a_first_value(self):
        pose_limiter = retargeters.EePoseRateLimiter("pose_limiter")
        joint_limiter = retargeters.JointRateLimiter("joint_limiter", ["j1", "j2"])
        for _ in range(STEPS):
            with pytest.raises(ValueError):
                pose_limiter({}, _context())["ee_pose"][0]
            with pytest.raises(ValueError):
                joint_limiter({}, _context())["joint_targets"][0]


# As printed on the robot state page.
def operator_holds_grip(command):
    operator = command["operator"]
    if operator.is_none:  # no tracker data in this step
        return False
    return float(operator[ControllerInputIndex.SQUEEZE_VALUE]) > 0.5


def _snapshot(squeeze: float) -> ControllerSnapshot:
    """The payload a controller tracker hands out for a tracked controller."""
    pose = Pose(Point(*CONTROLLER_POSITION), Quaternion(0.0, 0.0, 0.0, 1.0))
    state = ControllerInputState(
        primary_click=False,
        secondary_click=False,
        thumbstick_click=False,
        menu_click=False,
        thumbstick_x=0.0,
        thumbstick_y=0.0,
        squeeze_value=squeeze,
        trigger_value=0.0,
    )
    return ControllerSnapshot(
        ControllerPose(pose, True), ControllerPose(pose, True), state
    )


class TestTrackerDataPresentInThisStep:
    """The source's own output among the pipeline's outputs shows whether data is present."""

    @pytest.fixture
    def pipeline(self):
        controllers = ControllersSource(name="controllers")
        right = controllers.output(ControllersSource.RIGHT)
        ee_pose = retargeters.Se3AbsRetargeter(
            retargeters.Se3RetargeterConfig(input_device=ControllersSource.RIGHT),
            name="ee_pose",
        ).connect({ControllersSource.RIGHT: right})
        gripper = retargeters.GripperRetargeter(
            retargeters.GripperRetargeterConfig(hand_side="right"), name="gripper"
        ).connect({ControllersSource.RIGHT: right})
        pose_elements = ["x", "y", "z", "qx", "qy", "qz", "qw"]
        action = retargeters.TensorReorderer(
            input_config={"ee_pose": pose_elements, "gripper_command": ["gripper"]},
            output_order=pose_elements + ["gripper"],
            name="action",
            input_types={"ee_pose": "array", "gripper_command": "scalar"},
        ).connect(
            {
                "ee_pose": ee_pose.output("ee_pose"),
                "gripper_command": gripper.output("gripper_command"),
            }
        )
        combined = OutputCombiner(
            {
                "ee_pose": ee_pose.output("ee_pose"),
                "gripper": gripper.output("gripper_command"),
                "action": action.output("output"),
                "operator": right,  # the source's own output
            }
        )

        def step(right_payload):
            """One step as a session runs it: a tracker payload, or None, per controller."""
            spec = controllers.input_spec()
            raw = {}
            for name, payload in (
                ("deviceio_controller_left", None),
                ("deviceio_controller_right", right_payload),
            ):
                raw[name] = TensorGroup(spec[name])
                raw[name][0] = payload
            return combined({"controllers": raw}, _context())

        return step

    def test_without_tracker_data_the_pose_is_written_and_the_operator_is_absent(
        self, pipeline
    ):
        for _ in range(STEPS):
            command = pipeline(None)
            assert command["operator"].is_none
            assert not command["ee_pose"].is_none
            assert _numbers(command["ee_pose"]) == ORIGIN_POSE
            assert _numbers(command["gripper"]) == [1.0]
            assert not operator_holds_grip(command)

    def test_one_flat_tensor_does_not_show_that_the_input_is_absent(self, pipeline):
        command = pipeline(None)
        assert not command["action"].is_none
        assert _numbers(command["action"]) == [*ORIGIN_POSE, 1.0]

    def test_the_operator_input_needs_tracker_data_and_a_held_grip(self, pipeline):
        assert not operator_holds_grip(pipeline(None))
        released = pipeline(_snapshot(squeeze=0.0))
        assert not released["operator"].is_none
        assert not operator_holds_grip(released)
        assert operator_holds_grip(pipeline(_snapshot(squeeze=1.0)))
        assert not operator_holds_grip(pipeline(None))


class TestTeleopControlState:
    """The teleop control state machine with absent buttons, and who follows its state."""

    def test_the_default_state_manager_stays_stopped_without_a_reset(self):
        manager = DefaultTeleopStateManager(name="control")
        for _ in range(STEPS):
            outputs = manager({}, _context(ExecutionState.UNKNOWN))
            state = outputs["teleop_state"]
            names = [tensor_type.name for tensor_type in state.group_type.types]
            assert [name for index, name in enumerate(names) if state[index]] == [
                "stopped"
            ]
            assert not outputs["reset_event"][0]

    def test_a_stopped_state_does_not_stop_the_pose_and_gripper_retargeters(self):
        stopped = _context(ExecutionState.STOPPED)
        ee_pose = retargeters.Se3AbsRetargeter(
            retargeters.Se3RetargeterConfig(input_device="controller_right"),
            name="ee_pose",
        )
        gripper = retargeters.GripperRetargeter(
            retargeters.GripperRetargeterConfig(hand_side="right"), name="gripper"
        )
        inputs = {"controller_right": _controller(trigger=1.0)}
        assert _numbers(ee_pose(inputs, stopped)["ee_pose"])[:3] == pytest.approx(
            CONTROLLER_POSITION
        )
        assert _numbers(gripper(inputs, stopped)["gripper_command"]) == [-1.0]
