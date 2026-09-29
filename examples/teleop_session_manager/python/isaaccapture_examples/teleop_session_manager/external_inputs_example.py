# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
External Inputs Example with TeleopSessionManager

Demonstrates using TeleopSession with a pipeline that has both DeviceIO sources
(auto-polled from hardware trackers) and external leaf nodes whose inputs are
provided by the caller at each step().

This example shows:
1. Creating a pipeline with mixed DeviceIO and non-DeviceIO leaf nodes
2. Using ValueInput for simple external values (robot EE state, transform matrix)
3. Using get_external_input_specs() to discover required external inputs
4. Passing external inputs to step()

External input options -- any non-DeviceIO leaf node becomes an external input:

    1. ValueInput (shortcut): Use for any external value you want to feed into
       the graph without any computation. One typed input, one output, data
       passed straight through. Good for sim state, transform matrices,
       configuration parameters, etc.

    2. Any existing BaseRetargeter: Any retargeter that is a leaf node (i.e. not
       connected to a DeviceIO source) automatically becomes an external input.
       You provide its inputs via step(external_inputs=...). This is useful when
       you already have a node class and want the caller to drive it directly --
       no need to wrap it in ValueInput.

    3. Custom BaseRetargeter subclass: Write a new subclass when you need
       computation on the external inputs before they enter the graph (e.g.,
       fusing multiple sensor readings, applying filtering).

Scenario:
    A "delta position" retargeter takes two inputs:
    - Controller grip position (from DeviceIO via ControllersSource, transformed)
    - Current robot end-effector position (from the caller / simulator)

    A coordinate-frame transform is applied to the controller data before
    computing the delta, allowing the caller to align the VR tracking frame
    with the robot's world frame.

    Both the robot EE state and the transform matrix are simple external values,
    so they both use ValueInput rather than custom subclasses.
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.retargeting_engine.deviceio_source_nodes import ControllersSource
from isaaccapture.retargeting_engine.interface import (
    BaseRetargeter,
    OutputCombiner,
    ValueInput,
    TensorGroup,
    TensorGroupType,
    OptionalType,
)
from isaaccapture.retargeting_engine.interface.retargeter_core_types import (
    ComputeContext,
    RetargeterIO,
)
from isaaccapture.retargeting_engine.tensor_types import (
    ControllerInput,
    ControllerInputIndex,
    TransformMatrix,
    NDArrayType,
    DLDataType,
)
from isaaccapture.teleop_session_manager import (
    TeleopSession,
    TeleopSessionConfig,
    PluginConfig,
)


# Six levels up is the tree root -- the checkout, or the install prefix
# when running from install/examples/.
PLUGIN_ROOT_DIR = Path(__file__).resolve().parents[5] / "plugins"


# ==============================================================================
# Custom Tensor Types
# ==============================================================================


def RobotEndEffectorState() -> TensorGroupType:
    """End-effector state provided by the simulator each frame.

    Contains:
        position: (3,) float32 array [x, y, z]
    """
    return TensorGroupType(
        "robot_ee_state",
        [
            NDArrayType("position", shape=(3,), dtype=DLDataType.FLOAT, dtype_bits=32),
        ],
    )


def DeltaCommand() -> TensorGroupType:
    """Velocity / delta-position command for the robot.

    Contains:
        delta_position: (3,) float32 array [dx, dy, dz]
    """
    return TensorGroupType(
        "delta_command",
        [
            NDArrayType(
                "delta_position", shape=(3,), dtype=DLDataType.FLOAT, dtype_bits=32
            ),
        ],
    )


# ==============================================================================
# Delta Position Retargeter (custom node -- has real computation)
# ==============================================================================


class DeltaPositionRetargeter(BaseRetargeter):
    """Computes delta between controller grip position and robot end-effector.

    This is an example of a custom BaseRetargeter that performs actual
    computation (subtraction) on its inputs. Unlike ValueInput, this node
    transforms data rather than passing it through.

    Inputs:
        - "controller": OptionalType(ControllerInput) (from ControllersSource)
        - "ee_state":   RobotEndEffectorState (from ValueInput)

    Outputs:
        - "delta": OptionalType(DeltaCommand) -- (controller_pos - ee_pos),
                   absent when controller is absent
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def input_spec(self):
        return {
            "controller": OptionalType(ControllerInput()),
            "ee_state": RobotEndEffectorState(),
        }

    def output_spec(self):
        return {
            "delta": OptionalType(DeltaCommand()),
        }

    def _compute_fn(
        self,
        inputs: RetargeterIO,
        outputs: RetargeterIO,
        context: ComputeContext,
    ) -> None:
        if inputs["controller"].is_none:
            outputs["delta"].set_none()
            return

        grip_pos = np.asarray(
            inputs["controller"][ControllerInputIndex.GRIP_POSITION], dtype=np.float32
        )
        ee_pos = np.asarray(inputs["ee_state"][0], dtype=np.float32)
        outputs["delta"][0] = grip_pos - ee_pos


# ==============================================================================
# Main
# ==============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args()

    # ==================================================================
    # 1. Build Pipeline
    # ==================================================================
    #
    # Leaf nodes (auto-polled by DeviceIO):
    #   - controllers    (ControllersSource)  -> hardware tracker
    #
    # Leaf nodes (external -- caller provides values via step()):
    #   - ee_state        (ValueInput)  -> robot end-effector position from sim
    #   - transform_input (ValueInput)  -> 4x4 coordinate-frame transform
    #
    # ValueInput is used for both external leaves because they just pass a
    # value into the graph without any computation. No custom subclass needed.
    #
    # Graph:
    #   controllers ──> .transformed() ──> delta_left  ──┐
    #   transform_input ──────────────────> delta_left  ──┤
    #   ee_state ─────────────────────────> delta_left  ──┤
    #                                                     ├──> OutputCombiner
    #   controllers ──> .transformed() ──> delta_right ──┤
    #   transform_input ──────────────────> delta_right ──┤
    #   ee_state ─────────────────────────> delta_right ──┘
    #

    controllers = ControllersSource(name="controllers")

    # ValueInput for the robot's current end-effector position.
    # The simulator provides this value each frame via step(external_inputs=...).
    ee_state = ValueInput("ee_state", RobotEndEffectorState())

    # ValueInput for the coordinate-frame transform matrix.
    # The caller provides this 4x4 matrix to align VR frame with robot frame.
    transform_input = ValueInput("transform_input", TransformMatrix())

    # Apply the transform to controller data before retargeting.
    # .transformed() wires up the internal transform node and returns a subgraph
    # with the same outputs ("controller_left", "controller_right") but in the
    # new coordinate frame.
    transformed_controllers = controllers.transformed(
        transform_input.output(ValueInput.VALUE)
    )

    delta_left = DeltaPositionRetargeter(name="delta_left")
    pipeline_left = delta_left.connect(
        {
            "controller": transformed_controllers.output(ControllersSource.LEFT),
            "ee_state": ee_state.output(ValueInput.VALUE),
        }
    )

    delta_right = DeltaPositionRetargeter(name="delta_right")
    pipeline_right = delta_right.connect(
        {
            "controller": transformed_controllers.output(ControllersSource.RIGHT),
            "ee_state": ee_state.output(ValueInput.VALUE),
        }
    )

    pipeline = OutputCombiner(
        {
            "delta_left": pipeline_left.output("delta"),
            "delta_right": pipeline_right.output("delta"),
        }
    )

    # ------------------------------------------------------------------
    # Alternative: using an existing retargeter directly as an external leaf
    # ------------------------------------------------------------------
    # Any BaseRetargeter that ends up as a leaf (not connected to a DeviceIO
    # source) automatically becomes an external input. You don't need to wrap
    # it in ValueInput. For example, DeltaPositionRetargeter could itself be
    # used as a standalone external leaf -- the caller would then provide
    # *all* of its inputs (both "controller" and "ee_state") via step():
    #
    #     delta = DeltaPositionRetargeter(name="delta_manual")
    #     pipeline = delta  # delta is the only (leaf) node -- no DeviceIO
    #
    #     with TeleopSession(config) as session:
    #         specs = session.get_external_input_specs()
    #         # specs == {"delta_manual": {"controller": ..., "ee_state": ...}}
    #
    #         controller_tg = TensorGroup(ControllerInput())
    #         controller_tg[0] = my_controller_data
    #         ee_tg = TensorGroup(RobotEndEffectorState())
    #         ee_tg[0] = my_ee_position
    #
    #         result = session.step(external_inputs={
    #             "delta_manual": {"controller": controller_tg, "ee_state": ee_tg}
    #         })
    #
    # This works with any retargeter -- ValueInput is just a convenience
    # shortcut for the common single-value passthrough case.
    # ------------------------------------------------------------------

    # ==================================================================
    # 2. Create TeleopSession
    # ==================================================================

    config = TeleopSessionConfig(
        app_name="ExternalInputsExample",
        pipeline=pipeline,
        plugins=[
            PluginConfig(
                plugin_name="controller_synthetic_hands",
                plugin_root_id="synthetic_hands",
                search_paths=[PLUGIN_ROOT_DIR],
            )
        ],
    )

    with CloudXRLauncher.launch_context(args), TeleopSession(config) as session:
        # ==============================================================
        # 3. Discover what external inputs the pipeline expects
        # ==============================================================
        ext_specs = session.get_external_input_specs()
        print("\n" + "=" * 70)
        print("External Inputs Example (with transform)")
        print("=" * 70)
        print(f"\nPipeline has external inputs: {session.has_external_inputs()}")
        print("External leaf nodes and their input specs:")
        for leaf_name, input_spec in ext_specs.items():
            print(f"  - {leaf_name}:")
            for input_name, tensor_group_type in input_spec.items():
                print(f"      {input_name}: {tensor_group_type}")
        print("\nPress Ctrl+C to exit\n")

        # ==============================================================
        # 4. Simulated robot state and coordinate transform
        # ==============================================================
        sim_ee_position = np.array([0.3, 0.0, 0.5], dtype=np.float32)

        # Example: 90-degree rotation about Y to align VR frame with robot frame,
        # plus a 1m translation along Z.
        vr_to_robot = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        # ==============================================================
        # 5. Main loop -- pass external inputs to step()
        # ==============================================================
        while True:
            # Build TensorGroup for the ee_state leaf's "value" input
            ee_input = TensorGroup(RobotEndEffectorState())
            ee_input[0] = sim_ee_position

            # Build TensorGroup for the transform_input leaf's "value" input
            xform_input = TensorGroup(TransformMatrix())
            xform_input[0] = vr_to_robot

            # Call step() with external inputs for the non-DeviceIO leaves.
            # Keys match the ValueInput node names; inner keys are always "value"
            # for ValueInput nodes.
            result = session.step(
                external_inputs={
                    "ee_state": {ValueInput.VALUE: ee_input},
                    "transform_input": {ValueInput.VALUE: xform_input},
                }
            )

            # Read outputs (may be absent if controller tracking is lost)
            if session.frame_count % 30 == 0:
                elapsed = session.get_elapsed_time()
                left_str = "absent"
                right_str = "absent"
                if not result["delta_left"].is_none:
                    delta_l = result["delta_left"][0]
                    left_str = (
                        f"[{delta_l[0]:+.3f}, {delta_l[1]:+.3f}, {delta_l[2]:+.3f}]"
                    )
                if not result["delta_right"].is_none:
                    delta_r = result["delta_right"][0]
                    right_str = (
                        f"[{delta_r[0]:+.3f}, {delta_r[1]:+.3f}, {delta_r[2]:+.3f}]"
                    )
                print(f"[{elapsed:5.1f}s] delta_L={left_str}  delta_R={right_str}")

            # Slowly drift the simulated EE position (pretend robot is moving)
            sim_ee_position += np.array([0.0001, 0.0, 0.0], dtype=np.float32)

            time.sleep(0.016)  # ~60 FPS

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit(0)
