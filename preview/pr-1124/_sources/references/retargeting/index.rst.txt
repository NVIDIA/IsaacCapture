.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Retargeting Interface
=====================

Isaac Teleop uses a graph-based retargeting pipeline. Data flows from **source nodes** through
**retargeters** and is combined into a single action tensor.

Source Nodes
------------

* ``HeadSource`` -- provides head pose.
* ``HandsSource`` -- provides hand tracking data (left/right, 26 joints each).
* ``ControllersSource`` -- provides motion controller data (grip pose, trigger, thumbstick, etc.).
* ``Generic3AxisPedalSource`` -- provides 3-axis foot pedal data (left/right pedals, rudder).
* ``JointStateSource`` -- provides name-keyed joint positions from a generic joint-space device
  (leader arm, exoskeleton, ...). See :doc:`joint_space`.
* ``FullBodySource`` -- provides full-body pose (e.g. Pico tracking).

Available Retargeters
---------------------

.. dropdown:: Se3AbsRetargeter / Se3RelRetargeter

   Maps hand or controller tracking to end-effector pose. ``Se3AbsRetargeter`` outputs a 7D
   absolute pose (position + quaternion). ``Se3RelRetargeter`` outputs a 6D delta (position
   delta + rotation vector). Both use ``Se3RetargeterConfig`` with:

   * ``input_device``: ``"hand_left"``, ``"hand_right"``, ``"controller_left"``, or
     ``"controller_right"`` (default ``"hand_right"``).
   * Rotation offsets: ``target_offset_roll``, ``target_offset_pitch``, ``target_offset_yaw``
     (degrees, intrinsic XYZ Euler).
   * Position offsets: ``target_offset_x``, ``target_offset_y``, ``target_offset_z`` (meters).
   * Options: ``zero_out_xy_rotation``, ``use_wrist_rotation``, ``use_wrist_position``.
   * ``Se3RelRetargeter`` also uses ``delta_pos_scale_factor``, ``delta_rot_scale_factor``,
     ``alpha_pos``, ``alpha_rot``.

   ``Se3AbsRetargeter`` supports live-tunable parameters via ``ParameterState`` (rotation and
   position offsets, and the above options).

.. dropdown:: GripperRetargeter

   Outputs a single float (-1.0 closed, 1.0 open). Uses controller trigger (priority) or
   thumb-index pinch distance from hand tracking. ``GripperRetargeterConfig`` includes
   ``hand_side`` (``"left"`` or ``"right"``), ``gripper_close_meters``, ``gripper_open_meters``,
   and ``controller_threshold`` for trigger-based closing.

.. dropdown:: WujiHandRetargeter

   Maps OpenXR hand tracking to the 20 joint commands of a Wuji Hand or Wuji Hand 2 through
   ``wuji_sdk``. See :doc:`wuji` for installation, configuration, and hardware examples.

.. dropdown:: SO101ClutchRetargeter / SO101GripperRetargeter

   Retargeters for the SO-101 5-DOF arm under full-pose SE3 IK. ``SO101ClutchRetargeter``
   outputs a 7-D ``ee_pose`` like ``Se3AbsRetargeter`` but clutch-rebases the **full pose** around
   an origin captured on engage (no teleport): it re-latches both the home position and the home
   orientation on every engage and composes the orientation delta on the **left** (base frame),
   with no fixed calibration offset. It engages on ``RUNNING`` **and**
   ``squeeze > squeeze_threshold``, re-seeds its held pose from the configured home on ``reset``,
   and can latch its home position from the arm's measured EE pose.
   ``SO101GripperRetargeter`` maps the trigger to a proportional jaw closedness in ``[0, 1]``.
   See :doc:`so101` for the full setup.

.. dropdown:: JointStateRetargeter

   Maps a name-keyed joint-state input (from ``JointStateSource``) to an action for a generic
   joint-space device -- leader arm, exoskeleton, etc. Two modes via ``JointStateRetargeterConfig``:
   ``"joint"`` (lossless leader -> follower mirror with optional per-joint affine; no extra deps)
   and ``"ee_pose"`` (URDF forward kinematics -> 7D EE pose + gripper, requires ``pinocchio``).
   See :doc:`joint_space` for the full setup, modes, and the SO-101 example.

.. dropdown:: DexHandRetargeter / DexBiManualRetargeter

   Accurate hand tracking retargeter using the ``dex-retargeting`` library. It maps full hand
   tracking (26 joints) to robot-specific hand joint angles.

   **Features:**

   * Optimization-based retargeting for accurate joint angle estimation
   * Custom robot hands via URDF and YAML configuration
   * OpenXR hand tracking data (26 joints) → robot-specific joint angles
   * Configurable coordinate frame transformations

   **Requirements:**

   * ``dex-retargeting``: ``pip install dex-retargeting``
   * ``scipy``: ``pip install scipy``
   * Robot hand URDF file
   * dex_retargeting YAML configuration file

   .. warning::

      The links used for retargeting must be defined at the actual fingertips, not in the middle
      of the fingers, to ensure accurate optimization.

   **Configuration (DexHandRetargeter):**

   .. code-block:: python

      from isaacteleop.retargeters import (
          DexHandRetargeter,
          DexHandRetargeterConfig,
      )

      config = DexHandRetargeterConfig(
          hand_joint_names=[
              "thumb_proximal_yaw_joint",
              "thumb_proximal_pitch_joint",
              "index_proximal_joint",
              "middle_proximal_joint",
              "ring_proximal_joint",
              "pinky_proximal_joint",
          ],
          hand_retargeting_config="/path/to/hand_config.yml",
          hand_urdf="/path/to/robot_hand.urdf",
          handtracking_to_baselink_frame_transform=(0, 0, 1, 1, 0, 0, 0, 1, 0),  # 3x3 matrix flattened
          hand_side="left",  # or "right"
      )

      retargeter = DexHandRetargeter(config, name="dex_hand_left")

   **YAML configuration example:**

   A typical dex_retargeting config includes finger tip link names, low-pass filter, scaling,
   target joint names, type (e.g. ``DexPilot``), ``urdf_path``, and ``wrist_link_name``:

   .. code-block:: yaml

      retargeting:
        finger_tip_link_names:
        - thumb_tip
        - index_tip
        - middle_tip
        - ring_tip
        - pinky_tip
        low_pass_alpha: 0.2
        scaling_factor: 1.2
        target_joint_names:
        - thumb_proximal_yaw_joint
        - thumb_proximal_pitch_joint
        - index_proximal_joint
        - middle_proximal_joint
        - ring_proximal_joint
        - pinky_proximal_joint
        type: DexPilot
        urdf_path: /path/to/robot_hand.urdf
        wrist_link_name: hand_base_link

   **DexBiManualRetargeter:** Bimanual wrapper around two ``DexHandRetargeter`` instances. Create
   ``DexHandRetargeterConfig`` for left and right hands, then instantiate with
   ``left_config``, ``right_config``, and ``target_joint_names`` (combined left + right joint
   names):

   .. code-block:: python

      from isaacteleop.retargeters import (
          DexBiManualRetargeter,
          DexHandRetargeterConfig,
      )

      left_config = DexHandRetargeterConfig(
          hand_joint_names=left_hand_joints,
          hand_retargeting_config="/path/to/left_hand_config.yml",
          hand_urdf="/path/to/left_hand.urdf",
          hand_side="left",
      )

      right_config = DexHandRetargeterConfig(
          hand_joint_names=right_hand_joints,
          hand_retargeting_config="/path/to/right_hand_config.yml",
          hand_urdf="/path/to/right_hand.urdf",
          hand_side="right",
      )

      bimanual_retargeter = DexBiManualRetargeter(
          left_config=left_config,
          right_config=right_config,
          target_joint_names=left_hand_joints + right_hand_joints,
          name="dex_bimanual",
      )

   **Coordinate frame:** The ``handtracking_to_baselink_frame_transform`` parameter is a 3x3
   rotation matrix flattened to 9 elements. Applied as
   ``target_pos = joint_pos @ wrist_rotation @ transform_matrix``. Config default is **Identity**
   ``(1, 0, 0, 0, 1, 0, 0, 0, 1)``. Common value for **G1/Inspire**: ``(0, 0, 1, 1, 0, 0, 0, 1, 0)``
   (OpenXR Z→X, X→Y, Y→Z).

.. dropdown:: TriHandMotionControllerRetargeter

   Simple VR controller-based hand control. Maps trigger and squeeze inputs to G1 TriHand finger
   joint angles (7 DOF per hand). No external dependencies.

   **Mapping:** trigger → index finger, squeeze → middle finger, both → thumb. Good for quick
   prototyping and testing.

   **Configuration:**

   .. code-block:: python

      from isaacteleop.retargeters import (
          TriHandMotionControllerRetargeter,
          TriHandMotionControllerConfig,
      )

      config = TriHandMotionControllerConfig(
          hand_joint_names=[
              "thumb_rotation",
              "thumb_proximal",
              "thumb_distal",
              "index_proximal",
              "index_distal",
              "middle_proximal",
              "middle_distal",
          ],
          controller_side="left",  # or "right"
      )

      controller = TriHandMotionControllerRetargeter(config, name="trihand_motion_left")

   **Output DOF mapping (7 DOF):**

   ====== ====================== ==========================================
   Index  Joint                  Control
   ====== ====================== ==========================================
   0      Thumb rotation         (trigger - squeeze) * 0.5 (sign per hand)
   1      Thumb proximal         -max(trigger, squeeze) * 0.4
   2      Thumb distal           -max(trigger, squeeze) * 0.7
   3      Index proximal         trigger
   4      Index distal           trigger
   5      Middle proximal        squeeze
   6      Middle distal          squeeze
   ====== ====================== ==========================================

.. dropdown:: TriHandBiManualMotionControllerRetargeter

   Bimanual wrapper around two ``TriHandMotionControllerRetargeter`` instances for controlling
   both hands with left and right VR controllers.

   .. tip::

      For a complete hand retargeting example (e.g. ``HandsSource`` + ``DexHandRetargeter`` or
      ``TriHandMotionControllerRetargeter``, connect and compute), see
      ``g1_trihand_retargeting_example.py`` and ``dex_bimanual_example.py`` in the
      :code-dir:`examples/teleop/python` directory.

.. dropdown:: LocomotionRootCmdRetargeter

   Maps controller thumbsticks to a 4D locomotion command:
   ``[vel_x, vel_y, rot_vel_z, hip_height]``. Left thumbstick: linear velocity (X, Y). Right
   thumbstick X: angular velocity (Z). Right thumbstick Y: hip height adjustment.
   ``LocomotionRootCmdRetargeterConfig`` includes ``initial_hip_height``, ``movement_scale``,
   ``rotation_scale``, and ``dt`` (time step for height integration).

.. dropdown:: LocomotionFixedRootCmdRetargeter

   Outputs a fixed 4D root command ``[0, 0, 0, hip_height]`` (no velocity). Use when standing
   still or when controllers are not available but the pipeline expects locomotion commands.
   ``LocomotionFixedRootCmdRetargeterConfig`` has ``hip_height`` only; no inputs required.

.. dropdown:: FootPedalRootCmdRetargeter

   Maps 3-axis foot pedal input to the same 4D root command ``[vel_x, vel_y, rot_vel_z, hip_height]``.
   Uses ``Generic3AxisPedalSource``. Two modes (``FootPedalRootCmdRetargeterConfig.mode``):
   ``"horizontal"`` (right/left pedals = forward/back, rudder = yaw or strafe when pedal pressed)
   and ``"vertical"`` (left pedal drives hip height, rudder = yaw). Config includes velocity
   limits, squat range, deadzone and rudder thresholds.

.. dropdown:: TensorReorderer

   Utility that flattens and reorders outputs from multiple retargeters into a single 1D action
   tensor. The ``output_order`` must match the action space expected by the environment.


.. _isaac-teleop-pipeline-builder:

Build a Retargeting Pipeline
----------------------------

A pipeline builder is a callable that constructs the retargeting graph and returns an
``OutputCombiner`` with a single ``"action"`` key. Here is a complete example for a Franka
manipulator (from ``stack_ik_abs_env_cfg.py``):

.. code-block:: python
   :class: code-100col

   from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HandsSource
   from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
   from isaacteleop.retargeters import (
       GripperRetargeter, GripperRetargeterConfig,
       Se3AbsRetargeter, Se3RetargeterConfig,
       TensorReorderer,
   )
   from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

   def build_franka_stack_pipeline():

       # 1. Create input sources
       controllers = ControllersSource(name="controllers")
       hands = HandsSource(name="hands")

       # 2. Apply coordinate-frame transform (world_T_anchor provided by IsaacTeleopDevice)
       transform_input = ValueInput("world_T_anchor", TransformMatrix())
       transformed_controllers = controllers.transformed(
           transform_input.output(ValueInput.VALUE)
       )

       # 3. Create and connect retargeters
       se3_cfg = Se3RetargeterConfig(
           input_device=ControllersSource.RIGHT,
           target_offset_roll=90.0,
       )
       se3 = Se3AbsRetargeter(se3_cfg, name="ee_pose")
       connected_se3 = se3.connect({
           ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT),
       })

       gripper_cfg = GripperRetargeterConfig(hand_side="right")
       gripper = GripperRetargeter(gripper_cfg, name="gripper")
       connected_gripper = gripper.connect({
           ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT),
           HandsSource.RIGHT: hands.output(HandsSource.RIGHT),
       })

       # 4. Flatten into a single action tensor with TensorReorderer
       ee_elements = ["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"]
       reorderer = TensorReorderer(
           input_config={
               "ee_pose": ee_elements,
               "gripper_command": ["gripper_value"],
           },
           output_order=ee_elements + ["gripper_value"],
           name="action_reorderer",
           input_types={"ee_pose": "array", "gripper_command": "scalar"},
       )
       connected_reorderer = reorderer.connect({
           "ee_pose": connected_se3.output("ee_pose"),
           "gripper_command": connected_gripper.output("gripper_command"),
       })

       # 5. Return OutputCombiner with "action" key
       return OutputCombiner({"action": connected_reorderer.output("output")})

.. tip::

   The ``output_order`` of the ``TensorReorderer`` must match the action space of your environment.
   Mismatches will cause silent control errors.

.. _isaac-teleop-new-retargeter:

Add a New Retargeter
--------------------

If the built-in retargeters do not cover your use case, you can implement a custom one in the
`Isaac Teleop repository <https://github.com/NVIDIA/IsaacTeleop>`_:

#. Inherit from ``BaseRetargeter`` and implement ``input_spec()``, ``output_spec()``, and
   ``_compute_fn()``. (``compute()`` is concrete on the base class — it fills optional inputs
   and syncs parameters, then calls your ``_compute_fn()``.)
#. Optionally add a ``ParameterState`` for parameters that should be live-tunable via the
   retargeter tuning UI.
#. Connect to existing source nodes (``HandsSource``, ``ControllersSource``) or create a new
   ``IDeviceIOSource`` subclass for custom input devices.

See the :doc:`Contributing Guide <../../getting_started/contributing>` for details.

.. toctree::
   :maxdepth: 1
   :caption: Retargeter setup guides

   sharpa
   so101
   joint_space
   wuji
