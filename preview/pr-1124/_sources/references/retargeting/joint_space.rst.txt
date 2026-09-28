.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Retargeter: Joint-Space Device
==============================

``JointStateRetargeter`` maps a name-keyed joint-state input (from
:doc:`/device/joint_space`'s ``JointStateSource``) onto an Isaac Lab action, in one of two modes.
It is the generic retargeter for leader arms, exoskeletons, and other joint-encoder devices; the
SO-101 leader arm is the reference instance.

At a glance
-----------

.. list-table::
   :header-rows: 1
   :widths: 16 30 54

   * - Mode
     - Output
     - Use
   * - ``joint``
     - one float per target joint (``joint_targets``)
     - Lossless leader -> follower mirror for same-kinematics teleoperation. Name remap + optional
       per-joint affine. No extra dependencies.
   * - ``ee_pose``
     - 7-D ``ee_pose`` ``[x,y,z,qx,qy,qz,qw]`` + ``gripper_command``
     - Task-space / cross-embodiment teleoperation via URDF forward kinematics. Requires
       ``pinocchio`` (the ``[retargeters]`` extra).

``joint`` mode
--------------

Each target joint is filled from a device joint (by name) with an optional affine
``offset + sign * scale * value``. Defaults are an identity mirror. ``JointStateRetargeterConfig``:

* ``device_joints`` -- ordered device DOF names (must match the source's ``joint_names`` order).
* ``target_joints`` -- robot joint names to emit (defaults to ``device_joints``).
* ``joint_map`` -- ``{device_name: target_name}`` overrides; ``scale`` / ``offset`` / ``sign`` --
  per-target affine.

``ee_pose`` mode
----------------

Forward-kinematics the device joints through a URDF and emit the end-effector pose plus a gripper
command. Config: ``urdf_path``, ``ee_link``, ``gripper_joint`` (and optional ``gripper_open`` /
``gripper_close`` to emit normalized closedness in ``[0, 1]`` instead of the raw value).

* FK uses ``pinocchio`` (imported lazily; ``joint`` mode never needs it). Install via
  ``pip install 'isaacteleop[retargeters]'``.
* Assumes a fixed-base model of single-DOF joints (the common leader-arm / exoskeleton case).
* The schema's device ``ee_pose`` field is **not** consumed yet -- FK is always computed from the
  joint positions.
* ``clutch=True`` rebases the EE around an origin captured on the first ``RUNNING`` frame so
  engaging teleop does not jump the robot; when the optional ``robot_ee_pos`` input (the live
  ``world_T_ee``) is connected, the latched home is the robot's current end-effector.

.. note::

   The ``joints`` input is read positionally in ``device_joints`` order, so the upstream source's
   ``joint_names`` must list the same names in the same order. A name mismatch is rejected by the
   graph's type check at ``connect`` time.

Use it from Python
------------------

A pipeline builder returns an ``OutputCombiner`` with a single ``"action"`` key (the layout your
environment's action space expects):

.. code-block:: python

   from isaacteleop.retargeting_engine.deviceio_source_nodes import JointStateSource
   from isaacteleop.retargeting_engine.interface import OutputCombiner
   from isaacteleop.retargeters import (
       JointStateRetargeter,
       JointStateRetargeterConfig,
       TensorReorderer,
   )

   SO101_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

   def build_so101_joint_pipeline():
       source = JointStateSource(name="leader", collection_id="so101_leader", joint_names=SO101_JOINTS)
       retargeter = JointStateRetargeter(
           name="leader",
           mode="joint",
           config=JointStateRetargeterConfig(device_joints=SO101_JOINTS, target_joints=SO101_JOINTS),
       )
       head = retargeter.connect({JointStateRetargeter.JOINTS: source.output(JointStateSource.JOINTS)})
       reorderer = TensorReorderer(
           input_config={"joint_targets": SO101_JOINTS},
           output_order=SO101_JOINTS,
           name="action_reorderer",
           input_types={"joint_targets": "scalar"},
       )
       connected = reorderer.connect({"joint_targets": head.output("joint_targets")})
       return OutputCombiner({"action": connected.output("output")})

For ``ee_pose`` mode, build the retargeter with ``mode="ee_pose"`` + a ``urdf_path`` / ``ee_link``
and flatten ``ee_pose`` + ``gripper_command`` into the env's task-space action layout.

Run the example
---------------

The repo ships ``examples/teleop/python/joint_space_device_example.py``:

.. code-block:: console

   # Consumes the so101_leader plugin over OpenXR (source cloudxr.env first):
   $ python joint_space_device_example.py --launch-plugin --mode joint --frames 8
   $ python joint_space_device_example.py --launch-plugin --mode ee --urdf so101_new_calib.urdf

Validate
--------

Sim-free unit tests cover both modes (joint affine/remap/hold/reset, EE forward kinematics, clutch
rebasing, and the flattened action width/order):

.. code-block:: console

   $ ctest --test-dir build -R 'retargeting_test_joint_state' --output-on-failure

.. seealso::

   :doc:`/device/joint_space` -- the schema, ``JointStateTracker``, ``JointStateSource``, the
   SO-101 plugin, and MCAP record/replay.

   :doc:`index` -- the broader retargeting interface and pipeline-builder pattern.
