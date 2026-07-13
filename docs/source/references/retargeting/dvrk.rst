.. SPDX-FileCopyrightText: Copyright (c) 2026 Chris von Csefalvay (HCLTech)
.. SPDX-License-Identifier: Apache-2.0

dVRK PSM retargeters
====================

The dVRK Patient Side Manipulator (PSM) retargeters map XR controllers to one or
two simulated PSMs.  They convert controller state into absolute tool-pose and
paired-jaw targets.  Isaac Lab reads those targets, uses the live articulation
Jacobian and joint state, and solves differential IK.

They are not a physical dVRK driver or force-feedback system, and are not
intended for clinical workflow or surgical validation.

At a glance
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 26 44

   * - Retargeter
     - Output
     - Contract
   * - ``DVRKPSMClutchRetargeter``
     - ``ee_pose``: 7-D ``[x, y, z, qx, qy, qz, qw]``
     - Updates an absolute pose while squeeze is held.  Release holds the last
       target; the next valid squeezed sample re-bases without moving it.
   * - ``DVRKPSMGripperRetargeter``
     - ``jaw_targets``: 2-D ``[jaw_1, jaw_2]`` [rad]
     - Maps latched trigger movement to two mirrored PSM jaw targets.  The
       output is not a scalar closedness value, and releasing the arm clutch
       does not change it.

The clutch accepts a sample only when controller tracking is valid and every
pose value is finite.  If a pose sample is missing or invalid, the retargeter
holds the last safe target and clears its controller origin.  The next valid
sample with squeeze above threshold captures a fresh origin without moving the
tool.  An inactive teleoperation session holds both pose and jaw targets.

Controller mapping
------------------

OpenXR's *grip pose* is the tracked controller-handle transform.  It is not the
side grip button, which is called ``squeeze`` in the input packet.  The pose and
squeeze control tool motion; the index trigger controls the jaws.

For each PSM:

.. code-block:: text

   controller grip pose in shared reference frame
       -> squeeze deadman + origin-rebased absolute tool pose
       -> six-DOF DLS IK at psm_tool_tip_link

   controller trigger
       -> [psm_tool_gripper1_joint, psm_tool_gripper2_joint]

These retargeters do not consume face buttons, thumbsticks, thumbstick clicks,
or menu buttons.  A task or hosting XR application may bind those inputs
separately.

The squeeze threshold defaults to ``0.5``.  The trigger value captured at
squeeze engagement is neutral.  The first jaw movement requires at least
``0.05`` travel from that value.  Positive travel past the threshold starts
closing immediately.  Negative travel must remain at least ``0.05`` below its
reference for ``0.1`` seconds before opening begins.  Once opening is active,
further release opens the jaws continuously.  Releasing squeeze, losing
tracking, or stopping the session cancels a pending opening and holds the last
target.  Travel is capped at the configured physical endpoints:

.. code-block:: text

   open endpoint:   [-0.50, +0.50] rad
   closed endpoint: [-0.09, +0.09] rad

Those are library defaults, not a claim about every PSM asset.  Pass the
articulation's measured open and grip targets through
``DVRKPSMGripperConfig`` when the simulator uses different endpoints.

``initial_closedness`` selects the reset target: ``0`` is open and ``1`` is
closed.  It does not inspect contact or attach an object.

Reference frames
^^^^^^^^^^^^^^^^

``DVRKPSMClutchConfig`` does not prescribe a reference frame, but
``home_reference_T_ee``, controller input, and workspace bounds must all use the
same one.  This can be a PSM base frame for a single arm or a shared world frame
for a bimanual setup.  The home pose must lie within the workspace bounds so
the first clutch engagement remains continuous with the reset pose instead of
clipping to another target.

For a bimanual setup, express both controller streams and home poses in one
world frame.  Convert each target to that PSM's base frame immediately before
its IK solve.  Reusing one base transform for independently placed PSMs is
incorrect.

Use the retargeters from Python
-------------------------------

.. code-block:: python

   from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
   from isaacteleop.retargeters import (
       DVRKPSMClutchConfig,
       DVRKPSMClutchRetargeter,
       DVRKPSMGripperConfig,
       DVRKPSMGripperRetargeter,
   )

   left_pose = DVRKPSMClutchRetargeter(
       DVRKPSMClutchConfig(
           input_device=ControllersSource.LEFT,
           home_reference_T_ee=left_home_world_T_ee,
           workspace_lower=(-0.50, -0.40, -0.10),
           workspace_upper=(0.50, 0.40, 0.60),
           clutch_threshold=0.5,
       ),
       name="left_psm_pose",
   )
   left_jaws = DVRKPSMGripperRetargeter(
       DVRKPSMGripperConfig(input_device=ControllersSource.LEFT),
       name="left_psm_jaws",
   )

Create one pose retargeter and one jaw retargeter for each PSM.  Concatenate the
outputs for a bimanual action in this order:

.. code-block:: text

   left xyzxyzw, left jaws, right xyzxyzw, right jaws

Use ``DVRKPSMClutchRetargeter.OUTPUT_POSE`` and
``DVRKPSMGripperRetargeter.OUTPUT_JAW_TARGETS`` when wiring those output ports.
These names are part of the public retargeter API, so callers do not need to
import implementation modules.

Validation
----------

Tests that do not require Isaac Sim cover jaw endpoints, time-based opening
intent, jaw hold across clutch release and re-engagement, Cartesian clutch
behaviour, continuous motion, workspace bounds, invalid tracking, and malformed
samples.  Fixed-trace tests also compare the public retargeter wrappers with the
shared NumPy kernels:

.. code-block:: console

   $ ctest --test-dir build -R retargeting_test_dvrk_psm_retargeters --output-on-failure
