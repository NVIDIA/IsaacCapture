.. SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

dVRK PSM retargeters
====================

The da Vinci Research Kit (dVRK) fills a role in surgical robotics similar to
the :doc:`SO-101 </getting_started/lerobot/index>` in general manipulation: it
provides a shared research embodiment built from repurposed first-generation
da Vinci hardware, with open controllers and software.  The patient-side
architecture combines an Endoscopic Camera Manipulator (ECM) with up to three
Patient Side Manipulators (PSMs), which carry instruments such as needle drivers
and forceps.

Physical dVRK systems are scarce and are primarily available to eligible
non-profit education and research institutions as non-clinical research
platforms.  Simulation broadens access to the PSM embodiment for teleoperation,
demonstration collection, policy development, and reproducible evaluation.
dVRK research setups are typically bimanual, with two PSMs coordinating in a
shared workspace.  Collaborative manipulation tasks therefore abound,
including passing a needle from one instrument to the other, as shown below.
See the `Intuitive Foundation dVRK programme
<https://www.intuitive-foundation.org/dvrk/>`_, the `dVRK Research Wiki
<https://research.intusurg.com/index.php/Main_Page>`_, and the `JHU dVRK
software wiki <https://github.com/jhu-dvrk/sawIntuitiveResearchKit/wiki>`_ for
the hardware, software, and research community.

The dVRK Patient Side Manipulator (PSM) retargeters map XR controllers to one or
two simulated PSMs.  They convert controller state into absolute tool-pose and
paired-jaw targets.  A simulator integration can read those targets, use its
live articulation Jacobian and joint state, and solve differential IK.

.. figure:: ../../_static/dvrk-needle-pass.gif
   :alt: Real dVRK PSM teleoperation passing a surgical needle in Isaac Lab
   :width: 720px
   :align: center
   :class: no-image-zoom

   A real bimanual teleoperation run of the Isaac Lab needle-passing task using
   the same dVRK tool-pose and jaw-target contract.

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
closing immediately.  Negative travel must first be observed at least ``0.05``
below its reference, then remain there for ``0.1`` seconds before opening
begins.  Once opening is active, further release opens the jaws continuously.
Releasing squeeze, losing tracking, or stopping the session cancels a pending
opening and holds the last target.  Travel is capped at the configured physical
endpoints:

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

   import numpy as np

   from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
   from isaacteleop.retargeting_engine.interface import OutputCombiner
   from isaacteleop.retargeters import (
       DVRKPSMClutchConfig,
       DVRKPSMClutchRetargeter,
       DVRKPSMGripperConfig,
       DVRKPSMGripperRetargeter,
   )

   def build_right_psm_pipeline() -> OutputCombiner:
       controllers = ControllersSource(name="controllers")

       # The home transform, raw controller grip pose, and workspace are all
       # expressed in this example's shared tracking reference frame.
       home_reference_T_ee = np.eye(4, dtype=np.float64)
       home_reference_T_ee[:3, 3] = (0.0, 0.0, 0.18)

       pose = DVRKPSMClutchRetargeter(
           DVRKPSMClutchConfig(
               input_device=ControllersSource.RIGHT,
               home_reference_T_ee=home_reference_T_ee,
               workspace_lower=(-0.16, -0.14, 0.06),
               workspace_upper=(0.16, 0.14, 0.28),
               clutch_threshold=0.5,
           ),
           name="right_psm_pose",
       )
       jaws = DVRKPSMGripperRetargeter(
           DVRKPSMGripperConfig(input_device=ControllersSource.RIGHT),
           name="right_psm_jaws",
       )

       right_controller = controllers.output(ControllersSource.RIGHT)
       connected_pose = pose.connect(
           {ControllersSource.RIGHT: right_controller}
       )
       connected_jaws = jaws.connect(
           {ControllersSource.RIGHT: right_controller}
       )

       return OutputCombiner(
           {
               "ee_pose": connected_pose.output(
                   DVRKPSMClutchRetargeter.OUTPUT_POSE
               ),
               "jaw_targets": connected_jaws.output(
                   DVRKPSMGripperRetargeter.OUTPUT_JAW_TARGETS
               ),
           }
       )


   pipeline = build_right_psm_pipeline()

Create one pose retargeter and one jaw retargeter for each PSM.  Each instance
publishes an independent named output; the downstream consumer chooses any
packing and ordering required by its own action contract.  Use
``DVRKPSMClutchRetargeter.OUTPUT_POSE`` and
``DVRKPSMGripperRetargeter.OUTPUT_JAW_TARGETS`` when wiring those output ports.
These names are part of the public retargeter API, so callers do not need to
import implementation modules.

Validate
--------

Tests that do not require Isaac Sim cover jaw endpoints, time-based opening
intent, jaw hold across clutch release and re-engagement, Cartesian clutch
behaviour, continuous motion, workspace bounds, invalid tracking, and malformed
samples.  Fixed-trace tests also compare the public retargeter wrappers with the
shared NumPy kernels:

.. code-block:: console

   $ ctest --test-dir build -R retargeting_test_dvrk_psm_retargeters --output-on-failure
