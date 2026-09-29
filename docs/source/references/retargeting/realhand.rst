.. SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Retargeters: P7 and RealHand
============================

The RealHand integration separates device acquisition, robot-independent retargeting, and
simulation control along Isaac Teleop's existing boundaries.

Components
----------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Component
     - Input
     - Output
   * - ``P7ControllerPoseRetargeter``
     - One OpenXR controller grip pose
     - Absolute seven-value P7 end-effector pose
   * - ``P7HandPoseRetargeter``
     - One OpenXR hand wrist pose
     - Absolute seven-value P7 end-effector pose
   * - ``ControllerTriggerRealHandRetargeter``
     - One controller trigger / squeeze value
     - L6, O6, or L20 hand joints
   * - ``RealHandHandTrackingRetargeter``
     - One 26-joint OpenXR hand
     - L6, O6, or L20 hand joints
   * - ``RealHandFFGGloveRetargeter``
     - One RealHand FFG Glove 21-sensor ``JointStateSource``
     - L6, O6, or L20 hand joints

P7 arm control
--------------

The P7 nodes perform only the calibrated absolute workspace transform, pose limiting, and temporal
filtering. They deliberately do not solve robot joint angles. In Isaac Lab, feed their two
``ee_pose`` outputs to ``PinkInverseKinematicsActionCfg`` so collision settings, joint limits, and
the simulated robot state remain owned by the environment.

RealHand control
----------------

``get_realhand_profile()`` provides the active joint limits and the combined bimanual action order
for each hand model. Controller mode maps the analog trigger linearly from open to the model's
closed posture. Hand-tracking mode derives each digit from OpenXR joint geometry. RealHand FFG
Glove mode maps a calibrated glove independently per side and per digit.

Run the combined example
------------------------

Install the lightweight retargeting dependency and build the package first:

.. code-block:: console

   $ pip install 'isaaccapture[retargeters-lite]'
   $ cmake --build build --target python_package

The example supports all three hand models and all three input modes:

.. code-block:: console

   $ python examples/teleop/python/p7_realhand_bimanual_example.py \
       --mode handtracking --hand-model l20

   $ python examples/teleop/python/p7_realhand_bimanual_example.py \
       --mode controller --hand-model l6

   $ python examples/teleop/python/p7_realhand_bimanual_example.py \
       --mode ffg --hand-model o6 \
       --plugin-path install/plugins \
       --realhand-ffg-glove-calibration /path/to/calibration_o6.yml

The action layout is always ``left_ee_pose + right_ee_pose + profile.action_joint_names``. Its
width is 36 for L6, 26 for O6, and 46 for L20. Use ``--dry-run`` to validate graph construction
without starting OpenXR.

Robot assets
------------

The P7 + L6, O6, and L20 URDF assemblies and meshes are published separately to keep binary robot
assets out of the Isaac Teleop source repository. Download all three assemblies from the repository
root with:

.. code-block:: console

   $ python3 examples/teleop/python/scripts/fetch_realhand_assets.py

To download only one hand model or to use a cache outside the source tree:

.. code-block:: console

   $ python3 examples/teleop/python/scripts/fetch_realhand_assets.py \
       --hand-model l20 \
       --output-dir ~/.cache/isaaccapture/realhand

The script uses the immutable commit behind RealHand Teleop v0.2.0 and verifies the downloaded
files before use. Personal glove calibrations, source CAD files, controller mounts, scenes, and
demonstrations are not downloaded. Resolve the selected assembly in an Isaac Lab environment with
``get_realhand_profile("l20").resolve_urdf(asset_root)``.

Complete Isaac Lab reference package
------------------------------------

A complete reference package containing the P7 robot environments, RealHand L6, O6, and L20
assets, and runnable controller, hand-tracking, and RealHand FFG Glove examples is available from
the `RealHand Teleop v0.2.0 release
<https://huggingface.co/realhandinc/realhand-teleop/tree/v0.2.0>`_.

This package is optional and is not required to build or use the Isaac Teleop plugins and
retargeters in this repository.

Coordinate calibration
----------------------

``P7WorkspacePoseConfig`` exposes controller or wrist center, robot workspace center, per-axis
scale and limits, fixed orientation and local position offsets, and filtering limits. The example
contains the measured neutral P7 configurations used by the reference setup. A different robot
mount, OpenXR anchor, or controller-on-glove bracket should override those values in the consuming
Isaac Lab environment rather than changing the device plugin.

.. seealso::

   :doc:`/device/realhand_ffg_glove` covers RealHand FFG Glove discovery, permissions, and
   calibration poses.
