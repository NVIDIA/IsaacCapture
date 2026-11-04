.. SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Retargeters: P7 and RealHand
============================

The RealHand integration separates device acquisition, hand retargeting, and
simulation control along IsaacCapture's existing boundaries.

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
   * - Official ``DexHandRetargeter``
     - One 26-joint OpenXR hand (optical tracking or a glove plugin)
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
closed posture. Optical hand tracking and FFG both supply standard 26-joint
``HandsSource`` data to the same official ``DexHandRetargeter``. The mechanical
hand retargeter is independent of which device supplies the human joints.
FFG mode keeps P7 end-effector targets on the controller grip poses.

The model-specific YAML and mesh-free hand URDF files are in
``examples/teleop/python/isaaccapture_examples/teleop/config/realhand/``. They configure the official
vector optimizer, joint order, mimic relationships, and coordinate transform.
There is no custom optical hand algorithm, FFG robot mapping, gesture anchor,
or fingertip snapping. These small kinematic files do not require downloading
the full assemblies to run the pipeline example.

Human skeleton reconstruction and robot-independent glove calibration belong to
the device plugin. Robot control has no raw-sensor bypass or FFG-specific mapping.
Other compatible hand-tracking devices can drive RealHand hands through the same
``HandInput`` contract; FFG can feed other manufacturers' compatible retargeters.

Run the combined example
------------------------

Build and install the package from this checkout, following
:doc:`/development/build_from_source/index`. Install the locally built wheel
so the example uses the retargeters in this branch:

.. code-block:: console

   $ cmake --build build
   $ cmake --install build
   $ pip install --force-reinstall --no-deps isaaccapture --find-links=./install/wheels/ --no-index
   $ pip install 'isaaccapture[retargeters]'
   $ pip install -e ./examples/teleop

The wheel installation assumes the base runtime dependencies are already installed as
described in the build guide; ``--no-deps`` keeps the offline wheel lookup from
trying to reinstall those dependencies. The official hand solver requires the
``retargeters`` extra (including dex-retargeting and PyTorch); the P7 pose
nodes require SciPy.
Setting up a development install from this
checkout with ``pip install -e '.[retargeters]'`` is another supported path;
install ``./examples/teleop`` there as well.

The example supports all three hand models and all three input modes:

.. code-block:: console

   $ python -m isaaccapture_examples.teleop.p7_realhand_bimanual_example \
       --mode handtracking --hand-model l20

   $ python -m isaaccapture_examples.teleop.p7_realhand_bimanual_example \
       --mode controller --hand-model l6

   $ python -m isaaccapture_examples.teleop.p7_realhand_bimanual_example \
       --mode ffg --hand-model o6 \
       --plugin-path install/plugins \
       --realhand-ffg-glove-ports=/dev/ttyUSB0,/dev/ttyUSB1 \
       --realhand-ffg-glove-calibration /path/to/ffg_human.yml

Use only verified glove ports; omit ``--realhand-ffg-glove-ports`` for devices
identified as FFG hardware by their USB serial identifiers.
``--realhand-ffg-glove-calibration`` is independent of ``--hand-model`` and passed
only to a plugin started with ``--plugin-path``. For an already running calibrated
plugin, omit both options. Graph construction does not load a glove calibration;
one human calibration can be reused with any compatible robot-hand retargeter.

The action layout is always ``left_ee_pose + right_ee_pose + profile.action_joint_names``. Its
width is 36 for L6, 26 for O6, and 46 for L20. Use ``--dry-run`` to validate graph construction
without starting OpenXR.

Robot assets
------------

The P7 + L6, O6, and L20 URDF assemblies and meshes are published separately to keep binary robot
assets out of the Isaac Teleop source repository. Download all three assemblies from the repository
root with:

.. code-block:: console

   $ python3 -m isaaccapture_examples.teleop.fetch_realhand_assets

To download only one hand model or to use a cache outside the source tree:

.. code-block:: console

   $ python3 -m isaaccapture_examples.teleop.fetch_realhand_assets \
       --hand-model l20 \
       --output-dir ~/.cache/isaaccapture/realhand

The default destination is ``~/.cache/isaaccapture/realhand`` (under ``XDG_CACHE_HOME``
when set), so installed examples do not write into their package directory.
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

This package is optional and is not required to build or use the IsaacCapture plugins and
retargeters in this repository.
It is a standalone release. Its legacy robot-specific FFG calibration cannot be
used as this plugin's human-skeleton calibration; capture a device calibration
with the example in this checkout.

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
