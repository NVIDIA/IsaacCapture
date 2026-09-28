.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Retargeter: Manus to Sharpa
===========================

``SharpaHandRetargeter`` maps live hand-tracking poses from a `Manus glove
<https://www.manus-meta.com/>`_ (or any other source feeding the OpenXR
hand-tracking layer) onto Sharpa hand joint angles, frame by frame, via
optimization-based inverse kinematics. ``SharpaBiManualRetargeter`` is a
thin combiner that interleaves left and right outputs into a single
target-ordered vector for downstream control.

At a glance
-----------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Stage
     - What happens
   * - Input
     - 26-joint OpenXR ``HandInput`` for the configured side (xyzw quats),
       sourced from a Manus glove plugin or any other OpenXR hand-tracking
       provider.
   * - Repack
     - Drop OpenXR palm + non-thumb metacarpals to land on the canonical
       MANO 21-joint layout, and convert quaternions to wxyz.
   * - IK
     - ``robotic_grounding.retarget.hand_kinematics.SharpaHandKinematics``
       runs Pink IK on a Pinocchio model loaded from the Sharpa MJCF, with
       a FreeFlyer root that re-anchors the wrist each frame.
   * - Warm-start
     - Previous-frame qpos is kept and reused (with the wrist re-pinned to
       the new tracker reading) to keep the IK locally smooth. A frame
       with any invalid input joint zeros the output and resets the
       warm-start.
   * - Output
     - Sharpa finger DOFs (everything Pinocchio reports past the
       FreeFlyer), optionally reordered by ``hand_joint_names``.

The retargeter intentionally contains no IK math itself: joint orderings,
frame mappings, rotation corrections, and Pink/Pinocchio configuration all
live in ``robotic_grounding`` (V2D). This module is the OpenXR-shaped
adapter on top of it.

.. seealso::

   :doc:`/device/manus` -- installing the Manus plugin so its tracking
   shows up on the OpenXR hand layer that this retargeter consumes.

   :doc:`index` -- the broader retargeting interface and pipeline-builder
   pattern.

Public V2D source and the ``[grounding]`` extra
-----------------------------------------------

The IK implementation comes from the public `Video to Data (V2D) v0.2.0
release <https://github.com/nvidia-isaac/video_to_data/releases/tag/v0.2.0>`_.
When ``BUILD_PYTHON_BINDINGS=ON``, CMake fetches the pinned public commit
and stages only the ``robotic_grounding.retarget`` modules used here,
the public Sharpa MJCFs and license notices, and generated mesh-free
variants. V2D's meshes, Git LFS pointers, and unrelated packages are not
included.

Every Isaac Teleop Python wheel contains that small staged source bundle.
The implementation remains lazy-loaded, so importing other Isaac Teleop
features does not import its heavier numerical stack. The
``[grounding]`` extra installs that runtime stack:

.. code-block:: console

   $ pip install "isaacteleop[grounding]"

For a source or editable install, CMake performs the same fetch and
staging automatically:

.. code-block:: console

   $ pip install -e ".[grounding]"

There is no separate V2D wheel, authentication step, source setup script,
or CMake bundle option. A build configured with
``BUILD_PYTHON_BINDINGS=OFF`` does not fetch V2D.

Use it from Python
------------------

.. code-block:: python

   from isaacteleop.retargeters import (
       SharpaHandRetargeter,
       SharpaHandRetargeterConfig,
   )

The generated mesh-free Sharpa MJCFs ship inside ``robotic_grounding``.
Resolve them with ``importlib.resources``:

.. code-block:: python

   from importlib.resources import files

   xml_dir = files("robotic_grounding") / "assets" / "xmls" / "sharpawave"
   right_mjcf = str(xml_dir / "right_sharpawave_nomesh.xml")

   cfg = SharpaHandRetargeterConfig(hand_side="right", robot_asset_path=right_mjcf)
   retargeter = SharpaHandRetargeter(cfg, name="sharpa_right")

Key ``SharpaHandRetargeterConfig`` fields:

* ``robot_asset_path`` — the generated mesh-free Sharpa MJCF path.
* ``hand_side`` — ``"left"`` or ``"right"``.
* ``hand_joint_names`` — optional output ordering override; defaults to
  whatever finger joints Pinocchio discovers in the MJCF, in model order.
* ``source_to_robot_scale`` — MANO-to-robot length scale.
* ``solver`` / ``max_iter`` / ``frequency`` /
  ``frame_tasks_converged_threshold`` — Pink IK knobs forwarded to
  ``SharpaHandKinematics``.

For bimanual control, instantiate two ``SharpaHandRetargeter``\ s and
wrap them with ``SharpaBiManualRetargeter`` so a single output vector is
produced in your target joint order.

Run the example
---------------

The repo ships a bimanual demo in ``examples/retargeting``:

.. code-block:: console

   $ uv pip install -e ./examples/retargeting

   # Synthetic curl animation (no headset, no GUI required):
   $ python -m isaacteleop_examples.retargeting.sharpa_hand_retargeter_demo --synthetic

   # Live bimanual from a connected Quest headset:
   $ python -m isaacteleop_examples.retargeting.sharpa_hand_retargeter_demo

   # Custom MJCFs (e.g. the mesh-bearing variants):
   $ python -m isaacteleop_examples.retargeting.sharpa_hand_retargeter_demo \
       --left-mjcf  /path/to/left_sharpawave.xml \
       --right-mjcf /path/to/right_sharpawave.xml

The synthetic mode is the smoke test: if it animates a curl trajectory
and prints non-zero finger qpos each frame, the install is good.

Validate
--------

Run both checks after configuring and building with Python bindings:

**End-to-end pytest** -- exercises the full Pinocchio + Pink IK pipeline
through the Teleop wrapper (init, warm-start persistence, open vs. curled
hand, absent-hand zeros, etc.):

.. code-block:: console

   $ ctest --test-dir build -R retargeting_test_sharpa_hand_retargeter --output-on-failure
   ...
   100% tests passed, 0 tests failed out of 1

**Full retargeting suite** -- regression coverage in case the wrapper
introduced a typing or import regression elsewhere:

.. code-block:: console

   $ ctest --test-dir build -R '^retargeting_' --output-on-failure
   ...
   100% tests passed, 0 tests failed

CI
--

Ubuntu, release-wheel, editable-install, and ROS 2 jobs use the same
CMake-managed public dependency. The ROS 2 replay matrix includes
``hand_retargeter:=pink_ik`` and verifies that the installed wheel turns
a deterministic finger-curl trajectory into finite, ordered, nonzero,
changing Sharpa joint commands.

Bumping public V2D
------------------

Update ``VIDEO_TO_DATA_COMMIT`` in ``deps/v2d/CMakeLists.txt`` to the
full commit for the desired public release. Configure a clean build so
FetchContent checks out that revision and the staging script validates
that mesh references are removed without changing joints or IK target
sites. Then run the Sharpa test and inspect the wheel contents before
submitting the change.
