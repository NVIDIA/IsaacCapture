.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Data Collection in Sim
======================

Collect SO-101 demonstrations in simulation with `NVIDIA Isaac Lab
<https://isaac-sim.github.io/IsaacLab>`_, on the cube-stacking task. You drive the simulated
follower through Isaac Teleop (see :doc:`devices`) and record episodes to an HDF5 dataset.

Three SO-101 stack tasks are registered in Isaac Lab — pick the one that matches your teleop
device:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Task id
     - Use
   * - ``IsaacContrib-Stack-Cube-SO101-IK-Abs-v0``
     - Absolute-pose IK + Isaac Teleop teleoperation, driven by an **XR controller**.
   * - ``IsaacContrib-Stack-Cube-SO101-Joint-Teleop-v0``
     - Joint-space mirror, driven by a physical **SO-101 leader arm** (no headset or IK).
   * - ``IsaacContrib-Stack-Cube-SO101-v0``
     - Joint-position control baseline (no teleop).

Before you start
----------------

.. important::

   The steps below are **required** — complete them first. The teleoperation and recording
   commands later on will not work until you have.

**Step 1 — Install Isaac Lab.** Follow the `Isaac Lab installation guide`_ to set up the ``Lab``
repository, then run every script through its launcher: ``./isaaclab.sh -p <script> ...`` (or
plain ``python`` inside the activated Isaac Lab environment). The SO-101 USD assets stream from
the NVIDIA Nucleus server, so there is no manual asset download.

**Step 2 — Set up CloudXR.** Both teleop devices reach the simulator over the CloudXR / OpenXR
transport, so the CloudXR runtime is always needed — follow the :doc:`/getting_started/quick_start`
and the `CloudXR teleoperation in Isaac Lab`_ guide. Isaac Lab auto-launches the runtime; pick the
profile with ``--cloudxr_env`` (``cloudxrjs`` for Quest/Pico, ``avp`` for Apple Vision Pro,
``standalone`` for headless, ``none`` to disable auto-launch).

A **headset** is needed only for the XR-controller path; no physical headset? Open the CloudXR web
client in a desktop browser, which emulates a headset. The **SO-101 Leader** path needs no headset
at all — omit ``--xr`` and the sim runs standalone in the Kit viewport.

**Step 3 — (SO-101 Leader only) Build the** ``so101_leader`` **plugin.** The leader arm is served by
an Isaac Teleop C++ plugin that is **not** part of the ``isaacteleop`` pip package or of Isaac Lab —
you must build this repository from source to get it. See
:ref:`Build and install the plugin <sim-so101-leader-build>` in the **SO-101 Leader** tab below.

Collect Teleop Data
-------------------

.. tab-set::

   .. tab-item:: XR controller

      The controller pose drives the simulated follower's end-effector through the clutch + IK
      pipeline, streamed over CloudXR — the same controls as on real hardware.

      #. **(Optional) Try teleoperation without recording.** A good way to check the setup first:

         .. code-block:: bash

            ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
                --task IsaacContrib-Stack-Cube-SO101-IK-Abs-v0 \
                --xr \
                --viz kit

         ``--xr`` enables the XR/CloudXR path and ``--viz kit`` opens the Omniverse Kit viewport.
         Squeeze and hold the grip to engage the clutch and move the arm; the trigger controls the
         gripper.

      #. **Record a dataset.** ``record_demos.py`` runs the same teleoperation while saving
         episodes to HDF5. It records ``--num_demos`` demonstrations, marking one successful after
         ``--num_success_steps`` consecutive success frames:

         .. code-block:: bash

            ./isaaclab.sh -p scripts/tools/record_demos.py \
                --task IsaacContrib-Stack-Cube-SO101-IK-Abs-v0 \
                --dataset_file ./datasets/so101_stack_demos.hdf5 \
                --num_demos 10 \
                --step_hz 30 \
                --xr \
                --viz kit

         The demos are written to the ``--dataset_file`` path in HDF5 format.

   .. tab-item:: SO-101 Leader

      A back-drivable **SO-101 leader arm** whose joint angles are mirrored 1:1 onto the simulated
      follower — no headset, no inverse kinematics, no XR anchor. Use the joint-teleop task
      ``IsaacContrib-Stack-Cube-SO101-Joint-Teleop-v0`` (not the IK task); its pipeline is
      ``JointStateSource`` → ``JointStateRetargeter`` (``mode="joint"``) → ``TensorReorderer``.

      The leader's encoders are streamed by Isaac Teleop's ``so101_leader`` plugin, a standalone
      C++ binary you run in a **second terminal** alongside the sim. Isaac Lab does not spawn it
      for you.

      .. _sim-so101-leader-build:

      **1. Build and install the plugin**

      .. important::

         ``so101_leader_plugin`` ships **only as source** — it is not in the ``isaacteleop`` pip
         package, not in Isaac Lab, and not in any release archive. It is produced by building
         this repository (`NVIDIA/IsaacTeleop <https://github.com/NVIDIA/IsaacTeleop>`_) from
         source. If ``./install/plugins/so101_leader/so101_leader_plugin`` does not exist, this
         step has not been completed.

      Install the build prerequisites first — a missing ``clang-format-14`` is the most common
      cause of a failed build, because the format check is enforced by default on Linux:

      .. code-block:: bash

         sudo apt-get update
         sudo apt-get install -y build-essential cmake libx11-dev clang-format-14 ccache patchelf pkg-config glslang-tools

      Then clone, configure, build, and install (see
      :doc:`/getting_started/build_from_source/index` for the full prerequisite list and all
      build options):

      .. code-block:: bash

         git clone https://github.com/NVIDIA/IsaacTeleop.git
         cd IsaacTeleop
         cmake -B build                       # configure
         cmake --build build --parallel       # build
         cmake --install build     # install into ./install

      The plugin lands at::

         <IsaacTeleop>/install/plugins/so101_leader/so101_leader_plugin

      Verify it before going further — with no arguments it runs the **synthetic** backend, so it
      starts without any hardware attached:

      .. code-block:: bash

         ./install/plugins/so101_leader/so101_leader_plugin

      Only the plugin is needed here; the rest of the build (Python wheel, examples) is harmless
      but optional. To build just this target:

      .. code-block:: bash

         cmake --build build --target so101_leader_plugin

      .. admonition:: Build troubleshooting
         :class: tip

         .. list-table::
            :header-rows: 1
            :widths: 42 58

            * - Symptom
              - Fix
            * - ``clang-format not found but ENABLE_CLANG_FORMAT_CHECK is ON`` (from
                ``cmake/ClangFormat.cmake``)
              - Install the pinned formatter: ``sudo apt-get install -y clang-format-14``. To skip
                the gate instead, reconfigure with
                ``cmake -B build -DENABLE_CLANG_FORMAT_CHECK=OFF``.
            * - ``so101_leader_plugin: No such file or directory``
              - The build was never installed. Re-run ``cmake --install build``,
                or run the binary from the build tree at
                ``build/src/plugins/so101_leader/so101_leader_plugin``.
            * - Configure fails downloading dependencies
              - OpenXR SDK, yaml-cpp, pybind11, FlatBuffers, and MCAP are fetched by CMake
                ``FetchContent`` — an internet connection (and any proxy settings) is required at
                configure time.
            * - ``Could NOT find Python`` / wrong interpreter
              - Set ``-DISAAC_TELEOP_PYTHON_VERSION`` on a **fresh** build directory; it cannot
                be changed on an existing one.
            * - Stale cache after changing options
              - ``cmake -B build --fresh`` wipes the cache and reconfigures.

      **2. Set up and calibrate the leader arm**

      Assemble the leader per `SO-101 support in LeRobot`_ / `SO-ARM100
      <https://github.com/TheRobotStudio/SO-ARM100>`_: remove the gearbox gears so the joints
      back-drive freely, give each servo a unique id ``1..6`` at a common baud rate, and make sure
      your user can open the serial device (add it to the ``dialout`` group).

      The plugin talks to the FEETECH STS3215 servos directly and has **no** ``lerobot`` or
      FEETECH SDK dependency — calibrate with its own ``calibrate`` subcommand, which needs no
      OpenXR runtime:

      .. code-block:: bash

         ./install/plugins/so101_leader/so101_leader_plugin calibrate /dev/ttyACM1 so101_leader.calib

      It runs two interactive steps — hold the arm at mid-range and press ENTER (homing), then
      sweep every joint through its full range and press ENTER — and writes the calibration file.
      Pass a path ending in ``.json`` to write LeRobot's format instead; an **existing LeRobot
      calibration** (``~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/<id>.json``)
      can be handed to the plugin as-is. See :ref:`so101-leader-plugin` and the
      :code-file:`plugin README <src/plugins/so101_leader/README.md>` for the file format and the
      LeRobot interoperability details.

      .. note::

         Calibration matters more in sim than on real hardware: the joint angles are applied to
         the follower **absolutely**, in radians, so an uncalibrated leader maps to the wrong pose
         rather than merely a shifted one.

      **3. Launch the simulation**

      Start Isaac Lab first — it brings up the CloudXR runtime that the plugin connects through.
      Without ``--xr`` the sim runs standalone in the Kit viewport and teleoperation starts
      automatically:

      .. code-block:: bash

         ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
             --task IsaacContrib-Stack-Cube-SO101-Joint-Teleop-v0 \
             --num_envs 1 \
             --viz kit

      Add ``--xr`` if you also want the immersive headset view; the leader arm still drives the
      follower and the retargeting pipeline is unchanged.

      **4. Start the plugin**

      In a second terminal, source the environment file the CloudXR runtime writes on startup —
      this points the OpenXR loader at CloudXR — then start the plugin on the leader's serial port
      with the calibration file:

      .. code-block:: bash

         source ~/.cloudxr/run/cloudxr.env
         ./install/plugins/so101_leader/so101_leader_plugin /dev/ttyACM1 so101_leader so101_leader.calib

      Arguments are positional: ``[device_path] [collection_id] [calibration_file]``. The
      ``collection_id`` must stay ``so101_leader`` — that is the tensor collection the task's
      ``JointStateSource`` subscribes to. Omit the device path to stream the synthetic trajectory
      instead, which is a good way to confirm the sim side is wired up before touching hardware.
      See :ref:`run-cloudxr-server` and :ref:`load-cloudxr-environment-variables` for the full
      runtime setup.

      Back-drive the leader by hand and the simulated follower mirrors it. With the Kit viewport
      focused, ``B`` starts/resumes teleoperation, ``P`` pauses (the follower holds position), and
      ``R`` resets the environment.

      **5. Record a dataset**

      ``record_demos.py`` runs the same teleoperation while saving episodes to HDF5, with the
      plugin running in its second terminal exactly as above:

      .. code-block:: bash

         ./isaaclab.sh -p scripts/tools/record_demos.py \
             --task IsaacContrib-Stack-Cube-SO101-Joint-Teleop-v0 \
             --dataset_file ./datasets/so101_leader_stack_demos.hdf5 \
             --num_demos 10 \
             --step_hz 30 \
             --viz kit

      .. admonition:: Runtime troubleshooting
         :class: tip

         .. list-table::
            :header-rows: 1
            :widths: 42 58

            * - Symptom
              - Fix
            * - Plugin exits with an OpenXR/runtime error
              - The CloudXR runtime is not up, or ``~/.cloudxr/run/cloudxr.env`` was not sourced in
                the plugin's terminal. Launch the sim first, then source the file and retry.
            * - ``Permission denied`` on ``/dev/ttyACM*``
              - Add your user to the ``dialout`` group (``sudo usermod -aG dialout $USER``) and log
                back in.
            * - Plugin runs, follower does not move
              - The ``collection_id`` must be ``so101_leader``, and teleoperation must be started
                (press ``B`` in the Kit viewport, or send **start** from the headset when using
                ``--xr``).
            * - Follower moves to the wrong pose or hits limits
              - Re-run ``calibrate`` and pass the resulting file; flip ``sign`` for any joint that
                moves the wrong way.

Convert to LeRobot Dataset
--------------------------

.. admonition:: 🚧 Work in progress
   :class: caution

   **Export to a LeRobot dataset.** Converting these sim HDF5 demos to the
   :doc:`LeRobot dataset format <training_groot>` is **not yet provided** for the stack task. The
   closest reference is the locomanipulation converter `convert_dataset.py`_ from the ``develop``
   branch in Isaac Lab, which targets a different task and must be adapted.

..
   References
.. _Isaac Lab installation guide: https://isaac-sim.github.io/IsaacLab/develop/source/setup/installation/index.html#isaaclab-installation-root
.. _CloudXR teleoperation in Isaac Lab: https://isaac-sim.github.io/IsaacLab/develop/source/how-to/cloudxr_teleoperation.html
.. _convert_dataset.py: https://github.com/isaac-sim/IsaacLab/blob/develop/scripts/imitation_learning/locomanipulation_sdg/gr00t/convert_dataset.py
.. _SO-101 support in LeRobot: https://huggingface.co/docs/lerobot/en/so101
