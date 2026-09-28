.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Data Collection in Real
=======================

Record demonstrations on a physical SO-101 into a `LeRobot dataset
<https://huggingface.co/docs/lerobot/en/index>`_, driving the follower with either teleop device
(see :doc:`devices`). The example scripts live in ``examples/isaac_teleop_to_so101/`` in the
`LeRobot <https://github.com/huggingface/lerobot>`_ repository — they ship there only, and are
part of neither the ``lerobot`` pip package nor the Isaac Teleop repository, so everything on this
page runs from a LeRobot source checkout. ``teleoperate.py`` drives the arm live, and
``record.py`` does the same while saving a dataset. Both take the same
``--robot.*`` / ``--teleop.*`` flags; ``--teleop.type`` selects the device
(``xr_controller`` | ``so101_leader``).

Before you start
----------------

Follow the necessary one-time steps to set up your environment and hardware:

#. A **LeRobot source checkout**, since that is where the example scripts live. Clone it and
   change into its root — the installs below and every later command run from there, except where
   a step is explicitly scoped to another checkout (the ``so101_leader`` plugin is built from an
   Isaac Teleop checkout):

   .. code-block:: bash

      git clone https://github.com/huggingface/lerobot.git
      cd lerobot

#. A working **SO-101 follower** — assembled, motors set up, and calibrated. See
   `SO-101 support in LeRobot`_.

#. The example dependencies, installed into an environment created at that checkout root. The
   LeRobot extras cover the SO-101 motor bus (``feetech``), the IK solver for the XR path
   (``kinematics``), and dataset recording (``dataset``). For Isaac Teleop, ``cloudxr`` brings the
   CloudXR runtime bindings and ``retargeters-lite`` is the default retargeter path on both
   x86_64 and aarch64; the full ``retargeters`` extra is optional. The ``isaacteleop`` pin follows
   the release series this page documents — see :ref:`install-isaacteleop-pip-package` for the
   other install options:

   .. parsed-literal::

      uv venv --python 3.12 .venv
      source .venv/bin/activate
      uv pip install -e ".[feetech,kinematics,dataset]" "huggingface_hub>=1.5"
      uv pip install "isaacteleop[cloudxr,retargeters-lite]\ |pip_version_pin|\ " "scipy>=1.14" \\
            --extra-index-url https://pypi.nvidia.com --prerelease=allow

#. Log in to the Hugging Face Hub — recorded datasets are pushed to the Hub by default (pass
   ``--dataset.push_to_hub=false`` to keep them local):

   .. code-block:: bash

      hf auth login

#. Accept the CloudXR EULA once and start the runtime. The EULA prompts on stdin, which would hang
   a headless run, so do both ahead of time; the service outlives each script you run below:

   .. code-block:: bash

      python -m isaacteleop.cloudxr.service start --accept-eula

Teleop and data recording
-------------------------

Run the scripts as modules from the **LeRobot repository root**, with the environment from
`Before you start`_ activated (they use relative imports, so ``python -m`` is required).

.. note::

   The ``examples`` package resolves only from that root. Running these commands from an Isaac
   Teleop checkout or any other directory fails with::

      ModuleNotFoundError: No module named 'examples.isaac_teleop_to_so101'

   If you hit this, ``cd`` into your LeRobot clone and re-run.

Then follow the steps for your teleop device:

.. tab-set::

   .. tab-item:: XR controller

      The controller pose drives the follower's end-effector through the clutch + IK pipeline,
      streamed over CloudXR.

      #. **Connect a headset.** On ``connect()`` the script auto-launches the CloudXR runtime
         (~30 s) — you do **not** need a separate terminal or to source ``cloudxr.env``. Set
         ``LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH=1`` to opt out when running CloudXR yourself. For headset
         pairing and firewall setup, follow the :doc:`/getting_started/quick_start`.

         .. note::

            The XR path solves inverse kinematics, so it needs the SO-101 URDF and meshes. These are
            fetched automatically from the ``lerobot/robot-urdfs`` Hugging Face bucket into the
            LeRobot cache on first run — no manual download step.

      #. **(Optional) Try teleoperation without recording.** A good way to check the setup before
         committing to a dataset:

         .. code-block:: bash

            python -m examples.isaac_teleop_to_so101.teleoperate \
                --robot.type=so101_follower \
                --robot.port=/dev/ttyACM0 \
                --robot.id=so101_follower_arm \
                --teleop.type=xr_controller

         Squeeze and hold the grip to engage the clutch and move the arm; the trigger controls the
         gripper. Release the grip to pause.

      #. **Record a dataset.** Add cameras and the dataset parameters:

         .. code-block:: bash

            python -m examples.isaac_teleop_to_so101.record \
                --robot.type=so101_follower \
                --robot.port=/dev/ttyACM0 \
                --robot.id=so101_follower_arm \
                --teleop.type=xr_controller \
                --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
                --dataset.repo_id=$(hf auth whoami --format json | jq -r '.user')/my_test_dataset \
                --dataset.single_task="Pick up vial from rack on the left side" \
                --dataset.num_episodes=3 \
                --dataset.episode_time_s=20 \
                --dataset.reset_time_s=5

      .. note::

         **Customizing the reset pose.** On startup the XR path slews the arm to a built-in default
         reset pose (a comfortable mid-range pose) before handing control to the clutch — you do
         **not** need to record anything. To tailor it to your setup, back-drive the arm to the
         pose you want and run:

         .. code-block:: bash

            python -m examples.isaac_teleop_to_so101.override_reset_pose \
               --port /dev/ttyACM0 \
               --id so101_follower_arm

         This writes ``$HF_LEROBOT_HOME/reset_poses/<robot.name>/<robot.id>.json`` (``<robot.name>``
         is the follower type — ``so_follower`` for SO-100/SO-101 arms — and ``<robot.id>`` is the
         same arm identifier you pass as ``--robot.id``, given here as ``--id``). The pose is keyed
         per arm by ``--robot.id``, so later runs with the same ``--robot.id`` pick it up
         automatically and slew to it instead of the default. Pass ``--reset_to_origin=false`` to
         skip the slew and keep the arm where it is.

   .. tab-item:: SO-101 Leader

      A back-drivable SO-101 leader arm mirrored 1:1 to the follower. Its joints are streamed by
      Isaac Teleop's ``so101_leader`` plugin, which the script launches for you.

      #. **Build the so101_leader plugin.** It is part of Isaac Teleop's C++ source, not the
         ``isaacteleop`` pip package, so build it from an Isaac Teleop checkout:

         .. code-block:: bash

            cmake -B build && cmake --build build --parallel && cmake --install build

         The binary lands at ``install/plugins/so101_leader/so101_leader_plugin``. For details see
         :ref:`so101-leader-plugin` and :doc:`/getting_started/build_from_source/index`.

      #. **Calibrate the leader** so the leader and follower agree on each joint's zero and range.
         This reuses the serial SO-101 leader's calibration (stored under ``so_leader/<id>.json``
         and reused on every run):

         .. code-block:: bash

            lerobot-calibrate \
                --teleop.type=so101_leader \
                --teleop.port=/dev/ttyACM1 \
                --teleop.id=so101_leader_arm

      #. **(Optional) Try teleoperation without recording.** ``--launch_plugin`` spawns the plugin
         after CloudXR is up; ``--teleop.port`` is the leader's serial port:

         .. code-block:: bash

            python -m examples.isaac_teleop_to_so101.teleoperate \
                --robot.type=so101_follower \
                --robot.port=/dev/ttyACM0 \
                --robot.id=so101_follower_arm \
                --teleop.type=so101_leader \
                --teleop.port=/dev/ttyACM1 \
                --teleop.id=so101_leader_arm \
                --launch_plugin=/path/to/IsaacTeleop/install/plugins/so101_leader/so101_leader_plugin

         Back-drive the leader arm by hand to move the follower.

      #. **Record a dataset.** Same flags as teleoperation, plus the cameras and dataset parameters
         (keep ``--launch_plugin`` so the plugin is started):

         .. code-block:: bash

            python -m examples.isaac_teleop_to_so101.record \
                --robot.type=so101_follower \
                --robot.port=/dev/ttyACM0 \
                --robot.id=so101_follower_arm \
                --teleop.type=so101_leader \
                --teleop.port=/dev/ttyACM1 \
                --teleop.id=so101_leader_arm \
                --launch_plugin=/path/to/IsaacTeleop/install/plugins/so101_leader/so101_leader_plugin \
                --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
                --dataset.repo_id=$(hf auth whoami --format json | jq -r '.user')/my_test_dataset \
                --dataset.single_task="Pick up vial from rack on the left side" \
                --dataset.num_episodes=3 \
                --dataset.episode_time_s=20 \
                --dataset.reset_time_s=5

Recording controls
------------------

``record.py`` records ``--dataset.num_episodes`` episodes of ``--dataset.episode_time_s`` seconds
each, with a ``--dataset.reset_time_s`` window between episodes to reposition the scene. While
it is running, **press** these keys in the terminal where ``record.py`` is running — the example
reads them from that terminal, so they work over SSH and in a plain terminal (Linux/macOS), with
no desktop session required:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - Action
   * - Right arrow → (or ``n``)
     - End the current episode early and save it.
   * - Left arrow ← (or ``r``)
     - Discard the current take and re-record it.
   * - Escape (or ``q``)
     - Stop after the current episode (already-saved episodes are kept).

Keys are read from the terminal when stdin is a TTY (so they work over SSH); with no TTY the
example falls back to LeRobot's default keyboard listener. No configuration is needed.
The dataset is written under ``$HF_LEROBOT_HOME/<repo_id>`` and pushed to the Hub when recording
finishes (unless ``--dataset.push_to_hub=false``). Next, train a policy on it:
:doc:`training_groot`.

.. _SO-101 support in LeRobot: https://huggingface.co/docs/lerobot/en/so101
