.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Supported Teleop devices
========================

Isaac Teleop drives the SO-101 from two input devices today. Both command the same SO-101 follower
arm and produce the same LeRobot-format demonstrations — pick the one that matches the hardware you
have and the control feel you want.

XR controller
-------------

A VR controller — from a Meta Quest, PICO, or Apple Vision Pro headset — streamed to your
workstation over `NVIDIA CloudXR <https://docs.nvidia.com/cloudxr-sdk>`_. The controller pose drives
the SO-101 **end-effector** target, which inverse kinematics turns into arm joint commands, so you
steer the gripper through space rather than posing each joint.

A **clutch** (deadman) keeps teleoperation safe: the arm only moves while you hold the grip engaged.
Releasing pauses motion, and re-engaging re-centers on the arm's current pose, so the arm never
jumps when you reposition your hand.

**Controls**

- **Squeeze / grip** — the clutch. Hold to engage teleoperation; release to pause.
- **Trigger** — the gripper, controlled proportionally (analog).
- **Controller orientation** — the wrist. On the 5-DOF SO-101 the wrist follows your hand only
  partially, by design.

The XR controller is **self-calibrating** — there is no leader arm and no manual calibration step —
and it works in both simulation and on real hardware. You will need a CloudXR-connected headset; see
the :doc:`Quick Start </getting_started/quick_start>` for headset setup.

SO-101 Leader
-------------

A second, back-drivable SO-101 arm used as a **leader**: you move the leader by hand and the SO-101
follower mirrors it. Because the leader and follower share the same kinematics, the mapping is a
**direct joint-space mirror** — there is no inverse kinematics and no clutch; the follower tracks the
leader's joints one-to-one.

The leader's joint stream is published by Isaac Teleop's native ``so101_leader`` plugin, which the
teleoperation example launches for you.

This path needs the physical leader arm and a one-time **calibration** so the leader and follower
agree on each joint's zero and range. Calibrate with ``lerobot-calibrate`` (or the plugin's own
calibration step); the result is reused on every run.

The SO-101 Leader gives the most direct, intuitive feel for SO-101 teleoperation, at the cost of
needing a second arm.

The SO-101 Leader is the first :doc:`joint-space device </device/joint_space>` in Isaac Teleop — one
that streams joint positions directly instead of an end-effector pose. The interface is general, so
the SO-101 Leader is just the starting point; feature requests and/or contributions that add more
joint-space devices are highly welcome.

Choosing a device
-----------------

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * -
     - XR controller
     - SO-101 Leader
   * - Hardware
     - XR headset (CloudXR) that can be reused to teleoperate other robots
     - A leader arm dedicated to SO-101
   * - Mapping
     - End-effector pose → IK, with a clutch
     - Direct joint-space mirror (1:1)
   * - Extra Features
     - - No dedicated space needed for setting up a second arm
       - Haptics support (coming soon)
     - - No clutch or IK weights to tune

Prerequisites and calibration
-----------------------------

Both devices share the same high-level setup — see the per-flow steps in
:doc:`data_collection_real` and :doc:`data_collection_sim`. Calibration differs by device:

- **XR controller** — **self-calibrating**: the clutch re-centers on every engage, so there is no
  manual calibration step. It just needs a CloudXR-connected headset
  (:doc:`/getting_started/quick_start`).
- **SO-101 Leader** — needs a **one-time calibration** so the leader and follower agree on each
  joint's zero and range:

  .. code-block:: bash

     lerobot-calibrate \
         --teleop.type=so101_leader \
         --teleop.port=/dev/ttyACM1 \
         --teleop.id=so101_leader_arm

  The result is stored under ``so_leader/<id>.json`` and reused on every run.

Learn more
----------

For the device and retargeting internals behind these flows, see the Isaac Teleop references:

- :doc:`/device/trackers` — the device interface (trackers).
- :doc:`/device/add_device` — the device plugin model.
- :doc:`/device/joint_space` — the SO-101 leader (joint-space) device and its plugin.
- :doc:`/references/retargeting/index` — the retargeting interface.
- :doc:`/references/retargeting/so101` — the SO-101 (5-DOF arm) retargeters used by the XR
  controller.
- :doc:`/references/retargeting/joint_space` — the joint-space retargeter used by the leader.
