.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

IO Exoskeleton Glove (ROS 2)
============================

A Linux plugin for integrating an IO exoskeleton glove into the Isaac Teleop
framework. The glove is the *sensor*: an exoskeleton worn over the hand whose
ROS 2 driver publishes the linkage's joint angles and every link pose. The
plugin bridges those streams into the generic device paths — the glove's own
joint angles, the five fingertip poses per hand, and an OpenXR hand skeleton —
so the mapping onto a particular dexterous robot hand stays downstream in a
retargeter instead of being baked into the recording.

.. contents:: On this page
   :local:
   :depth: 2

Data flow
---------

The plugin is an ``rclcpp`` node that consumes the vendor driver's two topics
and republishes them through the Isaac Teleop session:

.. code-block:: text

   IO exoskeleton SDK / ROS 2 driver
     --/io_esk/joint_data-->        sensor_msgs/JointState   (link DOF angles, radians)
     --/io_fusion/tf_exoskeleton--> tf2_msgs/TFMessage       (every link pose, in base_link)
                        |
                        v
          io_exo_glove_ros2_plugin
            joints -> JointStateOutput     -> exo_glove_joints_left / _right
            tips   -> JointSe3PoseOutput   -> exo_glove_tips_left  / _right
            links  -> OpenXR hand skeleton -> /hand/left, /hand/right
                        |
                        v
          JointStateTracker / JointSe3PoseTracker / OpenXR hand tracker
                        |
                        v
          hand retargeters (one collection per hand)

Nothing is retargeted in the plugin: joint names, joint values and the
fingertip reference frame are forwarded as published. The exception is the
injected OpenXR skeleton, which cannot be a linkage mechanism — the exoskeleton's
own segments are about twice a human phalanx — so the plugin drives a human
anatomy from the glove's *articulation* instead (see `Hand tracking`_).

Prerequisites
-------------

- **Linux** — x86_64 (tested on Ubuntu 22.04).
- **ROS 2 Humble** — ``rclcpp``, ``sensor_msgs`` and ``tf2_msgs``. No custom
  messages and no ``colcon``/``ament`` wrapper; the plugin only consumes stock
  message types.
- **IO exoskeleton driver** — publishing ``/io_esk/joint_data`` and
  ``/io_fusion/tf_exoskeleton``. Start it before the plugin; the plugin does not
  open the device itself.
- **CloudXR / OpenXR runtime** — the plugin reaches the Teleop session through
  it, and the injected hand skeleton is what the runtime tracks.

Build
-----

Source ROS 2 and configure the plugin target:

.. code-block:: bash

   source /opt/ros/humble/setup.bash
   cmake -B build -DBUILD_PLUGIN_IO_EXO_GLOVE_ROS2=ON
   cmake --build build --target io_exo_glove_ros2_plugin --parallel

The resulting binary is ``build/src/plugins/io_exo_glove_ros2/io_exo_glove_ros2_plugin``.
No vendor SDK is linked: the driver is a separate ROS 2 process.

Configuration
-------------

Topics, collection ids and the fingertip reference frame live in
``config/io_exo_glove_ros2.yaml`` next to the plugin, not in the source. Every
key is optional; precedence, lowest to highest, is the built-in defaults, the
config file, then the command-line flags.

.. code-block:: yaml

   joints_topic: "/io_esk/joint_data"
   links_topic: "/io_fusion/tf_exoskeleton"
   joints_left_collection_id: "exo_glove_joints_left"
   joints_right_collection_id: "exo_glove_joints_right"
   tips_left_collection_id: "exo_glove_tips_left"
   tips_right_collection_id: "exo_glove_tips_right"
   tips_reference_frame: "base_link"
   wrist_source: "auto"
   left_aim_to_wrist: "0,0,0,0,0,0,1"
   right_aim_to_wrist: "0,0,0,0,0,0,1"
   wrist_smoothing_ms: 60

The file is searched in the working directory and the plugin directory, then
via ``$IO_EXO_GLOVE_ROS2_CONFIG``. Unknown keys are rejected rather than
silently ignored, so a typo in a topic name fails at startup.

Published collections
---------------------

Both hands share the two input topics but get **one collection per hand**, so a
glove that drops out is visible as that hand's collection going quiet instead
of as zero angles in a merged collection.

.. list-table::
   :widths: 30 22 48
   :header-rows: 1

   * - Collection id
     - Schema
     - Contents
   * - ``exo_glove_joints_left``
     - ``joint_state``
     - The glove's ``joint_Left...`` DOF angles from ``/io_esk/joint_data``, names
       and radians unchanged.
   * - ``exo_glove_joints_right``
     - ``joint_state``
     - The ``joint_Right...`` half of the same message.
   * - ``exo_glove_tips_left``
     - ``joint_se3_pose``
     - The five left ``<side>_<finger>_tip`` poses from the TF stream, expressed in
       ``tips_reference_frame`` (default ``base_link``).
   * - ``exo_glove_tips_right``
     - ``joint_se3_pose``
     - The five right fingertip poses.

A ``JointState`` whose name and position arrays differ in size is dropped
rather than forwarded partially. A fingertip frame that is not in the configured
reference frame is skipped with a warning instead of being relabelled.

The normal path for a retargeter is the glove's own joint angles plus its
fingertip poses; the hand skeleton is the path the device conformance suite
records, because it reads Isaac Teleop's OpenXR hand tracker rather than a
plugin-owned tensor.

Hand tracking
-------------

The same TF message is injected as the OpenXR hand skeleton (``/hand/left`` and
``/hand/right``, ``type: hand_tracker``) through ``plugin_utils::HandInjector``.
OpenXR requires every one of the 26 joints to be valid while a hand is active,
so all 26 are filled, including ``PALM`` and the four non-thumb metacarpals.

The glove's links are a *linkage mechanism*, not a hand. Injecting them
directly would give a palm a few millimetres deep and fingers twice a palm
long. What carries over is the articulation:

- the **knuckle spread** and the three **bends** per finger, read from each
  published link quaternion against the v5 URDF's rest rotation and revolute
  axis;
- the thumb's **revolute mount** (``ThumbBase``), its **lift** (``Thumb1``) and
  its three **bends** (``Thumb2``..``Thumb4``), which are not a fifth finger's
  axes: the mount and lift are in the palm plane and the bends turn in it.

The injected landmarks come from the vendor's human-hand model
(``blender_human.urdf``), with its CMC positions and segment lengths. Because
the exported model leaves the hand joints un-fanned, the plugin applies an
A-pose metacarpal fan so the knuckles are a physical 56 mm apart and the finger
plane is level, and gives the thumb its own frame so it leaves the palm. The
constants, their derivation and the data-driven checks live in
``io_exo_glove_ros2_kinematics.cpp`` and
``io_exo_glove_ros2_kinematics_tables.py``.

The exoskeleton reports the hand in its own frame, which is not anchored in the
room, so the skeleton is placed at a **wrist pose read from the headset**:

- ``hand_tracking`` — optical hand tracking, the preference under ``auto``;
- ``controller`` — the controller aim pose plus a per-hand rigid offset
  (``left_aim_to_wrist`` / ``right_aim_to_wrist``, ``x,y,z,qx,qy,qz,qw``,
  identity by default);
- ``auto`` — optical first, controller as the fallback.

``$IO_EXO_GLOVE_ROS2_WRIST_SOURCE`` overrides the setting. ``wrist_smoothing_ms``
low-passes the tracked wrist (0 disables); the glove's articulation itself is not
filtered. With no wrist source tracked the skeleton is still injected at the
origin and flagged untracked, rather than frozen at the last known pose.

Running
-------

Start the CloudXR runtime, source its environment, start the vendor driver, then
launch the plugin:

.. code-block:: bash

   source /opt/ros/humble/setup.bash
   ./build/src/plugins/io_exo_glove_ros2/io_exo_glove_ros2_plugin

Applications can instead launch it through its ``plugin.yaml`` using
``TeleopSession``. ``--print-config`` prints the resolved settings and the config
file that supplied them, then exits; ``--help`` lists the command-line overrides.
The plugin runs the ROS 2 executor on the calling thread, so ``Ctrl-C`` shuts it
down cleanly. A fatal ``DeviceIOSession::update()`` failure is not swallowed:
it propagates out of ``rclcpp::spin`` and the process exits non-zero.

Troubleshooting
---------------

.. list-table::
   :widths: 42 58
   :header-rows: 1

   * - Symptom
     - Fix
   * - The plugin starts but no hand is tracked
     - Check the driver is publishing ``/io_fusion/tf_exoskeleton`` and that its
       ``link_<Side>SkeletonBase`` frames are in the message. The plugin logs, once,
       the mechanism root each hand bound to and how many joints it filled.
   * - A hand's collection goes quiet
     - That glove dropped out of the joint or TF stream. Both hands share the input
       topics but each keeps its own collection, so the other hand is unaffected.
   * - Fingertip poses are missing
     - The published tip frame is not ``tips_reference_frame``. The plugin skips
       frames it cannot express in the configured reference frame and logs the
       count; check the driver's frame names or change the setting.
   * - The injected hand sits at the origin
     - No wrist source is tracked. Check optical hand tracking or the controller,
       and ``wrist_source`` / ``$IO_EXO_GLOVE_ROS2_WRIST_SOURCE``.
   * - The injected hand is rotated relative to the glove
     - The controller aim pose is not the wrist joint. Set
       ``left_aim_to_wrist`` / ``right_aim_to_wrist`` for the mount.
   * - Startup fails with "found unknown ROS arguments"
     - The plugin strips the launcher's ``--plugin-root-id``; an unknown argument
       of your own is still passed to ``rclcpp::init``. Check the launch command.

.. seealso::

   :code-file:`src/plugins/io_exo_glove_ros2/README.md` — the plugin's operator
   notes, topic names and the joint/fingertip schemas.

   :doc:`/device/joint_space` — consuming the glove's joint angles through the
   generic joint-space path.
