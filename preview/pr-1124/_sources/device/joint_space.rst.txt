.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Generic Joint-Space Device
==========================

A reusable device path for any **joint-encoder source** -- leader arms, exoskeletons, haptic
gloves, or other articulated input devices. A device streams a name-keyed ``JointStateOutput``
FlatBuffer over the OpenXR tensor transport; one schema, one tracker, one source, and one
retargeter serve them all, so adding a new joint-space device is just a new **plugin** plus a
small **config**.

The **SO-101 leader arm** (`TheRobotStudio SO-ARM100 <https://github.com/TheRobotStudio/SO-ARM100>`_,
6 Feetech STS3215 bus servos) is the reference instance; the **reBot DevArm leader**
(`Seeed reBot DevArm <https://github.com/Seeed-Projects/reBot-DevArm>`_, 7 Damiao DM-series or
RobStride RS-series motors) is a second instance built from the same recipe.

At a glance
-----------

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Layer
     - Component
   * - Schema
     - :code-file:`src/core/schema/fbs/joint_state.fbs` -- ``JointState`` (name + position +
       optional velocity/effort) and ``JointStateOutput`` (a vector of joints + ``device_id``).
   * - Plugin
     - :code-dir:`src/plugins/so101_leader` -- pushes ``JointStateOutput`` via ``SchemaPusher``.
       Reads the FEETECH STS3215 servos over serial (``FeetechBus``); synthetic fallback when no
       device path is given.
   * - Tracker
     - ``JointStateTracker`` (facade) with live (``LiveJointStateTrackerImpl``) and MCAP-replay
       (``ReplayJointStateTrackerImpl``) backends, registered in the live/replay factories.
   * - Source
     - ``JointStateSource`` (``IDeviceIOSource``) -- converts the FlatBuffer into a name-keyed
       group of joint positions for the retargeting graph.
   * - Retargeter
     - ``JointStateRetargeter`` -- ``joint`` (mirror) or ``ee_pose`` (URDF FK) mode. See
       :doc:`/references/retargeting/joint_space`.

Data schema
-----------

Joints are modeled as **name -> value** records so consumers read them by name, independent of
wire order:

.. code-block:: idl
   :class: code-100col

   table JointState {
     name: string (id: 0, key);   // e.g. "shoulder_pan", "gripper"
     position: float (id: 1);     // [rad] revolute, [m] prismatic
     velocity: float (id: 2);     // optional (JointStateOutput.has_velocity)
     effort: float (id: 3);       // optional (JointStateOutput.has_effort)
     valid: bool = true (id: 4);
   }

   table JointStateOutput {
     joints: [JointState] (id: 0);
     device_id: string (id: 1);
     has_velocity: bool (id: 2);
     has_effort: bool (id: 3);
     ee_pose: Pose (id: 4);       // RESERVED: device-side FK; not consumed yet
     ee_pose_valid: bool (id: 5);
   }

The gripper is just another named DOF (conventionally ``"gripper"``). ``velocity``, ``effort``,
and ``ee_pose`` are optional/reserved: the reference plugin and ``JointStateSource`` populate and
surface joint **positions** only.

.. _so101-leader-plugin:

The SO-101 leader plugin
------------------------

``so101_leader`` reads the six SO-101 servos (``shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex, wrist_roll, gripper``) and pushes them to a tensor collection. With a serial device
path it talks to the FEETECH STS3215 bus servos directly via ``FeetechBus`` -- the same SMS/STS
wire protocol the FEETECH SCServo SDK / LeRobot's ``FeetechMotorsBus`` use, with no SDK dependency:
it disables torque (so the leader can be back-driven) and reads ``Present_Position`` each frame,
converting ticks to radians with per-joint calibration. With no device path it falls back to a
**synthetic** trajectory so the pipeline runs hardware-free (CI and the headless example).

.. code-block:: bash

   # Synthetic backend (no hardware), default collection id "so101_leader":
   ./install/plugins/so101_leader/so101_leader_plugin

   # Real SO-101 leader on a serial port (Linux), optional calibration file:
   ./install/plugins/so101_leader/so101_leader_plugin /dev/ttyACM0 so101_leader so101_leader.calib

See the :code-file:`plugin README <src/plugins/so101_leader/README.md>` for hardware setup
(unique servo ids, gear removal, back-driving) and the calibration file format.

The consumer side creates a ``JointStateSource(name=..., collection_id="so101_leader",
joint_names=[...])`` on the same ``collection_id``; ``TeleopSession`` discovers and polls the
``JointStateTracker`` each frame.

.. _rebot-devarm-leader-plugin:

The reBot DevArm leader plugin
------------------------------

``rebot_devarm_leader`` reads the seven joints of the Seeed reBot DevArm (``joint1 .. joint6,
gripper``) and pushes them to a tensor collection, mirroring the SO-101 plugin's structure and
CLI shape. The arm ships in **two motor builds**, and the plugin picks the backend from the shape
of the device argument: a serial path containing ``/`` (e.g. ``/dev/ttyACM0``) selects the
**Damiao** build (:ref:`below <rebot-devarm-damiao-build>`), a bare SocketCAN interface name
(e.g. ``can0``) selects the **RobStride** build (:ref:`below <rebot-devarm-robstride-build>`).
With no device argument it falls back to a **synthetic** trajectory, exactly like
``so101_leader``:

.. code-block:: bash

   # Synthetic backend (no hardware), default collection id "rebot_devarm_leader":
   ./install/plugins/rebot_devarm_leader/rebot_devarm_leader_plugin

Both builds share the calibration file format; see the
:code-file:`plugin README <src/plugins/rebot_devarm_leader/README.md>` for the format
(name, command/feedback CAN ids, motor model, sign, zero offset) and hardware notes.

.. _rebot-devarm-damiao-build:

The Damiao build
~~~~~~~~~~~~~~~~

The **Damiao build** (7 DM-series MIT-protocol motors: DM4340P on joints 1-3, DM4310 on
joints 4-6 and the gripper) puts the motors on a CAN bus behind a Damiao USB-to-CAN serial adapter
(USB CDC-ACM); ``DamiaoBus`` speaks the adapter's fixed-size binary framing directly -- no SDK
dependency. As a *leader*, the plugin sends the **disable** control frame so the arm can be
back-driven by hand (Damiao motors keep answering feedback requests while disabled), then
requests one feedback frame per motor per cycle (command ``0xCC`` addressed via CAN id ``0x7FF``)
and decodes the fixed-point position/velocity feedback, which lands directly in radians -- no
tick conversion, only an optional per-joint sign and zero offset from a calibration file.

.. code-block:: bash

   # Real reBot DevArm on the Damiao USB-to-CAN adapter (Linux), optional calibration file:
   ./install/plugins/rebot_devarm_leader/rebot_devarm_leader_plugin /dev/ttyACM0 rebot_devarm_leader rebot_devarm.calib

   # Probe wiring, motor ids, and the decode path -- no OpenXR runtime needed:
   ./install/plugins/rebot_devarm_leader/rebot_devarm_leader_plugin probe /dev/ttyACM0

``probe`` exits ``0`` when every motor replied, and ``3`` when the motors replied but the gripper
reads outside its physical travel: the Damiao multi-turn counter is volatile across power cycles,
so the gripper (whose geared travel exceeds one turn) can wake up reading ``physical + 2*pi*k``
and must be re-homed (closed against the mechanical stop and re-zeroed) before teleoperating.
While wrapped, the running plugin streams the gripper joint with ``valid = false`` so consumers
hold it instead of executing garbage.

.. _rebot-devarm-robstride-build:

The RobStride build
~~~~~~~~~~~~~~~~~~~

The **RobStride build** (7 RS-series motors) speaks classic CAN at **1 Mbps** through any
SocketCAN adapter (PCAN, candleLight, ...); this backend is Linux-only. ``RobStrideBus``
implements the leader subset of the RobStride private 29-bit extended-id CAN protocol directly --
again no SDK dependency: it sends the **stop** frame (comm type ``0x04``) so the arm can be
back-driven by hand, then alternates single-parameter reads (comm type ``0x11``) of ``mechPos``
(``0x7019``) and ``mechVel`` (``0x701A``) per motor per cycle. Replies carry exact little-endian
IEEE ``f32`` radians / rad/s, so the decode is **model-independent** -- no fixed-point limit
tables. Each channel refreshes at 45 Hz with the plugin's 90 Hz loop, under 20% load of the
1 Mbps bus for 7 motors.

.. code-block:: bash

   # Bring up the SocketCAN interface (once per boot):
   sudo ip link set can0 up type can bitrate 1000000

   # Real reBot DevArm (RobStride build) on a SocketCAN interface:
   ./install/plugins/rebot_devarm_leader/rebot_devarm_leader_plugin can0

   # Probe wiring, motor ids, and the decode path -- no OpenXR runtime needed:
   ./install/plugins/rebot_devarm_leader/rebot_devarm_leader_plugin probe can0

``probe`` uses the same exit codes as the Damiao build (``0`` all motors replied, ``1`` some
missing, ``2`` no device, ``3`` gripper out of travel). The calibration file format is shared;
on the RobStride backend only ``motor_id`` is used (replies are matched by device id, the
``feedback_id`` column is ignored), and ``rs-*`` model names are accepted and ignored since the
decode is model-independent. Factory RobStride device ids are ``1..7``; the host id is ``0xFD``.

Record and replay
-----------------

The live tracker records to MCAP, and ``ReplayJointStateTrackerImpl`` replays it back with no
OpenXR runtime, so a recorded session drives the retargeting graph headlessly:

.. code-block:: python

   from isaacteleop.deviceio import McapRecordingConfig, McapReplayConfig
   from isaacteleop.teleop_session_manager import SessionMode, TeleopSession, TeleopSessionConfig

   # Record (live): TeleopSessionConfig(..., mcap_config=McapRecordingConfig("leader.mcap"))
   # Replay (headless): TeleopSessionConfig(..., mode=SessionMode.REPLAY,
   #                                        mcap_config=McapReplayConfig("leader.mcap"))

Add another joint-space device
------------------------------

Reuse everything above by writing only:

#. A **plugin** that reads your hardware and fills ``JointStateOutput`` (positions; optionally
   velocity/effort), modeled on :code-dir:`src/plugins/so101_leader` (or
   :code-dir:`src/plugins/rebot_devarm_leader`, a second instance of the same recipe on a
   different motor bus).
#. A **config**: a ``collection_id``, the device joint names, and -- for ``ee_pose`` mode -- a URDF
   and end-effector link.

The schema, ``JointStateTracker``, ``JointStateSource``, and ``JointStateRetargeter`` are unchanged.

.. seealso::

   :doc:`add_device` -- the general four-step device-plugin recipe (foot-pedal reference).

   :doc:`/references/retargeting/joint_space` -- the ``JointStateRetargeter`` (joint / EE modes),
   the end-to-end example, and validation.
