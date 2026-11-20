.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _robot-state:

Robot State
===========

Robot state is data about the robot: its joint positions, a gripper's opening,
status readings, faults, and whatever else a robot has. A robot app in a
:ref:`remote session <remote-sessions>` hands it to its ``TeleopSession`` as
numbers; Isaac Capture sends it to the Teleop Session Server, where a reader,
for example an agent's harness, imports ``isaaccapture.robot_state`` and asks
for it by name.

Overview
--------

.. code-block:: text

    robot machine                           server machine
    robot app                               Teleop Session Server
      TeleopSession (REMOTE)                  keeps the newest reading of each series
        publish_robot_state("arm", ...) ══►        │
                                                    ▼
                                             reader program
                                               RobotState.connect().joints("arm")

Three things are needed:

#. A **robot description**: a small file that says what the robot is. A
   six-joint arm and a two-arm robot differ only in their files.
#. The **robot app** names the file in its ``TeleopSessionConfig`` and calls
   ``publish_robot_state()`` with the numbers it reads from the robot.
#. A **reader** on the server machine calls ``RobotState.connect()`` and reads
   the description, the joints by name, a gripper, the status or any other
   series, each with its age.

The robot app and the reader see no socket, port or message format.

Robot state is made of **data series**: a series is a named sequence of small
records over time, for example the series ``arm`` with one record of joint
positions per reading, some tens a second. It is not video. A **shape** is the
layout of a series' records. There are two standard shapes to pick from,
``joints`` and ``status``, and a robot can define shapes of its own.

**Requirements.** Robot state travels in a remote session, so everything under
:ref:`remote-sessions` applies. Run the same Isaac Capture version in the robot
app, the Teleop Session Server and the reader. The reader runs on the server
machine, as the same user as the server, in a Python environment that has the
``isaaccapture`` package.

Describe the Robot
------------------

The description is a `TOML <https://toml.io>`_ file. This one describes a
six-joint arm with one gripper
(:code-file:`six_joint_arm.toml <examples/robot_state/python/isaaccapture_examples/robot_state/descriptions/six_joint_arm.toml>`):

.. code-block:: toml

   robot = "rebot_b601"

   [series.arm]
   shape = "joints"
   rate_hz = 20.0
   joints = [
     { name = "joint1",  kind = "revolute", min = -2.8,  max = 2.8 },
     { name = "joint2",  kind = "revolute", min = -0.05, max = 3.14 },
     { name = "joint3",  kind = "revolute", min = -0.05, max = 3.14 },
     { name = "joint4",  kind = "revolute", min = -1.69, max = 1.79 },
     { name = "joint5",  kind = "revolute", min = -1.57, max = 1.57 },
     { name = "joint6",  kind = "revolute", min = -3.14, max = 3.14 },
     { name = "gripper", kind = "gripper",  min = 0.0,   max = 1.0 },  # 0 closed, 1 open
   ]

   [series.status]
   shape = "status"
   readings = ["motors_on", "fault"]

It declares two series. ``arm`` has the standard shape ``joints``: one record
holds a position for each of the seven joints, in the order of the file. The
gripper is one more named joint. ``status`` has the standard shape ``status``:
one record holds the two named readings and the faults present at that moment.

A two-arm robot with a waist and a force sensor in each wrist
(:code-file:`two_arm_robot.toml <examples/robot_state/python/isaaccapture_examples/robot_state/descriptions/two_arm_robot.toml>`)
needs a longer file and no other change:

.. code-block:: toml

   robot = "two_arm_example"

   # A shape this robot defines: one record holds these fields.
   [shapes.force]
   fields = [
     { name = "fx", unit = "N" },
     { name = "fy", unit = "N" },
     { name = "fz", unit = "N" },
     { name = "tx", unit = "N*m" },
     { name = "ty", unit = "N*m" },
     { name = "tz", unit = "N*m" },
   ]

   [series.left_arm]
   shape = "joints"
   has_effort = true
   joints = [
     { name = "l_shoulder_pan",  kind = "revolute",   min = -3.1, max = 3.1, max_velocity = 2.0, max_effort = 40.0 },
     { name = "l_shoulder_lift", kind = "revolute",   min = -2.0, max = 2.0, max_velocity = 2.0, max_effort = 40.0 },
     { name = "l_elbow",         kind = "revolute",   min = -2.5, max = 2.5, max_velocity = 2.5, max_effort = 25.0 },
     { name = "l_wrist_roll",    kind = "continuous" },
     { name = "l_gripper",       kind = "gripper",    min = 0.0, max = 0.08, unit = "m" },
   ]

   [series.right_arm]
   shape = "joints"
   has_effort = true
   joints = [
     { name = "r_shoulder_pan",  kind = "revolute",   min = -3.1, max = 3.1, max_velocity = 2.0, max_effort = 40.0 },
     { name = "r_shoulder_lift", kind = "revolute",   min = -2.0, max = 2.0, max_velocity = 2.0, max_effort = 40.0 },
     { name = "r_elbow",         kind = "revolute",   min = -2.5, max = 2.5, max_velocity = 2.5, max_effort = 25.0 },
     { name = "r_wrist_roll",    kind = "continuous" },
     { name = "r_gripper",       kind = "gripper",    min = 0.0, max = 0.08, unit = "m" },
   ]

   [series.waist]
   shape = "joints"
   joints = [
     { name = "waist_yaw", kind = "revolute", min = -1.5, max = 1.5 },
   ]

   [series.left_wrist_force]
   shape = "force"
   rate_hz = 50.0
   attributes = { at = "l_wrist" }

   [series.right_wrist_force]
   shape = "force"
   rate_hz = 50.0
   attributes = { at = "r_wrist" }

   [series.status]
   shape = "status"
   stale_after_s = 2.0
   readings = ["motors_on", "fault", { name = "battery", kind = "number", unit = "V" }]

The force sensors use a shape the robot defines, ``force``. The server needs
no change to carry it: each series tells the server its shape when the robot
app connects.

.. list-table::
   :header-rows: 1
   :widths: 22 24 54

   * - Where
     - Key
     - Value
   * - top of the file
     - ``robot`` (required)
     - The robot's name: 1 to 64 letters, digits, ``_``, ``.`` or ``-``.
   * - top of the file
     - ``attributes``
     - A table of texts, handed to the reader as given.
   * - ``[series.NAME]``
     - ``shape`` (required)
     - ``joints``, ``status``, or the name of a shape under ``[shapes]``.
   * - ``[series.NAME]``
     - ``stale_after_s``
     - A reading older than this many seconds is not fresh. Default 0.5;
       0 means it never goes stale.
   * - ``[series.NAME]``
     - ``rate_hz``
     - How many readings a second the robot app intends to publish. It is
       handed to the reader and changes nothing else.
   * - ``[series.NAME]``
     - ``attributes``
     - A table of texts, handed to the reader as given.
   * - a ``joints`` series
     - ``joints`` (required)
     - A list of joints: ``name``, ``kind`` (``revolute``, ``prismatic``,
       ``continuous`` or ``gripper``), ``min`` and ``max`` (required except for
       ``continuous``), and optionally ``max_velocity``, ``max_effort``,
       ``unit``, and for a gripper ``open`` and ``closed``.
   * - a ``joints`` series
     - ``has_velocity``, ``has_effort``
     - ``true`` when every record also holds a velocity or an effort for each
       joint. Default ``false``.
   * - a ``status`` series
     - ``readings`` (required)
     - A list of names, or of ``{ name, kind, unit }`` with ``kind`` one of
       ``any`` (default), ``number``, ``flag``, ``text``.
   * - ``[shapes.NAME]``
     - ``fields`` (required)
     - A list of names, or of ``{ name, kind, unit }`` with ``kind`` one of
       ``number`` (default), ``flag``, ``text``, ``numbers``, ``any``.
   * - ``[grippers.NAME]``
     - ``series``, ``joints``, ``open``, ``closed`` (all required)
     - A gripper made of several joints of one ``joints`` series. The first
       joint measures the opening; ``open`` and ``closed`` are its positions
       when fully open and fully closed.

- **Order matters.** The order of joints, readings and fields in the file is
  the order of the numbers the robot app passes.
- **Names** of series, shapes, joints, readings, fields and grippers are 1 to
  64 letters, digits or ``_`` and start with a letter. Joint names are unique
  within a series. The series name ``robot_description`` and the shape names
  ``joints`` and ``status`` are reserved.
- **Units.** A ``revolute`` or ``continuous`` joint is in rad and a
  ``prismatic`` joint in m unless ``unit`` says otherwise. A ``gripper`` joint
  is in the unit the file states.
- **Grippers.** A joint of kind ``gripper`` declares a gripper of the same
  name, closed at ``min`` and open at ``max``; give ``open`` and ``closed`` in
  the joint entry when it is the other way round. Use ``[grippers.NAME]`` for
  a gripper of several joints. Gripper names are unique across the robot: two
  series that each have a joint of kind ``gripper`` name them differently, as
  ``l_gripper`` and ``r_gripper`` above.
- **Limits.** At most 64 series, 256 joints in a series, 256 readings or
  fields, and 256 characters in an attribute value.
- **Mistakes are refused.** A key the library does not know, a missing limit,
  a name used twice and every other mistake raise ``RobotDescriptionError`` (a
  ``ValueError``) whose text starts with the file's path and names the series
  and joint. Check a file without a robot:

  .. code-block:: python

     from isaaccapture.robot_state import load_robot_description

     description = load_robot_description("my_robot.toml")
     print(description.robot, list(description.series))

Send Robot State
----------------

Name the description in the configuration of a remote session and publish
readings while the session is entered:

.. code-block:: python

   import time

   from isaaccapture.teleop_session_manager import (
       SessionMode,
       TeleopSession,
       TeleopSessionConfig,
   )

   config = TeleopSessionConfig(
       app_name="MyRobotApp",
       pipeline=pipeline,
       mode=SessionMode.REMOTE,
       remote_endpoint="server-machine:47600",
       robot_description="six_joint_arm.toml",
   )
   with TeleopSession(config) as session:
       while True:
           send_to_robot(session.step())

           read_time_ns = time.monotonic_ns()
           positions = read_joints()            # 7 numbers, in the order of the file
           session.publish_robot_state("arm", positions, time_ns=read_time_ns)
           session.publish_robot_state("status", [motors_on(), fault_text()])

``publish_robot_state(series, values, *, velocities=None, efforts=None,
valid=None, faults=None, time_ns=None)`` sends one record of one series. The
library checks the values against the description, adds the names and the time
of the reading, and sends the record at once; it does not wait for ``step()``.
It can be called from any thread while the session is entered.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Shape of the series
     - What to pass
   * - ``joints``
     - ``values``: one position per joint, in the order of the file.
       ``velocities`` and ``efforts``: one number per joint; required when the
       series declares ``has_velocity`` or ``has_effort`` and refused
       otherwise. ``valid``: optionally one ``bool`` per joint; pass ``False``
       for a joint whose read failed. A reader gets nan for the numbers of
       that joint, whatever was passed for them.
   * - ``status``
     - ``values``: one value per reading, in the order of the file: a
       ``bool``, a number, a text, or ``None`` for a reading the robot could
       not take. ``faults``: the faults present at this moment, each
       ``(code, text)``, ``(code, text, source)`` or
       ``(code, text, source, severity)`` with severity ``"warning"``,
       ``"error"`` or ``"fatal"``. A fault that ended is left out of the next
       record.
   * - a shape the robot defines
     - ``values``: a mapping from field name to value, or the values in the
       order of the file. ``publish_robot_record(series, record, *,
       time_ns=None)`` does the same and reads better for this case.

For the two-arm robot:

.. code-block:: python

   session.publish_robot_state("left_arm", positions, efforts=efforts)
   session.publish_robot_state(
       "status",
       [True, "", 47.9],
       faults=[("overtemperature", "left elbow motor is hot", "l_elbow", "warning")],
   )
   session.publish_robot_record(
       "left_wrist_force",
       {"fx": 0.1, "fy": -0.3, "fz": 9.6, "tx": 0.0, "ty": 0.02, "tz": 0.0},
   )

- **Time of the reading.** ``time_ns`` is ``time.monotonic_ns()`` on the robot
  machine at the moment the robot was read; leave it out to use the moment of
  the call. Do not pass ``time.time_ns()`` or ``time.perf_counter_ns()``: the
  reader's ages would be wrong.
- **Return value.** ``True`` when the record was handed to the link. ``False``
  when the server does not take robot state or the link is down; ``step()``
  raises the lost link as described under :ref:`remote-sessions`.
  ``session.robot_state_active`` is ``True`` once the server has agreed to
  take robot state.
- **Errors.** ``ValueError`` naming the series and what is wrong: a series the
  description does not have, a wrong number of values, missing or undeclared
  ``velocities`` or ``efforts``, a number that is not finite, a status value
  of another kind than declared, a record with missing or unknown fields.
  ``RuntimeError`` when the configuration has no ``robot_description`` or the
  session is not entered. A ``robot_description`` in another mode than
  ``SessionMode.REMOTE`` raises ``ValueError`` when the configuration is
  created, and a mistake in the file raises ``RobotDescriptionError`` when the
  session is entered, before it connects.
- **When sending starts.** The description is sent as soon as the server has
  accepted the robot app. Readings can be published once the session is
  entered, and entering waits for a tracked headset as every remote session
  does, so a robot app whose server has no headset sends its description
  only.
- **A server without robot state.** A Teleop Session Server of a version
  without robot state serves the robot app's trackers as usual. The robot app
  logs ``The Teleop Session Server at <endpoint> does not take robot state;
  none is sent`` once, and the publish calls return ``False``.
- **Slow network.** Of a series' readings only the newest matters, so when the
  network cannot keep up, older readings are dropped instead of queued. The
  description always arrives. A record is at most 1 MiB.

Without ``robot_description`` a remote session behaves as described under
:ref:`remote-sessions`, with nothing added.

Read Robot State
----------------

On the server machine, in any program:

.. code-block:: python

   from isaaccapture.robot_state import RobotState

   robot = RobotState.connect()

   what = robot.description()              # None until a robot app has connected
   what.robot                              # "rebot_b601"
   list(what.series)                       # ["arm", "status"]
   what.series["arm"].joint_names          # ["joint1", ..., "joint6", "gripper"]
   what.series["arm"].limits["joint2"]     # (-0.05, 3.14)

   arm = robot.joints("arm")               # the newest reading, or None
   if arm is not None:
       arm["joint2"].position              # 1.204 (rad)
       arm.positions                       # all seven, in the order of the file
       arm.age_s                           # 0.03: how old this reading is

   robot.gripper("gripper").opening        # 0.96: 0 is closed, 1 is open
   robot.status()["fault"]                 # ""
   robot.status().faults                   # ()

The same reader works with any robot, because the description says which names
exist. For the two-arm robot it calls ``robot.joints("left_arm")`` and
``robot.joints("right_arm")``. A reader that wants every joint of whatever
robot is connected never names one:

.. code-block:: python

   for name, series in robot.series().items():
       if series.shape == "joints":
           reading = robot.joints(name)
           if reading is not None:
               for joint in reading.values():
                   if joint.valid:
                       print(name, joint.name, joint.position)

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Call
     - Result
   * - ``RobotState.connect(socket_path=None, *, timeout_s=1.0, auto_refresh=True)``
     - A reader. It never raises because the server is not running.
   * - ``robot.description()``
     - The ``RobotDescription`` the robot app sent: ``robot``, ``series``,
       ``grippers``, ``attributes``. ``None`` while there is none.
   * - ``robot.series()``
     - The series by name, in the order of the file; empty without a
       description. Each has ``shape`` (``"joints"``, ``"status"`` or
       ``"custom"``), ``shape_name``, ``joints``, ``joint_names``, ``limits``,
       ``values``, ``value_names``, ``has_velocity``, ``has_effort``,
       ``rate_hz``, ``stale_after_s`` and ``attributes``.
   * - ``robot.joints(series)``
     - A mapping from joint name to a reading with ``position``, ``velocity``,
       ``effort`` (``None`` unless the series has them) and ``valid``; also
       ``names`` and ``positions``. The numbers of a joint whose ``valid`` is
       ``False`` are nan, in ``positions`` too.
   * - ``robot.gripper(name)``
     - ``position``, ``valid``, and ``opening`` from 0 (closed) to 1 (open);
       ``position`` and ``opening`` are nan when ``valid`` is ``False``.
   * - ``robot.status(series="status")``
     - A mapping from reading name to its value (``None`` for a reading the
       robot could not take), and ``faults``: a tuple of ``Fault(code, text,
       source, severity)``.
   * - ``robot.record(series)``
     - The newest record of any series: as ``joints()`` or ``status()`` for
       those shapes; for a shape the robot defines, ``values`` holds its
       fields as a ``dict``, for example
       ``robot.record("left_wrist_force").values["fz"]``.
   * - ``robot.connected``
     - ``True`` while a robot app that sends robot state is connected to the
       server.
   * - ``robot.server_reachable``
     - ``True`` while the reader gets answers from the server.
   * - ``robot.refresh()``
     - Fetch the newest readings; ``False`` when the server did not answer.
       Needed only with ``auto_refresh=False``.
   * - ``robot.close()``
     - Close the reader. ``RobotState`` is also a context manager.

Every reading also has ``series``, ``age_s``, ``receive_age_s``, ``fresh``,
``clock_two_way``, ``sequence`` and ``sample_time_ns`` (see `Ages`_).

- **A read can return None.** ``joints()``, ``gripper()``, ``status()`` and
  ``record()`` return ``None`` when the series has no reading yet, when its
  newest reading is not fresh, while no robot app is connected, and while the
  server cannot be reached. Check for ``None`` before using a reading. Pass
  ``allow_stale=True`` to get the newest reading the server has even when it
  is not fresh or the robot app has left; ``fresh`` and ``age_s`` then say how
  far to trust it.
- **A joint the robot could not read.** Its ``valid`` is ``False`` and its
  numbers are nan, never a number that could pass for a position
  (``velocity`` and ``effort`` stay ``None`` when the series has none). Check
  ``valid`` before using a joint.
- **Unknown names.** A series or gripper the description does not have raises
  ``KeyError`` naming the ones it has. A call for another shape than the
  series has, for example ``joints("status")``, raises ``ValueError``.
- **A new robot app.** When a robot app connects, the readings and the
  description of the one before it are gone. ``description()`` and
  ``series()`` stay readable after a robot app has left, until the next one
  connects.
- **Refreshing.** With ``auto_refresh=True`` each call fetches the newest
  readings first. With ``auto_refresh=False``, call ``refresh()`` once per
  cycle; every read until the next ``refresh()`` then sees the same moment.
- **Inside the server's process.** Code that runs in the Teleop Session
  Server's own process gets the same object from
  ``isaaccapture.teleop_session_server.hosted_robot_state()``.
- **Threads.** One ``RobotState`` can be used from several threads.

A reader program and the Teleop Session Server exchange robot state over a
local socket that belongs to the isaaccapture package. Its format is private
and may change in any release; it is not an interface. Use the calls of
``isaaccapture.robot_state``.

The server opens that socket when the first robot app that sends robot state
connects; until then ``server_reachable`` is ``False`` although the server
runs. The default path is ``$XDG_RUNTIME_DIR/isaaccapture/robot_state.sock``,
or ``/tmp/isaaccapture-<uid>/robot_state.sock`` without ``XDG_RUNTIME_DIR``.
To use another path, start the server with ``--robot-state-socket PATH`` and
pass the same path as ``RobotState.connect(socket_path=PATH)``. The directory
of the socket must belong to the user that runs the server and be closed to
group and others, and the path must be at most 103 bytes long. A server and a
reader that see different values of ``XDG_RUNTIME_DIR``, for example a server
started as a system service and a reader started in a login shell, look in
different places; give both the same path then.

At most 16 reader programs read one server at a time; one ``RobotState`` is
one of them, however many threads use it. The server closes a further one
and logs one warning; that reader sees ``server_reachable`` ``False`` and its
reads return ``None`` until another reader has closed.

Ages
----

``age_s`` is the time since the robot app took the reading, measured on the
server machine: the library converts the robot machine's clock to the server
machine's with the clock estimate of the remote session.

- ``fresh`` is ``True`` while the robot app is connected and ``age_s`` is at
  most the series' ``stale_after_s``. A series with ``stale_after_s = 0`` is
  fresh for as long as its robot app is connected.
- During the first moments of a session the clock estimate is one-way:
  ``clock_two_way`` is ``False`` and ``age_s`` is too large by the network
  delay, never too small.
- ``receive_age_s`` is the time since the reading arrived at the server. It
  needs no clock estimate and leaves out the time the reading spent on the
  way.
- ``age_s`` is never less than ``receive_age_s``. A reading whose ``time_ns``
  lies ahead of the robot machine's ``time.monotonic_ns()`` therefore ages
  from its arrival and goes stale like any other.
- ``sample_time_ns`` is the time of the reading on the robot machine's clock.
  Do not compare it with a time taken on the server machine; use ``age_s``.
- ``sequence`` counts the readings of the series the server has taken since
  the robot app connected.

When Something Goes Away
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Event
     - What the reader and the robot app see
   * - The robot app leaves, or its link is lost
     - ``connected`` turns ``False`` and reads return ``None``. The robot
       app's ``step()`` raises on a lost link and its publish calls return
       ``False``; it does not reconnect by itself. Robot state resumes when
       the robot app enters a new session.
   * - The headset disconnects and connects again
     - The Teleop Session Server keeps running and robot state keeps
       arriving. A robot app that enters a session while no headset is
       tracked waits, and publishes no readings until one is.
   * - CloudXR crashes
     - **The Teleop Session Server exits** (see **Failures** under
       "Teleop Session Server" in :doc:`teleop_session`). Robot state stops
       reaching the
       server machine until the server runs again and the robot app has
       entered a new session, which again needs CloudXR and a tracked
       headset. Meanwhile the reader's ``server_reachable`` is ``False`` and
       its reads return ``None``. Whether the server should exit in this case
       is undecided; run the server under a supervisor that restarts it.
   * - The Teleop Session Server is stopped or restarted
     - As above: ``server_reachable`` is ``False`` and every read, also with
       ``allow_stale=True``, returns ``None`` until the server answers again.
       The reader keeps trying; it need not be restarted.
   * - The reader starts before the server, or before any robot app has
       connected
     - ``RobotState.connect()`` succeeds, ``server_reachable`` is ``False``,
       reads return ``None``.
   * - More than 16 reader programs read one server
     - Each one above 16 sees ``server_reachable`` ``False`` and its reads
       return ``None``; the server logs one warning. It reads as soon as
       another reader has closed; share one ``RobotState`` among the threads
       of a program.
   * - The reader and the server run different Isaac Capture versions
     - They may not exchange robot state: the reader logs one warning that
       says so, and its reads return ``None``.

Security
--------

The link between the robot app and the Teleop Session Server has no
authentication and no encryption (see :ref:`remote-sessions`, **Security**).
With robot state this also means:

- Any machine that reaches the server's port can connect as a robot app and
  send robot state. A reader takes what it sent as the robot's: a false
  description, false joint positions, false faults. Readers cannot tell the
  difference. Do not let robot state alone decide that a robot may move.
- Such a machine cannot read robot state through the link: robot state
  travels only from the robot app to the server.
- On the server machine, programs of the user that runs the server can read
  robot state; other users cannot. The local socket is closed to other users
  and is not reachable from another machine. A reader cannot change robot
  state through it.

Run the Teleop Session Server on a trusted network or a VPN, and open its
ports only toward the robot machine.

Example
-------

:code-dir:`examples/robot_state` runs all three parts on one machine, without
a robot and without a headset: a stand-in robot publishes every series of a
description through a remote session, and a stand-in harness prints what it
reads.

.. code-block:: bash

   uv pip install -e ./examples/robot_state
   python -m isaaccapture_examples.robot_state
   python -m isaaccapture_examples.robot_state --description two_arm_robot

Its server listens on a loopback port the system picks and keeps its reader
socket in a temporary folder, so it does not disturb a Teleop Session Server
that runs on the same machine.

Not Included
------------

Robot state is what the robot reports. These stay in the application: an
agent's commands and their answers, who controls the robot, stop, the handover
between an agent and a person, and camera pictures.

Not available yet: robot state from the server to the robot app; a history of
readings (the server keeps the newest of each series); a gripper's position in
space; more than one robot app at a time; recording robot state to MCAP; a C++
call to publish from a description file.
