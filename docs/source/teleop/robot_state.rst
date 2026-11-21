.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _robot-state:

Robot State
===========

Robot state is data about the robot: its joint positions, a gripper's opening,
status readings, faults, and whatever else a robot has. A robot app hands it
to its ``TeleopSession`` as numbers; Isaac Capture carries it to a reader, for
example an agent's harness, which imports ``isaaccapture.robot_state`` and
asks for it by name. The robot app and the reader are written once: they are
the same in a local session and in a :ref:`remote session <remote-sessions>`.

Overview
--------

.. code-block:: text

    Remote session (SessionMode.REMOTE)

    robot machine                           server machine
    robot app                               Teleop Session Server
      TeleopSession                           keeps the newest reading of each series
        publish_robot_state("arm", ...) ══►        │
                                                    ▼
                                             reader program
                                               RobotState.connect().joints("arm")

    Local session (SessionMode.LIVE or SessionMode.REPLAY)

    one machine
    robot app
      TeleopSession
        publish_robot_state("arm", ...)
        keeps the newest reading of each series
             │
             ▼
    reader program
      RobotState.connect().joints("arm")

Three things are needed:

#. A **robot description**: a small file that says what the robot is. A
   six-joint arm and a two-arm robot differ only in their files.
#. The **robot app** names the file in its ``TeleopSessionConfig`` and calls
   ``publish_robot_state()`` with the numbers it reads from the robot.
#. A **reader** calls ``RobotState.connect()`` and reads the description, the
   joints by name, a gripper, the status or any other series, each with its
   age. It runs on the machine that runs the headset runtime: the server
   machine in a remote session, the robot app's own machine in a local one.

The robot app and the reader see no socket, port or message format.

Robot state is made of **data series**: a series is a named sequence of small
records over time, for example the series ``arm`` with one record of joint
positions per reading, some tens a second. It is not video. A **shape** is the
layout of a series' records. There are two standard shapes to pick from,
``joints`` and ``status``, and a robot can define shapes of its own.

**Requirements.** Robot state works in a local session (``SessionMode.LIVE``
and ``SessionMode.REPLAY``) and in a remote session (``SessionMode.REMOTE``,
where everything under :ref:`remote-sessions` applies). The reader runs on the
machine that runs the headset runtime, as the same user as the program that
serves robot state there (the Teleop Session Server in a remote session, the
robot app in a local one), in a Python environment that has the
``isaaccapture`` package. Run the same Isaac Capture version in the robot app,
the Teleop Session Server and the reader. Reader programs need Unix sockets:
on a system without them a robot app in a local session logs one warning and
its publish calls return ``False``. One robot's state is served on a machine:
a second robot is refused when it starts (see `One Robot per Machine`_).

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

The force sensors use a shape the robot defines, ``force``. Neither the
library nor a Teleop Session Server needs a change to carry it: each series
declares its shape when the robot app enters its session.

.. list-table::
   :header-rows: 1
   :widths: 22 24 54

   * - Where
     - Key
     - Value
   * - top of the file
     - ``robot`` (required)
     - The robot's name: 1 to 64 letters, digits, ``_``, ``.`` or ``-``. A
       reader can ask for this robot by name (see `Read Robot State`_).
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

Name the description in the session's configuration and publish readings
while the session is entered. One robot app runs in a local and in a remote
session:

.. code-block:: python

   import time

   from isaaccapture.teleop_session_manager import (
       SessionMode,
       TeleopSession,
       TeleopSessionConfig,
   )

   endpoint = settings.get("server")  # "server-machine:47600", or None for a local session
   config = TeleopSessionConfig(
       app_name="MyRobotApp",
       pipeline=pipeline,
       mode=SessionMode.LIVE if endpoint is None else SessionMode.REMOTE,
       remote_endpoint=endpoint,
       robot_description="six_joint_arm.toml",
       wait_for_headset=False,        # see "With Nobody in the Headset"
   )
   with TeleopSession(config) as session:
       while True:
           command = session.step()   # tracker outputs are absent until a headset is tracked
           if operator_holds_grip(command):     # the robot app's own test, see below
               send_to_robot(command)

           read_time_ns = time.monotonic_ns()
           positions = read_joints()            # 7 numbers, in the order of the file
           session.publish_robot_state("arm", positions, time_ns=read_time_ns)
           session.publish_robot_state("status", [motors_on(), fault_text()])

The pair ``mode`` and ``remote_endpoint`` is the one place where the robot app
names its session: ``SessionMode.REMOTE`` requires an endpoint and the other
modes refuse one. Everything else, the publish calls included, is the same. In
a remote session the readings go through the Teleop Session Server to readers
on the server machine; in a local session the robot app serves them to readers
on its own machine.

``operator_holds_grip`` is a function of the robot app, not of the library: it
says whether this step has tracker data and a deliberate operator input. With
``wait_for_headset=False`` the pipeline runs before any tracker has data, and
the outputs of most retargeters do not show that. `What a Pipeline Gives With
No Tracker Data`_ lists what each retargeter writes then and shows such a
function.

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
- **Return value.** ``True`` when the record was handed over for the readers:
  to the link in a remote session, to the robot app's own readers in a local
  one. ``False`` while robot state is not active or the link is down;
  ``step()`` raises the lost link as described under :ref:`remote-sessions`.
- **Whether robot state is active.** ``session.robot_state_active`` is
  ``True`` while the session is entered and the Teleop Session Server has
  agreed to take robot state (remote session), or the robot app serves it to
  reader programs (local session). It is ``False`` with a server that does not
  take robot state, and in a local session while the robot app cannot open the
  socket that reader programs use (see **A reader socket that cannot be
  opened** under `Read Robot State`_). Keep publishing while it is ``False``:
  in a local session the socket is tried again only when a publish call asks
  for it. After a remote session's link is lost it stays ``True`` until the
  session is left; the publish calls return ``False`` then.
- **Errors.** ``ValueError`` naming the series and what is wrong: a series the
  description does not have, a wrong number of values, missing or undeclared
  ``velocities`` or ``efforts``, a number that is not finite, a status value
  of another kind than declared, a record with missing or unknown fields, a
  record above 1 MiB. ``RuntimeError`` when the configuration has no
  ``robot_description`` or the session is not entered. A mistake in the file
  raises ``RobotDescriptionError`` when the session is entered, before it
  connects to a server or opens an OpenXR session or a recording. Entering
  also raises ``RuntimeError`` for a robot that would be the second one on its
  machine (see `One Robot per Machine`_). The calls return and raise the same
  in every mode; only the text for a record above 1 MiB differs.
- **When sending starts.** Once entering the session has returned, in every
  mode. The description is available to readers earlier: a remote session
  sends it as soon as the server has accepted the robot app, a local session
  serves it before it opens its OpenXR session or recording. By default
  entering waits for a headset; with ``wait_for_headset=False`` it does not
  (see `With Nobody in the Headset`_).
- **A server without robot state.** A Teleop Session Server of a version
  without robot state serves the robot app's trackers as usual. The robot app
  logs ``The Teleop Session Server at <endpoint> does not take robot state;
  none is sent`` once, and the publish calls return ``False``.
- **Slow network.** Of a series' readings only the newest matters, so when the
  network cannot keep up, older readings are dropped instead of queued. The
  description always arrives. A record is at most 1 MiB. A local session has
  no network: every record is kept at once as the newest of its series.

Without ``robot_description`` a session behaves as it does without robot
state, with nothing added: a local session opens no socket and starts no
thread for it.

One Robot per Machine
---------------------

Reader programs get robot state at one place on a machine: the **reader
socket**, a local socket of the program that serves robot state. Its default
path is the same for a Teleop Session Server and for every robot app in a
local session of one user (see **Where the socket is** under `Read Robot
State`_), and one program serves a path. So one robot's state is served on a
machine. The machine meant is the one that runs the headset runtime: the
server machine in a remote session, the robot app's own machine in a local
one. One headset cannot sensibly drive two robots, for example a real robot
in a remote session and a simulated robot in a local session on the server
machine. A robot that would be the second one is therefore **refused when it
starts** and does not teleoperate.

- **Which one is refused.** The one that starts second: the robot whose
  program finds, at the moment it starts, that another running program serves
  the path. Which of the two is the real robot, and which is in a remote
  session, makes no difference. A program that has ended holds nothing, also
  one that was killed: the next robot starts. A child process made by a fork
  that skips Python's fork hooks keeps the path until it ends.
- **A second robot app in a local session.** Entering the session raises
  ``RuntimeError``: ``Another program on this machine already serves a robot's
  state at <path>: a Teleop Session Server or another robot app. One robot per
  machine is served; stop that program, then start this robot app again.`` It
  is raised before the session opens its OpenXR session or its recording and
  before it starts a plugin, so nothing of the refused robot app runs.
- **A second robot at a Teleop Session Server.** A robot app with a
  ``robot_description`` connects to a server while another program on the
  server machine serves the server's reader socket path, for example a robot
  app in a local session there. The server refuses the robot app, and entering
  the session raises ``RuntimeError`` with the server's text: ``Another
  program on the Teleop Session Server's machine already serves a robot's
  state at <path>. One robot per machine is served; stop that program, or
  connect this robot app without a robot description.`` It is raised when the
  server answers, before any wait for a headset. The server logs ``Refused
  the robot app at <address>:`` with the same text, keeps running and serves
  the next robot app.
- **What the refused robot app should do.** Stop the other program, then
  start again. Entering does not try again by itself, in either case; a robot
  app that wants to wait for the other program catches the error and enters
  the same ``TeleopSession`` again. The error is a plain ``RuntimeError``,
  told from others by its text.
- **Without a robot description nothing is refused.** A robot app that sets no
  ``robot_description`` enters a local session, and is served by a Teleop
  Session Server, whichever program serves the path. It publishes no robot
  state.
- **A Teleop Session Server keeps the path until it stops.** The server opens
  its reader socket for the first robot app that sends robot state and keeps
  it until the server stops, so a harness stays connected while one robot app
  leaves and the next one connects. While such a server runs, a robot app with
  a ``robot_description`` in a local session on that machine is refused, also
  in the time between two robot apps. Stop the server first. A server that has
  served no robot app with robot state yet holds no path, and a robot app it
  refused does not make it take one.
- **Two robots on purpose.** Give each its own path, and each reader the path
  of its robot. Start a Teleop Session Server with ``--robot-state-socket
  PATH`` and pass the same path to its readers:
  ``RobotState.connect(socket_path=PATH)``. A robot app in a local session has
  no option for the path: start it and its readers with ``XDG_RUNTIME_DIR``
  set to another existing directory of that user. Other software reads that
  variable too, and this way has not been run with an OpenXR runtime.
  Programs of another user have a default path of their own.
- **The rule holds per path.** Two programs that see different values of
  ``XDG_RUNTIME_DIR``, for example a Teleop Session Server started as a system
  service, which sees none, and a robot app started in a login shell, have two
  default paths: neither robot is refused, and each is read at its own path.
  Where one robot per machine is meant, give both programs the same value.
- **What stays a warning.** Only another running program at the path refuses a
  robot at its start. A reader socket that cannot be opened for any other
  reason is a warning: the robot teleoperates, and the socket is tried again
  (see **A reader socket that cannot be opened** under `Read Robot State`_).

With Nobody in the Headset
--------------------------

By default entering a session waits for a headset, and a robot app publishes
nothing while it waits. A robot app that reports its robot at all times, for
example to an agent's harness, sets ``wait_for_headset=False``:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Session
     - Entering with ``wait_for_headset=False``
   * - Remote
     - Returns once the Teleop Session Server has answered; neither the server
       nor the robot app needs a headset. Publishing works at once.
   * - Local, ``SessionMode.LIVE``
     - Returns at once, also while no headset has reached the OpenXR runtime
       of the machine. Publishing works at once. The OpenXR runtime must be
       running.
   * - Local, ``SessionMode.REPLAY``; ``LIVE`` with ``oxr_handles``,
       ``oxr_session_factory`` or ``joint_publisher``
     - No effect. A replay session and a session on the caller's own OpenXR
       session never wait for a headset; the robot twin of
       ``joint_publisher`` waits as its ``twin_render`` configuration says.

Three terms are used below. The **OpenXR runtime** is the program that serves
headsets on a machine, here CloudXR. A headset **has reached** the runtime
once it has connected to it since the runtime started; the runtime goes on
reporting it after the headset has disconnected. A headset is **tracked**
while the runtime reports its pose: it is awake, not necessarily worn.

In a remote session entered this way:

- **Until a headset is tracked, the trackers report no data.** The server
  starts the robot app's trackers only once a headset has connected to
  CloudXR; until then it sends no samples, ``step()`` runs the pipeline with
  every tracker input absent, and ``session.last_step_info.remote_tick_age_s``
  is ``None``. Each tracker logs one warning while the session is entered,
  the controller and hand trackers one for each side, such as ``left
  controller data not found``. The robot app's plugin trackers
  also get data only from then on. ``get_status()`` shows ``openxr/headset``
  as ``DISCONNECTED`` (see :ref:`remote-sessions`, **Status**).
- **Move the robot only on a deliberate operator input.** The pipeline runs
  with no tracker data from the first ``step()``, and most retargeters write
  an output all the same: ``Se3AbsRetargeter`` writes the pose at the origin
  and ``GripperRetargeter`` an open gripper. A robot app that sends every
  output to the motors would move the robot there at start. Move only on
  tracker data present in this step together with an operator input such as a
  held grip, or on the ``DefaultTeleopStateManager`` buttons of a
  ``teleop_control_pipeline`` (see :doc:`control_state_machine`). `What a
  Pipeline Gives With No Tracker Data`_ lists every retargeter and shows how
  to test for tracker data. The library adds no guard of its own.
- **When a headset arrives nothing restarts.** The same session gets tracker
  data, and ``step()`` logs ``Teleop Session Server at <endpoint> is ready``
  once the server reports the headset tracked for about a second.
- **A headset that arrives late can end the session.** What entering raises
  by default after the server's answer is raised by a later ``step()``
  instead. When the server cannot start the robot app's trackers it ends the
  connection: ``step()`` raises ``RuntimeError("The Teleop Session Server
  could not start the trackers: ...")``, the publish calls return ``False``
  and the reader's ``connected`` turns ``False``. Robot state does not outlive
  that failure. When the server sends tracker data in a schema the robot app
  cannot read, ``step()`` raises ``RuntimeError("MCAP schema mismatch on
  '<topic>': ...")``. In both cases bring the robot to a safe state in the
  robot app's own loop, then enter a new session.
- **The robot app holds the server's one place.** The server serves one robot
  app at a time, so a robot app left running without a headset keeps every
  other robot app out (``busy``).
- **The timeout.** ``remote_connect_timeout_s`` bounds connecting, the retries
  and the server's answer only; ``TimeoutError`` then names ``server did not
  answer``.

In a local session entered this way (``SessionMode.LIVE`` without
``oxr_handles``, ``oxr_session_factory`` or ``joint_publisher``):

- **The OpenXR runtime must be running.** Entering asks the runtime once for
  an OpenXR session, its connection to the runtime. With a headset at the
  runtime the session starts as any ``LIVE`` session and has tracker data
  from the first step. With the runtime up and no headset, entering returns
  and logs ``No headset has reached the OpenXR runtime; the session starts
  without one. Its trackers report no data and its plugins are not started
  until one does``. With no runtime running, entering raises the OpenXR error
  as it does by default, for example ``RuntimeError("Failed to create OpenXR
  instance: -51")``: it does not wait for the runtime. A robot app that does
  not start CloudXR itself with ``CloudXRLauncher`` finds the runtime only
  from a terminal that has loaded the CloudXR environment (``source
  ~/.cloudxr/run/cloudxr.env``, see
  :ref:`load-cloudxr-environment-variables`); without it entering raises
  ``RuntimeError("Environment variable NV_CXR_RUNTIME_DIR is not set")``.
- **What entering still refuses while no headset is there.** A plugin with
  ``required=True`` that cannot be found raises as it does by
  default (``Required plugin '<name>' ...``). With ``mcap_config``, a folder
  that is missing or cannot be written raises ``RuntimeError``: ``The folder
  of the recording file does not exist or cannot be written: <folder>``. Both
  are raised by entering, so they do not come up in a later ``step()``,
  and nothing of the session stays running.
- **Until a headset has reached the runtime, nothing gives data.** ``step()``
  runs the pipeline with every tracker input absent: head, hands and
  controllers, and also the keyboard and plugin devices such as a gamepad or
  a SpaceMouse. A message channel reports ``DISCONNECTED`` and no messages; a
  message given to ``MessageChannelSink`` waits and is sent once the channel
  is connected. Only the newest ``outbound_queue_capacity`` batches wait (256
  by default; a batch is what the sink gets in one step): older ones are
  dropped with no log line, and those that waited are all sent in one
  ``step()``. A tactile tracker has no forces and no layout.
  No plugin is started and nothing is recorded: the recording
  file of ``mcap_config`` is made when a headset arrives. ``sinks`` run every
  step, but their data, for example haptics, is dropped. Meanwhile a
  background thread named ``isaaccapture-headset-wait`` asks the runtime for
  an OpenXR session once a second, and each attempt logs ``Created OpenXR
  instance``.
- **Move the robot only on a deliberate operator input**, as in a remote
  session (see above): the pipeline runs with no tracker data from the first
  ``step()``.
- **When a headset arrives, one** ``step()`` **returns late.** Nothing
  restarts. The first ``step()`` after a headset has reached the runtime
  starts the trackers, then the plugins, and the recording. It logs ``A
  headset has reached the OpenXR runtime; starting <n> tracker(s) and the
  plugins now. This step() returns late, and every Python thread of this
  program waits until it does`` before, and ``Started the trackers and <n>
  plugin(s) in <t> ms; the session now reads the headset`` after. That one
  call takes the tracker start plus at least 0.1 s for each plugin, and every
  Python thread of the robot app stands still for it, also a thread that
  commands the robot or publishes robot state. ``session.last_step_info``
  does not show the late step: measure how long each ``step()`` takes
  yourself, and make sure the robot is safe when one command comes that much
  later. **Readers get no answer during that step**, because the thread that
  answers them stands still too. A reader's call waits for the step for at
  most its ``timeout_s`` (1 s by default). When the step takes longer,
  ``description()`` and every read return ``None``, also with
  ``allow_stale=True``, until the step has returned; the reader logs
  ``Nothing serves robot state at <path> (no answer within 1 s)`` and reads
  again by itself afterwards. When the step is shorter the call returns
  late, and a reading that is by then older than its series'
  ``stale_after_s`` (0.5 s by default) is not fresh, so the read returns
  ``None``. Check ``description()`` and every read for ``None``. A reader of
  a remote session sees none of this. Tracker data follows once the headset
  is tracked. In pipelined mode (``RetargetingExecutionMode.PIPELINED``) that
  call also runs the pipeline itself, so
  ``last_step_info.ran_synchronously`` is ``True`` for it.
- **Two attributes change in that** ``step()``. While the session waits,
  ``session.oxr_session`` is ``None`` and ``session.deviceio_session`` is a
  session whose trackers report no data. Afterwards they are the OpenXR
  session and the DeviceIO session of any ``LIVE`` session. Read both from
  the session in each step. A ``deviceio_session`` kept from before goes on
  giving no data; only ``send_message()`` on it raises.
- **Status.** While the session waits, ``get_status()`` shows
  ``openxr/runtime`` as ``AVAILABLE`` and ``openxr/headset`` as
  ``DISCONNECTED`` with the reason ``NO_CURRENT_DATA`` and the text ``No
  headset has reached the OpenXR runtime yet; waiting for one``. The
  controllers are ``INITIALIZING`` and, after one second of steps,
  ``DISCONNECTED``. Plugins and their devices are ``INITIALIZING`` until the
  plugins are started.
- **A failure after entering makes every later** ``step()`` **raise.** Three
  things can fail once entering has returned, and the session tries none of
  them again:

  - The runtime stops, or answers with another error, while the session
    waits: ``The OpenXR session could not be opened while waiting for a
    headset: <error>``. The background thread logs the same text as an error
    and ends.
  - A headset arrives and the trackers cannot start: ``A headset reached the
    OpenXR runtime, but the trackers could not be started: <error>``.
  - A headset arrives and a plugin cannot start: ``A headset reached the
    OpenXR runtime, but a plugin could not be started: <error>``. The plugins
    and trackers that step had started are stopped again.

  That ``step()`` and every later one raise a new ``RuntimeError`` with the
  text, and the first error is its ``__cause__``. ``get_status()`` shows the
  providers as ``FAILED`` with the reason ``RUNTIME_UPDATE_FAILED`` and the
  text. **Robot state stays served:** ``publish_robot_state()`` keeps
  returning ``True`` and readers keep reading. The robot app should catch the
  error, bring the robot to a safe state in its own loop, stay in the
  session, stop calling ``step()`` and keep publishing (the rule under the
  table below says the same for both modes). Leaving the session ends its
  robot state, and a new session starts only once the cause is fixed. Until
  then entering raises the cause itself: with a headset at the runtime it
  starts the trackers and the plugins at once.
- **Ctrl-C and leaving.** Leaving the session while it waits ends the
  background thread. If an attempt is in progress, leaving waits for it for
  at most 5 s, then logs ``The thread that waits for a headset is still
  inside the OpenXR runtime after 5 s; it ends, and closes a session it still
  opens, when the runtime answers`` and goes on. Ctrl-C in the ``step()``
  that starts the trackers raises ``KeyboardInterrupt`` once; what that step
  had started is stopped again, and a ``step()`` called afterwards starts the
  trackers anew.
- **One OpenXR session per program.** As in any ``LIVE`` session, the robot
  app must not open an OpenXR session of its own (``oxr.OpenXRSession``)
  beside the one of its ``TeleopSession``, also not while that session
  waits: the background attempt is then expected to fail, and ``step()``
  raises the first text above.

Where a local and a remote session differ with ``wait_for_headset=False``:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Moment
     - Local session (``LIVE``)
     - Remote session
   * - Start
     - The OpenXR runtime must be running when the robot app enters, and the
       robot app must find it: without the CloudXR environment entering
       raises ``Environment variable NV_CXR_RUNTIME_DIR is not set``.
       Entering does not wait for the runtime.
     - The Teleop Session Server may start later: entering waits for it for
       up to ``remote_connect_timeout_s``.
   * - A headset arrives
     - One ``step()`` returns late, with every Python thread of the robot app
       waiting. Readers get no answer during it; a read can return ``None``.
     - No late ``step()``: the server starts the trackers. Readers read
       throughout.
   * - The OpenXR runtime stops while the session waits for a headset
     - ``step()`` raises ``The OpenXR session could not be opened while
       waiting for a headset: <error>``. Robot state stays served: the
       publish calls keep returning ``True``.
     - The Teleop Session Server exits with code 1. ``step()`` raises
       ``RuntimeError`` with the server's text, for example ``The Teleop
       Session Server could not open an OpenXR session: <error>``, the
       publish calls return ``False`` and the reader's ``server_reachable``
       turns ``False``.
   * - ``step()`` raised because the trackers or a plugin could not start
     - Robot state stays served: the publish calls keep returning ``True``.
     - The server has closed the connection: the publish calls return
       ``False`` and the reader's ``connected`` turns ``False``.
   * - ``session.oxr_session`` and ``session.deviceio_session``
     - Change in the ``step()`` that starts the trackers; read them in each
       step.
     - Do not change while the session is entered; ``oxr_session`` is
       ``None``.

**One rule after** ``step()`` **raises serves both modes**, so the robot app
need not test its mode: bring the robot to a safe state in the robot app's
own loop, stop calling ``step()`` and keep publishing; leave the session once
a publish call returns ``False``. In a local session the publish calls keep
returning ``True`` and readers keep reading; in a remote session whose link
is closed they return ``False``. A new session starts once the cause is
fixed.

This behaviour is tested with stand-ins: a scripted server and a scripted
OpenXR session for a remote session; a replaced OpenXR session, trackers and
plugins for a local one. **None of it has been run with CloudXR, a headset or
a real OpenXR runtime.** Not measured: how a Teleop Session Server behaves
over hours with CloudXR up and no headset (it tries to open its OpenXR
session once a second); and for a local session, whether the runtime answers
an attempt with the error the session expects when no headset is there, how
long an attempt takes and whether it pauses the robot app's loop, whether
hours of attempts are clean, what an attempt raises once the runtime has
stopped, how long the late ``step()`` is, and whether the runtime accepts an
OpenXR session that one thread opened and another uses.

.. _robot-state-no-tracker-data:

What a Pipeline Gives With No Tracker Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A retargeter computes in every ``step()``, also while its input is absent, and
most of them write an output then. What the retargeters of
``isaaccapture.retargeters`` write while their input is absent:

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Retargeter
     - Output while its input is absent
   * - ``Se3AbsRetargeter``
     - **A pose:** position (0, 0, 0) with no rotation. Once it has had
       input, the last pose it computed, until a reset puts it back at the
       origin. The pose jumps from the origin to the controller's or hand's
       when data arrives.
   * - ``JointStateRetargeter``
     - **Joint values:** 0.0 for every joint in ``joint`` mode; the pose at
       the origin and gripper 0.0 in ``ee_pose`` mode. Once it has had input,
       the last values.
   * - ``TriHandMotionControllerRetargeter``, ``DexHandRetargeter``,
       ``SharpaHandRetargeter``, ``WujiHandRetargeter``, and the two-hand
       retargeters that combine them
     - **Joint values:** 0.0 for every hand joint, each time the input is
       absent. ``WujiHandRetargeter`` also writes ``hand_valid`` 0.0.
   * - ``SO101ClutchRetargeter``
     - **A pose:** its configured home pose. Once the clutch has been
       engaged, the last pose it wrote.
   * - ``LocomotionRootCmdRetargeter``
     - **A root command:** zero velocity at the hip height
       ``initial_hip_height``. Once it has had input, zero velocity at the
       last height.
   * - ``FootPedalRootCmdRetargeter``
     - **A root command:** zero velocity at the hip height ``max_squat_pos``,
       each time its input is absent.
   * - ``LocomotionFixedRootCmdRetargeter``
     - **A root command:** it has no input and writes its fixed command in
       every step.
   * - ``GripperRetargeter``
     - **A gripper command:** 1.0 (open), each time its input is absent. A
       closed gripper opens when the data goes away.
   * - ``SO101GripperRetargeter``
     - **A gripper command:** 0.0 (open). Once it has had input, the last
       value.
   * - ``KeyboardGripperRetargeter``, ``SpaceMouseGripperRetargeter``,
       ``GamepadGripperRetargeter``
     - **A gripper command:** 1.0 (open) until toggled; the toggled state is
       kept while the input is absent.
   * - ``Se3RelRetargeter``, ``KeyboardToSe3RelRetargeter``,
       ``GamepadToSe3RelRetargeter``, ``SpaceMouseToSe3RelRetargeter``
     - **No motion:** a change of pose of zero.
   * - ``KeyboardToSe2Retargeter``, ``GamepadToSe2Retargeter``,
       ``SpaceMouseToSe2Retargeter``
     - **No motion:** a base command of zero.
   * - ``ControllerPoseSource``
     - **Nothing:** its output is absent.
   * - ``EePoseJumpGuard``
     - **Nothing** until it has accepted a pose: its output is absent. After
       that, while the teleop state is running, the last pose it accepted.
   * - ``EePoseRateLimiter``, ``JointRateLimiter``
     - **Nothing** until they have had a first value: reading the output
       raises ``ValueError``. After that, the last value they wrote.
   * - ``TensorReorderer``
     - What the retargeters before it wrote, packed into one tensor.

The guard, the rate limiters and ``TensorReorderer`` read another
retargeter's output, not a tracker's: after ``Se3AbsRetargeter`` their input
is the pose at the origin, which is not absent. ``VehicleControlRetargeter``
is not part of a pipeline, and the tactile retargeters drive the operator's
haptic devices, not the robot.

**The teleop state does not hold these outputs back.** With a
``teleop_control_pipeline``, ``DefaultTeleopStateManager`` is ``STOPPED``
while its buttons are absent, sends no reset at the start, and does not take
a button that is already held when the data appears for a press. Only
``SO101ClutchRetargeter``, ``EePoseJumpGuard`` and ``JointStateRetargeter``
in ``ee_pose`` mode with its clutch follow that state; ``Se3AbsRetargeter``,
``GripperRetargeter`` and the others write the same in every state. The robot
app tests the state itself, as :doc:`control_state_machine` shows with
``session.last_context``.

**Testing for tracker data.** The output of a source node, such as
``ControllersSource``, is absent in every step in which its tracker has no
data. Put the source's own output among the pipeline's outputs and test
``is_none`` on it:

.. code-block:: python

   from isaaccapture.retargeters import (
       GripperRetargeter,
       GripperRetargeterConfig,
       Se3AbsRetargeter,
       Se3RetargeterConfig,
   )
   from isaaccapture.retargeting_engine.deviceio_source_nodes import ControllersSource
   from isaaccapture.retargeting_engine.interface import OutputCombiner
   from isaaccapture.retargeting_engine.tensor_types import ControllerInputIndex

   controllers = ControllersSource(name="controllers")
   right = controllers.output(ControllersSource.RIGHT)
   ee_pose = Se3AbsRetargeter(
       Se3RetargeterConfig(input_device=ControllersSource.RIGHT), name="ee_pose"
   ).connect({ControllersSource.RIGHT: right})
   gripper = GripperRetargeter(
       GripperRetargeterConfig(hand_side="right"), name="gripper"
   ).connect({ControllersSource.RIGHT: right})
   pipeline = OutputCombiner(
       {
           "ee_pose": ee_pose.output("ee_pose"),
           "gripper": gripper.output("gripper_command"),
           "operator": right,                   # the source's own output
       }
   )

   def operator_holds_grip(command):
       operator = command["operator"]
       if operator.is_none:                     # no tracker data in this step
           return False
       return float(operator[ControllerInputIndex.SQUEEZE_VALUE]) > 0.5

Without tracker data this pipeline's ``ee_pose`` is the pose at the origin,
its ``gripper`` is 1.0, and only ``operator`` is absent. A pipeline that packs
its outputs into one tensor, as :ref:`isaac-teleop-pipeline-builder` does with
``TensorReorderer``, holds the same numbers in that tensor and shows no
absence: add the source's output beside it.

Read Robot State
----------------

On the machine that runs the headset runtime (the server machine in a remote
session, the robot app's machine in a local one), in any program of the same
user:

.. code-block:: python

   from isaaccapture.robot_state import RobotState

   robot = RobotState.connect(robot="rebot_b601")   # or connect(): whichever robot is served

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

The same calls work with any robot, because the description says which names
exist. For the two-arm robot a reader calls ``robot.joints("left_arm")`` and
``robot.joints("right_arm")``. A reader made with ``RobotState.connect()``,
without ``robot``, that wants every joint of whatever robot is connected never
names one:

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
   * - ``RobotState.connect(socket_path=None, *, robot=None, timeout_s=1.0,
       auto_refresh=True)``
     - A reader. It never raises because nothing serves robot state yet.
       ``robot`` names the robot it expects (see **Expecting one robot**).
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
       server or, in a local session, is in its session.
   * - ``robot.server_reachable``
     - ``True`` while the reader gets answers from the program that serves
       robot state: the Teleop Session Server, or a robot app in a local
       session.
   * - ``robot.refresh()``
     - Fetch the newest readings; ``False`` when that program did not answer.
       Needed only with ``auto_refresh=False``.
   * - ``robot.close()``
     - Close the reader. ``RobotState`` is also a context manager.

Every reading also has ``series``, ``age_s``, ``receive_age_s``, ``fresh``,
``clock_two_way``, ``sequence`` and ``sample_time_ns`` (see `Ages`_).

- **A read can return None.** ``joints()``, ``gripper()``, ``status()`` and
  ``record()`` return ``None`` when the series has no reading yet, when its
  newest reading is not fresh, while no robot app is connected, and while
  nothing serves robot state. Check for ``None`` before using a reading. Pass
  ``allow_stale=True`` to get the newest reading there is even when it is not
  fresh or the robot app has left a remote session; ``fresh`` and ``age_s``
  then say how far to trust it.
- **Expecting one robot.** ``RobotState.connect(robot="rebot_b601")`` reads
  only the robot whose description has that ``robot`` name. While another
  robot is served, ``server_reachable`` is ``True``, ``connected`` is
  ``False``, ``description()`` is ``None``, ``series()`` is empty, and every
  read returns ``None``, also with ``allow_stale=True``; the reader logs
  ``Robot state at <path> is of robot '<served>', not '<expected>'; nothing is
  read`` once. Reads come back by themselves when the expected robot is
  served. A harness that commands a robot passes ``robot``: a Teleop Session
  Server and every robot app in a local session share one default socket
  path, so a reader without it reads whichever robot is served there. The
  guard is the name alone: give a simulated robot's description another
  ``robot`` name than the real robot's. A reader logs ``Robot state at <path>:
  robot '<name>'`` for each description it reads.
- **A joint the robot could not read.** Its ``valid`` is ``False`` and its
  numbers are nan, never a number that could pass for a position
  (``velocity`` and ``effort`` stay ``None`` when the series has none). Check
  ``valid`` before using a joint.
- **Unknown names.** A series or gripper the description does not have raises
  ``KeyError`` naming the ones it has. A call for another shape than the
  series has, for example ``joints("status")``, raises ``ValueError``.
- **A new robot app.** When a robot app connects, the readings and the
  description of the one before it are gone. In a remote session
  ``description()`` and ``series()`` stay readable after a robot app has left,
  until the next one connects; a robot app that leaves a local session takes
  them with it (see `When Something Goes Away`_).
- **Refreshing.** With ``auto_refresh=True`` each call fetches the newest
  readings first. With ``auto_refresh=False``, call ``refresh()`` once per
  cycle; every read until the next ``refresh()`` then sees the same moment. A
  harness that reads several values per cycle should do that: each fetch is
  one exchange with the program that serves robot state, and a robot app in a
  local session answers it on a thread beside its own loop (see **Cost in the
  robot app**).
- **Inside the server's process.** Code that runs in the Teleop Session
  Server's own process gets the same object from
  ``isaaccapture.teleop_session_server.hosted_robot_state()``. A robot app in
  a local session has no such call; ``RobotState.connect()`` also reads from
  inside the robot app's own process.
- **Threads.** One ``RobotState`` can be used from several threads.

A reader program gets robot state over a local socket that belongs to the
isaaccapture package. Its format is private and may change in any release; it
is not an interface. Use the calls of ``isaaccapture.robot_state``.

- **Where the socket is.** The default path is
  ``$XDG_RUNTIME_DIR/isaaccapture/robot_state.sock``, or
  ``/tmp/isaaccapture-<uid>/robot_state.sock`` without ``XDG_RUNTIME_DIR``.
  The program that serves robot state logs ``Serving robot state to reader
  programs at <path>`` when it opens the socket; that line is the place to
  look when a reader finds nothing. The Teleop Session Server opens the socket
  when the first robot app that sends robot state connects, and keeps it until
  the server stops; until that robot app connects, ``server_reachable`` is
  ``False`` although the server runs. A robot app in a local session opens it
  while it enters its session and removes it when it leaves. The directory of
  the socket must belong to the user that runs the serving program and be
  closed to group and others, and the path must be at
  most 103 bytes long.
- **Another path.** Start the server with ``--robot-state-socket PATH`` and
  pass the same path as ``RobotState.connect(socket_path=PATH)``. A robot app
  in a local session has no option for the path: it and its reader follow
  ``XDG_RUNTIME_DIR``. A serving program and a reader that see different
  values of ``XDG_RUNTIME_DIR``, for example a server started as a system
  service and a reader started in a login shell, look in different places;
  give both the same value or the same path then.
- **One program serves a path.** One running program serves a socket path, and
  it keeps the path until it stops serving. A Teleop Session Server and every
  robot app in a local session, of one user on one machine, share the default
  path, so the readers there read one robot. A robot that would be the second
  one at a path is refused when it starts (see `One Robot per Machine`_). An
  empty file ``<socket path>.lock`` marks the program that serves a path; it
  stays after the program has ended and is not to be removed while a program
  serves the path.
- **A reader socket that cannot be opened.** For every reason other than
  another program at the path, the robot is not refused: it teleoperates, and
  its program logs one warning per reason and tries the socket again. The
  reasons are a path longer than 103 bytes, a directory that belongs to
  another user or is open to group or others, a file at the path that is not a
  socket, a socket file left at the path that the program is not permitted to
  connect to, a system without Unix sockets, and a thread that cannot be
  started.

  - A robot app in a local session logs ``Reader programs cannot get robot
    state from this robot app: <reason>. Publish calls return False until the
    socket can be opened; trying again every 2 s``. Entering the session
    succeeds and teleoperation goes on; ``robot_state_active`` is ``False``
    and every publish call returns ``False`` at once. At most once every 2 s
    a publish call asks for another attempt. A thread of the robot app named
    ``isaaccapture-robot-state-socket-retry`` makes the attempt, and the call
    that asked returns ``False``; the publish calls return ``True`` once that
    thread has opened the socket. A robot app that publishes nothing asks for
    no attempt.
  - A Teleop Session Server logs ``Reader programs cannot get robot state from
    this Teleop Session Server: <reason>``, serves its robot app, and tries
    again once a second. Its robot app still sends, with
    ``robot_state_active`` ``True``, and ``hosted_robot_state()`` still reads.

  Another program at the path is such a reason only when a later attempt
  finds it: the socket could not be opened at the start for one of the reasons
  above, or was lost while the program ran, and another program has taken the
  path since. The reason then reads ``another program serves robot state at
  <path>``; the robot app or the server goes on and keeps trying. A robot app
  with a ``robot_description`` that connects to a server in that state is
  refused.
- **Readers at a time.** At most 16 reader programs read one serving program
  at a time; one ``RobotState`` is one of them, however many threads use it.
  The serving program closes a further one and logs one warning; that reader
  sees ``server_reachable`` ``False`` and its reads return ``None`` until
  another reader has closed.
- **Cost in the robot app.** In a local session the robot app gets one more
  Python thread, ``isaaccapture-robot-state-readers``, which answers the
  readers. While the socket cannot be opened it has the thread
  ``isaaccapture-robot-state-socket-retry`` in its place, which waits until a
  publish call asks for an attempt; no publish call opens the socket itself
  or waits for it. Measured on a development
  workstation, a loop of one publish call and 2 ms of Python work kept its
  period (2.0 ms median, 2.1 ms at the 99th percentile, single loops up to
  3.5 ms late) with no reader, with one reader and with 16 readers that each
  asked up to 50 times a second. The readers are the ones that wait, because
  that thread shares the Python interpreter with the robot app's loop. While
  the loop never paused, a fetch took 5 to 10 ms for one reader and 25 to
  100 ms, at times 0.3 s, for each of 16 readers; when the loop paused 3 ms
  in each round, as a loop that sleeps or waits for its robot does, a fetch
  took about 0.2 ms.
- **fork().** A robot app in a local session that calls ``os.fork()`` keeps
  serving in the parent; the child serves nothing and holds nothing of the
  socket. Python 3.12 and later warn about ``fork()`` in a program that has
  threads, which such a robot app does; start other programs with
  ``subprocess`` instead.

Ages
----

``age_s`` is the time since the robot app took the reading, measured on the
reader's machine. In a remote session the library converts the robot
machine's clock to the server machine's with the clock estimate of the remote
session; in a local session both are one machine.

- ``fresh`` is ``True`` while the robot app is connected and ``age_s`` is at
  most the series' ``stale_after_s``. A series with ``stale_after_s = 0`` is
  fresh for as long as its robot app is connected.
- During the first moments of a remote session the clock estimate is one-way:
  ``clock_two_way`` is ``False`` and ``age_s`` is too large by the network
  delay, never too small. In a local session ``clock_two_way`` is ``True``
  from the first reading.
- ``receive_age_s`` is the time since the reading arrived at the server, or in
  a local session since the publish call. It needs no clock estimate and
  leaves out the time the reading spent on the way.
- ``age_s`` is never less than ``receive_age_s``. A reading whose ``time_ns``
  lies ahead of the robot machine's ``time.monotonic_ns()`` therefore ages
  from its arrival and goes stale like any other.
- ``sample_time_ns`` is the time of the reading on the robot machine's clock.
  Do not compare it with a time taken on another machine; use ``age_s``.
- ``sequence`` counts the readings of the series since the robot app entered
  its session: every publish call that returned ``True`` in a local session,
  every reading that reached the server in a remote one, where a slow network
  drops older readings.

When Something Goes Away
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Event
     - What the reader and the robot app see
   * - The robot app leaves a remote session, or its link is lost
     - ``connected`` turns ``False`` and reads return ``None``;
       ``description()`` and reads with ``allow_stale=True`` still answer. The
       robot app's ``step()`` raises on a lost link and its publish calls
       return ``False``; it does not reconnect by itself. Robot state resumes
       when the robot app enters a new session.
   * - The robot app leaves a local session
     - The socket goes with it: ``server_reachable`` turns ``False``, and
       ``description()`` and every read, also with ``allow_stale=True``,
       return ``None``. The reader keeps trying and reads again when a robot
       app enters a session; it need not be restarted.
   * - The headset disconnects and connects again
     - Robot state keeps arriving, in a local and in a remote session. A
       robot app that enters a session while no headset is tracked waits, and
       publishes no readings until one is, unless it sets
       ``wait_for_headset=False`` (see `With Nobody in the Headset`_).
   * - With ``wait_for_headset=False`` in a remote session, a headset arrives
       and the server cannot start the robot app's trackers, or sends data in
       a schema the robot app cannot read
     - The robot app's ``step()`` raises ``RuntimeError``. After a tracker
       failure the server has ended the connection: the publish calls return
       ``False`` and the reader's ``connected`` turns ``False``. Bring the
       robot to a safe state in the robot app's own loop, then enter a new
       session.
   * - With ``wait_for_headset=False`` in a local session, a headset arrives
       and one ``step()`` of the robot app returns late
     - The reader gets no answer for the length of that ``step()``. A call
       returns late; when the step is longer than the reader's ``timeout_s``,
       ``server_reachable`` turns ``False`` and ``description()`` and every
       read, also with ``allow_stale=True``, return ``None`` until the step
       has returned. The reader reads again by itself (see `With Nobody in
       the Headset`_).
   * - With ``wait_for_headset=False`` in a local session, the OpenXR runtime
       stops while the session waits for a headset, or a headset arrives and
       the trackers or a plugin cannot start
     - The robot app's ``step()`` raises ``RuntimeError``, in that call and in
       every later one. Robot state stays served: the publish calls return
       ``True`` and the reader keeps reading. Bring the robot to a safe state
       in the robot app's own loop, stay in the session, stop calling
       ``step()`` and keep publishing (see `With Nobody in the Headset`_).
   * - CloudXR crashes
     - **The Teleop Session Server exits** (see **Failures** under
       "Teleop Session Server" in :doc:`teleop_session`). Robot state stops
       reaching the server machine until the server runs again and the robot
       app has entered a new session, which again needs CloudXR, and by
       default a tracked headset. Meanwhile the reader's ``server_reachable``
       is ``False`` and its reads return ``None``. Whether the server should
       exit in this case is undecided; run the server under a supervisor that
       restarts it.
   * - The Teleop Session Server is stopped or restarted
     - As above: ``server_reachable`` is ``False`` and every read, also with
       ``allow_stale=True``, returns ``None`` until the server answers again.
       The reader keeps trying; it need not be restarted.
   * - The reader starts before the program that serves robot state, or
       before any robot app has connected to the server
     - ``RobotState.connect()`` succeeds, ``server_reachable`` is ``False``,
       reads return ``None``.
   * - A robot starts while another program already serves the socket path
     - The robot that starts second is refused: entering its session raises
       ``RuntimeError`` and it does not teleoperate. The readers there go on
       reading the program that serves the path (see `One Robot per
       Machine`_).
   * - The reader socket cannot be opened for another reason
     - The robot teleoperates. Its program logs one warning and tries again;
       until then its robot state reaches no reader program, and in a local
       session the publish calls return ``False`` (see **A reader socket that
       cannot be opened** under `Read Robot State`_).
   * - More than 16 reader programs read one serving program
     - Each one above 16 sees ``server_reachable`` ``False`` and its reads
       return ``None``; the serving program logs one warning. It reads as soon
       as another reader has closed; share one ``RobotState`` among the
       threads of a program.
   * - The reader and the program that serves robot state run different Isaac
       Capture versions
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
- On the machine that serves robot state, programs of the user that runs the
  serving program can read robot state; other users cannot. The local socket
  is closed to other users and is not reachable from another machine. A reader
  cannot change robot state through it.
- A local session has no network side: its robot state does not leave the
  robot app's machine. Any program of the same user can serve the default
  socket path first, and a reader then reads that program's robot; a reader
  that passes ``robot`` refuses another robot name, not a program that uses
  the same name. The robot app that starts after such a program is refused
  (see `One Robot per Machine`_), so it does not run unnoticed without
  readers.

Run the Teleop Session Server on a trusted network or a VPN, and open its
ports only toward the robot machine.

Example
-------

:code-dir:`examples/robot_state` runs all three parts on one machine, without
a robot and without a headset: a stand-in robot publishes every series of a
description through a remote session entered with ``wait_for_headset=False``,
and a stand-in harness prints what it reads.

.. code-block:: bash

   uv pip install -e ./examples/robot_state
   python -m isaaccapture_examples.robot_state
   python -m isaaccapture_examples.robot_state --description two_arm_robot

Its server listens on a loopback port the system picks and keeps its reader
socket in a temporary folder (``--robot-state-socket``), so the example runs
beside a Teleop Session Server or a robot app on the same machine: it
disturbs neither, and its stand-in robot is not refused as a second robot.

The stand-in robot started by itself without ``--endpoint`` runs a local
session instead; that needs the OpenXR runtime on the machine and no headset
(see the example's README).

Not Included
------------

Robot state is what the robot reports. These stay in the application: an
agent's commands and their answers, who controls the robot, stop, the handover
between an agent and a person, and camera pictures.

Not available yet: starting a ``LIVE`` session before the OpenXR runtime runs;
a second try after the OpenXR session, the trackers or a plugin could not be
started in a ``LIVE`` session; robot state that outlives a tracker failure on
the server;
a reader on the robot machine in a remote session; robot state from the server
to the robot app; a history of readings (the newest of each series is kept); a
gripper's position in space; more than one robot app at a time; more than one
robot on a machine, each answering under its own name to a harness that asks
for a robot by name; recording robot state to MCAP; a C++ call to publish
from a description file.
