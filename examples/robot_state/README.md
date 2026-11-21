<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Robot state example

Robot state on one machine, without a robot and without a headset. A stand-in
robot publishes joint angles, a gripper, status and faults through a remote
teleop session; a stand-in harness reads them back by name through
`isaaccapture.robot_state` and prints them.

```bash
uv pip install -e ./examples/robot_state
python -m isaaccapture_examples.robot_state
```

That one command starts three programs and stops them again on Ctrl+C:

```text
stand-in robot                     Teleop Session Server          stand-in harness
  TeleopSession (REMOTE)             (--no-headset: no CloudXR,     RobotState.connect()
    publish_robot_state()  ══════►    no OpenXR)            ────►    description(), joints(),
                           remote     keeps the newest reading       gripper(), status(), record()
                           teleop     of each data series
                           session
```

It prints what the harness reads, once a second:

```text
robot rebot_b601
  arm (joints), 0.033 s old
    joint1              2.186 rad  limits -2.800 to 2.800
    joint2              2.229 rad  limits -0.050 to 3.140
    ...
    gripper             0.587  limits 0.000 to 1.000
  status (status), 0.033 s old
    motors_on        True
    fault            'overtemperature'
    warning overtemperature at joint1: stand-in fault: a motor is hot
  gripper gripper: 59 % open
```

The server is started with `--no-headset`: it serves robot apps and their
robot state without a headset, and uses neither CloudXR nor OpenXR, so the
machine needs neither. It listens on a loopback port the system picks and
keeps its reader socket in a temporary folder, so the example disturbs nothing
else that runs on the machine and is not refused beside a robot that runs
there (see "One robot per machine" below). `--duration-s 10` stops it by
itself.

## Two robots, two files

A robot is described by a file, not by code. The example ships two:

| File | Robot |
| --- | --- |
| [`six_joint_arm.toml`](python/isaaccapture_examples/robot_state/descriptions/six_joint_arm.toml) | Six joints and a gripper in one data series, and a status series |
| [`two_arm_robot.toml`](python/isaaccapture_examples/robot_state/descriptions/two_arm_robot.toml) | Two arms that report effort, a waist, a force sensor in each wrist (a shape this robot defines), status with a battery reading |

```bash
python -m isaaccapture_examples.robot_state --description two_arm_robot
python -m isaaccapture_examples.robot_state --description /path/to/my_robot.toml
```

Neither the stand-in robot nor the stand-in harness names a joint: both take
the names from the description.

## The two sides

The robot app names its description and hands over numbers, in the order of
the file (`stand_in_robot.py`):

```python
from isaaccapture.teleop_session_manager import SessionMode, TeleopSession, TeleopSessionConfig

config = TeleopSessionConfig(
    app_name="MyRobot",
    pipeline=pipeline,
    mode=SessionMode.REMOTE,
    remote_endpoint="tcp://<server>:<port>",
    robot_description="six_joint_arm.toml",
    wait_for_headset=False,                                  # publish with nobody in the headset
)
with TeleopSession(config) as session:
    while True:
        session.step()
        session.publish_robot_state("arm", positions)        # seven numbers
        session.publish_robot_state("status", [True, ""])
```

A reader on the machine that serves robot state, here the server machine, asks
by name (`stand_in_harness.py`):

```python
from isaaccapture.robot_state import RobotState

robot = RobotState.connect()
what = robot.description()            # None until a robot app has connected
what.series["arm"].joint_names        # ["joint1", ..., "joint6", "gripper"]
arm = robot.joints("arm")             # None when there is no fresh reading
arm["joint2"].position, arm.age_s
robot.gripper("gripper").opening      # 0 closed to 1 open
robot.status()["fault"]
```

| Module | What it is |
| --- | --- |
| `__main__` | Starts the server, the stand-in robot and the stand-in harness |
| `stand_in_robot` | A robot app: every value is a function of the record number, so a reader can check it |
| `stand_in_harness` | A reader: `describe()` turns a `RobotState` into the lines above |

## With a real Teleop Session Server

The two stand-ins also run by themselves, the robot on the robot machine and
the harness on the server machine:

```bash
python -m isaaccapture_examples.robot_state.stand_in_robot --endpoint tcp://<server>:<port>
python -m isaaccapture_examples.robot_state.stand_in_harness
```

The stand-in robot sets `wait_for_headset=False`, so it publishes as soon as
the server has answered, with no headset. The harness runs as the same user as
the server, and reads nothing until a robot app that sends robot state has
connected.

The example does not drive a robot: commands to the robot, who controls it and
stopping it stay in the application.

## The same robot app in a local session

The stand-in robot always uses a remote session. A robot app of your own runs
the same publish calls without a Teleop Session Server; it names its mode in
one place:

```python
endpoint = None                       # or "tcp://<server>:<port>" for a remote session
config = TeleopSessionConfig(
    app_name="MyRobot",
    pipeline=pipeline,
    mode=SessionMode.REMOTE if endpoint else SessionMode.LIVE,
    remote_endpoint=endpoint,
    robot_description="six_joint_arm.toml",
    wait_for_headset=False,
)
```

In a local session the robot app serves its robot state itself, and the
harness runs unchanged on the robot app's machine, as the same user. A local
session with `wait_for_headset=False` starts only when a headset has reached
the OpenXR runtime of that machine since the runtime started; with no headset
at all, run a Teleop Session Server on that machine and use a remote session
to `127.0.0.1`. The user page lists the limits.

## One robot per machine

Reader programs get robot state at one place on a machine: the reader socket,
a local socket whose default path a Teleop Session Server and every robot app
in a local session share. One robot's state is served there, because one
headset cannot sensibly drive two robots. A robot that starts while another
running program serves that path is refused, whichever of the two is the real
robot:

- A robot app in a local session: entering the session raises `RuntimeError`,
  before anything else of the session starts:

  ```text
  Another program on this machine already serves a robot's state at <path>: a Teleop Session Server or another robot app. One robot per machine is served; stop that program, then start this robot app again.
  ```

- A robot app that connects to a Teleop Session Server, such as the stand-in
  robot started by itself: the server refuses it, and entering the session
  raises `RuntimeError` with the server's text:

  ```text
  Another program on the Teleop Session Server's machine already serves a robot's state at <path>. One robot per machine is served; stop that program, or connect this robot app without a robot description.
  ```

Stop the other program, then start the robot app again; it does not try again
by itself. A Teleop Session Server keeps the path from its first robot app
that sends robot state until the server stops, also between two robot apps,
so stop the server before a robot app with a description runs a local session
on that machine. A robot app without `robot_description` is never refused. A
program that has ended, also a killed one, refuses nobody.

To run two robots on one machine on purpose, give each its own path, as this
example does: start the server with `--robot-state-socket PATH` and give the
harness the same path (`RobotState.connect(socket_path=PATH)`; the stand-in
harness takes `--robot-state-socket PATH`). A robot app in a local session
has no option for the path; start it and its harness with `XDG_RUNTIME_DIR`
set to another existing directory.

The rule holds per path. Two programs that see different values of
`XDG_RUNTIME_DIR`, for example a server started as a system service, which
sees none, and a robot app started in a login shell, have two default paths:
neither robot is refused, and each is read at its own path.

A reader socket that cannot be opened for any other reason, for example a
directory that is open to group or others, refuses nobody: the robot
teleoperates, and its program logs one warning and tries again.

## Test

```bash
ctest --test-dir build -R robot_state_example_
```

## See also

- User page: `docs/source/teleop/robot_state.rst`
- Design: `docs/design/remote_teleop_session.md`
