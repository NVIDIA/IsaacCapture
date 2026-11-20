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
  TeleopSession (REMOTE)             (test mode: no CloudXR,        RobotState.connect()
    publish_robot_state()  ══════►    no headset)           ────►    description(), joints(),
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

The server listens on a loopback port the system picks and keeps its reader
socket in a temporary folder, so the example disturbs nothing else that runs
on the machine. `--duration-s 10` stops it by itself.

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
)
with TeleopSession(config) as session:
    while True:
        session.step()
        session.publish_robot_state("arm", positions)        # seven numbers
        session.publish_robot_state("status", [True, ""])
```

A reader on the server machine asks by name (`stand_in_harness.py`):

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

The stand-in robot starts to publish once its session is entered, which needs
a headset that the server tracks. The harness runs as the same user as the
server, and reads nothing until a robot app that sends robot state has
connected.

The example does not drive a robot: commands to the robot, who controls it and
stopping it stay in the application.

## Test

```bash
ctest --test-dir build -R robot_state_example_
```

## See also

- User page: `docs/source/teleop/robot_state.rst`
- Design: `docs/design/remote_teleop_session.md`
