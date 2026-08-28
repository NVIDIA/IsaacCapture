<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SO-101 Leader Arm plugin

Streams the SO-101 (5-DOF arm + gripper) leader joint angles as a `JointStateOutput` FlatBuffer
over the OpenXR tensor transport, using the generic **joint-space device** path
(`JointStateTracker` / `JointStateSource` / `JointStateRetargeter`).

The SO-101 leader is 6 FEETECH STS3215 bus servos on a half-duplex TTL serial bus (the same
hardware [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) and HuggingFace
LeRobot drive via the FEETECH SCServo SDK). `FeetechBus` (`feetech_bus.{hpp,cpp}`) speaks that
SMS/STS wire protocol directly — no SDK dependency — and implements just what a *leader* needs:
disable torque so the arm can be back-driven by hand, then read `Present_Position` (register 56,
4096 ticks / 360°) for all six servos in a **single SYNC READ** per frame (one bus round-trip, not
six — matching LeRobot's `sync_read`). Ticks are converted to radians with per-joint calibration.

When no serial device is given, the plugin falls back to a **synthetic** trajectory so the
device → tracker → retargeter pipeline runs with no hardware (CI and the headless example).

## Run

```bash
# Synthetic backend (no hardware):
./install/plugins/so101_leader/so101_leader_plugin

# Real SO-101 leader on a serial port (Linux), default collection id "so101_leader":
./install/plugins/so101_leader/so101_leader_plugin /dev/ttyACM0

# ... with a custom collection id and a calibration file:
./install/plugins/so101_leader/so101_leader_plugin /dev/ttyACM0 so101_leader so101_leader.calib
```

Args are positional: `[device_path] [collection_id] [calibration_file]`. The serial backend is
Linux/macOS only (POSIX `termios`); the STS bus runs at 1,000,000 bps by default.

### Generate a calibration file

The `calibrate` subcommand reads the live servos and writes a calibration file (it does **not** need
the OpenXR runtime). It mirrors LeRobot's `lerobot-calibrate`: a homing step then a range-of-motion
sweep.

```bash
# Two interactive steps: (1) hold the arm at mid-range, ENTER; (2) sweep every joint, ENTER:
./install/plugins/so101_leader/so101_leader_plugin calibrate /dev/ttyACM0 so101_leader.calib

# Write a LeRobot-format calibration instead (chosen by the .json extension):
./install/plugins/so101_leader/so101_leader_plugin calibrate /dev/ttyACM0 so101_leader.json

# Omit the output path to just print the measurements (a "dump" for inspection):
./install/plugins/so101_leader/so101_leader_plugin calibrate /dev/ttyACM0
```

It disables torque, then:
1. **Home** — prompts you to hold all joints at the middle of their range and averages a few sync
   reads into each servo's `home_ticks` (the middle pose is also the SO-101 URDF/operating zero).
2. **Range sweep** — while you move every joint through its full range, it tracks per-joint
   `range_min`/`range_max` until you press ENTER.

It writes the file (plain text, or LeRobot JSON if the path ends in `.json`; all `sign` default to
`+1` — flip any inverted joint by hand) and prints the gripper's range endpoints in radians, which
you drop into the retargeter's `gripper_open`/`gripper_close`.

### LeRobot calibration interoperability

The calibration files are interchangeable with [LeRobot](https://github.com/huggingface/lerobot)
(`SO101Leader`). A `.json` path is read/written in LeRobot's format
(`{ "joint": {"id", "drive_mode", "homing_offset", "range_min", "range_max"}, ... }`); anything else
is the plain-text format above.

```bash
# Use an existing LeRobot calibration directly:
./install/plugins/so101_leader/so101_leader_plugin /dev/ttyACM0 so101_leader \
    ~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/<id>.json
```

Mapping: LeRobot's `range_min`/`range_max` → our range (reads are clamped to it), the range midpoint
→ `home_ticks` (LeRobot's zero), `drive_mode` → `sign`. LeRobot stores each joint's `homing_offset`
in the **servo EEPROM** (its runtime normalization doesn't use it), so on connect we read the
servo's live `Homing_Offset` (register 31) and shift home/range by
`homing_offset_file − homing_offset_servo` — reconciling the frames without writing to the servo.
The result reproduces LeRobot's joint angles (in radians rather than degrees).

### Hardware setup (per SO-ARM100 / LeRobot)

- Assemble the leader arm and **remove the gearbox gears** so the joints move freely and only the
  position encoders are used (the plugin disables torque on connect, but the leader is meant to be
  back-driven).
- Give each servo a unique id `1..6` on the bus and set them all to the same baud rate. Use the
  FEETECH tool (`FT_SCServo_Debug_Qt` on Ubuntu) or LeRobot's `lerobot-setup-motors` to do this.
- Make sure your user can access the serial device (e.g. add it to the `dialout` group).

### Calibration file (optional)

The plain-text format (used unless the path ends in `.json`) is whitespace-separated, one joint per
line; `#` starts a comment. Columns:
`name  servo_id  sign(+1/-1)  home_ticks(0..4095)  [range_min range_max]` (the two range columns are
optional). The conversion is
`angle [rad] = sign * (clamp(ticks, range_min, range_max) - home_ticks) * 2π / 4096`.

```
# name           id  sign  home_ticks  range_min  range_max
shoulder_pan      1   1     2048        800        3300
shoulder_lift     2   1     2048        900        3100
elbow_flex        3   1     2048        700        3400
wrist_flex        4   1     2048        800        3300
wrist_roll        5   1     2048        0          4095
gripper           6   1     2048        2000       3000
```

Defaults (no file, or only the first four columns): ids `1..6` in DOF order, `sign +1`,
`home_ticks 2048` (servo center), full range `0..4095` (clamp is a no-op). Set `home_ticks` to each
servo's raw `Present_Position` at the joint's URDF-zero pose, `sign` to `-1` for any joint whose
servo turns opposite the URDF convention (LeRobot's `drive_mode`), and the optional `range_min/max`
to the swept extremes (reads are clamped to them; `range_min < range_max` required or they're
ignored). For **joint-mirror** mode the retargeter's per-joint `offset`/`sign`/`scale` can also
absorb calibration; for **EE (URDF FK)** mode the joint angles must already match the URDF, so set
`home_ticks`/`sign` here. The `calibrate` subcommand fills all of this in for you.

The consumer side creates a `JointStateTracker("so101_leader")` (via
`JointStateSource(name=..., collection_id="so101_leader", joint_names=[...])`) on the same
`collection_id`. See `examples/teleop/python/isaaccapture_examples/teleop/joint_space_device_example.py` for the retargeting
pipeline (joint-mirror and task-space EE modes).

DOF order / names: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`.
