<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Noitom G1 Teleop

This example registers an external Isaac Lab task based on
`Isaac-PickPlace-Locomanipulation-G1-Abs-v0` and drives its upper-body action
pipeline from Noitom full-body mocap.

The `noitom_mocap` plugin reads Noitom Hybrid Data Server data through
MocapApi, converts each avatar update to the existing IsaacTeleop
`FullBodyPose` layout, and publishes it on tensor identifier `full_body`
inside the `noitom_mocap` OpenXR tensor collection. The Python task consumes
that stream through the generic `FullBodyTracker()` with the `body.noitom`
vendor selected. Plugin launch defaults, including the HDS endpoint, live in
[`plugin.yaml`](../../src/plugins/noitom_mocap/plugin.yaml). Before installing,
set its `--host` and `--port` arguments to the Hybrid Data Server TCP endpoint.

## Build

The Noitom SDK is not vendored in this repository. Build the optional plugin
when you need Noitom support:

```bash
cmake -B build -DBUILD_PLUGIN_NOITOM_MOCAP=ON
cmake --build build --target python_package noitom_mocap_plugin --parallel
cmake --install build
uv pip install --find-links=install/wheels "isaaccapture[cloudxr]"
```

Run `cmake --install build` again whenever `plugin.yaml` changes so the updated
configuration is copied to `install/plugins/noitom_mocap/plugin.yaml`.

By default CMake fetches MocapApi from `https://github.com/pnmocap/MocapApi`.
For offline builds, pass `-DNOITOM_MOCAP_API_ROOT=/path/to/MocapApi`.

## Example 1: Record And Replay

Record uses a Noitom wrapper around `TeleopSession` so the Noitom plugin can be
launched and consumed through the `noitom_mocap` tensor collection without
changing the shared MCAP examples. Replay uses the existing generic full-body
MCAP script because the recording uses the standard `full_body` channel.

![Noitom full-body recording](assets/record.gif)

Record the Noitom full-body stream:

```bash
uv run python examples/noitom/record_noitom_full_body.py \
  10 examples/noitom/recordings/noitom_full_body.mcap
```

The resulting MCAP uses the standard `core.FullBodyPoseRecord` schema and
the standard `full_body` channel, so the generic replay script can play it back:

![Noitom MCAP replay](assets/replay.gif)

```bash
uv pip install -e ./examples/mcap_record_replay
python -m isaaccapture_examples.mcap_record_replay.replay_full_body \
  examples/noitom/recordings/noitom_full_body.mcap
```

## Example 2: Teleop

![Noitom G1 live teleoperation](assets/teleop.gif)

Run Isaac Lab with the external task registration callback. The task launches
`noitom_mocap_plugin` through IsaacTeleop's plugin manager by default.

```bash
cd ~/dependence/IsaacLab3-0
PYTHONPATH=~/IsaacTeleop/examples/noitom:$PYTHONPATH \
  ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-PickPlace-Locomanipulation-G1-Noitom-Abs-v0 \
  --visualizer kit \
  --xr \
  --external_callback noitom_tasks.register_tasks
```

For advanced manual plugin control, start the plugin yourself and disable
auto-launch in the Isaac Lab terminal:

```bash
./install/plugins/noitom_mocap/noitom_mocap_plugin

NOITOM_MOCAP_AUTO_LAUNCH=0 \
PYTHONPATH=~/IsaacTeleop/examples/noitom:$PYTHONPATH \
  ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-PickPlace-Locomanipulation-G1-Noitom-Abs-v0 \
  --visualizer kit \
  --xr \
  --external_callback noitom_tasks.register_tasks
```

If you run a dedicated CloudXR runtime yourself, source its environment before
starting Isaac Lab and pass Isaac Lab's flags for using the existing runtime
instead of auto-launching another one.

## Behavior

Retargeting lives in `noitom_retargeting.py` and is wired by
`noitom_tasks.py`. It maps Noitom shoulder, elbow, and wrist bones into G1 Pink
IK frame targets, with optional elbow and shoulder frame tasks enabled by
default.

Pipeline:

```text
FullBodyPose (`body.noitom` vendor)
  -> torso frame (pelvis, SPINE3, shoulders)
  -> posture-based arm targets (shoulder, elbow, wrist)
  -> Pink IK frame-task action [wrists, elbows, shoulders, hands, locomotion]
```

Calibration:

1. After teleop reset, retargeting clears its neutral reference.
2. Hold a stable upper-body pose; the next valid frame becomes the neutral
   reference.
3. Motion is applied relative to that neutral pose.

Default settings live in `NoitomG1Settings` and
`NoitomRetargetingSettings`. When Kit visualization is enabled, the incoming
Noitom pose is shown as a cyan stick figure anchored to the robot pelvis.
