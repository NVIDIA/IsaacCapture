<!--
SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# j-mex AgileMaster full body

An example in two parts, both reading one [j-mex AgileMaster](https://jmex.com.tw/agilemaster/)
motion capture channel through the [`jmex` plugin](../../src/plugins/jmex/README.md): a teleop task
that drives a G1 in Isaac Lab from the operator's body pose, and a script that records the same body
pose to a standard Isaac Teleop full-body MCAP.

## Prerequisites

1. **The plugin, installed.** `cmake --install build` — a plugin that has only been built is not
   discoverable. The [plugin README](../../src/plugins/jmex/README.md) lists the SDK it needs and
   the build prerequisites.
2. **MOXI Player**, with the **standard human skeleton** loaded, on a known channel. The plugin's
   default is channel 1; Player's file replay is fixed on 255.
3. **A CloudXR runtime.** It is required even though no headset is.
4. **Isaac Lab 3.0**, for the teleop half.
5. **This repository's `isaaccapture`, in Isaac Lab's Python**, also for the teleop half: no
   released package carries the `body.jmex` vendor yet. Configure the build with
   `-DISAAC_TELEOP_PYTHON_VERSION=3.12`, the Python Isaac Sim 6 runs, and `cmake --install` leaves
   the wheel in `install/wheels`. Install that wheel by its path:

   ```bash
   cd <IsaacLab>
   WHEEL=$(ls -t <IsaacTeleop>/install/wheels/isaaccapture-*-cp312-*.whl | head -1)
   ./isaaclab.sh -p -m pip install "${WHEEL}[cloudxr]" \
     --find-links=<IsaacTeleop>/install/wheels
   ```

   A rebuilt wheel keeps its version, and pip keeps the one already installed. After a rebuild,
   replace it:

   ```bash
   ./isaaclab.sh -p -m pip install --force-reinstall --no-deps "${WHEEL}"
   ```

   pip reports a conflict with the `isaacteleop` version Isaac Lab pins. That is expected: Isaac Lab
   works with this package. `./isaaclab.sh --install` puts the pinned package back, so repeat this
   step after it.
6. **This example, in Isaac Lab's Python**, for the teleop half as well. Install it editable — the
   task finds `jmex_plugin` under this checkout's `install/plugins`:

   ```bash
   cd <IsaacLab>
   ./isaaclab.sh -p -m pip install -e <IsaacTeleop>/examples/jmex
   ```

## Teleop: driving the G1

`jmex_g1_tasks.py` registers `Isaac-PickPlace-Locomanipulation-G1-Jmex-Abs-v0`: Isaac Lab's
locomanipulation G1 task, with its action taken from the operator instead of from XR controllers.
Nothing in Isaac Lab is modified.

This is **upper-body teleop**: the operator's arms drive the G1's arms, and their torso — leaning,
bending sideways, twisting — drives its waist. Turning on the spot does not move the robot, and the
legs belong to the locomotion policy. The scene is the G1 on its own.

Running it takes four steps:

1. **Start the runtime** in a terminal of its own, and leave it running; `Ctrl-C` there stops it
   once Isaac Lab has exited. It runs on Isaac Lab's Python, which has this repository's
   `isaaccapture` from [Prerequisites](#prerequisites) step 5:

   ```bash
   cd <IsaacLab>
   ./isaaclab.sh -p -m isaaccapture.cloudxr.service run
   ```

2. **Load the environment it wrote**, in the terminal you start Isaac Lab from:

   ```bash
   source ~/.cloudxr/run/cloudxr.env
   ```

3. **Start the task** in that terminal, with `--cloudxr_env none`:

   ```bash
   cd <IsaacLab>
   ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
     --task Isaac-PickPlace-Locomanipulation-G1-Jmex-Abs-v0 \
     --visualizer kit \
     --xr \
     --cloudxr_env none \
     --external_callback isaaccapture_examples.jmex.jmex_g1_tasks.register_tasks
   ```

4. **Press Start XR** in Kit's XR panel. Nothing moves until then, and Isaac Lab logs
   `OpenXR handles not yet available (waiting for XR session)`. With `--headless` the XR session
   starts by itself.

Things to know:

- **Keep `--cloudxr_env none`.** Without it, Isaac Lab launches a runtime of its own with the
  `auto-webrtc` profile, which waits for a headset: no data arrives, and that runtime is still
  running after Isaac Lab exits.
- **The callback path uses a dot, not a colon**, and resolves only once this example is installed in
  Isaac Lab's Python ([Prerequisites](#prerequisites) step 6).
- **The task launches `jmex_plugin` itself.** `JMEX_PLUGIN_AUTO_LAUNCH=0` leaves that to you, and
  `JMEX_CHANNEL` and `JMEX_COLLECTION_ID` change the channel and the collection.
- **Until Player pairs**, the G1 holds its own rest pose, and the console says so once and then
  every 300 frames.
- **If Player disconnects mid-run**, the G1 holds the last pose it received: the collection keeps its
  last frame. The plugin logs `MOXI Player disconnected; waiting for it to come back`, and the G1
  follows again once Player pairs.
- **Load the standard human skeleton in Player.** Only a frame that carries all 24 slots moves the
  robot. A robot URDF loaded by mistake fills only some of them; when slots stay missing for 30
  frames in a row — a few seconds when driven live — the console names them, again every 300 frames.

### What is connected, and what is not

| | Source |
| --- | --- |
| Wrist position | The operator's arm, rebuilt with the G1's arm lengths from the G1's shoulder |
| Wrist orientation | The operator's hand |
| Elbow position | The operator's elbow, on the same rebuilt arm |
| Shoulder position | The G1's shoulder, turned about its waist by the operator's torso |
| Fingers | Zero |
| Legs, base motion | A fixed standing command; the locomotion policy keeps the G1 standing |

Ten of the 24 slots reach the action: the shoulders, elbows and wrists for the arms, the two hips
for the operator's heading, and `SPINE1` and `SPINE3` for the torso. The rest are drawn in the
overlay and drive nothing.

### Reading the overlay

The task draws two figures over the robot. Once XR is running, Kit's main viewport shows the XR
camera's view; open **Window → Viewport 2** to watch them:

| Color | What it is | Should the robot line up with it? |
| --- | --- | --- |
| Blue | The body pose as it arrived, turned to face the way the robot does, with its pelvis on the robot's | No. It has the operator's proportions |
| Orange | Each arm as the IK is asked to place it, with the shoulder, elbow and wrist drawn as large dots | Yes. The robot's shoulders, elbows and wrists should sit on the large dots |

Before Player pairs, orange is a dot on each of the robot's resting wrists, elbows and shoulders,
and there is no blue figure. Without Kit's debug draw extension nothing is drawn, and the console
says so once.

## Recording a body pose

Install the example once, then start the runtime in its own terminal and leave it running:

```bash
uv pip install -e ./examples/jmex
python -m isaaccapture.cloudxr.service run
```

Then, in the terminal you record from, load the environment it wrote and let the script attach to
what is already up:

```bash
source ~/.cloudxr/run/cloudxr.env

python -m isaaccapture_examples.jmex.record_jmex_full_body \
  10 examples/jmex/recordings/jmex_full_body.mcap \
  --no-launch-cloudxr-runtime
```

This records ten seconds to that path on the plugin's default channel. Add `--channel=255` to
record from Player's file replay, and `--collection-id=` if this machine runs more than one of
these.

The script starts `jmex_plugin` itself and stops it on exit. `--no-plugin` leaves starting it to
you, which is also how to give the plugin any of its [other
flags](../../src/plugins/jmex/README.md#running). Start the runtime first, as above: if the script
starts a runtime itself and that runtime is still coming up, the script can wait at
`OpenXR HMD form factor is unavailable; waiting for a system...` with no timeout, and `Ctrl-C` does
not stop it.

Each progress line reports how many of the 24 slots the frame carried:

```text
[record] t= 1.03s  frame=60  joints=24/24
```

`00/24` on every line means no sample has arrived: Player has not paired, or the channel is wrong.

## Replaying

There is no j-mex replay script. The recording uses the standard `core.FullBodyPoseRecord` schema on
the standard `full_body` channel, so the shared script plays it back:

```bash
uv pip install -e ./examples/mcap_record_replay
python -m isaaccapture_examples.mcap_record_replay.replay_full_body \
  examples/jmex/recordings/jmex_full_body.mcap
```

## What is in the recording

| Channel | Schema | Contents |
| --- | --- | --- |
| `full_body/full_body` | `core.FullBodyPoseRecord` | 24 body-pose slots, OpenXR axes, meters |
| `controllers/left_controller` | `core.ControllerSnapshotRecord` | No tracked pose on a mocap-only session |
| `controllers/right_controller` | `core.ControllerSnapshotRecord` | No tracked pose on a mocap-only session |

The two controller channels are recorded because the shared replay pipeline expects them. Near the
end of each pass, replay logs `left controller data not found` and `right controller data not found`
once each; neither is a fault.

## Known limitations

- **Nothing can be picked up.** The body pose carries no finger joints, so the fingers stay at zero,
  and the table and its object are moved away.
- **The feet shuffle.** The pelvis is pinned, and the locomotion policy keeps counterbalancing
  whatever the arms do.
- **An arm thrown out to the shoulder's joint limit can stay there.** Isaac Lab's "Press 'R'" does
  not reset it; restart Isaac Lab. Short visits to the limit that clear on their own are more
  common.
- **Stay inside the waist's range**: ±15° of lean and side bend, ±30° of twist, as logged at
  startup. Past them the waist stops at the limit and the G1 stops following, arms included.
- **The G1 runs behind the operator.** With XR on, the simulation runs well below real time.
- **Stand upright before starting.** For the first couple of seconds the robot catches up to
  wherever the operator already is.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Isaac Lab fails to import `pinocchio`, or its `pip install` lands in another environment | The shell has a venv active or a ROS environment sourced. Put `env -u VIRTUAL_ENV -u LD_LIBRARY_PATH -u PYTHONPATH` in front of Isaac Lab's commands |

## Verified on

Isaac Lab v3.0.0-beta2.patch1 on Isaac Sim 6.0.1, with the `LocomanipulationG1EnvCfg` base task and
the Pink IK controller that ships with it.

## Running the offline tests

The retargeting tests need only the staged `isaaccapture` package: no Isaac Sim, Isaac Lab, device
or MOXI Player. They carry the ctest label `jmex`, as do the plugin's C++ tests, which need no MOXI
Receiver SDK. A build registers the Python ones only with `BUILD_EXAMPLES` and `BUILD_VIZ` on:

```bash
ctest --test-dir build -L jmex --output-on-failure
```

To run them under `pytest` from the repository root instead, put the staged package on `PYTHONPATH`
and use the Python the build was configured for, 3.11 unless `ISAAC_TELEOP_PYTHON_VERSION` set
another:

```bash
PYTHONPATH=build/python_package/Release \
  uv run --project tests/python/examples/jmex --python 3.11 --extra dev \
  pytest tests/python/examples/jmex/ -v
```
