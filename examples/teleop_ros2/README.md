<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Teleop ROS 2 Reference (Python)

Reference ROS 2 publisher for Isaac Teleop data.

## Prerequisite: Foot Pedal For `hand_teleop`

`hand_teleop` uses `Generic3AxisPedalSource` + `FootPedalRootCmdRetargeter` for
`xr_teleop/root_twist` and `xr_teleop/root_pose`. Start `foot_pedal_reader`,
`pedal_pusher`, or a compatible pedal plugin so pedal data is available through
the matching collection ID.

The default collection ID is `generic_3axis_pedal`. Override it with
`--ros-args -p pedal_collection_id:=<your_collection_id>` when your pedal
publisher uses a different ID.

## Prerequisite: Hand Retargeting

`hand_teleop` and the tracked-hand variants of `controller_teleop` retarget
OpenXR hand tracking to robot hand joint commands via the `hand_retargeter`
parameter:

- `hand_retargeter:=mode_default` (default): keeps the mode-specific default
  behavior: `controller_teleop` uses TriHand controller retargeting, while
  `hand_teleop` uses DexPilot Sharpa retargeting.
- `hand_retargeter:=trihand`: valid only with `controller_teleop`; retargets
  controller trigger/squeeze input to TriHand finger joints.
- `hand_retargeter:=dexpilot`: uses `DexHandRetargeter` with DexPilot configs from
  `examples/teleop_ros2/configs/`. It requires `isaaccapture[retargeters]` and
  official standalone Sharpa Wave URDFs at:
  `examples/teleop_ros2/assets/urdf/sharpa_standalone/left_sharpa_wave.urdf`
  and
  `examples/teleop_ros2/assets/urdf/sharpa_standalone/right_sharpa_wave.urdf`.
  Set `config_asset_root` to use a different directory containing `configs/`
  and `assets/`; the empty default uses the installed or source tree root.
- `hand_retargeter:=pink_ik`: uses `SharpaHandRetargeter`. It requires the
  `isaaccapture[grounding]` runtime dependencies and the bundled
  `robotic_grounding` package data that provides the Sharpa MJCF assets.
- `hand_retargeter:=wuji`: uses `WujiHandRetargeter` and requires
  `isaaccapture[wuji]`. `wuji_hand_model` selects `wuji_hand` or `wuji_hand_2`
  for both sides and defaults to `wuji_hand_2`. Each side publishes 20
  firmware-order joints named `left_thumb_j0` through
  `left_pinky_j3` and `right_thumb_j0` through `right_pinky_j3`.

In `controller_teleop`, `hand_retargeter` selects the finger-joint mapping.
With `hand_tracking_provider:=native`, EE poses and wrist TFs come from XR
controller aim poses. With `hand_tracking_provider:=manus`, they come from
controller aim poses with the MANUS controller-to-hand calibration. With
`hand_tracking_provider:=wuji`, they come from the provider-injected OpenXR
wrists. Controllers continue to provide locomotion and `controller_data`.

The Docker build fetches the pinned official Sharpa Wave URDFs and installs them
at `/opt/isaaccapture/install/examples/teleop_ros2/assets/urdf/sharpa_standalone/`.
Source-tree users can populate the same local asset directory from the repo root:

```bash
python3 examples/teleop_ros2/scripts/fetch_sharpa_wave_urdfs.py
```

Robot assets are never downloaded by the node at runtime.

### OpenXR hand input sources

The DexPilot, Pink IK, and Wuji hand retargeters can use native OpenXR hand
tracking, MANUS gloves, or Wuji gloves. Set `hand_tracking_provider` to
`native` (default), `manus`, or `wuji`. Use only one provider at a time.

When `mcap_replay_path` is unset, the ROS node starts the selected MANUS or Wuji
plugin by default. For manual launch, set
`use_external_hand_tracking_plugin:=true`; the plugin and node must use the same
CloudXR runtime path and network namespace, and the glove must be reachable from
the plugin environment. A non-native provider requires `hand_teleop`, or
`controller_teleop` with `dexpilot`, `pink_ik`, or `wuji`.

#### MANUS glove input

Follow the [MANUS device guide](../../docs/source/device/manus.rst) to install
host USB permissions, the MANUS SDK, and the MANUS plugin.

After the ROS node starts, run the following from the IsaacTeleop repository
root in the environment where the MANUS plugin was built:

```bash
source "$HOME/.cloudxr/run/cloudxr.env"
./install/plugins/manus/manus_hand_plugin --datasets=human
```

#### Wuji glove input

Build the Wuji glove plugin from the repository root:

```bash
./src/plugins/wuji_glove/install.sh
```

After the ROS node starts, attach the plugin to the same CloudXR runtime:

```bash
source "$HOME/.cloudxr/run/cloudxr.env"
./install/plugins/wuji_glove/wuji_glove_plugin
```

The Wuji glove plugin defaults to automatic wrist-source selection: optical
hand tracking is preferred, with the mounted controller as fallback. Set
`WUJI_GLOVE_WRIST_SOURCE` to `hand_tracking` or `controller` to force one
source. Override mount calibration when needed:

```bash
export WUJI_GLOVE_AIM_TO_WRIST_LEFT="px,py,pz,qx,qy,qz,qw"
export WUJI_GLOVE_AIM_TO_WRIST_RIGHT="px,py,pz,qx,qy,qz,qw"
```

## Published Topics

- `xr_teleop/hand` (`teleop_ros2_interfaces/msg/NamedPoseArray`)
  - Named OpenXR hand joint poses; always contains 25 `left_` entries followed by 25 `right_` entries, including `WRIST` and omitting `PALM`; missing or invalid joints use zero poses with `is_valid=false`
- `xr_teleop/ee_poses` (`teleop_ros2_interfaces/msg/NamedPoseArray`)
  - Named hand/controller EE poses; always contains `left` and `right` entries, with zero poses and `is_valid=false` for invalid sides
- `xr_teleop/root_twist` (`geometry_msgs/TwistStamped`)
- `xr_teleop/root_pose` (`geometry_msgs/PoseStamped`)
- `xr_teleop/head_pose` (`geometry_msgs/PoseStamped`)
  - Head pose
- `xr_teleop/controller_data` (`std_msgs/ByteMultiArray`, msgpack-encoded dictionary)
- `xr_teleop/finger_joints` (`sensor_msgs/JointState`)
  - Retargeted finger joint angles for the robot; contains joint names and position arrays corresponding to the robot finger joints (TriHand in default `controller_teleop`, or the selected Sharpa/Wuji retargeter in tracked-hand modes)
- `/tf` (`tf2_msgs/TFMessage`)
  - `world_frame` → `right_wrist_frame`: Right wrist transform (published in `controller_teleop` and `hand_teleop` modes)
  - `world_frame` → `left_wrist_frame`: Left wrist transform (published in `controller_teleop` and `hand_teleop` modes)
  - `world_frame` → `head_frame`: Head transform (published in `controller_teleop` and `hand_teleop` modes)
  - With `ee_poses_frame:=head` the wrist transforms are parented to `head_frame` instead of `world_frame`

## Run in Docker

### Build the container

From the repo root.

**Jazzy (default):**
```bash
docker build -f examples/teleop_ros2/Dockerfile -t teleop_ros2_ref .
```

ROS 2 Humble is no longer supported: it ships Python 3.10, which is below the
`isaaccapture` wheel's minimum of 3.11. Another distro needs `ROS_DISTRO` and
`PYTHON_VERSION` overridden together, and its interpreter must be 3.11 or newer:

```bash
docker build -f examples/teleop_ros2/Dockerfile --build-arg ROS_DISTRO=<distro> --build-arg PYTHON_VERSION=<py> -t teleop_ros2_ref:<distro> .
```

You can tag by distro (e.g. `teleop_ros2_ref:jazzy`) to build and run several side by side.

The default image includes the Wuji glove plugin on amd64 and arm64. To include
the MANUS plugin in an amd64 image, review the
[MANUS Software License Agreement](https://www.manus-meta.com/software-license-agreement),
confirm that your license permits this use, and explicitly accept it:

```bash
export MANUS_ACCEPT_EULA=YES
docker build --build-arg MANUS_ACCEPT_EULA \
  -f examples/teleop_ros2/Dockerfile -t teleop_ros2_ref:manus .
```

MANUS is skipped on non-amd64 builds.

Incremental rebuilds use Docker BuildKit cache. Ensure BuildKit is enabled (default in Docker 23+), or run with `DOCKER_BUILDKIT=1 docker build ...`.

### Run the container

Use host networking (recommended for ROS 2 DDS):
```bash
docker run --rm --gpus all --net=host --ipc=host \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e ROS_LOCALHOST_ONLY=1 \
  -v $HOME/.cloudxr:/root/.cloudxr \
  --name teleop_ros2_ref \
  teleop_ros2_ref --ros-args -p cloudxr_accept_eula:=true
```

Select the hand provider independently from the robot-hand retargeter. The node
starts the selected plugin automatically:

- Wuji gloves driving Sharpa hands:
  `-p hand_tracking_provider:=wuji -p hand_retargeter:=dexpilot`
- MANUS gloves driving Wuji Hand 2:
  `-p hand_tracking_provider:=manus -p hand_retargeter:=wuji -p wuji_hand_model:=wuji_hand_2`

Wuji gloves must be reachable through the container's host network. Pass Wuji
configuration overrides into the container with `docker run -e <name>=<value>`.
MANUS also requires the
[host udev setup](../../docs/source/device/manus.rst) and USB access; add
`-v /dev/bus/usb:/dev/bus/usb` to the MANUS `docker run` command.

### Overriding parameters and remapping topics

It's possible to set ROS 2 parameters and remap topics from the command line when running the container. Append `--ros-args -p param_name:=value` to set parameters, or `--ros-args -r old_topic:=new_topic` to remap topics after the image name:

```bash
docker run --rm --gpus all --net=host --ipc=host \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e ROS_LOCALHOST_ONLY=1 \
  -v $HOME/.cloudxr:/root/.cloudxr \
  --name teleop_ros2_ref \
  teleop_ros2_ref --ros-args -p cloudxr_accept_eula:=true \
  -p world_frame:=odom -p rate_hz:=30.0 \
  -r xr_teleop/hand:=my_robot/hand \
  -r xr_teleop/ee_poses:=my_robot/ee_poses
```

Available parameters: `rate_hz`, `mode`, `hand_retargeter`, `hand_tracking_provider`, `use_external_hand_tracking_plugin`, `wuji_hand_model`, `config_asset_root`, `cloudxr_install_dir`, `cloudxr_env_config`, `cloudxr_client_route`, `cloudxr_accept_eula`, `cloudxr_setup_oob`, `cloudxr_usb_local`, `pedal_collection_id`, `world_frame`, `right_wrist_frame`, `left_wrist_frame`, `head_frame`, `ee_poses_frame`, `left_finger_joint_names`, `right_finger_joint_names`. Use `ros2 param list /teleop_ros2_node` and `ros2 param describe /teleop_ros2_node <param>` (with the node running) for the full set.

By default, `left_finger_joint_names` and `right_finger_joint_names` use the selected mode's retargeter joint names. They can be overridden to publish robot-specific names on `xr_teleop/finger_joints`, but each override must provide the same number of names as the joints emitted by that mode's retargeter.

Available topics for remapping: `xr_teleop/hand`, `xr_teleop/ee_poses`, `xr_teleop/root_twist`, `xr_teleop/root_pose`, `xr_teleop/head_pose`, `xr_teleop/controller_data`, `xr_teleop/full_body`, `xr_teleop/finger_joints`. Active remaps can be inspected with `ros2 node info /teleop_ros2_node`.

### Mode

The `mode` parameter selects the teleoperation scenario and which topics are published:

| Mode | Topics published |
|------|------------------|
| `controller_teleop` (default) | `ee_poses`, `root_twist`, `root_pose`, `head_pose`, `finger_joints`, `controller_data`, and `tf`; TriHand uses controller-derived finger joints and EE poses. Explicit Sharpa or Wuji retargeters add `hand`; `native` uses controller aim for EE poses, `manus` applies the MANUS calibration to controller aim, and `wuji` uses provider-injected wrist poses |
| `hand_teleop` | `ee_poses` (from hand tracking wrists), `hand` (named left and right joint poses), `finger_joints` (finger joints in joint space), `root_twist`, `root_pose`, `head_pose`, `tf` (from hand tracking wrists and head pose); locomotion comes from the configured foot pedal collection |
| `controller_raw` | `controller_data` only |
| `full_body` | `full_body` and `controller_data` |

Example: `--ros-args -p mode:=controller_raw`

### EE Poses Frame

The `ee_poses_frame` parameter selects the reference frame of `xr_teleop/ee_poses`:

| Value | Behavior |
|-------|----------|
| `world` (default) | Absolute poses in `world_frame`; wrist TFs parented to `world_frame` |
| `head` | Poses relative to the headset, stamped `head_frame`; wrist TFs parented to `head_frame` |

Head mode transforms both position and orientation using
`T_head_ee = inverse(T_world_head) @ T_world_ee`. The head and EE inputs come from
the same session step, after the configured world-frame coordinate conversion
and any controller-to-hand calibration. Head-relative poses stay fixed when the
head and EEs undergo the same rigid motion; moving only the head changes them.

If the head pose is absent or invalid, the EE message keeps the configured
`head_frame`, both entries are marked `is_valid: false`, and no wrist TFs are
broadcast until a valid head pose returns. There is no fallback to `world_frame`.
Consumers must honor `is_valid`; invalid entries contain placeholder poses, not
robot targets. The head pose and head TF remain relative to `world_frame`, and
other topics are unchanged.

```bash
docker run --rm --gpus all --net=host --ipc=host \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e ROS_LOCALHOST_ONLY=1 \
  -v $HOME/.cloudxr:/root/.cloudxr \
  --name teleop_ros2_ref \
  teleop_ros2_ref --ros-args -p cloudxr_accept_eula:=true \
  -p ee_poses_frame:=head
```

### OOB Teleop Control

For live sessions, the node can enable the out-of-band (OOB) teleop control hub
that the in-process `CloudXRLauncher` provides. The hub shares the CloudXR proxy
TLS port (default 48322) and lets you read streaming metrics, inspect connected
headsets, and push config from outside the headset:

```bash
docker run --rm --gpus all --net=host --ipc=host \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e ROS_LOCALHOST_ONLY=1 \
  -v $HOME/.cloudxr:/root/.cloudxr \
  --name teleop_ros2_ref \
  teleop_ros2_ref --ros-args -p cloudxr_accept_eula:=true \
  -p cloudxr_setup_oob:=true
```

`cloudxr_usb_local:=true` additionally routes teleop signalling, the web client,
and WebRTC media over the USB cable via `adb reverse`. It requires
`cloudxr_setup_oob:=true` (the node raises a parameter error otherwise) and the
`adb` and `coturn` host tools. Both parameters are ignored in MCAP replay mode,
since no CloudXR runtime is launched.

See the [Out-of-Band Teleop Control](../../docs/source/references/oob_teleop_control.rst)
reference for the hub HTTP/WebSocket API, ADB automation, and USB-local details.

### MCAP Replay

Set `mcap_replay_path` to run the same ROS 2 publisher from recorded DeviceIO
tracker data instead of live OpenXR/DeviceIO inputs:

```bash
docker run --rm --net=host --ipc=host \
  -v /tmp:/tmp \
  --name teleop_ros2_ref \
  teleop_ros2_ref --ros-args -p mode:=controller_raw \
  -p mcap_replay_path:=/tmp/teleop_ros2_input.mcap
```

The installed integration test utility
`examples/teleop_ros2/cpp/integration_tests/teleop_ros2_mcap_generator` creates a
deterministic fixture with controller, hand, pedal, and full-body samples for CI
coverage.

## Echo Topics

```bash
docker exec -it teleop_ros2_ref /bin/bash

ros2 topic echo /xr_teleop/hand teleop_ros2_interfaces/msg/NamedPoseArray
ros2 topic echo /xr_teleop/ee_poses teleop_ros2_interfaces/msg/NamedPoseArray
ros2 topic echo /xr_teleop/root_twist geometry_msgs/msg/TwistStamped
ros2 topic echo /xr_teleop/root_pose geometry_msgs/msg/PoseStamped
ros2 topic echo /xr_teleop/head_pose geometry_msgs/msg/PoseStamped
ros2 topic echo /xr_teleop/controller_data std_msgs/msg/ByteMultiArray
ros2 topic echo /xr_teleop/full_body std_msgs/msg/ByteMultiArray
ros2 topic echo /xr_teleop/finger_joints sensor_msgs/msg/JointState
ros2 topic echo /tf tf2_msgs/msg/TFMessage
```

## Controller Data Decoding

```python
import msgpack
import msgpack_numpy as mnp
from std_msgs.msg import ByteMultiArray

def controller_callback(msg: ByteMultiArray):
    data = msgpack.unpackb(
        bytes([ab for a in msg.data for ab in a]),
        object_hook=mnp.decode,
    )
    print(data.keys())
```
