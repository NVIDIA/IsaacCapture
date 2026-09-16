<!--
SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Preface
First, the required ROS2 topics need to be published through the IO exoskeleton SDK, and then the following process can be initiated.

# Exoskeleton Glove ROS 2 Bridge plugin

Bridges an exoskeleton glove's ROS 2 (Humble) driver into Isaac Teleop's generic
**joint-space device** path (`JointStateTracker` / `JointStateSource` /
`JointStateRetargeter`), using the same `JointStateOutput` FlatBuffer schema as
`so101_leader` / `rebot_devarm_leader`.

## Data path

```
Exoskeleton ROS 2 driver (sensor_msgs/msg/JointState, already retargeted to a
dexterous hand's URDF joint names, radians)
  --/io_teleop/joint_cmd_finger_left-->  ┐
  --/io_teleop/joint_cmd_finger_right--> ┤
                                          ▼
                        io_exo_glove_ros2_plugin (rclcpp node)
                          left  -> JointStateOutputT -> SchemaPusher(collection "exo_glove_left")
                          right -> JointStateOutputT -> SchemaPusher(collection "exo_glove_right")
                                          ▼
                        JointStateTracker x2 -> JointStateSource x2 -> JointStateRetargeter x2
```

The plugin is a **pure transport bridge**: it assumes the upstream ROS 2 driver has already
retargeted the exoskeleton's raw joint angles to the target dexterous hand's URDF joint names
and units (radians), so `JointState.name[i]` / `JointState.position[i]` are copied through
unchanged into `JointStateOutput.joints`. No unit conversion, calibration, or name remapping is
performed here.

## Build

Requires a sourced ROS 2 Humble environment (`rclcpp`, `sensor_msgs` are located via
`find_package()`, same as any ROS 2 C++ node -- no colcon/ament wrapper needed since this plugin
only *consumes* stock message types, no custom `.msg` generation):

```bash
source /opt/ros/humble/setup.bash
cmake -B build -DBUILD_PLUGIN_IO_EXO_GLOVE_ROS2=ON
cmake --build build --target io_exo_glove_ros2_plugin --parallel
```

## Configuration

Topics and collection ids live in `config/io_exo_glove_ros2.yaml`, **not** in the source: edit that
file and restart the plugin, with no rebuild and no recompilation of the header defaults.

```yaml
left_topic: "/io_teleop/joint_cmd_finger_left"
right_topic: "/io_teleop/joint_cmd_finger_right"
left_collection_id: "exo_glove_left"
right_collection_id: "exo_glove_right"
```

Every key is optional. Precedence, lowest to highest:

1. the built-in defaults in `io_exo_glove_ros2_plugin.hpp`
2. the config file
3. the command-line flags `--left-topic=`, `--right-topic=`, `--left-collection-id=`,
   `--right-collection-id=`

The file is looked up in this order, first hit wins:

1. `--config=PATH`
2. `$IO_EXO_GLOVE_ROS2_CONFIG`
3. `<working directory>/config/io_exo_glove_ros2.yaml` -- the plugin launcher `chdir()`s into the
   plugin directory, which is where the file is installed next to `plugin.yaml`

No path is baked in at build time, so a binary run from a directory without a `config/`
subdirectory keeps the built-in defaults; name the file explicitly to pick up the source-tree copy:

```bash
./build/src/plugins/io_exo_glove_ros2/io_exo_glove_ros2_plugin \
    --config=src/plugins/io_exo_glove_ros2/config/io_exo_glove_ros2.yaml
# or:
IO_EXO_GLOVE_ROS2_CONFIG=src/plugins/io_exo_glove_ros2/config/io_exo_glove_ros2.yaml \
    ./build/src/plugins/io_exo_glove_ros2/io_exo_glove_ros2_plugin
```

Naming a file explicitly (`--config=` or the environment variable) makes a missing or malformed file
an error instead of a silent fallback to defaults. Unknown keys are rejected too, so a typo such as
`left_topics:` fails loudly rather than quietly keeping the old value.

## Usage


```bash

python -m isaacteleop.cloudxr

source /opt/ros/humble/setup.bash
source ~/.cloudxr/run/cloudxr.env
./build/src/plugins/io_exo_glove_ros2/io_exo_glove_ros2_plugin

# Show which config file was loaded and the effective settings, then exit (no OpenXR runtime,
# no ROS graph needed -- the fastest way to confirm an edit took effect):
./build/src/plugins/io_exo_glove_ros2/io_exo_glove_ros2_plugin --print-config

# Point at a config file, and/or override a single key for one run:
./build/src/plugins/io_exo_glove_ros2/io_exo_glove_ros2_plugin \
    --config=src/plugins/io_exo_glove_ros2/config/io_exo_glove_ros2.yaml \
    --left-topic=/io_teleop/joint_cmd_finger_left
```

`--help` lists all flags. Everything from `--ros-args` onwards is forwarded to ROS 2 untouched
(remappings, parameter files, logging), apart from the launcher-owned `--plugin-root-id=...`, e.g.
`... --ros-args --log-level io_exo_glove_ros2_plugin:=debug`.

When the plugin is launched by the framework rather than by hand, it is started from the
`plugin.yaml` `command` line, and the session config's per-plugin `plugin_args` list is appended to
it. Both are places to pass a config file or an override without touching code:

```yaml
# plugin.yaml
command: "./io_exo_glove_ros2_plugin --config=./config/io_exo_glove_ros2.yaml"
```

```python
# teleop session config
PluginConfig(
    plugin_name="io_exo_glove_ros2",
    plugin_root_id="io_exo_glove_ros2",
    search_paths=[install_dir / "plugins"],
    plugin_args=["--left-topic=/io_teleop/joint_cmd_finger_left"],
)
```

The consumer side creates a `JointStateSource(name=..., collection_id="exo_glove_left", ...)`
(and the `_right` counterpart) via `JointStateTracker`, feeding `JointStateRetargeter` with
`device_joints` set to the same URDF joint names the exoskeleton was calibrated against. The
collection ids there must match `left_collection_id` / `right_collection_id` above. See
`docs/source/device/joint_space.rst` for the shared schema/tracker/retargeter reference.

## Notes

- Each hand gets its own OpenXR tensor collection (`SchemaPusher` instance) so the two sides can
  be tracked, recorded, and retargeted independently.
- The `--plugin-root-id=...` argument injected by the `PluginManager` is dropped wherever it appears
  -- including after `--ros-args`, where the launcher appends it -- because ROS 2 rejects unknown
  arguments inside the `--ros-args` section and startup would fail. This plugin does not use the id:
  it publishes under the fixed collection ids from its config file.
- `JointState.header.stamp` is used as the raw device clock when the driver sets it (non-zero);
  otherwise the local monotonic clock is used for both timestamps, per `SchemaPusher`'s documented
  fallback convention.
- A message whose `name` and `position` arrays differ in length is dropped with a throttled warning
  rather than forwarded as a partial hand state, as is a message whose serialized payload would
  exceed the tensor buffer (`SchemaPusher::push_buffer()` throws on oversized data).
