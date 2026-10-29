<!--
SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Preface

The ROS 2 streams are published by the IO exoskeleton SDK; start that first, then run this plugin.
The link poses (`/io_fusion/tf_exoskeleton`) are the required input: the injected hand's flexion and
spread both come from that stream. The joint-angle topic is optional -- on the current rig it is not
published, and nothing the plugin injects depends on it.

# Exoskeleton Glove ROS 2 Bridge plugin

Bridges the exoskeleton glove's own ROS 2 streams into Isaac Capture's device paths, so the mapping
onto a particular dexterous hand stays on this side (`JointStateRetargeter` or the hand retargeters)
and a recording is not tied to one hand:

- the glove's own joint angles, as `JointStateOutput` (the generic joint-space path, the same schema
  `so101_leader` / `rebot_devarm_leader` use);
- the five fingertip poses per hand, as `JointSe3PoseOutput` -- the schema for a sparse,
  vendor-chosen set of 6-DoF frames, the same one `manus` publishes its raw flex tips with.

## Data path

```
Exoskeleton SDK / ROS 2 driver
  --/io_esk/joint_data-->        sensor_msgs/msg/JointState  (optional: glove DOF angles, radians)
  --/io_fusion/tf_exoskeleton--> tf2_msgs/msg/TFMessage      (every link pose, children of base_link)
                     |
                     v
        io_exo_glove_ros2_plugin (rclcpp node)
          joints -> JointStateOutput    -> SchemaPusher("exo_glove_joints_left",
                                                        "exo_glove_joints_right", joint_state)
          tips L -> JointSe3PoseOutput  -> SchemaPusher("exo_glove_tips_left",  joint_se3_pose)
          tips R -> JointSe3PoseOutput  -> SchemaPusher("exo_glove_tips_right", joint_se3_pose)
                     |
                     v
        JointStateTracker -> JointStateSource -> retargeters
        JointSe3PoseTracker (fingertip poses)
```

The TF stream is published at about 120 Hz in the reference setup and is all the injected hand
needs. Nothing is changed on the way through: names, values and the reference frame are forwarded as
published. The joint-angle collection is optional: when the driver does not publish that topic it
simply stays quiet, which is the correct "not tracked" signal for its consumer.

## Build

Requires a sourced ROS 2 Humble environment (`rclcpp`, `sensor_msgs`, `tf2_msgs` are located via
`find_package()`, same as any ROS 2 C++ node -- no colcon/ament wrapper needed since this plugin only
*consumes* stock message types, no custom `.msg` generation):

```bash
source /opt/ros/humble/setup.bash
cmake -B build -DBUILD_PLUGIN_IO_EXO_GLOVE_ROS2=ON
cmake --build build --target io_exo_glove_ros2_plugin --parallel
```

## Configuration

Topics, collection ids and the fingertip reference frame live in `config/io_exo_glove_ros2.yaml`,
**not** in the source: edit that file and restart the plugin, with no rebuild and no recompilation of
the header defaults.

```yaml
joints_topic: "/io_esk/joint_data"
links_topic: "/io_fusion/tf_exoskeleton"
joints_left_collection_id: "exo_glove_joints_left"
joints_right_collection_id: "exo_glove_joints_right"
tips_left_collection_id: "exo_glove_tips_left"
tips_right_collection_id: "exo_glove_tips_right"
tips_reference_frame: "base_link"
wrist_source: "auto"
left_aim_to_wrist: "0,0,0,0,0,0,1"
right_aim_to_wrist: "0,0,0,0,0,0,1"
```

`left_joint_zero_offsets` / `right_joint_zero_offsets` are accepted for compatibility with older
config files but **ignored**: flexion is derived from the fingertip chords, so nothing about a
session's open pose can bias it and no open-hand calibration is needed.

Every key is optional. Precedence, lowest to highest:

1. the built-in defaults in `io_exo_glove_ros2_plugin.hpp`
2. the config file
3. the command-line flags `--joints-topic=`, `--links-topic=`, `--joints-left-collection-id=`,
   `--joints-right-collection-id=`, `--tips-left-collection-id=`, `--tips-right-collection-id=`,
   `--tips-reference-frame=`, `--wrist-source=`, `--left-aim-to-wrist=`, `--right-aim-to-wrist=`,
   `--wrist-smoothing-ms=`

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
`joint_topics:` fails loudly rather than quietly keeping the old value.

## Joint angles (collections `exo_glove_joints_left`, `exo_glove_joints_right`)

`JointState.name` is the lookup key, so the `JointStateSource` consumer reads joints **by name** and
the order on the wire does not matter. The reference setup publishes 42 joints (radians), right hand
first:

| Group | Names |
| --- | --- |
| Right hand | `joint_RightSkeletonThumbBase`, `joint_RightSkeletonThumb1..4`, `joint_RightSkeletonIndex1..4`, `joint_RightSkeletonMiddle1..4`, `joint_RightSkeletonRing1..4`, `joint_RightSkeletonPinky1..4` |
| Left hand | `joint_LeftSkeletonThumbBase`, `joint_LeftSkeletonThumb1..4`, `joint_LeftSkeletonIndex1..4`, `joint_LeftSkeletonMiddle1..4`, `joint_LeftSkeletonRing1..4`, `joint_LeftSkeletonPinky1..4` |

List them from a running driver with
`ros2 topic echo --once /io_esk/joint_data --field name`.

The collection is sized 4096 bytes -- `JointStateTracker`'s own default -- so the consumer needs no
explicit size; the reference 42-joint payload serializes to about 2 KB.

> Names you do not list are simply unused, but a name that does **not** match what the driver
> publishes is not an error: `JointStateSource` falls back to `0.0` for it, so a finger would read a
> silent zero. Check the list above against your driver before wiring a retargeter.

## Fingertip poses (collections `exo_glove_tips_left` / `exo_glove_tips_right`)

Every transform in the TF message is one link pose and every frame is a child of the same reference
frame (`base_link`), so each transform already is that frame's pose in `base_link`; no forward
kinematics is composed here. The five fingertips of each hand are published as a `JointSe3PoseOutput`
-- one collection per hand, tensor id `joint_se3_pose`, frame `type = JointType.HAND_RAW`, stamped
with the message's own header stamp -- keyed by `JointName`:

| Frame in `/io_fusion/tf_exoskeleton` | `JointName` key |
| --- | --- |
| `<left\|right>_thumb_tip` | `HAND_RAW_THUMB_TIP` |
| `<left\|right>_index_tip` | `HAND_RAW_INDEX_TIP` |
| `<left\|right>_middle_tip` | `HAND_RAW_MIDDLE_TIP` |
| `<left\|right>_ring_tip` | `HAND_RAW_RING_TIP` |
| `<left\|right>_pinky_tip` | `HAND_RAW_LITTLE_TIP` |

The schema is sparse by design: a fingertip the driver does not publish is simply absent from the
frame (there is no per-pose validity flag), so consumers must treat a missing key as "not tracked"
rather than as an invalid pose. Positions are in meters and orientations are Hamilton quaternions
(x, y, z, w).

`JointSe3PoseOutput` has no per-pose frame: one output carries poses in a single frame, which is why
`tips_reference_frame` exists. It defaults to `base_link` (the exoskeleton's own root, the frame
every transform in the reference setup is a child of) and is the frame consumers read the poses as. A
fingertip that arrives in any other frame is skipped with a throttled warning instead of being
relabelled, so a reparented driver fails visibly rather than silently publishing poses in the wrong
frame. Set the key to your exoskeleton's root frame if it is not `base_link`.

The other links in the message are not duplicated here: they follow from the joint angles by forward
kinematics, and the raw topic remains available for tooling that wants all 75 frames.

Each collection is sized 1024 bytes -- `JointSe3PoseTracker`'s own default -- so the consumer needs
no explicit size; five poses are about 200 B.

### Consumer example

```python
from isaaccapture.deviceio_trackers import JointSe3PoseTracker, JointStateTracker
from isaaccapture.schema import JointName, JointType

joints = JointStateTracker("exo_glove_joints_left")       # 4096-byte default; one tracker per hand
tips = JointSe3PoseTracker("exo_glove_tips_left")         # 1024-byte default

# ... register both with the DeviceIOSession, then each tick:
angles = {j.name: j.position for j in joints.get_data(session).data.joints}   # radians

data = tips.get_data(session).data
assert data is None or data.type == JointType.HAND_RAW
thumb = data.lookup(JointName.HAND_RAW_THUMB_TIP) if data is not None else None
if thumb is not None:
    position, orientation = thumb.pose.position, thumb.pose.orientation
```

## Hand tracking (devices `/hand/left`, `/hand/right`)

The same link poses are injected as the **OpenXR hand skeleton**, one push device per hand
(`plugin_utils::HandInjector`). This is the output a hand-tracking consumer records: it reads
Isaac Teleop's OpenXR hand tracker (`core.HandPoseRecord`, 26 joints per hand), not a tensor
collection of the plugin's own, so a glove that publishes only joint angles or fingertips has nothing
this device can drive. OpenXR requires every one of the 26 joints to be valid while a hand is active, so
all 26 are filled -- `PALM` and the four non-thumb metacarpals included.

The exoskeleton's links are a **linkage mechanism**, not a hand: in the vendor's own URDF its finger
segments are 7.6 cm and 7.5 cm (about twice a human phalanx) and the linkage leaves the finger plane
between two joints, so injecting those frames directly gives a palm a few millimetres deep and fingers
twice a palm long. What carries over from the glove is its **articulation**; the injected skeleton is
the anatomy from the vendor's human-hand model, driven by that articulation:

| What the plugin reads (mechanism) | What it injects (anatomy) |
| --- | --- |
| `link_<Side>Skeleton<Finger>1` and `<side>_<finger>_tip` translations | the fingertip **chord**, normalised against the operator's own travel, that sets the bend |
| the thumb chord's in-plane angle around the palm normal | the thumb's **opposition** (CMC, plus a share at the MCP) |
| the four `link_…Skeleton<Finger>1` local `+Z` axes and the knuckle row | the palm frame: `+X` across, `+Y` the back of the hand, `-Z` along the fingers |
| `link_<Side>Skeleton<Finger>1` (or `Thumb1`) quaternion | the knuckle spread / thumb lift (not flexion) |
| `link_<Side>SkeletonThumbBase` (revolute) | the thumb's mount, not a fifth finger |
| `<side>_<finger>_tip` | the distal phalanx beyond `…_DISTAL` |

Flexion is derived from the **fingertip**, not from the linkage's joint angles. For each finger the
chord is `|tip - knuckle|`, where the knuckle is `link_<Side>Skeleton<Finger>1` (the thumb's is
`link_<Side>SkeletonThumbBase`) and the tip is the vendor's own `<side>_<finger>_tip`. The palm frame
is built from the knuckle row: `across` is index minus little knuckle (mirrored for the right hand),
`dorsal` is the mean of the four knuckles' local `+Z` axes -- the URDF's knuckle-spread axis, i.e.
the palm normal, which is live in every message -- and `distal` is `dorsal x across`. The chord's
palmward angle is measured in that frame, so its sign is geometric: a tip on the palm side always
curls the hand towards the palm. The linkage's second frame is the first bend joint's origin only
1.1 cm from the first along the knuckle row, so "knuckle -> next frame" is not a distal direction on
this rig; the spread axis is the reliable live palm normal.

The glove's chord travel is smaller than the anatomical chain's, so an absolute chord-to-bend map
left a half-closed glove looking open: the operator had to make a full fist before the injected hand
registered. Each hand and finger therefore learns its own **travel gain** from a rolling 6 s window
of the chord: the 5th and 95th percentiles of that window are the short and long extents, and the
current chord is normalised to `fraction = (max - chord) / (max - min)`. The extents are low-passed
with a 1 s time constant, so a window update cannot step the pose; a window whose 5th..95th span is
below 5 mm has not seen a gesture, so its extents are ignored and the previous mapping kept. This is
a **per-session, per-finger gain learned from the operator's own range**, not a rig constant: a
half-range gesture still drives the full anatomical range, and a hand held still cannot make the gain
chase noise. The state lives in `FingerChordGain`/`HandChordGain` and resets when the plugin
restarts.

The normalised chord is converted to **one total bend of the anatomical chain**. The chain is the
same human model as before (MCP -> PIP -> DIP -> tip, with its fixed segment lengths; CMC -> MCP ->
IP -> tip for the thumb), and the bend is solved by bisection so that the chain's tip lands on the
anatomical chord the fraction names: the longest chord seen maps to the straight chain (0 deg), the
shortest to the fully curled one. That keeps bone lengths constant by construction. The total is
split with fixed weights 0.45 / 0.35 / 0.20 over MCP / PIP / DIP (the thumb's CMC / MCP / IP) and
each joint is clamped to its human maximum (finger MCP 85, PIP 95, DIP 70 deg; thumb CMC 35, MCP 55,
IP 75). The weights and maxima live in `io_exo_glove_ros2_kinematics.hpp`; the landmark table is
unchanged.

**Opposition.** A thumb that only flexes in the palm's plane cannot reach the index, so the chord's
lateral angle around the palm normal is tracked in the same window and normalised the same way. Its
normalised travel (0 at the widest angle seen, 1 at the narrowest) drives an extra CMC rotation
about an axis in the palm plane, plus a share at the MCP; the IP is left to the flexion solve. The
injected thumb therefore swings out of the palm's plane towards the index as the glove's thumb does.
The lateral guard is 0.1 rad (~6 deg) because it is an angle, not a length.

Because the tip chord cannot invert, the runtime bend-direction check, the per-joint open-hand zeros
and the per-joint open-to-fist gains are gone, and `--calibrate-zero` is removed. The knuckle spread
and the thumb's mount still come from the linkage's joint angles: their zero and sign are rig
constants in `io_exo_glove_ros2_kinematics.cpp`, and
`io_exo_glove_ros2_kinematics_tables.py` re-derives them from a recorded
`/io_fusion/tf_exoskeleton` stream. The plugin logs, once, how many joints each hand filled and the
mechanism root it bound to.

The human model's hand joints are exported un-fanned (every hand-joint `rpy` is zero), which would
leave the knuckles 31 mm apart and roll the finger plane 13 degrees. The landmark table therefore
also carries the A-pose: a metacarpal fan that puts the index-to-little knuckle span at 56 mm with a
level finger plane, and the thumb's own rest frame so it leaves the palm. The table is driven by the
glove's articulation on top of that rest pose; `io_exo_glove_ros2_kinematics_tables.py` prints the
neutral-frame numbers (span, `table.level`, the thumb metacarpal's position along the knuckle line)
and checks the retained spread and thumb-mount constants against the recording.

The exoskeleton reports the hand in its own frame, which is not anchored in the room, so the skeleton
is placed at a **wrist pose read from the headset**. `wrist_source` picks it: `hand_tracking` (optical,
the preference under the default `auto`), `controller` (aim pose, the fallback), or `auto`.
`$IO_EXO_GLOVE_ROS2_WRIST_SOURCE` overrides the setting; set it to match how the capture was
configured. A controller aim pose is not the wrist joint, so
`left_aim_to_wrist` / `right_aim_to_wrist` carry the rigid offset per hand (`x,y,z,qx,qy,qz,qw`,
identity by default); the plugin cannot know a controller's mounting, so it is configuration.
With no wrist source tracked the skeleton is still injected, at the origin and flagged untracked:
honest degradation rather than a frozen pose at the last known place.

The offset can be measured instead of guessed: switch to `--wrist-source=controller` (or set
`wrist_source: "controller"`), run the plugin once with `--calibrate-aim`, hold both hands flat, palm
down and fingers forward for about a second, and it prints the two `left_aim_to_wrist` /
`right_aim_to_wrist` values for that pose. Paste them into `config/io_exo_glove_ros2.yaml` and record
normally; `--aim-behind=` / `--aim-height=` move the wrist joint under the controller (defaults 4 cm
behind the aim origin along the fingers and 6 cm below it), and `--aim-flip` rolls the measured
constant 180 degrees about the fingers for a controller whose aim frame has its up axis towards the
palm (instead of the back of the hand).

The injected skeleton is expressed in the OpenXR wrist frame: `WRIST` is the origin and the rest of
the anatomy follows from the glove's articulation alone. Nothing is captured from the first frame, so
a take does not depend on how the hand was held when the plugin started. Joint orientations are built
from the bones (`-Z` along the segment, `+Y` the back of the hand, flipped for the right hand).

`wrist_smoothing_ms` (default 60, 0 disables) low-passes the wrist pose, including a valid but not
actively tracked one: optical hand tracking is noisy -- and a glove in front of the headset makes it
noisier -- while the glove's own articulation is not, so only the anchor is filtered. Smoothing a
pose the source repeats also keeps consecutive injected frames distinct instead of bit-identical.

A wrist source that stops updating is detected: when the raw pose repeats bit-identically while any
fingertip chord moved more than 2 mm, and that persists for 0.2 s, the plugin logs once and either
switches to the controller source (under `wrist_source: "auto"`, when a controller is tracked) or
drops the hand's push device until the pose changes, so the runtime reports it inactive instead of
recording a frozen wrist. Any change in the pose clears the state; a live source never trips it.

Poses are metres in the session's base space, as OpenXR requires, and the injected rate follows the
glove's 120 Hz TF stream, well above the 30 Hz a hand-tracking consumer may expect. A hand with no data for
200 ms has its push device dropped, so the runtime reports it inactive instead of holding a stale pose.

## Usage

```bash
source /opt/ros/humble/setup.bash
./build/src/plugins/io_exo_glove_ros2/io_exo_glove_ros2_plugin

# Show which config file was loaded and the effective settings, then exit (no OpenXR runtime,
# no ROS graph needed -- the fastest way to confirm an edit took effect):
./build/src/plugins/io_exo_glove_ros2/io_exo_glove_ros2_plugin --print-config

# Point at a config file, and/or override a single key for one run:
./build/src/plugins/io_exo_glove_ros2/io_exo_glove_ros2_plugin \
    --config=src/plugins/io_exo_glove_ros2/config/io_exo_glove_ros2.yaml \
    --joints-topic=/io_esk/joint_data
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
    plugin_args=["--joints-topic=/io_esk/joint_data"],
)
```

## Notes

- Every collection carries the `joint_state` or `joint_se3_pose` tensor identifier its consumer
  tracker reads, and each gets its own `SchemaPusher`, so the streams are tracked and recorded
  independently.
- The `--plugin-root-id=...` argument injected by the `PluginManager` is dropped wherever it appears
  -- including after `--ros-args`, where the launcher appends it -- because ROS 2 rejects unknown
  arguments inside the `--ros-args` section and startup would fail. This plugin does not use the id:
  it publishes under the fixed collection ids from its config file.
- `JointState.header.stamp`, or the first transform's stamp for the TF message, is used as the raw
  device clock when the publisher sets one; otherwise the local monotonic clock is used for both
  timestamps, per `SchemaPusher`'s documented fallback convention.
- A joint message whose `name` and `position` arrays differ in length is dropped with a throttled
  warning rather than forwarded as a partial device state, as is any frame whose serialized payload
  would exceed its tensor buffer (`SchemaPusher::push_buffer()` throws on oversized data).
- A fingertip frame is skipped (with a throttled warning) when it is not expressed in
  `tips_reference_frame`: one `JointSe3PoseOutput` carries a single reference frame for all of its
  poses, and the schema requires the producer to document it, so the pose is never relabelled to fit.
- A hand with no fingertip frame in the message, or none in `tips_reference_frame`, pushes nothing for
  that hand: the schema is sparse, and a missing tensor means "not tracked", not "identity pose".
