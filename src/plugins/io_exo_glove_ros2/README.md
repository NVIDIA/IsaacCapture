<!--
SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Preface

The required ROS 2 topics are published by the IO exoskeleton SDK; start that first, then run this
plugin.

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
  --/io_esk/joint_data-->        sensor_msgs/msg/JointState  (glove skeleton DOF angles, radians)
  --/io_fusion/tf_exoskeleton--> tf2_msgs/msg/TFMessage      (every link pose, children of base_link)
                     |
                     v
        io_exo_glove_ros2_plugin (rclcpp node)
          joints -> JointStateOutput    -> SchemaPusher("exo_glove_joints",    joint_state)
          tips L -> JointSe3PoseOutput  -> SchemaPusher("exo_glove_tips_left",  joint_se3_pose)
          tips R -> JointSe3PoseOutput  -> SchemaPusher("exo_glove_tips_right", joint_se3_pose)
                     |
                     v
        JointStateTracker -> JointStateSource -> retargeters
        JointSe3PoseTracker (fingertip poses)
```

Both topics are published at about 120 Hz in the reference setup. Nothing is changed on the way
through: names, values and the reference frame are forwarded as published.

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
joints_collection_id: "exo_glove_joints"
tips_left_collection_id: "exo_glove_tips_left"
tips_right_collection_id: "exo_glove_tips_right"
tips_reference_frame: "base_link"
wrist_source: "auto"
wrist_alignment: "auto"
# The persistent session reference (see "Session reference" below): paste the snippet that
# --export-session-reference= writes. Both keys are optional; a missing hand captures as before.
reference_left: "chord thumb=...; offset thumb=...; gap index=.../...; angle thumb0=... ..."
reference_right: "chord thumb=...; offset thumb=...; gap index=.../...; angle thumb0=... ..."
# Where to write the reference snippet after a good capture (and at start-up when one is loaded).
export_session_reference: "/tmp/glove_reference.yaml"
```

Every key is optional. Precedence, lowest to highest:

1. the built-in defaults in `io_exo_glove_ros2_plugin.hpp`
2. the config file
3. the command-line flags `--joints-topic=`, `--links-topic=`, `--joints-collection-id=`,
   `--tips-left-collection-id=`, `--tips-right-collection-id=`, `--tips-reference-frame=`,
   `--wrist-source=`, `--wrist-alignment=`, `--session-reference=` (a file carrying the two reference
   keys; overrides the config per hand) and `--export-session-reference=`

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

## Joint angles (collection `exo_glove_joints`)

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

joints = JointStateTracker("exo_glove_joints")            # 4096-byte default
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
(`plugin_utils::HandInjector`). This is the output the device conformance suite records: it reads
Isaac Teleop's OpenXR hand tracker (`core.HandPoseRecord`, 26 joints per hand), not a tensor
collection of the plugin's own, so a glove that publishes only joint angles or fingertips cannot be
conformance tested. OpenXR judges 21 of the 26 joints (`PALM` and the four non-thumb metacarpals are
optional), and the exoskeleton's own chain ends at the fingertips, which is exactly 21 poses per hand:

| OpenXR joint | Frame in `/io_fusion/tf_exoskeleton` |
| --- | --- |
| `WRIST` | `joint_<Side>SkeletonBase` (`<side>_hand` also accepted) |
| `THUMB_METACARPAL`, `THUMB_PROXIMAL`, `THUMB_DISTAL` | `link_<Side>SkeletonThumbBase`, `…Thumb1`, `…Thumb2` |
| `THUMB_TIP` | `<side>_thumb_tip` |
| `INDEX_PROXIMAL`, `INDEX_INTERMEDIATE`, `INDEX_DISTAL` | `link_<Side>SkeletonIndex1..3` |
| `INDEX_TIP` | `<side>_index_tip` |
| `MIDDLE_*` | `link_<Side>SkeletonMiddle1..3`, `<side>_middle_tip` |
| `RING_*` | `link_<Side>SkeletonRing1..3`, `<side>_ring_tip` |
| `LITTLE_*` | `link_<Side>SkeletonPinky1..3`, `<side>_little_tip` |

A rig whose link names differ only needs that table changed in `io_exo_glove_ros2_hand.cpp`; the plugin
logs every slot it resolved on the first frame that carries a hand and lists the ones it could not
fill, so one run says exactly what bound to what.

The exoskeleton reports the hand in its own frame, which is not anchored in the room, so the skeleton
is placed at a **wrist pose read from the headset**. `wrist_source` picks it: `hand_tracking` (optical,
the preference under the default `auto`), `controller` (aim pose, the fallback), or `auto`.
`$IO_EXO_GLOVE_ROS2_WRIST_SOURCE` overrides the setting -- the conformance capture's `--wrist-source`
only labels the take, so set the variable to match. The glove's wrist frame and the headset's wrist
frame differ by a fixed rotation per hand, and `wrist_alignment: "auto"` captures it from the first
frame where both are tracked, logging it as `x,y,z,w` to paste into the config and make a take
reproducible; `identity` says the frames already agree, and an explicit `x,y,z,w` pins the rotation.
With no wrist source tracked the skeleton is still injected, at the origin and flagged untracked:
honest degradation rather than a frozen pose at the last known place.

Poses are metres in the session's base space, as OpenXR requires, and the injected rate follows the
glove's 120 Hz TF stream -- above the 30 Hz the conformance suite asks for. A hand with no data for
200 ms has its push device dropped, so the runtime reports it inactive instead of holding a stale pose.

### Session reference: calibrate once, store it in the config

The injected hand is a fixed-length human chain (MCP/PIP/DIP 85/95/70 deg, thumb CMC/MCP/IP
35/55/75 deg) driven by the exoskeleton's **own joint angles**, not by a chord length. Per hand and
per mechanism joint the plugin computes the absolute joint angle from the published link quaternions
with the v5 URDF rest rotations and revolute axes:

```
theta = signed_angle(conj(R_rest) * conj(R_parent) * R_child, axis)
```

Each joint's angle is unwrapped around its baked open reading, so it is continuous across the
+-pi boundary. The session reference stores the **open-pose angle of every mechanism joint** (20
values per hand: four per digit), and the fraction is

```
f = clamp((theta - theta_open) / (theta_closed - theta_open), 0, 1)
```

where `theta_closed` is the joint's baked fist angle. The model's joints are `f * limit` (85/95/70,
thumb 35/55/75), so at the session's open pose every joint reads exactly 0 and at the mechanism's
own closed angle every joint reads exactly its human maximum. The four fingers combine their MCP and
DIP (the PIP is a coupling link that barely articulates in this mechanism's fist), so the whole
finger follows the hinges that carry its curl.

r35 adds an **open deadband** at the open end, stored per hand in the reference:

```
d     = deadband / 85 deg                 # hand-level, model degrees
f_raw = clamp((theta - theta_open) / (theta_closed - theta_open), 0, 1)
f     = clamp((f_raw - d) / (1 - d), 0, 1)
      = clamp((moved - D) / (|travel| - D), 0, 1)      with D = d * |travel|, moved = |theta - theta_open|
```

So the deadband is one hand-level number, but it is applied as a fraction of each hinge's own
travel (the equivalent per-joint angle `D` in the formula). Everything within `d` of the session's
open pose reads exactly 0; `theta = theta_closed` still reads exactly 1 because both ends are
shifted by the same fraction, so the fist is **not** compressed. An r31 four-section block has no
deadband (`d = 0`) and runs the old map bit for bit. The value is the measured relaxed-vs-stiff
offset per hand, plus 2 deg, capped at **25 deg**; it is printed by `--print-config` and carried by
`format -> parse -> format`.

The **closed table** (`kAngleClosed`) is the current mounting's fist (live_20261112_095821, frame
6968). The r31 table was the 09:18 mounting's deeper fist and left the 09:58 fist at 80-84 deg; the
09:58 fist now reads exactly 85/95/70, and the deeper 09:18 fist still clamps to the maxima.

The **tip-frame orientation is not a rig constant** -- the same finger has been measured 147 deg from
the human axis in one driver session and 54 deg in another -- so the reference also keeps the open
chord and the rotation from the device's open tip direction to that digit's human rest axis, plus the
opening and tightest thumb-to-finger gap. The chord values are **not** used by the angle driver; they
are what a reference written by an earlier round (or a failed capture) falls back to.

Those measurements are a **session reference**. It is captured automatically from the first frames,
but it can also be written to the config: then every later start loads it and the capture window is
skipped entirely. The operator calibrates once and never repeats the start-up ritual. A **loaded
five-section (r35)** reference is authoritative: r35 no longer widens it to the take's most-extended
frame, because that is exactly what turned a relaxed or stiff straight hand into curl. A **captured**
reference, and a four-section r31 block without the deadband, still widen as in r31; the open
deadband absorbs the session-to-session variation instead.

#### The block format

One quoted scalar per hand, five `;`-separated sections: `chord`, `offset`, `gap`, `angle`,
`deadband`. Chords and gaps are millimetres with three decimals; each offset is a unit quaternion
`x,y,z,w` with six decimals; a gap is `open/tightest`; each angle is **degrees** with three
decimals; the deadband is one number in **model degrees** (0..25). Digit order is always thumb,
index, middle, ring, little; the gap section has the four fingers only. The angle section has all 20
mechanism joints, keyed `<digit><slot>` with slot 0 the knuckle spread/mount and slots 1..3
MCP/PIP/DIP (thumb CMC/MCP/IP), in the URDF chain order. The names make the block auditable and the
section order does not matter:

```yaml
reference_left: "chord thumb=83.827 index=83.800 middle=84.670 ring=87.160 little=65.540; offset thumb=-0.017458,-0.089227,-0.144430,0.985329 index=0.588302,0.073364,-0.067845,0.802444 middle=0.620804,0.068593,0.021338,0.780668 ring=0.576115,0.072535,0.012770,0.814043 little=0.583541,0.027232,-0.034353,0.810899; gap index=120.218/97.356 middle=143.309/116.131 ring=156.262/132.404 little=155.598/138.059; angle thumb0=-32.296 thumb1=-32.566 thumb2=33.574 thumb3=229.892 index0=-23.883 index1=23.115 index2=-138.535 index3=25.291 middle0=-19.983 middle1=17.154 middle2=-138.890 middle3=14.634 ring0=-17.361 ring1=18.193 ring2=-134.844 ring3=10.217 little0=-14.530 little1=20.237 little2=-153.437 little3=22.661; deadband 25.000"
reference_right: "chord thumb=87.419 index=84.280 middle=87.610 ring=82.450 little=62.320; offset thumb=-0.267492,-0.041677,0.079838,0.959342 index=0.606993,0.067513,-0.057525,0.789742 middle=0.529825,0.054947,0.034908,0.845605 ring=0.485412,0.072808,0.024949,0.870891 little=0.527497,0.071893,-0.022518,0.846210; gap index=120.870/99.990 middle=143.430/117.250 ring=154.300/129.790 little=161.490/139.010; angle thumb0=0.641 thumb1=24.795 thumb2=27.815 thumb3=240.079 index0=12.023 index1=-10.989 index2=-131.306 index3=7.602 middle0=1.995 middle1=-4.593 middle2=-131.421 middle3=0.762 ring0=-3.001 ring1=-5.398 ring2=-136.496 ring3=-4.835 little0=-30.148 little1=-7.794 little2=-154.440 little3=14.334; deadband 19.700"
```

A block is **complete** when both sections carry every digit; a hand with a complete block never runs
the window. A missing or malformed block falls back to the auto-capture, with a warning. A block is
rejected (and the window captures instead) when:

- a chord is outside **20-160 mm**, or a finger's open chord is not above its baked closed chord
  (the legacy chord map would have no travel);
- an offset quaternion's norm is not 1 within `1e-3` (it is normalised after the check);
- a gap is negative, or its tightest value exceeds its open value;
- an angle is outside **-720..720 deg**;
- a section or a digit is missing, duplicated or truncated.

A **three-section block** (chord, offset, gap; one written by an earlier round) still loads: it is
accepted, but the hand then runs the **chord driver**, exactly as before, and the load line is a
warning:

```
session reference: left: loaded from config (no angle section: using the CHORD driver; re-export the reference to drive from the mechanism's joint angles)
```

A **four-section block** (chord, offset, gap, angle; an r31 block) still loads with the
**joint-angle driver** but no deadband: `d = 0`, the r31 map bit for bit, and a warning asks for a
re-export.

A **five-section block** loads with the **joint-angle driver and the open deadband**:

```
session reference: left: loaded from config (joint-angle driver, open deadband 25.000 deg)
```

`--print-config` prints both blocks and their driver (`<valid, joint-angle driver, open deadband
25.000 deg>`, `<valid, joint-angle driver, no open deadband (r31 map)>` or `<valid, chord driver
(legacy block)>`), so a typo is visible before the plugin runs. The captured reference is
**quantized to the same 3/6-decimal spelling at the close of the window**, so the values a run uses
are exactly the values the config will carry: capture -> export -> reload reproduces the same
mapping bit for bit. A deadband above 25 deg is rejected; the cap is enforced by the parser too.

#### Writing the reference (export)

After a good capture (and at start-up when a block is loaded), the plugin writes a YAML snippet to
the path given by `export_session_reference` (config) or `--export-session-reference=PATH`:

```bash
./io_exo_glove_ros2_plugin --export-session-reference=/tmp/glove_reference.yaml
```

```yaml
# SPDX-FileCopyrightText: Copyright (c) 2026 IO. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# io_exo_glove_ros2 session reference, generated 2026-11-12T09:50:00+0800 by io_exo_glove_ros2 1.0.0.
# Paste these keys into config/io_exo_glove_ros2.yaml, or pass this file as
# --session-reference=<this file>. Format and operator workflow: README.md, "Session reference".
# left: captured this run, 40 frames in 0.400 s
# right: captured this run, 40 frames in 0.400 s
reference_left: "chord ...; angle ..."
reference_right: "chord ...; angle ..."
```

The header carries the date, the plugin version and, per hand, the capture statistics (frames and
seconds). A hand without a usable reference is listed in the header and omitted from the keys, so a
one-handed rig exports exactly what it has. Exporting a *loaded* reference writes it back unchanged
(that is the start-up write).

#### Loading the reference

Precedence, lowest to highest:

1. built-in empty block (the window captures)
2. `reference_left` / `reference_right` in the config file
3. `--session-reference=PATH`: reads the same two keys from that file (the exported snippet, or the
   config itself). A hand the file omits keeps the config's block, so a one-handed file never
   disturbs the other hand.

With a loaded block the plugin logs one load line and never logs a capture line for that hand: the
first frame is already mapped. Without a block it logs
`session reference: left: not in config; capturing from the first frames`, and an invalid block logs
`session reference: left: rejected (<reason>); capturing from the first frames` as a warning.

The stale guard (a stream that never changes in the window is held open and reported) belongs to the
capture window; a hand that loads a reference from the config starts mapped, exactly as its stored
reference says. A hand whose frames stop arriving is still dropped by the plugin after 200 ms, as
before.

#### Capture window (the fallback)

The window is 40 valid frames or 2 s (from 20 frames on), and never later than 4 s. Until it closes
every digit is held open, because the open reference is what makes an angle (or a chord) meaningful.
The window stores the open-pose angle of every mechanism joint, the open chords, the tip-direction
offsets and the thumb gaps. A window that cannot be satisfied falls back to the baked table and the
hand is injected regardless; a hand whose frames never changed in the window (an idle or frozen
glove) is held open/neutral and reported as stale instead of being read as flexion. Every outcome
logs one line per hand:

```
session open reference captured: left: thumb 90.1 mm (+0.0 deg) index 98.9 mm (+50.5 deg) ...
session open reference not captured: left: only 3 valid frame(s) in 4.00 s; using the baked table
```

After the window a **captured** reference only **widens**: a pose more open than the stored one
replaces the open angles of the whole hand, and a longer chord or a wider gap replaces the stored
one. It can never collapse, so a window that caught the hand part-way bent repairs itself the next
time the operator opens the hand. A **loaded five-section (r35)** reference does not widen; the
stored calibration is authoritative and the open deadband absorbs the session variation. A loaded
four-section r31 block keeps the old widen-on-a-more-open-pose behaviour.

The angle fraction maps **linearly** to the bend above the deadband: each joint moves from 0 at the
session's open angle (and stays 0 for the first `deadband` of model bend) to its own limit at the
baked closed angle, so the fist is still exactly 85/95/70. The raw mechanism travel differs per joint
(the linkage moves roughly half as far as the finger), which is why the per-joint gain is baked with
the closed table; the operator-visible transfer above the deadband is the 1:1 one.

The **thumb** is driven by the glove's own thumb-to-finger gaps, the only sign-stable signal the
device gives: at the session's opening gap it is neutral/abducted, and as the gap closes it rotates
about its CMC towards the injected tip of the finger being pinched, the three hinges staying the same
fraction of their 35/55/75 deg limits. At the session's open pose it is held exactly neutral (the
thumb is the one digit whose amount the angle table does not drive); its open angle is still stored
in the reference. Its old device-direction aim (which reached 67 deg) is gone.

### Recording a conformance take

Device conformance is recorded from a build of this repository, with the CTS copied into
`src/python/isaaccapture/cts/`:

```bash
src/python/isaaccapture/cts/hand/checker/setup_env.sh --panel
src/python/isaaccapture/cts/hand/capture/setup_env.sh
src/python/isaaccapture/cts/hand/capture/record.sh exo_glove --plugin io_exo_glove_ros2 --wrist-source hand_tracking
```

`record.sh` launches the plugin from `install/plugins/` (`cmake --install build`), so install before
recording. The capture panel then drives the eleven scripted gestures and writes the take; check it
with `hand_cts.cli` and package it with `hand_cts.panel`. See **Device Conformance Testing** for the
full procedure.

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

# Persist a reference: capture once, write the snippet, then load it on every later start.
./build/src/plugins/io_exo_glove_ros2/io_exo_glove_ros2_plugin \
    --export-session-reference=/tmp/glove_reference.yaml       # run once, hand open, then stop
./build/src/plugins/io_exo_glove_ros2/io_exo_glove_ros2_plugin \
    --session-reference=/tmp/glove_reference.yaml              # every later start: no window
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
