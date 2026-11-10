<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Spatial Trackers and Device Data

Spatial tracking includes headset and controller poses, articulated hands and
bodies, and generic SE3 pose sources such as tracker pucks and mocap rigid bodies.
This guide is the common tracker reference; device-specific setup belongs in the
individual device guides. The same interfaces also carry camera metadata and
joint-space inputs.

* [VIVE Ultimate Tracker](vive_ultimate_tracker.md)

Trackers (defined in [src/core/deviceio_trackers](https://github.com/NVIDIA/IsaacCapture/tree/main/src/core/deviceio_trackers)) are the consumer-side API for reading device
data from an active [DeviceIOSession](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/deviceio_session/cpp/inc/deviceio_session/deviceio_session.hpp).
Each tracker manages one logical device, queries the OpenXR runtime every frame,
and exposes the latest sample through typed `get_*()` accessors.

There are two categories of trackers:

**OpenXR-direct trackers** – read pose and input data through standard OpenXR
APIs (`xrLocateSpace`, `xrSyncActions`, etc.):

- [HeadTracker](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/deviceio_trackers/cpp/inc/deviceio_trackers/head_tracker.hpp) – HMD head pose
- [HandTracker](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/deviceio_trackers/cpp/inc/deviceio_trackers/hand_tracker.hpp) – articulated hand joints (left and right)
- [ControllerTracker](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/deviceio_trackers/cpp/inc/deviceio_trackers/controller_tracker.hpp) – controller poses and button/axis inputs (left and right)
- [FullBodyTracker](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/deviceio_trackers/cpp/inc/deviceio_trackers/full_body_tracker.hpp) – vendor-agnostic 24-joint full body pose; default vendor reads the PICO `XR_BD_body_tracking` extension (see [Vendor Selection]())

**SchemaTracker-based trackers** – create new device type by defining a FlatBuffer schema and
reading it from OpenXR tensor collections via the
[SchemaTracker](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/live_trackers/cpp/inc/live_trackers/schema_tracker.hpp) utility.

- [FrameMetadataTrackerOak](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/deviceio_trackers/trackers.toml) – frame metadata for one OAK camera stream (generated)
- [Generic3AxisPedalTracker](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/deviceio_trackers/trackers.toml) – foot pedal axis values (generated)
- [JointStateTracker](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/deviceio_trackers/trackers.toml) – named joint-space device state (leader arms, exoskeletons, gloves, …) (generated)
- [Se3Tracker](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/deviceio_trackers/trackers.toml) – generic SE3 (6-DoF) pose sources (tracker pucks, mocap rigid bodies, logical trackers) (generated)
- [SomaBodyJointRotationsTracker](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/deviceio_trackers/trackers.toml) – SOMA body joint rotations with 77 controls (generated)
- [SomaBodyJointPosesTracker](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/deviceio_trackers/trackers.toml) – evaluated SOMA body poses for 77 joints (generated)
- [SomaHandJointRotationsTracker](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/deviceio_trackers/trackers.toml) – SOMA hand joint rotations with 25 controls (generated)
- [SomaHandJointPosesTracker](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/deviceio_trackers/trackers.toml) – evaluated SOMA hand poses for 25 joints (generated)

All trackers follow the same lifecycle:

1. Construct the tracker.
2. Pass it (along with any other trackers) to `DeviceIOSession::run()`.
3. Call `session.update()` each frame.
4. Read data with the tracker’s `get_*()` method.

#### NOTE
The `DeviceIOSession` is considered a low-level API. In practice, it is recommended to
use the [Teleop Session](../../teleop/teleop_session.md) to manage a teleop session with multiple
device trackers and retargeters to work together.

<a id="data-schema-convention"></a>

## Data Schema Convention

Every tracker’s data is defined by a FlatBuffers schema under
[src/core/schema/fbs](https://github.com/NVIDIA/IsaacCapture/tree/main/src/core/schema/fbs). Each schema follows a two-tier convention:

```idl
// 1. Payload table -- the actual data, and what trackers hand to consumers.
table Xxx {
    field_a: SomeType (id: 0);
    field_b: AnotherType (id: 1);
}

// 2. Record wrapper -- used as the MCAP recording root type.
//    Adds a DeviceDataTimestamp alongside the payload.
table XxxRecord {
    data: Xxx (id: 0);
    timestamp: DeviceDataTimestamp (id: 1);
}
```

<a id="tracked-sub-channel"></a>

### The `_tracked` recording sub-channel

A tracker that reads from a tensor collection can see **several samples per
frame**, and it records both views of that: every sample goes to `<channel>`,
while only the final sample of each `update()` – the one the live consumer
actually observed – goes to `<channel>_tracked`. Both carry the same
`XxxRecord` root type; they differ only in which samples reach them.

Replay reads `<channel>_tracked` **exclusively**, so that a replayed session
yields exactly the values the live session did rather than the intermediate
samples. The per-sample channel is there for offline analysis.

Both names come from [src/core/deviceio_trackers/defaults.toml](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/deviceio_trackers/defaults.toml) and
apply to every generated pull tracker:

```toml
mcap_channels = ["%channel%", "%channel%_tracked"]
replay_channels = ["%channel%_tracked"]
```

A manifest entry that overrides `mcap_channels` must keep the `_tracked`
entry and list it in `replay_channels`. Recording still succeeds without it,
but the resulting file cannot be replayed.

> root_type XxxRecord;
- **Payload table** (e.g. `HeadPose`, `HandPose`, `ControllerSnapshot`) –
  contains the device-specific fields. All fields are present whenever the table
  itself is present.
- **Record wrapper** (e.g. `HeadPoseRecord`) – wraps the payload plus a
  `DeviceDataTimestamp`. This is the `root_type` written to MCAP channels by
  the recorder.

### Reading a payload

The `get_*()` accessors hand out the payload table itself as an owning handle
over the encoded bytes – `Serialized<HeadPose>` in C++
([src/core/schema/cpp/inc/schema/serialized.hpp](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/schema/cpp/inc/schema/serialized.hpp)), a read-only view
class (`HeadPose`) in Python. Reads go straight into the buffer, so there is no
unpack step and joint arrays come back as zero-copy NumPy views.

An **empty handle is the absent payload**: the device is inactive, no sample has
arrived yet, or replay hit a gap. Test it with `if (handle)` in C++; in Python
the accessor returns `None`.

Each `session.update()` publishes a *new* buffer rather than refilling the
previous one, so a handle read this frame keeps its values after the next update.

To build a payload from Python, pass every field to its constructor – the view
classes expose no setters.

#### WARNING
Immutability is a **contract, not an enforcement**. The joint-array properties
hand out *writable* NumPy views, because NumPy cannot export a read-only array
over DLPack before 2.1. Writing through one changes what every holder of that
buffer sees, including handles read on earlier frames. Copy first if you mean
to modify.

#### NOTE
`MessageChannelMessagesTracked` wraps its payload in a table, because that
payload is a **list** and something has to hold the vector. Once a tracker has
run one `update()` the handle is non-empty for the rest of the session, and
an empty `data` vector – not an empty handle – means no messages arrived
this frame. Before that first update the handle is empty like any other.

### Shared Types

**DeviceDataTimestamp** ([src/core/schema/fbs/timestamp.fbs](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/schema/fbs/timestamp.fbs))

All timestamp fields are `int64` nanoseconds.

| Field                               | Description                                                                                                                                                   |
|-------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `available_time_local_common_clock` | System monotonic time when the sample became available to the recording<br/>system. Useful for measuring pipeline latency.                                    |
| `sample_time_local_common_clock`    | System monotonic time when the sample was captured. Enables<br/>cross-device synchronization (values from different devices share the<br/>same clock domain). |
| `sample_time_raw_device_clock`      | Timestamp from the device’s own clock. Values from different devices<br/>are **not** directly comparable.                                                     |

**Pose** ([src/core/schema/fbs/pose.fbs](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/schema/fbs/pose.fbs))

```idl
struct Point      { x: float; y: float; z: float; }
struct Quaternion  { x: float; y: float; z: float; w: float; }
struct Pose {
  position: Point;       // meters
  orientation: Quaternion;
}
```

<a id="tracker-reference"></a>

## Tracker Reference

### HeadTracker

Tracks the HMD head pose via the OpenXR view space.

- Schema: [src/core/schema/fbs/head.fbs](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/schema/fbs/head.fbs)
- C++ header: `#include <deviceio_trackers/head_tracker.hpp>`
- Python import: `from isaaccapture.deviceio import HeadTracker`
- Record channels: `head` | MCAP schema: `core.HeadPoseRecord`
- Tests:
  - [tests/cpp/core/schema/test_head.cpp](https://github.com/NVIDIA/IsaacCapture/blob/main/tests/cpp/core/schema/test_head.cpp)
  - [tests/python/core/schema/test_head.py](https://github.com/NVIDIA/IsaacCapture/blob/main/tests/python/core/schema/test_head.py)
- Examples:
  - [examples/oxr/cpp/oxr_simple_api_demo.cpp](https://github.com/NVIDIA/IsaacCapture/blob/main/examples/oxr/cpp/oxr_simple_api_demo.cpp)
  - [examples/oxr/python/isaaccapture_examples/oxr/modular_example.py](https://github.com/NVIDIA/IsaacCapture/blob/main/examples/oxr/python/isaaccapture_examples/oxr/modular_example.py)

### HandTracker

Tracks articulated hand joints (26 joints per hand, following the OpenXR
`XrHandJointEXT` ordering) using the `XR_EXT_hand_tracking` extension.

- Schema: [src/core/schema/fbs/hand.fbs](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/schema/fbs/hand.fbs)
- C++ header: `#include <deviceio_trackers/hand_tracker.hpp>`
- Python import: `from isaaccapture.deviceio import HandTracker`
- Record channels: `left_hand`, `right_hand` | MCAP schema: `core.HandPoseRecord`
- Tests:
  - [tests/cpp/core/schema/test_hand.cpp](https://github.com/NVIDIA/IsaacCapture/blob/main/tests/cpp/core/schema/test_hand.cpp)
  - [tests/python/core/schema/test_hand.py](https://github.com/NVIDIA/IsaacCapture/blob/main/tests/python/core/schema/test_hand.py)
  - [tests/python/xr_runtime/test_synthetic_hands.py](https://github.com/NVIDIA/IsaacCapture/blob/main/tests/python/xr_runtime/test_synthetic_hands.py)
- Examples:
  - [examples/oxr/cpp/oxr_simple_api_demo.cpp](https://github.com/NVIDIA/IsaacCapture/blob/main/examples/oxr/cpp/oxr_simple_api_demo.cpp)
  - [examples/oxr/python/isaaccapture_examples/oxr/modular_example.py](https://github.com/NVIDIA/IsaacCapture/blob/main/examples/oxr/python/isaaccapture_examples/oxr/modular_example.py)
  - [examples/retargeting/python/isaaccapture_examples/retargeting/sources_example.py](https://github.com/NVIDIA/IsaacCapture/blob/main/examples/retargeting/python/isaaccapture_examples/retargeting/sources_example.py)

### ControllerTracker

Tracks both left and right controllers – grip and aim poses, plus button and
axis inputs. Uses standard OpenXR action bindings.

- Schema: [src/core/schema/fbs/controller.fbs](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/schema/fbs/controller.fbs)
- C++ header: `#include <deviceio_trackers/controller_tracker.hpp>`
- Python import: `from isaaccapture.deviceio import ControllerTracker`
- Record channels: `left_controller`, `right_controller` | MCAP schema: `core.ControllerSnapshotRecord`
- Tests:
  - [tests/cpp/core/schema/test_controller.cpp](https://github.com/NVIDIA/IsaacCapture/blob/main/tests/cpp/core/schema/test_controller.cpp)
  - [tests/python/core/schema/test_controller.py](https://github.com/NVIDIA/IsaacCapture/blob/main/tests/python/core/schema/test_controller.py)
  - [tests/python/xr_runtime/test_controller_tracker.py](https://github.com/NVIDIA/IsaacCapture/blob/main/tests/python/xr_runtime/test_controller_tracker.py)
- Examples:
  - [examples/retargeting/python/isaaccapture_examples/retargeting/sources_example.py](https://github.com/NVIDIA/IsaacCapture/blob/main/examples/retargeting/python/isaaccapture_examples/retargeting/sources_example.py)
  - [examples/teleop/python/isaaccapture_examples/teleop/locomotion_retargeting_example.py](https://github.com/NVIDIA/IsaacCapture/blob/main/examples/teleop/python/isaaccapture_examples/teleop/locomotion_retargeting_example.py)
  - [examples/teleop/python/isaaccapture_examples/teleop/gripper_retargeting_example_simple.py](https://github.com/NVIDIA/IsaacCapture/blob/main/examples/teleop/python/isaaccapture_examples/teleop/gripper_retargeting_example_simple.py)

### FullBodyTracker

Tracks 24 body joints through a vendor-selected backend. The tracker itself is
a vendor-agnostic marker and carries no vendor or live/replay state: a live
session picks the backend via `VendorConfig` (see [Vendor Selection]()), and
replay reads the recorded `full_body` channel regardless of which vendor
produced it. When no vendor is selected, the default vendor `body.pico-xr`
reads the PICO `XR_BD_body_tracking` extension directly.

- Schema: [src/core/schema/fbs/full_body.fbs](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/schema/fbs/full_body.fbs)
- C++ header: `#include <deviceio_trackers/full_body_tracker.hpp>`
- Python import: `from isaaccapture.deviceio import FullBodyTracker`
- Record channels: `full_body` | MCAP schema: `core.FullBodyPoseRecord`
- Tests:
  - [tests/cpp/core/schema/test_full_body.cpp](https://github.com/NVIDIA/IsaacCapture/blob/main/tests/cpp/core/schema/test_full_body.cpp)
  - [tests/python/core/schema/test_full_body.py](https://github.com/NVIDIA/IsaacCapture/blob/main/tests/python/core/schema/test_full_body.py)
  - [tests/python/xr_runtime/test_full_body_tracker.py](https://github.com/NVIDIA/IsaacCapture/blob/main/tests/python/xr_runtime/test_full_body_tracker.py)
- Examples:
  - [examples/schemaio/full_body_printer.cpp](https://github.com/NVIDIA/IsaacCapture/blob/main/examples/schemaio/full_body_printer.cpp)
  - [examples/mcap_record_replay/cpp/record_full_body.cpp](https://github.com/NVIDIA/IsaacCapture/blob/main/examples/mcap_record_replay/cpp/record_full_body.cpp)
  - [examples/mcap_record_replay/python/isaaccapture_examples/mcap_record_replay/live_full_body.py](https://github.com/NVIDIA/IsaacCapture/blob/main/examples/mcap_record_replay/python/isaaccapture_examples/mcap_record_replay/live_full_body.py)
  - [examples/mcap_record_replay/python/isaaccapture_examples/mcap_record_replay/record_full_body.py](https://github.com/NVIDIA/IsaacCapture/blob/main/examples/mcap_record_replay/python/isaaccapture_examples/mcap_record_replay/record_full_body.py)
  - [examples/mcap_record_replay/python/isaaccapture_examples/mcap_record_replay/replay_full_body.py](https://github.com/NVIDIA/IsaacCapture/blob/main/examples/mcap_record_replay/python/isaaccapture_examples/mcap_record_replay/replay_full_body.py)

#### NOTE
`FullBodyTrackerPico` remains available as a deprecated alias for
`FullBodyTracker` so existing scripts run unchanged.

To run the examples from the repo root (the C++ ones need the CloudXR runtime
started first; the Python modules start it themselves):

```bash
# C++ (after cmake --install)
python -m isaaccapture.cloudxr.service start   # runs in the background
source ~/.cloudxr/run/cloudxr.env
./install/examples/schemaio/full_body_printer
./install/examples/mcap_record_replay/cpp/record_full_body 5 recordings/

# Python
uv pip install -e ./examples/mcap_record_replay
python -m isaaccapture_examples.mcap_record_replay.live_full_body --accept-eula
python -m isaaccapture_examples.mcap_record_replay.record_full_body --accept-eula
python -m isaaccapture_examples.mcap_record_replay.replay_full_body
```

See [Body Tracking](../body_tracking/index.md) for headset setup and
[MCAP Recording & Replay](../../references/mcap_record_replay.md) for the record and replay options.

### SOMA body trackers

Reads the SOMA body joint-rotation transport schema from an OpenXR tensor collection. The schema is
defined and verified against SOMA-X v0.3.1 and its v0.3 pose interface. A vendor adapter converts
its native skeleton to `SomaBodyJointRotations`, serializes the checked-in schema, and
publishes it through `SchemaPusher`. The tracker transports and records the fixed payload; it does
not perform vendor-specific skeleton or reference-pose conversion.

The payload contains the 77 user-facing joint rotations for SOMA’s 78-joint public skeleton. It omits
the virtual Root, which is always identity and is added internally by `SOMALayer`. Hips is the
global rotation; the remaining rotations use SOMA-X v0.3.1 with `reference_pose=None` and
`absolute_pose=False`. Producers must convert historical, custom, or vendor-native references
before publishing. If the native skeleton has a separate Root, the producer composes it into the
global Hips rotation and translation. Rotations are unit XYZW quaternions; `q` and `-q` represent
the same rotation, and the transport requires no quaternion-sign convention. `global_translation`
is applied to Hips and is measured in meters. The reference frame is right-handed with +Y up and +Z
forward. Each joint rotation and the global translation have independent validity.

A SOMA-X consumer can convert the quaternion array with
[soma.geometry.transforms.quaternion_xyzw_to_matrix](https://nvlabs.github.io/SOMA-X/stable/api/geometry.html)
and pass the resulting matrices to `SOMALayer.pose(..., pose2rot=False)`.

When a producer already has evaluated global positions and orientations, it can
publish `SomaBodyJointPoses` through `SomaBodyJointPosesTracker` instead.
That profile carries a global pose and independent validity for each of the same
77 public joints. Positions use meters and orientations use unit XYZW
quaternions in the same reference-frame convention. It requires no downstream
FK.

The integration defines the reference frame and keeps it stable for the collection. To align this
pose with another tracker, such as HMD full-body tracking, the producer transforms its native
tracking frame into the same physical reference frame before publishing.

For a retargeting graph, `SomaBodySource(name="body", collection_id="vendor.soma", layer=layer)`
registers the rotation tracker by default. Pass `representation="joint-poses"`
to select the evaluated-pose tracker; that profile does not require `layer`.
The source evaluates rotations once per graph step against a caller-supplied
SOMA identity or directly maps evaluated poses, then emits `SomaBodyInput`
with global poses for all 77 joints.
An inactive tracker produces an absent output. The raw tracker and transport do
not require SOMA-X or PyTorch. The live-view example supplies those evaluation
dependencies, renders the SOMA public hierarchy, and offers launch-time
body-schema selection. It does not map SOMA into `FullBodyPose`.

- Schemas: [src/core/schema/fbs/soma_body_common.fbs](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/schema/fbs/soma_body_common.fbs), [src/core/schema/fbs/soma_body_joint_rotations.fbs](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/schema/fbs/soma_body_joint_rotations.fbs), [src/core/schema/fbs/soma_body_joint_poses.fbs](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/schema/fbs/soma_body_joint_poses.fbs)
- Manifest: [src/core/deviceio_trackers/trackers.toml](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/deviceio_trackers/trackers.toml) (`soma_body_joint_rotations` and `soma_body_joint_poses`)
- C++ header: `#include <deviceio_trackers/soma_body_joint_rotations_tracker.hpp>`
- Python import: `from isaaccapture.deviceio_trackers import SomaBodyJointRotationsTracker`
- Evaluated-pose C++ header: `#include <deviceio_trackers/soma_body_joint_poses_tracker.hpp>`
- Evaluated-pose Python import: `from isaaccapture.deviceio_trackers import SomaBodyJointPosesTracker`
- Graph source: [src/python/isaaccapture/retargeting_engine/deviceio_source_nodes/soma_body_source.py](https://github.com/NVIDIA/IsaacCapture/blob/main/src/python/isaaccapture/retargeting_engine/deviceio_source_nodes/soma_body_source.py)
- Live-view example: [examples/deviceio_live_view/README.md](https://github.com/NVIDIA/IsaacCapture/blob/main/examples/deviceio_live_view/README.md)
- Demo publisher: [examples/soma_body_publisher/README.md](https://github.com/NVIDIA/IsaacCapture/blob/main/examples/soma_body_publisher/README.md)
- Record channels: `soma_body_joint_rotations`, `soma_body_joint_rotations_tracked` | MCAP schema: `core.SomaBodyJointRotationsRecord`
- Evaluated-pose record channels: `soma_body_joint_poses`, `soma_body_joint_poses_tracked` | MCAP schema: `core.SomaBodyJointPosesRecord`
- Tests:
  - [tests/cpp/core/schema/test_soma_body_joint_rotations.cpp](https://github.com/NVIDIA/IsaacCapture/blob/main/tests/cpp/core/schema/test_soma_body_joint_rotations.cpp)
  - [tests/cpp/core/schema/test_soma_body_joint_poses.cpp](https://github.com/NVIDIA/IsaacCapture/blob/main/tests/cpp/core/schema/test_soma_body_joint_poses.cpp)
  - [tests/python/core/schema/test_soma_body_joint_rotations.py](https://github.com/NVIDIA/IsaacCapture/blob/main/tests/python/core/schema/test_soma_body_joint_rotations.py)
  - [tests/python/core/retargeting_engine/test_soma_body_source.py](https://github.com/NVIDIA/IsaacCapture/blob/main/tests/python/core/retargeting_engine/test_soma_body_source.py)

### SomaHandJointRotationsTracker and SomaHandJointPosesTracker

Reads either of the two SOMA hand transport profiles defined and verified against SOMA-X v0.3.1.
`SomaHandJointRotations` contains 25 unit XYZW quaternions in `SomaHandJoint` order. Wrist
is the global rotation; the remaining rotations are joint-local relative to the fixed v0.3.1
reference contract. `global_translation` is applied to Wrist and uses the same units and
reference-frame convention as `SomaBodyJointRotations`.

When a producer has already evaluated forward kinematics, it can instead publish
`SomaHandJointPoses`. That profile carries a global position, orientation, and validity for each
of the same 25 joints and requires no downstream FK. Create one tracker per hand collection; each
payload identifies its side through `hand_side`. As with the body contract, a vendor adapter
converts its device-native hand data before calling `SchemaPusher`.

For a retargeting graph, `SomaHandSource` selects the rotation tracker by default and emits an
evaluated `SomaHandInput`. Pass `representation="joint-poses"` to select the evaluated-pose
tracker. A rotation source requires a prepared SOMA hand layer; an evaluated-pose source does not.
The DeviceIO live-view example selects body and hand inputs independently and creates one source for
each hand.

- Schemas: [src/core/schema/fbs/soma_common.fbs](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/schema/fbs/soma_common.fbs), [src/core/schema/fbs/soma_hand_joint_rotations.fbs](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/schema/fbs/soma_hand_joint_rotations.fbs), [src/core/schema/fbs/soma_hand_joint_poses.fbs](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/schema/fbs/soma_hand_joint_poses.fbs)
- Manifest: [src/core/deviceio_trackers/trackers.toml](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/deviceio_trackers/trackers.toml) (`soma_hand_joint_rotations` and `soma_hand_joint_poses`)
- Rotation C++ header: `#include <deviceio_trackers/soma_hand_joint_rotations_tracker.hpp>`
- Rotation Python import: `from isaaccapture.deviceio_trackers import SomaHandJointRotationsTracker`
- Evaluated-pose C++ header: `#include <deviceio_trackers/soma_hand_joint_poses_tracker.hpp>`
- Evaluated-pose Python import: `from isaaccapture.deviceio_trackers import SomaHandJointPosesTracker`
- Graph source: [src/python/isaaccapture/retargeting_engine/deviceio_source_nodes/soma_hand_source.py](https://github.com/NVIDIA/IsaacCapture/blob/main/src/python/isaaccapture/retargeting_engine/deviceio_source_nodes/soma_hand_source.py)
- Live-view example: [examples/deviceio_live_view/README.md](https://github.com/NVIDIA/IsaacCapture/blob/main/examples/deviceio_live_view/README.md)
- Demo publisher: [examples/soma_hand_publisher/README.md](https://github.com/NVIDIA/IsaacCapture/blob/main/examples/soma_hand_publisher/README.md)
- Rotation record channels: `soma_hand_joint_rotations`, `soma_hand_joint_rotations_tracked` | MCAP schema: `core.SomaHandJointRotationsRecord`
- Evaluated-pose record channels: `soma_hand_joint_poses`, `soma_hand_joint_poses_tracked` | MCAP schema: `core.SomaHandJointPosesRecord`
- Tests:
  - [tests/cpp/core/schema/test_soma_hand_joint_rotations.cpp](https://github.com/NVIDIA/IsaacCapture/blob/main/tests/cpp/core/schema/test_soma_hand_joint_rotations.cpp)
  - [tests/cpp/core/schema/test_soma_hand_joint_poses.cpp](https://github.com/NVIDIA/IsaacCapture/blob/main/tests/cpp/core/schema/test_soma_hand_joint_poses.cpp)
  - [tests/python/core/schema/test_soma_hand_joint_rotations.py](https://github.com/NVIDIA/IsaacCapture/blob/main/tests/python/core/schema/test_soma_hand_joint_rotations.py)
  - [tests/python/core/schema/test_soma_hand_joint_poses.py](https://github.com/NVIDIA/IsaacCapture/blob/main/tests/python/core/schema/test_soma_hand_joint_poses.py)
  - [tests/python/core/retargeting_engine/test_soma_hand_source.py](https://github.com/NVIDIA/IsaacCapture/blob/main/tests/python/core/retargeting_engine/test_soma_hand_source.py)

### FrameMetadataTrackerOak

Per-frame metadata for a **single** OAK camera stream. Create one tracker per
stream, passing the tensor collection the plugin publishes that stream under –
`{collection_prefix}/{StreamName}`, e.g. `"oak_camera/Color"`. Uses the
[SchemaTracker](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/live_trackers/cpp/inc/live_trackers/schema_tracker.hpp)
utility internally.

- Schema: [src/core/schema/fbs/oak.fbs](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/schema/fbs/oak.fbs)
- Manifest: [src/core/deviceio_trackers/trackers.toml](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/deviceio_trackers/trackers.toml) (`frame_metadata_oak`)
- C++ header: `#include <deviceio_trackers/frame_metadata_tracker_oak.hpp>`
- Python import: `from isaaccapture.deviceio import FrameMetadataTrackerOak`
- Record channels: `oak`, `oak_tracked` | MCAP schema: `core.FrameMetadataOakRecord`
- Tests:
  - [tests/cpp/core/schema/test_oak.cpp](https://github.com/NVIDIA/IsaacCapture/blob/main/tests/cpp/core/schema/test_oak.cpp)
  - [tests/python/core/schema/test_camera.py](https://github.com/NVIDIA/IsaacCapture/blob/main/tests/python/core/schema/test_camera.py)
  - [tests/python/xr_runtime/test_oak_camera.py](https://github.com/NVIDIA/IsaacCapture/blob/main/tests/python/xr_runtime/test_oak_camera.py)
- Examples:
  - [examples/schemaio/frame_metadata_printer.cpp](https://github.com/NVIDIA/IsaacCapture/blob/main/examples/schemaio/frame_metadata_printer.cpp)

The printer reads `{prefix}/Color`, `{prefix}/MonoLeft` and
`{prefix}/MonoRight` (default prefix `oak_camera`). Both processes wait for
a headset to connect to the CloudXR runtime; after that the printer reports
`Waiting for data from prefix: ...` until `camera_plugin_oak` pushes with
the same `--collection-prefix`. Run each in its own shell:

```bash
python -m isaaccapture.cloudxr.service start   # runs in the background

# Shell 1: the plugin
source ~/.cloudxr/run/cloudxr.env
./build/src/plugins/oak/camera_plugin_oak \
  --add-stream=camera=Color,output=./color.h264 \
  --collection-prefix=oak_camera

# Shell 2: the printer
source ~/.cloudxr/run/cloudxr.env
./install/examples/schemaio/frame_metadata_printer --collection-prefix=oak_camera
```

See [OAK Camera Plugin](../cameras/oak.md) to build the plugin and
[CloudXR Service](../../references/cloudxr.md#dedicated-cloudxr-runtime) for the runtime setup.

### Generic3AxisPedalTracker

Reads foot pedal axis values pushed by a device plugin through OpenXR tensor
collections. Uses the [SchemaTracker](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/live_trackers/cpp/inc/live_trackers/schema_tracker.hpp)
utility internally.

- Schema: [src/core/schema/fbs/pedals.fbs](https://github.com/NVIDIA/IsaacCapture/blob/main/src/core/schema/fbs/pedals.fbs)
- C++ header: `#include <deviceio_trackers/generic_3axis_pedal_tracker.hpp>`
- Python import: `from isaaccapture.deviceio import Generic3AxisPedalTracker`
- Record channels: `pedals`, `pedals_tracked` | MCAP schema: `core.Generic3AxisPedalOutputRecord`
- Tests:
  - [tests/cpp/core/schema/test_pedals.cpp](https://github.com/NVIDIA/IsaacCapture/blob/main/tests/cpp/core/schema/test_pedals.cpp)
  - [tests/python/core/schema/test_pedals.py](https://github.com/NVIDIA/IsaacCapture/blob/main/tests/python/core/schema/test_pedals.py)
- Examples:
  - [examples/schemaio/pedal_printer.cpp](https://github.com/NVIDIA/IsaacCapture/blob/main/examples/schemaio/pedal_printer.cpp)
  - [examples/teleop/python/isaaccapture_examples/teleop/foot_pedal_locomotion_example.py](https://github.com/NVIDIA/IsaacCapture/blob/main/examples/teleop/python/isaaccapture_examples/teleop/foot_pedal_locomotion_example.py)

#### NOTE
The Python method is named `get_pedal_data()` (instead of the C++
`get_data()`).

<a id="vendor-selection"></a>

## Vendor Selection

Some trackers are **vendor-agnostic markers**: the tracker declares *what*
device data it represents, while a live session chooses *which* backend
(“vendor”) produces that data. This mirrors how live-vs-replay is chosen at the
session level – the same tracker instance works across vendors and across live
and replay. `FullBodyTracker` is currently the only vendored tracker; its
default vendor `body.pico-xr` reads the PICO `XR_BD_body_tracking`
extension.

Select a vendor by passing a `VendorConfig` to both
`DeviceIOSession.get_required_extensions()` and `DeviceIOSession.run()`. A
`VendorConfig` maps tracker instances to a `TrackerVendor(id, params)`,
where `id` selects the backend from the live factory’s vendor registry and
`params` carries free-form string key/value options for it. Trackers left out
of the config use their default vendor. Vendor selections on non-vendored
trackers, and unknown vendor ids, are rejected at session construction.

```python
import isaaccapture.deviceio as deviceio

body = deviceio.FullBodyTracker()

# Select the backend for the vendored tracker (default shown explicitly).
vendor_config = deviceio.VendorConfig([
    (body, deviceio.TrackerVendor("body.pico-xr")),
])

required_extensions = deviceio.DeviceIOSession.get_required_extensions(
    [body], vendor_config
)
with deviceio.DeviceIOSession.run(
    [body], handles, None, vendor_config
) as session:
    ...
```

Replay is always vendor-neutral: the replay full-body impl reads the recorded
`full_body` channel regardless of which live vendor produced it, so
`VendorConfig` applies to live sessions only. The vendor registry is open for
additional pre-built plugin vendors without changing the tracker marker.

When driving devices through the higher-level teleop session manager, vendor
selection is carried on the DeviceIO source itself via its `vendor` argument
(e.g. `FullBodySource(name="full_body", vendor=deviceio.TrackerVendor("body.pico-xr"))`),
so it travels with the pipeline into both extension discovery and session
construction; see [Teleop Session](../../teleop/teleop_session.md).

<a id="tracker-usage-example"></a>

## Usage Examples

For end-to-end usage patterns combining trackers with a `DeviceIOSession`, see:

- **C++**: [examples/oxr/cpp/oxr_simple_api_demo.cpp](https://github.com/NVIDIA/IsaacCapture/blob/main/examples/oxr/cpp/oxr_simple_api_demo.cpp)
- **Python**: [examples/oxr/python/isaaccapture_examples/oxr/modular_example.py](https://github.com/NVIDIA/IsaacCapture/blob/main/examples/oxr/python/isaaccapture_examples/oxr/modular_example.py)

For higher-level usage with the teleop session manager and retargeting, see:

- [examples/retargeting/python/isaaccapture_examples/retargeting/sources_example.py](https://github.com/NVIDIA/IsaacCapture/blob/main/examples/retargeting/python/isaaccapture_examples/retargeting/sources_example.py)
- [examples/teleop/python/isaaccapture_examples/teleop/gripper_retargeting_example_simple.py](https://github.com/NVIDIA/IsaacCapture/blob/main/examples/teleop/python/isaaccapture_examples/teleop/gripper_retargeting_example_simple.py)
- [examples/teleop/python/isaaccapture_examples/teleop/locomotion_retargeting_example.py](https://github.com/NVIDIA/IsaacCapture/blob/main/examples/teleop/python/isaaccapture_examples/teleop/locomotion_retargeting_example.py)
