<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Pose Provider — Interface Design

This design introduces a backend-neutral Pose Provider that converts synchronized camera
and optional IMU measurements into a 6-DoF pose stream. The initial OAK-D integration targets
OAK-D stereo and stereo-inertial data collection while allowing different SLAM/VIO backends
behind one API. Existing pose transport, recording/replay, and consumer APIs remain
unchanged. Capture APIs, vendor synchronization, recording formats, and SLAM algorithms are
outside this design.

## Motivation

IsaacTeleop needs 6-DoF pose from cameras that expose images and optional IMU data but do
not provide pose through the existing teleop device path. A proven OAK-D and ORB-SLAM3
data-collection pipeline provides the behavioral baseline; cuVSLAM is the intended
production backend.

## Goals

- Define a common Pose Provider contract that supports interchangeable SLAM/VIO backends and
  produces one 6-DoF pose stream from synchronized camera and optional IMU measurements.
- Preserve measurement timing and source identity so pose output remains correlatable with
  recorded sensor data.
- Define consistent input, calibration, frame, lifecycle, and health contracts across
  backends, including a stable pose frame for each run.
- Reuse the existing pose transport, recording, replay, and consumer APIs.
- Keep capture, estimation, and transport responsibilities separate so the design can grow
  without coupling the core provider contract to vendor-specific mechanisms.

## Initial release scope

The initial release has one supported input topology and two input modes:

| Topology | Input mode |
|---|---|
| One OAK-D stereo pair | `Stereo` |
| One OAK-D stereo pair + its onboard IMU | `StereoImu` |

For the initial release:

- The supported device is one physical OAK-D providing one stereo pair and, optionally, its
  onboard IMU. It forms one rigid rig and one raw device-clock domain.
- The existing OAK plugin remains the device owner. One process coordinates capture,
  sensor recording, pose estimation, and publication from the same capture pipeline without
  depending on the unfinished generic capture interface.
- Input uses owned CPU buffers from two pinhole camera streams. Configuration selects one
  fixed body frame, and resolved calibration describes the exact delivered streams.
- The supported use case is data collection: measurement correctness and sensor/pose
  correlation take priority over low output latency.
- One binary contains one selected backend and produces one pose stream through one existing
  `se3_tracker` collection. ORB-SLAM3 is the functional reference, cuVSLAM is the production
  reference, and the stub is the contract-test reference.
- Output uses the backend's fixed `SlamWorld` by default or one fixed startup transform into
  `Session`. Dynamic map-frame or alignment changes are not supported within a run.

## Design boundaries and non-goals

These are responsibilities this design does not own, including when a deferred capability
eventually uses them:

- Designing camera or IMU capture APIs, delivery to multiple consumers, vendor
  synchronization, or clock mapping mechanisms.
- Defining sensor-recording infrastructure or file formats.
- Defining backend algorithms or backend-native configuration formats.
- Defining unrelated camera consumers, cross-machine transport, or collaborative-map
  protocols. Those components may integrate with the Pose Provider through separate designs.

## Deferred capabilities and extension directions

These capabilities are related to the Pose Provider but are not initial-release requirements
or delivery commitments. They may require interface changes or separate designs:

1. **Capture-interface migration.** Replace the direct OAK-D integration after a generic
   capture interface can provide the required synchronization, calibration, source identity,
   delivery to recording and other consumers, and lifecycle outcomes.
2. **Input expansion.** Add other devices, mono and larger camera groups, multi-adapter or
   cross-clock rigs, additional camera and timing models, and accelerated buffer transport.
3. **SLAM and map behavior.** Add operating modes such as localization against an existing
   map, map reset or dynamic alignment, and collaborative/shared-map operation.
4. **Consumption and output.** Add forward prediction, real-time control and safety
   contracts, or a richer pose schema when a concrete consumer requires them.

The initial release stabilizes only the seams it exercises; these directions are not
compatibility promises.

## Architecture and ownership boundary

There are three independent seams:

1. **Capture seam.** The capture layer owns devices, synchronization, sensor recording, and
   delivery to its consumers.
2. **Estimator seam.** `IPoseProvider` owns SLAM/VIO state and backend-native translation.
3. **Transport seam.** Existing `se3_tracker` schema, consumer API, recording, and replay
   remain unchanged; live consumers use the collection name derived from `instance_id`.

`SensorAdapter` is in scope only as the initial OAK-D capture adapter. This design defines
its provider-facing normalization, `AppliedCaptureSpec` construction, fusion-time ordering,
delivery, and lifecycle obligations—not device access, vendor synchronization, capture APIs,
or recording infrastructure.

### Runtime ownership

The session integration owner composes these runtime responsibilities and coordinates their
lifecycle. Each child retains the internal responsibility defined by the three seams above.

```text
Session integration owner
        ├── Capture integration
        │       ├── capture/device source
        │       ├── sensor recording
        │       └── capture adapter
        ├── IPoseProvider backend
        ├── output integration
        ├── telemetry
        └── lifecycle and shutdown
```

The table maps those ownership responsibilities to the initial OAK-D implementation without
adding new contract behavior.

| Generic responsibility | Initial OAK-D integration |
|---|---|
| Session integration owner | `plugins::oak::DataCollectionSession` |
| Capture integration | `DataCollectionSession` coordinates `OakCamera`, the recorder, and `plugins::oak::SensorAdapter` |
| Capture/device source | `OakCamera` |
| Sensor recording | Session-owned recorder |
| Capture adapter | `plugins::oak::SensorAdapter` |
| `IPoseProvider` backend | ORB-SLAM3, cuVSLAM, or stub adapter |
| Output integration | `DataCollectionSession` update loop and output-frame transform |
| Telemetry | `DataCollectionSession` live health view and episode summary |
| Lifecycle and shutdown | `DataCollectionSession` |

The existing OAK plugin process remains the single device-owning process. Recording and pose
estimation branch from the same capture pipeline; a separate pose process must not reopen
the device. `OakCamera`, the recorder, `SchemaPusher`, and `Se3Tracker` remain existing or
separately designed dependencies.

### End-to-end data flow

The ownership view does not show the data path. This diagram instead makes the provider's
inputs and outputs visible at a glance.

```text
capture/device source ── sensor data ──┬──▶ sensor recording
                                       │
                                       └──▶ capture adapter
                                                 │
                                      fusion-time-ordered sensor input
                                      (ImageGroup; optional ImuSample)
                                                 │
                                                 ▼
                                           IPoseProvider
                                      ┌──────────┴──────────┐
                                      │                     │
                                PoseEstimate          ProviderHealth
                                      │                     │
                                      ▼                     ▼
                             output integration          telemetry ◀── diagnostics
                                      │                              from capture/session
                                      ▼
                                  pose stream
                               ┌──────┴──────┐
                               ▼             ▼
                         live consumers  pose recording
```

The two capture branches receive the same source measurement, but provider delivery through
the adapter occurs only after recorder acceptance succeeds, as required below.

The session assembles one immutable setup and queries capabilities and identity before
initialization. The capture integration then submits normalized measurements for asynchronous
processing in fusion-time order, while the session drains estimates and health without
exposing provider pull calls to pose consumers.

### Interface and type relationships

```text
ImageGroup ──contains──▶ ImageFrame
ImageGroup ─┐
ImuSample ──┴──accepted by──▶ ISensorSink ◀──implemented by── IPoseProvider

SlamInstanceConfig ─┐
AppliedCaptureSpec ──┼──validate + initialize──▶ IPoseProvider
RigCalibration ──────┘                                  │
                                                       ├── advertises ProviderCapabilities
                                                       ├── identifies ProviderIdentity
                                                       ├── produces PoseEstimate
                                                       └── reports ProviderHealth

IPoseProvider lifecycle:
validate_configuration → initialize → start → finish_input → input_drained → stop
```

In contrast to the runtime view above, this UML-lite view shows only contract-level type
relationships, not ownership or end-to-end flow. The API sections below define the supporting
types without turning the diagram into a field-by-field data model.

## Capture-integration requirements

The direct OAK-D integration and any future generic capture interface must provide the same
provider-facing contract:

- **Synchronized input.** Deliver each image measurement as one atomic `ImageGroup`, with
  stable sensor and capture identities, source sequences, and device and local measurement
  timestamps. Optional IMU samples use the same clock domain. Initial OAK-D stereo frames
  have exactly equal device timestamps; future capture implementations may use another
  declared grouping rule. Host arrival time is never treated as measurement time.
- **Recording correlation.** The data-collection session records the same measurements it
  offers to the provider, including their identities and timestamps. Recorded image-group
  metadata retains each image's sensor ID and source sequence. Recorder acceptance precedes
  provider delivery; a measurement that recording cannot accept is not sent to the provider.
  A later recording failure fails the session rather than leaving an apparently valid pose
  stream without its source data. The provider remains independent of the recording
  implementation and carries source identity and measurement time into `PoseEstimate`.
- **Immutable applied setup.** Before provider initialization, the integration supplies one
  selected `ProviderInputMode`, one `AppliedCaptureSpec` describing the streams actually
  delivered, and one fully resolved `RigCalibration`. Calibration describes the delivered
  images after any capture-side processing and is established before measurement delivery
  begins. The same records are used for validation, provider initialization, fusion-time
  ordering, and episode metadata, and remain fixed for the run.

For the initial OAK-D integration, `OakCamera` reports the applied capture-native stream
properties and device calibration without depending on Pose Provider types.
`plugins::oak::SensorAdapter` converts the stream report into the provider-facing
`AppliedCaptureSpec`, while the session orchestrator applies any file override to the device
calibration and resolves the final `RigCalibration` once. Future capture-interface migration
may change where the source information comes from, but not these provider-facing outcomes or
ownership boundaries.

### Initial OAK-D fusion-time ordering

`SensorAdapter` merges complete image groups and IMU samples into one monotonic fusion-time
stream using a reorder buffer bounded by a configured time window and item capacity. Capacity
must cover the items expected within that window at the applied stream rates, plus the
qualified delivery-jitter margin. Camera fusion time is the group device timestamp; IMU
fusion time applies the resolved calibration offset without changing the recorded raw
timestamp. An item is normally emitted when its fusion time is no greater than the greatest
observed fusion time minus the window.

An incoming item older than the last emitted fusion time is late, counted, and withheld
before buffering. Otherwise, when admitting it would exceed capacity, the adapter emits the
earliest item from the union of the buffer and the incoming item. If the incoming item is
earliest, it is emitted directly; otherwise, the earliest buffered item is emitted and the
incoming item is inserted. The adapter never rejects an on-time item solely because the
reorder buffer is full. Normal shutdown flushes buffered items in order.

The applied window and capacity, forced-emission count, and late-input count are recorded in
episode metadata. They belong to capture/session configuration and telemetry rather than
`SlamInstanceConfig` or `ProviderHealth`.

## Pose Provider API

### Input ingestion — `ISensorSink`

```cpp
namespace core {

// Stable identifiers shared across capture, calibration, and provider records.
using SensorId = std::string;
using RigId = std::string;
using FrameId = std::string;
using ClockDomainId = std::string;
using CaptureGroupId = uint64_t;

// Pixel formats accepted at the provider boundary.
enum class PixelFormat { Gray8, NV12, BGR888, RGBA8 };

// One captured image with owned pixel data and source timestamps.
struct ImageFrame {
    SensorId sensor_id;
    uint64_t source_sequence;
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    PixelFormat format;
    std::vector<uint8_t> data;       // owned CPU buffer
    int64_t sample_time_local_ns;    // local common monotonic clock
    int64_t sample_time_device_ns;   // raw capture-device clock
};

// One atomic, synchronized camera measurement.
struct ImageGroup {
    CaptureGroupId capture_group_id;
    int64_t sample_time_local_ns;    // group measurement time in local common clock
    int64_t sample_time_device_ns;   // group measurement time in raw device clock
    std::vector<ImageFrame> images;  // owned frames; matched by sensor_id
};

// One accelerometer and gyroscope measurement.
struct ImuSample {
    SensorId sensor_id;
    uint64_t source_sequence;
    double accel_mps2[3];
    double gyro_radps[3];
    int64_t sample_time_local_ns;
    int64_t sample_time_device_ns;
};

// Result of a non-blocking input-admission attempt.
enum class EnqueueResult {
    // The complete measurement was queued.
    Accepted,

    // A full queue rejected the complete measurement.
    DroppedBackpressure,

    // The measurement does not match the initialized input contract.
    RejectedInvalidInput,

    // The provider is not accepting input in its current lifecycle state.
    RejectedNotRunning,
};

// Non-blocking sensor-input boundary implemented by each Pose Provider.
class ISensorSink {
public:
    virtual ~ISensorSink() = default;

    // One atomic, owned image measurement.
    virtual EnqueueResult on_image_group(ImageGroup group) = 0;

    // Available only in StereoImu mode.
    virtual EnqueueResult on_imu(const ImuSample& sample) = 0;
};

} // namespace core
```

#### Ownership and call model

The capture integration satisfies the synchronization, recording, and applied-setup
requirements above before calling this interface. `ISensorSink` only admits work; SLAM/VIO
processing happens asynchronously.

- `on_image_group()` transfers one complete group and its image buffers. `on_imu()` copies
  the small sample before returning.
- Each image group contains exactly one frame for every configured camera sensor. Sensor ID,
  not vector position, identifies the stream.
- The capture integration serializes input and delivers it in monotonic fusion-time order
  under the capture-side contract above. Camera groups use their device measurement time;
  IMU samples apply the resolved calibration offset for ordering without changing their
  recorded raw timestamps.
- Input calls are legal only while the provider is running. One input producer may operate
  concurrently with one thread draining estimates and reading health; lifecycle operations
  remain serialized.

#### Backpressure behavior

Input methods use finite per-provider queues and never block:

- A full image queue rejects the incoming `ImageGroup` as one atomic unit.
- Every dropped IMU sample increments `dropped_imu_samples`. The first drop before the next
  accepted IMU opens one gap episode and increments `imu_gap_events`; further drops in that
  episode only increment the sample count. The next accepted IMU carries an internal pending
  gap marker, and the provider wrapper notifies the backend adapter before ingesting that
  sample. The adapter owns the resulting tracking behavior; `ISensorSink` does not expose the
  internal notification.
- Drops and rejections are counted in provider health.
- A full pose-output queue stalls the backend worker until the output consumer drains it;
  `pose_output_stalls` increments once when the worker enters each such stall. The condition
  is non-fatal, does not block input callbacks, and does not discard an already-computed pose.
  Continued pressure can instead fill the input queues and produce normal backpressure drops.

Queue defaults are deployment choices based on measured device and backend behavior.

### Provider lifecycle and estimates — `IPoseProvider`

```cpp
namespace core {

struct AppliedCaptureSpec;
struct RigCalibration;
struct SlamInstanceConfig;

// Sensor combinations accepted for one provider run.
enum class ProviderInputMode { Stereo, StereoImu };

// Canonical pinhole distortion models negotiated with a backend.
enum class PinholeDistortionModel {
    None,
    RadTan5,
    Rational8,
    ThinPrism12,
    Tilted14,
};

// Provider tracking state for one output estimate.
enum class TrackingState {
    Initializing,
    Tracking,
    Lost,
    Relocalizing,
};

// One pose estimate and the source measurement used to produce it.
struct PoseEstimate {
    Pose pose;                          // existing core::Pose; T_slamworld_body
    TrackingState state;
    CaptureGroupId source_capture_group_id;
    int64_t sample_time_local_ns;       // source image group's reference time
    int64_t sample_time_device_ns;      // source image group's device time
};

// Supported input forms; vector order has no meaning.
struct ProviderCapabilities {
    std::vector<ProviderInputMode> input_modes;
    std::vector<PixelFormat> pixel_formats;
    std::vector<PinholeDistortionModel> distortion_models;
};

// Runtime identity of the selected backend implementation.
struct ProviderIdentity {
    std::string backend_name;
    std::string backend_version;
};

// Aggregated admission, processing, and fatal-error health.
struct ProviderHealth {
    uint64_t dropped_image_groups;
    uint64_t dropped_imu_samples;
    uint64_t imu_gap_events;
    uint64_t rejected_invalid_input;
    uint64_t rejected_not_running;
    uint64_t pose_output_stalls;
    bool fatal_error; // once true, remains true for the run
};

// Backend-neutral Pose Provider lifecycle and output interface.
class IPoseProvider : public ISensorSink {
public:
    virtual ~IPoseProvider() = default;

    // Valid before initialize().
    virtual ProviderCapabilities capabilities() const = 0;
    virtual ProviderIdentity identity() const = 0;

    // Called after generic validation. Empty string means valid.
    virtual std::string validate_configuration(const SlamInstanceConfig& config,
                                               const AppliedCaptureSpec& capture,
                                               const RigCalibration& calibration) const = 0;

    // Normal lifecycle:
    // Uninitialized -> Initialized -> Running -> Draining -> Stopped.
    virtual void initialize(const SlamInstanceConfig& config,
                            const AppliedCaptureSpec& capture,
                            const RigCalibration& calibration) = 0;
    virtual void start() = 0;
    virtual void finish_input() = 0;
    virtual bool input_drained() const = 0;
    virtual void stop() = 0;

    // Legal while Running or Draining and for the final drain after input_drained().
    // Removes one estimate in production order. The integration owner drains until false
    // on each non-blocking tick; this is not a consumer-facing call.
    virtual bool try_get_next_pose(PoseEstimate& out) = 0;
    virtual ProviderHealth health() const = 0;
};

} // namespace core
```

Each `PoseEstimate` copies the identity and measurement timestamps of its originating
`ImageGroup`. The recorded group metadata resolves that identity to the exact source images
and sequences, avoiding duplicate frame-level source details in every estimate. The group ID
remains internal because the existing `se3_tracker` schema does not carry it.

`ProviderCapabilities` advertises only input modes, pixel formats, and distortion models the
backend accepts without approximation; each advertised list is non-empty. Generic startup
validation checks membership without assigning meaning to list order, then runs
backend-specific validation.

#### Lifecycle

The session integration owns the provider lifecycle:

1. **Configure.** Resolve the instance configuration, applied capture description,
   calibration, and backend configuration; inspect capabilities and identity; then validate
   and record that immutable setup.
2. **Start.** Arm recording, initialize and start the provider, and only then begin sensor
   delivery.
3. **Run.** Enqueue prepared measurements, drain all ready estimates without blocking, and
   monitor health. A fatal provider error fails the data-collection session.
4. **Finish.** Stop new sensor delivery, flush the capture adapter's buffered input, call
   `finish_input()`, and continue draining until `input_drained()` is true and no estimate
   remains. Then call `stop()` and finalize the recording.

`input_drained()` means every accepted input has been processed and no additional estimate
can be produced. Output must continue to drain while finishing so a bounded output queue
cannot stall shutdown. Repeating `finish_input()` or `stop()` after the first successful
call has no additional effect; `stop()` also supports abort and partial-startup cleanup.

#### Stable map frame

One provider run uses one fixed `SlamWorld` frame. Relocalization must return to that frame;
a backend that cannot preserve it remains `Lost` or fails the session rather than silently
creating a new origin. Starting a new map requires a new provider run and recording episode.

## Provider setup and calibration

### Instance configuration

```cpp
namespace core {

// One image stream as it will actually be delivered to the provider.
struct AppliedImageStreamSpec {
    SensorId sensor_id;
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    PixelFormat format;
    double rate_hz;
};

// Optional IMU stream as it will actually be delivered to the provider.
struct AppliedImuStreamSpec {
    SensorId sensor_id;
    double rate_hz;
};

// Immutable description of all capture streams for one provider run.
struct AppliedCaptureSpec {
    AppliedImageStreamSpec left;
    AppliedImageStreamSpec right;
    std::optional<AppliedImuStreamSpec> imu;
    ClockDomainId clock_domain_id;
};

// Configured sensor topology and output body frame.
struct StereoRigSpec {
    RigId rig_id;
    FrameId body_frame_id;
    SensorId left_sensor_id;
    SensorId right_sensor_id;
    std::optional<SensorId> imu_sensor_id;
    ClockDomainId clock_domain_id;
};

// Selects the published pose frame and optional fixed Session alignment.
struct OutputSpec {
    enum class Frame { SlamWorld, Session };

    Frame output_frame{Frame::SlamWorld};
    std::optional<Pose> T_session_slamworld;
};

// Finite capacities for provider input and output queues.
struct QueueConfig {
    size_t max_image_groups;
    size_t max_imu_samples;
    size_t max_pose_estimates;
};

// User-configured setup for one Pose Provider instance.
struct SlamInstanceConfig {
    std::string instance_id;
    ProviderInputMode input_mode;
    StereoRigSpec rig;
    OutputSpec output;
    QueueConfig queues;
    std::string backend_config_path;
};

} // namespace core
```

`SlamInstanceConfig`, the applied capture description, and the resolved calibration form one
immutable setup. The same records are validated, passed to `initialize()`, and stored with
the recording.

Before `initialize()`, the session performs generic startup validation across this complete
setup. The provider then applies backend-specific restrictions through
`validate_configuration()`. Generic validation requires:

- Identifiers that satisfy the rules below, with distinct left and right sensors.
- Agreement among the selected input mode, rig topology, optional IMU, applied streams,
  resolved calibration, and advertised provider capabilities.
- Valid applied stream geometry, format, rate, and clock domain, with matching canonical
  calibration.
- A present, finite rigid `T_session_slamworld` when `Session` output is selected, and no
  `T_session_slamworld` when `SlamWorld` output is selected.
- Positive `max_image_groups` and `max_pose_estimates`; positive `max_imu_samples` in
  `StereoImu` mode. `max_imu_samples` is ignored in `Stereo` mode.

A fixed `T_session_slamworld` has no automatic compatibility identifier. It must be
re-established whenever the backend map-frame convention or repeatable physical start
relationship changes. Startup validates its form, not its physical correctness. The
recording retains the resolved setup, provider identity, backend configuration, and
application-build information for later audit. The recorded backend configuration is the
resolved content used for the run, not only its machine-local source path.

### Identifier semantics

These identifiers connect configuration, capture, calibration, recording, and pose output
for one provider run:

| Identifier | Assigned by | Scope and lifetime | Purpose |
|---|---|---|---|
| `instance_id` | Session configuration | Unique among active providers in one runtime; stable for the run | Names the provider and derives its transport collection |
| `collection_id` | Derived from `instance_id` | `"pose/" + instance_id`; stable for the run | Shared stream name used by publication, recording setup, and consumers |
| `rig_id` | Rig configuration and calibration | Stable for the physical rig | Prevents applying calibration for another rig |
| `sensor_id` | Capture configuration and calibration | Unique within the rig; stable for the run | Connects measurements and applied streams to calibration |
| `body_frame_id` | Rig configuration and calibration | One fixed rigid frame for the run | Defines the body frame of the reported pose and sensor extrinsics |
| `clock_domain_id` | Capture integration | Stable for one raw clock domain | Identifies timestamps that can be fusion-ordered together |
| `capture_group_id` | Capture integration | Unique within one recording episode | Identifies one atomic image measurement across recording and estimation |
| `source_capture_group_id` | Pose Provider | One value per estimate, copied from its image group | Links an internal pose estimate back to its recorded image group |

#### Pose transport naming

`instance_id` is configured rather than generated so independently launched components can
derive the same transport name. It is unique among active providers in one runtime, begins
with an ASCII letter or digit, and otherwise uses only ASCII letters, digits, `.`, `_`, and
`-`.

The collection name is not separately configurable:

```cpp
std::string make_pose_collection_id(std::string_view instance_id);
// Example: make_pose_collection_id("oak_front") == "pose/oak_front"
```

The session, live consumers, and recording setup derive the same value. Startup rejects a
derived name longer than `XR_MAX_TENSOR_IDENTIFIER_SIZE - 1` bytes—255 bytes with the
current OpenXR extension header, excluding the terminating null byte.

### Frames and calibration

Before provider initialization, the session resolves one `RigCalibration` from the capture
self-report and any file override. An override replaces only explicitly supplied fields and
takes precedence over the self-report; any required field still missing after the merge fails
startup. The result describes the exact images and sample timing delivered to the provider
after capture-side processing. Device loaders and backend adapters are the only points that
convert vendor-specific conventions.

#### Frame conventions

**Frames.** Named frames in prose use CamelCase, while transform subscripts use lowercase
tokens.

| Frame | Meaning | Transform token |
|---|---|---|
| `Body` | The configured rigid frame whose pose is reported; selected by the matching `body_frame_id` in `StereoRigSpec` and `RigCalibration` | `body` |
| `SlamWorld` | The backend's fixed map frame for the run | `slamworld` |
| `Session` | Optional fixed output frame | `session` |

For the initial OAK-D integration, the body may be the physical left or center/RGB optical
frame when calibration supplies every required transform. A virtual stereo midpoint is
allowed only through a calibration override that defines the body frame and all sensor
extrinsics; the integration does not synthesize it. The body need not correspond to a
delivered stream.

**Transforms.** All frames are right-handed. `T_A_B` transforms coordinates from frame B
into frame A; 4x4 matrices are row-major homogeneous transforms in meters. `core::Pose` uses
meters and a unit Hamilton quaternion ordered `(x,y,z,w)`, matching the existing
`se3_tracker` schema.

| Transform | Meaning |
|---|---|
| `T_body_left`, `T_body_right`, and optional `T_body_imu` | Fixed extrinsics from each delivered sensor frame into `Body` |
| `T_slamworld_body` | Body pose in `SlamWorld`; stored in `PoseEstimate::pose` before optional Session alignment |
| `T_session_slamworld` | Optional fixed alignment producing `T_session_body = T_session_slamworld * T_slamworld_body` |

Changing `body_frame_id` changes the fixed conversion used for the reported body pose; it
does not redefine `SlamWorld`. The backend-native conversion is described next, and the
fixed Session-alignment validity rules remain under Instance configuration.

#### Calibration-to-output pose flow

Resolved calibration affects pose output in two ways:

1. Camera intrinsics, distortion, stereo extrinsics, and optional IMU calibration configure
   the estimator.
2. A fixed extrinsic converts the backend's native pose into the configured `Body` frame.

During initialization, each backend adapter declares the rigid frame in which its backend
reports pose. This may be a calibrated sensor frame or another backend frame with a known
fixed relation to the rig. The adapter resolves one immutable `T_body_native` from the final
calibration and backend configuration; initialization fails if that transform is ambiguous
or unavailable.

The output path is:

```text
resolved RigCalibration
    ├── estimator parameters ──▶ backend
    │                               │
    │                               ▼
    │                    T_slamworld_native
    │                               │
    └── T_body_native ──────────────┤  compose with inverse(T_body_native)
                                    ▼
                             T_slamworld_body
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
             SlamWorld output             apply T_session_slamworld
             T_slamworld_body                      │
                                                   ▼
                                           Session output
                                           T_session_body
```

The resulting provider pose is:

```text
T_slamworld_body = T_slamworld_native * inverse(T_body_native)
```

The inverse is required because calibration stores the transform from the backend-native
frame into `Body`, while pose composition needs the opposite direction. Other calibration
fields have already affected the estimate through backend initialization.

For example, when the OAK-D center/RGB optical frame is the configured body and a backend
reports the delivered left-camera pose:

```text
T_slamworld_body = T_slamworld_left * inverse(T_body_left)
```

`PoseEstimate::pose` always contains `T_slamworld_body`. Pose publication applies the optional
fixed `Session` alignment when that output frame is selected.

#### Initial calibration support

The initial release accepts one canonical `RigCalibration` with:

- Two pinhole cameras and, for `StereoImu`, one IMU.
- Camera parameters for the delivered pixel grids, preserving every coefficient of the
  declared distortion model without truncation or guessing.
- IMU extrinsics, timing offset, and noise parameters when IMU input is enabled.
- Sensor identities and stream properties that match the applied capture description.

A backend may advertise a supported subset of the canonical pinhole models and impose
tighter constraints through `validate_configuration()`. Exact records, coefficient ordering,
normalization, and validation rules are in the
[Canonical calibration reference](#appendix-canonical-calibration-reference).

## Backend selection and reference roles

One build contains exactly one provider backend: ORB-SLAM3, cuVSLAM, or the stub. An unknown
selection fails explicitly; there is no fallback. Every backend reports its name, library
version, and supported capabilities through the common API. The recording stores that
identity and the IsaacTeleop application build separately.

| Backend | Role | Boundary |
|---|---|---|
| [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) | Proven OAK-D stereo and stereo-inertial baseline for development validation | Developer-supplied under its GPLv3 or commercial terms; not vendored, downloaded, or included in normal release CI/artifacts. Shipping requires an approved commercial license or GPL-compliant distribution plan. |
| [cuVSLAM](https://github.com/nvidia-isaac/cuVSLAM) | Intended production backend | The selected SDK version must be qualified for its advertised modes, calibration conversion, redistribution terms, CUDA requirements, and target platforms. |
| `Stub` | Deterministic conformance backend with no external dependency | Supports the full initial contract and exercises trajectories, tracking states, source matching, draining, gaps, backpressure, and map-frame invariants. |

Each adapter advertises only the input modes, pixel formats, and distortion models it can
accept without approximation, and converts the canonical calibration and frame convention
to its backend API.

## Pose publication and timing

### Existing transport mapping

`DataCollectionSession` drains ready estimates and maps them to the existing
[`se3_tracker` schema](../../src/core/schema/fbs/se3_tracker.fbs). This integration does not
change the schema or recorded channel names.

Conceptually, the existing transport contains these small records:

```text
Se3TrackerPose        { pose: position + orientation, is_valid }
Se3TrackerPoseTracked { data: Se3TrackerPose or no sample }       // live API
Se3TrackerPoseRecord  { data: Se3TrackerPose, timestamp }         // recording
```

Position is in meters and orientation is a quaternion in `(x, y, z, w)` order. The recorded
timestamp contains availability time, local measurement time, and raw device measurement
time. The producer defines the pose's reference frame. The transport does not carry a
detailed tracking state, source image-group ID, or map version.

| Provider result | Published behavior |
|---|---|
| `pose` | Apply the selected output-frame transform described in [Calibration-to-output pose flow](#calibration-to-output-pose-flow). |
| `state` | Set `is_valid=true` only for `Tracking`. Other states publish `is_valid=false`; the filler pose is unspecified. The transport does not expose the detailed state. |
| Measurement timestamps | Publish the source image group's device and local measurement times, not the time at which SLAM completes or publication occurs. |
| `source_capture_group_id` | Keep it available internally for diagnostics. The current transport cannot carry it, so recorded source data and poses are correlated by measurement timestamp. |

The pose collection name is derived from `instance_id` as described in
[Pose transport naming](#pose-transport-naming). Only fresh backend estimates are published:
the integration neither repeats an old pose nor fabricates an invalid estimate when no
output is available. Missing output therefore remains a visible gap in the pose stream.

### Measurement and availability time

This subsection defines the timestamp and synchronization guarantees visible to the Pose
Provider; it does not define how capture establishes those guarantees.

The following three times serve different purposes and must not be mixed up:

| Time concept | Provider field | `SchemaPusher` argument | Recorded `DeviceDataTimestamp` field |
|---|---|---|---|
| Raw device measurement time | `PoseEstimate::sample_time_device_ns` | `sample_time_raw_device_clock_ns` | `sample_time_raw_device_clock` |
| Local common measurement time | `PoseEstimate::sample_time_local_ns` | `sample_time_local_common_clock_ns` | `sample_time_local_common_clock` |
| Availability time | No provider field | No argument | `available_time_local_common_clock` |

`ImageFrame`, `ImageGroup`, and `ImuSample` use the same provider-side measurement-time
field names as `PoseEstimate`.

Capture supplies the two measurement times, and the Pose Provider preserves them even though
it produces the result later. Camera fusion time uses raw device measurement time directly;
IMU fusion time applies the recorded calibration offset without overwriting the raw
timestamp. Local common measurement time uses the host monotonic timeline to align the pose
with other recorded streams.

The receiving runtime sets availability time when the published pose reaches it and becomes
available to a live consumer or recorder. This is not the time when the backend finishes
computing the pose or when a recorder writes it to disk. Because availability time and local
measurement time use the same host clock, their difference measures pose availability
latency:

```text
pose availability latency = available_time_local_common_clock
                           - sample_time_local_common_clock
```

For accepted image group N:

```text
recorded image-group metadata: group N, group device time T, group local time L
provider input:                group N, group device time T, group local time L
provider pose output:          source group N, sample device time T, sample local time L
runtime receives pose:         availability time A
recorded pose timestamp:       sample device T, sample local L, availability A
```

`T` and `L` are the group measurement times selected by capture's declared grouping rule.
Initial OAK-D frames share the same device timestamp; other stereo integrations may retain
different per-frame timestamps.

When IMU is enabled, each sample retains its own device and local measurement times. The
pose remains stamped at the source image group's time; `time_offset_ns` relates the IMU
timeline to it without changing the recorded raw timestamps.

To make timestamp correlation unambiguous, accepted image groups have strictly increasing
device timestamps. A duplicate is rejected without ending the episode; a timestamp
regression fails the episode because the source timeline is no longer trustworthy. Unknown
or mismatched clock domains are rejected before provider initialization. Clock mapping,
camera pairing, skew estimation, and drift correction remain capture responsibilities.

## Telemetry and diagnostics

Telemetry here means the health and performance data needed to decide whether a collection
run is usable and to explain missing or delayed poses. It stays separate from pose output,
must not block capture or estimation, and does not change the existing `se3_tracker` schema.

| Source | Minimum data | What it explains |
|---|---|---|
| Existing pose stream | Every emitted pose is recorded once with `is_valid`, measurement timestamps, and availability time | Primary pose output and the source for counts, gaps, valid-tracking periods, and pose availability latency; telemetry does not create a duplicate per-pose stream |
| `ProviderHealth` | Dropped image groups and IMU samples, IMU gaps, rejected input, pose-output stalls, and fatal-error state | Provider input loss, queue pressure, invalid calls or data, and provider failure |
| Capture and session integration | Measurements offered, recorded, and accepted; fusion-time ordering bounds, forced emissions, and late input; timestamp violations; recorder pressure or failure; final session state and failure reason | Whether recording and estimation saw the same source data and where a run failed |

`ProviderHealth` counters start at zero for each provider run, increase until final drain,
and never reset during that run. `health()` is non-blocking and safe for the one documented
output/health consumer while input and backend processing run concurrently; additional
concurrent callers are not supported. `fatal_error` stays true after the first fatal provider
error. The session derives its own input and output totals from enqueue results and drained
estimates rather than duplicating those totals in the provider API.

The initial OAK-D integration makes current health available for live inspection and records
the final provider, capture, and session summary with the episode. That summary is stored
with the applied capture and calibration, provider identity, backend configuration, and
IsaacTeleop application build already required as episode setup. The pose records themselves
remain the time-based source for validity, gaps, and latency analysis.

The exact encoding of the final telemetry summary is an implementation detail of the
existing episode metadata path; this design defines its required content, not a new
telemetry schema.

A future need for high-rate queue depth, detailed tracking-state transitions, or
backend-specific metrics may add a separate time-series telemetry schema; those fields must
not become requirements for all Pose Provider backends.

## Integration and verification

Verification is divided by ownership so provider behavior, capture integration, physical
hardware, and backend-specific behavior can be tested independently.

### Provider conformance

The deterministic stub is the common contract-test target. The same tests run against each
enabled backend where applicable.

| Area | Required outcome |
|---|---|
| Lifecycle | Only documented transitions succeed; input is rejected outside the running interval; final draining completes without deadlock or loss of an already-produced estimate. |
| Configuration and input validation | Unsupported capabilities, malformed image groups, mismatched clocks or geometry, and invalid calibration fail before backend ingestion. Supported calibration is preserved without truncation or approximation. |
| Concurrency and backpressure | One producer and one output consumer operate without races. Queues stay bounded, image groups remain atomic, individual IMU drops and gap episodes are counted as specified, and pose-output pressure neither blocks capture callbacks nor discards an already produced pose. |
| Pose semantics | Estimates preserve their source group and measurement timestamps. A missing estimate produces no repeated or fabricated pose, and one run never silently changes its `SlamWorld` frame. |
| Frames | Known trajectories verify body and optional session transforms; invalid frame configurations fail at startup. |
| Provider health | Health counters are cumulative, pose-output stall transitions are counted once, the fatal flag stays set, and non-blocking snapshots match the injected outcomes. |

### Capture and recording integration

Run a common integration suite using deterministic recorded or synthetic sensor fixtures.
Apply it to the initial OAK-D integration and future implementations of the
capture-integration requirements.

- Verify recorder acceptance precedes provider delivery; rejected measurements never reach
  the provider.
- Verify the same applied capture description and calibration are used for fusion-time
  ordering, provider initialization, and episode metadata.
- Verify grouping and timestamp rules before backend ingestion, including duplicate,
  regressed, or late input, while preserving raw measurement timestamps.
- Exercise the applied fusion-time ordering window and item capacity; verify emitting the
  union minimum under capacity pressure guarantees progress without discarding an on-time
  incoming minimum, and the applied bounds and counters are recorded.
- Verify every provider input is present in the resulting episode and every emitted pose
  correlates with exactly one recorded image group.
- Verify pose publication preserves both measurement times and the receiving runtime adds
  availability time.
- Verify coordinated shutdown processes every accepted input, preserves every produced
  estimate, and records session totals and any counted loss or failure consistently.
- Verify episode metadata reconstructs the configured provider setup.
- Verify published poses through `Se3Tracker` and recorded poses through the existing replay
  API, using the derived collection name and unchanged sensor channels.

This suite runs routinely in CI and does not require physical hardware.

### Initial OAK-D hardware qualification

Run live OAK-D sessions in both `Stereo` and `StereoImu` modes:

- Verify one process owns the device and recording pipeline; no pose component reopens it.
- Verify `OakCamera` reports the actual applied stream configuration and calibration.
- Verify synchronized stereo frames have equal, strictly increasing device timestamps.
- Verify an IMU `time_offset_ns` override takes precedence over the self-report, and that it
  and any crop, resize, or rectification are reflected in the applied calibration and recorded
  episode.
- Run a live smoke test of recording, pose publication, coordinated shutdown, and subsequent
  replay.

These checks require physical hardware and are not required for every CI run.

### Backend qualification

- Compare the ORB-SLAM3 adapter with the proven OAK-D reference pipeline using the same
  recorded sensor data.
- Qualify cuVSLAM independently for every input mode it advertises.
- Verify each backend's calibration and frame conversions with known transforms and
  trajectories.
- Inject IMU drop episodes and verify each inertial adapter receives the internal gap
  notification before the next accepted sample, then verify backend-specific tracking-state
  changes and recovery.
- Verify release builds and artifacts exclude ORB-SLAM3 unless a separate licensing
  approval exists.

## Implementation impact

Implementation adds a core Pose Provider module and backend adapters, and evolves the
existing OAK plugin for the initial OAK-D integration. Exact file placement, build targets,
and dependency wiring belong in the implementation plan.

| Area | Implementation effect |
|---|---|
| Core Pose Provider | Add the interfaces, normalized sensor types, configuration, calibration, validation, lifecycle, and health defined in this document. |
| Backend adapters | Add the deterministic stub, production cuVSLAM adapter, and developer-supplied ORB-SLAM3 reference adapter. Each implements the same provider contract and preserves the licensing boundaries above. |
| Existing pose transport | Reuse `SchemaPusher`, `Se3Tracker`, recording, and replay without changing their schemas or consumer APIs. |

### Initial OAK-D integration

| Component | Change | Responsibility |
|---|---|---|
| `OakCamera` | Evolve existing component | Remain the sole device and DepthAI-pipeline owner, and report the applied capture configuration and device calibration without depending on Pose Provider types. |
| `SensorAdapter` | New component | Convert OAK-D capture output into provider input, construct `AppliedCaptureSpec`, apply bounded fusion-time ordering, and deliver accepted measurements. |
| `DataCollectionSession` | New integration owner | Coordinate recording, resolved calibration, provider lifecycle, pose publication, telemetry, output draining, and shutdown. |

The initial OAK-D integration remains in the OAK plugin process so the camera is opened only
once. Internal class or file organization may change as long as these ownership boundaries
remain intact.

## Appendix: Canonical calibration reference

This appendix defines the normalized calibration passed to every backend. Capture-specific
loaders resolve it before provider initialization; backends do not interpret vendor-native
calibration records directly.

### Calibration types

```cpp
namespace core {

// Returns nullopt for an unknown/deserialized enum value.
std::optional<size_t> distortion_coefficient_count(PinholeDistortionModel model);

struct PinholeCameraModel {
    double fx;
    double fy;
    double cx;
    double cy;
    PinholeDistortionModel distortion_model;
    std::vector<double> distortion_coefficients;
};

struct CameraCalibration {
    SensorId sensor_id;
    uint32_t width;
    uint32_t height;
    PinholeCameraModel model;
    std::array<double, 16> T_body_camera;
};

struct ImuCalibration {
    SensorId sensor_id;
    double rate_hz;
    std::array<double, 16> T_body_imu;
    double accel_noise_density;
    double gyro_noise_density;
    double accel_random_walk;
    double gyro_random_walk;
    // Sign convention: t_camera = t_imu + time_offset_ns.
    int64_t time_offset_ns;
};

struct RigCalibration {
    RigId rig_id;
    FrameId body_frame_id;
    CameraCalibration left;
    CameraCalibration right;
    std::optional<ImuCalibration> imu;
};

} // namespace core
```

### Supported distortion models

The initial release uses the pinhole projection family. Coefficients follow the cumulative
OpenCV/DepthAI Perspective ordering:

| Distortion model | Count | Coefficient order |
|---|---:|---|
| `None` | 0 | Empty |
| `RadTan5` | 5 | `k1, k2, p1, p2, k3` |
| `Rational8` | 8 | `k1, k2, p1, p2, k3, k4, k5, k6` |
| `ThinPrism12` | 12 | Rational8 followed by `s1, s2, s3, s4` |
| `Tilted14` | 14 | ThinPrism12 followed by `tau_x, tau_y` |

Enum values are semantic tags, not encoded counts. Configuration serializes their symbolic
names, and `distortion_coefficient_count()` plus this table define the enum-to-length
mapping.

### Normalization and validation

- Calibration dimensions, sensor identities, and rates must match `AppliedCaptureSpec`.
  Intrinsics and distortion describe the delivered pixel grid after capture-side crop,
  resize, and rectification. A distortion-free rectified grid uses `None` with no
  coefficients, and its extrinsics must follow the same rectification convention.
- Generic validation requires positive finite `fx` and `fy`, finite principal-point and
  coefficient values, and the coefficient count declared by the model. It rejects unknown
  enum values and never guesses or rescales calibration.
- A source explicitly tagged as OpenCV/DepthAI Perspective with four coefficients is
  normalized to `RadTan5` by appending `k3 = 0`. An untagged four-element list is rejected;
  coefficient count alone cannot identify a projection convention. DepthAI `Perspective`
  maps to the matching pinhole model without truncation. Fisheye and other projection
  families are deferred.
- Startup rejects a distortion model not advertised by the selected backend. A backend may
  impose tighter constraints through `validate_configuration()`, but its adapter must not
  silently discard coefficients.
- `RigCalibration::imu` is present in `StereoImu` mode and absent in `Stereo` mode. When
  present, its sensor ID and rate match `AppliedCaptureSpec::imu`; a backend may impose
  tighter noise, rate, or stream constraints.
- Initial OAK-D input is treated as global shutter; rolling-shutter timing models are
  deferred. Changing rectification or `body_frame_id` requires newly resolved calibration
  and a new provider run.
- The complete resolved calibration, including any file override, is recorded with the
  data-collection episode.
