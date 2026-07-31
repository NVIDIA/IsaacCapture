<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Pose Provider — Interface Design

## Context

IsaacTeleop needs 6-DoF pose from cameras that do not emit a pose through the existing
teleop device path. A Pose Provider consumes synchronized camera and optional IMU
measurements, runs a SLAM/VIO backend, and republishes the estimate through the existing
pose transport.

The initial hardware target is an OAK-D stereo camera. We already have a working data
collection pipeline using OAK-D with ORB-SLAM3; that pipeline is the functional reference
for input modes and expected behavior. cuVSLAM is the intended production reference
backend.

This document designs the Pose Provider boundary and its integration requirements. It does
not design the camera capture interface, the vendor pipeline, or the recording subsystem.

## Goals

- Define a common Pose Provider contract that supports interchangeable SLAM/VIO backends and
  produces one 6-DoF pose stream from synchronized camera and optional IMU measurements.
- Preserve measurement timing and provenance so pose output remains correlatable with
  recorded sensor data.
- Define consistent input, calibration, frame, lifecycle, and health contracts across
  backends, including a stable pose frame for each run.
- Reuse the existing pose transport, recording, replay, and consumer APIs.
- Keep capture, estimation, and transport responsibilities separate so the design can grow
  without coupling the core provider contract to vendor-specific mechanisms.

## Initial release scope

The initial release has one supported input topology and two input modes:

| Topology | Input mode | Pose output |
|---|---|---|
| One OAK-D stereo pair | `Stereo` | One pose stream |
| One OAK-D stereo pair + its onboard IMU | `StereoImu` | One pose stream |

Normative initial-release constraints:

- The supported device is one physical OAK-D providing one stereo pair and, optionally, its
  onboard IMU. It forms one rigid rig and one raw device-clock domain.
- The existing OAK integration remains the device owner. One process coordinates capture,
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

## Deferred capabilities and extension directions

These capabilities are related to the Pose Provider but are not initial-release requirements
or delivery commitments. They may require interface changes or separate designs:

1. **Capture-interface migration.** Replace the direct OAK integration after a generic
   capture interface can provide the required synchronization, calibration, provenance,
   recording fan-out, and lifecycle outcomes.
2. **Input expansion.** Add other devices, mono and larger camera groups, multi-adapter or
   cross-clock rigs, additional camera and timing models, and accelerated buffer transport.
3. **SLAM and map behavior.** Add operating modes such as localization against an existing
   map, map reset or dynamic alignment, and collaborative/shared-map operation.
4. **Consumption and output.** Add forward prediction, real-time control and safety
   contracts, or a richer pose schema when a concrete consumer requires them.

The initial release stabilizes only the seams it exercises; these directions are not
compatibility promises.

## Design boundaries and non-goals

These are responsibilities this design does not own, including when a deferred capability
eventually uses them:

- Designing camera or IMU capture APIs, capture fan-out, vendor synchronization, or clock
  mapping mechanisms.
- Defining sensor-recording infrastructure or file formats.
- Defining backend algorithms or backend-native configuration formats.
- Defining unrelated camera consumers, cross-machine transport, or collaborative-map
  protocols. Those components may integrate with the Pose Provider through separate designs.

## Architecture and ownership boundary

```text
Generic responsibility                                                    Initial OAK-D realization
──────────────────────                                                    ─────────────────────────
Session integration owner ──────────────────────────────────────────────▶ plugins::oak::DataCollectionSession
        ├── Capture/device owner ───────────────────────────────────────▶ OakCamera
        ├── Sensor recording ───────────────────────────────────────────▶ DataCollectionSession-owned recorder
        └── Capture adapter ────────────────────────────────────────────▶ SensorAdapter
                    │
         normalized, fusion-ordered,
     owned image groups + optional IMU
                    │
                    ▼
             IPoseProvider implementation ──────────────────────────────▶ ORB-SLAM3 / cuVSLAM / stub adapter
                              │
                         PoseEstimate
                              │
                              ▼
          Output drain and frame transform ─────────────────────────────▶ DataCollectionSession update loop
                              │
                              ▼
              Existing pose transport ──────────────────────────────────▶ SchemaPusher / Se3Tracker
                                                                          live + record/replay consumers
```

The left column defines architectural responsibilities; the right column shows their initial
OAK-D realization. Horizontal arrows mean “realized by,” while vertical and branch connectors
show runtime flow and coordination. Concrete names in the right column are not part of the
generic provider contract.

There are three independent seams:

1. **Capture seam.** The capture layer owns devices, synchronization, sensor recording, and
   fan-out. `SensorAdapter` translates its output into the normalized provider types below.
2. **Estimator seam.** `IPoseProvider` owns SLAM/VIO state and backend-native translation.
3. **Transport seam.** Existing `se3_tracker` schema, consumer API, recording, and replay
   remain unchanged; live consumers use the collection name derived from `instance_id`.

For the initial OAK-D integration, the existing OAK capture process remains the single
device-owning process. `OakCamera` owns the device and DepthAI pipeline;
`plugins::oak::DataCollectionSession` owns their integration lifecycle and non-blocking
update loop. Recording and pose estimation branch from that one pipeline. A separate pose
process must not reopen the same device.

### Architecture scope inventory

The top-level types and components designed here are:

- **Core interfaces:** `ISensorSink` and `IPoseProvider`.
- **Input and identity types:** `SensorId`, `RigId`, `FrameId`, `ClockDomainId`,
  `CaptureGroupId`, `PixelFormat`, `ImageFrame`, `ImageGroup`, `ImuSample`, and
  `EnqueueResult`.
- **Provider mode, output, and status types:** `ProviderInputMode`,
  `PinholeDistortionModel`, `TrackingState`, `SourceImageRef`, `PoseEstimate`,
  `ProviderCapabilities`, `ProviderIdentity`, and `ProviderHealth`.
- **Applied configuration types:** `AppliedImageStreamSpec`, `AppliedImuStreamSpec`,
  `AppliedCaptureSpec`, `StereoRigSpec`, `OutputSpec`, `QueueConfig`, and
  `SlamInstanceConfig`.
- **Calibration types:** `PinholeCameraModel`, `CameraCalibration`, `ImuCalibration`, and
  `RigCalibration`.
- **Initial integration components:** `plugins::oak::SensorAdapter` and the Pose
  Provider-related orchestration in `plugins::oak::DataCollectionSession`, plus the stub,
  ORB-SLAM3, and cuVSLAM implementations of `IPoseProvider`.

`SensorAdapter` is in scope only as the initial OAK-specific provider-input adapter: this
design defines its provider-facing normalization, applied-descriptor construction, fusion
ordering, and delivery and lifecycle obligations. It does not own device access, vendor
synchronization, capture APIs, or recording infrastructure. `OakCamera`, the recorder,
`SchemaPusher`, and `Se3Tracker` are existing or separately designed dependencies rather than
types designed here.

## Capture-integration requirements

The direct OAK integration and any future generic capture interface must provide the same
observable contract to the Pose Provider:

- **Synchronized input.** Deliver each image measurement as one atomic `ImageGroup`, with
  stable sensor and capture identities, source sequences, and device and local measurement
  timestamps. Optional IMU samples use the same clock domain. Initial OAK-D stereo frames
  have exactly equal device timestamps; future capture implementations may use another
  declared grouping rule. Host arrival time is never treated as measurement time.
- **Recording correlation.** The data-collection session records the same measurements it
  offers to the provider, including their identities and timestamps. Recorder acceptance
  precedes provider delivery; a measurement that recording cannot accept is not sent to the
  provider. A later recording failure fails the session rather than leaving an apparently
  valid pose stream without its source data. The provider remains independent of the
  recording implementation and carries source identity and measurement time into
  `PoseEstimate`.
- **Immutable applied setup.** Before provider initialization, the integration supplies one
  selected `ProviderInputMode`, one `AppliedCaptureSpec` describing the streams actually
  delivered, and one fully resolved `RigCalibration`. Calibration describes the delivered
  images after any capture-side processing and is established before measurement delivery
  begins. The same records are used for validation, provider initialization, fusion
  ordering, and episode metadata, and remain fixed for the run.

For the initial OAK-D path, `OakCamera` reports the applied capture-native stream properties
without depending on Pose Provider types. `plugins::oak::SensorAdapter` converts that report
into the provider-facing `AppliedCaptureSpec`, while the session orchestrator resolves the
final calibration once. Future capture-interface migration may change where the source
information comes from, but not these provider-facing outcomes or ownership boundaries.

## Interface 1 — provider input

```cpp
namespace core {

using SensorId = std::string;
using RigId = std::string;
using FrameId = std::string;
using ClockDomainId = std::string;
using CaptureGroupId = uint64_t;

enum class PixelFormat { Gray8, NV12, BGR888, RGBA8 };

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

struct ImageGroup {
    CaptureGroupId capture_group_id;
    int64_t sample_time_local_ns;    // capture-selected group reference time
    int64_t sample_time_device_ns;   // capture-selected group reference time
    std::vector<ImageFrame> images;  // owned frames; cardinality is mode-dependent
};

struct ImuSample {
    SensorId sensor_id;
    uint64_t source_sequence;
    double accel_mps2[3];
    double gyro_radps[3];
    int64_t sample_time_local_ns;
    int64_t sample_time_device_ns;
};

enum class EnqueueResult {
    Accepted,
    DroppedBackpressure,
    RejectedInvalidInput,
    RejectedNotRunning,
};

class ISensorSink {
public:
    virtual ~ISensorSink() = default;

    // The group is one indivisible queue item. Initial OAK-D requires:
    // - exactly two images
    // - one image for each initialized left/right sensor ID
    // - each image's device timestamp matches the group device timestamp
    virtual EnqueueResult on_image_group(ImageGroup group) = 0;

    // Called only in StereoImu mode.
    virtual EnqueueResult on_imu(const ImuSample& sample) = 0;
};

} // namespace core
```

### Input ownership, ordering, and concurrency

- `on_image_group()` transfers ownership of the group and all image buffers.
- `on_imu()` copies the small sample into the provider queue before returning.
- Calls are enqueue-only and do not execute SLAM compute.
- `ImageGroup::sample_time_local_ns` and `sample_time_device_ns` are the capture-selected
  reference measurement times for the group. The provider uses them for ordering and pose
  output without choosing or averaging frame timestamps; each `ImageFrame` retains its own
  source timestamps for recording and diagnostics.
- One `SensorAdapter` serializes all calls into an `ISensorSink` instance. Callbacks are not
  concurrent in the initial release.
- `SensorAdapter` owns a bounded fusion-ordering buffer. Its capture-side configuration has
  positive `fusion_reorder_window_ns` and `max_reorder_items` bounds; the applied values are
  recorded with the episode and selected from measured OAK-D behavior before release.
  `max_reorder_items` must be at least one greater than the maximum aggregate number of
  stereo-group and optional IMU items that the configured sensor rates can place within one
  full reorder window.
- Image-group fusion time is `ImageGroup::sample_time_device_ns`. IMU fusion time is
  `sample_time_device_ns + RigCalibration::imu.time_offset_ns`, following the convention
  `t_camera = t_imu + time_offset_ns`.
- The adjustment is an ordering/backend-input operation only. `ImuSample` and recorded
  metadata retain the original raw device timestamp.
- On every non-late arrival, the adapter advances `greatest_observed_fusion_time` before its
  admission decision, then computes `watermark = greatest_observed_fusion_time -
  fusion_reorder_window_ns`. It emits buffered items whose fusion time is at or before that
  watermark.
- Delivery into the provider is monotonic by fusion time; for equal fusion times, IMU
  precedes stereo and source sequence breaks same-type ties.
- A sample whose ordering key precedes an already emitted item is late. It is not delivered
  to the provider and increments capture health `late_fusion_samples`. Sensor data already
  accepted by the recorder remains in the episode and the drop is attributable to source
  identity.
- If admitting an item would exceed `max_reorder_items` after watermark-eligible emissions,
  the adapter immediately emits the lowest item from the union of the buffer and incoming
  item, ignoring the watermark, and retains the rest. This preserves monotonic delivery and
  forward progress while reducing effective reorder depth. Each such transition increments
  capture health `fusion_reorder_forced_emits`; no item is dropped by the capacity event.
- During normal shutdown, the adapter flushes its remaining items in order before becoming
  quiescent. These capture-side outcomes do not add an `EnqueueResult` because rejected
  items never reach `ISensorSink`.
- Calls are legal only after `IPoseProvider::start()` returns and before the capture adapter
  becomes quiescent during shutdown.
- No sensor callback may arrive after `IPoseProvider::finish_input()` begins.
- The input-callback thread may run concurrently with one output/observer thread calling
  `try_get_next_pose()`, `health()`, and `input_drained()`. Implementations must be safe for
  that one-producer/one-consumer pattern. Lifecycle calls remain serialized: `finish_input()`
  follows adapter quiescence, and `stop()` follows the final output drain.

### Bounded backpressure

`SlamInstanceConfig::queues` sets finite capacities for stereo input, IMU input, and pose
output.

- A full stereo queue drops the incoming `ImageGroup` as one unit and returns
  `DroppedBackpressure`. It never splits a group.
- A full IMU queue returns `DroppedBackpressure`, increments the IMU-drop counter, and
  records one IMU-gap event. The wrapper sets an implementation-internal pending-gap marker;
  the next successfully queued IMU item carries that marker to the worker. A
  source-sequence discontinuity marks the affected accepted item directly.
- Before ingesting a marked IMU item, the worker invokes the backend adapter's internal
  gap-notification entry point. Every backend adapter must implement that entry point;
  `ISensorSink` is unchanged. If no later IMU item is accepted, health still records the gap
  and no backend notification is needed because there is no post-gap sample to qualify.
  Backend-specific configuration determines whether tracking remains valid and what
  recovery requires. The generic wrapper does not override `TrackingState`.
- A full pose-output queue stalls the backend worker until the consumer drains it.
  Already-computed poses are never dropped. The stall increments `pose_output_stalls` once
  when the worker begins waiting and does not set `fatal_error`.
- Output pressure never blocks `on_image_group()` or `on_imu()`. While the worker is stalled,
  bounded input queues may fill and their normal `DroppedBackpressure` outcomes apply to
  complete stereo groups and/or IMU samples.
- Structurally invalid input is rejected before backend ingestion and returns
  `RejectedInvalidInput`.
- All drops, invalid inputs, and rejected-state calls are counted and exposed through
  `ProviderHealth`.

`imu_gap_events` increments once when a local IMU drop or source-sequence discontinuity
opens a new gap episode. Delivery of the next marked accepted IMU item notifies the backend
and closes that episode; further contiguous accepted input does not create another event.
`dropped_imu_samples` counts samples dropped by the provider queue, while capture-side drops
remain in capture health. `pose_output_stalls` increments once each time the backend worker
transitions from running to waiting on a full output queue.

The initial capacities are configuration values, not ABI constants. Deployment defaults
must be selected from measured OAK-D/backend behavior before implementation release.

## Interface 2 — `IPoseProvider`

```cpp
namespace core {

enum class ProviderInputMode { Stereo, StereoImu };

enum class PinholeDistortionModel {
    None,
    RadTan5,
    Rational8,
    ThinPrism12,
    Tilted14,
};

enum class TrackingState {
    Initializing,
    Tracking,
    Lost,
    Relocalizing,
};

struct SourceImageRef {
    SensorId sensor_id;
    uint64_t source_sequence;
};

struct PoseEstimate {
    Pose pose;                          // T_slamworld_body
    TrackingState state;
    CaptureGroupId source_capture_group_id;
    std::vector<SourceImageRef> source_images;
    int64_t sample_time_local_ns;       // source image group's reference time
    int64_t sample_time_device_ns;      // source image group's device time
};

struct ProviderCapabilities {
    std::vector<ProviderInputMode> input_modes;
    std::vector<PixelFormat> pixel_formats; // most preferred first
    std::vector<PinholeDistortionModel> distortion_models;
};

struct ProviderIdentity {
    std::string backend_name;
    std::string backend_version;
};

struct ProviderHealth {
    uint64_t dropped_stereo_groups;
    uint64_t dropped_imu_samples;
    uint64_t imu_gap_events;
    uint64_t rejected_invalid_input;
    uint64_t rejected_not_running;
    uint64_t pose_output_stalls;
    bool fatal_error;
};

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

`PoseEstimate::source_images` contains exactly one `{sensor_id, source_sequence}` reference
for every image in the originating `ImageGroup`, preserving group order for determinism
while using `sensor_id`, not vector position, as the semantic association. The initial OAK
modes therefore emit exactly the configured left and right references.
`source_capture_group_id` and these references remain internal provenance; the existing
`se3_tracker` schema does not transport them.

`ProviderCapabilities` advertises only modes, pixel formats, and distortion models the
selected adapter can ingest without approximation. All three lists are non-empty; generic
startup validation checks the applied configuration against them before backend-specific
validation.

### Lifecycle and map-frame invariant

Startup order:

1. Load and validate `SlamInstanceConfig`.
2. Create the selected provider and inspect `capabilities()` and `identity()`.
3. Prepare the capture session, then obtain the immutable `AppliedCaptureSpec` from
   `SensorAdapter` plus fusion-ordering bounds and calibration self-report from the capture
   integration.
4. Resolve any calibration-file override once, then install that final `RigCalibration` into
   `SensorAdapter`.
5. Load the exact backend configuration contents, run generic validation, then
   `provider.validate_configuration()` with the same final calibration.
6. Persist the complete resolved setup required by Recording correlation.
7. `DataCollectionSession` arms the recorder, then calls `provider.initialize()` and
   `provider.start()`.
8. Start capture delivery. For every measurement, recorder acceptance still precedes
   provider delivery.

On each running tick, `DataCollectionSession::update()` non-blockingly polls `OakCamera`,
allows capture callbacks and `SensorAdapter` to enqueue fusion-ready input, drains
`try_get_next_pose()` until it returns false, publishes each estimate through
`SchemaPusher`, and checks `provider.health()`. Application code continues to consume poses
through `DeviceIOSession::update()` and `Se3Tracker`; it does not call
`try_get_next_pose()`. A provider `fatal_error` ends the session and prevents it from being
finalized as a successful recording.

Shutdown order:

1. Stop capture delivery and wait for the `SensorAdapter` to become quiescent.
2. Call `provider.finish_input()`. It closes the input side and returns immediately; later
   sensor callbacks return `RejectedNotRunning`.
3. Continue draining pose output while the backend processes every previously `Accepted`
   input. `input_drained()` becomes true only after all accepted input has been processed and
   no more pose output can be produced.
4. After `input_drained()` is true, drain `try_get_next_pose()` until it returns false.
5. Call `provider.stop()`; on the normal path it releases backend resources after the worker
   is already quiescent.
6. Finalize pose and sensor recording, then close the capture session.

`DataCollectionSession` must drain output while waiting in step 3 so a finite output queue
cannot deadlock shutdown. `finish_input()` is idempotent. On a fatal-error/abort path,
`stop()` may be called before normal draining completes; it quiesces the worker and may
discard queued input, and the recording episode is marked failed. `stop()` is idempotent and
is also valid for cleanup after a partially completed `initialize()` or failed `start()`.

There is no in-process reset in the initial release. Within one successful
`start()`/`stop()` run:

- `PoseEstimate::pose` is always expressed in the same `SlamWorld` frame.
- Relocalization must return to that same map frame.
- A backend that cannot preserve the frame after tracking loss remains `Lost` or fails the
  session. It must not silently initialize a new origin.
- Starting a new map requires a full session restart and produces a new recording episode.

## Configuration

```cpp
struct AppliedImageStreamSpec {
    SensorId sensor_id;
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    PixelFormat format;
    double rate_hz;
};

struct AppliedImuStreamSpec {
    SensorId sensor_id;
    double rate_hz;
};

struct AppliedCaptureSpec {
    AppliedImageStreamSpec left;
    AppliedImageStreamSpec right;
    std::optional<AppliedImuStreamSpec> imu;
    ClockDomainId clock_domain_id;
};

struct StereoRigSpec {
    RigId rig_id;
    FrameId body_frame_id;
    SensorId left_sensor_id;
    SensorId right_sensor_id;
    std::optional<SensorId> imu_sensor_id;
    ClockDomainId clock_domain_id;
};

struct OutputSpec {
    enum class Frame { SlamWorld, Session };

    Frame output_frame{Frame::SlamWorld};
    std::optional<Pose> T_session_slamworld;
};

struct QueueConfig {
    size_t max_stereo_groups;
    size_t max_imu_samples;
    size_t max_pose_estimates;
};

struct SlamInstanceConfig {
    std::string instance_id;
    ProviderInputMode input_mode;
    StereoRigSpec rig;
    OutputSpec output;
    QueueConfig queues;
    std::string backend_config_path;
};
```

The transport integration derives the existing `SchemaPusher` collection identifier from
the validated provider instance identifier:

```cpp
std::string make_pose_collection_id(std::string_view instance_id);
// Returns "pose/" + instance_id.
```

The exact mapping is part of the integration contract so `DataCollectionSession`,
`Se3Tracker` consumers, and recording setup rendezvous on the same name. The helper's
placement is an implementation detail; `collection_id` is not separately configurable in
`SlamInstanceConfig`. The derived name must occupy at most
`XR_MAX_TENSOR_IDENTIFIER_SIZE - 1` bytes—255 bytes with the current OpenXR extension
header—excluding the terminating null byte.

Existing live-consumer configurations that pass `"se3_tracker"` must migrate to
`make_pose_collection_id(instance_id)` or the equivalent resolved string. A mismatched
collection name produces no data rather than a connection error, so the session launcher
must single-source `instance_id` and supply the same derived value to
`DataCollectionSession` and every live consumer. This changes consumer configuration, not
the `Se3Tracker` API or the recorded MCAP channel names.

Generic startup validation rejects:

- An empty `instance_id`, an instance ID that is not a portable identifier, or one whose
  derived pose collection name exceeds `XR_MAX_TENSOR_IDENTIFIER_SIZE - 1` bytes.
- Empty `rig_id`, `body_frame_id`, sensor IDs, or clock-domain ID.
- Equal left/right sensor IDs.
- `StereoImu` without `imu_sensor_id`.
- `Stereo` with an IMU stream wired into the provider or an IMU record in
  `AppliedCaptureSpec`; `StereoImu` without matching configured, applied, and calibrated IMU
  records.
- A selected input mode, pixel format, or applied pinhole distortion model absent from
  `capabilities()`.
- Applied descriptors with non-positive dimensions, stride, or rates; a stride too small for
  the selected format; or a clock-domain mismatch.
- Applied descriptors whose sensor IDs, dimensions, formats, or clock domain do not match
  the config and resolved calibration.
- Duplicate calibration records, an incorrect calibration `rig_id`, a calibration
  `body_frame_id` that differs from the configured value, missing configured sensor records,
  or calibration records for unknown sensors.
- A camera model with an unknown distortion tag, a coefficient count that does not match the
  authoritative model table, non-positive or non-finite `fx`/`fy`, or a non-finite principal
  point or distortion coefficient.
- `output_frame == Session` without a finite, valid rigid `T_session_slamworld`;
  `output_frame == SlamWorld` with `T_session_slamworld`.
- Non-positive queue capacities.

`instance_id` uses letters, digits, `.`, `_`, and `-`, begins with a letter or digit, and is
unique among active pose providers in one OpenXR runtime. Therefore its derived
`collection_id` is also unique. It is configured rather than randomly generated so
independently launched consumers can derive the same transport name.

A fixed `T_session_slamworld` has no automatic compatibility identifier. It must be
re-established whenever the backend map-frame convention, backend-native pose frame, input
mode, or repeatable-start convention changes. Changes to calibration or backend
configuration also require re-establishment when they change that map-frame convention or
physical start relationship. Startup validates only the transform's presence and numeric
form; it cannot verify its physical correctness. The complete resolved setup and
application-build provenance are recorded for post-hoc audit.

### Identifier semantics

| Identifier | Assigned by | Scope and lifetime | Purpose |
|---|---|---|---|
| `instance_id` | Session configuration | Unique among active pose providers; stable for one provider run | Names the provider instance and deterministically derives its pose transport collection |
| `collection_id` | `make_pose_collection_id()` | `"pose/" + instance_id`; stable for the run | Existing `SchemaPusher`/`Se3Tracker` rendezvous name; not a separate configuration field |
| `rig_id` | Rig configuration and resolved calibration | Stable for the configured physical rig | Prevents applying calibration for a different rig |
| `sensor_id` | Capture configuration and resolved calibration | Unique within a rig; stable for the run | Binds delivered image/IMU samples and applied descriptors to calibration records; `left_sensor_id`, `right_sensor_id`, and `imu_sensor_id` are references to these values |
| `body_frame_id` | Rig configuration and resolved calibration | Names one fixed rigid frame for the run | Defines the body frame used by the reported pose and every `T_body_<sensor>` extrinsic |
| `clock_domain_id` | Capture integration | Stable for one raw device-clock domain during the run | Establishes which raw timestamps may be compared and fusion-ordered |
| `capture_group_id` | Capture integration | Unique within one recording episode, normally a monotonic counter | Identifies one atomic image measurement shared by the recorder and provider branches |
| `source_capture_group_id` | Pose Provider | Copies the originating `capture_group_id` into `PoseEstimate` | Preserves internal input provenance for diagnostics and tests |

## Canonical calibration and frame convention

The orchestrator supplies one fully resolved record after combining capture self-report and
any file override. Device loaders and backend adapters are the only convention-conversion
points. Calibration describes the exact pixel arrays and sample timing delivered through
`AppliedCaptureSpec` and `ImageFrame`, not a sensor mode upstream of crop, resize, ISP, or
rectification.

### Frames and transforms

- All frames are right-handed.
- `StereoRigSpec::body_frame_id` selects one fixed rigid frame for the run, and
  `RigCalibration::body_frame_id` must match it. In the initial OAK-D release, the physical
  left or center/RGB optical frame may be selected when resolved device calibration supplies
  all required transforms. A virtual stereo-midpoint frame is permitted only through a
  calibration-file override that supplies its `body_frame_id` and every
  `T_body_<sensor>`; the initial release does not synthesize virtual body frames. A body
  frame need not correspond to a delivered stream.
- `T_A_B` transforms coordinates expressed in frame B into frame A.
- `T_body_left`, `T_body_right`, and optional `T_body_imu` express delivered sensor frames in
  the configured body frame; none is required to be identity.
- All 4x4 matrices are row-major homogeneous transforms in meters.
- `PoseEstimate::pose` is `T_slamworld_body`.
- `core::Pose` uses position in meters and a unit Hamilton quaternion ordered `(x,y,z,w)`,
  matching the existing `se3_tracker` schema.
- A backend adapter that estimates `T_slamworld_native` converts it with
  `T_slamworld_body = T_slamworld_native * inverse(T_body_native)`.
- `SlamWorld` is the backend's fixed map frame, not the configured body frame at tracking
  initialization. Changing `body_frame_id` changes `T_slamworld_body` only through the
  extrinsic conversion above; it does not redefine `SlamWorld`.
- `T_session_slamworld * T_slamworld_body` produces `T_session_body`.
  `OutputSpec::T_session_slamworld` aligns the fixed map frame to `Session`; the operational
  validity rules under Configuration determine when that fixed transform must be
  re-established.

### Calibration-to-output pose flow

Resolved calibration serves two distinct purposes:

1. Camera intrinsics, distortion, stereo extrinsics, and optional IMU calibration configure
   the estimator and therefore affect the pose it computes.
2. The fixed extrinsic between the backend's estimated frame and the configured body frame
   converts each valid backend estimate into the pose exposed by `IPoseProvider`.

During initialization, each backend adapter identifies the native rigid frame in which its
backend reports pose. The native frame may be a calibrated sensor frame such as the delivered
left camera or IMU, or a deterministic backend frame whose fixed relation to the calibrated
sensors is known. The adapter resolves one immutable `T_body_native` from the final
`RigCalibration` and backend configuration. Backend-specific validation fails initialization
if the native frame is ambiguous or its transform to the configured body frame cannot be
resolved; the adapter must not guess it or infer a changing transform while running.

The output path is:

```text
resolved RigCalibration
    ├── estimator calibration ──▶ backend ──▶ T_slamworld_native ──┐
    │                                                              │
    └── fixed T_body_native ───────────────────────────────────────┤
                                                                   ▼
             T_slamworld_body =
                 T_slamworld_native * inverse(T_body_native)
                                   │
                 ┌─────────────────┴──────────────────┐
                 ▼                                    ▼
         SlamWorld output                      Session output
         T_slamworld_body          T_session_slamworld * T_slamworld_body
```

The inverse is required because calibration stores `T_body_native`, while pose composition
needs `T_native_body` to move from the estimated native frame to the configured body frame.
This post-estimation composition uses only the resolved native-to-body extrinsic; the other
calibration fields already influenced the estimate through backend initialization.

For example, when the OAK-D center/RGB optical frame is the configured body and a backend
reports the delivered left-camera pose:

```text
T_slamworld_body = T_slamworld_left * inverse(T_body_left)
```

Changing the configured body frame changes this final fixed extrinsic composition, not the
backend's `SlamWorld`. `PoseEstimate::pose` always contains the resulting
`T_slamworld_body`; `DataCollectionSession` applies the optional `Session` transform when
constructing the valid wire pose.

### Initial camera model

The initial release supports one pinhole projection family using the
`PinholeDistortionModel` tag declared in the provider interface:

```cpp
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
```

The coefficient list uses the cumulative OpenCV/DepthAI Perspective ordering:

| Distortion model | Count | Coefficient order |
|---|---:|---|
| `None` | 0 | Empty |
| `RadTan5` | 5 | `k1, k2, p1, p2, k3` |
| `Rational8` | 8 | `k1, k2, p1, p2, k3, k4, k5, k6` |
| `ThinPrism12` | 12 | Rational8 followed by `s1, s2, s3, s4` |
| `Tilted14` | 14 | ThinPrism12 followed by `tau_x, tau_y` |

Enum values are semantic model tags, not encoded coefficient counts, and configuration
serializes their symbolic names. Generic validation uses the single helper above and the
table as its authoritative enum-to-length mapping. It also requires positive finite
`fx`/`fy` and finite principal-point and coefficient values. The resolved canonical record
uses `None` with an empty list for a distortion-free delivered pixel grid. A source
record explicitly tagged by its device/file API as OpenCV/DepthAI Perspective with four
coefficients is normalized by that loader to `RadTan5` by appending `k3 = 0`. A bare
four-element list without a source projection/convention tag is rejected: the canonical
model has no four-coefficient variant, and selecting one would require guessing the source
convention.

DepthAI `Perspective` calibration maps to the matching pinhole variant without truncating
coefficients. DepthAI `Fisheye` and other non-pinhole projection families remain deferred
capabilities even though a fisheye source may also carry four coefficients; coefficient count
alone never selects the model. Initial OAK-D camera streams are treated as global-shutter
inputs. Rolling-shutter timing models are also deferred capabilities.

The canonical validator understands every pinhole variant above. Generic startup validation
rejects an applied model absent from the selected backend's `distortion_models` capability;
backend-specific `validate_configuration()` may impose tighter constraints, but an adapter
must not silently discard higher-order coefficients.

For each camera, calibration width and height must equal its applied stream descriptor.
Intrinsics and distortion coefficients describe the delivered pixel grid after every
capture-side crop, scale, and rectification step. If capture delivers rectified images, it
must supply the calibration of those rectified images using `None` when the delivered grid
is distortion-free, and `T_body_camera` must remain consistent with that rectification
convention. A mismatch fails startup; the provider never guesses or rescales calibration.
Changing rectification or `body_frame_id` requires a newly resolved calibration and a new
provider run.

### IMU calibration

```cpp
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
```

`RigCalibration::imu` must be present in `StereoImu` mode and absent in `Stereo` mode. Its
sensor ID and rate must match `AppliedCaptureSpec::imu`. A backend may impose tighter
model/rate/noise or applied-stream constraints through
`validate_configuration()`.

The resolved calibration, including any file override, is part of the complete provider
setup recorded with the data-collection episode.

## Backend selection and reference roles

One CMake selection builds exactly one provider implementation into a binary:

```cmake
set(POSE_PROVIDER_BACKEND stub CACHE STRING "cuvslam | orbslam3 | stub")
set_property(CACHE POSE_PROVIDER_BACKEND PROPERTY STRINGS cuvslam orbslam3 stub)
```

Configuration rejects any value outside that set. The factory has an explicit branch for
each value and no silent fallback. Every implementation returns a non-empty
`ProviderIdentity`; the backend version identifies the linked or developer-supplied SLAM
library. The recording session captures IsaacTeleop application-build provenance separately
instead of making provider adapters report their own build identifiers.

### ORB-SLAM3 — proven functional reference

- The existing OAK-D stereo and stereo-inertial pipeline is the behavioral baseline for
  initial integration.
- ORB-SLAM3 supports the two initial input modes and is useful for development validation.
- The adapter advertises only the applied pixel formats and pinhole distortion models it
  converts without dropping coefficients.
- The upstream project is GPLv3 and offers separate commercial licensing. Under the normal
  development path it is developer-supplied, not vendored or downloaded, and absent from
  release CI and release artifacts.
- Shipping an ORB-SLAM3 binary requires an approved commercial license or a separately
  approved GPL-compliant distribution plan.

### cuVSLAM — production reference

- cuVSLAM is the intended production backend.
- The adapter reports only input modes actually supported by the selected cuVSLAM version.
  The design does not assume IMU is mandatory: `Stereo` and `StereoImu` are negotiated
  independently through `ProviderCapabilities::input_modes`.
- It likewise reports only pixel formats and pinhole distortion models supported by the
  selected SDK adapter.
- The backend adapter converts the canonical calibration and frame convention to the
  selected cuVSLAM API.
- Exact SDK version, redistribution, CUDA, and target-platform requirements are release
  qualification gates.

### Stub — contract-test reference

The stub has no external dependencies and supports both initial input modes, every declared
pixel format, and every canonical pinhole distortion model. It is scriptable to:

- Produce a deterministic trajectory.
- Emit every `TrackingState`.
- Hold one final estimate until `finish_input()` to exercise lossless draining.
- Simulate an IMU gap, queue overflow, and attempted map-frame reset.
- Preserve and echo capture provenance/timestamps for conformance tests.

## Output contract

`DataCollectionSession` drains `try_get_next_pose()` until false on each integration tick
and maps every estimate to the existing schema:

```cpp
core::Pose apply_output_frame(const OutputSpec& output,
                              const core::Pose& T_slamworld_body) {
    if (output.output_frame == OutputSpec::Frame::SlamWorld) {
        return T_slamworld_body;
    }
    return compose(*output.T_session_slamworld,
                   T_slamworld_body); // presence and rigid-transform form validated at startup
}

core::Se3TrackerPoseT to_wire(const PoseEstimate& estimate,
                              const OutputSpec& output) {
    core::Se3TrackerPoseT wire;
    wire.is_valid = estimate.state == TrackingState::Tracking;
    wire.pose = std::make_shared<core::Pose>(
        wire.is_valid ? apply_output_frame(output, estimate.pose)
                      : core::Pose{}); // contents unspecified when invalid
    return wire;
}
```

`DataCollectionSession` creates `SchemaPusher` with
`make_pose_collection_id(config.instance_id)`. Consumers use the same helper or the
equivalent resolved string, so no separately configured collection name can drift from the
provider instance name.

`SchemaPusher::push_buffer()` receives
`estimate.sample_time_local_ns` and `estimate.sample_time_device_ns`. These are always the
measurement timestamps of `source_capture_group_id`, never the time at which SLAM completed
or the plugin published the estimate.

`se3_tracker` cannot carry `source_capture_group_id` or source sequences. In the initial
OAK-D recording, the exact raw-device measurement timestamp is therefore the wire-level
join key between:

- Recorded left/right frame metadata.
- The pose computed from that stereo group.

The capture integration enforces a strictly increasing raw device timestamp for accepted
stereo groups before either recorder or provider delivery. A duplicate is rejected and
increments capture health `nonmonotonic_stereo_groups`; the episode may continue because the
previously accepted group retains sole ownership of that timestamp. A regression is rejected,
increments the same counter, fails the episode, and requires a new episode before delivery
can resume because the device timeline is no longer trustworthy. The internal
`PoseEstimate` provenance remains available for tests and diagnostics. A future need to
transport provenance explicitly would trigger a schema extension or a new pose schema.

### Tracking-state publication

- `is_valid` is true only in `Tracking`.
- `Initializing`, `Lost`, and `Relocalizing` publish an identity/unspecified filler with
  `is_valid=false`.
- A fresh invalid estimate uses its own source capture timestamps.
- `DataCollectionSession` publishes only fresh estimates returned by the backend. It does
  not synthesize an invalid measurement or reuse an earlier capture timestamp when output
  is absent.
- Missing pose output remains a timestamp-visible gap in the recorded pose stream. Liveness
  monitoring may fail the data-collection session, but it must not fabricate a measurement.

Detailed tracking state is intentionally dropped at the `se3_tracker` boundary.

## Timing contract

Three times must not be conflated:

1. **Raw device measurement time** — the source timestamp preserved for provenance. Camera
   fusion time uses this value directly; IMU fusion time applies the recorded calibration
   offset without overwriting it.
2. **Local common measurement time** — the corresponding host monotonic timeline used to
   align the pose with other recorded streams.
3. **Availability time** — when the computed pose becomes observable.

Capture supplies the first two; the recording infrastructure supplies availability time.
The Pose Provider preserves the two measurement timestamps and may report the result later.

For stereo group N:

```text
recorded left/right metadata: group N, device time T, local time L
provider input:              group N, device time T, local reference time L
pose output (later):         source group N, sample device time T, sample local time L
```

When IMU is enabled, individual IMU samples retain their own device/local measurement
times. The pose remains stamped at the stereo group time; the recorded IMU interval can be
located around that time using the recorded `time_offset_ns`.

The integration rejects unknown/mismatched clock domains before provider initialization. It
does not perform device-to-host clock mapping, camera pairing, skew estimation, or drift
correction.

## Integration and verification

### Provider conformance

Run all tests against the stub and each enabled backend where applicable:

- Lifecycle accepts only the documented transition sequence.
- Sensor callbacks are rejected outside the running interval.
- Capability lists are non-empty, and generic startup validation rejects an applied input
  mode, pixel format, or distortion model absent from them.
- Initialization receives exactly one immutable applied-capture descriptor; descriptor,
  calibration, and frame-buffer geometry mismatches fail before backend ingestion.
- Generic calibration validation accepts `None`, `RadTan5`, `Rational8`, `ThinPrism12`, and
  `Tilted14` only with their exact coefficient counts and finite values; it rejects an
  unknown tag, a tag/count mismatch, or an untagged four-element list.
- For every enabled backend, preserve every coefficient of each supported pinhole variant;
  reject an unsupported variant before initialization rather than truncating it.
- Initial stereo input is one atomic `ImageGroup` containing exactly the configured left and
  right images; concurrent worker processing cannot observe a partial group.
- One input producer delivers callbacks while one output consumer concurrently drains poses
  and reads health/drain state without races.
- An incorrect image count, duplicate or unconfigured sensor IDs, or an image timestamp that
  does not match the initial OAK-D group timestamp returns `RejectedInvalidInput` before
  backend ingestion; mismatched clock domains fail integration validation before
  initialization.
- Input and output queues remain within configured capacities.
- Stereo overflow drops one complete `ImageGroup` and increments one group counter.
- IMU overflow increments the drop and gap-event counters; the next accepted IMU item carries
  the internal gap marker, the worker notifies the stub/backend adapter before ingestion, and
  the generic wrapper does not override tracking state.
- IMU callbacks are ordered by calibration-adjusted fusion time while their raw timestamps
  remain unchanged.
- Pose output preserves the source capture ID, the complete set of sensor-ID/sequence
  references, and both measurement timestamps.
- A full pose-output queue stalls the worker, increments `pose_output_stalls`, loses no pose,
  never sets `fatal_error`, and never blocks an input callback. Sustained pressure surfaces
  through normal counted input drops.
- `finish_input()` rejects new callbacks; `DataCollectionSession` drains output while
  waiting for `input_drained()`, then observes an empty final output queue without deadlock
  or loss.
- A simulated internal map reset cannot emit a pose in a new map frame under the same run.
- Provider silence does not synthesize an output or reuse an old measurement timestamp.
- Backend-specific validation rejects an ambiguous native pose frame or a native frame with
  no fixed transform to the configured body frame.
- With a non-identity `T_body_native`, a known backend-native trajectory produces the expected
  `T_slamworld_body`; selecting `Session` additionally pre-multiplies exactly
  `T_session_slamworld`.
- `Session` output requires a finite valid `T_session_slamworld`; `SlamWorld` output rejects
  one. Confirm that the documented map-frame/start changes require operational
  re-establishment rather than implying that startup validates physical correctness.
- The derived collection ID is exactly `"pose/" + instance_id`, satisfies the transport
  identifier limit, and is used by both `SchemaPusher` and the `Se3Tracker` consumer.

### OAK-D data-collection integration

Using a recorded or live OAK-D episode:

- Run both `Stereo` and `StereoImu`.
- Confirm every provider stereo input corresponds to recorded left/right metadata with the
  same strictly increasing device timestamp and source sequences.
- Inject a duplicate stereo timestamp; confirm capture rejects it before recorder/provider
  delivery, increments `nonmonotonic_stereo_groups`, and continues the episode. Inject a
  regressed timestamp; confirm capture rejects it, increments the same counter, and fails the
  episode.
- In `StereoImu`, confirm every provider IMU input is present in the recorded episode.
- Exercise the configured fusion reorder window, late-sample rejection, item-capacity forced
  emission, counters, and ordered shutdown flush while preserving raw recorded timestamps.
- Reject a fusion-ordering configuration whose item capacity cannot hold one full configured
  aggregate-rate window plus one item.
- Apply a calibration override to `time_offset_ns`; confirm the same resolved record drives
  adapter ordering, provider initialization, and recording.
- Confirm the OAK loader uses the device camera-model tag, preserves all Perspective
  coefficients in canonical order, normalizes distortion-free delivered pixels to `None`,
  and rejects Fisheye under the initial scope.
- Confirm the recorded episode contains the resolved instance/output configuration, applied
  descriptors, body-frame selection, fusion-ordering settings/counters, calibration,
  provider identity, IsaacTeleop application-build provenance, exact backend configuration
  contents, and `T_session_slamworld` when Session output is selected.
- Force recorder backpressure and confirm rejected sensor measurements never reach the
  provider.
- Confirm every emitted pose joins to exactly one recorded stereo group by device
  measurement timestamp.
- Exercise capture and provider backpressure; verify all gaps are attributable and no
  partial image group is delivered.
- Exercise any capture-side crop, resize, or rectification and verify the recorded/applied
  calibration describes the exact delivered pixel geometry and configured body frame.
- Read the pose through the existing `Se3Tracker`/replay API using the derived live
  collection name and unchanged recorded channel names.

### Backend qualification

- Compare the provider's ORB-SLAM3 path with the proven OAK-D reference pipeline on the same
  recorded episodes.
- Qualify cuVSLAM independently for both input modes it advertises.
- Confirm backend-native calibration/frame conversions against known transforms and
  trajectories.
- Qualify each inertial backend's configured gap tolerance, tracking-state behavior, and
  recovery criteria; the generic wrapper must not override those results.
- Keep ORB-SLAM3 absent from default/release dependency resolution and artifacts unless a
  separate licensing approval exists.

## Design-level files

This pass defines interfaces and integration contracts; implementation is separate.

- `src/core/pose_provider/cpp/inc/pose_provider/sensor_sink.hpp` — normalized input types,
  `ISensorSink`, and `EnqueueResult`.
- `src/core/pose_provider/cpp/inc/pose_provider/pose_provider.hpp` — `IPoseProvider`,
  capabilities, identity, estimates, health, and factory.
- `src/core/pose_provider/cpp/inc/pose_provider/calibration.hpp` — canonical initial-release
  calibration.
- `src/core/pose_provider/cpp/inc/pose_provider/config.hpp` — stereo-rig, output, and queue
  configuration, immutable applied-capture descriptors, and generic validation.
- `src/plugins/oak/core/oak_camera.hpp/.cpp` — evolve the existing OAK device owner to report
  capture-native applied stream information without depending on Pose Provider types.
- `src/plugins/oak/core/oak_sensor_adapter.hpp/.cpp` —
  `plugins::oak::SensorAdapter`, which constructs and owns the initial
  `AppliedCaptureSpec`, performs bounded fusion ordering, and delivers provider input.
- `src/plugins/oak/core/data_collection_session.hpp/.cpp` —
  `plugins::oak::DataCollectionSession`, which owns the initial integration lifecycle,
  recorders, provider, pose `SchemaPusher`, non-blocking update loop, output drain, and
  coordinated shutdown.
- `src/core/pose_provider/backends/stub/` — deterministic conformance backend.
- `src/core/pose_provider/backends/cuvslam/` — production backend.
- `src/core/pose_provider/backends/orbslam3/` — developer-supplied functional-reference
  adapter, subject to the licensing boundary above.

The initial integration lives in the existing OAK capture process so one process owns the
device and recording pipeline. The responsibility split among `OakCamera`, `SensorAdapter`,
and `DataCollectionSession` is normative; internal refactoring within those boundaries
remains an implementation detail.
