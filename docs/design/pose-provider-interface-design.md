<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Pose Provider — Interface Design

This design introduces a Pose Provider that converts synchronized camera and optional IMU
measurements into a 6-DoF pose stream using a selected pose estimation engine. Any capture
integration that meets the input contract can feed it; the initial OAK-D integration targets
stereo and stereo-inertial data collection. Recordings preserve the live pose and exact
provider input so a later offline tool can reproduce a run or attach an optimized trajectory.
Existing pose schemas and consumer APIs remain unchanged. Capture mechanisms, recording
encodings, and SLAM algorithms remain the responsibilities of their respective components.

This is the detailed implementation contract. The
[Google Doc](https://docs.google.com/document/d/1ec61WH_JcFmm3zlSfyYbSx9SNdDYt59V9y-O85i7QbM/edit)
is its condensed review presentation. Unresolved decisions are listed under
[Implementation decision gates](#implementation-decision-gates).

## Motivation

Isaac Capture needs 6-DoF pose from cameras that expose images and optional IMU data but do
not provide pose through an existing device plugin. A proven OAK-D and ORB-SLAM3
data-collection pipeline provides the behavioral baseline; cuVSLAM is the intended
production pose estimation engine. Data-collection users need live pose to assess a take
and recorded inputs to evaluate or improve the trajectory afterward.

## Goals

- Define a common Pose Provider contract that supports build-selectable pose estimation
  engines and produces one 6-DoF pose stream from synchronized camera and optional IMU
  measurements.
- Preserve measurement timing and source identity so pose output remains correlatable with
  recorded sensor data.
- Preserve the exact provider inputs, their admission outcomes, and resolved setup so an
  offline tool can reconstruct the live run or estimate a new trajectory.
- Define consistent input, calibration, frame, lifecycle, and health contracts across
  engines, including a stable pose frame for each run.
- Reuse the existing pose transport, recording, replay, and consumer APIs.
- Keep capture, estimation, and transport responsibilities separate so the design can grow
  without coupling the core provider contract to vendor-specific mechanisms.

## Initial release scope

The initial release has one supported input topology and two input modes:

| Topology | Input mode |
|---|---|
| One OAK-D stereo pair | `Stereo` |
| One OAK-D stereo pair + its onboard IMU | `StereoImu` |

One data-collection session owns one provider run and produces one recording episode.
Each new episode starts a new provider run; keeping a map across multiple takes remains a
proposal outside the initial scope.

For the initial release:

- The supported device is one physical OAK-D providing one stereo pair and, optionally, its
  onboard IMU. It forms one rigid rig and one raw device-clock domain.
- A new C++ data-collection executable reuses shared OAK capture components and owns one
  device pipeline. It coordinates sensor recording, pose estimation, and publication in
  that process. The existing `camera_plugin_oak` executable retains its current behavior;
  this integration does not depend on the unfinished generic capture interface.
- Input uses owned CPU buffers from two pinhole camera streams. Configuration selects one
  fixed body frame, and resolved calibration describes the exact delivered streams.
- The supported use case is data collection: measurement correctness and sensor/pose
  correlation take priority over low output latency.
- One application build links one selected provider and produces one pose stream
  through one existing `se3_tracker` collection. ORB-SLAM3 is the functional reference,
  cuVSLAM is the production reference, and the stub is the contract-test reference.
- Output uses the engine's fixed `SlamWorld` by default or one fixed startup transform into
  `Session`. Dynamic map-frame or alignment changes are not supported within a run.

## Design boundaries and non-goals

These are responsibilities this design does not own, including when a deferred capability
eventually uses them:

- Designing camera or IMU capture APIs, delivery to multiple consumers, vendor
  synchronization, or clock mapping mechanisms.
- Defining sensor-recording infrastructure or file formats. This design specifies the
  information recording must preserve; capture integration owns its encoding and storage.
- Defining pose-estimation algorithms or engine-native configuration formats.
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
5. **Offline re-estimation tool.** Re-run an engine on recorded inputs and attach a new
   trajectory with its provenance. Recording the required inputs is in initial scope;
   delivering this tool, blocking provider admission, and a refined-trajectory API are not.

The initial release stabilizes only the seams it exercises; these directions are not
compatibility promises.

## Architecture and ownership boundary

There are three independent seams:

1. **Capture seam.** The capture layer owns devices, synchronization, sensor recording, and
   delivery to its consumers.
2. **Estimator seam.** `IPoseProvider` owns engine state and engine-native translation.
3. **Transport seam.** Existing `se3_tracker` schema, consumer API, recording, and replay
   remain unchanged; live consumers use the collection name derived from `instance_id`.

Capture integration may prepare and deliver provider input directly or through an
implementation-specific adapter. The provider contract requires the same input and lifecycle
outcomes in either case; it does not require an adapter type. The
[initial OAK-D integration](#initial-oak-d-integration) uses `SensorAdapter` for this role.

### Runtime ownership

The data collection coordinator composes these runtime responsibilities and coordinates their
lifecycle. Each child retains the internal responsibility defined by the three seams above.

```text
Data collection coordinator
        ├── Capture integration
        │       ├── capture/device source
        │       ├── sensor recording
        │       └── provider input handling
        ├── IPoseProvider implementation
        ├── output integration
        ├── telemetry
        └── lifecycle and shutdown
```

The table maps those ownership responsibilities to the initial OAK-D implementation without
adding new contract behavior.

| Generic responsibility | Initial OAK-D integration |
|---|---|
| Data collection coordinator | `DataCollectionCoordinator` in the new DC executable |
| Capture integration | Coordinator composes the OAK capture source, recorder, `SensorAdapter`, and recording admission gate |
| Capture/device source | Shared `OakCamera` behind the application's proposed `OakCaptureSource` facade |
| Sensor recording | Coordinator-owned `EpisodeRecorder`, with recorder acceptance enforced by `RecordingAdmissionSink` |
| Provider input handling | `plugins::oak::SensorAdapter` feeds `RecordingAdmissionSink`, which forwards to the provider |
| `IPoseProvider` implementation | Adapter for ORB-SLAM3, cuVSLAM, or the deterministic test engine |
| Output integration | Coordinator output worker publishes through `PoseOutput`, including the output-frame transform; recording preserves that same published pose through the existing transport/recording path |
| Telemetry | `DataCollectionCoordinator` live health view and episode summary |
| Lifecycle and shutdown | `DataCollectionCoordinator` |

The new executable is the single device-owning process for a DC run; it does not launch
`camera_plugin_oak`. Recording and pose estimation branch from the same capture pipeline.
The recorder and capture facade are proposed integration components; `OakCamera`,
`SchemaPusher`, and `Se3Tracker` are existing components to reuse or extend. Source namespaces
such as `plugins::oak` identify code ownership, not the process that must host that code.
Concrete responsibilities are detailed under
[Initial OAK-D integration](#initial-oak-d-integration).

The coordinator assembles one immutable setup and queries capabilities and identity before
initialization.

### End-to-end data flow

The ownership view does not show the data path. This diagram instead makes the provider's
inputs and outputs visible at a glance.

```text
capture integration ── synchronized sensor data ──┬──▶ sensor recording
                                                  │
                                                  └──▶ provider input handling
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

The two capture branches receive the same source measurement, but provider delivery occurs
only after recorder acceptance succeeds, as required by the
[capture-integration contract](#capture-integration-requirements).

The capture integration submits normalized measurements for asynchronous processing in
fusion-time order, while the coordinator drains estimates and health without exposing
provider pull calls to pose consumers.

### Interface and type relationships

```text
ImageGroup ──contains──▶ ImageFrame
ImageGroup ─┐
ImuSample ──┴──accepted by──▶ ISensorSink ◀──implemented by── IPoseProvider

SlamInstanceConfig ──┐
AppliedCaptureSpec ──┼──validate + initialize──▶ IPoseProvider
RigCalibration ──────┘                                  │
                                                        ├── advertises ProviderCapabilities
                                                        ├── identifies ProviderIdentity
                                                        ├── produces PoseEstimate
                                                        └── reports ProviderHealth

IPoseProvider lifecycle:
validate_configuration → initialize → start → finish_input → input_drained → stop
```

In contrast to the [Runtime ownership](#runtime-ownership) view, this UML-lite view shows only
contract-level type relationships, not ownership or end-to-end flow. The
[Pose Provider API](#pose-provider-api) defines the supporting types without turning the
diagram into a field-by-field data model.

### Independent rigs

The provider contract describes one rigid rig and one raw device-clock domain per run.
Several independent rigs can each have their own capture process, provider, `instance_id`,
pose stream, and recording. Process isolation also accommodates engines with process-global
calibration state; the [engine survey](pose-provider-backend-survey.md) records those limits.

On one host, local common measurement time correlates the recordings. Each provider still
has its own `SlamWorld`; a known fixed transform may express a common output frame, but no
shared map or automatic spatial alignment is implied. Coordinated multi-rig episode control
and two-device hardware qualification remain scope decisions, not requirements of this API.

## Capture-integration requirements

The provider-facing capture contract is defined by the
[Ego Data Capture Interface](https://docs.google.com/document/d/1C56pIKonEo3HbCZKl5OffWj65qrx7GBy40X5GK5tntc/edit?tab=t.0#heading=h.rb7yjh6gaa3i).
It supplies synchronized sensor groups, measurement timestamps and clock mapping, applied
calibration and stream configuration, and recording correlation. This design states what the
Pose Provider requires from that contract, not how capture implements it.

The direct OAK-D integration and any future generic capture interface must provide the same
provider-facing contract:

- **Synchronized input.** Deliver each image measurement as one atomic `ImageGroup`, with
  stable sensor and capture identities, source sequences, and device and local measurement
  timestamps. Optional IMU samples use the same clock domain. Initial OAK-D stereo frames
  have exactly equal device timestamps; future capture implementations may use another
  declared grouping rule. Host arrival time is never treated as measurement time.
- **Fusion-time ordering.** Deliver image groups and optional IMU samples in nondecreasing
  fusion-time order using device measurement times and resolved timing offsets. At equal
  fusion time, IMU samples precede the image group and retain source-sequence order. A
  regression within one source stream fails the episode. A cross-stream item that arrives
  too late to preserve this order is withheld and counted instead. The applied ordering policy
  and any bounds or counters are recorded with the episode; the Pose Provider does not repair
  or reorder its input. Capture-side IMU loss also requires the explicit treatment identified
  under [Implementation decision gates](#implementation-decision-gates).
- **Recording correlation.** The data collection coordinator ensures that the same
  measurements offered to the provider are recorded, including their identities and
  timestamps. Provider images are recorded losslessly after capture-side processing, with
  their delivered geometry, format, and buffer layout. Recorded image-group metadata retains
  each image's sensor ID and source sequence. Recorder acceptance precedes provider delivery;
  a measurement that recording cannot accept is not sent to the provider.
  A later recording failure fails the session rather than leaving an apparently valid pose
  stream without its source data. The provider remains independent of the recording
  implementation and carries source identity and measurement time into `PoseEstimate`.
  The integration records each withheld, dropped, rejected, or accepted input outcome and
  the actual admission order; aggregate counters alone cannot reconstruct the live input.
- **Immutable applied setup.** Before provider initialization, the integration supplies one
  selected `ProviderInputMode`, one `AppliedCaptureSpec` describing the streams actually
  delivered, and one fully resolved `RigCalibration`. Calibration describes the delivered
  images after any capture-side processing and is established before measurement delivery
  begins. The same records are used for validation, provider initialization, fusion-time
  ordering, and episode metadata, and remain fixed for the run.

The [Initial OAK-D integration](#initial-oak-d-integration) shows how the new DC application
constructs this applied setup, prepares ordered input, and coordinates delivery. That mapping
is one implementation of these requirements rather than part of the generic provider contract.

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
    uint32_t stride;                 // bytes per row; validated for PixelFormat
    PixelFormat format;
    std::vector<uint8_t> data;       // owned contiguous CPU buffer
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

Before calling this interface, the capture integration satisfies the
[capture-integration requirements](#capture-integration-requirements) for synchronization,
recording, ordering, and applied setup. `ISensorSink` only admits work; SLAM/VIO processing
happens asynchronously.

Each frame's dimensions, format, stride, and buffer layout must match its
`AppliedImageStreamSpec`. The [canonical image buffer layouts](#canonical-image-buffer-layouts)
define the accepted layouts and validation rules. An invalid or undersized buffer returns
`RejectedInvalidInput`.

- `on_image_group()` transfers one complete group and its image buffers. `on_imu()` copies
  the small sample before returning.
- Each image group contains exactly one frame for every configured camera sensor. Sensor ID,
  not vector position, identifies the stream.
- The capture integration serializes input and delivers it in nondecreasing fusion-time order
  under the [capture-integration requirements](#capture-integration-requirements). Camera
  groups use their device measurement time; IMU samples apply the resolved calibration offset
  for ordering without changing their recorded raw timestamps.
- The provider preserves accepted admission order across its image and IMU queues when
  passing measurements to the engine adapter. Separate queue capacities do not permit
  separate workers to overtake one another at engine ingestion. Engine-specific batching
  must preserve this order and the association of any IMU gap with the following sample.
- Input calls are legal only while the provider is running. One input producer may operate
  concurrently with one thread draining estimates and reading health; lifecycle operations
  remain serialized.

#### Admission outcomes and backpressure

Input methods use finite per-provider queues and never block:

- `on_imu()` in `Stereo` mode returns `RejectedInvalidInput` because the sample does not match
  the initialized input contract.
- Any `RejectedInvalidInput` result is a capture-integration contract violation. The coordinator
  marks the run failed and begins coordinated shutdown; it does not finalize the episode as a
  successful but degraded pose stream.

- `RejectedNotRunning` during abort or partial-startup cleanup is counted in
  `rejected_not_running` and does not replace the failure already in progress. At any other
  time, including normal coordinated shutdown, it is a lifecycle-ordering violation that
  fails the session.
- A full image queue rejects the incoming `ImageGroup` as one atomic unit.
- Every dropped IMU sample is counted. One or more drops before the next accepted IMU sample
  form one gap event, and the Pose Provider informs a stereo-inertial engine of that gap before it
  ingests the accepted sample. The detailed handoff is in
  [IMU gap handoff](#imu-gap-handoff); tracking behavior remains engine-specific.
- Drops and rejections are counted in provider health.
- A full pose-output queue stalls the provider worker until the output consumer drains it;
  `pose_output_stalls` increments once when the worker enters each such stall. The condition
  is non-fatal, does not block input callbacks, and does not discard an already-computed pose.
  Continued pressure can instead fill the input queues and produce normal backpressure drops.

Queue defaults are deployment choices based on measured device and engine behavior.

### Provider lifecycle and estimates — `IPoseProvider`

```cpp
namespace core {

struct AppliedCaptureSpec;
struct RigCalibration;
struct SlamInstanceConfig;

// Sensor combinations accepted for one provider run.
enum class ProviderInputMode { Stereo, StereoImu };

// Canonical pinhole distortion models negotiated with a pose estimation engine.
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

// Runtime identity of the selected pose estimation engine.
struct ProviderIdentity {
    std::string engine_name;
    std::string engine_version;
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

// Pose Provider lifecycle and output interface shared by all engines.
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
    // Removes one estimate in production order. The coordinator drains until false
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
engine accepts without approximation; each advertised list is non-empty. Generic startup
validation checks membership without assigning meaning to list order, then runs
engine-specific validation.

#### Lifecycle

The data collection coordinator owns the provider lifecycle:

1. **Configure.** Resolve the instance configuration, applied capture description,
   calibration, and engine configuration; inspect `capabilities()` and `identity()`; run
   generic validation; then call `validate_configuration()` and record that immutable setup.
2. **Initialize and start.** Arm recording, call `initialize()` and then `start()`, and begin
   sensor delivery only after `start()` succeeds.
3. **Run.** Deliver prepared measurements through `on_image_group()` and optional `on_imu()`,
   drain `try_get_next_pose()` without blocking, and monitor `health()`. A fatal provider
   error fails the data-collection session.
4. **Finish.** Stop new sensor delivery, wait for any in-flight delivery call to return, flush
   and drain any capture-side buffered input, call `finish_input()`, and continue draining
   until `input_drained()` is true and `try_get_next_pose()` reports that no estimate remains.
   Then call `stop()` and finalize the recording.

`input_drained()` means every accepted input has been processed and no additional estimate
can be produced. Output must continue to drain while finishing so a bounded output queue
cannot stall shutdown. Repeating `finish_input()` or `stop()` after the first successful
call has no additional effect; `stop()` also supports abort and partial-startup cleanup.

Abort must wake a worker stalled on its output queue and release queued buffers without
requiring a functioning output consumer. The coordinator marks the recording incomplete or
failed and preserves the original failure reason; graceful-drain guarantees apply only to
normal finish. Engine qualification must establish a bounded shutdown strategy, including
when an engine cannot cancel an in-progress call. If the DC application is integrated with
`PluginManager`, its current two-second forced-stop deadline cannot be assumed sufficient;
that optional launcher integration needs a qualified finish handshake or deadline.

#### Stable map frame

One provider run uses one fixed `SlamWorld` frame, established by its first `Tracking`
estimate. An engine that redefines its map frame during startup, such as when aligning to
gravity, stays `Initializing` until that frame is stable. The adapter must detect subsequent
resets, new maps, re-anchoring, and other frame changes. Relocalization must return to the
original frame; an engine that cannot preserve it remains `Lost` or fails the session rather
than silently creating a new origin. Starting a new map requires a new provider run and
recording episode. Loop-closure correction can change an estimated trajectory within a fixed
frame; it does not itself authorize changing that frame.

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
    std::string engine_config_path;
};

} // namespace core
```

`SlamInstanceConfig`, the applied capture description, and the resolved calibration form one
immutable setup. The same records are validated, passed to `initialize()`, and stored with
the recording.

`AppliedCaptureSpec` is intentionally fixed to left and right streams for the initial stereo
topology. `ImageGroup` uses sensor-keyed vector storage so frame identity does not depend on
position, but that does not promise that the current applied-setup type supports mono or
larger camera groups. Those extensions will revise `AppliedCaptureSpec` and its validation.

Before `initialize()`, the session performs generic startup validation across this complete
setup. The provider then applies engine-specific restrictions through
`validate_configuration()`. Generic validation requires:

- Identifiers that satisfy [Identifier semantics](#identifier-semantics), with distinct left
  and right sensors.
- Agreement among the selected input mode, rig topology, optional IMU, applied streams,
  resolved calibration, and advertised provider capabilities.
- Valid applied stream geometry and
  [canonical image buffer layout](#canonical-image-buffer-layouts), plus valid rate and clock
  domain, with matching canonical calibration.
- A present, finite rigid `T_session_slamworld` when `Session` output is selected, and no
  `T_session_slamworld` when `SlamWorld` output is selected.
- Positive `max_image_groups` and `max_pose_estimates`; positive `max_imu_samples` in
  `StereoImu` mode. `max_imu_samples` is ignored in `Stereo` mode.

A fixed `T_session_slamworld` has no automatic compatibility identifier. It must be
re-established whenever the engine's map-frame convention or repeatable physical start
relationship changes. Startup validates its form, not its physical correctness. The
recording retains the resolved setup, provider identity, engine configuration, and
application-build information for later audit. The recorded engine configuration is the
resolved content used for the run, not only its machine-local source path. Referenced assets,
such as a vocabulary, need content hashes and a way to retrieve the matching bytes.
`engine_version` identifies the exact library build, including a source commit or equivalent
immutable build identifier.

Engine configuration supplies tuning and assets. Rig geometry, calibration, and timing come
from the canonical setup; conflicting engine-native overrides fail validation. The adapter
generates any required native calibration settings from that setup and records the resolved
settings actually used.

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
| `source_capture_group_id` | Capture integration; copied by Pose Provider | One value per estimate, copied from its image group | Links an internal pose estimate back to its recorded image group |

#### Pose transport naming

`instance_id` is configured rather than generated so independently launched components can
derive the same transport name. The collection name is not separately configurable: an
instance such as `oak_front` produces `pose/oak_front`. The coordinator, live consumers, and
recording setup derive the same value.

Exact character, length, and helper requirements are in
[Pose transport naming constraints](#pose-transport-naming-constraints).

### Frames and calibration

Before provider initialization, the session resolves one `RigCalibration` from the capture
self-report and any file override. Independent fields in an override replace only explicitly
supplied values and take precedence over the self-report or documented integration defaults.
The frame group is atomic: it contains `body_frame_id`, every configured camera's
`T_body_camera`, and `T_body_imu` when an IMU is configured. Supplying any frame-group member
requires supplying all of them; a partial frame-group override, including an IMU-only
extrinsic override, fails startup. IMU noise and timing fields remain independently
overridable.

All supplied transforms must be finite rigid transforms and match the configured rig,
sensors, and delivered raw/rectified frame conventions. Any required field still missing
after resolution fails startup. The result describes the exact images and sample timing
delivered to the provider after capture-side processing. Device loaders and engine adapters
are the only points that convert vendor-specific conventions.

For the initial OAK-D integration, `OakCamera` reports the device's IMU extrinsic, falling
back to the board-design value for that device when necessary, and identifies which source
was used. The device does not store IMU noise parameters. `DataCollectionCoordinator`
supplies documented integration defaults for those missing fields before applying file
overrides and resolving `RigCalibration`. Default values follow the OAK-D baseline and must
be pinned and qualified for the supported hardware; they are not per-device measurements.
The recording retains the effective values and their origins—device calibration, board
design, documented defaults, or file override—alongside the resolved calibration.

#### Frame conventions

**Frames.** Named frames in prose use CamelCase, while transform subscripts use lowercase
tokens.

| Frame | Meaning | Transform token |
|---|---|---|
| `Body` | The configured rigid frame whose pose is reported; selected by the matching `body_frame_id` in `StereoRigSpec` and `RigCalibration` | `body` |
| `SlamWorld` | The pose estimation engine's fixed map frame for the run | `slamworld` |
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
does not redefine `SlamWorld`. The engine-native conversion is described in
[Calibration-to-output pose flow](#calibration-to-output-pose-flow), and the fixed
Session-alignment validity rules remain under
[Instance configuration](#instance-configuration).

#### Calibration-to-output pose flow

Resolved calibration affects pose output in two ways:

1. Camera intrinsics, distortion, stereo extrinsics, and optional IMU calibration configure
   the estimator.
2. A fixed extrinsic converts the engine's native pose into the configured `Body` frame.

During initialization, each engine adapter declares the rigid frame in which its selected
live output reports pose. This may be a calibrated sensor frame or another engine frame with
a known fixed relation to the rig, including a rectified optical frame. The adapter resolves
one immutable `T_body_native` from the final calibration and engine configuration, including
any rectification rotation; initialization fails if that transform is ambiguous or unavailable.
Any future refined-output path must declare its own native frame rather than assuming it
matches live output.

The output path is:

```text
resolved RigCalibration
    ├── estimator parameters ──▶ pose estimation engine
    │                                      │
    │                                      ▼
    │                           T_slamworld_native
    │                                      │
    └── T_body_native ─────────────────────┤  compose with inverse(T_body_native)
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

The inverse is required because calibration stores the transform from the engine-native
frame into `Body`, while pose composition needs the opposite direction. Other calibration
fields have already affected the estimate through engine initialization.

For example, when the OAK-D center/RGB optical frame is the configured body and an engine
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

An engine may advertise a supported subset of the canonical pinhole models and impose
tighter constraints through `validate_configuration()`. Exact records, coefficient ordering,
normalization, and validation rules are in the
[Canonical calibration reference](#appendix-canonical-calibration-reference).

## Pose estimation engine selection and reference roles

One application build links exactly one provider implementation: the stub by default, or a
selected engine adapter package. An unknown or unavailable selection fails explicitly;
there is no fallback. The application calls a generic `make_pose_provider()` factory returning
`std::unique_ptr<core::IPoseProvider>`. Every engine reports its name, library version, and
supported capabilities through the common API. The recording stores that identity and the
Isaac Capture application build separately.

| Pose estimation engine | Role | Boundary |
|---|---|---|
| [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) | Proven OAK-D stereo and stereo-inertial baseline for development validation | User-supplied engine under its applicable terms. A proposed NVIDIA-authored example adapter, including its source distribution and license, is pending OSRB review. |
| [cuVSLAM](https://github.com/nvidia-isaac/cuVSLAM) | Intended production engine | Confirm the exact SDK and applicable license/review scope, and qualify its advertised modes, calibration conversion, redistribution terms, CUDA requirements, and target platforms. |
| `Stub` | Deterministic conformance engine with no external dependency | Supports the full initial contract and exercises trajectories, tracking states, source matching, draining, gaps, backpressure, and map-frame invariants. |

Each adapter advertises only the input modes, pixel formats, and distortion models it can
accept without approximation, and converts the canonical calibration and frame convention
to its engine API.

### Build and distribution boundary

The core public headers, application wiring, and generic build hooks do not refer to native
engine APIs. A selected adapter package supplies the factory implementation and owns native
headers, API calls, and SDK discovery. A user builds the chosen engine and an approved or
externally supplied adapter, then links them into a matching application build. Installing
an engine library alone does not turn a stub binary into that engine. This is a coordinated
C++ source/build integration, not a runtime plugin loader or stable binary-plugin ABI.

The OSRB request has been filed; adapter publication and licensing remain pending. The
default build and release artifacts exclude ORB-SLAM3 source, headers, binaries, assets, and
ORB-dependent compiled adapters or applications, including containers, wheels, and system
packages. User-built or source-only arrangements do not establish approval. Any NVIDIA-authored
ORB-SLAM3 example adapter requires the review outcome before publication; the surrounding
Apache-2.0 components retain their existing license. The generic example below can be reviewed
and exercised with the stub while those decisions remain open.

### Engine adapter obligations

Every adapter documents and qualifies the following behavior for its exact library build:

| Concern | Required adapter contract |
|---|---|
| Live output | Identify the selected native output, such as odometry or loop-corrected SLAM, and its cadence. Record the choice with the setup. Missing estimates remain missing; the adapter does not invent per-frame poses. |
| Tracking state | Map native status to `TrackingState`; explain any derived status and how permanent failure becomes fatal. |
| World and body frames | Detect startup frame changes and later resets or re-anchoring. Define native-to-body conversion, including rectification. |
| Calibration | Use all coefficients of an advertised model, possibly through an explicitly qualified image-remapping path. Keep the applied canonical setup immutable; any internal online estimates are additional engine diagnostics. |
| IMU gaps | Receive each provider gap notification before the next accepted sample and qualify the resulting tracking behavior. |
| Finishing | Define when engine work is complete, how worker threads are joined, and how abort terminates or isolates a non-cancellable call. |
| Repeatability | State whether repeated runs can be deterministic and which settings alter the computation path. Exact recorded inputs alone do not promise identical floating-point trajectories. |

The [source survey](pose-provider-backend-survey.md) provides evidence for adapter planning,
not conformance certification. For the two reference engines, it identifies these concrete
implementation requirements:

- **ORB-SLAM3.** Account for startup gravity alignment and later map changes; convert the
  rectified camera frame correctly; translate the canonical distortion model without
  truncation; generate native settings from the resolved calibration. The surveyed upstream
  API lacks the map identity and join guarantees available in the internal OAK-D fork.
  Choosing that fork or providing equivalent adapter hooks is a decision gate. A fixed
  elapsed initialization delay alone is not proof that no later frame change can occur.
- **cuVSLAM.** Preserve the selected odometry or SLAM frame and stop valid output after a
  frame-breaking tracking failure unless return to the original frame is demonstrated.
  Translate `RadTan5` coefficient order to the native Brown model, use the native rational
  model for `Rational8`, and qualify any other remapping path. Apply the IMU time offset at
  engine ingestion without changing recorded raw timestamps.

End-of-run history extraction, unbounded pose-graph retention, and a new
`try_get_refined_pose()` method are proposals for a future output extension. They are not
required to implement the current live provider contract.

## Pose publication and timing

### Existing transport mapping

The data collection coordinator's output integration drains ready estimates and maps them to
the existing [`se3_tracker` schema](../../src/core/schema/fbs/se3_tracker.fbs). This integration
does not change the schema or recorded channel names. The concrete initial coordinator is
identified under [Initial OAK-D integration](#initial-oak-d-integration).

Conceptually, the existing transport contains these small records:

```text
Se3TrackerPose        { pose: position + orientation, is_valid }  // live API
Se3TrackerPoseRecord  { data: Se3TrackerPose, timestamp }         // recording
```

`Se3Tracker` returns a nullable `Se3TrackerPose` handle (`None` in Python); it is empty
when no sample is available. There is no separate `Se3TrackerPoseTracked` schema type.

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
[Pose transport naming](#pose-transport-naming). Only fresh engine estimates are published:
the integration neither repeats an old pose nor fabricates an invalid estimate when no
output is available. Missing output remains a gap in the recorded per-sample pose stream.
A live `Se3Tracker` retains its last sample and does not expose a timestamp, so its value
alone cannot establish freshness. Live health must expose missing or stalled output; the
status channel and thresholds are identified under
[Implementation decision gates](#implementation-decision-gates).

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
available to a live consumer or recorder. This is not the time when the engine finishes
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
device timestamps. Capture withholds and counts a duplicate before calling `ISensorSink`,
without ending the episode; it records that outcome separately from a provider rejection.
If invalid duplicate input reaches the provider, `RejectedInvalidInput` retains its fatal
contract-violation meaning. A timestamp
regression within that source stream fails the episode because its timeline is no longer
trustworthy. This differs from cross-stream lateness: a source may remain internally
monotonic but arrive after the fusion timeline has advanced, in which case the
[capture-integration contract](#capture-integration-requirements) withholds and counts the
item. Unknown or mismatched clock domains are rejected before provider initialization. Clock
mapping, camera pairing, skew estimation, and drift correction remain capture
responsibilities.

## Recording requirements and offline re-estimation

One provider run produces one recording episode under the current lifecycle. Its recording
must preserve the live result and the evidence needed to reconstruct the input and setup.
The capture/recording integration owns storage; the provider has no recorder dependency.
Lossy video for viewing may coexist with the lossless provider inputs but cannot replace them.

### Required recording contents

| Content | Required information |
|---|---|
| Setup | Instance configuration, applied capture specification, fully resolved calibration and value provenance (including defaults and overrides), provider identity, resolved engine settings and asset hashes, application build, and capture ordering policy and limits |
| Images and groups | Lossless delivered images with format and layout; group ID, sensor IDs, source sequences, and group and per-image device/local measurement times |
| IMU | Sensor ID, source sequence, acceleration, angular velocity, and device/local measurement times |
| Input outcomes | Measurement identity, ordered provider admission attempts and results, capture-side exclusions and their reasons, and the location of IMU gaps relative to accepted input |
| Live poses | Existing pose records, with validity and measurement/availability timestamps; source groups resolve unambiguously by measurement time |
| Final summary | Final provider and capture counters, successful or failed completion, failure reason, and any incomplete tail |

The initial `DataCollectionCoordinator` assembles the setup record from `SensorAdapter`'s
applied capture report, the resolved calibration, provider identity, and engine settings.
It supplies that record and the final summary to its recorder. `SensorAdapter` reports
capture exclusions and `RecordingAdmissionSink` records delivery results; the coordinator
preserves those outcomes in their actual admission order. The input ledger must distinguish
“recorded” from “accepted by the
provider,” including when recorder or provider failure interrupts an attempt.

Recording integration must supply durable correlations between input records, outcomes,
and the episode. A crash or forced stop may leave an incomplete tail; it must not produce a
successful final summary. The schema/encoding and exact writer process are implementation
decision gates. They must be settled before recording integration starts, because today's
tracker-only recording configuration does not supply these records.

### What a later offline tool can reconstruct

- **Reproducing the live run** uses the recorded setup and exact engine build, the accepted
  input sequence, and the same gaps experienced by the live engine. The admission ledger
  records observed decisions rather than asking a replay to reproduce queue pressure.
  A threaded engine may still differ numerically; qualification defines an appropriate
  trajectory tolerance.
- **Re-estimating** can use other recorded measurements, engine tuning, or another engine.
  It is a new provider run with its own `SlamWorld`, and must retain its input selection,
  setup, and relationship to the source recording.

New results are separate derived artifacts, preserving the original recording and live pose.
The future tool needs a lossless feeding/gap-injection strategy and may use engine-native
trajectory export. This design does not add blocking admission or a refined-output method
to the provider API merely to anticipate that tool.

## Telemetry and diagnostics

Telemetry here means the health and performance data needed to decide whether a collection
run is usable and to explain missing or delayed poses. It stays separate from pose output,
must not block capture or estimation, and does not change the existing `se3_tracker` schema.

| Source | Minimum data | What it explains |
|---|---|---|
| Existing pose stream | Every emitted pose is recorded once with `is_valid`, measurement timestamps, and availability time | Primary pose output and the source for counts, gaps, valid-tracking periods, and pose availability latency; telemetry does not create a duplicate per-pose stream |
| `ProviderHealth` | Dropped image groups and IMU samples, IMU gaps, rejected input, pose-output stalls, and fatal-error state | Provider input loss, queue pressure, invalid calls or data, and provider failure |
| Capture integration and data collection coordinator | Measurements offered, recorded, and accepted; fusion-time ordering policy and any applied bounds, forced emissions, or late input; timestamp violations; recorder pressure or failure; final session state and failure reason | Whether recording and estimation saw the same source data and where a run failed |

`ProviderHealth` counters start at zero for each provider run, increase until final drain,
and never reset during that run. `health()` is non-blocking and safe for the one documented
output/health consumer while input and engine processing run concurrently; additional
concurrent callers are not supported. `fatal_error` stays true after the first fatal provider
error. The coordinator derives its own input and output totals from enqueue results and
drained estimates rather than duplicating those totals in the provider API.

The initial OAK-D integration makes current health available for live inspection and records
the final provider, capture, and session summary with the episode. That summary is stored
with the applied capture and calibration, provider identity, engine configuration, and
Isaac Capture application build already required as episode setup. The pose records themselves
remain the time-based source for validity, gaps, and latency analysis.

Isaac Capture has no episode setup/metadata path today: `McapRecordingConfig` specifies a
filename and tracker channels, and the current implementation does not write MCAP metadata
or attachments. Recording integration therefore adds the setup and summary records described
above. Their encoding and writer placement remain explicit integration decisions.

A future need for high-rate queue depth, detailed tracking-state transitions, or
engine-specific metrics may add a separate time-series telemetry schema; those fields must
not become requirements for all pose estimation engines.

## Example: data collection with OAK-D and ORB-SLAM3

An operator records one manipulation take with a head-mounted OAK-D. They want its 6-DoF
pose during collection, the recorded live trajectory, and the exact stereo/IMU input needed
to evaluate the result offline. Each take is one provider run and one recording episode.
Starting the next take creates a new map; this workflow does not require multi-take sessions.

This is an integration example for design review. `oak_pose_capture` is the provisional name
of a new C++ executable. Its build options, configuration, factories, coordinator helpers,
and capture callbacks are illustrative proposed APIs, not existing compilable application
code. Calls on `IPoseProvider` and `SensorAdapter` follow the declarations in this document.
The ORB-SLAM3 adapter's publication remains subject to the review described under
[Build and distribution boundary](#build-and-distribution-boundary).

### A dedicated data-collection executable

The existing `camera_plugin_oak` executable keeps its camera/video workflow and CLI.
Its [`main.cpp`](../../src/plugins/oak/main.cpp) constructs `OakCamera` with a `FrameSink`,
then services capture until a stop signal arrives. The current sink writes H.264 video and
optionally writes frame metadata to MCAP or publishes it through OpenXR.

The new `oak_pose_capture` executable reuses shared OAK capture code. It does not spawn
`camera_plugin_oak`; it opens its OAK-D once and hosts the coordinator, recorder, input
adapter, provider, and pose output in that process. A separate application can consume the
published poses through `Se3Tracker` without handling camera buffers. These are alternative
applications for a device, not two processes to run against the same OAK-D:

```text
Shared OAK capture code (OakCamera / DepthAI)
    used by:
    ├── camera_plugin_oak     existing camera/video application
    └── oak_pose_capture     new data-collection application
```

Within the new process, the ownership relationships are:

```text
oak_pose_capture: main()
    └── DataCollectionCoordinator
        ├── OakCaptureSource
        │   └── OakCamera / DepthAI device and pipeline
        ├── EpisodeRecorder
        ├── SensorAdapter
        ├── RecordingAdmissionSink
        ├── IPoseProvider (build-selected adapter and engine)
        └── PoseOutput
```

The coordinator owns lifecycle, workers, and health reporting for these components.
`OakCaptureSource` is an illustrative application facade over shared capture code, not a
new generic capture interface. `OakCamera` needs raw synchronized stereo delivery, optional
IMU, and applied stream/calibration reports beside the encoded-video path. Its current
constructor starts the pipeline immediately; the DC path must permit setup inspection while
measurement delivery stays gated until recording and the provider are ready. Shared-code
extraction and these optional capabilities must preserve the existing camera application's
behavior.

The raw branch supplies exact provider input to recording. Decoding H.264 does not satisfy
that contract. Optional encoded video remains another branch from the same device owner.
The lossless writer, live-status mapping, and publication wiring remain integration work.

### Build and configure the integration

Select a provider package that implements the generic factory and meets the fixed-frame and
shutdown requirements. For ORB-SLAM3, the user supplies the engine and the selected adapter
package, subject to the pending review. Exact engine sources/hooks remain an
[implementation decision](#implementation-decision-gates). The proposed build hooks select
a provider package without native-engine headers or calls in the application:

```bash
cmake -B build -DBUILD_OAK_POSE_CAPTURE=ON \
    -DPOSE_PROVIDER_PACKAGE_DIR="/path/to/provider-package" \
    -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
cmake --build build --target oak_pose_capture --parallel
```

Omitting the package selection uses the stub. The package owns native SDK discovery;
these option names and the executable path below remain provisional.

```yaml
# oak_head.yaml
capture:
  device_id: "OAK_MXID"
  stereo_streams: [MonoLeft, MonoRight]
pose_provider:
  instance_id: oak_head
  input_mode: StereoImu
  body_frame_id: CAM_A  # center/RGB optical frame
  output_frame: SlamWorld
  engine_config_path: orbslam3_head.yaml
# calibration_override: measured_noise_and_timing.yaml  # optional
```

The integration resolves rig/sensor identities, actual geometry and rates, clock domain,
and qualified queue/reorder limits into `SlamInstanceConfig` and `AppliedCaptureSpec`.
`CAM_A` names the chosen body frame; calibration relates the delivered left/right optical
frames and IMU to it. The body need not be a delivered image stream. `orbslam3_head.yaml`
supplies tuning and assets; the adapter generates native calibration from `RigCalibration`.

For `StereoImu`, the coordinator resolves the device IMU extrinsic or board-design fallback
and documented IMU noise defaults. An optional override may replace noise or timing fields.
Changing any extrinsic requires the complete frame group. Missing required values, including
a usable timing offset, fail startup. Selecting `Stereo` removes IMU delivery and calibration;
neither mode requires an IMU-only extrinsic override.

### Application entry point

`main()` constructs the components and connects capture callbacks to the adapter. The
coordinator's preparation step connects the rest of the input path:
`SensorAdapter -> RecordingAdmissionSink -> IPoseProvider`.

```cpp
int main(int argc, char** argv)
{
    const auto options = parse_dc_options(argc, argv);
    auto stop_request = make_stop_request_from_signals();

    DataCollectionCoordinator dc(
        make_oak_capture_source(),
        make_pose_provider(),
        make_episode_recorder(options.episode_dir),
        make_pose_output(options.instance_id));

    dc.prepare(options);  // Inspect and validate; delivery stays disabled.
    auto& adapter = dc.sensor_adapter();

    dc.capture().set_callbacks({
        .on_image_group = [&adapter](OakImageGroup group) {
            adapter.on_image_group(std::move(group));
        },
        .on_imu = [&adapter](const OakImuSample& sample) {
            adapter.on_imu(sample);
        },
    });

    return dc.run_one_episode(stop_request).exit_code();
}
```

The capture facade dispatches both callbacks on one serialized input worker. The provider
never opens the camera or configures its streams. The signal bridge only requests finish;
shutdown work runs in normal coordinator code. Argument-error reporting and other terminal
UX are omitted from this sketch.

### Coordinator lifecycle

The coordinator takes ownership of the four components passed by `main()`. Its helpers below
show the coordination contract; worker, storage, and error-reporting implementations are
intentionally omitted.

```cpp
class DataCollectionCoordinator {
public:
    DataCollectionCoordinator(std::unique_ptr<OakCaptureSource> capture,
                              std::unique_ptr<core::IPoseProvider> provider,
                              std::unique_ptr<EpisodeRecorder> recorder,
                              std::unique_ptr<PoseOutput> output);
    ~DataCollectionCoordinator();  // Quiesce and join before destroying components.

    void prepare(const DcOptions& options)
    {
        capture_->prepare(options.capture);  // Measurement delivery remains gated.
        setup_ = resolve_setup(options, *capture_, *provider_);
        validate_generic_setup(setup_);
        const auto error = provider_->validate_configuration(
            setup_.config, setup_.capture, setup_.calibration);
        if (!error.empty()) {
            throw ConfigurationError(error);
        }

        admission_ = std::make_unique<RecordingAdmissionSink>(*recorder_, *provider_);
        adapter_ = std::make_unique<SensorAdapter>(
            setup_.capture, setup_.calibration, *admission_, setup_.adapter_limits);
    }

    auto& capture() { return *capture_; }
    auto& sensor_adapter() { return *adapter_; }

    EpisodeResult run_one_episode(StopRequest& stop_request)
    {
        try {
            recorder_->write_setup(setup_);
            recorder_->arm();
            provider_->initialize(setup_.config, setup_.capture, setup_.calibration);
            provider_->start();

            start_output_and_health_worker();
            start_input_worker_and_capture();
            wait_for_stop_or_failure(stop_request);

            finish_capture_and_flush_adapter();
            join_input_worker();
            provider_->finish_input();
            request_final_output_drain();
            join_output_worker();
            provider_->stop();
            return write_summary_and_finalize();
        } catch (...) {
            return abort_and_finalize(std::current_exception());
        }
    }

private:
    // Helper declarations and worker state omitted.
    std::unique_ptr<OakCaptureSource> capture_;
    std::unique_ptr<core::IPoseProvider> provider_;
    std::unique_ptr<EpisodeRecorder> recorder_;
    std::unique_ptr<PoseOutput> output_;
    AppliedEpisodeSetup setup_;
    std::unique_ptr<RecordingAdmissionSink> admission_;
    std::unique_ptr<SensorAdapter> adapter_;
};
```

`AppliedEpisodeSetup` is an illustrative integration record holding the immutable provider
configuration, capture specification, calibration, limits, identity, and provenance.
`DataCollectionCoordinator::resolve_setup()` obtains the applied capture report, uses
`SensorAdapter::make_applied_capture_spec()` to construct `AppliedCaptureSpec`, resolves
`RigCalibration` from device data and any permitted overrides, and snapshots provider
`identity()` and `capabilities()`. Validation, initialization, ordering, and recording all
consume those same resolved records.

`RecordingAdmissionSink` implements `core::ISensorSink`. For each ordered measurement it
obtains recorder acceptance, calls the corresponding provider ingestion method, and records
the `EnqueueResult` before the next admission. Recording failure prevents further admission
and fails the episode. `SensorAdapter` reports any exclusions before this gate so the ledger
also explains withheld input. At equal fusion time, IMU samples precede the image group.

The output worker is the sole consumer of `try_get_next_pose()`. It drains until false on
each tick, publishes through `PoseOutput`, and checks provider health and recorder outcomes.
`PoseOutput` applies the configured frame transform and maps tracking state to validity
without replacing measurement timestamps. The existing transport/recording path records
that published pose once, with availability time supplied by the receiving runtime; source
group identity remains internal, and recorded correlation uses measurement time. This
example publishes `SlamWorld` poses under `pose/oak_head`.

Three helper contracts keep finishing and failures unambiguous:

- `finish_capture_and_flush_adapter()` stops new capture, waits for in-flight callbacks,
  and runs `adapter.finish_capture()` on the still-live input worker. It returns only after
  `capture_drained()` is true and input delivery has quiesced; only then is that worker joined.
- The output worker stays active throughout input flushing. After `finish_input()`,
  `request_final_output_drain()` lets it exit only after `input_drained()` is true and a
  subsequent `try_get_next_pose()` returns false. A worker failure interrupts coordinator
  waits, including finish waits, and enters the abort path.
- `abort_and_finalize()` supports partial startup and failed workers. It cancels input and
  output work and quiesces callbacks and output pulls before calling provider `stop()`.
  Output cancellation exits without waiting for `input_drained()` and prevents pulls after
  stop. It joins started workers and marks the episode failed or incomplete while preserving
  the original reason. Destructor cleanup also handles failed preparation and is safe after
  completion.

### Collect one take and inspect the result

The proposed launch shape runs one episode, with the device selected by the configuration:

```bash
./build/bin/oak_pose_capture \
    --pose-config=oak_head.yaml --episode-dir=episodes/take_001

# After the first take has drained and finalized, start a new provider run.
./build/bin/oak_pose_capture \
    --pose-config=oak_head.yaml --episode-dir=episodes/take_002
```

The operator observes initialization/tracking health and requests coordinated finish with
a stop signal. A launcher, if used, must allow the qualified drain duration. The next
invocation uses the same calibration and configuration but creates a new `SlamWorld`.

Each episode contains the resolved setup and calibration provenance, lossless stereo input,
optional IMU samples, ordered admission outcomes, live poses, and final health summary.
The image/container encoding and writer placement remain recording integration decisions.
Existing `Se3Tracker` replay exposes the recorded live pose. The inputs enable a later offline
reproduction or optimized re-estimation tool; that tool remains deferred. Separate takes
have independent map frames, so their trajectories are not automatically spatially aligned.

Selecting cuVSLAM changes the provider package, application build, and settings. The input,
calibration, lifecycle, recording, and pose-consumer contracts stay the same, subject to that
adapter's capabilities, qualification, and approved SDK/distribution configuration.

## Integration and verification

Verification is divided by ownership so provider behavior, capture integration, physical
hardware, and engine-specific behavior can be tested independently.

### Provider conformance

The deterministic stub is the common contract-test target. The same tests run against each
enabled engine where applicable.

| Area | Required outcome |
|---|---|
| Lifecycle | Only the [documented lifecycle transitions](#lifecycle) succeed; graceful shutdown quiesces input before `finish_input()` without `RejectedNotRunning`; abort cleanup may count that result without replacing the original failure; final draining completes without deadlock or loss of an already-produced estimate. |
| Configuration and input validation | Unsupported capabilities, malformed image groups, invalid [canonical image buffer layouts](#canonical-image-buffer-layouts), mismatched clocks or geometry, and invalid calibration fail before engine ingestion. Supported calibration is preserved without truncation or approximation. |
| Concurrency and backpressure | One producer and one output consumer operate without races. Queues stay bounded, image groups remain atomic, individual IMU drops and gap episodes follow [Admission outcomes and backpressure](#admission-outcomes-and-backpressure), and pose-output pressure neither blocks capture callbacks nor discards an already produced pose. |
| Pose semantics | Estimates preserve their source group and measurement timestamps. A missing estimate produces no repeated or fabricated pose, and one run never silently changes its `SlamWorld` frame. |
| Frames | Known trajectories verify body and optional session transforms; invalid frame configurations fail at startup. |
| Provider health | Health counters are cumulative, pose-output stall transitions are counted once, the fatal flag stays set, and non-blocking snapshots match the injected outcomes. |
| Input order | Accepted image and IMU inputs reach the engine adapter in admission order across queue boundaries, including equal-time ties and an IMU gap immediately before an image. |

### Capture and recording integration

Run a common integration suite using deterministic recorded or synthetic sensor fixtures.
Apply it to the initial OAK-D integration and future capture integrations that satisfy the
[capture-integration requirements](#capture-integration-requirements).

- Verify recorder acceptance precedes provider delivery, every provider input is recorded,
  and every emitted pose correlates with exactly one recorded image group.
- Verify one applied capture description and resolved calibration drive input ordering,
  provider initialization, and reconstructable episode metadata.
- Verify a frame override replaces the complete frame group and that partial groups,
  including an IMU-only extrinsic override, fail startup. Verify independent noise/timing
  overrides take precedence over device values or integration defaults, and record the
  effective calibration and its provenance.
- Verify grouping, fusion-time ordering, and timestamp-failure rules before engine ingestion
  while preserving raw measurement timestamps and recording any applied bounds and counters.
- Verify pose publication and replay through the existing transport preserve measurement and
  availability times, derived collection naming, and unchanged sensor channels.
- Verify invalid input and recording or provider failures produce a failed episode, while
  coordinated shutdown preserves every accepted input and produced estimate.
- Verify recorded pixel buffers and IMU values reconstruct the delivered measurements
  losslessly, and the ordered admission ledger reconstructs accepted input and gaps.
- Verify duplicate images are excluded at capture without a provider-invalid-input result;
  late equal-time IMU input cannot overtake an already-emitted image.
- Verify abort terminates even when the output consumer has failed, and that incomplete
  recordings cannot be mistaken for successful episodes.
- Verify the new DC application gates delivery until setup and startup succeed, and shared
  capture changes preserve the existing `camera_plugin_oak` camera/video behavior.

For the initial OAK-D adapter, deterministic fixtures also verify the
[reorder-buffer policy](#initial-oak-d-reorder-buffer-policy). This algorithm-specific test is
not a requirement on future capture implementations that meet the same ordering outcomes.

This suite will run routinely in CI and does not require physical hardware.

### Initial OAK-D hardware qualification

Run live OAK-D sessions in both `Stereo` and `StereoImu` modes:

- Verify one process owns the device and recording pipeline; no pose component reopens it.
- Verify `OakCamera` reports the actual applied stream configuration and calibration.
- Verify device IMU extrinsics take precedence over board-design fallback, that either
  source matches the delivered camera frames, and that missing required extrinsics fail
  startup. Qualify the documented IMU noise defaults and record their values and provenance.
- Verify synchronized stereo frames have equal, strictly increasing device timestamps.
- Verify an IMU `time_offset_ns` override takes precedence over the self-report, and that it
  and any crop, resize, or rectification are reflected in the applied calibration and recorded
  episode.
- Run a live smoke test of recording, pose publication, coordinated shutdown, and subsequent
  replay.

These checks require physical hardware and are not required for every CI run.

### Pose estimation engine qualification

- Compare the ORB-SLAM3 adapter with the proven OAK-D reference pipeline using the same
  recorded sensor data.
- Qualify cuVSLAM independently for every input mode it advertises.
- Verify each engine's calibration and frame conversions with known transforms and
  trajectories.
- Verify startup gravity alignment, map reset, and re-anchoring never produce a valid pose
  in a different `SlamWorld` within one run.
- Qualify each stereo-inertial engine's tracking behavior and recovery after an IMU gap.
- Use a test harness to reconstruct the recorded accepted input and compare the resulting
  trajectory under a declared engine-specific tolerance. This verifies recording completeness
  without committing to a user-facing replay tool.
- Verify the default build uses the stub and generic application sources contain no native
  engine dependencies. Verify ORB-SLAM3 and ORB-dependent binaries are absent from default
  artifacts; adapter source publication and any additional distribution require the
  applicable review outcome.

## Implementation impact

Implementation adds a core Pose Provider module, engine adapters, and a new C++ DC executable.
The initial OAK-D integration reuses and evolves shared capture components while preserving
the existing `camera_plugin_oak` executable's behavior. Exact file placement, target names,
and dependency wiring belong in the implementation plan.

| Area | Implementation effect |
|---|---|
| Core Pose Provider | Add the interfaces, normalized sensor types, configuration, calibration, validation, lifecycle, and health defined in this document. |
| Engine adapters | Add the deterministic stub and generic factory/package selection. Plan cuVSLAM and ORB-SLAM3 adapters against the same contract, subject to the qualification and review boundaries in [Pose estimation engine selection and reference roles](#pose-estimation-engine-selection-and-reference-roles). |
| DC application | Add the C++ entry point and coordinator composition shown in the example. Reuse shared OAK capture code without launching or changing the CLI of `camera_plugin_oak`. |
| Existing pose transport | Reuse `SchemaPusher`, `Se3Tracker`, recording, and replay without changing their schemas or consumer APIs. |
| Recording integration | Add setup and final-summary records, lossless provider-input records, and an admission ledger. Resolve encoding and writer ownership before implementation; do not assume a pre-existing metadata path. |
| Live status and shutdown | Expose pose freshness and run failure, and qualify the application's signal-to-finish path. Any launcher integration must support the same drain duration; status mapping and launcher handshake remain decision gates. |

### Initial OAK-D integration

| Component | Change | Responsibility |
|---|---|---|
| `OakCamera` | Reuse and evolve shared component | Remain the sole device and DepthAI-pipeline owner. Add raw stereo delivery, optional IMU, calibration readout, and stereo grouping beside the current H.264 path. Report capture-native applied configuration and calibration without depending on Pose Provider types. Preserve the existing camera application's behavior. |
| `OakCaptureSource` | Proposed application facade | Expose capture preparation, applied reports, gated delivery, serialized callbacks, and quiescence for the DC coordinator. Wrap shared OAK capture code; do not define a generic capture API. |
| `SensorAdapter` | New component | Convert OAK-D capture output into provider input, construct `AppliedCaptureSpec`, apply bounded fusion-time ordering, forward to its destination, and report exclusions and ordering outcomes. |
| `RecordingAdmissionSink` | New `ISensorSink` implementation | Enforce recorder acceptance before provider admission and record each admission result. Owned by the coordinator and supplied to `SensorAdapter` as its destination. |
| `DataCollectionCoordinator` | New application coordinator | Own capture, recorder, adapter, admission gate, provider, and output. Construct setup and summary records, resolve calibration, and coordinate workers, provider lifecycle, pose publication, telemetry, and shutdown. |

`OakCamera` reports the applied capture-native stream properties and device calibration
without depending on Pose Provider types. `SensorAdapter` converts that report into
`AppliedCaptureSpec`; `DataCollectionCoordinator` applies any file override to the device
calibration and resolves the final `RigCalibration` once.

`SensorAdapter` also merges complete image groups and IMU samples into one nondecreasing
fusion-time stream using bounded buffering selected from the applied rates and qualified
delivery jitter. It preserves raw measurement timestamps, records the applied bounds and
ordering counters with the episode, and flushes buffered input during coordinated shutdown.
The exact admission rule is in the
[Initial OAK-D reorder-buffer policy](#initial-oak-d-reorder-buffer-policy).

The initial integration uses this declaration-level shape. The capture-native types remain
owned by the OAK integration and may change as `OakCamera` evolves:

```cpp
namespace plugins::oak {

// Capture-native types owned by the OAK integration.
struct OakAppliedCaptureReport;
struct OakImageGroup;
struct OakImuSample;
struct SensorAdapterLimits;
struct SensorAdapterStats;

class SensorAdapter {
public:
    // Convert OakCamera's applied stream report into the provider contract.
    static core::AppliedCaptureSpec make_applied_capture_spec(
        const OakAppliedCaptureReport& report);

    SensorAdapter(const core::AppliedCaptureSpec& capture,
                  const core::RigCalibration& calibration,
                  core::ISensorSink& destination,
                  const SensorAdapterLimits& limits);

    // Normalize, fusion-time order, and forward captured measurements.
    void on_image_group(OakImageGroup group);
    void on_imu(const OakImuSample& sample);

    // Flush buffered measurements after capture delivery has stopped.
    void finish_capture();
    bool capture_drained() const;

    // Ordering, late-input, and delivery counters for session telemetry.
    SensorAdapterStats stats() const;
};

} // namespace plugins::oak
```

`SensorAdapter` holds and calls an `ISensorSink` destination; it does not implement
`ISensorSink`. Its capture-facing types and internal buffering remain OAK integration details.
The `plugins::oak` namespace reflects source ownership and does not require hosting in the
existing plugin executable.

The initial OAK-D integration runs in the new DC executable, with one camera owner and one
provider run per episode. Internal class or file organization may change as long as these
ownership boundaries remain intact.

## Implementation decision gates

The declarations and settled contracts above are the basis for the core/stub implementation.
The following choices must be resolved before their affected integration work can be planned
as complete. Proposed alternatives do not override this document's current lifecycle, API,
calibration, or naming rules.

| Decision | Current contract or constraint | Work gated |
|---|---|---|
| Capture-side IMU loss | Provider queue loss has an internal gap handoff. Withheld or missing capture input currently has no declared path to that handoff. Choose an explicit gap signal or fail the run before continuing; do not silently integrate across it. | Stereo-inertial capture integration and offline reconstruction |
| Recording format and writer | Preserve the required content above. Choose lossless image storage, record schemas, writer process, crash recovery, and how pose records enter that recording. | Complete recording integration |
| Live pose semantics | Each adapter must declare its output. Decide whether the production cuVSLAM adapter publishes odometry or loop-corrected SLAM; loop corrections may jump within one frame. | Engine output configuration and qualification |
| ORB-SLAM3 build | The fixed-frame and shutdown guarantees are mandatory. Choose the internal fork or equivalent hooks; stock upstream is not presumed conformant. | ORB-SLAM3 adapter |
| Adapter licensing and distribution | OSRB ticket filed; NVIDIA-authored ORB-SLAM3 adapter source/license and distribution remain pending. Confirm the selected cuVSLAM SDK's license and applicable review scope. | Engine-specific adapter publication and release packaging |
| Graceful stop | Qualify the new application's signal-to-finish path. If integrated with PluginManager, its current SIGINT-to-SIGKILL limit is two seconds; choose a configurable deadline or finish handshake that permits draining. | Shutdown qualification and optional launcher integration |
| Live health | Last live pose alone cannot establish freshness. Choose a status mapping and measured silence/drop thresholds; the existing device-status channel is a candidate if plugin-launcher integration is selected. | Operator-facing readiness and failure reporting |
| Namespace integration | Current collection is `pose/<instance_id>` and `Session` is a fixed user output frame. Confirm OpenXR consumer-frame assumptions and, if used, plugin-prefix integration. | Publication wiring and consumer validation |
| Hardware scope and bounds | One OAK-D is the current qualification target. Measure supported rates, reorder bounds, queue capacities, and recording throughput; decide whether coordinated two-OAK-D operation is required. | Hardware acceptance criteria |

`Session` in this design does not by itself mean the OpenXR session space. A consumer that
requires that space needs an explicitly established transform. If the integration cannot
satisfy the existing consumer's frame assumptions, it must fail validation rather than
publish a differently framed pose under the same assumptions.

## References

- [Ego Data Capture Interface](https://docs.google.com/document/d/1C56pIKonEo3HbCZKl5OffWj65qrx7GBy40X5GK5tntc/edit) —
  companion design for sensor capture, synchronization, calibration, and recording
  responsibilities.
- [SLAM engine survey](pose-provider-backend-survey.md) — source-based findings at pinned
  commits, including native frames, map behavior, completion, and licensing boundaries.

## Appendix: Operational contract details

These mechanics support the main contracts but are separated from the primary design flow so
implementers can change internal structure without changing observable behavior.

### Initial OAK-D reorder-buffer policy

The initial `SensorAdapter` uses a nonnegative fusion-time window and a finite positive item
capacity, configured from the applied stream rates and qualified delivery-jitter margin.
It normally emits an item when its fusion time is no greater than the greatest observed
fusion time minus that window. Ordering uses fusion time, then IMU-before-image priority,
then source sequence for equal-time IMU samples. After per-source timestamp validation, an
otherwise valid item preceding the last emitted item under that complete ordering is late,
counted, and withheld before buffering.

When admitting an on-time item would exceed capacity, the adapter emits the earliest item from
the union of the buffer and the incoming item. If the incoming item is earliest, it is emitted
directly; otherwise, the earliest buffered item is emitted and the incoming item is inserted.
This guarantees progress without rejecting an on-time item solely because the reorder buffer
is full. Normal shutdown flushes buffered items in order.

The setup records the window and capacity; the final summary records forced emissions and
late inputs, and the admission ledger records each exclusion. Capacity-forced progress may
make a later arrival late even when it is within the nominal time window. If no new input
arrives, the remaining tail waits until more input advances the watermark or
`finish_capture()` flushes it; this policy does not promise a wall-clock latency bound.

### IMU gap handoff

Every dropped IMU sample increments `dropped_imu_samples`. The first drop before the next
accepted IMU sample opens one gap event and increments `imu_gap_events`; further drops in that
interval only increment the sample count. Before ingesting the next accepted sample, the Pose
Provider implementation notifies the engine adapter that a gap occurred. `ISensorSink` does
not expose this internal notification, and the adapter owns the resulting tracking behavior.

### Pose transport naming constraints

`instance_id` is unique among active providers in one runtime, begins with an ASCII letter or
digit, and otherwise uses only ASCII letters, digits, `.`, `_`, and `-`. All participants use
the same helper to derive the collection name:

```cpp
std::string make_pose_collection_id(std::string_view instance_id);
```

Startup rejects a derived name longer than `XR_MAX_TENSOR_IDENTIFIER_SIZE - 1` bytes—255 bytes
with the current OpenXR extension header, excluding the terminating null byte.

### Canonical image buffer layouts

`ImageFrame::stride` is measured in bytes. Each frame must use the layout declared by its
`PixelFormat`:

| Format | Layout | Stride and minimum `data.size()` |
|---|---|---|
| `Gray8` | One packed byte per pixel | `stride >= width`; size at least `stride * height` |
| `BGR888` | One packed three-byte pixel | `stride >= 3 * width`; size at least `stride * height` |
| `RGBA8` | One packed four-byte pixel | `stride >= 4 * width`; size at least `stride * height` |
| `NV12` | Contiguous Y plane followed immediately by an interleaved UV plane | Width and height are even; one shared `stride >= width`; UV starts at `stride * height`; size at least `stride * (height + height / 2)` |

The provider rejects an invalid stride, plane layout, or undersized buffer as
`RejectedInvalidInput` before engine ingestion.

## Appendix: Canonical calibration reference

This appendix defines the normalized calibration passed to every pose estimation engine.
Capture-specific loaders normalize device records; the capture integration resolves
overrides before provider initialization. Engines consume the canonical result.

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
  maps to the smallest listed pinhole model whose omitted trailing coefficients are all
  exactly zero. This is lossless normalization, not coefficient truncation or fitting.
  Fisheye and other projection families are deferred.
- Startup rejects a distortion model not advertised by the selected engine. An engine may
  impose tighter constraints through `validate_configuration()`, but its adapter must not
  silently discard coefficients.
- `RigCalibration::imu` is present in `StereoImu` mode and absent in `Stereo` mode. When
  present, its sensor ID and rate match `AppliedCaptureSpec::imu`; an engine may impose
  tighter noise, rate, or stream constraints.
- Initial OAK-D input is treated as global shutter; rolling-shutter timing models are
  deferred. Changing rectification or `body_frame_id` requires newly resolved calibration
  and a new provider run.
- The complete resolved calibration and its value provenance, including integration defaults
  and any file override, are recorded with the data-collection episode.
