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

- Define a small backend-neutral `IPoseProvider` contract that maps synchronized stereo and
  optional IMU measurements to one 6-DoF pose stream.
- Make the initial release independently implementable with the proven OAK-D capture path;
  it must not depend on the unfinished generic capture-interface design.
- Preserve capture measurement time and provenance through the provider so recorded sensor
  data and recorded poses remain correlatable.
- Keep sensor ingestion non-blocking, bounded, atomic for stereo, and explicit about drops
  and IMU gaps.
- Define one canonical calibration and frame convention at the provider boundary. Vendor
  capture implementations and SLAM backend adapters perform all conversions.
- Keep one stable SLAM world frame for a running session; a backend may not silently create
  a new map origin under the same fixed alignment.
- Reuse `core::Se3TrackerPose` and the existing `SchemaPusher`/record/replay path so no
  consumer changes are required.
- Use ORB-SLAM3 as the proven functional reference, cuVSLAM as the production reference,
  and a stub as the deterministic contract-test backend.
- Leave a narrow capture seam that a future generic capture implementation can satisfy
  without making that future interface an initial-release dependency.

## Initial release scope

The initial release has one supported input topology and two input modes:

| Topology | Input mode | Pose output |
|---|---|---|
| One OAK-D stereo pair | `Stereo` | One pose stream |
| One OAK-D stereo pair + its onboard IMU | `StereoImu` | One pose stream |

Normative initial-release constraints:

- One physical OAK-D, one raw device-clock domain, one rigid body, and one process owning
  capture, recording, and pose estimation.
- Exactly two camera streams, identified as `left` and `right`; `left` is the body/reference
  camera.
- The IMU is optional. When `StereoImu` is selected, the same IMU samples delivered to the
  provider must also be recorded.
- The capture layer records the same identified stereo captures that feed the provider.
- Owned CPU image buffers; no device/GPU buffer lifetime crosses the provider boundary.
- Data collection only. Correct measurement timestamps matter; low pose availability
  latency does not.
- One fixed backend per binary.
- One `OutputSpec`, one `SchemaPusher`, and one `se3_tracker` collection.
- `output_frame: SlamWorld` is the default. `output_frame: Session` uses a fixed transform
  loaded at startup and is valid only for a repeatable physical start pose.

## Planned extensions

These are ordered integration directions, not initial-release requirements or compatibility
promises:

1. **Generic capture-interface migration.** Replace direct OAK capture integration after
   that interface can provide the capture outcomes listed below: atomic stereo groups,
   optional IMU, two timestamps, calibration, stable provenance, recording fan-out, and
   coordinated lifecycle.
2. **Additional stereo devices.** Add capture implementations for devices such as ZED and
   NVIDIA Sensing/GMSL without changing `IPoseProvider`.
3. **Mono input.** Add and validate a mono input mode only when a backend/device use case
   requires it.
4. **Multi-adapter rigid rigs.** Add a `SensorMultiplexer` only when one provider must
   consume streams from more than one capture adapter.
5. **Larger or cross-clock rigs.** Design a general atomic image-group type for 3+ cameras,
   plus hardware-sync/clock-mapping health contracts.
6. **New consumption modes.** Design forward prediction and live-control safety separately;
   design collaborative/shared-map multi-rig operation separately.

Adding a planned extension may change interfaces. The initial release stabilizes the seams
it exercises; it does not claim that every future topology is non-breaking.

## Non-goals

- Designing `ICamera`, `IImu`, capture fan-out, vendor synchronization, or recording file
  formats.
- Implementing or configuring DepthAI Sync, ZED synchronization, GMSL triggers, PTP, or
  software clock estimation in the Pose Provider.
- Supporting mono, non-OAK cameras, multiple physical devices, 3+ cameras, or multiple clock
  domains in the initial release.
- Real-time robot control, forward prediction, a latency profile API, or tracking-loss
  safety policy for live motion.
- Collaborative/shared-map SLAM, cross-machine image transport, or dynamic multi-rig
  alignment.
- In-process map reset, dynamic Session alignment, or a map epoch on the existing
  `se3_tracker` schema.
- GPU/zero-copy image transport.
- Shipping, vendoring, downloading, or redistributing GPL ORB-SLAM3 through the normal
  release build.
- A VINS backend in the initial release.
- A new consumer-facing pose schema. Detailed tracking state, covariance, map quality, and
  capture-group identity remain internal because `se3_tracker` cannot carry them.
- Camera consumers unrelated to pose estimation, such as detection. A capture session may
  fan out to them, but they do not affect this provider contract.

## Architecture and ownership boundary

```text
Capture layer / data-collection session (outside this design)
  owns one camera device and its vendor pipeline
        │
        ├──▶ Sensor recorder
        │      stereo video + frame metadata + optional IMU
        │
        └──▶ SensorAdapter
               fusion-time-ordered, owned stereo groups + optional IMU
                              │
                              ▼
                       IPoseProvider impl
                    (ORB-SLAM3 / cuVSLAM / stub)
                              │
                              ▼
                         PoseEstimate
                              │
                     output-frame transform
                              │
                              ▼
                    Se3TrackerPose + timestamps
                              │
                       SchemaPusher / recorder
```

There are three independent seams:

1. **Capture seam.** The capture layer owns devices, synchronization, sensor recording, and
   fan-out. `SensorAdapter` translates its output into the normalized provider types below.
2. **Estimator seam.** `IPoseProvider` owns SLAM/VIO state and backend-native translation.
3. **Transport seam.** Existing `se3_tracker` transport, recording, replay, and consumers
   remain unchanged.

For the initial OAK-D integration, the existing OAK capture process remains the single
device owner. Recording and pose estimation branch from one OAK pipeline. A separate pose
process must not reopen the same device. How the existing `OakCamera` is refactored into a
capture-session owner is an OAK/capture implementation concern, not part of
`IPoseProvider`.

## Contract required from the capture layer

The future generic capture interface and the direct initial OAK integration must provide
the same observable outcomes.

### Capture identity and synchronization

- Every stereo capture has a `capture_group_id` that is unique within its recording episode.
- Left and right frames carry their vendor source sequence numbers.
- The two frames are delivered atomically as one complete group. A partial group is never
  delivered.
- For initial OAK-D, the left and right `sample_time_device_ns` values must be exactly
  equal. The capture layer rejects and counts a pair that violates this rule.
- Both frames expose `sample_time_local_ns`. The group's local measurement timestamp is the
  left/reference frame's value.
- The capture layer reports one stable `clock_domain_id` for the OAK-D. Camera and optional
  IMU samples delivered to one provider instance must use that domain.
- IMU samples and stereo groups share one device clock domain. Delivery order uses the
  calibration-adjusted fusion-time rule defined under Input ownership, ordering, and
  concurrency while preserving every raw device timestamp.
- Host arrival time is not a measurement timestamp.

Future devices may form a synchronized group using a hardware trigger, SDK capture ID, or a
bounded skew rather than exact timestamp equality. That grouping rule belongs to the
capture implementation; the Pose Provider continues to receive one atomic group with a
declared reference measurement time.

### Recording correlation

The data-collection session, not `IPoseProvider`, owns recording. It must guarantee:

- The recorded stereo frames and the frames delivered to the provider describe the same
  capture groups.
- Recorded frame metadata preserves `capture_group_id`, source sequence numbers,
  `clock_domain_id`, and both timestamps.
- When `StereoImu` is selected, every IMU sample delivered to the provider is also recorded
  with the same identity and timestamps.
- The applied capture specification and resolved calibration are recorded with the episode.
- Recording and provider queues are independent and bounded, so SLAM compute cannot block
  capture.
- A drop on either branch is counted and attributable to a capture/sequence identity.
- Recorder acceptance precedes provider delivery. If the recording branch cannot accept a
  stereo group or IMU sample, that measurement is not delivered to the provider either.
- If the provider rejects a measurement after the recorder accepted it, the sensor data
  remains in the episode and no pose is expected for that group.
- A recorder failure discovered after provider delivery fails the data-collection session;
  the system must not silently retain poses whose source sensor data is missing.
- The episode records the complete resolved `SlamInstanceConfig` (including `OutputSpec` and
  `T_session_slamworld` when present), `AppliedCaptureSpec`, `RigCalibration`, provider
  identity, and exact backend configuration contents plus a digest. Recording only a
  backend-configuration path is insufficient.

The provider does not need to know how or where sensor data is recorded. Its responsibility
is to preserve the source capture identity and measurement timestamps into
`PoseEstimate`.

### Calibration and applied descriptors

Before the provider initializes, the integration supplies:

- One immutable `AppliedCaptureSpec` containing the applied left/right width, height, stride,
  pixel format, rate, sensor IDs, device clock-domain identity, and optional IMU descriptor.
- A fully resolved `RigCalibration` in the canonical convention below.
- The selected `ProviderInputMode`.

The capture layer determines how device self-report is obtained. An orchestrator may apply
a calibration-file override, but `IPoseProvider` receives only the final resolved records.
The applied descriptor and calibration remain fixed for one provider run.

## Interface 1 — provider input

```cpp
namespace core {

using SensorId = std::string;
using RigId = std::string;
using ClockDomainId = std::string;
using CaptureGroupId = uint64_t;

enum class PixelFormat { Gray8, NV12, BGR888, RGBA8 };

struct ImageFrame {
    SensorId sensor_id;
    uint64_t source_sequence;
    CaptureGroupId capture_group_id;
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    PixelFormat format;
    std::vector<uint8_t> data;       // owned CPU buffer
    int64_t sample_time_local_ns;    // local common monotonic clock
    int64_t sample_time_device_ns;   // raw capture-device clock
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

    // Both frames are one indivisible queue item. Initial OAK-D requires:
    // - matching capture_group_id
    // - matching sample_time_device_ns
    // - left.sensor_id/right.sensor_id match the initialized StereoRigSpec
    virtual EnqueueResult on_stereo_pair(ImageFrame left, ImageFrame right) = 0;

    // Called only in StereoImu mode.
    virtual EnqueueResult on_imu(const ImuSample& sample) = 0;
};

} // namespace core
```

### Input ownership, ordering, and concurrency

- `on_stereo_pair()` transfers ownership of both image buffers.
- `on_imu()` copies the small sample into the provider queue before returning.
- Calls are enqueue-only and do not execute SLAM compute.
- One `SensorAdapter` serializes all calls into an `ISensorSink` instance. Callbacks are not
  concurrent in the initial release.
- Delivery is monotonic by calibration-adjusted fusion time; the provider does not sort.
  Stereo fusion time is its raw `sample_time_device_ns`. IMU fusion time is
  `sample_time_device_ns + RigCalibration::imu.time_offset_ns`, following the convention
  `t_camera = t_imu + time_offset_ns`.
- The adjustment is an ordering/backend-input operation only. `ImuSample` and recorded
  metadata retain the original raw device timestamp.
- For equal adjusted fusion times, IMU samples are delivered before the stereo group so a
  backend has received all inertial data through the frame measurement time.
- Calls are legal only after `IPoseProvider::start()` returns and before the capture adapter
  becomes quiescent during shutdown.
- No sensor callback may arrive after `IPoseProvider::finish_input()` begins.

### Bounded backpressure

`SlamInstanceConfig::queues` sets finite capacities for stereo input, IMU input, and pose
output.

- A full stereo queue drops the incoming pair as one unit and returns
  `DroppedBackpressure`. It never splits a pair.
- A full IMU queue returns `DroppedBackpressure`, increments the IMU-drop counter, and
  forces the provider out of `Tracking` until the backend has recovered from the resulting
  gap.
- A full pose-output queue is a session error: dropping an already computed pose would make
  the recorded trajectory silently incomplete.
- Structurally invalid input is rejected before backend ingestion and returns
  `RejectedInvalidInput`.
- All drops, invalid inputs, and rejected-state calls are counted and exposed through
  `ProviderHealth`.

The initial capacities are configuration values, not ABI constants. Deployment defaults
must be selected from measured OAK-D/backend behavior before implementation release.

## Interface 2 — `IPoseProvider`

```cpp
namespace core {

enum class ProviderInputMode { Stereo, StereoImu };

enum class TrackingState {
    Initializing,
    Tracking,
    Lost,
    Relocalizing,
};

struct PoseEstimate {
    Pose pose;                          // T_slamworld_body
    TrackingState state;
    CaptureGroupId source_capture_group_id;
    uint64_t source_left_sequence;
    uint64_t source_right_sequence;
    int64_t sample_time_local_ns;       // source stereo group's reference time
    int64_t sample_time_device_ns;      // source stereo group's device time
    std::optional<std::array<double, 36>> covariance; // 6x6 row-major
};

struct ProviderCapabilities {
    std::vector<ProviderInputMode> input_modes;
    std::vector<PixelFormat> pixel_formats; // most preferred first
};

struct ProviderIdentity {
    std::string backend_name;
    std::string backend_version;
    std::string adapter_build_id;
};

struct ProviderHealth {
    uint64_t dropped_stereo_groups;
    uint64_t dropped_imu_samples;
    uint64_t rejected_invalid_input;
    uint64_t rejected_not_running;
    uint64_t pose_queue_overflows;
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
    // Returns estimates in production order; the caller drains until false each tick.
    virtual bool try_get_next_pose(PoseEstimate& out) = 0;
    virtual ProviderHealth health() const = 0;
};

} // namespace core
```

### Lifecycle and map-frame invariant

Startup order:

1. Load and validate `SlamInstanceConfig`.
2. Create the selected provider and inspect `capabilities()`.
3. Obtain the applied capture descriptors and resolved calibration from the capture
   integration.
4. Run generic validation, then `provider.validate_configuration()`.
5. Persist the complete resolved setup required by Recording correlation.
6. Arm the recorder, then `provider.initialize()` and `provider.start()`.
7. Start capture delivery. For every measurement, recorder acceptance still precedes
   provider delivery.

The producer checks `provider.health()` while running. `fatal_error` ends the session and
prevents it from being finalized as a successful recording.

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

The producer must drain output concurrently with step 3 so a finite output queue cannot
deadlock shutdown. `finish_input()` is idempotent. On a fatal-error/abort path, `stop()` may
be called before normal draining completes; it quiesces the worker and may discard queued
input, and the recording episode is marked failed. `stop()` is idempotent and is also valid
for cleanup after a partially completed `initialize()` or failed `start()`.

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
    SensorId left_sensor_id;            // body/reference camera
    SensorId right_sensor_id;
    std::optional<SensorId> imu_sensor_id;
    ClockDomainId clock_domain_id;
};

struct OutputSpec {
    std::string collection_id;
    enum class Frame { SlamWorld, Session } output_frame;
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

Generic startup validation rejects:

- Empty `instance_id`, `rig_id`, sensor IDs, clock-domain ID, or output `collection_id`.
- Equal left/right sensor IDs.
- `StereoImu` without `imu_sensor_id`.
- `Stereo` with an IMU stream wired into the provider or an IMU record in
  `AppliedCaptureSpec`; `StereoImu` without matching configured, applied, and calibrated IMU
  records.
- A selected input mode or pixel format absent from `capabilities()`.
- Applied descriptors with non-positive dimensions, stride, or rates; a stride too small for
  the selected format; or a clock-domain mismatch.
- Applied descriptors whose sensor IDs, dimensions, formats, or clock domain do not match
  the config and resolved calibration.
- Duplicate calibration records, an incorrect calibration `rig_id`, missing configured
  sensor records, or calibration records for unknown sensors.
- `output_frame == Session` without `T_session_slamworld`.
- Non-positive queue capacities.

## Canonical calibration and frame convention

The capture integration supplies one fully resolved record. Device loaders and backend
adapters are the only conversion points. Calibration describes the exact pixel arrays and
sample timing delivered through `AppliedCaptureSpec` and `ImageFrame`, not a sensor mode
upstream of crop, resize, ISP, or rectification.

### Frames and transforms

- All frames are right-handed.
- The rig body frame is the OAK-D left optical camera frame:
  `+X` right in the image, `+Y` down, `+Z` forward along the optical axis.
- `T_A_B` transforms coordinates expressed in frame B into frame A.
- `T_body_left` is identity.
- All 4x4 matrices are row-major homogeneous transforms in meters.
- `PoseEstimate::pose` is `T_slamworld_body`.
- `core::Pose` uses position in meters and a unit Hamilton quaternion ordered `(x,y,z,w)`,
  matching the existing `se3_tracker` schema.
- `T_session_slamworld * T_slamworld_body` produces `T_session_body`.

### Initial camera model

The initial release supports one canonical model:

```cpp
struct PinholeRadTanCalibration {
    double fx;
    double fy;
    double cx;
    double cy;
    // OpenCV radial-tangential order.
    double k1;
    double k2;
    double p1;
    double p2;
    double k3;
};

struct CameraCalibration {
    SensorId sensor_id;
    uint32_t width;
    uint32_t height;
    PinholeRadTanCalibration model;
    std::array<double, 16> T_body_camera;
};
```

Initial OAK-D camera streams are treated as global-shutter inputs. Rolling-shutter models
and other distortion models are planned extensions and are rejected by generic validation.

For each camera, calibration width and height must equal its applied stream descriptor.
Intrinsics and distortion coefficients describe the delivered pixel grid after every
capture-side crop, scale, and rectification step. If capture delivers rectified images, it
must supply the calibration of those rectified images (normally with zero residual
radial-tangential distortion), and `T_body_camera` must remain consistent with that
rectification convention. A mismatch fails startup; the provider never guesses or rescales
calibration.

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
`ProviderIdentity`; the adapter build ID identifies the exact IsaacTeleop adapter build, and
the backend version identifies the linked or developer-supplied SLAM library.

### ORB-SLAM3 — proven functional reference

- The existing OAK-D stereo and stereo-inertial pipeline is the behavioral baseline for
  initial integration.
- ORB-SLAM3 supports the two initial input modes and is useful for development validation.
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
- The backend adapter converts the canonical calibration and frame convention to the
  selected cuVSLAM API.
- Exact SDK version, redistribution, CUDA, and target-platform requirements are release
  qualification gates.

### Stub — contract-test reference

The stub has no external dependencies and supports both initial input modes. It is
scriptable to:

- Produce a deterministic trajectory.
- Emit every `TrackingState`.
- Hold one final estimate until `finish_input()` to exercise lossless draining.
- Simulate an IMU gap, queue overflow, and attempted map-frame reset.
- Preserve and echo capture provenance/timestamps for conformance tests.

## Output contract

The producer drains `try_get_next_pose()` each tick and maps each estimate to the existing
schema:

```cpp
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

`SchemaPusher::push_buffer()` receives
`estimate.sample_time_local_ns` and `estimate.sample_time_device_ns`. These are always the
measurement timestamps of `source_capture_group_id`, never the time at which SLAM completed
or the plugin published the estimate.

`se3_tracker` cannot carry `source_capture_group_id` or source sequences. In the initial
OAK-D recording, the exact raw-device measurement timestamp is therefore the wire-level
join key between:

- Recorded left/right frame metadata.
- The pose computed from that stereo group.

Initial-release validation requires one unique device timestamp per stereo capture group.
The internal `PoseEstimate` provenance remains available for tests and diagnostics. A
future need to transport provenance explicitly would trigger a schema extension or a new
pose schema.

### Tracking-state publication

- `is_valid` is true only in `Tracking`.
- `Initializing`, `Lost`, and `Relocalizing` publish an identity/unspecified filler with
  `is_valid=false`.
- A fresh invalid estimate uses its own source capture timestamps.
- The producer publishes only fresh estimates returned by the backend. It does not synthesize
  an invalid measurement or reuse an earlier capture timestamp when output is absent.
- Missing pose output remains a timestamp-visible gap in the recorded pose stream. Liveness
  monitoring may fail the data-collection session, but it must not fabricate a measurement.

Covariance and detailed tracking state are intentionally dropped at the `se3_tracker`
boundary.

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
- Initialization receives exactly one immutable applied-capture descriptor; descriptor,
  calibration, and frame-buffer geometry mismatches fail before backend ingestion.
- Stereo input is one atomic queue item; concurrent worker processing cannot observe a
  half-pair.
- Mismatched capture IDs, timestamps, or configured sensor IDs return
  `RejectedInvalidInput` before backend ingestion; mismatched clock domains fail integration
  validation before initialization.
- Input and output queues remain within configured capacities.
- Stereo overflow drops a complete pair and increments one group counter.
- IMU overflow/gap forces the backend out of `Tracking`.
- IMU callbacks are ordered by calibration-adjusted fusion time while their raw timestamps
  remain unchanged.
- Pose output preserves source capture ID, source sequences, and both measurement
  timestamps.
- Pose-output overflow fails the session rather than silently dropping a pose.
- `finish_input()` rejects new callbacks; the producer drains output concurrently until
  `input_drained()`, then observes an empty final output queue without deadlock or loss.
- A simulated internal map reset cannot emit a pose in a new map frame under the same run.
- Provider silence does not synthesize an output or reuse an old measurement timestamp.

### OAK-D data-collection integration

Using a recorded or live OAK-D episode:

- Run both `Stereo` and `StereoImu`.
- Confirm every provider stereo input corresponds to recorded left/right metadata with the
  same unique device timestamp and source sequences.
- In `StereoImu`, confirm every provider IMU input is present in the recorded episode.
- Confirm the recorded episode contains the resolved instance/output configuration, applied
  descriptors, calibration, provider identity, and exact backend configuration contents.
- Force recorder backpressure and confirm rejected sensor measurements never reach the
  provider.
- Confirm every emitted pose joins to exactly one recorded stereo group by device
  measurement timestamp.
- Exercise capture and provider backpressure; verify all gaps are attributable and no
  partial stereo pair is delivered.
- Exercise any capture-side crop, resize, or rectification and verify the recorded/applied
  calibration describes the exact delivered pixel geometry.
- Read the pose through the existing `Se3Tracker`/replay path without consumer changes.

### Backend qualification

- Compare the provider's ORB-SLAM3 path with the proven OAK-D reference pipeline on the same
  recorded episodes.
- Qualify cuVSLAM independently for both input modes it advertises.
- Confirm backend-native calibration/frame conversions against known transforms and
  trajectories.
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
- `src/core/pose_provider/backends/stub/` — deterministic conformance backend.
- `src/core/pose_provider/backends/cuvslam/` — production backend.
- `src/core/pose_provider/backends/orbslam3/` — developer-supplied functional-reference
  adapter, subject to the licensing boundary above.

The initial integration lives in the existing OAK capture process so one component owns the
device and recording pipeline. This design intentionally does not prescribe the OAK capture
class refactor or the future generic capture-interface implementation.
