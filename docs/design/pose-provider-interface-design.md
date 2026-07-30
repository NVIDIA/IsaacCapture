<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Pose Provider — Interface Design

## Context

IsaacTeleop needs 6-DoF pose from cameras that don't emit pose natively (e.g. an OAK
camera streams video only). We want a **Pose Provider** layer that runs SLAM/VIO on a
device's sensor inputs (image + IMU) and republishes the estimated pose into teleop.

Two hard requirements shape the design:
- **Swappable backends** — cuVSLAM (in-house, GPU), ORB-SLAM3, VINS-Fusion, etc. must be
  interchangeable without touching the plugin, the sensor layer, or the consumer.
- **Fits the existing architecture** — reuse the producer-plugin + `SchemaPusher` +
  `se3_tracker` path so no consumer-side work is needed.

Deliverable for this pass: **interface design only** — the C++ interfaces, data
structures, factory, and integration points documented here. No plugin/build wiring or
real SLAM library integration yet.

## What "v1" means

**v1** is the first shipped *implementation* of this design — not this design pass, which is
interfaces only. Its scope is deliberately narrower than the interfaces can express: the
types are shaped so the deferred items can be added later without breaking existing backends
or consumers, but none of them are built, specified, or validated in v1.

v1 ships in two **phases** that differ only in where sensor samples come from. **Phase 1**
uses an OAK-specific adapter talking DepthAI directly, because the Ego Data Capture
interface (IsaacTeleop#571) is still in progress. **Phase 2** swaps that for a generic
adapter over `ICamera`/`IImu` once that design lands. Both phases are *within* v1. The swap
touches exactly two things — which class produces `ISensorSink` callbacks, and which object
`DeviceCalibrationSource` reads its self-report from. Every type and rule above that seam —
`IPoseProvider`, the canonical `RigCalibration` and its merge, the output path — is
unchanged. Full detail in "Phased implementation — OAK-first path".

**In v1:**

- **One rigid rig per instance, at most 2 cameras** — mono or a stereo pair (+ optional IMU)
  produce **one** pose stream, **one** `OutputSpec`, **one** `SchemaPusher`, in one process.
  The 2-camera ceiling is deliberate: the target device is an OAK-D stereo pair, and the
  atomic-capture primitive (`on_stereo_pair()`) is defined for exactly two frames. Rigs of
  3+ cameras need the deferred `ImageSet` grouping type and are rejected at startup.
- **Backends:** cuVSLAM (release target), `stub` (build default, no deps), ORB-SLAM3
  (dev-only validation; opt-in, developer-supplied, never shipped or in release CI).
- **One raw device-clock domain per instance**, validated at startup.
- **OAK-first sensor path** — `OakSensorAdapter` in Phase 1, generic `SensorAdapter` in
  Phase 2.
- **Egocentric data collection is the only supported consumption mode** — pose recorded and
  aligned offline, latency tolerated. There is no profile enum; live robot control is
  deferred (see below).
- Owned CPU image buffers, atomic stereo pairs, whole-record calibration merge, and a fixed
  `T_session_slamworld` read from yaml at startup.

**Deferred (not in v1; listed so the boundary is explicit):**

- **Real-time control of a robot from a SLAM pose.** v1 has no forward-prediction API
  (`predict_to()`) and no profile enum; a SLAM pose is tens of ms old when pushed, which is
  fine for recording but not for live control. Prediction is deliberately deferred so it
  lands together with a pinned horizon, a device-domain "now" derivation, and measured
  end-to-end latency — see "Real-time teleop (future work)".
- Collaborative / shared-map multi-rig — future work, and its eventual interface shape is
  intentionally undecided.
- Cross-clock-domain fusion, software clock mapping, and drift estimation.
- In-process `reset()`; a map reset is a stop/restart with updated yaml.
- IMU recording to MCAP; zero-copy and GPU buffer sharing.
- Rigs of **3+ cameras**, and the general `ImageSet` capture-group type they need for atomic
  delivery. v1's `on_stereo_pair()` covers exactly two frames; startup rejects more.
- VINS backend; image transport across machines.

Decisions locked in:

- Sensor input: **device interface layer** (`ICamera` defined in the Ego Data Capture
  Interface design; `IImu` not yet specified there — v1 IMU access is owned by
  `OakSensorAdapter` directly); bridged to the Pose Provider via a `SensorAdapter` in the
  plugin. A device that estimates pose natively (e.g. ZED SDK's
  built-in tracking) may instead be wrapped directly as an `IPoseProvider`, bypassing the
  sensor layer entirely.
- Backend selection: **compile-time** (CMake option); heterogeneous backends across
  instances come from deploying **different per-backend binaries** (see Topologies).
- Output: **reuse `se3_tracker`** (`core::Se3TrackerPose`), no new device type.
- Topology: **one rigid rig per instance** (M sensors → 1 pose). Collaborative/shared-map
  multi-rig is future work and not part of the v1 interface or types.
- Consumption: **egocentric data collection** (pose recorded, aligned offline). Driving a
  robot live from a SLAM pose is materially stricter and is **future work** — it needs IMU
  forward-prediction, which v1 does not ship. See "Real-time teleop (future work)".

## Architecture

```
Device layer                   IPoseProvider impl             producer plugin
(ICamera + IImu, out of scope) (cuVSLAM, ORB-SLAM3, ...)      (SE3 push path)
        │                            │                            │
  images + IMU  ──on_image/on_imu──▶ SLAM/VIO estimator           │
  [via SensorAdapter]                │                            │
                                     └── get_latest_pose() ──▶ Se3TrackerPoseT
                                                                  │  SchemaPusher
                                                                  ▼  .push_buffer()
                                                     OpenXR tensor collection
                                                     (collection_id, "se3_tracker")
                                                                  ▼
                                             core::Se3Tracker (consumer, ALREADY built)
                                                → Python IDeviceIOSource → teleop graph
```

Three seams, each independently swappable: **device layer** (`ICamera` WIP in
IsaacTeleop#571; IMU accessed directly by `OakSensorAdapter` in Phase 1 — no `IImu` yet),
**estimator** (`IPoseProvider`), **transport** (existing `SchemaPusher` +
`se3_tracker`, unchanged).

**Design alternative considered: device-locked providers.** The simpler approach is one
plugin per (device, backend) pair — `oak_orb_slam3`, `oak_cuvslam`, etc. — mirroring how
`controller_se3_tracker` owns its device directly. This is rejected because it scales
multiplicatively: N devices × M backends = N×M plugins, each a partial copy of the others.
The layered design scales additively (N+M), so adding one backend or one device is always
one self-contained unit of work.

## Module placement

A new core library `src/core/pose_provider/` holding the interfaces and data types
(public headers under `inc/pose_provider/`, per the `cmake-structure` conventions).
Backend impls live in sibling dirs selected by CMake: `backends/cuvslam/` (release),
`backends/orbslam3/` (dev-only), `backends/stub/` (default). A future
`slam_pose_provider` plugin wires the device layer → `SensorAdapter` → `IPoseProvider`
→ `SchemaPusher`, modeled on `src/plugins/controller_se3_tracker/`. The device layer
(`ICamera`) is defined in the Ego Data Capture Interface design (`IImu` not yet specified
there); both are out of scope here.

## Interface 1 — Device interface requirements and `ISensorSink`

> **Out of scope:** The design of the device interface layer (`ICamera`; `IImu` is not yet
> specified in IsaacTeleop#571) is defined in the **Ego Data Capture Interface design**.
> `SensorAdapter` is internal to the `slam_pose_provider` plugin and is not defined there.
> This document only specifies what the Pose Provider needs from the device layer, and
> defines `ISensorSink` as the boundary the provider exposes.

### What the Pose Provider requires from the device interface layer

The `slam_pose_provider` plugin bridges the device interface layer to `ISensorSink` via a
`SensorAdapter`. The adapter adapts whatever delivery model the capture interface provides
to meet these outcome guarantees before forwarding into `ISensorSink`:

1. **Non-blocking SLAM consumption** — camera and IMU samples can be consumed without
   blocking or stalling the capture path. The adapter achieves this regardless of whether
   the capture interface uses polling or subscription; `on_image`/`on_imu` are enqueue-only
   and the provider runs SLAM on its own thread.
2. **Two timestamps per sample** — `sample_time_device_ns` (hardware clock, the clock VIO
   fuses on) and `sample_time_local_ns` (host monotonic, for teleop stream alignment).
   The adapter populates both from whatever the capture interface exposes; neither field is
   rewritten downstream (see Timing C).
3. **Raw image data in a SLAM-accepted format** — the adapter learns the applied pixel
   format from the capture interface's stream descriptor at open time and converts only when
   the applied format is not in the backend's accepted list.
4. **IMU at the required rate** — ≥200 Hz on the same device-clock domain as the camera
   (intra-device IMU is strongly preferred; see Timing C and the v1 clock-domain constraint).
5. **Calibration available after open** — per-camera intrinsics, inter-camera extrinsics,
   and IMU parameters are obtainable after the stream is opened. These become the device
   self-report source in the calibration merge (see Calibration).
6. **Fan-out** *(Phase 2 only)* — recording and SLAM can consume the same stream
   independently. Phase 1 does not depend on this: `OakSensorAdapter` owns its own
   DepthAI pipeline and serves both paths from separate queues (see Phased implementation).

### `SensorAdapter` contract (plugin layer, not a public interface)

The `SensorAdapter` lives in the `slam_pose_provider` plugin. It is not a reusable
interface — it is the glue that bridges the device layer to `ISensorSink`. Its obligations:

- Consumes camera and IMU samples from their available sources using whatever delivery model
  those sources provide (polling or callbacks). Fan-out to the recording path is handled at
  the source level (Phase 2) or via separate queues in the adapter (Phase 1; see Phased
  implementation).
- Populates both timestamps on every `ImageFrame`/`ImuSample`. When the instance has a
  `SensorMultiplexer`, the adapter forwards **in the order the device produced them** and
  does not buffer or reorder. When it is the *only* adapter (Phase 1) there is no mux, so it
  additionally interleaves its own camera and IMU streams by `sample_time_device_ns` before
  dispatch — see Timing A. Cross-*adapter* ordering is always the `SensorMultiplexer`'s job
  (see Timing & synchronization C).
- Performs pixel format conversion only when the device could not natively produce a format
  from `capabilities().pixel_formats`.
- Never blocks on SLAM compute; all downstream calls are enqueue-only.

### `ISensorSink` — the Pose Provider's input boundary

```cpp
namespace core {

// Identifiers: a SensorId names one physical stream (camera or IMU); a RigId names a
// rigid body of 1+ sensors with known extrinsics — one pose per rig. Stable strings so
// config files can reference them.
using SensorId = std::string;
using RigId = std::string;

enum class PixelFormat { Gray8, NV12, BGR888, RGBA8 };

struct ImageFrame {
    SensorId sensor_id;               // stable routing key -> (rig, role) via instance config
    uint32_t width, height, stride;
    PixelFormat format;
    std::vector<uint8_t> data;        // owned; SensorAdapter copies device buffer before delivery
    int64_t sample_time_local_ns;    // host monotonic clock (os_monotonic_now_ns)
    int64_t sample_time_device_ns;   // hardware clock; VIO fuses on this
};

struct ImuSample {
    SensorId sensor_id;
    double accel[3];                 // m/s^2
    double gyro[3];                  // rad/s
    int64_t sample_time_local_ns;
    int64_t sample_time_device_ns;
};

class ISensorSink {
public:
    virtual ~ISensorSink() = default;
    virtual void on_image(ImageFrame) = 0;      // takes ownership; enqueue only, must be cheap
    virtual void on_imu(const ImuSample&) = 0;  // must be cheap; enqueue only

    // Atomic stereo delivery: both frames MUST be enqueued as one indivisible work item, so
    // a concurrent consumer can never observe a half-pair. Pure virtual by design — there is
    // no splitting default, because a sink that forgot to override one would break the
    // invariant silently. Mono-only sinks implement a rejecting stub:
    //     void on_stereo_pair(ImageFrame, ImageFrame) override {
    //         throw std::logic_error("stereo source wired to a mono-only sink");
    //     }
    // Reaching that stub is a wiring bug startup validation should have caught.
    virtual void on_stereo_pair(ImageFrame left, ImageFrame right) = 0;
};

} // namespace core
```

`IPoseProvider` inherits `ISensorSink` (see Interface 2). The `SensorAdapter` in the plugin
calls these after pulling from its sensor source — DepthAI directly in Phase 1,
`ICamera`/`IImu` in Phase 2 — and resolving ordering and format.

## Interface 2 — `IPoseProvider` (swappable estimator)

The provider is an `ISensorSink` (consumes tagged image+IMU from **one or more** sensors in
one rigid rig) and estimates a single pose. v1 scope: one rig per instance. The plugin polls
for a fresh pose each tick.

```cpp
namespace core {

enum class TrackingState { Initializing, Tracking, Lost, Relocalizing };

struct PoseEstimate {
    Pose pose;                   // position (m) + Hamilton quat, in the instance's map frame
    TrackingState state;
    int64_t sample_time_local_ns;
    int64_t sample_time_device_ns;
    // Optional SLAM-only extras (dropped when emitting se3_tracker; see Output):
    std::optional<std::array<double, 36>> covariance;  // 6x6 row-major
};

// Returned by IPoseProvider::capabilities() before initialize(). Drives startup validation
// and format negotiation without opening hardware.
struct ProviderCapabilities {
    std::vector<PixelFormat> pixel_formats;  // accepted pixel formats, most-preferred first
    bool supports_multi_sensor_rig;          // rigid multi-camera rig (>1 camera per rig)
    bool requires_imu;                       // false = visual-only backend
};

class IPoseProvider : public ISensorSink {
public:
    virtual ~IPoseProvider() = default;

    // --- Capability introspection (valid before initialize()) -----------------------
    virtual ProviderCapabilities capabilities() const = 0;
    virtual const char* backend_name() const = 0;
    // Backend-specific calibration validation. Called after generic structural validation
    // (Orchestration step 5). Returns empty string on success, error message on failure.
    // Each backend checks its own constraints: distortion models, camera counts, rates, etc.
    virtual std::string validate_calibration(const RigCalibration&) const = 0;

    // --- Lifecycle: Uninitialized → Initialized → Running → Stopped ----------------
    // initialize() is called after calibration is resolved and validated. start() begins
    // accepting sensor input. stop() blocks until the provider worker thread is quiescent;
    // the plugin must stop all SensorAdapters before calling stop() so no on_image/on_imu
    // calls can arrive after stop() returns.
    virtual void initialize(const SlamInstanceConfig& config,
                            const RigCalibration& calibration) = 0;
    virtual void start() = 0;
    virtual void stop() = 0;
    // v1: no in-process reset. Map reset = stop() all adapters + provider, update yaml if
    // T_session_slamworld changed, restart. Internal events (tracking loss, relocalization,
    // LC, BA) are reported via TrackingState; the plugin does not call any reset API for them.

    // --- Pose output ----------------------------------------------------------------
    // Inherited from ISensorSink: on_image(...), on_imu(...), on_stereo_pair(...).
    // Returns true and fills `out` if a fresh estimate is available since the last call.
    // Provider contract: must not remain in TrackingState::Tracking through an IMU gap that
    // exceeds its internal VIO tolerance. Transition to Lost is the correct response.
    virtual bool get_latest_pose(PoseEstimate& out) = 0;
    // v1 has no forward-prediction API. IMU forward-prediction (predict_to()) is required
    // before a SLAM pose can drive a robot live; see "Real-time teleop (future work)".
};

} // namespace core
```

## Compile-time backend selection

A CMake option chooses exactly one backend to build/link, so only its heavy deps
(Pangolin/Eigen/Ceres/…) are pulled in. A single factory, guarded by the same macro,
returns the concrete provider — the plugin only ever sees `IPoseProvider`.

`stub` is the build default so a checkout with no SLAM deps installed still configures and
compiles; **cuVSLAM is the intended production backend** (see below) and is what deployment
builds select explicitly.

```cmake
set(POSE_PROVIDER_BACKEND stub CACHE STRING "cuvslam | orbslam3 | stub")
# orbslam3: dev-only validation backend; opt-in, not vendored, not in release CI.
# vins: reserved future slot; licensing TBD.
```

```cpp
// pose_provider/factory.cpp
std::unique_ptr<IPoseProvider> core::create_pose_provider() {
#if defined(POSE_PROVIDER_CUVSLAM)
    return std::make_unique<CuVslamPoseProvider>();    // release backend: GPU, vendored prebuilt
#elif defined(POSE_PROVIDER_ORBSLAM3)
    return std::make_unique<OrbSlam3PoseProvider>();   // dev-only: local dep, GPLv3, not shipped
#else
    return std::make_unique<StubPoseProvider>();       // default: no deps, for testing
#endif
}
```

This mirrors the existing runtime dispatch pattern (`create_frame_sink` in
`src/plugins/oak/core/frame_sink.cpp:194`, and the tracker dispatch table in
`src/core/live_trackers/cpp/live_deviceio_factory.cpp:130`) but resolved at compile time.
Each backend is a separate static lib linked only when selected.

**Heterogeneous backends across instances (without runtime selection).** Because each SLAM
instance is its own process/binary (see Topologies), the compile-time choice is *per
binary*, not per deployment. Build one binary per backend
(`slam_pose_provider_cuvslam`, `slam_pose_provider_orbslam3`, …) and assign instances to
binaries in the topology config. You get cuVSLAM on one rig and ORB-SLAM3 on another
simultaneously — no runtime switch, no linking every SLAM lib into one process. The only
thing forbidden is two backends fused inside a single estimator instance, which is
meaningless anyway.

### cuVSLAM (in-house) — the natural default backend

cuVSLAM fits with **no interface changes**, and its own API model mirrors this design:
- **Rig maps 1:1.** cuVSLAM's C API (`cuvslam.h`) is built around a `CUVSLAM_Rig` (a set of
  `CUVSLAM_Camera`s with intrinsics + extrinsics + optional IMU). `CuVslamPoseProvider`
  mostly translates `RigCalibration` (cameras + IMU) into that registration.
- **Capabilities:** `capabilities()` returns `{supports_multi_sensor_rig=true,
  requires_imu=true, pixel_formats={Gray8}}`.
- **Timing:** cuVSLAM requires image+IMU on one synchronized clock — exactly the Timing &
  synchronization contract; reinforces it rather than straining it.
- **Output:** returns pose + covariance + tracking state → `PoseEstimate`; adapter converts
  cuVSLAM's axis convention into the `se3_tracker` `Pose` convention.
- **Loop-closure caveat:** full-SLAM mode does pose-graph optimization → pose *jumps* on map
  correction. Harmless for recording (better global consistency), and
  `SlamInstanceConfig::odometry_only` disables it if a run needs continuity instead. Live
  robot control would have to force it on — see "Real-time teleop (future work)".
- **Build/deploy:** ships as a **prebuilt binary** (`.so` + header) + CUDA runtime + GPU —
  so its CMake backend links a *vendored prebuilt lib* (like `deps/cloudxr`), not
  FetchContent-from-source. The repo already uses `find_package(CUDAToolkit)` (viz,
  `examples/camera_viz/codec`), so the CUDA/GPU linkage story exists.

## SLAM topologies & multi-instance orchestration

v1 supports two topologies, both expressed as `SlamInstanceConfig` with no interface change:

| Topology | Cameras in | Pose out | Instances | Notes |
|---|---|---|---|---|
| Mono (+ optional IMU) | 1 | 1 | N | Frames delivered via `on_image()` |
| Stereo rig (+ optional IMU) | 2 | 1 | N | Atomic pairs via `on_stereo_pair()`; the OAK-D target |

Each instance is its own process, so N independent rigs = N processes, fault-isolated.
Rigs of 3+ cameras, collaborative/shared-map multi-rig, and hybrid topologies are future work.

### The load-bearing constraint: no image transport → topology = process decomposition

Raw frames never cross the tensor bus (only tiny metadata/poses do). So **all sensors
feeding one estimator instance must live in the same OS process as that estimator.** This
fixes the process model:

- **Per-device** → N separate `slam_pose_provider` processes (one per device), which the
  existing `PluginManager` forks independently → fault isolation (one SLAM crash doesn't
  kill the others). This is the natural fit for the current plugin-process architecture.
- **Centralized, rigid multi-cam rig** → one process that opens all its sensors directly
  (DepthAI supports multiple devices per process). Collaborative SLAM is future work.

### Config (declarative topology; lives in the orchestration/plugin layer, not the estimator)

One `SlamInstanceConfig` per process. yaml-cpp is already a dependency (`plugin_manager`
parses `plugin.yaml` with it), so a sibling `slam_topology.yaml` per instance is natural.

```cpp
struct SensorSpec {              // how to open one physical stream
    SensorId sensor_id;
    std::string device;          // "oak" | "zed" | "sensing" — selects the source impl
    std::string device_serial;   // DepthAI mxid; empty = any
    // Vendor-neutral role name, kept a string so slam_topology.yaml stays readable and no
    // vendor enum leaks into the config schema. Each source impl owns the mapping to its
    // own enum (e.g. OakSensorAdapter: "left" -> core::StreamType_MonoLeft); an unknown
    // name for that device is a startup error.
    // RESERVED: "imu" always denotes the inertial stream. Every other value names a camera.
    // This is the only role the generic layer interprets, and it is what lets Orchestration
    // step 3 count cameras before any device impl is constructed.
    std::string stream;          // "imu", or a camera role: "left" | "right" | "color" | ...
};
struct CalibrationSourceSpec {
    enum class Kind { Device, File } kind;
    std::string path;   // when kind == File; empty for Device
};
struct RigSpec {
    RigId rig_id;
    // The rig OWNS its sensors rather than referencing shared IDs. Sensors are a property of
    // the rigid body they are bolted to, so there is no case for sharing one across rigs —
    // a camera cannot be rigidly attached to two independently moving bodies. Owning them
    // here keeps one source of truth (no dangling or orphaned SensorId references) and
    // extends unchanged to collaborative topology, where each RigSpec carries its own set.
    // v1: 1-2 cameras + at most one "imu" entry.
    std::vector<SensorSpec> sensors;
    // Ordered calibration sources (later replaces earlier), e.g. [device-self-report,
    // kalibr.yaml]. Resolved into a canonical RigCalibration at startup. See "Calibration".
    std::vector<CalibrationSourceSpec> calibration;
};
struct OutputSpec {
    std::string collection_id;                    // se3_tracker wire instance
    enum class Frame { SlamWorld, Session } output_frame;
    // Required when output_frame == Session; startup validation rejects Session with this
    // unset. Rigid transform aligning the SLAM world origin into the OpenXR session base
    // space. Always read from slam_topology.yaml at startup — the plugin never computes it.
    //
    // VALIDITY CONDITION. The SLAM world origin is the rig body pose at tracking init
    // (gravity-aligned when an IMU is present) — it is re-established on every start, not
    // preserved across runs. A constant written in yaml is therefore only meaningful when
    // the rig reliably starts from the SAME physical pose (fixed mount, docking fixture,
    // known home position). Configure Session output only under that assumption.
    //
    // Otherwise use output_frame: SlamWorld and align offline — which is the normal v1
    // data-collection path, since recorded poses are aligned in post anyway. A live
    // alignment procedure that survives restarts is future work.
    std::optional<Pose> T_session_slamworld;
};
struct SlamInstanceConfig {
    std::string instance_id;
    RigSpec rig;                      // owns its sensors; v1 has exactly one rig
    OutputSpec output;
    std::string backend_config_path; // backend-specific config file path (e.g. cuVSLAM params)

    // v1 serves egocentric data collection only: the pose is recorded and aligned offline,
    // so pipeline latency is tolerated as long as timestamps are correct. Driving a robot
    // live is future work — see "Real-time teleop (future work)".

    // Disables loop closure / pose-graph optimisation, running the backend as pure
    // visual-inertial odometry. Drift accumulates, but the pose never jumps on map
    // correction. Optional in v1; loop closure is usually preferable for recording.
    bool odometry_only = false;

    // Reorder window + drop/skew/IMU-gap diagnostic thresholds. Applied by the
    // SensorMultiplexer when there is one, otherwise by the single SensorAdapter.
    StreamHealthConfig stream_health;

    // If get_latest_pose() has not produced a fresh estimate for this long, the plugin
    // synthesizes an invalid Se3TrackerPose so the consumer cannot hold stale validity.
    // (A fresh non-Tracking estimate already publishes is_valid = false on its own; this
    // timeout covers silence — the provider returning false tick after tick.)
    int64_t pose_timeout_ns = 500'000'000;  // 500 ms default
};
```

### Orchestration (one process = one instance = one backend binary)

The `slam_pose_provider` plugin generalizes `controller_se3_tracker` by potentially owning
**multiple** sensor sources (one per camera/IMU) but a **single** `SchemaPusher`. The backend
is created first so its capabilities drive format negotiation at hardware-open time.
Config-level validation runs before hardware opens
(fast fail on wiring errors); calibration resolution and provider initialization run after
(device self-report is only available post-open).

1. Load `SlamInstanceConfig`.
2. `create_pose_provider()` — this binary's backend. `provider.capabilities()` is valid
   immediately (no hardware open required) and drives steps 3 and 4.
3. **Validate config before opening hardware.** Camera count is `config.rig.sensors.size()`
   minus the
   `"imu"` entries — `"imu"` is the one reserved, vendor-neutral role name in `SensorSpec`,
   so the generic layer can partition camera vs non-camera roles without knowing any vendor's
   stream vocabulary. Then: reject **> 2 cameras** (v1 ceiling — no atomic delivery primitive
   exists for 3+; see `ImageSet` in Deferred); reject **2 cameras** when
   `supports_multi_sensor_rig == false`, so a stereo config can never reach a mono-only
   sink's `on_stereo_pair()` stub; reject `requires_imu == true` with no `"imu"` sensor.
   Plus config self-consistency (`output_frame == Session` implies `T_session_slamworld` is
   set). Fail fast.
4. Open every `SensorSpec` and instantiate the **`SensorAdapter`**(s) — Phase 1 constructs a
   single `OakSensorAdapter` over DepthAI; Phase 2 opens via the device interface layer
   (`ICamera`/`IImu` — designed elsewhere), one adapter per source. Request the first format
   from `capabilities().pixel_formats` the device can produce natively. Opening reports the
   applied stream descriptor (pixel format, dimensions, clock domain); the adapter exposes
   the clock domain to the plugin, which rejects mismatched domains here (see Timing C).
5. Resolve `config.rig`'s `CalibrationSourceSpec` list into a canonical `RigCalibration`:
   merge sources in list order (later entries override earlier); e.g. `[device, kalibr.yaml]`
   uses device self-report as base and Kalibr as override. Run generic structural validation
   (extrinsic graph completeness, IMU noise fields when `requires_imu`, readout time when
   `shutter == Rolling`), then `provider.validate_calibration(calibration)` for backend-
   specific constraints. Fail fast on either.
6. Wire every `SensorAdapter` into a **`SensorMultiplexer`** (single time-ordered stream on
   one fusion timeline; see Timing & synchronization C) whose output is the provider, each
   sample tagged with its `sensor_id`. **Single-adapter instances skip the mux** — one
   adapter owns all its streams and merge-orders them itself (Phase 1 does exactly this; see
   Phased implementation), so the mux exists to merge *across* adapters. Either way the
   provider's input contract is identical. Then `provider.initialize(config, calibration)` →
   `provider.start()`.
7. Per tick: drain into the provider — `mux.update()` when a mux is present, otherwise
   `adapter.update()` directly — then `get_latest_pose()` → apply the `output_frame`
   transform and push `Se3TrackerPose` on `config.output.collection_id` via the single
   `SchemaPusher`. Apply the silence-timeout policy (`pose_timeout_ns`) to prevent stale
   validity.

```cpp
// Merges N per-adapter input streams into one time-ordered ISensorSink output. Instantiated
// only when an instance has MORE THAN ONE adapter; a single adapter merge-orders its own
// streams instead (see Timing A). Each SensorAdapter writes into the slot returned by
// add_source(); the mux k-way merges by sample_time_device_ns through a bounded reorder
// buffer and forwards to the output.
//
// These are the stream-health thresholds. They are a member of SlamInstanceConfig (loaded
// from slam_topology.yaml) rather than of the mux, because a single-adapter instance has no
// mux yet still needs the same reorder window and the same drop/skew/gap diagnostics —
// OakSensorAdapter applies them directly in Phase 1.
struct StreamHealthConfig {
    int64_t reorder_window_ns;   // buffer depth: max out-of-order skew absorbed
    int64_t skew_alert_ns;       // inter-sensor skew above this logs a warning
    // Gap between consecutive IMU samples above this threshold logs a warning. This is
    // diagnostic only; the mandatory behaviour is the IPoseProvider contract — a provider
    // must not remain in Tracking through a gap exceeding its internal VIO tolerance.
    int64_t imu_gap_alert_ns;    // default: 5 × (1s / imu_rate_hz)
};

class SensorMultiplexer {
public:
    explicit SensorMultiplexer(const StreamHealthConfig&);
    // Call once per SensorAdapter; returns its input slot. The returned sink implements
    // on_stereo_pair() by buffering the pair as ONE reorder-buffer entry keyed on the shared
    // device timestamp — the mux never splits a pair, and backpressure drops the entry whole.
    ISensorSink* add_source();
    void set_output(ISensorSink* sink);   // the IPoseProvider
    // Releases every buffered entry older than (newest_seen - reorder_window_ns) to the
    // output, in device-time order; a stereo entry is released as one on_stereo_pair() call.
    // Called each plugin tick so a quiet source cannot stall the stream indefinitely.
    void update();
};
```

Deployment example (v1: two independent single-rig cuVSLAM processes):
```
slam_pose_provider_cuvslam --instance head_a.yaml   # 1 rig -> se3_tracker "slam_head_a"
slam_pose_provider_cuvslam --instance head_b.yaml   # 1 rig -> se3_tracker "slam_head_b"
# Multi-backend and collaborative topologies: deferred (see Topology table and Open questions)
```

### Frame semantics

Pose is in the instance's own SLAM world frame, whose origin is the rig body pose at
tracking init (gravity-aligned when an IMU is present) and is re-established on every start.
`output_frame: SlamWorld` passes that frame through and is the v1 default — recorded poses
are aligned offline anyway. `output_frame: Session` applies the fixed `T_session_slamworld`
from yaml, which is only valid when the rig reliably starts from the same physical pose
(see `OutputSpec`).

## Calibration (heterogeneous inputs → one canonical model)

Different setups need different calibration (mono: intrinsics only; stereo/multi-cam: +cam↔cam
extrinsics; VIO: +cam↔IMU spatial + time-offset + IMU noise; rolling shutter: +readout time),
from different sources (device self-report vs Kalibr file vs mixed), consumed by backends that
each want a different format. Handle all three with a **canonical model + layered sources +
backend adapters**, so none of that variety touches the interfaces.

**Canonical `RigCalibration` — backend/vendor-neutral, optional-field superset.** One
convention is pinned here (frame direction, quaternion order, units); loaders and backend
adapters are the *only* places conversions happen.

```cpp
struct CameraCalibration {
    SensorId sensor_id;
    uint32_t width, height;
    enum class Model { Pinhole, RadTan, KannalaBrandt /*fisheye*/, DoubleSphere } model;
    std::vector<double> intrinsics;              // fx,fy,cx,cy (+ model params); required
    std::vector<double> distortion;
    enum class Shutter { Global, Rolling } shutter;
    std::optional<double> rolling_readout_ns;    // present only when shutter == Rolling
    std::array<double,16> T_body_cam;            // body(rig)-from-camera; required
};
struct ImuCalibration {
    SensorId sensor_id;
    // Noise params: typically absent from device self-report; come from file source (Kalibr).
    // Startup rejects if requires_imu and any are absent after merge.
    std::optional<double> accel_noise_density, gyro_noise_density;
    std::optional<double> accel_random_walk, gyro_random_walk;
    double rate_hz;
    std::array<double,16> T_body_imu;            // body-from-IMU; required
    std::optional<double> time_offset_ns;    // cam↔IMU offset (Kalibr); nullopt = unknown
};
struct RigCalibration {                  // one rig; optionality encodes setup variety
    RigId rig_id;
    std::vector<CameraCalibration> cameras;   // length = mono/stereo/multi
    std::optional<ImuCalibration> imu;        // absent = visual-only
};
```

**Layered sources (`RigSpec.calibration`).** Sources are merged in list order: a later source
replaces the entire `CameraCalibration` or `ImuCalibration` record for any `sensor_id` it
covers (whole-record replacement, not field-level overlay). The orchestrator produces a fully
populated `RigCalibration` after merging all sources; missing required fields (not covered by
any source) are a startup error.

```cpp
class ICalibrationSource {
public:
    virtual ~ICalibrationSource() = default;
    // Returns complete CameraCalibration/ImuCalibration records for the sensors this source
    // covers. A later source in the ordered list replaces earlier records for the same
    // sensor_id wholesale. Required sensor records absent from all sources, and required
    // optional fields (e.g. IMU noise params) that remain nullopt after merge, trigger a
    // startup error.
    virtual RigCalibration load(const RigSpec&) const = 0;
};

// CalibrationSourceSpec::Kind::Device — the opened hardware's self-report. The SOURCE of
// that self-report is phase-dependent, and this is the one seam the phase swap moves:
//   Phase 1: OakSensorAdapter::get_calibration(), from dai::CalibrationHandler.
//   Phase 2: the device interface at open time (ICamera.open() / IImu.open()).
// Both return the same canonical RigCalibration, so the merge rule, validation, and every
// consumer above are identical; only which object is asked changes.
class DeviceCalibrationSource : public ICalibrationSource { /* impl is phase-dependent */ };

// Parses a Kalibr yaml or native calibration file.
class FileCalibrationSource : public ICalibrationSource { /* path from CalibrationSourceSpec */ };
```

- `DeviceCalibrationSource` — provides complete camera intrinsics, extrinsics, and IMU
  spatial/rate from the device self-report (Phase 1: `OakSensorAdapter::get_calibration()`;
  Phase 2: the device interface). IMU noise params are typically absent (the device doesn't
  measure its own noise floor); `ImuCalibration` is returned without them, leaving noise
  fields as `nullopt`.
- `FileCalibrationSource` — Kalibr yaml provides a **complete** `ImuCalibration` record
  (including `T_body_imu`, `rate_hz`, and noise params), which wholly replaces the device
  self-report record for that `sensor_id`.
- Canonical v1 example `[device, kalibr.yaml]`: device provides base camera and IMU records;
  Kalibr replaces the entire `ImuCalibration` (not just noise fields).

**Backend adapters translate canonical → native** (cuVSLAM rig struct / ORB-SLAM3 yaml / VINS
yaml), owning distortion-model and axis-convention mapping. The interface only ever carries
`RigCalibration`.

**Validation at startup** (Orchestration step 5): two passes — generic structural check
(extrinsic star graph complete, IMU noise fields present when `requires_imu`,
`rolling_readout_ns` set when `shutter == Rolling`) followed by
`provider.validate_calibration(calibration)` for backend-specific constraints. Config-level
checks (`output_frame == Session` implies `T_session_slamworld` set) run earlier in step 3.
Online refinement (VIO backends refine cam↔IMU/time-offset live) is a backend concern; the
canonical model supplies the *prior*.

## Output — reuse `se3_tracker` (recommended)

The plugin's per-tick loop maps the single `PoseEstimate` onto the existing SE3 push path
(`controller_se3_tracker_plugin.cpp:67-105` is the template):

```cpp
// SlamWorld -> pass through; Session -> T_session_slamworld * pose.
core::Pose apply_output_frame(const core::OutputSpec& spec, const core::Pose& slam_world_pose) {
    if (spec.output_frame == core::OutputSpec::Frame::SlamWorld)
        return slam_world_pose;
    return compose(*spec.T_session_slamworld, slam_world_pose);  // non-null validated at startup
}

// Per-tick loop.
core::PoseEstimate estimate;
if (provider.get_latest_pose(estimate)) {
    core::Se3TrackerPoseT out;
    if (estimate.state == core::TrackingState::Tracking) {
        out.pose = std::make_shared<core::Pose>(
            apply_output_frame(config.output, estimate.pose));
        out.is_valid = true;
    } else {
        out.pose = std::make_shared<core::Pose>(/* identity */);
        out.is_valid = false;               // tracking-lost filler; consumers freeze
    }
    // Pack -> builder.Finish() -> pusher.push_buffer(ptr, size,
    //     estimate.sample_time_local_ns, estimate.sample_time_device_ns)
}
// If get_latest_pose() returns false, apply pose_timeout_ns policy.
```

Why this is the whole point of the design:
- **Zero consumer-side work.** `core::Se3Tracker` is already registered in the live and
  replay factory dispatch tables (`live_deviceio_factory.cpp:141`); record/replay via MCAP
  and the Python `IDeviceIOSource` bridge come for free.
- The SLAM state machine collapses onto the schema's `is_valid` two-level validity
  (`se3_tracker.fbs`): `is_valid = (state == Tracking)`.
- Wire rendezvous: `tensor_identifier = Se3Tracker::TENSOR_IDENTIFIER ("se3_tracker")`,
  `max_flatbuffer_size = Se3Tracker::DEFAULT_MAX_FLATBUFFER_SIZE (256)`;
  `collection_id` comes from `config.output.collection_id` (e.g. `"slam_head_a"`).

**Key integration concern — reference frame.** `se3_tracker.fbs` requires the producer to
document its reference frame. A SLAM pose starts in a map frame whose origin is the rig body
pose at tracking init, *not* the OpenXR session base space that head/controllers use. v1
documents the device as streaming in its own SLAM world frame (`output_frame: SlamWorld`);
`output_frame: Session` applies a fixed `T_session_slamworld` and is valid only when the rig
starts from a repeatable physical pose (see `OutputSpec`).

**What we give up:** covariance / detailed tracking state / map quality don't fit
`Se3TrackerPose` (only pose + `is_valid`). If those become required, add a dedicated
`slam_pose` device type later (new schema + tracker triple + two factory rows). Not needed
for the first pass — the `is_valid` bool already carries the essential tracking-lost signal.

## Timing & synchronization

Two distinct delays exist between the seams; they need different handling.

**A. Inter-interface delay (adapter → mux → provider).** In-process callbacks, so the
delay is small — but the hazard is **backpressure and ordering**, not latency:
- Provider `on_image`/`on_imu` are enqueue-only; SLAM runs on the provider's own worker
  thread. A blocking call into the provider would stall the whole chain and drop
  frames / back up IMU.
- The provider is guaranteed samples in **monotonic device-time order** (IMU interleaved
  with frames), and never sorts. VIO integrates IMU *between* consecutive frames, so
  out-of-order or dropped IMU corrupts the estimate. Who produces that guarantee depends on
  the wiring, and exactly one component owns it in each case:
  - **Multi-adapter instances:** the `SensorMultiplexer`. Adapters forward as-arrived; the
    mux k-way merges and is the single owner of reordering. A sample later than
    `reorder_window_ns` is dropped and counted rather than delivered out of order.
  - **Single-adapter instances (Phase 1):** the adapter itself. `OakSensorAdapter` holds all
    its own queues, so it interleaves its camera and IMU streams by `sample_time_device_ns`
    before dispatch and applies the same late-drop rule. No mux is instantiated.

**B. Provider processing latency (the one that reaches teleop).** SLAM/VIO compute is tens
of ms, so an emitted pose is *for a past frame*, not "now." Rules:
- `PoseEstimate.sample_time_*` = the timestamp of the **source frame the pose was computed
  for**, never wall-clock at emit. The plugin passes these straight into
  `push_buffer(..., t_local, t_device)` so downstream knows the pose is *for* time T.
- Teleop consequence: a correctly-stamped but late SLAM pose, fused with near-realtime
  head/controller channels, can lag. Propagating accurate timestamps is the floor;
  IMU forward-prediction to "now" inside the provider is the mitigation if latency matters.

**Are the inputs time-synced?** They must be, on **one clock domain**. Image and IMU are
fused on `sample_time_device_ns` (hardware clock), not host-arrival time (USB/scheduling
jitter would desync them). `sample_time_local_ns` is carried in parallel purely so the
resulting SE3 pose aligns with the other teleop device streams on the common clock. How the
device layer provides these timestamps is a device interface concern (out of scope here);
the `SensorAdapter` is responsible for populating both fields correctly on `ImageFrame` and
`ImuSample` before calling into `ISensorSink`.

**C. Multi-sensor / inter-device time sync.** Four layers, easiest to hardest:
1. *Intra-device (cam ↔ its own IMU):* same on-device oscillator → trivially synced. **The
   preferred config is one physical camera providing stereo + IMU** (per-device topology,
   zero sync work).
2. *Device → host clock:* device oscillators drift (ppm) vs host; DepthAI/ZED estimate the
   offset and emit both stamps. `sample_time_local_ns` is what aligns the SLAM pose with the
   rest of teleop. Neither timestamp field is rewritten by the adapter or multiplexer.
3. *Inter-device (multi-cam rig, or cam + separate IMU) — the hard case.* Independent clocks
   are **not directly comparable**. Handle, best first: (a) **hardware sync / genlock**
   (shared trigger; GMSL native, some OAK via FSIN) → one capture clock, comparable device
   stamps — the right answer for rigid multi-cam rigs; (b) **PTP** for networked sensors;
   (c) software-offset estimation — deferred, not supported in v1 (see below).
4. *Exposure convention:* per-frame stamp is **mid-exposure** (not arrival/shutter-start),
   consistent across cameras, latency-corrected; rolling-shutter caveat. Never fuse on
   host-arrival time.

**v1 clock-domain constraint:** all cameras and IMUs feeding one provider instance must share
one raw device-clock domain. Orchestration step 4 validates this at startup and rejects
mismatched or unknown domains. Cross-domain fusion, drift estimation, and software clock
mapping are deferred.

**Who owns alignment: the orchestrator, not the estimator.** The device layer exposes each
source's clock domain; the `SensorAdapter` signals it to the plugin. A **`SensorMultiplexer`**
merges N source streams into one coherent, monotonic stream (k-way merge by
`sample_time_device_ns` + a bounded reorder buffer) and **monitors inter-sensor skew,
flagging threshold breaches** (VIO degrades silently otherwise). The provider stays
sync-agnostic. Both timestamp fields pass through unchanged.

Recommendation ranking: (1) single camera with built-in IMU (v1 OAK path); (2) hardware-
synced multi-cam rig (GMSL/FSIN), IMU on the same sync domain; (3) software-offset fallback
— not supported in v1, not suitable for tight VIO.

## Real-time teleop (future work)

v1 serves **egocentric data collection**: the pose is recorded and aligned offline, so a
50 ms pipeline latency is irrelevant as long as timestamps are correct. This section records
what a *second* use case — **replacing a live tracked device** (e.g. swapping a Pico
controller for an OAK-D + SLAM rig) so the pose drives a robot in real time — would additionally
require. None of it is built or validated in v1; it is captured here so the gap is explicit
and so the v1 interfaces are not accidentally shaped against it.

The wire path would be identical: a SLAM rig is just another `se3_tracker` device, so no
consumer work is needed either way. The difference is entirely in what the producer must
guarantee.

| Concern | v1 (data collection) | Real-time control (future) |
|---|---|---|
| Pipeline latency | Tolerated; correct timestamps suffice | Must be hidden by prediction |
| Loop closure / pose jumps | Fine (better global consistency) | Forbidden — jumps move the robot |
| Tracking loss | Gap in the recording | Robot stops mid-motion |
| Backend prediction | Not required | Required |
| Drift over a session | Correctable offline | Accumulates uncorrected |

### The four things real-time control would add

**1. Latency must be hidden, not just measured — this is the gap that removes it from v1.**
A SLAM pose is *for a past frame* — with OAK-D and a CPU-bound backend, roughly 40–70 ms old
by the time it is pushed (USB transfer, tracking compute, tick and transport). Worse than the
absolute number is the *asymmetry*: OpenXR gives head and controller poses already
extrapolated to display time, so a SLAM channel fused with XR channels reads as lagging by
the full gap plus the XR runtime's forward prediction. Propagating timestamps (Timing B)
makes this correctly *describable*, but not usable for live control.

Closing it needs an IMU forward-prediction call — a `predict_to(target_time_device_ns)` that
integrates IMU past the last processed frame — plus three things v1 does not specify: a
normative derivation of a device-domain "now" from the host clock, a pinned maximum
prediction horizon beyond which the call refuses rather than degrades, and measured
end-to-end latency per backend. **v1 deliberately ships no prediction API**, so that it is
added together with the horizon and quality contract rather than as a bare method whose
accuracy is unspecified.

**2. Loop closure becomes forbidden.** Full-SLAM pose-graph optimisation corrects the map and
the pose *jumps*. During recording that is a feature — better global consistency. Driving a
robot, it is a discontinuity in the commanded end-effector pose. `odometry_only` exists as a
config knob and would need to be forced on, trading drift (bounded, slow) for continuity
(safety-critical). Note this is a sharper constraint for ORB-SLAM3 than for cuVSLAM: loop
closing and multi-map merging are ORB-SLAM3's headline features, so odometry-only mode gives
up most of what distinguishes it.

**3. Tracking loss would need a policy beyond `is_valid = false`.** The `se3_tracker` schema
carries two-level validity, and the consumer contract is "freeze on invalid." For a
recording, a gap is a gap. Mid-teleop, freezing means the robot stops accepting input at an
arbitrary moment — possibly mid-grasp — and `Initializing` means there is *no* usable pose
for the first seconds of every session, unlike a controller which is tracked from the start.
`PoseEstimate` already carries the full `TrackingState`; what is missing is a decision about
what the teleop layer does with `Lost` vs `Relocalizing` vs `Initializing`. That is a
consumer-side policy question (see Open questions), not an interface gap.

**4. Relative drift would be the failure mode, not absolute drift.** What teleop actually
needs is the hand pose *relative to the head*. On a Pico, both come from one tracking system,
so the relative geometry stays consistent even as the whole map drifts. Replacing only the
controller leaves two independent trackers with independent drift — each individually
healthy, their relative pose degrading over a session. A fixed `T_session_slamworld` cannot
correct this, since the error is time-varying. Mitigations (periodic re-alignment against a
shared observable, or tracking both head and hand in one shared-map instance so they share a
drift-correlated frame) would require the deferred collaborative topology.

### Practical verdict

Data collection works with the design as specified. Real-time control would be viable for
slow, deliberate manipulation once prediction and forced odometry-only exist, and is not
recommended for dynamic motion regardless — the residual latency after prediction, plus the
tracking-loss failure modes, are qualitatively worse than a dedicated XR controller. Prefer
a native XR tracked device where one is available; use a SLAM rig where none is (which is
exactly the egocentric, no-headset case v1 targets).

## Files (design-level; no implementation this pass)

- `src/core/pose_provider/cpp/inc/pose_provider/sensor_sink.hpp` — `ISensorSink`,
  `ImageFrame`, `ImuSample`, `SensorId`, `RigId`, `PixelFormat`.
- `src/core/pose_provider/cpp/inc/pose_provider/pose_provider.hpp` — `IPoseProvider`
  (including `get_latest_pose()`, `validate_calibration()`),
  `ProviderCapabilities`, `PoseEstimate`, `TrackingState`, `create_pose_provider()`.
- `src/core/pose_provider/cpp/inc/pose_provider/topology.hpp` — `SensorSpec`, `RigSpec`,
  `OutputSpec`, `SlamInstanceConfig` (+ a yaml-cpp loader for `slam_topology.yaml`), and the
  config-level validation from Orchestration step 3.
- `src/core/pose_provider/cpp/.../sensor_multiplexer.*` — `SensorMultiplexer` +
  `StreamHealthConfig`: merges N `SensorAdapter` outputs into one time-ordered stream (only
  instantiated for multi-adapter instances; a lone adapter merge-orders its own streams).
  Owns reordering across adapters — bounded reorder buffer, late-sample drop counter,
  inter-sensor skew monitor, IMU gap alert. Buffers a stereo pair as one entry and releases
  it via a single `on_stereo_pair()` call — never splits a pair.
- `src/core/pose_provider/cpp/inc/pose_provider/calibration.hpp` — canonical `RigCalibration`
  (`CameraCalibration`/`ImuCalibration`), `ICalibrationSource` (`Device`/`File`),
  `CalibrationSourceSpec`, the precedence-merge resolver (whole-record replacement per
  `sensor_id`, in list order → final `RigCalibration`). Validation is two-pass: generic
  structural check in the orchestrator + `provider.validate_calibration(calibration)` for
  backend-specific rules. Backend adapters translate canonical → native.
- `src/core/pose_provider/backends/cuvslam/` — release backend; links vendored prebuilt
  `.so` + CUDA.
- `src/core/pose_provider/backends/orbslam3/` — **dev-only** validation backend; opt-in
  (`-DPOSE_PROVIDER_BACKEND=orbslam3`), not vendored, not in release CI, GPLv3 dep supplied
  locally by developer.
- `src/core/pose_provider/backends/stub/` — default; no deps, always built. Reports
  `{pixel_formats={Gray8}, supports_multi_sensor_rig=true, requires_imu=false}` and
  implements `on_stereo_pair()` non-throwing, so the end-to-end verification below can run
  the real OAK stereo path against it. Emits a synthetic pose (identity or a scripted
  trajectory) with a `Tracking` state so `is_valid` gating is exercised.
- `src/plugins/slam_pose_provider/` — producer plugin: owns `SensorAdapter` per source
  (bridges device layer → `ISensorSink`), feeds `SensorMultiplexer` → one
  `create_pose_provider()` → one `SchemaPusher` on `se3_tracker`. One process per
  `SlamInstanceConfig`. Device interface (`ICamera`/`IImu`) backends are out of scope here.

Reused as-is: `core::SchemaPusher` (`src/core/pusherio/`), `core::Se3Tracker` +
`Se3TrackerPose` schema, `os_monotonic_now_ns()`, the `controller_se3_tracker` plugin
skeleton, `DeviceIOSession`/`OpenXRSession`.

## Verification (of the design, before/after implementation)

- **Design review acceptance:** a new backend can be added by implementing `IPoseProvider`
  + adding one CMake branch + one factory `#if`, touching neither the device interface
  layer, the plugin, nor any consumer. A new camera vendor = one `SensorAdapter` impl in
  the plugin (device interface design concern). v1 topologies (per-device and rigid-rig)
  express as `SlamInstanceConfig` data with no interface change; collaborative topology is
  deferred and its interface shape is intentionally undecided. Confirm `PoseEstimate` →
  `Se3TrackerPose` mapping is lossless for the fields `se3_tracker` carries.
  Confirm `ISensorSink` exposes no fallback path that can split a stereo pair:
  `on_stereo_pair()` is pure virtual, and every sink on a stereo path either enqueues the
  pair as one work item or explicitly rejects stereo input. Confirm a multi-camera rig on a
  backend with `supports_multi_sensor_rig == false` is rejected in Orchestration step 3 —
  before any hardware opens, and before any sink's rejecting stub could be reached.
- **Once implemented (out of scope now):** build with `POSE_PROVIDER_BACKEND=stub`, run the
  plugin, read the pose back with `examples/schemaio/se3_printer.cpp` on the chosen
  `collection_id` — the same reader used for `controller_se3_tracker` — to confirm the pose
  and `is_valid` gating flow end-to-end without any consumer changes. Confirm that
  artificially starving the IMU queue causes the backend to transition out of `Tracking`
  within its declared tolerance (provider IMU-gap requirement).
- **Stereo pair atomicity (concurrency test):** with the provider worker thread running,
  feed a long stereo sequence through `on_stereo_pair()` and assert the worker never
  dequeues a work item containing one frame of a pair — every dequeue yields either a
  complete pair or no pair. Repeat with the `SensorMultiplexer` in the path and with
  backpressure forcing drops, asserting drops remove whole pairs only.

## Phased implementation — OAK-first path

The `ICamera`/`IImu` device interface design (IsaacTeleop#571) is still in progress.
Rather than block on it, the Pose Provider can be implemented end-to-end with an
OAK-specific adapter that talks DepthAI directly. Once the device interface lands, the
adapter is swapped out; everything above `ISensorSink` is untouched.

### Phase 1 — `OakSensorAdapter` (temporary, OAK-only)

A new class in `src/plugins/oak/core/oak_sensor_adapter.*`, following the same DepthAI
patterns as `OakCamera` but targeting `ISensorSink` instead of `FrameSink`:

```text
OAK (DepthAI direct)
  dai::Pipeline
  ├─ MonoLeft raw  ─┐
  │                 ├─ dai::node::Sync (≤1µs) ─▶ MessageGroup ─┐
  ├─ MonoRight raw ─┘                                          │
  │                                          on_stereo_pair(L,R) ─▶ ISensorSink ─▶ IPoseProvider
  ├─ MonoLeft/Right H.264 queue ─────────────────────────────▶ RawDataWriter (.h264 sidecar)
  ├─ IMU queue (dai::node::IMU, 200 Hz) ─────── on_imu() ─────▶ ISensorSink ─▶ IPoseProvider
  └─ readCalibration() ──────────────────────────────────────▶ RigCalibration

  IPoseProvider ──▶ SchemaPusher ──▶ se3_tracker MCAP channel  (pose recording)
```

`update()` dequeues one `MessageGroup` per tick and calls `on_stereo_pair(left, right)` —
both frames delivered as one atomic call. After dequeue, the adapter verifies exact
device-timestamp equality and discards the group if they differ. A `MessageGroup` that times
out without one side is discarded whole.

Both recording and SLAM draw from the same DepthAI pipeline. Video is hardware-encoded
to an H.264 sidecar (same as the existing `OakCamera`/`RawDataWriter` pattern); raw frames
go to SLAM only. IMU is delivered to SLAM only (IMU recording to MCAP is not in v1 scope).
Pose is recorded via the existing `se3_tracker` path, timestamped with the source frame's
`sample_time_device_ns` — so sensor data, IMU, and pose are all correlated on the device
clock in the final MCAP episode.

Key differences from the existing `OakCamera`:

- **Sync node for stereo SLAM input.** Left and right raw streams pass through a
  `dai::node::Sync` node (threshold ≤ 1 µs) before reaching the host. `update()` dequeues
  a `MessageGroup` and calls `on_stereo_pair(left, right)` — one atomic call, both frames
  as a single work item. A second path through `dai::node::VideoEncoder` feeds
  `RawDataWriter` for the H.264 sidecar independently; both paths share the same source
  timestamps.
- **IMU node added.** Adds `dai::node::IMU` to the pipeline
  (`ACCELEROMETER_RAW` + `GYROSCOPE_RAW` at 200 Hz, `setBatchReportThreshold(1)` for
  per-sample delivery). `update()` polls `m_imu_queue` and calls `m_slam_sink->on_imu()`.
  IMU recording to MCAP is not in scope for v1.
- **Calibration from device.** Calls `m_device->readCalibration()` at construction and
  translates `dai::CalibrationHandler` intrinsics + `getCameraExtrinsics()` into a
  `core::RigCalibration` — the device self-report source in the calibration merge.
- **Timestamps.** Both `getTimestampDevice()` (→ `sample_time_device_ns`) and
  `getTimestamp()` (→ `sample_time_local_ns`) already exist on `dai::ImgFrame` and
  `dai::IMUReport`; the same extraction pattern as `oak_camera.cpp:155-166` applies.

```cpp
// src/plugins/oak/core/oak_sensor_adapter.hpp
class OakSensorAdapter {
public:
    // Takes SensorSpecs directly and owns the "left"/"right"/"imu" -> core::StreamType
    // mapping, so the vendor enum never appears in slam_topology.yaml. Throws on an
    // unrecognized stream name. `format` is the orchestrator's chosen entry from the
    // backend's capabilities().pixel_formats.
    // Builds a mono or stereo pipeline from the camera count in `sensors` (1 or 2; step 3
    // has already rejected 3+). Stereo inserts a dai::node::Sync; mono takes the raw
    // camera output directly.
    OakSensorAdapter(const OakConfig& config,
                     const std::vector<core::SensorSpec>& sensors,
                     core::PixelFormat format,
                     const core::StreamHealthConfig& stream_health,
                     core::ISensorSink* slam_sink);

    core::RigCalibration get_calibration() const;
    // Opaque identity of the raw device clock these streams are stamped on. Orchestration
    // step 4 rejects an instance whose adapters do not all report the same domain. For OAK
    // this is the device mxid: one device, one oscillator, so cameras and IMU share it.
    std::string clock_domain() const;
    // Drains the camera queue (mono: one ImgFrame; stereo: one MessageGroup) and the IMU
    // queue, then dispatches BOTH interleaved by sample_time_device_ns — Phase 1 has no
    // SensorMultiplexer, so this class owns the monotonic-order guarantee and applies
    // m_stream_health's reorder window, late-drop counter, and skew/IMU-gap alerts.
    void update();

private:
    std::shared_ptr<dai::Device> m_device;
    std::unique_ptr<dai::Pipeline> m_pipeline;
    // SLAM inputs (raw frames + IMU; v1 does not record IMU). Exactly one camera queue is
    // populated, chosen by camera count at construction:
    //   1 camera  -> m_mono_queue,        update() calls on_image()
    //   2 cameras -> m_stereo_sync_queue, update() calls on_stereo_pair()
    // 3+ cameras is rejected by Orchestration step 3 and never reaches this class.
    // WHICH camera is never inferred: the mono rig's single non-"imu" SensorSpec names it
    // via `stream`, mapped to a socket exactly as in the stereo case —
    // "color" -> CAM_A, "left" -> CAM_B, "right" -> CAM_C (matches oak_camera.cpp).
    std::shared_ptr<dai::MessageQueue> m_mono_queue;          // raw ImgFrame
    std::shared_ptr<dai::MessageQueue> m_stereo_sync_queue;   // Sync node MessageGroup
    std::shared_ptr<dai::MessageQueue> m_imu_queue;
    core::ISensorSink* m_slam_sink;
    core::StreamHealthConfig m_stream_health;  // no mux in Phase 1: this class applies it
    core::RigCalibration m_calibration;
    // Video recording (existing OakCamera/RawDataWriter pattern)
    std::map<core::StreamType, std::shared_ptr<dai::MessageQueue>> m_h264_queues;
    std::map<core::StreamType, std::unique_ptr<RawDataWriter>> m_h264_writers;
};
```

The `slam_pose_provider` plugin in Phase 1 is a simplified variant of the full
orchestration: one `OakSensorAdapter` — the sole adapter, so **no `SensorMultiplexer` is
instantiated and the adapter owns merge-ordering and stream-health itself** (Timing A) —
feeding one `IPoseProvider` and one `SchemaPusher`. The per-tick drain is
`adapter.update()` rather than `mux.update()` (Orchestration step 7). Video recording is
opt-in: `m_h264_queues` and `m_h264_writers` are omitted from the pipeline when video
recording is off. IMU recording to MCAP is **not in v1** — the existing OAK plugin has no
IMU support at all, so it is new scope rather than a port; when added it would build on the
capture-interface recording path that Phase 2 introduces.

### Phase 2 — migration to the generic device interface

Once `ICamera`/`IImu` land with built-in fan-out, `OakSensorAdapter` is replaced by a
generic `SensorAdapter` bridging `ICamera`/`IImu` callbacks to `ISensorSink`. The
recording path moves to the device layer (the recording sink subscribes to `ICamera`
directly, as described in the Ego Data Capture Interface design). A `FanOutSensorSink`
can optionally sit between `SensorAdapter` and `IPoseProvider` if any recording concern
needs to tap the post-adapter, post-format-conversion stream. The `IPoseProvider`,
calibration stack, and pose recording path are untouched in this migration.

```cpp
// Distributes ISensorSink callbacks to N downstream sinks (e.g. IPoseProvider + RecordingSink).
// COST: ISensorSink takes frames BY VALUE, so N sinks means the last one is moved into and
// the other N-1 get a full image-sized copy. That is acceptable for the 1-extra-sink
// recording tap this is for, and unacceptable as a general bus — if a fan-out with several
// heavy consumers is ever needed, move ImageFrame to a shared immutable buffer first.
// Every sink registered here must be stereo-capable if the source is stereo; the mono-only
// rejecting stub would otherwise fire at runtime, which step 3 cannot catch (it validates
// only the provider's capabilities, not sinks attached behind a fan-out).
class FanOutSensorSink : public ISensorSink {
public:
    void add_sink(ISensorSink* sink);
    void on_image(ImageFrame) override;                              // fans out to each sink
    void on_imu(const ImuSample&) override;                          // fans out to each sink
    // Forwards the pair as a pair, preserving atomicity for stereo-aware downstream sinks.
    void on_stereo_pair(ImageFrame left, ImageFrame right) override;
};
```

## Open questions to resolve at implementation time

- OAK IMU + intrinsics extraction from DepthAI (absent today) — spike needed. Confirm the
  IMU node's timestamps come from the same `getTimestampDevice()` clock domain as frames.
- **`T_session_slamworld` source:** the plugin always reads it from `slam_topology.yaml` at
  startup and never computes it. Because the SLAM origin is re-established at each tracking
  init, a yaml constant is only valid for rigs that start from a repeatable physical pose
  (fixed mount / docking fixture); v1's default and tested path is `output_frame: SlamWorld`
  with offline alignment. **Open:** whether to add a live alignment procedure whose result
  survives restarts — that needs either a persisted map with relocalization, or an origin
  convention tied to an observable landmark rather than to the init pose.
- **Tracking-loss policy for live control** *(future work — not needed for v1 recording)*:
  `PoseEstimate` carries the full `TrackingState`, but `se3_tracker` only carries
  `is_valid`, and the consumer contract is "freeze on invalid." Decide what the teleop layer
  should do for `Initializing` (no pose for the first seconds of a session) vs `Lost`
  mid-motion vs `Relocalizing`. If freeze is wrong, this is the concrete trigger for the
  dedicated `slam_pose` device type (see Output) that can carry state rather than a bool.
- **Prediction design** *(future work — prerequisite for live control)*: a forward-prediction
  call integrating IMU past the last processed frame. Before adding it, pin: the maximum sane
  horizon (beyond which prediction error exceeds the latency it hides) and whether the call
  refuses rather than degrades past it; the normative derivation of a device-domain target
  time from the host monotonic clock; and the measured end-to-end latency budget per backend.
- **Relative head↔hand drift** when a SLAM rig replaces one tracked device while the head
  stays on XR: two independent trackers drift independently and a fixed
  `T_session_slamworld` cannot correct a time-varying error. Evaluate tracking both as rigs
  in one collaborative instance (shared map frame) versus periodic re-alignment.
- **ORB-SLAM3 (dev-only):** used as a local validation backend only. Developer supplies the
  GPLv3 dep; it is never vendored, FetchContent'd, redistributed, or included in release CI.
  Release-supported backends are cuVSLAM and `stub`. A formal licensing decision is required
  before ORB-SLAM3 can be included in any release artifact.
- Collaborative SLAM across separate machines needs an inter-process **image transport**
  (frames don't cross the tensor bus today) — deferred; single-host multi-device only for now.
- Calibration: pin the canonical convention (frame direction, quaternion order, units) and
  the supported `CameraCalibration::Model` set; adopt Kalibr yaml as the file format (vs the
  existing `so101_leader.calib` convention)? Confirm each backend adapter's native mapping.
  Decide handling of rolling-shutter readout and per-sensor mid-exposure/latency offsets.
- cuVSLAM: confirm the target version's multi-camera + visual-inertial support and its
  exact distortion/axis conventions; decide odometry-only vs full-SLAM (loop-closure) mode
  for teleop; settle vendoring of the prebuilt lib under `deps/`.
- Vendor SDKs to depend on: ZED SDK (CUDA, EULA) and the Sensing/GMSL access path
  (v4l2 vs Argus/NvSIPL — Jetson vs x86+dGPU); per-vendor IMU availability and calibration.
- Time sync: which rigs get hardware genlock vs software-offset fallback; the
  `SensorMultiplexer` reorder-buffer depth and skew-alert threshold; whether to expose a
  time-sync health signal to teleop; mid-exposure/latency-offset calibration per sensor.
