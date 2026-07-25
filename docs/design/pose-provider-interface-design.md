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

Decisions locked in:
- Sensor input: **generic sensor abstraction** (`ISensorSource`); OAK, ZED (Stereolabs),
  and Sensing (GMSL) are each one impl. A device that estimates pose natively (e.g. ZED
  SDK's built-in tracking) may instead be wrapped as an `IPoseProvider` — the source vs
  provider split handles both "dumb sensor" and "smart sensor" without new interfaces.
- Backend selection: **compile-time** (CMake option); heterogeneous backends across
  instances come from deploying **different per-backend binaries** (see Topologies).
- Output: **reuse `se3_tracker`** (`core::Se3TrackerPose`), no new device type.
- Topology: one interface spans all configs; must support **both** a rigid multi-camera
  rig (M sensors → 1 pose) and **collaborative** multi-device SLAM (M sensors → N poses in
  one shared map frame). See "SLAM topologies & multi-instance orchestration".

## Architecture

```
ISensorSource impl            IPoseProvider impl            producer plugin
(OakSensorSource, ...)        (OrbSlam3, Vins, ...)         (SE3 push path)
        │                            │                            │
  images + IMU  ──on_image/on_imu──▶ SLAM/VIO estimator           │
  + intrinsics                       │                            │
                                     └── get_latest_pose() ──▶ Se3TrackerPoseT
                                                                  │  SchemaPusher
                                                                  ▼  .push_buffer()
                                                     OpenXR tensor collection
                                                     (collection_id, "se3_tracker")
                                                                  ▼
                                             core::Se3Tracker (consumer, ALREADY built)
                                                → Python IDeviceIOSource → teleop graph
```

Three seams, each independently swappable: **sensor** (`ISensorSource`), **estimator**
(`IPoseProvider`), **transport** (existing `SchemaPusher` + `se3_tracker`, unchanged).

## Module placement

A new core library `src/core/pose_provider/` holding the interfaces and data types
(public headers under `inc/pose_provider/`, per the `cmake-structure` conventions).
Backend impls live in sibling dirs selected by CMake, e.g.
`src/core/pose_provider/backends/orbslam3/`, `.../vins/`, `.../stub/`. `ISensorSource`
lives here too; the OAK implementation lives with the OAK plugin
(`src/plugins/oak/`, which already owns the DepthAI pipeline). A future
`slam_pose_provider` plugin wires `OakSensorSource → IPoseProvider → SchemaPusher`,
modeled on `src/plugins/controller_se3_tracker/`.

## Interface 1 — `ISensorSource` (generic camera/IMU input)

Push model (matches OAK's existing `FrameSink::on_frame` callback style; frames and IMU
arrive asynchronously at different rates). The source owns hardware; the provider is the
sink.

**Delivery contract (timing-critical — see Timing & synchronization below):**
- Every sample carries **two clocks**: `sample_time_device_ns` (single hardware clock
  domain shared by camera + IMU — the clock VIO fuses on) and `sample_time_local_ns`
  (host monotonic, for teleop alignment). Host-arrival time alone desyncs image/IMU.
- The sink MUST receive samples in **monotonically non-decreasing `sample_time_device_ns`
  order** across `on_image`/`on_imu` (IMU interleaved with frames, not batched after).
- `on_image`/`on_imu` MUST be cheap (enqueue only). The provider runs SLAM on its own
  thread so a slow estimator cannot block the source and drop frames (backpressure).

```cpp
namespace core {

// Identifiers (see "SLAM topologies"): a SensorId names one physical stream (a camera or
// an IMU); a RigId names a rigid body of 1+ sensors with known extrinsics — one pose is
// estimated per rig. Stable strings so config files can reference them.
using SensorId = std::string;
using RigId = std::string;

enum class PixelFormat { Gray8, NV12, BGR888 };   // SLAM usually wants Gray8

struct ImageFrame {
    SensorId sensor_id;          // stable routing key -> (rig, role) via instance config
    uint32_t width, height, stride;
    PixelFormat format;
    const uint8_t* data;         // borrowed; valid only during the callback
    size_t size;
    int64_t sample_time_local_ns;   // common/monotonic clock (os_monotonic_now_ns)
    int64_t sample_time_device_ns;  // device clock
};

struct ImuSample {
    SensorId sensor_id;
    double accel[3];             // m/s^2
    double gyro[3];              // rad/s
    int64_t sample_time_local_ns;
    int64_t sample_time_device_ns;
};

// Per-sensor calibration below is the self-reportable subset; the full, resolved,
// backend-neutral model is RigCalibration (see "Calibration"). ISensorSource exposes what
// it can self-report as one calibration *source*; files/Kalibr provide the rest.
struct CameraIntrinsics {        // per sensor_id
    SensorId sensor_id;
    uint32_t width, height;
    double fx, fy, cx, cy;
    enum class Distortion { None, RadTan, Fisheye } model;
    std::vector<double> coeffs;  // model-dependent
};

class ISensorSink {
public:
    virtual ~ISensorSink() = default;
    virtual void on_image(const ImageFrame&) = 0;
    virtual void on_imu(const ImuSample&) = 0;
};

class ISensorSource {
public:
    virtual ~ISensorSource() = default;
    // Self-reported calibration = one calibration *source* (empty when the device can't
    // self-report, e.g. GMSL — then a FileCalibrationSource supplies it). See "Calibration".
    virtual std::vector<CameraIntrinsics> get_self_reported_intrinsics() const = 0;
    // Clock domain of this source's sample_time_device_ns, so the orchestrator knows whether
    // two sources' device stamps are directly comparable (same synced domain) or must be
    // bridged via the host clock. See "Multi-sensor / inter-device time sync".
    virtual SensorTimebase get_timebase() const = 0;
    // Register sink, then drive: run() blocks; or poll()/update() per tick.
    virtual void set_sink(ISensorSink* sink) = 0;
    virtual void update() = 0;    // drains available samples -> sink callbacks
};

struct SensorTimebase {
    // Sources sharing domain_id have directly-comparable sample_time_device_ns (hardware
    // synced). Distinct ids are only comparable after mapping to the host clock.
    uint32_t domain_id;
    enum class Sync { HardwareTrigger, Ptp, SoftwareOffset } method;
    bool device_to_host_available;   // true if sample_time_local_ns is a trustworthy mapping
};

} // namespace core
```

Note the two **gaps in the OAK plugin today** that an `OakSensorSource`
impl must fill — neither exists now:
- **IMU capture** — no `dai::node::IMU` in `src/plugins/oak/`; must be added.
- **Intrinsics/extrinsics** — nothing calls DepthAI `getCalibrationData()`; must be added.
Also, OAK currently H.264-encodes frames to file; SLAM needs raw `Gray8`, so the source
impl exposes a raw output queue (like `preview_stream.cpp` does for the SDL preview).

### Multiple camera vendors (each one `ISensorSource` impl)

The estimator/topology layers never change across vendors — only the source impl differs:
- **OAK** (`OakSensorSource`, DepthAI) — built-in IMU (needs adding), self-reported calib.
- **ZED** (`ZedSensorSource`, ZED SDK, CUDA) — built-in IMU + factory calib. ZED also does
  its own VIO, so it can alternatively be a `ZedSdkPoseProvider : IPoseProvider` that
  forwards the SDK pose (skip the external SLAM entirely).
- **Sensing** (`SensingSensorSource`, GMSL via v4l2/Argus/NvSIPL) — often **no IMU**
  (visual-only, or pair with an IMU from a *different* sensor in the same rig), external
  calibration file (Kalibr). HW-synchronized multi-camera capture suits the rigid-rig
  topology and the device-clock timestamp requirement.

Design consequences already covered by the interfaces:
- **IMU can be a separate physical device from the cameras** in a rig — this is why
  `SensorId` is per-stream and `RigSpec.sensors` is a list; `ImuExtrinsics` ties an IMU
  sensor to a reference camera by id.
- **Self-reported vs file-based calibration** — `CameraIntrinsics` (source-reported) and
  `RigSpec.calibration` sources (loaded) cover both.

## Interface 2 — `IPoseProvider` (swappable estimator, multi-in / multi-out)

The provider is an `ISensorSink` (consumes tagged image+IMU from **one or more** sensors)
and estimates a pose **per rig**. This single shape covers every topology: 1 rig = per-
device or rigid-rig SLAM; N rigs in one shared map frame = collaborative SLAM (see
Topologies). The plugin polls all currently-tracked rig poses each tick.

```cpp
namespace core {

enum class TrackingState { Initializing, Tracking, Lost, Relocalizing };

struct PoseEstimate {
    Pose pose;                   // core::Pose: position (m) + Hamilton quat, in the instance's map frame
    TrackingState state;
    int64_t sample_time_local_ns;
    int64_t sample_time_device_ns;
    // Optional SLAM-only extras (dropped when emitting se3_tracker; see Output):
    std::optional<std::array<double, 36>> covariance;  // 6x6 row-major
};

struct RigPose {
    RigId rig_id;
    PoseEstimate pose;           // for collaborative instances, all rigs share ONE map frame
};

class IPoseProvider : public ISensorSink {
public:
    virtual ~IPoseProvider() = default;
    // Topology (rigs, sensors, backend config path) plus each rig's resolved, canonical
    // RigCalibration (see Calibration) come from the instance config. The backend adapter
    // translates RigCalibration into its own native format.
    virtual void initialize(const SlamInstanceConfig&) = 0;
    // Inherited from ISensorSink: on_image(...), on_imu(...) — samples carry sensor_id, which
    // the provider maps to a rig via its config.
    // Appends a RigPose for every rig with a fresh estimate since the last call; returns count.
    virtual size_t get_latest_poses(std::vector<RigPose>& out) = 0;
    virtual void reset() = 0;
    virtual const char* backend_name() const = 0;

    // Capability introspection so the plugin can validate a config against the linked backend
    // and fail fast (e.g. a mono-VIO backend rejects a 2-rig collaborative config).
    virtual bool supports_multi_sensor_rig() const = 0;  // rigid multi-camera rig
    virtual bool supports_multi_rig() const = 0;         // collaborative, shared frame
};

} // namespace core
```

## Compile-time backend selection

A CMake option chooses exactly one backend to build/link, so only its heavy deps
(Pangolin/Eigen/Ceres/…) are pulled in. A single factory, guarded by the same macro,
returns the concrete provider — the plugin only ever sees `IPoseProvider`.

```cmake
set(POSE_PROVIDER_BACKEND stub CACHE STRING "cuvslam | orbslam3 | vins | stub")
```

```cpp
// pose_provider/factory.cpp
std::unique_ptr<IPoseProvider> core::create_pose_provider() {
#if defined(POSE_PROVIDER_CUVSLAM)
    return std::make_unique<CuVslamPoseProvider>();   // in-house, GPU (default target)
#elif defined(POSE_PROVIDER_ORBSLAM3)
    return std::make_unique<OrbSlam3PoseProvider>();
#elif defined(POSE_PROVIDER_VINS)
    return std::make_unique<VinsPoseProvider>();
#else
    return std::make_unique<StubPoseProvider>();   // pass-through / mock
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
  mostly translates `RigSpec`/`CameraIntrinsics`/`ImuExtrinsics` into that registration.
- **Capabilities:** `supports_multi_sensor_rig() = true` (multi-camera rig is a headline
  cuVSLAM feature); `supports_multi_rig() = false` (cuVSLAM tracks one body — collaborative
  multi-agent needs per-agent cuVSLAM instances + an external map-merge layer, out of
  cuVSLAM's scope). The capability flags make an over-scoped config fail fast.
- **Timing:** cuVSLAM requires image+IMU on one synchronized clock — exactly the Timing &
  synchronization contract; reinforces it rather than straining it.
- **Output:** returns pose + covariance + tracking state → `PoseEstimate`; adapter converts
  cuVSLAM's axis convention into the `se3_tracker` `Pose` convention.
- **Loop-closure caveat (teleop):** full-SLAM mode does pose-graph optimization → pose
  *jumps* on map correction. Prefer pure visual-inertial **odometry** mode for teleop
  unless the consumer tolerates discontinuities.
- **Build/deploy:** ships as a **prebuilt binary** (`.so` + header) + CUDA runtime + GPU —
  so its CMake backend links a *vendored prebuilt lib* (like `deps/cloudxr`), not
  FetchContent-from-source. The repo already uses `find_package(CUDAToolkit)` (viz,
  `examples/camera_viz/codec`), so the CUDA/GPU linkage story exists.

## SLAM topologies & multi-instance orchestration

All requested topologies are the **same `IPoseProvider` interface** (M sensors → N rig
poses) at different configs. Topology is a *wiring/config concern*, not new interfaces:

| Topology | Sensors in | Rigs out | Instances | `IPoseProvider` config |
|---|---|---|---|---|
| Per-device (independent SLAM each) | 1 | 1 | N | 1 rig, 1 sensor set |
| Centralized, rigid multi-cam rig | M | 1 | 1 | 1 rig, M sensors (known extrinsics) |
| Centralized, collaborative | M | N | 1 | N rigs, shared map frame |
| Hybrid / groups | mixed | mixed | mixed | mix of the above |

### The load-bearing constraint: no image transport → topology = process decomposition

Raw frames never cross the tensor bus (only tiny metadata/poses do). So **all sensors
feeding one estimator instance must live in the same OS process as that estimator.** This
fixes the process model:

- **Per-device** → N separate `slam_pose_provider` processes (one per device), which the
  existing `PluginManager` forks independently → fault isolation (one SLAM crash doesn't
  kill the others). This is the natural fit for the current plugin-process architecture.
- **Centralized (rigid or collaborative)** → one process that opens *all* its sensors
  directly (DepthAI supports multiple devices per process). Bounded to sensors reachable
  from that one host — **collaborative SLAM across physically separate machines is out of
  scope until we have an image transport** (a real future work item, not a config flag).

### Config (declarative topology; lives in the orchestration/plugin layer, not the estimator)

One `SlamInstanceConfig` per process. yaml-cpp is already a dependency (`plugin_manager`
parses `plugin.yaml` with it), so a sibling `slam_topology.yaml` per instance is natural.

```cpp
struct SensorSpec {              // how to open one physical stream
    SensorId sensor_id;
    std::string device;          // "oak"
    std::string device_serial;   // DepthAI mxid; empty = any
    std::string stream;          // "left" | "right" | "color" | "imu"
};
struct RigSpec {
    RigId rig_id;
    std::vector<SensorId> sensors;   // 1 = per-device; >1 = rigid multi-cam rig
    // Ordered calibration sources (later overrides earlier), e.g. [device-self-report,
    // kalibr.yaml]. Resolved into a canonical RigCalibration at startup. See "Calibration".
    std::vector<CalibrationSourceSpec> calibration;
};
struct OutputSpec {
    RigId rig_id;
    std::string collection_id;                 // se3_tracker wire instance for this rig
    enum class Frame { SlamWorld, Session } output_frame;
};
struct SlamInstanceConfig {
    std::string instance_id;
    std::vector<SensorSpec> sensors;
    std::vector<RigSpec> rigs;       // 1 rig = per-device/rigid; N rigs = collaborative
    std::vector<OutputSpec> outputs;
    std::string backend_config_path; // ORB vocab / VINS yaml
};
```

### Orchestration (one process = one instance = one backend binary)

The `slam_pose_provider` plugin generalizes `controller_se3_tracker` by owning **multiple**
sources and **multiple** pushers:
1. Load `SlamInstanceConfig`; open every `SensorSpec` via an `ISensorSource` impl.
2. Feed all sources through a **`SensorMultiplexer`** (single time-ordered stream on one
   fusion timeline; see Timing & synchronization C) into the one provider
   (`create_pose_provider()` = this binary's backend), each sample tagged with its
   `sensor_id`.
3. Validate the config against the backend's capabilities (`supports_multi_rig()` /
   `supports_multi_sensor_rig()`) **and against clock domains** (reject an unsynced
   multi-device rig unless software-offset fallback is explicitly allowed); fail fast with a
   clear message on mismatch.
4. Per tick: `get_latest_poses()` → for each `RigPose`, look up its `OutputSpec`, apply the
   `output_frame` transform, and push `Se3TrackerPose` on that rig's `collection_id` via a
   **per-output `SchemaPusher`** (N pushers in one process; each is an independent
   `se3_tracker` device on the consumer).

Deployment example (three processes, mixed backends, one collaborative):
```
slam_pose_provider_orbslam3 --instance head_a.yaml   # 1 rig  -> se3_tracker "slam_head_a"
slam_pose_provider_orbslam3 --instance head_b.yaml   # 1 rig  -> se3_tracker "slam_head_b"
slam_pose_provider_vins     --instance room.yaml     # 2 rigs -> "slam_head_a2","slam_head_b2", shared frame
```

### Frame semantics per topology (refines the reference-frame concern below)

- **Per-device / rigid rig (1 rig):** pose in that instance's own arbitrary SLAM world
  frame. `output_frame: Session` applies a fixed `T_session_slamworld`; else document it as
  its own frame.
- **Collaborative (N rigs):** all rigs share ONE map frame (the whole value — poses are
  mutually consistent), so a single instance-level `T_session_slamworld` aligns the whole
  group into session space at once.

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
    std::vector<double> intrinsics;      // fx,fy,cx,cy (+ model params)
    std::vector<double> distortion;
    enum class Shutter { Global, Rolling } shutter;
    double rolling_readout_ns;           // when Rolling
    std::array<double,16> T_body_cam;    // body(rig)-from-camera — star graph, any pair derivable
};
struct ImuCalibration {
    SensorId sensor_id;
    double accel_noise_density, gyro_noise_density, accel_random_walk, gyro_random_walk;
    double rate_hz;
    std::array<double,16> T_body_imu;    // body-from-IMU
    double time_offset_ns;               // cam↔IMU temporal calibration (Kalibr output)
};
struct RigCalibration {                  // one rig; optionality encodes setup variety
    RigId rig_id;
    std::vector<CameraCalibration> cameras;   // length = mono/stereo/multi
    std::optional<ImuCalibration> imu;        // absent = visual-only
};
```

**Layered sources (`RigSpec.calibration`).** An `ICalibrationSource` yields a partial
`RigCalibration`; the orchestrator merges an ordered list (later overrides earlier) into the
resolved model at startup:
- `DeviceCalibrationSource` — from the `ISensorSource` self-report (OAK `getCalibrationData()`,
  ZED SDK factory calib).
- `FileCalibrationSource` — Kalibr yaml / native file (the only option for GMSL/Sensing).
- Mixed is the common case: device intrinsics as base, Kalibr file overriding cam↔IMU +
  IMU noise. `CalibrationSourceSpec` = `{ kind: device|file, path?, overrides? }`.

**Backend adapters translate canonical → native** (cuVSLAM rig struct / ORB-SLAM3 yaml / VINS
yaml), owning distortion-model and axis-convention mapping. The interface only ever carries
`RigCalibration`.

**Validation at startup** (`validate(RigCalibration, backend_capabilities)`): distortion model
supported by the backend? extrinsic star graph complete? IMU noise present when the backend
needs VIO? units/convention sane? Fail fast — reuses the capability introspection from
Topologies. Online refinement (VIO backends refine cam↔IMU/time-offset live) is a backend
concern; the canonical model supplies the *prior*.

## Output — reuse `se3_tracker` (recommended)

The plugin's per-tick loop maps each `RigPose` onto the existing SE3 push path — one
`SchemaPusher` per output rig (`controller_se3_tracker_plugin.cpp:67-105` is the single-rig
template):

```cpp
std::vector<core::RigPose> poses;
provider.get_latest_poses(poses);
for (const auto& rp : poses) {
    core::SchemaPusher& pusher = pusher_for(rp.rig_id);   // per-OutputSpec collection_id
    core::Se3TrackerPoseT out;
    if (rp.pose.state == core::TrackingState::Tracking) {
        out.pose = std::make_shared<core::Pose>(apply_output_frame(rp));  // -> Session if configured
        out.is_valid = true;
    } else {
        out.pose = std::make_shared<core::Pose>(/* identity */);
        out.is_valid = false;               // tracking-lost filler; consumers freeze
    }
    // Pack -> builder.Finish() -> pusher.push_buffer(ptr, size, t_local, t_device)
}
```

Why this is the whole point of the design:
- **Zero consumer-side work.** `core::Se3Tracker` is already registered in the live and
  replay factory dispatch tables (`live_deviceio_factory.cpp:141`); record/replay via MCAP
  and the Python `IDeviceIOSource` bridge come for free.
- The SLAM state machine collapses onto the schema's `is_valid` two-level validity
  (`se3_tracker.fbs`): `is_valid = (state == Tracking)`.
- Wire rendezvous constants must match the reader: `tensor_identifier =
  Se3Tracker::TENSOR_IDENTIFIER ("se3_tracker")`, `max_flatbuffer_size =
  Se3Tracker::DEFAULT_MAX_FLATBUFFER_SIZE (256)`; pick a unique `collection_id` per **rig**
  (from `OutputSpec`, e.g. `"slam_head_a"`). Multiple rigs = multiple `se3_tracker` devices,
  each independently registerable on the consumer.

**Key integration concern — reference frame.** `se3_tracker.fbs` requires the producer to
document its reference frame. A SLAM pose starts in an arbitrary map/world frame (origin at
init), *not* the OpenXR session base space that head/controllers use. The plugin must apply
a fixed `T_session_slamworld` transform (from calibration or an init-time alignment) before
pushing, or explicitly document that this device streams in its own SLAM world frame.

**What we give up:** covariance / detailed tracking state / map quality don't fit
`Se3TrackerPose` (only pose + `is_valid`). If those become required, add a dedicated
`slam_pose` device type later (new schema + tracker triple + two factory rows). Not needed
for the first pass — the `is_valid` bool already carries the essential tracking-lost signal.

## Timing & synchronization

Two distinct delays exist between the seams; they need different handling.

**A. Inter-interface delay (source → provider).** In-process callbacks, so the delay is
small — but the hazard is **backpressure and ordering**, not latency:
- Provider `on_image`/`on_imu` are enqueue-only; SLAM runs on the provider's own worker
  thread. A synchronous heavy estimator would stall the source's `update()` loop and drop
  frames / back up IMU.
- The sensor sink delivers in **monotonic device-time order** (IMU interleaved with
  frames). If a source cannot guarantee this, the provider sorts its input buffer by
  `sample_time_device_ns`. VIO integrates IMU *between* consecutive frames, so out-of-order
  or dropped IMU corrupts the estimate.

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
jitter would desync them). OAK/DepthAI supplies this via `getTimestampDevice()` — a single
device clock across camera and IMU (`src/plugins/oak/core/oak_camera.cpp:155-166` already
reads it for the camera). `sample_time_local_ns` is carried in parallel purely so the
resulting SE3 pose aligns with the other teleop device streams on the common clock.

**C. Multi-sensor / inter-device time sync.** Four layers, easiest to hardest:
1. *Intra-device (cam ↔ its own IMU):* same on-device oscillator → trivially synced. **The
   preferred config is one physical camera providing stereo + IMU** (per-device topology,
   zero sync work).
2. *Device → host clock:* device oscillators drift (ppm) vs host; DepthAI/ZED estimate the
   offset and emit both stamps. The host stamp is what aligns the SLAM pose with the rest of
   teleop (transport already stamps everything on `os_monotonic_now_ns()`).
3. *Inter-device (multi-cam rig, or cam + separate IMU) — the hard case.* Independent clocks
   are **not directly comparable**. Handle, best first: (a) **hardware sync / genlock**
   (shared trigger; GMSL native, some OAK via FSIN) → one capture clock, comparable device
   stamps — the right answer for rigid multi-cam rigs; (b) **PTP** for networked sensors;
   (c) **software offset estimation** to the host timeline — sub-ms to few-ms, needs online
   drift tracking, **marginal for tight VIO** (fallback only, avoid for the primary cam+IMU).
4. *Exposure convention:* per-frame stamp is **mid-exposure** (not arrival/shutter-start),
   consistent across cameras, latency-corrected; rolling-shutter caveat. Never fuse on
   host-arrival time.

**Who owns alignment: the orchestrator, not the estimator.** Sources *declare* their clock
domain via `get_timebase()`; a **`SensorMultiplexer`** in the plugin merges the N source
streams into one coherent, monotonic, single-timeline stream (k-way merge by fusion
timestamp + a small bounded reorder buffer to absorb per-source jitter), and **monitors
inter-sensor skew, flagging threshold breaches** (VIO degrades silently otherwise). The
provider stays sync-agnostic: by the time samples reach it they are already one coherent
fusion timeline satisfying the monotonic-ordering contract in (A). Same-domain sources pass
device stamps through; cross-domain sources are rewritten to the host timeline first.

Recommendation ranking: (1) single camera with built-in IMU; (2) hardware-synced multi-cam
rig (GMSL/FSIN), IMU on the same sync domain; (3) software-offset fallback only, with skew
monitoring — not for tight VIO.

## Files (design-level; no implementation this pass)

- `src/core/pose_provider/cpp/inc/pose_provider/sensor_source.hpp` — `ISensorSource`,
  `ISensorSink`, `ImageFrame`, `ImuSample`, `CameraIntrinsics`, `SensorTimebase`.
- `src/core/pose_provider/cpp/inc/pose_provider/pose_provider.hpp` — `IPoseProvider`,
  `PoseEstimate`, `RigPose`, `TrackingState`, `SensorId`/`RigId`, `create_pose_provider()`.
- `src/core/pose_provider/cpp/inc/pose_provider/topology.hpp` — `SensorSpec`, `RigSpec`,
  `OutputSpec`, `SlamInstanceConfig` (+ a yaml-cpp loader for `slam_topology.yaml`).
- `src/core/pose_provider/cpp/.../sensor_multiplexer.*` — merges N `ISensorSource`s into one
  time-ordered, single-fusion-timeline stream; reorder buffer + inter-sensor skew monitor.
- `src/core/pose_provider/cpp/inc/pose_provider/calibration.hpp` — canonical `RigCalibration`
  (`CameraCalibration`/`ImuCalibration`), `ICalibrationSource` (`Device`/`File`), the
  precedence-merge resolver, and `validate(RigCalibration, capabilities)`. Backend adapters
  live with each backend and translate canonical → native.
- `src/core/pose_provider/backends/{cuvslam,orbslam3,vins,stub}/` — one lib each, selected
  by `POSE_PROVIDER_BACKEND`; built into per-backend plugin binaries for heterogeneity.
  cuVSLAM links a vendored prebuilt `.so` (+ CUDA); others FetchContent-from-source.
- Sensor sources (one per vendor): `src/plugins/oak/core/oak_sensor_source.*`
  (`OakSensorSource`, adds IMU node + raw-frame queue + `getCalibrationData()`);
  `ZedSensorSource` (ZED SDK; or a `ZedSdkPoseProvider` using built-in tracking);
  `SensingSensorSource` (GMSL via v4l2/Argus/NvSIPL, external calib, optional external IMU).
- `src/plugins/slam_pose_provider/` — producer plugin generalizing `controller_se3_tracker`:
  opens N `ISensorSource`s → multiplex into one `create_pose_provider()` → **per-rig**
  `SchemaPusher`s on `se3_tracker`. One process per `SlamInstanceConfig`.

Reused as-is: `core::SchemaPusher` (`src/core/pusherio/`), `core::Se3Tracker` +
`Se3TrackerPose` schema, `os_monotonic_now_ns()`, the `controller_se3_tracker` plugin
skeleton, `DeviceIOSession`/`OpenXRSession`.

## Verification (of the design, before/after implementation)

- **Design review acceptance:** a new backend can be added by implementing `IPoseProvider`
  + adding one CMake branch + one factory `#elif`, touching neither `ISensorSource`, the
  plugin, nor any consumer. A new camera = one `ISensorSource` impl. All four topologies
  (per-device, rigid rig, collaborative, hybrid) express as `SlamInstanceConfig` data with
  no interface change. Confirm `PoseEstimate` → `Se3TrackerPose` mapping is lossless for the
  fields `se3_tracker` carries.
- **Once implemented (out of scope now):** build with `POSE_PROVIDER_BACKEND=stub`, run the
  plugin, read the pose back with `examples/schemaio/se3_printer.cpp` on the chosen
  `collection_id` — the same reader used for `controller_se3_tracker` — to confirm the pose
  and `is_valid` gating flow end-to-end without any consumer changes.

## Open questions to resolve at implementation time

- OAK IMU + intrinsics extraction from DepthAI (absent today) — spike needed. Confirm the
  IMU node's timestamps come from the same `getTimestampDevice()` clock domain as frames.
- SLAM world-frame → OpenXR session-space alignment strategy (fixed calib vs init-time).
- Latency budget: is propagating the correct pose timestamp enough, or does teleop need
  the provider to forward-predict to "now" (see Timing & synchronization B)? Decide per
  use case; affects whether `IPoseProvider` gains a `predict_to(t_ns)` method.
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
