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

- Sensor input: **device interface layer** (`ICamera`/`IImu`, defined in the Ego Data
  Capture Interface design — out of scope here); bridged to the Pose Provider via a
  `SensorAdapter` in the plugin. A device that estimates pose natively (e.g. ZED SDK's
  built-in tracking) may instead be wrapped directly as an `IPoseProvider`, bypassing the
  sensor layer entirely.
- Backend selection: **compile-time** (CMake option); heterogeneous backends across
  instances come from deploying **different per-backend binaries** (see Topologies).
- Output: **reuse `se3_tracker`** (`core::Se3TrackerPose`), no new device type.
- Topology: one interface spans all configs; must support **both** a rigid multi-camera
  rig (M sensors → 1 pose) and **collaborative** multi-device SLAM (M sensors → N poses in
  one shared map frame). See "SLAM topologies & multi-instance orchestration".
- Consumption profile: the primary target is **egocentric data collection** (pose recorded,
  aligned offline). Driving a robot live from a SLAM pose is supported but materially
  stricter — declared per instance via `SlamInstanceConfig::profile` and enforced at
  startup. See "Real-time teleop vs data collection".

## Architecture

```
Device layer                   IPoseProvider impl             producer plugin
(ICamera + IImu, out of scope) (cuVSLAM, ORB-SLAM3, ...)      (SE3 push path)
        │                            │                            │
  images + IMU  ──on_image/on_imu──▶ SLAM/VIO estimator           │
  [via SensorAdapter]                │                            │
                                     └── get_latest_poses() ──▶ Se3TrackerPoseT
                                                                  │  SchemaPusher
                                                                  ▼  .push_buffer()
                                                     OpenXR tensor collection
                                                     (collection_id, "se3_tracker")
                                                                  ▼
                                             core::Se3Tracker (consumer, ALREADY built)
                                                → Python IDeviceIOSource → teleop graph
```

Three seams, each independently swappable: **device layer** (`ICamera`/`IImu`, defined
elsewhere), **estimator** (`IPoseProvider`), **transport** (existing `SchemaPusher` +
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
Backend impls live in sibling dirs selected by CMake, e.g.
`src/core/pose_provider/backends/orbslam3/`, `.../vins/`, `.../stub/`. A future
`slam_pose_provider` plugin wires the device layer → `SensorAdapter` → `IPoseProvider`
→ `SchemaPusher`, modeled on `src/plugins/controller_se3_tracker/`. The device layer
(`ICamera`, `IImu`) is defined in the Ego Data Capture Interface design and is out of
scope here.

## Interface 1 — Device interface requirements and `ISensorSink`

> **Out of scope:** The design of the device interface layer (`ICamera`, `IImu`, vendor
> backends, and the `SensorAdapter` implementation) is defined in the **Ego Data Capture
> Interface design** (IsaacTeleop#571). This document only specifies what the Pose Provider
> needs from that layer, and defines `ISensorSink` as the boundary the provider exposes.

### What the Pose Provider requires from the device interface layer

The `slam_pose_provider` plugin bridges the device interface layer to `ISensorSink` via a
`SensorAdapter`. For that bridge to work correctly the device layer must provide:

1. **Push delivery** — frames and IMU arrive via callbacks, not polled. The provider's
   `on_image`/`on_imu` are enqueue-only; the provider runs SLAM on its own thread. A
   blocking or synchronous delivery model would stall the capture path and drop frames.
2. **Two timestamps per sample** — `sample_time_device_ns` (hardware clock, the clock VIO
   fuses on) and `sample_time_local_ns` (host monotonic, for teleop stream alignment).
   Host-arrival time alone is not sufficient; USB/scheduling jitter desyncs image and IMU.
3. **Format flexibility** — the ability to open a stream in a requested pixel format. The
   backend declares which formats it accepts (Gray8 for classical SLAM, BGR/RGB for learned
   approaches) via `supported_pixel_formats()`; the orchestrator requests the first one the
   device can produce natively, and the `SensorAdapter` converts only if none matched.
4. **IMU at the required rate** — ≥200 Hz with device-clock timestamps on the same clock
   domain as the camera where possible (intra-device IMU is strongly preferred; cross-device
   IMU requires clock bridging — see Timing & synchronization C).
5. **Calibration at open time** — per-camera intrinsics and inter-camera extrinsics from
   the camera interface; IMU spatial transform, time offset, and noise parameters from the
   IMU interface. These are the device self-report source; file-based sources (Kalibr)
   override them in the calibration merge (see Calibration).
6. **Fan-out** *(Phase 2 only)* — recording and SLAM subscribe to the same camera
   independently, without either consumer knowing about the other. Phase 1 does not depend
   on this: `OakSensorAdapter` owns its own DepthAI pipeline and serves both paths from
   separate queues (see Phased implementation).

### `SensorAdapter` contract (plugin layer, not a public interface)

The `SensorAdapter` lives in the `slam_pose_provider` plugin. It is not a reusable
interface — it is the glue that bridges the device layer to `ISensorSink`. Its obligations:

- Subscribes to `ICamera` and `IImu` independently (fan-out: the recording path also
  subscribes to the same camera).
- Populates both timestamps on every `ImageFrame`/`ImuSample`, and forwards samples **in the
  order the device produced them** — it does not buffer or reorder. Cross-source ordering is
  the `SensorMultiplexer`'s job (see Timing & synchronization C); a single adapter only
  guarantees it does not itself introduce reordering between its own camera and IMU streams.
- Performs pixel format conversion only when the device could not natively produce a format
  from the backend's `supported_pixel_formats()`.
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
    const uint8_t* data;             // borrowed; valid only during the callback
    size_t size;
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
    virtual void on_image(const ImageFrame&) = 0;  // must be cheap; enqueue only
    virtual void on_imu(const ImuSample&) = 0;     // must be cheap; enqueue only
};

} // namespace core
```

`IPoseProvider` inherits `ISensorSink` (see Interface 2). The `SensorAdapter` in the plugin
calls these after pulling from `ICamera`/`IImu` and resolving ordering and format.

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
    // The orchestrator resolves each rig's CalibrationSourceSpec list into a canonical
    // RigCalibration before calling initialize(), then passes the results alongside the
    // config. Entries are matched to config.rigs by RigCalibration::rig_id (not by
    // position). The backend adapter translates each RigCalibration into its native format.
    virtual void initialize(const SlamInstanceConfig& config,
                            const std::vector<RigCalibration>& calibrations) = 0;
    // Inherited from ISensorSink: on_image(...), on_imu(...) — samples carry sensor_id, which
    // the provider maps to a rig via its config.
    // Clears `out`, then fills it with one RigPose per rig that has a fresh estimate since
    // the last call; returns out.size(). Safe to reuse the same vector across ticks.
    virtual size_t get_latest_poses(std::vector<RigPose>& out) = 0;
    virtual void reset() = 0;
    virtual const char* backend_name() const = 0;

    // Forward-predicts each tracked rig to target_time_device_ns (typically "now" or the
    // next display time) by integrating IMU past the last processed frame. Required for
    // real-time teleop, where a pose stamped tens of ms in the past reads as lag; optional
    // for data collection, where correct timestamps suffice. Backends that cannot predict
    // return the unpredicted estimate and report supports_prediction() == false, which the
    // orchestrator rejects when the instance is configured for teleop use.
    // See "Real-time teleop vs data collection".
    virtual size_t predict_to(int64_t target_time_device_ns, std::vector<RigPose>& out) = 0;
    virtual bool supports_prediction() const = 0;

    // --- Capability introspection ---------------------------------------------------
    // All const and valid BEFORE initialize(), so the plugin can validate a config and pick
    // stream formats before opening any hardware (see Orchestration).
    virtual bool supports_multi_sensor_rig() const = 0;  // rigid multi-camera rig
    virtual bool supports_multi_rig() const = 0;         // collaborative, shared frame

    // Pixel formats this backend accepts, most-preferred first. The orchestrator opens each
    // stream in the first entry the device supports natively; the SensorAdapter converts
    // only if none matched. Classical SLAM typically returns {Gray8}; a learned backend may
    // return {BGR888, Gray8}.
    virtual std::vector<PixelFormat> supported_pixel_formats() const = 0;
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
set(POSE_PROVIDER_BACKEND stub CACHE STRING "cuvslam | orbslam3 | vins | stub")
```

```cpp
// pose_provider/factory.cpp
std::unique_ptr<IPoseProvider> core::create_pose_provider() {
#if defined(POSE_PROVIDER_CUVSLAM)
    return std::make_unique<CuVslamPoseProvider>();   // in-house, GPU (production target)
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
  mostly translates `RigCalibration` (cameras + IMU) into that registration.
- **Capabilities:** `supports_multi_sensor_rig() = true` (multi-camera rig is a headline
  cuVSLAM feature); `supports_multi_rig() = false` (cuVSLAM tracks one body — collaborative
  multi-agent needs per-agent cuVSLAM instances + an external map-merge layer, out of
  cuVSLAM's scope). The capability flags make an over-scoped config fail fast.
- **Timing:** cuVSLAM requires image+IMU on one synchronized clock — exactly the Timing &
  synchronization contract; reinforces it rather than straining it.
- **Output:** returns pose + covariance + tracking state → `PoseEstimate`; adapter converts
  cuVSLAM's axis convention into the `se3_tracker` `Pose` convention.
- **Loop-closure caveat (teleop):** full-SLAM mode does pose-graph optimization → pose
  *jumps* on map correction. This is what `SlamInstanceConfig::odometry_only` disables, and
  it is forced on for `profile == RealtimeControl` (see "Real-time teleop vs data
  collection").
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
    std::string device;          // "oak" | "zed" | "sensing" — selects the source impl
    std::string device_serial;   // DepthAI mxid; empty = any
    // Vendor-neutral role name, kept a string so slam_topology.yaml stays readable and no
    // vendor enum leaks into the config schema. Each source impl owns the mapping to its
    // own enum (e.g. OakSensorAdapter: "left" -> core::StreamType_MonoLeft); an unknown
    // name for that device is a startup error.
    std::string stream;          // "left" | "right" | "color" | "imu"
};
struct CalibrationSourceSpec {
    enum class Kind { Device, File } kind;
    std::string path;   // when kind == File; empty for Device
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
    std::string collection_id;                    // se3_tracker wire instance for this rig
    enum class Frame { SlamWorld, Session } output_frame;
    // Required when output_frame == Session; startup validation rejects Session with this
    // unset. Rigid transform aligning the SLAM world origin (set at tracker init) into the
    // OpenXR session base space. Typed as core::Pose — the same type it is composed with
    // per-frame — rather than a 4x4, unlike the calibration extrinsics which stay matrices
    // because that is what calibration files and backend APIs use. Source: a calibration
    // file or an init-time alignment procedure (see Open questions).
    std::optional<Pose> T_session_slamworld;
};
struct SlamInstanceConfig {
    std::string instance_id;
    std::vector<SensorSpec> sensors;
    std::vector<RigSpec> rigs;       // 1 rig = per-device/rigid; N rigs = collaborative
    std::vector<OutputSpec> outputs;
    std::string backend_config_path; // ORB vocab / VINS yaml

    // Declares what this instance's pose is consumed for. Not a hint — it changes which
    // backend behaviours are legal and is enforced at startup.
    // See "Real-time teleop vs data collection".
    enum class Profile {
        DataCollection,  // pose is recorded and aligned offline; latency tolerated
        RealtimeControl, // pose drives a robot live; requires prediction, forbids pose jumps
    } profile = Profile::DataCollection;

    // Disables loop closure / pose-graph optimisation, running the backend as pure
    // visual-inertial odometry. Drift accumulates, but the pose never jumps on map
    // correction. Forced true when profile == RealtimeControl.
    bool odometry_only = false;
};
```

### Orchestration (one process = one instance = one backend binary)

The `slam_pose_provider` plugin generalizes `controller_se3_tracker` by owning **multiple**
sources and **multiple** pushers. Ordering matters: the backend is created **first** so its
capabilities and accepted formats are known, and validation runs **before** any hardware is
opened, so a bad config fails without touching a device.

1. Load `SlamInstanceConfig`.
2. `create_pose_provider()` — this binary's backend. Its capability methods
   (`supports_multi_rig()`, `supports_multi_sensor_rig()`, `supported_pixel_formats()`) are
   const and valid before `initialize()`, so they can drive the next two steps.
3. **Validate before opening hardware:** config vs backend capabilities (reject e.g. a 2-rig
   collaborative config on a single-body backend; reject `profile == RealtimeControl` on a
   backend reporting `supports_prediction() == false`), and config self-consistency (every
   `OutputSpec.rig_id` resolves to a `RigSpec`; `output_frame == Session` implies
   `T_session_slamworld` is set). Fail fast with a clear message.
4. Resolve each `RigSpec`'s ordered `CalibrationSourceSpec` list into a canonical
   `RigCalibration` (see Calibration); run `validate(RigCalibration, provider)`.
5. Open every `SensorSpec` via the device interface layer (`ICamera`/`IImu` — out of scope
   here), requesting a stream in the first format from `supported_pixel_formats()` the device
   can produce natively. Instantiate a **`SensorAdapter`** per source, which bridges device
   callbacks into `ISensorSink` and converts format only if no native match was available.
6. Wire every `SensorAdapter` into a **`SensorMultiplexer`** (single time-ordered stream on
   one fusion timeline; see Timing & synchronization C) whose output is the provider, each
   sample tagged with its `sensor_id`. Then `provider.initialize(config, calibrations)`.
7. Per tick: `mux.update()` to drain the reorder buffer into the provider, then
   `get_latest_poses()` (or `predict_to(now)` when `profile == RealtimeControl`) → for each
   `RigPose`, look up its `OutputSpec`, apply the `output_frame` transform, and push
   `Se3TrackerPose` on that rig's `collection_id` via a **per-output `SchemaPusher`**
   (N pushers in one process; each is an independent `se3_tracker` device on the consumer).

Validation against **clock domains** (reject an unsynced multi-device rig unless
software-offset fallback is explicitly allowed) happens in step 5, since the clock domain is
only known once the device layer reports it.

```cpp
// Merges N per-adapter input streams into one time-ordered ISensorSink output.
// Each SensorAdapter writes into the slot returned by add_source(); the mux k-way merges
// by sample_time_device_ns through a bounded reorder buffer and forwards to the output.
// This is the ONE place reordering happens — adapters push as samples arrive, and the
// provider receives an already-coherent monotonic stream.
struct MultiplexerConfig {
    int64_t reorder_window_ns;   // buffer depth: max out-of-order skew absorbed
    int64_t skew_alert_ns;       // inter-sensor skew above this logs a warning
};

class SensorMultiplexer {
public:
    explicit SensorMultiplexer(const MultiplexerConfig&);
    ISensorSink* add_source();            // call once per SensorAdapter; returns its input slot
    void set_output(ISensorSink* sink);   // the IPoseProvider
    // Releases every buffered sample older than (newest_seen - reorder_window_ns) to the
    // output, in device-time order. Called each plugin tick so a quiet source cannot stall
    // the stream indefinitely.
    void update();
};
```

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

```cpp
class ICalibrationSource {
public:
    virtual ~ICalibrationSource() = default;
    // Returns a partial RigCalibration; unset fields are left at defaults.
    // The orchestrator merges an ordered list, later sources overriding earlier ones.
    virtual RigCalibration load(const RigSpec&) const = 0;
};

// Reads from the device interface at open time (ICamera.open() / IImu.open()).
class DeviceCalibrationSource : public ICalibrationSource { /* out of scope: device layer */ };

// Parses a Kalibr yaml or native calibration file.
class FileCalibrationSource : public ICalibrationSource { /* path from CalibrationSourceSpec */ };
```

- `DeviceCalibrationSource` — from the device interface self-report (intrinsics/extrinsics
  from `ICamera.open()`, IMU calibration from `IImu.open()`). How the device layer exposes
  this is out of scope here.
- `FileCalibrationSource` — Kalibr yaml / native file (the only option for GMSL/Sensing).
- Mixed is the common case: device intrinsics as base, Kalibr file overriding cam↔IMU +
  IMU noise.

**Backend adapters translate canonical → native** (cuVSLAM rig struct / ORB-SLAM3 yaml / VINS
yaml), owning distortion-model and axis-convention mapping. The interface only ever carries
`RigCalibration`.

**Validation at startup** (`validate(const RigCalibration&, const IPoseProvider&)`): queries
the provider's capability methods against the resolved calibration — distortion model
supported by the backend? extrinsic star graph complete (every camera has `T_body_cam`)?
IMU noise present when the backend needs VIO? `rolling_readout_ns` set when
`shutter == Rolling`? units/convention sane? Fail fast. Config-level checks that need no
calibration (rig/output id consistency, `Session` output frame implies `T_session_slamworld`
is set, backend supports the requested rig count) run earlier — see Orchestration step 3.
Online refinement (VIO backends refine cam↔IMU/time-offset live) is a backend concern; the
canonical model supplies the *prior*.

## Output — reuse `se3_tracker` (recommended)

The plugin's per-tick loop maps each `RigPose` onto the existing SE3 push path — one
`SchemaPusher` per output rig (`controller_se3_tracker_plugin.cpp:67-105` is the single-rig
template):

```cpp
// Composes the SLAM-world pose into the configured output frame.
// SlamWorld -> pass through; Session -> T_session_slamworld * pose.
core::Pose apply_output_frame(const core::OutputSpec& spec, const core::Pose& slam_world_pose) {
    if (spec.output_frame == core::OutputSpec::Frame::SlamWorld)
        return slam_world_pose;
    return compose(*spec.T_session_slamworld, slam_world_pose);  // validated non-null at startup
}

// Per-tick loop. `poses` is reused across ticks; get_latest_poses() clears it.
std::vector<core::RigPose> poses;
provider.get_latest_poses(poses);
for (const auto& rp : poses) {
    const core::OutputSpec& spec = output_for(rp.rig_id);
    core::SchemaPusher& pusher = pusher_for(rp.rig_id);   // per-OutputSpec collection_id
    core::Se3TrackerPoseT out;
    if (rp.pose.state == core::TrackingState::Tracking) {
        out.pose = std::make_shared<core::Pose>(apply_output_frame(spec, rp.pose.pose));
        out.is_valid = true;
    } else {
        out.pose = std::make_shared<core::Pose>(/* identity */);
        out.is_valid = false;               // tracking-lost filler; consumers freeze
    }
    // Pack -> builder.Finish() -> pusher.push_buffer(ptr, size,
    //                                rp.pose.sample_time_local_ns, rp.pose.sample_time_device_ns)
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

**A. Inter-interface delay (adapter → mux → provider).** In-process callbacks, so the
delay is small — but the hazard is **backpressure and ordering**, not latency:
- Provider `on_image`/`on_imu` are enqueue-only; SLAM runs on the provider's own worker
  thread. A blocking call into the provider would stall the whole chain and drop
  frames / back up IMU.
- The provider is guaranteed samples in **monotonic device-time order** (IMU interleaved
  with frames). That guarantee is produced by the `SensorMultiplexer`, which is the single
  owner of reordering — adapters forward as-arrived, the provider never sorts. VIO
  integrates IMU *between* consecutive frames, so out-of-order or dropped IMU corrupts the
  estimate; a sample arriving later than the mux's `reorder_window_ns` is dropped and
  counted rather than delivered out of order.

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

**Who owns alignment: the orchestrator, not the estimator.** The device layer exposes each
source's clock domain (how is a device-layer concern, out of scope here); the `SensorAdapter`
uses this to populate timestamps correctly and signals the domain to the plugin. A
**`SensorMultiplexer`** in the plugin merges the N source streams into one coherent, monotonic,
single-timeline stream (k-way merge by fusion timestamp + a small bounded reorder buffer to
absorb per-source jitter), and **monitors inter-sensor skew, flagging threshold breaches**
(VIO degrades silently otherwise). The provider stays sync-agnostic: by the time samples
reach it they are already one coherent fusion timeline satisfying the monotonic-ordering
contract in (A). Same-domain sources pass device stamps through; cross-domain sources are
rewritten to the host timeline first.

Recommendation ranking: (1) single camera with built-in IMU; (2) hardware-synced multi-cam
rig (GMSL/FSIN), IMU on the same sync domain; (3) software-offset fallback only, with skew
monitoring — not for tight VIO.

## Real-time teleop vs data collection

The primary use case is **egocentric data collection**, where the pose is recorded and
aligned offline — a 50 ms pipeline latency is irrelevant as long as timestamps are correct.
A second, much stricter use case is **replacing a live tracked device** (e.g. swapping a
Pico controller for an OAK-D + SLAM rig) so the pose drives a robot in real time. The wire
path is identical for both — a SLAM rig is just another `se3_tracker` device, so no consumer
work is needed either way — but the two profiles have materially different requirements.
`SlamInstanceConfig::profile` declares which one an instance serves, and startup validation
enforces it.

| Concern | `DataCollection` | `RealtimeControl` |
|---|---|---|
| Pipeline latency | Tolerated; correct timestamps suffice | Must be hidden by prediction |
| Loop closure / pose jumps | Fine (better global consistency) | Forbidden — jumps move the robot |
| Tracking loss | Gap in the recording | Robot stops mid-motion |
| Backend prediction | Optional | Required (`supports_prediction()`) |
| Drift over a session | Correctable offline | Accumulates uncorrected |

### The four constraints `RealtimeControl` adds

**1. Latency must be hidden, not just measured.** A SLAM pose is *for a past frame* — with
OAK-D + a CPU-bound backend, roughly 40–70 ms old by the time it is pushed (USB transfer,
tracking compute, tick and transport). Worse than the absolute number is the *asymmetry*:
OpenXR gives head and controller poses already extrapolated to display time, so a SLAM
channel fused with XR channels reads as lagging by the full gap plus the XR runtime's
forward prediction. Propagating timestamps (Timing B) makes this correctly *describable*;
`predict_to()` is what makes it usable. The plugin calls `predict_to(now)` instead of
`get_latest_poses()` when `profile == RealtimeControl`, and startup fails if the linked
backend reports `supports_prediction() == false`.

**2. Loop closure is forbidden.** Full-SLAM pose-graph optimisation corrects the map and the
pose *jumps*. During recording that is a feature — better global consistency. Driving a
robot, it is a discontinuity in the commanded end-effector pose. `odometry_only` is forced
true for this profile, trading drift (bounded, slow) for continuity (safety-critical). Note
this is a sharper constraint for ORB-SLAM3 than for cuVSLAM: loop closing and multi-map
merging are ORB-SLAM3's headline features, so odometry-only mode gives up most of what
distinguishes it.

**3. Tracking loss needs a policy beyond `is_valid = false`.** The `se3_tracker` schema
carries two-level validity, and the consumer contract is "freeze on invalid." For a
recording, a gap is a gap. Mid-teleop, freezing means the robot stops accepting input at an
arbitrary moment — possibly mid-grasp — and `Initializing` means there is *no* usable pose
for the first seconds of every session, unlike a controller which is tracked from the start.
The interface already carries the full `TrackingState`; what is missing is a decision about
what the teleop layer does with `Lost` vs `Relocalizing` vs `Initializing`. This is a
consumer-side policy question (see Open questions), not an interface gap.

**4. Relative drift is the failure mode, not absolute drift.** What teleop actually needs is
the hand pose *relative to the head*. On a Pico, both come from one tracking system, so the
relative geometry stays consistent even as the whole map drifts. Replacing only the
controller leaves two independent trackers with independent drift — each individually
healthy, their relative pose degrading over a session. A fixed `T_session_slamworld` cannot
correct this, since the error is time-varying. Mitigations (periodic re-alignment against a
shared observable, or tracking both head and hand as rigs in *one* collaborative instance so
they share a map frame) are out of scope for the first pass but are the reason the
collaborative topology exists.

### Practical verdict

`DataCollection` works with the design as specified. `RealtimeControl` is viable for slow,
deliberate manipulation once `predict_to()` and `odometry_only` are implemented, and is not
recommended for dynamic motion regardless — the residual latency after prediction, plus the
tracking-loss failure modes, are qualitatively worse than a dedicated XR controller. Prefer
a native XR tracked device where one is available; use a SLAM rig where none is (which is
exactly the egocentric, no-headset case the design targets).

## Files (design-level; no implementation this pass)

- `src/core/pose_provider/cpp/inc/pose_provider/sensor_sink.hpp` — `ISensorSink`,
  `ImageFrame`, `ImuSample`, `SensorId`, `RigId`, `PixelFormat`.
- `src/core/pose_provider/cpp/inc/pose_provider/pose_provider.hpp` — `IPoseProvider`
  (including `predict_to()` / `supports_prediction()`), `PoseEstimate`, `RigPose`,
  `TrackingState`, `create_pose_provider()`.
- `src/core/pose_provider/cpp/inc/pose_provider/topology.hpp` — `SensorSpec`, `RigSpec`,
  `OutputSpec`, `SlamInstanceConfig` (+ a yaml-cpp loader for `slam_topology.yaml`), and the
  config-level validation from Orchestration step 3.
- `src/core/pose_provider/cpp/.../sensor_multiplexer.*` — `SensorMultiplexer` +
  `MultiplexerConfig`: merges N `SensorAdapter` outputs into one time-ordered,
  single-fusion-timeline stream; the sole owner of reordering (bounded reorder buffer,
  late-sample drop counter, inter-sensor skew monitor).
- `src/core/pose_provider/cpp/inc/pose_provider/calibration.hpp` — canonical `RigCalibration`
  (`CameraCalibration`/`ImuCalibration`), `ICalibrationSource` (`Device`/`File`),
  `CalibrationSourceSpec`, the precedence-merge resolver (input: ordered `CalibrationSourceSpec`
  list per rig → output: `RigCalibration`), and
  `validate(const RigCalibration&, const IPoseProvider&)`. Backend adapters live with each
  backend and translate canonical → native.
- `src/core/pose_provider/backends/{cuvslam,orbslam3,vins,stub}/` — one lib each, selected
  by `POSE_PROVIDER_BACKEND`; built into per-backend plugin binaries for heterogeneity.
  cuVSLAM links a vendored prebuilt `.so` (+ CUDA); others FetchContent-from-source.
- `src/plugins/slam_pose_provider/` — producer plugin: owns `SensorAdapter` per source
  (bridges device layer → `ISensorSink`), feeds `SensorMultiplexer` → one
  `create_pose_provider()` → **per-rig** `SchemaPusher`s on `se3_tracker`. One process per
  `SlamInstanceConfig`. Device interface (`ICamera`/`IImu`) backends are out of scope here.

Reused as-is: `core::SchemaPusher` (`src/core/pusherio/`), `core::Se3Tracker` +
`Se3TrackerPose` schema, `os_monotonic_now_ns()`, the `controller_se3_tracker` plugin
skeleton, `DeviceIOSession`/`OpenXRSession`.

## Verification (of the design, before/after implementation)

- **Design review acceptance:** a new backend can be added by implementing `IPoseProvider`
  + adding one CMake branch + one factory `#elif`, touching neither the device interface
  layer, the plugin, nor any consumer. A new camera vendor = one `SensorAdapter` impl in
  the plugin (device interface design concern). All four topologies
  (per-device, rigid rig, collaborative, hybrid) express as `SlamInstanceConfig` data with
  no interface change. Confirm `PoseEstimate` → `Se3TrackerPose` mapping is lossless for the
  fields `se3_tracker` carries. Confirm a `RealtimeControl` instance cannot be configured
  onto a backend without prediction, and that `odometry_only` is forced for that profile.
- **Once implemented (out of scope now):** build with `POSE_PROVIDER_BACKEND=stub`, run the
  plugin, read the pose back with `examples/schemaio/se3_printer.cpp` on the chosen
  `collection_id` — the same reader used for `controller_se3_tracker` — to confirm the pose
  and `is_valid` gating flow end-to-end without any consumer changes.

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
  ├─ MonoLeft/Right raw queue  ───────────── on_image() ──▶ ISensorSink ──▶ IPoseProvider
  ├─ MonoLeft/Right H.264 queue ──────────▶ RawDataWriter (.h264 sidecar, hardware enc)
  ├─ IMU queue (dai::node::IMU, 200 Hz) ─┬─ on_imu()  ──▶ ISensorSink ──▶ IPoseProvider
  │                                       └─ MCAP imu channel
  └─ readCalibration() ───────────────────────────────────▶ RigCalibration

  IPoseProvider ──▶ SchemaPusher ──▶ se3_tracker MCAP channel  (pose recording)
```

Both recording and SLAM draw from the same DepthAI pipeline. Video is hardware-encoded
to an H.264 sidecar (same as the existing `OakCamera`/`RawDataWriter` pattern); raw frames
go to SLAM only. IMU is delivered to both SLAM and an MCAP channel. Pose is recorded via
the existing `se3_tracker` path, timestamped with `sample_time_device_ns` of the source
frame — so sensor data, IMU, and pose are all correlated on the device clock in the final
MCAP episode.

Key differences from the existing `OakCamera`:

- **Two queues per camera stream.** The raw `requestOutput` queue feeds SLAM (`GRAY8` or
  `NV12`); a second queue through `dai::node::VideoEncoder` feeds `RawDataWriter` for the
  H.264 sidecar. Same device timestamps on both (same `dai::ImgFrame` source node).
- **IMU node added.** Adds `dai::node::IMU` to the pipeline
  (`ACCELEROMETER_RAW` + `GYROSCOPE_RAW` at 200 Hz, `setBatchReportThreshold(1)` for
  per-sample delivery). `update()` polls `m_imu_queue`, calls `m_slam_sink->on_imu()` and
  writes to the MCAP IMU channel.
- **Calibration from device.** Calls `m_device->readCalibration()` at construction and
  translates `dai::CalibrationHandler` intrinsics + `getCameraExtrinsics()` into a
  `core::RigCalibration` — the device self-report source in the calibration merge.
- **Timestamps.** Both `getTimestampDevice()` (→ `sample_time_device_ns`) and
  `getTimestamp()` (→ `sample_time_local_ns`) already exist on `dai::ImgFrame` and
  `dai::IMUReport`; the same extraction pattern as `oak_camera.cpp:155-166` applies.

```cpp
// Minimal interface for writing ImuSamples to an MCAP channel; impl lives in the plugin.
class IImuMcapWriter {
public:
    virtual ~IImuMcapWriter() = default;
    virtual void write(const ImuSample&) = 0;
};
```

```cpp
// src/plugins/oak/core/oak_sensor_adapter.hpp
class OakSensorAdapter {
public:
    // Takes SensorSpecs directly and owns the "left"/"right"/"imu" -> core::StreamType
    // mapping, so the vendor enum never appears in slam_topology.yaml. Throws on an
    // unrecognized stream name. `format` is the orchestrator's chosen entry from the
    // backend's supported_pixel_formats().
    OakSensorAdapter(const OakConfig& config,
                     const std::vector<core::SensorSpec>& sensors,
                     core::PixelFormat format,
                     core::ISensorSink* slam_sink,         // SLAM consumer
                     core::IImuMcapWriter* imu_writer);    // IMU recording (nullable)

    core::RigCalibration get_calibration() const;
    void update();   // poll all queues: raw+H.264 frames, IMU -> sinks

private:
    std::shared_ptr<dai::Device> m_device;
    std::unique_ptr<dai::Pipeline> m_pipeline;
    std::map<core::StreamType, std::shared_ptr<dai::MessageQueue>> m_raw_queues;   // SLAM
    std::map<core::StreamType, std::shared_ptr<dai::MessageQueue>> m_h264_queues;  // recording
    std::shared_ptr<dai::MessageQueue> m_imu_queue;
    core::ISensorSink* m_slam_sink;
    core::IImuMcapWriter* m_imu_writer;   // null = don't record IMU
    core::RigCalibration m_calibration;
    std::map<core::StreamType, std::unique_ptr<RawDataWriter>> m_video_writers;
};
```

The `slam_pose_provider` plugin in Phase 1 is a simplified variant of the full
orchestration: one `OakSensorAdapter` (no `SensorMultiplexer` needed for single-device),
feeds one `IPoseProvider`, one `SchemaPusher` per output rig. Recording is opt-in:
`imu_writer` is null when recording is disabled; `m_h264_queues` and `m_video_writers`
are omitted from the pipeline when video recording is off.

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
class FanOutSensorSink : public ISensorSink {
public:
    void add_sink(ISensorSink* sink);
    void on_image(const ImageFrame&) override;  // forwards to all registered sinks
    void on_imu(const ImuSample&) override;
};
```

## Open questions to resolve at implementation time

- OAK IMU + intrinsics extraction from DepthAI (absent today) — spike needed. Confirm the
  IMU node's timestamps come from the same `getTimestampDevice()` clock domain as frames.
- **`T_session_slamworld` source:** `OutputSpec` carries the transform but doesn't say where
  it comes from. Two options: (a) a fixed extrinsic from a calibration file (known rig
  mounting); (b) an init-time alignment procedure that runs once at session start (e.g.
  levelling to gravity + a known reference point). Decide per use case; (b) means the plugin
  cannot push a valid pose until alignment completes — decide what it emits until then
  (`is_valid = false`, or `SlamWorld` frame with a documented caveat).
- **Tracking-loss policy for `RealtimeControl`:** the interface carries the full
  `TrackingState`, but `se3_tracker` only carries `is_valid`, and the consumer contract is
  "freeze on invalid." Decide what the teleop layer should do for `Initializing` (no pose
  for the first seconds of a session) vs `Lost` mid-motion vs `Relocalizing`. If freeze is
  wrong, this is the concrete trigger for the dedicated `slam_pose` device type (see Output)
  that can carry state rather than a bool.
- **Prediction horizon and quality:** `predict_to()` integrates IMU past the last processed
  frame. Pin the maximum sane horizon (beyond which prediction error exceeds the latency it
  hides) and whether prediction should be refused rather than degraded past it. Measure the
  actual end-to-end budget per backend before committing to `RealtimeControl` support.
- **Relative head↔hand drift** when a SLAM rig replaces one tracked device while the head
  stays on XR: two independent trackers drift independently and a fixed
  `T_session_slamworld` cannot correct a time-varying error. Evaluate tracking both as rigs
  in one collaborative instance (shared map frame) versus periodic re-alignment.
- **ORB-SLAM3 licensing (GPLv3)** vs IsaacTeleop's Apache-2.0. The per-backend-binary
  architecture keeps it out of the main process, but this needs a legal answer before
  ORB-SLAM3 ships as a supported backend rather than a local experiment. cuVSLAM (in-house,
  vendored prebuilt) does not have this constraint.
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
