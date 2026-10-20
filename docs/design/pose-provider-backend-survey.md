<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Pose Provider — SLAM Backend Survey

This survey checks the backend-neutral contract in the
[Pose Provider interface design](pose-provider-interface-design.md) against ten public
SLAM/VIO libraries. It records how each library behaves where backends differ, and which
design rules those differences produced. It is not a support commitment.

## Method

Each library's source was read at the commit below; nothing was built or run. Evidence cites
`file:line` at that commit, abbreviated to the file name where a repository layout is deep.

| Library | Source | Commit |
|---|---|---|
| ORB-SLAM3 | [UZ-SLAMLab/ORB_SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) (v1.0) | `4452a3c` |
| cuVSLAM | [nvidia-isaac/cuVSLAM](https://github.com/nvidia-isaac/cuVSLAM) (17.0.0) | `2a0882c` |
| VINS-Fusion | [HKUST-Aerial-Robotics/VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) | `be55a93` |
| OpenVINS | [rpng/open_vins](https://github.com/rpng/open_vins) | `6948812` |
| Basalt | [VladyslavUsenko/basalt](https://gitlab.com/VladyslavUsenko/basalt) | `0f3b2b52` |
| Kimera-VIO | [MIT-SPARK/Kimera-VIO](https://github.com/MIT-SPARK/Kimera-VIO) | `ce8c59b7` |
| OKVIS2 | [ethz-mrl/okvis2](https://github.com/ethz-mrl/okvis2) | `a2ea0068` |
| stella_vslam | [stella-cv/stella_vslam](https://github.com/stella-cv/stella_vslam) | `e445b54` |
| RTAB-Map | [introlab/rtabmap](https://github.com/introlab/rtabmap) | `afb63bd` |
| DROID-SLAM | [princeton-vl/DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) | `2dfd39f` |

Every library was asked the same questions: inputs, live output, world frame, refined
output, behavior after losing tracking, maps, repeatable mode, process-global state,
finishing, calibration, and license.

## Summary

| Backend | IMU | Live output | Refined output | After losing tracking | Repeatable mode | License |
|---|---|---|---|---|---|---|
| ORB-SLAM3 | Optional | Every frame, camera | End of run, every frame; upstream only as a file export | New map, or reset | None | GPL-3.0 |
| cuVSLAM | Optional | Every frame: odometry and SLAM poses | Any time, every frame | Odometry restarts in a new frame | Offline tracker modes | NVIDIA Community License |
| VINS-Fusion | Optional | Every frame (half rate when threaded), IMU body | Separate loop-closure node, keyframes, any time | Not detected; a restart starts a new frame | Single-threaded mode, a different code path | GPL-3.0; its DBoW2 copy is non-commercial |
| OpenVINS | Required | Every frame, IMU body, with covariance | None | Not detected; keeps drifting | Single-threaded settings | GPL-3.0 |
| Basalt | Optional | Every frame, IMU body | Separate offline mapper, keyframes | Not detected | None documented | BSD-3-Clause |
| Kimera-VIO | Required | Keyframes only | Optional, keyframes, updated at each keyframe | Backend failure ends the run | Sequential mode with a fixed seed | BSD-2-Clause; vendored Triangle is non-commercial |
| OKVIS2 | Optional | Every frame, plus revisions of past states | Any time; end-of-run full BA, every frame, as a file | Flagged; frame kept | VIO mode only | BSD-3-Clause |
| stella_vslam | None | Every frame, rectified camera | Any time, every frame, as a file | Relocalizes in the same map; an early loss resets | None | BSD-2-Clause |
| RTAB-Map | Through VIO backends | Odometry every frame; map pose jumps at loop closures | Any time, graph nodes | Optional new map | Synchronous API; RANSAC seed is a global | BSD-3-Clause core |
| DROID-SLAM | None | Keyframes only, revised later | End of run, re-reading all input | Not detected | Synchronous mode | BSD-3-Clause |

Every backend fits the contract once its adapter handles the differences below. Two stretch
it: Kimera-VIO outputs poses only at keyframes, so per-frame output needs IMU propagation in
the adapter, and DROID-SLAM needs the whole input again at the end, so it fits only offline
re-estimate runs.

## What the survey changed in the design

| Finding | Design rule |
|---|---|
| Backends offer different live poses (odometry, loop-corrected) at different rates (every frame, keyframes) | Each adapter states which pose it publishes and at what rate; data collection uses the loop-corrected pose |
| VINS-Fusion, OpenVINS, Basalt, and DROID-SLAM report no tracking status | The adapter derives `Initializing` and `Lost`; a failure that ends tracking for good is fatal |
| Frames change in different ways: a new map, a new odometry frame, an early reset, a re-anchored map | The adapter detects every change and reports `Lost` until the backend is back in the original frame |
| Outputs use different native frames: rectified cameras, IMU bodies, a different frame for an export | Each output declares its own native frame, including any rectification rotation |
| Refined output is in memory, a file, an end-of-run step, or a separate offline tool, and may cover only keyframes | Refined output is optional, available after input ends, and may cover part of the run; a separate tool runs as part of re-estimate |
| A repeatable mode can take a different code path (VINS-Fusion) | Re-estimate uses the repeatable mode; reproduce keeps the live mode |
| VINS-Fusion, OpenVINS, and OKVIS2 can refine calibration online | The adapter states it and records the final estimates |
| Process-global state is common (ORB-SLAM3, VINS-Fusion, Kimera-VIO, and others) | One rig per capture process |
| Finishing differs: ORB-SLAM3 and stella_vslam return without waiting for optimization; VINS-Fusion never returns | The adapter states how long finishing takes, which sets the stop timeout |
| Backend databases do not keep raw input (RTAB-Map keeps no raw IMU) | The session recording, not a backend database, is the input for offline runs |
| Loop-corrected live poses jump without a frame change | The recording keeps loop-closure corrections as backend events |

## Per-library findings

### ORB-SLAM3

| Aspect | Finding |
|---|---|
| Input | Stereo, optionally with IMU batched into each `TrackStereo()` call; timestamps in double seconds (`include/System.h:110`) |
| Live output | `TrackStereo()` returns the camera pose once per frame; `GetTrackingState()` reports state. `MapChanged()` only flags a large change, and keeps its counter in a function-local static (`src/System.cc:244`, `:321`, `:490`) |
| World frame | With an IMU, the map is rotated to gravity at inertial initialization and again after 5 s and 15 s; `GetTimeFromIMUInit()` is 0 until initialization (`src/LocalMapping.cc:184-224`, `:1283`; `src/System.cc:1339`) |
| Refined output | Only `SaveTrajectoryEuRoC()`, a file written after `Shutdown()`: it uses the largest map, re-anchors to its first keyframe, skips lost and other-map frames, and writes IMU-body poses with an IMU. The per-map overload needs a `Map*` that `System` never exposes (`src/System.cc:662-777`; `include/System.h:158`, `:213`) |
| After losing tracking | Starts a new map, or resets the map if it has fewer than 10 keyframes. With an IMU, a frame gap over 1 s does the same; a timestamp regression starts a new map. A merge returns to the older map (`src/Tracking.cc:1805-1849`, `:2021-2024`; `src/LoopClosing.cc:1550`) |
| Maps | Atlas of several maps, atlas save and load, localization mode (`include/System.h:197-198`) |
| Repeatable mode | None: local mapping and loop closing always run on their own threads (`src/System.cc:197`, `:214`) |
| Process-global state | Camera intrinsics are static members of `Frame` (`include/Frame.h:204-209`) |
| Finishing | `Shutdown()` returns without waiting for the mapping threads; the wait loop is commented out (`src/System.cc:515-545`) |
| Calibration | Pinhole with k1, k2, p1, p2, optional k3, or Kannala-Brandt. It rectifies stereo itself and rewrites the IMU extrinsic for the rectified frame (`src/Settings.cc:205-219`, `:502`, `:525`) |
| License | GPL-3.0 |

The internal OAK-D baseline fork adds the `System` methods an adapter needs: the ID of the
map being tracked, map-reset and relocalization counters, per-frame poses for a chosen map,
an atlas merge, a final global bundle adjustment, and a shutdown that waits.

### cuVSLAM

| Aspect | Finding |
|---|---|
| Input | 1–32 cameras and 0–1 IMU; IMU through `RegisterImuMeasurement()` in timestamp order; timestamps in int64 ns (`libs/cuvslam/cuvslam2.h:229`, `:274`, `:649`) |
| Live output | `Tracker::Track()` returns an odometry pose with covariance, empty on failure, and a SLAM pose that jumps at loop closures; neither is revised (`cuvslam2.h:313`, `:1133`) |
| World frame | The rig frame at the first frame; gravity comes from `GetLastGravity()`, in the current rig frame (`cuvslam2.h:310`, `:674-682`) |
| Refined output | `Slam::GetAllSlamPoses()`, any time, every frame, rebuilt from keyframe-relative poses. The default `max_map_size = 300` caps the pose graph (`cuvslam2.h:810`, `:950`; `async_slam.cpp:275`) |
| After losing tracking | Odometry continues in a new frame, and SLAM cannot repair the break (`cuvslam2.h:593`; `TROUBLESHOOTING.md:322`) |
| Maps | `SaveMap()` and `LocalizeInMap()`; a successful localization replaces the current map and its frame (`cuvslam2.h:964`; `doc/load_map.md:100`) |
| Repeatable mode | The offline tracker modes run single-threaded and are reproducible (`cuvslam2.h:1062`) |
| Process-global state | Logging verbosity; the API is instance-based |
| Finishing | No end-of-run step is needed, because the history is readable at any time |
| Calibration | Pinhole, Fisheye (4), Brown (5, ordered k1, k2, k3, p1, p2), Polynomial (8, OpenCV rational order). IMU noise densities and random walks, no time-offset field (`cuvslam2.h:167-227`) |
| License | NVIDIA Community License, which limits use to NVIDIA platforms (`LICENSE`) |

### VINS-Fusion

| Aspect | Finding |
|---|---|
| Input | Mono with IMU, stereo with IMU, or stereo. IMU arrives as a separate stream, and an image waits for an IMU sample at or after its time plus the time offset; timestamps in double seconds (`estimator.h:49-51`) |
| Live output | IMU-body pose of the newest window frame; no status, no covariance; output starts after about 11 frames, and threaded mode processes every second frame (`visualization.cpp:124-170`; `estimator.cpp:114-118`) |
| World frame | Gravity from the initial accelerations, yaw zeroed, origin at the first frame; never re-aligned (`estimator.cpp:47-52`, `:345-364`) |
| Refined output | Only in the separate `loop_fusion` node: a pose graph over keyframes, readable any time through ROS or a CSV, never finalized; it stays in the VIO world (`pose_graph.cpp:481-508`, `:782-845`) |
| After losing tracking | Never detected: `failureDetection()` returns `false`. Only a manual restart resets, into a new world frame (`estimator.cpp:955-957`) |
| Maps | The core has none; `loop_fusion` saves and loads its pose graph and joins several sequences (`pose_graph_node.cpp:457-474`) |
| Repeatable mode | `multiple_thread: 0` runs synchronously but changes tracker behavior; a Ceres wall-clock cap also varies results (`estimator.cpp:187-195`, `:1122-1125`) |
| Process-global state | Configuration lives in process globals, and the estimator loop calls ROS publishers (`parameters.h:28-65`) |
| Finishing | No final optimization; the processing loop has no stop flag, so the destructor never returns in threaded mode (`estimator.cpp:20-27`) |
| Calibration | camodocal models (pinhole radial-tangential, Kannala-Brandt, MEI, Scaramuzza) on raw images; time offset and extrinsics fixed or estimated online (`parameters.cpp:116-140`) |
| License | GPL-3.0; the DBoW2 copy in `loop_fusion` is CC BY-NC-SA 3.0, which forbids commercial use (`DBoW2.h:42-50`) |

### OpenVINS

| Aspect | Finding |
|---|---|
| Input | Mono, stereo, or more cameras, always with IMU; IMU as a separate stream; timestamps in double seconds (`VioManager.h:75-81`) |
| Live output | Filter IMU state at the newest camera time, with a 6×6 covariance; the only status is `initialized()` (`ROS1Visualizer.cpp:591-624`) |
| World frame | Gravity-aligned at initialization and fixed for the run. The default static initialization needs the rig still, then moving (`StaticInitializer.cpp:121-140`) |
| Refined output | None; loop closure and mapping live in separate repositories |
| After losing tracking | Not detected; the filter continues on IMU and drifts. Some errors call `std::exit` (`VioManager.cpp:308-309`; `StateHelper.cpp:171-182`) |
| Maps | None |
| Repeatable mode | Synchronous API; repeatable with single-threaded settings (`ros1_serial_msckf.cpp:66-68`) |
| Process-global state | Each constructor sets OpenCV's thread count and random seed; the print level is static (`VioManager.cpp:66-67`) |
| Finishing | Nothing runs at the end |
| Calibration | Radial-tangential or equidistant on raw images; Kalibr noise model; time offset, intrinsics, and extrinsics optionally estimated online (`VioManagerOptions.h:233-278`) |
| License | GPL-3.0 |

### Basalt

| Aspect | Finding |
|---|---|
| Input | Stereo with IMU, or stereo only; IMU in a separate queue; timestamps in int64 ns (`optical_flow.h:54-59`; `vio_estimator.h:79-80`) |
| Live output | World-from-IMU state every processed frame; no status, no covariance; frames can be skipped (`sqrt_keypoint_vio.cpp:486-493`) |
| World frame | Gravity-aligned at the first frame and never re-aligned (`sqrt_keypoint_vio.cpp:185-220`) |
| Refined output | None in real time. The separate `basalt_mapper` optimizes keyframes offline from marginalization data saved during the run (`mapper.cpp:338-345`; `doc/VioMapping.md:50-79`) |
| After losing tracking | Not detected; no reset path |
| Maps | None |
| Repeatable mode | None documented; parallel reductions vary with the thread count (`ba_base.cpp:218`) |
| Process-global state | None in the library; the TBB scheduler is process-wide |
| Finishing | Pushing an empty item into both queues stops it, and joining blocks until then (`sqrt_keypoint_vio.cpp:263-267`) |
| Calibration | EUCM, Double Sphere, Kannala-Brandt, Unified, pinhole, and pinhole with 8 radial-tangential coefficients, on raw images; the camera time offset is not applied by the VIO (`sqrt_keypoint_vio.cpp:182-183`) |
| License | BSD-3-Clause |

### Kimera-VIO

| Aspect | Finding |
|---|---|
| Input | Stereo, mono, or RGB-D, always with IMU; IMU as a separate stream; timestamps in int64 ns (`Pipeline.h:64-85`) |
| Live output | Poses only at keyframes, 0.2–5 s apart; covariance optional; tracking status only on the frontend, behind a protected callback (`StereoImuPipeline.cpp:95-114`) |
| World frame | Gravity-aligned with static IMU initialization; the EuRoC default initializes from ground truth; never re-aligned (`VioBackend.h:143-194`) |
| Refined output | Optional loop closure with a pose-graph optimizer; the corrected keyframe trajectory updates at every keyframe through a callback or CSV (`LoopClosureDetector.cpp:198-391`) |
| After losing tracking | A smoother failure gets one retry, then ends the pipeline (`VioBackend.cpp:1407-1497`) |
| Maps | None |
| Repeatable mode | Sequential mode with a fixed random seed (`Pipeline.cpp:86-88`) |
| Process-global state | About 112 process-global flags, and statics that latch the first instance's settings (`Pipeline.cpp:18-61`) |
| Finishing | No final optimization; shutdown drains and joins |
| Calibration | Pinhole or omnidirectional cameras with none, radial-tangential, or equidistant distortion; rectifies stereo itself; IMU time shift fixed or estimated once at startup (`CameraParams.h:31-41`) |
| License | BSD-2-Clause, but the vendored Triangle library forbids commercial use without arrangement (`triangle.c:25-37`) |

### OKVIS2

| Aspect | Finding |
|---|---|
| Input | Mono, stereo, or more cameras, with IMU or vision only; IMU as a separate stream (`ViInterface.hpp:326-340`) |
| Live output | A per-frame callback with the world-from-IMU state, a tracking state (good, marginal, lost, place recognized), and revisions of past states (`ViInterface.hpp:143-174`) |
| World frame | Gravity-aligned, origin at the first frame; kept after initialization, though a clear before initialization recreates it (`ViGraph.cpp:79-118`) |
| Refined output | Revisions arrive any time; at the end, `doFinalBa()` and `writeFinalTrajectoryCsv()` give every frame as a file (`ThreadedSlam.hpp:156-159`) |
| After losing tracking | Flagged; no reset after initialization, and the frame is kept (`ThreadedSlam.cpp:627-634`) |
| Maps | Experimental map saving; loading is disabled |
| Repeatable mode | VIO mode only, because asynchronous loop closure depends on wall-clock timing (`ThreadedSlam.cpp:543-566`) |
| Process-global state | `doFinalBa()` sets a process-global flag that is never reset; a global timing singleton (`ImuError.hpp:151`) |
| Finishing | Stopping blocks while it drains frames and loop closure; the final bundle adjustment runs once |
| Calibration | Pinhole with radial-tangential (5 or 8) or equidistant distortion, on raw images; fixed image delay; optional online extrinsics |
| License | BSD-3-Clause; the bundled network weights have an unknown license |

### stella_vslam

| Aspect | Finding |
|---|---|
| Input | Mono, stereo, or RGB-D; no IMU; stereo images must be rectified; timestamps in double seconds (`system.h:138-140`) |
| Live output | World-from-camera for the rectified left camera, with a status string; the frame that goes lost still returns a pose (`system.cc:586-611`) |
| World frame | The left camera at the first initialized frame; not gravity-aligned; the root stays fixed (`initializer.cc:346-349`) |
| Refined output | A file, written any time, covering every non-lost frame since the last reset; only keyframes in memory (`trajectory_io.cc:64-81`) |
| After losing tracking | Relocalizes in the same map. A loss within 5 s of initialization, or a reset request, clears the map and starts a new world frame (`tracking_module.cc:28`, `:157-166`) |
| Maps | One map, saved and loaded as msgpack or SQLite; localization against a loaded map (`system.cc:259-307`) |
| Repeatable mode | None; build and runtime options reduce variation (`CMakeLists.txt:193-195`) |
| Process-global state | One static mutex shared by all map databases (`map_database.h:270`) |
| Finishing | Shutdown joins the mapping threads but not a loop bundle adjustment in progress (`system.cc:222-243`) |
| Calibration | Perspective (k1, k2, p1, p2, k3), fisheye, equirectangular, radial division; the caller rectifies; no IMU parameters |
| License | BSD-2-Clause; one g2o component is LGPL |

### RTAB-Map

| Aspect | Finding |
|---|---|
| Input | Mono, RGB-D, stereo, or lidar. Stereo with raw IMU only through a VIO odometry backend; IMU as a separate stream; timestamps in double seconds (`OdometryThread.cpp:181-217`) |
| Live output | Odometry every frame, empty when lost, with covariance; the map pose is a map correction times odometry and jumps at loop closures (`Odometry.h:94-102`; `Rtabmap.h:340-363`) |
| World frame | Odometry starts at identity or from the IMU orientation; the map is anchored at its oldest node, but the anchor can move with pose priors, long-term memory transfer, or `OptimizeFromGraphEnd` (`Rtabmap.cpp:5444-5470`) |
| Refined output | Any time in memory, for graph nodes only; explicit post-processing adds loop closures and global bundle adjustment (`Rtabmap.h:331`, `:503`) |
| After losing tracking | No reset by default; an optional reset countdown starts a new map in the same database, merged after a later loop closure (`Odometry.cpp:1113-1131`) |
| Maps | Multi-session SQLite database and a localization mode; the database keeps no raw IMU, only a per-node gravity orientation (`DBReader.cpp:760-770`) |
| Repeatable mode | Synchronous API, but RANSAC is seeded from the clock unless a test-only global is set (`util3d_motion_estimation.h:43-65`) |
| Process-global state | Logger, an event-manager singleton, and the RANSAC seed flag; the VINS-Fusion backend brings VINS-Fusion's globals |
| Finishing | No end-of-run optimization; closing saves to the database and clears poses in memory, so export first (`Rtabmap.cpp:472-540`) |
| Calibration | Pinhole with 4–14 radial-tangential coefficients, or equidistant; rectified input by default, or rectifies itself; IMU noise and time offset only in VIO backends |
| License | BSD-3-Clause core; optional backends carry their own licenses, several GPL |

### DROID-SLAM

| Aspect | Finding |
|---|---|
| Input | Mono, stereo, or RGB-D; no IMU; rectified images; timestamps stored as float32 (`droid.py:61-69`) |
| Live output | `track()` returns nothing; only keyframe poses exist, and the window revises and deletes them; no status (`depth_video.py:26`, `:41`) |
| World frame | The first frame; not gravity-aligned. The stereo baseline is hard-coded to 0.1, so translations need rescaling (`droid_kernels.cu:227-230`) |
| Refined output | `terminate()` runs global bundle adjustment, re-reads the entire input, and returns every frame (`droid.py:71-85`; `trajectory_filler.py:42-110`) |
| After losing tracking | No detection, reset, or relocalization |
| Maps | None; a fixed GPU buffer of 512 keyframes |
| Repeatable mode | The synchronous class; the asynchronous variant is documented as non-deterministic |
| Process-global state | The network is pinned to the first GPU, and PyTorch settings are process-wide |
| Finishing | `terminate()` blocks through global bundle adjustment and the fill pass |
| Calibration | Pinhole fx, fy, cx, cy only; the caller undistorts and rectifies |
| License | BSD-3-Clause; the network weights' license is unknown |

## Limits

The survey reads source code only; behavior can differ with configuration and between
releases. Statements about determinism and process-global state come from the code paths
inspected, not from running two instances.
