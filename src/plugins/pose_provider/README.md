<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Pose Provider

Pose Provider turns stereo camera images, and optionally IMU samples, into a 6-DoF pose
stream through an interchangeable pose estimation engine.

- **Engine:** the pose estimation software, for example a SLAM or VIO system.
- **Engine library:** the library built and linked for an engine. Its license governs
  distribution.
- **Engine adapter:** the `IPoseEngine` implementation for one engine library.
  `make_pose_engine()` creates it.

| Path | Target | Contents |
| --- | --- | --- |
| `engine_api/` | `pose_provider::pose_engine_api` | Header-only engine interface: `IPoseEngine`, its records and `ResolvedPoseSetup`; standard library only |
| `camera_input/` | `pose_provider::pose_camera_input` | Header-only camera input interface: `ICameraInput`, its events and `StereoFrame`, for one stereo camera source read without blocking |
| `undistort/` | `pose_provider::pose_undistort` | Undistortion of one camera image to a distortion-free pinhole of the same size and optical frame: the widest valid output calibration, its map and bilinear `Gray8` resampling |
| `engines/stub/` | `pose_provider::pose_engine_stub` | Adapter of the full contract with no engine library behind it |
| `core/` | `pose_provider::pose_provider_core` | `PoseProvider`: one run's lifecycle, camera input, setup resolution, input preparation, status and failure handling around one engine adapter; links no adapter |
| `app/` | `pose_provider_plugin` | The plugin executable; links the adapter that `POSE_ENGINE_TARGET` names. Its manifest is [`plugin.yaml`](plugin.yaml) |

## Build

The `BUILD_PLUGIN_POSE_PROVIDER` option is off by default and needs `BUILD_PLUGINS`:

```bash
cmake -B build -DBUILD_PLUGIN_POSE_PROVIDER=ON
cmake --build build --target pose_provider_plugin pose_provider_tests
ctest --test-dir build -L pose_provider
```

To use an engine adapter instead of the stub, also pass
`-DPOSE_ENGINE_<NAME>=ON -D<NAME>_ROOT=<engine library directory>`.

`POSE_ENGINE_TARGET` names the adapter that `pose_provider_plugin` links: the one enabled
adapter, or the stub. `pose_provider_tests` checks the stub against the engine contract, a
scripted camera input against the camera input contract and the undistortion on synthetic
lenses, and runs `PoseProvider` on a scripted camera input with the stub and with a scripted
engine; the `pose_engine_*` tests check adapter registration. The build places
`pose_provider_plugin` beside a copy of `plugin.yaml` in `build/src/plugins/pose_provider/`,
so `build/src/plugins` works as a `PluginManager` search path. No target here has install
rules, and a pip build with `BUILD_PLUGIN_POSE_PROVIDER` on stops at configuration.

## Run

`pose_provider_plugin` runs the linked engine adapter through one run: Preparing, Ready,
Running, Draining and Finished. It issues its own start request at launch, so the run starts
as soon as the engine is ready; an application cannot start it separately.

| Option | Meaning |
| --- | --- |
| `--startup-timeout=<seconds>` | Time allowed from launch for the camera input to become ready and the run to start; engine initialization is not interrupted |
| `--drain-timeout=<seconds>` | Time allowed from a finish request until the camera input has ended and the engine has processed its accepted input |
| `--plugin-root-id=<id>` | Set by the plugin launcher |

`--help` prints the defaults. SIGINT or SIGTERM finishes the run: a running engine drains,
and a run that has not started ends at once. The exit status is 0 when the run finished, 1
when it failed (the log gives the reason) and 2 for a usage error.

After its SIGINT, `PluginManager` waits `shutdown_timeout_seconds` from
[`plugin.yaml`](plugin.yaml) and then sends SIGKILL. Keep that value above the longest
finish: `--drain-timeout` plus the engine's stop time and the camera input's release, plus
the rest of engine initialization and start if a stop can arrive while the engine
initializes. When `PluginManager` stops the plugin itself, it reports the plugin as stopped
and drops its exit status; the plugin's last log line still gives the outcome.

The plugin executable creates no camera input, so it waits in Preparing until it is stopped
or the startup timeout fails it.

## Write an adapter

1. Copy `engines/stub/` to `engines/<name>/`. It defines `make_pose_engine()`. Rename its
   targets, and create them only when the adapter's option is on.
2. Register the adapter in its `CMakeLists.txt`. `DISTRIBUTION` is `redistributable` when
   the adapter and its engine library may be installed and packaged, and
   `user-built-only` when they are built only locally: such an adapter adds no install
   rules, and a pip build with it fails. With
   [`PoseEngine.cmake`](../../../cmake/PoseEngine.cmake), paths are relative to
   `<NAME>_ROOT`:

   ```cmake
   pose_engine_add(FOO
       TARGET pose_provider::pose_engine_foo
       DISTRIBUTION user-built-only
       SCRIPT build_foo.sh
       INCLUDE_DIRS include
       LIBRARIES lib/libfoo.so
   )
   if(POSE_ENGINE_FOO)
       add_library(pose_engine_foo STATIC foo_engine.cpp)
       target_include_directories(pose_engine_foo PRIVATE ${POSE_ENGINE_FOO_INCLUDE_DIRS})
       target_link_libraries(pose_engine_foo
           PUBLIC pose_provider::pose_engine_api
           PRIVATE ${POSE_ENGINE_FOO_LIBRARIES})
       add_library(pose_provider::pose_engine_foo ALIAS pose_engine_foo)
   endif()
   ```

   Without the helper, register the target yourself so that the one-adapter check sees it:

   ```cmake
   option(POSE_ENGINE_FOO "Build the FOO engine adapter" OFF)
   set(FOO_ROOT "" CACHE PATH "Directory of the FOO engine library")
   if(POSE_ENGINE_FOO)
       add_library(pose_engine_foo STATIC foo_engine.cpp)
       # Find and link the engine library under ${FOO_ROOT} here.
       add_library(pose_provider::pose_engine_foo ALIAS pose_engine_foo)
       pose_engine_register(pose_provider::pose_engine_foo DISTRIBUTION user-built-only)
   endif()
   ```

   In this directory's `CMakeLists.txt`, add the adapter directory before
   `pose_engine_select()`.
3. Implement the obligations documented on each declaration in the
   [`engine_api` headers](engine_api/inc/pose_engine_api/), starting with
   [`pose_engine.hpp`](engine_api/inc/pose_engine_api/pose_engine.hpp).
   Geometry and calibration come from `ResolvedPoseSetup`; engine-specific settings come
   from its opaque `engine_settings`. Prepared image storage is a fixed set
   (`ProviderStatus::prepared_image_groups`): release each hold after the image's last
   read, and copy any image kept longer. While the engine holds every group, the provider
   drops new frames and counts them in `frames_dropped`.
4. Start the adapter's tests from the stub's test,
   [`test_stub_contract.cpp`](../../../tests/cpp/plugins/pose_provider/test_stub_contract.cpp).
   Keep the cases that apply to the engine, and add cases for its own settings and modes.
5. Confirm that the engine library's license permits the intended distribution.
