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
| `engines/stub/` | `pose_provider::pose_engine_stub` | Adapter of the full contract with no engine library behind it |

## Build

The `BUILD_PLUGIN_POSE_PROVIDER` option is off by default and needs `BUILD_PLUGINS`:

```bash
cmake -B build -DBUILD_PLUGIN_POSE_PROVIDER=ON
cmake --build build --target pose_provider_tests
ctest --test-dir build -L pose_provider
```

To use an engine adapter instead of the stub, also pass
`-DPOSE_ENGINE_<NAME>=ON -D<NAME>_ROOT=<engine library directory>`.

`POSE_ENGINE_TARGET` names the adapter that executables link: the one enabled adapter, or
the stub. `pose_provider_tests` checks the stub against the contract, and the
`pose_engine_*` tests check adapter registration. This directory has no install rules. A pip
build with the option on still installs `pose_provider_tests`, as it does other test
executables.

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
   from its opaque `engine_settings`.
4. Start the adapter's tests from the stub's test,
   [`test_stub_contract.cpp`](../../../tests/cpp/plugins/pose_provider/test_stub_contract.cpp).
   Keep the cases that apply to the engine, and add cases for its own settings and modes.
5. Confirm that the engine library's license permits the intended distribution.
