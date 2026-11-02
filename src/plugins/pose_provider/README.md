<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Pose Provider

Pose Provider turns stereo camera images, and optionally IMU samples, into a 6-DoF pose
stream through an interchangeable pose estimation engine. An engine adapter implements
`IPoseEngine` on top of one engine library.

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

`pose_provider_tests` checks the stub against the contract. This directory has no install
rules. A pip build with the option on still installs `pose_provider_tests`, as it does other
test executables.

## Write an adapter

1. Copy `engines/stub/`. It defines `make_pose_engine()`, which creates the adapter.
2. Add a CMake option, default `OFF`, and a cache variable for the engine library's
   install prefix. In this directory's `CMakeLists.txt`, add the adapter directory behind
   that option.
3. Implement the obligations documented on each declaration in the
   [`engine_api` headers](engine_api/inc/pose_engine_api/), starting with
   [`pose_engine.hpp`](engine_api/inc/pose_engine_api/pose_engine.hpp).
   Geometry and calibration come from `ResolvedPoseSetup`; engine-specific settings come
   from its opaque `engine_settings`.
4. Start the adapter's tests from the stub's test,
   [`test_stub_contract.cpp`](../../../tests/cpp/plugins/pose_provider/test_stub_contract.cpp).
   Keep the cases that apply to the engine, and add cases for its own settings and modes.
5. Confirm that the engine library's license permits the intended distribution.
