<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Standalone camera streamer

- Preserve the standalone Argus → AVS1/TCP → NVDEC → CUDA IPC boundary; these tools must not import robot control packages or the Isaac Capture runtime.
- Deploy to a driver-ready Orin/Thor without assuming a vendor directory or existing camera services. Snapshot and restore camera-streamer artifacts and the platform service state changed for camera ownership; leave driver configuration to the BSP.
- Keep sender-only calibration assets, remaps and deployment helpers under argus_sender/. Resolve remap paths relative to the sender config so copying the sender includes its calibration assets.
- Treat the shipped calibration as an approximate viewing example. Preserve the 2x wrist lookup grid and box filtering when moving or regenerating it.
- Keep the sender build's Argus linkage consistent with the launcher's daemon ownership checks. The build wrapper selects single-process Argus; direct CMake builds also support multiprocess mode.
- Standalone Python loggers use the isaaccapture namespace without importing the runtime.
- The imported Argus sample helpers `Error.h` and `EGLGlobal.h` retain BSD-3-Clause notices and SPDX identifiers. Do not assign the example's Apache default to those files.
- Preserve upstream notices. Downloaded SDK sources remain outside this source export and retain their own license.
- Preserve the source layout when exporting standalone components; keep each component's tests inside its own folder. Reserve example-level tests/ for checks spanning components.
- Minimize differences from the source export: preserve CMake target definitions, include/header locations and flat component tests. This standalone import is an exception to the repository's CMake/include conventions; C++ formatting follows the repository's clang-format-14 configuration. Do not add build options or refactor working code solely for local verification.
- Normalize imported executable modes and final newlines before staging.
