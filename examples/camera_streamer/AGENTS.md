<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Standalone camera streamer

- Preserve the standalone capture → AVS1/TCP → NVDEC → CUDA IPC boundary; these tools must not import robot control packages or the Isaac Capture runtime.
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

- ZED USB capture uses V4L2 independently of Argus. Preserve atomic stereo pairs, bounded capture queues, and the existing AVS1 layout. Do not label USB timestamps as Tegra SOF; absent hardware timing stays unavailable.
- Keep ZED development in a named repository branch; standalone sender directories are deployment exports. Record the upstream base when creating an export.
- Multi-camera ZED sending is one bundle per camera, sent when that camera's frame is encoded. Do not pair or tile frames across cameras: USB cameras are unsynchronized, so waiting adds up to a frame period.
- Every ZED bundle describes all enabled cameras and marks the others absent; receivers rely on that to see a camera that has not delivered. One camera failing must not end the session for the rest.
- ZED stream names come from the camera's position in the sender config, carried in the AVS1 sample device index. Keep that position stable when a camera is disabled.
- Identify ZED units by the HID serial, not `/dev/v4l/by-id`: the video interface serial is shared across units of a model.
- A receiver config without a `streams` list discovers its layout from the wire. Keep discovery and configured streams as the two modes of one code path; do not add a per-setup receiver profile that restates a sender config.
- Do not free a replaced stream's ring while the receiver runs: consumers may still hold its CUDA IPC handles.
- After the handshake, only the IPC broadcast thread writes to consumer sockets, so a `streams` update always precedes the first frame of a new stream. Send layout changes through `streams_changed()`, not directly.
- The viewer draws a quiet stream as a placeholder, never its last image: a frozen frame reads as live video.
- Unix socket paths are limited to about 100 characters; tests take the IPC socket path separately from their output directory.
