<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Standalone Sensing/Argus streamer

- Keep this example scoped to Sensing/Argus capture, AVS1/TCP, NVDEC and CUDA IPC.
  Do not restore unrelated camera wrappers, USB senders, RTP/recording paths or
  rig-specific deployment assets without an explicit scope change.
- Preserve original hashes and license notices for retained imports. Public review is
  owner-approved; preserve unresolved license metadata for maintainer review.
- Keep the native sender and standalone receiver independent of robot control
  software and the Isaac Capture runtime. Use the isaaccapture logger tree without importing it.
- Document Install, Preview, Send, Receive and Viewer stages with explicit
  commands; launch.py is optional. Local Argus preview is a separate application
  and must release the sensor before network capture starts.
- Use HOST_IP/ROBOT_IP discovered for the installation, never developer addresses.
  Explain source transfer before referencing robot deployment directories.
- Select sensor modes from --list-modes and match PWM to capture rate. Distinguish
  validated profiles from candidates; configuration validation is not a live test.
- Deploy directory renames with documentation/launchers and rebuild stale CMake
  caches. Do not transfer virtual environments, builds or caches between machines.
- Receiver tests must establish the standalone import path. Check runtime imports
  and native/receiver tests after trimming dependencies, not just formatting.
- Preserve encoded frame dependencies across disconnects. CUDA IPC handles stay
  local to the receiving machine; they are not network transport handles.
- Preview loops must handle window-close events before rendering another frame.
- CuPy pixel diagnostics need matching NVRTC; exclude first-use JIT from timing.
- Normalize final newlines and imported executable permissions before staging.
