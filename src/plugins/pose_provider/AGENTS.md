<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — Pose Provider plugin

These notes also cover the plugin's tests in `tests/cpp/plugins/pose_provider/`.

- Define helper types and functions at namespace scope, in the source file's anonymous namespace, not inside a
  function. Keep a lambda only for a body of a line or two; anything longer is a named function.
- Name per-stream types and helpers after streams, as the camera input interface (`StreamDescription`) and the engine
  API (`StreamSetup`, `StreamId`) do. "Eye" names only a stream's left or right role in a stereo pair.
- Write US English (gray, signaled, behavior, -ize).
