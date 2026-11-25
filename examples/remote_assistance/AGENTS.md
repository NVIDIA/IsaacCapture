<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Remote Assistance transport example

- Keep transport implementations in their own packages; example code owns camera
  capture, encoder feedback, and operator setup. Never import the example from a plugin.
- Keep public installation and help text driven by available plugins. Do not add
  dependencies on optional SDKs to the core or assume multiple plugins are installed.
- Validate shipped camera presets with the same source/encoder validation used at
  runtime; control names and units must match that implementation.
- Keep generated credentials and experiment artifacts outside the source tree.
- A bundle transfer supplies configuration, not code or installed packages; test
  the exact built wheels on both architectures before calling a deployment ready.
