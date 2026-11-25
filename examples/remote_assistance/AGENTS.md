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
- In README setup snippets, set `IC_DIR` explicitly and use the requested simple
  `cd $IC_DIR` form instead of embedding required-variable checks in `cd`.
- Run setup helpers with the explicitly created virtual-environment interpreter;
  do not assume the client’s system `python3` meets the package minimum version.
- Order one-time preparation as shared installation, host prerequisites, then
  client prerequisites, with explicit instructions for both checkouts.
- Document core/plugin and example installation as separate required steps;
  verify the example entrypoint as well as plugin discovery before bundle setup.
- Describe camera selection according to the public CLI: bundle generation
  requires a template file and has no interactive camera-selection prompt.
- Keep recurring bundle transfer and signaling checks in Host Site Setup; put
  firewall configuration in its own section, with host commands before client commands.
- Reuse the earlier IP selections in Firewall Setup instead of repeating export
  commands; remind readers that each site’s terminal needs those variables set.
- Mark native receive builds as required for CUDA profiles and show the build
  command beside Python installation; optional to the package does not mean optional to a selected profile.
- Keep eye identity at the capture mapping; verify physical lenses before XR use
  and preserve that order through IPC and both display modes.
- Smoke-test documented module commands for actual CLI output; importing an
  implementation module without an entrypoint can exit successfully doing nothing.
- Make new-terminal process examples self-contained: set their paths, change
  directory, source credentials, and clearly identify the physical site.
