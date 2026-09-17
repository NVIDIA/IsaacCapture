<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sharpa Avatar plugin — agent notes

**CRITICAL:** Before editing this tree, read this file and the repo root
[`AGENTS.md`](../../../AGENTS.md).

- **Keep the Avatar SDK external.** `install_avatar_sdk.sh` installs the
  official package under `/opt/avatar-sdk`; CMake consumes that fixed host
  layout directly. Do not copy, patch, or package SDK files, and do not
  advertise extracted SDK trees as supported.
- Install the pinned production `avatar-sdk` package version from the signed
  production channel, and never substitute `-dev` or `-beta`. `install.sh`
  and CMake must refuse any other installed package rather than warn and
  continue. Read `production_version` from `install_avatar_sdk.sh`; do not
  duplicate that string. Pin the dpkg package version and
  `BUILD_TYPE=Production`, not the `VERSION=` field in `share/Version`.
- Linux only. `BUILD_PLUGIN_SHARPA_AVATAR=ON` with no usable SDK under
  `/opt/avatar-sdk` skips this plugin; it must not `FATAL_ERROR` the rest of
  the tree. Pin checks run only after that layout is present.
- APT signing keys: require the pinned fingerprints to be present. Do not
  demand exact set equality (extra fingerprints from key rotation are allowed).
- Keep installer responsibilities separate: SDK/APT in
  `install_avatar_sdk.sh`, host device permissions in
  `install_udev_rules.sh`, and plugin build/install orchestration in
  `install.sh`.
- Installers must not delete files derived from a configured install prefix.
- Do not mutate loader environment variables or SDK configuration at runtime.
  Runtime paths come from the selected SDK root and its `sdk_config.json`.
- Carry Avatar SDK enums such as `DeviceSide` and `DeviceDataCategory` through
  plugin internals; do not encode sides or categories as strings or booleans.
- Keep shared enum iteration tables in one scope; a public `kDeviceSides` or
  category table must not have a same-named private duplicate.
- Use `plugin_utils::WristPoseSource` for world-space wrist placement. Do not
  copy the Manus/XDev/controller fallback implementation into this plugin.
- OpenXR setup is required for publication; initialization errors must abort
  construction instead of leaving an Avatar-only idle plugin.
- Keep initialization non-blocking; `update()` owns retries for absent gloves.
- Resolve HUMAN-to-OpenXR landmarks from `human_joint_names`; do not hard-code
  SDK array indices or fill unsupported OpenXR slots with neighbouring poses.
- When reconciling the maintained feature branch, use it as the baseline and
  retain local divergence only for an explicit API or correctness requirement.
- **Sample lives in `tools/`, not `examples/`.** Path is
  `src/plugins/sharpa_avatar/tools/sharpa_avatar_sample.py` (ticket #9). Do not
  put it back under `examples/sharpa_avatar`.
