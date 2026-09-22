<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sharpa Avatar plugin — agent notes

**CRITICAL:** Before editing this tree, read this file and the repo root
[`AGENTS.md`](../../../AGENTS.md).

- **Keep the Avatar SDK external.** `install_avatar_sdk.sh` installs the
  official package under `/opt/avatar-sdk`; the build consumes the selected
  SDK root in place (`-DAVATAR_SDK_ROOT` > `$AVATAR_SDK_ROOT` >
  `/opt/avatar-sdk`). Do not copy, patch, or package SDK files.
- Install the pinned production `avatar-sdk` package from the signed production
  channel, and never substitute `-dev` or `-beta`. Do not parse or duplicate
  `production_version`: `install.sh` and CMake call
  `install_avatar_sdk.sh --check`, which holds the default root to the full pin
  (layout, `BUILD_TYPE=Production`, dpkg package version) and layout-checks a
  custom root, where a non-production build only warns. Do not pin `VERSION=`
  from `share/Version`; the dpkg package version is authoritative.
- Linux only. `BUILD_PLUGIN_SHARPA_AVATAR=ON` with no usable SDK under the
  selected root skips this plugin; it must not `FATAL_ERROR` the rest of the
  tree. The SDK check runs only after that layout is present.
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
  plugin internals; convert only at an existing external API boundary.
- Adapt this plugin to shared APIs; do not refactor other plugins or shared
  modules solely to change this plugin's internal representation.
- Keep shared enum iteration tables in one scope; a public `kDeviceSides` or
  category table must not have a same-named private duplicate.
- Use `plugin_utils::WristPoseSource` for world-space wrist placement. Do not
  copy the Manus/XDev/controller fallback implementation into this plugin.
  When that query is invalid (no headset optical pose and no controller),
  still inject HUMAN at identity with VALID-only flags; do not skip
  `HandInjector::push`. HandsSource/`live_view` otherwise stay empty.
- OpenXR setup is required for publication; initialization errors must abort
  construction instead of leaving an Avatar-only idle plugin.
- Keep initialization non-blocking; `update()` owns retries for absent gloves.
- Resolve HUMAN-to-OpenXR landmarks from the in-repo name snapshot copied from
  the pinned production SDK; do not read joint names from `sdk_config.json` at
  runtime, hard-code SDK array indices, or fill unsupported OpenXR slots with
  neighbouring poses. Refresh the snapshots (HUMAN in core, RAW/ROBOT in the
  example) together with the package pin.
- When reconciling the maintained feature branch, use it as the baseline and
  retain local divergence only for an explicit API or correctness requirement.
- Keep this plugin tree C++-only (`core/` + `app/`). The TeleopSession sample
  lives at `examples/sharpa_avatar/` (`live_view.py`) and runs as
  `python -m isaacteleop_examples.sharpa_avatar`. Do not put Python samples
  under `src/plugins/sharpa_avatar/`, and do not add SDK-only diagnostic
  printers that bypass Isaac Teleop.
