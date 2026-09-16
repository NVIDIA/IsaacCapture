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
- Keep the dependency installer on the production `avatar-sdk` package and its
  reviewed version; never substitute the `-dev` or `-beta` package channels.
- Keep installer responsibilities separate: SDK/APT in
  `install_avatar_sdk.sh`, host device permissions in
  `install_udev_rules.sh`, and plugin build/install orchestration in
  `install.sh`.
- Do not mutate loader environment variables or SDK configuration at runtime.
  Runtime paths come from the selected SDK root and its `sdk_config.json`.
- **Sample lives in `tools/`, not `examples/`.** Path is
  `src/plugins/sharpa_avatar/tools/sharpa_avatar_sample.py` (ticket #9). Do not
  put it back under `examples/sharpa_avatar`.
