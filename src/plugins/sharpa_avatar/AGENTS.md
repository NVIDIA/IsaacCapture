<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sharpa Avatar plugin — agent notes

**CRITICAL:** Before editing this tree, read this file and the repo root
[`AGENTS.md`](../../../AGENTS.md).

- **SDK is a staged package, not a git artifact.** `install.sh` /
  `vendor_avatar_sdk.sh` copy headers + `.so` into gitignored
  `vendor/avatar-sdk/`. Do not commit `.deb` / `.so` / `vendor/avatar-sdk/`.
- **Do not tell reviewers they must `dpkg -i` to `/opt` first.** Last delivery
  (`IsaacTeleop-SharpaPlugin`) packed `avatar-sdk/` + prebuilt into a customer
  tarball (`package_avatar_plugin_delivery.sh`); customers ran
  `install_prebuilt.sh` and never installed the SDK system-wide. `/opt` is only
  the packager default. `AVATAR_SDK_ROOT` may be an extracted tree.
- There is no Wuji-style `curl` in this plugin. Do not add one without a real
  release URL, and keep the plugin under `src/plugins`.
- **Sample lives in `tools/`, not `examples/`.** Path is
  `src/plugins/sharpa_avatar/tools/sharpa_avatar_sample.py` (ticket #9). Do not
  put it back under `examples/sharpa_avatar`.
- Generate external delivery manifests from the packaged file list. Exclude
  internal `AGENTS.md` files from both the package and its manifest.
