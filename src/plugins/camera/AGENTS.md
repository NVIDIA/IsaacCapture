<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — camera plugin

**CRITICAL (non-optional):** Before editing this package, complete the mandatory **`AGENTS.md` preflight** in [`../../../AGENTS.md`](../../../AGENTS.md) (read every applicable `AGENTS.md` on your paths, not just this file).

- **Backends are chosen at build time.** One CMake option per backend, off by default; a backend that is off is not configured, so its SDK is never looked for. Do not load backends at run time and do not add a backend ABI. The steps are in [`README.md`](README.md), "Adding a backend".
- **No SDK types outside a backend's directory.** `core/` and `app/` see `CameraSource` and `camera_core` types only, and readers never link a camera SDK.
- **A backend refuses what it does not understand.** Read options through `OptionReader` and call `finish()`, so a misspelt option is an error.
- **A camera drops a frame rather than wait.** When `acquire` has no free slot, abandon what was taken for that frame set and carry on; nothing on the capture path blocks on readers.
- **One plugin process hosts every backend.** A second instance on another endpoint is a configuration, not a different design.
