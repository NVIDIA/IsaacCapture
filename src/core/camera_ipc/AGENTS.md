<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — `camera_ipc`

**CRITICAL (non-optional):** Before editing this package, complete the mandatory **`AGENTS.md` preflight** in [`../../../AGENTS.md`](../../../AGENTS.md) (read every applicable `AGENTS.md` on your paths, not just this file).

Read [`README.md`](README.md) first; it explains why frames cross between processes this way.

- **Linux only.** The module and its tests are added under `CMAKE_SYSTEM_NAME STREQUAL "Linux"`. Platform calls stay here and out of `camera_core`.
- **Readers cannot write.** A pool is sealed before its descriptor leaves the process, and a reader maps it read-only. Do not hand a reader a writable mapping.
- **Derive what both sides can compute.** A pool's layout follows from the slot count and slot size in the stream's description; do not send it separately.
- **Tests need no camera and no GPU**, and carry the `unit` tag.
