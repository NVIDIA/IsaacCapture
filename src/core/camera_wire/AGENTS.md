<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — `camera_wire`

**CRITICAL (non-optional):** Before editing this package, complete the mandatory **`AGENTS.md` preflight** in [`../../../AGENTS.md`](../../../AGENTS.md) (read every applicable `AGENTS.md` on your paths, not just this file).

Read [`README.md`](README.md) before changing `fbs/camera_wire.fbs`: a reader and a plugin from different builds must keep understanding each other.

- **The schema only grows.** Append fields with fresh ids, optional; append union members; tables, never structs. Do not reuse or renumber.
- **FlatBuffers stays inside this module.** Its users see `camera_core` types; only the codec and its tests include the generated header.
- **Decoded bytes are untrusted.** Verify, then bound every count against the fixed arrays it fills, and check for a body behind a body type.
- **A new enum value needs both sides.** The core enum and the schema enum carry the same numbers (pinned by `static_assert`), and the decoder must accept the value before a plugin sends it.
- **The frame path does not allocate.** Keep `encode(const FrameEvent&)` free of containers that do.
