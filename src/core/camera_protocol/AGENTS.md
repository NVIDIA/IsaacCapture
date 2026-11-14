<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — `camera_protocol`

**CRITICAL (non-optional):** Before editing this package, complete the mandatory **`AGENTS.md` preflight** in [`../../../AGENTS.md`](../../../AGENTS.md) (read every applicable `AGENTS.md` on your paths, not just this file).

The schema is [`../schema/fbs/camera_protocol.fbs`](../schema/fbs/camera_protocol.fbs). Read [`../schema/AGENTS.md`](../schema/AGENTS.md) and [`README.md`](README.md) before changing it: a reader and a plugin from different releases must keep understanding each other, and the golden check enforces it.

- **Few messages.** A connection has one purpose, stated in the handshake; do not add a message for what a handshake field can say.
- **FlatBuffers stays inside this module.** Its users see `camera_core` types; only the codec and its tests include the generated header.
- **Decoded bytes are untrusted.** Verify, then bound every count against the fixed arrays it fills, and check for a body behind a body type.
- **A new enum value needs both sides.** The core enum and the schema enum carry the same numbers (pinned by `static_assert`), and the decoder must accept the value before a plugin sends it.
- **The frame path does not allocate.** Keep `encode(const FrameEvent&)` free of containers that do.
