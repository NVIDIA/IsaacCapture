<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — `camera_protocol`

**CRITICAL (non-optional):** Before editing this package, complete the mandatory **`AGENTS.md` preflight** in [`../../../AGENTS.md`](../../../AGENTS.md) (read every applicable `AGENTS.md` on your paths, not just this file).

The schema is [`../schema/fbs/camera_protocol.fbs`](../schema/fbs/camera_protocol.fbs). Read [`../schema/AGENTS.md`](../schema/AGENTS.md) and [`README.md`](README.md) before changing it: a reader and a plugin from different releases must keep understanding each other, and the golden check enforces it.

- **Few messages, and no fewer than correctness needs.** A connection has one purpose, stated in the handshake; do not add a message for what a handshake field can say. A newest-order reader asks for each frame set (`CameraAcquire`): do not fold that back into sending frame sets as they are published.
- **This is local IPC.** Lease ids, slot indices and attached descriptors name resources of one machine. Do not reuse the envelope as a network or recording format, and do not wrap it in the recorder's `Record` tables or its `DeviceDataTimestamp`.
- **A new message says whether it may be ignored.** One that only informs is sent with `skippable`; an operation the peer must carry out is not, and needs a protocol version so that it is only sent to a peer that knows it.
- **The numbers are the protocol.** Union members, flag bits and enum values are written out in the schema and pinned by `static_assert` and a test; a member keeps its number.
- **FlatBuffers stays inside this module.** Its users see `camera_core` types; only the codec and its tests include the generated header.
- **Decoded bytes are untrusted.** Verify, then bound every count against the fixed arrays it fills, and check for a body behind a body type. The decoder establishes structure only; what a description means is checked by `camera_core`'s `problem_with` where it is used.
- **The encoder refuses, it does not truncate.** A count that does not fit is an error to the caller, never a shorter message.
- **A group this build cannot read is an entry, not an error.** A kind added by a newer plugin marks its group `unsupported` and leaves the other groups of the same response readable.
- **A new enum value needs both sides.** The core enum and the schema enum carry the same numbers (pinned by `static_assert`), and the decoder must accept the value before a plugin sends it.
- **The frame path does not allocate.** Keep `encode(const FrameEvent&)` free of containers that do.
- **Evolution is tested with hand-built messages.** A passing golden comparison shows layout, not that two releases understand each other: a rule about newer or older peers gets a test that builds the message that peer would send.
