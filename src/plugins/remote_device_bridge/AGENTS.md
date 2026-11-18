<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — `src/plugins/remote_device_bridge`

**CRITICAL:** Complete the mandatory `AGENTS.md` preflight in [`../../../AGENTS.md`](../../../AGENTS.md) before editing here.

- `session/protocol_frame.*`, `session/protocol_payloads.*` and the `Reader`/`Writer` in `session/protocol_io.hpp` are the only serialization points. Every frame is encoded and decoded there and nowhere else; changing the wire format means changing them and their codec tests together.
- Wire format v1 (`kProtocolVersion` = 1, the only version number): 24-byte header `RPS0` magic, type u8, version u8, channel u16, seq u32, payload size u32, `t_send` i64; frame types 1..13 with no gaps; the hand byte is 0 = left, 1 = right in every payload; PING has no payload and PONG carries the PING's header `t_send` and the client's receive time. Any change to a header, payload or type number bumps `kProtocolVersion`; `decode_header` rejects other versions before reading the type, so old senders and bridges fail with a readable reason instead of mis-decoding.
- A frame's `t_send` is its send time for clock exchange: call `stamp_header` where the frame is written to the transport, never when it is queued.
- Never add `OpenXR::openxr_loader`, `Teleop::plugin_utils` or DeviceIO session libraries to `Teleop::remote_plugin_session`: a plugin that talks to a bridge through this library must run without an OpenXR runtime.
