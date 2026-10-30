<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — `src/plugins/remote_device_bridge`

**CRITICAL:** Complete the mandatory `AGENTS.md` preflight in [`../../../AGENTS.md`](../../../AGENTS.md) before editing here.

- `session/protocol_frame.*` and `session/protocol_payloads.*` are the only serialization points. Every frame is encoded and decoded there and nowhere else; changing the wire format means changing them and `tests/cpp/plugins/remote_device_bridge/test_wire_codec.cpp` together. Sessions hold an `ITransport` (`session/inc/remote_plugin_session/transport.hpp`) and only call `send/receive/shutdown` on complete frames; TCP framing lives in `TcpTransport`.
- WebSocket is a second transport, not a second protocol: `WebSocketTransport` carries exactly one wire frame per binary message (no TLS; terminate `wss://` in a proxy). Clients pick it with `--remote=ws://host:port[/path]` via `parse_remote`/`connect_transport`; the bridge serves it on a separate `--ws-listen` port sharing the fork-per-client loop and `--max-clients`. Never sniff the protocol on the TCP port. The server-side HTTP upgrade runs in the first `receive()` so the accept loop never blocks on a client.
- `Teleop::remote_plugin_session` links only `pusherio::pusherio` (OpenXR value types via headers). Never add `OpenXR::openxr_loader`, `Teleop::plugin_utils` or DeviceIO session libraries to it: a plugin using `--remote` must run without an OpenXR runtime.
