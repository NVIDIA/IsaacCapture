<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — `src/plugins/remote_device_bridge`

**CRITICAL:** Complete the mandatory `AGENTS.md` preflight in [`../../../AGENTS.md`](../../../AGENTS.md) before editing here.

- `session/protocol_frame.*` and `session/protocol_payloads.*` are the only serialization points. Every frame is encoded and decoded there and nowhere else; changing the wire format means changing them and `tests/cpp/plugins/remote_device_bridge/test_wire_codec.cpp` together. Sessions hold an `ITransport` (`session/inc/remote_plugin_session/transport.hpp`) and only call `send/receive/shutdown` on complete frames; TCP framing lives in `TcpTransport`.
- WebSocket is a second transport, not a second protocol: `WebSocketTransport` carries exactly one wire frame per binary message (no TLS; terminate `wss://` in a proxy). Clients pick it with `--remote=ws://host:port[/path]` via `parse_remote`/`connect_transport`; the bridge serves it on a separate `--ws-listen` port sharing the fork-per-client loop and `--max-clients`. Never sniff the protocol on the TCP port. The server-side HTTP upgrade runs in the first `receive()` so the accept loop never blocks on a client.
- Never add `OpenXR::openxr_loader`, `Teleop::plugin_utils` or DeviceIO session libraries to `Teleop::remote_plugin_session`: a plugin using `--remote` must run without an OpenXR runtime.
- ERROR on channel 0 ends the session; ERROR on a channel id refuses only that OPEN (schema above `--max-schema-bytes`, a hand someone else owns) and the session and its other channels continue. Refuse per channel whenever the session itself is still sound.
- `RemotePluginSession` never throws on the sample path for transport failures (`push`, `push_buffer`, `query`, `update`): after one they become no-ops; only channel creation and the constructor throw. Caller bugs (null joints or buffer) still throw `std::invalid_argument`.
- Timestamps cross the wire in the sender's `CLOCK_MONOTONIC`; the receiver maps them with `ClockOffsetEstimator`. The bridge's offset comes from PING/PONG (min-RTT of the last 32 exchanges, `--clock-source=two-way`); the one-way minimum of `local_recv - remote_send` hides the minimum network delay and is only the fallback (no PONG within 2 s) or the `--clock-source=one-way` A/B baseline. Do not push raw remote timestamps into OpenXR.
