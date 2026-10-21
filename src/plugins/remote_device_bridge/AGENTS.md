<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — `src/plugins/remote_device_bridge`

**CRITICAL:** Complete the mandatory `AGENTS.md` preflight in [`../../../AGENTS.md`](../../../AGENTS.md) before editing here.

- `session/protocol_frame.*` and `session/protocol_payloads.*` are the only serialization points. Every frame is encoded and decoded there and nowhere else; changing the wire format means changing them and `tests/cpp/plugins/remote_device_bridge/test_wire_codec.cpp` together. Sessions hold an `ITransport` (`session/inc/remote_plugin_session/transport.hpp`) and only call `send/receive/shutdown` on complete frames; TCP framing lives in `TcpTransport`.
- WebSocket is a second transport, not a second protocol: `WebSocketTransport` carries exactly one wire frame per binary message (no TLS; terminate `wss://` in a proxy). Clients pick it with `--remote=ws://host:port[/path]` via `parse_remote`/`connect_transport`; the bridge serves it on a separate `--ws-listen` port sharing the fork-per-client loop and `--max-clients`. Never sniff the protocol on the TCP port. The server-side HTTP upgrade runs in the first `receive()` so the accept loop never blocks on a client.
- `Teleop::remote_plugin_session` links only `pusherio::pusherio` (OpenXR value types via headers). Never add `OpenXR::openxr_loader`, `Teleop::plugin_utils` or DeviceIO session libraries to it: a plugin using `--remote` must run without an OpenXR runtime.
- ERROR on channel 0 ends the session; ERROR on a channel id refuses only that OPEN (schema above `--max-schema-bytes`, a hand someone else owns) and the session and its other channels continue. Refuse per channel whenever the session itself is still sound.
- `RemotePluginSession` never throws on the sample path (`push`, `push_buffer`, `query`, `update`). After a transport failure those become no-ops; only channel creation and the constructor throw.
- Timestamps cross the wire in the sender's `CLOCK_MONOTONIC`; the receiver maps them with `ClockOffsetEstimator`. The bridge's offset comes from PING/PONG (min-RTT of the last 32 exchanges, `--clock-source=two-way`); the one-way minimum of `local_recv - remote_send` hides the minimum network delay and is only the fallback (HELLO v0 client, no PONG within 2 s) or the `--clock-source=one-way` A/B baseline. Never PING a HELLO v0 client: it drops the connection on unknown frame types. Do not push raw remote timestamps into OpenXR.
- Tests that need a stalled link use `BlockingTransport` in `test_remote_plugin_session.cpp`, never a TCP peer that stops reading: loopback socket buffers differ per host (416 KiB in one sandbox, 2×8 MiB on the teleop workstation), so "the sender is blocked by now" is not portable.
