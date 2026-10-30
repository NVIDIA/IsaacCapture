<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — `src/plugins/remote_device_bridge`

**CRITICAL:** Complete the mandatory `AGENTS.md` preflight in [`../../../AGENTS.md`](../../../AGENTS.md) before editing here.

- `session/protocol_frame.*` and `session/protocol_payloads.*` are the only serialization points. Every frame is encoded and decoded there and nowhere else; changing the wire format means changing them and `tests/cpp/plugins/remote_device_bridge/test_wire_codec.cpp` together.
- `Teleop::remote_plugin_session` links only `pusherio::pusherio` (OpenXR value types via headers). Never add `OpenXR::openxr_loader`, `Teleop::plugin_utils` or DeviceIO session libraries to it: a plugin using `--remote` must run without an OpenXR runtime.
