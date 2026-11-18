<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — `src/plugins/remote_device_bridge`

**CRITICAL:** Complete the mandatory `AGENTS.md` preflight in [`../../../AGENTS.md`](../../../AGENTS.md) before editing here.

- `session/protocol_frame.*`, `session/protocol_payloads.*` and the `Reader`/`Writer` in `session/protocol_io.hpp` are the only serialization points. Every frame is encoded and decoded there and nowhere else; changing the wire format means changing them and their codec tests together.
- Never add `OpenXR::openxr_loader`, `Teleop::plugin_utils` or DeviceIO session libraries to `Teleop::remote_plugin_session`: a plugin that talks to a bridge through this library must run without an OpenXR runtime.
