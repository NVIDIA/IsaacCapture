<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Remote Assistance prototype

- Media integration checks must verify changing pixel content against source
  frame IDs. Fresh receive timestamps and correct dimensions do not prove that
  the decoder produced a fresh image.
- Keep robot-side capture and data transport importable without isaaccapture,
  CuPy, Vulkan, or CloudXR. GPU upload belongs to the camera_viz source adapter.
- Run network integration tests against an isolated room and test endpoints;
  never use the live SONIC command ports for automated test traffic.
- Preserve local target-before-command ordering when coalescing SONIC messages;
  a discrete command is a boundary across which state must not be replaced.
  Separate reliable/lossy channels do not guarantee remote cross-topic ordering.
- After a formatting hook rewrites files, stage the formatted versions and
  rerun the full hook set; the first run is not a passing verification.
- Check controller command semantics before using a message as a watchdog action.
  SONIC stop exits deployment: never synthesize it on input timeout or bridge
  shutdown. Delegate hold/idle to the controller; test loss and recovery as well
  as startup silence, and preserve explicit operator stop commands.
- Base input watchdog freshness on received control, not outbound lease readiness;
  bidirectional transport handshakes can become ready at different times.
- Keep startup documentation organized by site, with Essential, Robot and Camera
  Televiz process groups; separate one-time provisioning from session commands.
- Mark host-to-container transitions explicitly in deployment instructions;
  an environment prompt prefix does not establish that Docker was entered.
- Pair network port tables with site-specific firewall commands, status checks,
  and address scope; distinguish host rules from upstream network policy.
  Give each site's firewall configuration its own numbered setup step and keep
  subsequent step numbers and references consistent.
  Define shared site IP variables in step 0 on both systems, explain terminal
  scope, and reuse them throughout setup commands.
- Network profiles must keep server signing/TLS private keys on Edge Compute,
  distribute only site-scoped client tokens, and verify the generated profile
  through the actual SDK (including certificate rejection and changing video).
