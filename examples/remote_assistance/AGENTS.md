<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Remote Assistance prototype

- For shareable diagrams in CLI sessions, provide a local SVG fallback when
  connector generation requires a selection widget the user cannot access.

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
  In network setup, give Robot Site and Edge Compute separate startup steps,
  with Robot Site first after the shared Edge server is available.
  List Robot Site before Edge Compute in the process overview; keep the table,
  process headings and network setup references aligned when reordering them.
  Include MuJoCo and SONIC's host/container launch commands directly in network
  startup; state that MuJoCo is required for simulation and omitted for real robots;
  identify processes with comments above their command blocks in network setup,
  including the network SFU as Edge Process 0.
  Include the final Pico streamer launch and headset connection instruction
  directly in network startup; do not rely on a link to the process overview.
- Mark host-to-container transitions explicitly in deployment instructions;
  an environment prompt prefix does not establish that Docker was entered.
- Document fresh Robot Site provisioning (GR00T clone/LFS, deployment guide,
  matching model downloads and the installer for each venv); do not infer Thor
  policy support from working ARM64 transport packages or Orin instructions.
- Pair network port tables with site-specific firewall commands, status checks,
  and address scope; distinguish host rules from upstream network policy.
  Give each site's firewall configuration its own numbered setup step and keep
  subsequent step numbers and references consistent.
  Define shared site IP and login username variables in step 0 on both systems,
  explain terminal scope, and reuse them throughout setup commands.
- Network profiles must keep server signing/TLS private keys on Edge Compute,
  distribute only site-scoped client tokens, and verify the generated profile
  through the actual SDK (including certificate rejection and changing video).

- Keep repeated network setup operations in runnable helpers; README steps should
  use short commands rather than requiring operators to paste remote shell bodies.
- Mark shell helpers with shebangs executable before staging; pre-commit enforces
  the executable bit even when README commands invoke them through bash.
- In stack comparison diagrams, place Edge Compute on the left and Robot Site
  on the right; align the original stack from camera visualization to simulator.
- Put credential sourcing and the generated YAML path together in network launch
  examples; label localhost alternatives explicitly at the point of use.
- Resolve viewer import failures before suggesting a launch retry; configuration
  changes cannot repair a mixed legacy/current Python environment. Preserve local
  builds and shared package files when replacing a legacy distribution.

- Bundle generation must support first-time setup and renewal with the same command;
  preserve existing bundles in backups before installing replacements. Document
  that command once per setup step, with a short note explaining replacement.

- Reachability checks must name the site-specific certificate path, expected
  response, and recovery step for a missing transferred bundle.

- Pre-encoded video queues must recover at a keyframe after dropping dependent
  frames. Validate codec/decoder pairs with changing pixels in separate processes;
  decoder environment switches are process-wide and H.265 must not inherit the
  software-only H.264 test configuration. Keep private FFI adapters version-pinned.

- After replacing generated server bundles, recreate the network containers so
  bind mounts and loaded credentials match the new files. Include an Edge-local
  TLS check before Robot Site connectivity checks.

- When Pico Manager reports a missing OpenXR body-tracking extension then waits
  forever, check the shared CloudXR runtime's originating environment. Matching
  isaaccapture package versions do not guarantee matching native runtime builds.
  For the tested camera_viz/teleop mismatch, stop both clients and CloudXR, then
  launch Pico Manager first from its compatible teleop environment, followed by
  the viewer. Verify body samples; align native builds for a durable fix.

- Troubleshooting entries should reproduce the complete operator-facing error or
  waiting message on its own highlighted line so users can match terminal output.

- Experimental remote-camera instructions must transfer uncommitted backend code
  and dependency metadata before generating profiles that select the new backend.
  Copying configuration does not update the Robot Site implementation.

- Camera presets belong in Edge-generated bundles with matching viewer settings;
  transfer them with the Robot bundle rather than requiring per-site YAML edits.
- Keep camera source adapters separate from encoding and transport. Adding an
  Argus source must preserve V4L2 and DepthAI profile behavior and optional imports.
  Match GStreamer Python bindings to the venv interpreter, not just the OS.
- Verify repeated camera starts and actual output frames before treating a
  display/environment change as the fix for an intermittent Argus failure.
- Put operator commands in copyable conversation code blocks, not question
  widgets. Use SSH for authorized remote work; ask only for the sudo step needed.
