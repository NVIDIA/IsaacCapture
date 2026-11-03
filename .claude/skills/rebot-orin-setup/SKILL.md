---
name: rebot-orin-setup
description: >-
  Use when initializing a reBot B601-RS (RobStride 6-DOF, seven motors) arm
  from an NVIDIA Jetson AGX Orin using the Orin's built-in CAN (mttcan) via an external
  SN65HVD230 transceiver — no USB-CAN/PCAN adapter. Covers the uv/motorbridge
  env, bringing up can0 @ 1 Mbit, verifying the bus, reboot persistence,
  motor-ID assignment, zero calibration, and the CAN bring-up troubleshooting
  tree (loopback-passes-but-no-motors, pinmux, motor power).
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# reBot B601-RS on Jetson AGX Orin

The procedure, assets, and troubleshooting tree live with the example:
**[`integrations/rebot/SKILL.md`](../../../integrations/rebot/SKILL.md)**.

Read that file and work from `integrations/rebot/`.
