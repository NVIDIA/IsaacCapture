<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GMSL Argus sender

The native sender captures configured Argus sensors, hardware-encodes HEVC,
and publishes AVS1 bundles over TCP. It also supports raw NV12 output.

Use the [standalone setup and GMSL commands](../../README.md).
The supplied profiles select stereo sensors 0/1 at 2560×1984/30 or 1920×1080/60;
verify mode enumeration and hardware PWM for your adapter before starting.

See [WIRE_FORMAT.md](WIRE_FORMAT.md) for the protocol and
[TESTING.md](TESTING.md) for unit and opt-in hardware tests.
Calibration/remapping code accepts user-supplied maps; this example bundles no
rig calibration or system service installers.
