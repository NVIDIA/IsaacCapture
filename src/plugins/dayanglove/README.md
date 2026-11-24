<!--
SPDX-FileCopyrightText: Copyright (c) 2026 mwang259. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

![DAYANGlove hand pose, surface tactile heatmap, and particle tactile heatmap](dayanglove-effect-github.png)

<h1 align="center">DAYANGlove for Isaac Capture</h1>

<p align="center"><strong>Capture hand pose and palm tactile data from a liquid-metal sensing glove.</strong></p>

<p align="center">
  <img alt="Platform: Linux" src="https://img.shields.io/badge/Platform-Linux-315A84">
  <img alt="Python 3.11 or newer" src="https://img.shields.io/badge/Python-3.11%2B-315A84">
  <img alt="Plugin license: Apache 2.0" src="https://img.shields.io/badge/Plugin-Apache%202.0-315A84">
</p>

---

## Overview

DAYANGlove is a device integration plugin for NVIDIA Isaac Capture that provides
access to hand pose and palm tactile data from a custom liquid-metal tactile
glove. The external DGlove application communicates with the glove, processes
sensor data, and estimates hand pose. It sends Protobuf frames over UDP. The
plugin validates the incoming frames and publishes them as FlatBuffers
`DAYANGloveFrame` records for downstream applications.

The illustration shows hand pose, a continuous surface heatmap, and a particle
heatmap from left to right.

## Data

- **45** local MANO axis-angle components in radians.
- **21** wrist-relative 3D joint positions in metres.
- **210** raw palm tactile ADC channels.

## Quick start

1. Build the plugin on Linux with `-DBUILD_PLUGIN_DAYANGLOVE=ON`.
2. Start CloudXR, load its exported environment, and launch the DAYANGlove plugin.
3. Connect the glove in the DGlove application and enable UDP output. If the
   application and plugin run on the same machine, send data to
   `127.0.0.1:50051`. If you change the port, use the same port in the application
   and receiver. Run `dayanglove_reader` to inspect the published frames.

See the [setup and usage guide](USAGE.md) for commands, UDP diagnostics, and
synthetic tests.

**Validation:** The plugin was built and tested locally on Linux. An end-to-end
CloudXR test recovered all 240 synthetic frames. Live hardware validation is
pending.
