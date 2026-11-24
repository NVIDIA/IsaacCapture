<!--
SPDX-FileCopyrightText: Copyright (c) 2026 mwang259. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

![DGlove hand pose, surface tactile heatmap, and particle tactile heatmap](dglove-effect-github.png)

<h1 align="center">DGlove for Isaac Capture</h1>

<p align="center"><strong>Capture hand pose and palm tactile data from a liquid-metal sensing glove.</strong></p>

<p align="center">
  <img alt="Platform: Linux" src="https://img.shields.io/badge/Platform-Linux-315A84">
  <img alt="Python 3.11 or newer" src="https://img.shields.io/badge/Python-3.11%2B-315A84">
  <img alt="Plugin license: Apache 2.0" src="https://img.shields.io/badge/Plugin-Apache%202.0-315A84">
</p>

---

## Overview

DGlove is a tactile glove based on liquid-metal sensing. The DGlove application
estimates hand pose and sends UDP/Protobuf frames. The Isaac Capture plugin
validates the frames and publishes them as `DGloveFrame` records for downstream
applications.

The illustration shows hand pose, a continuous surface heatmap, and a particle
heatmap from left to right.

## Data

- **45** local MANO axis-angle components in radians.
- **21** wrist-relative 3D joint positions in metres.
- **210** raw palm tactile ADC channels.

## Quick start

1. Build the plugin on Linux with `-DBUILD_PLUGIN_DGLOVE=ON`.
2. Start CloudXR, load its exported environment, and launch the DGlove plugin.
3. Connect the glove in the DGlove application and enable UDP output. For a
   same-host setup, the default destination is `127.0.0.1:50051`. If you change
   the port, use the same port in the application and receiver. Run
   `dglove_reader` to inspect the published frames.

See the [setup and usage guide](USAGE.md) for commands, UDP diagnostics, and
synthetic tests.

**Validation:** The plugin was built and tested locally on Linux. An end-to-end
CloudXR test recovered all 240 synthetic frames. Live hardware validation is
pending.
