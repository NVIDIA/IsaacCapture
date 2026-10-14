<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA Isaac Capture

<div align="center">

**Shared capture infrastructure for teleoperation and egocentric data.**

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-3.0.0-orange.svg)](https://isaac-sim.github.io/IsaacLab/develop)
[![numpy](https://img.shields.io/badge/numpy-1.23%2B-lightgrey.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)


</div>

---

<!-- Keep one overview image, below the title and badges and immediately before Overview. -->
![Isaac Capture architecture: Isaac Teleop and Ego Pipeline share device plugins, retargeters, and data utilities, with integrations into simulation and real-robot environments](docs/source/_static/isaac-capture-architecture.png)

## Overview

**Isaac Capture** provides shared infrastructure for collecting human and robot demonstration data.
It brings together two components:

- **Isaac Teleop** supports robot-in-the-loop capture through teleoperation, including Teleop Session and Remote Assist.
- **Ego Pipeline** supports human and egocentric data capture through Ego Capture and Ego Post Processing.

Both components use a shared core of device plugins, retargeters, and data utilities for recording,
replaying, and converting data. Isaac Teleop remains a component of Isaac Capture, with its existing
functionality preserved.

## Key Features

- Unified stack for sim & real teleoperation
- Standardized device interface
- Flexible retargeting framework

### Isaac Teleop

- Currently supported use cases
  - Use XR headsets for gripper / tri-finger hand manipulation
  - Use XR headsets with gloves for dex-hand manipulation
  - Seated full body loco-manipulation (Homie)
  - Tracking based full body loco-manipulation with GR00T-WholeBodyControl (SONIC)
- Upcoming use cases
  - Teleoperate using only non-XR devices (e.g. gamepad, Gello, haply, etc.)
  - Teleoperate cloud based robotics simulations
  - Remote teleoperation with camera streaming to the desktop
  - Remote teleoperation with immersive camera streaming to XR headsets

### Ego Pipeline

- Capture egocentric data without a robot in the loop.
- Reconstruct hand and camera poses from egocentric video with the
  [Ego Post Processing pipeline](https://nvidia.github.io/IsaacCapture/main/references/egocentric_hand_reconstruction.html).

## Quick Start

### Documentation

Our [documentation page](https://nvidia.github.io/IsaacCapture) provides everything you need to get started, including detailed tutorials and step-by-step guides. Follow these links to learn more:

- [Architecture](https://nvidia.github.io/IsaacCapture/main/overview/architecture.html)
- [Quick installation steps](https://nvidia.github.io/IsaacCapture/main/getting_started/quick_start.html)
- [How to build from source](https://nvidia.github.io/IsaacCapture/main/getting_started/build_from_source.html)


### Install & Run Isaac Lab

The latest [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) release, [3.0.0 EA](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-EA), includes Isaac Capture, previously released as Isaac Teleop.

To get started, please refer to Isaac Lab's [Installation](https://isaac-sim.github.io/IsaacLab/develop/source/setup/installation/index.html) guide for more details. Then follow the [CloudXR teleoperation in Isaac Lab](https://isaac-sim.github.io/IsaacLab/develop/source/how-to/cloudxr_teleoperation.html) to get started with Teleoperation in Sim.
