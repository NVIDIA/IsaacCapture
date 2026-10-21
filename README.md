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
- [OVXR installation and source-built camera support](docs/source/getting_started/ovxr.rst) (optional public packages)


### Install & Run Isaac Lab

The latest [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) release, [3.0.0 EA](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-EA), includes Isaac Capture, previously released as Isaac Teleop.

To get started, please refer to Isaac Lab's [Installation](https://isaac-sim.github.io/IsaacLab/develop/source/setup/installation/index.html) guide for more details. Then follow the [CloudXR teleoperation in Isaac Lab](https://isaac-sim.github.io/IsaacLab/develop/source/how-to/cloudxr_teleoperation.html) to get started with Teleoperation in Sim.

## Licensing

The default license is [Apache-2.0](LICENSE.md).
EgoPost is licensed under [OpenMDW-1.1](src/python/isaaccapture_egopost/LICENSE.md).
Third-party materials may carry their own licenses.

### Separately retrieved optional materials

The optional `isaaccapture[ovxr]` extra retrieves OVXR and its OVRTX/OVStage
dependencies. The separate `isaaccapture[desktop-mirror]` extra retrieves OVUI.
These packages are downloaded separately, rather than bundled in the Isaac Capture
wheel. Their licenses apply independently of Isaac Capture's Apache-2.0 license.
OVXR, OVRTX and OVStage are NVIDIA proprietary software. OVUI 0.3.0 is Apache-2.0.
See [package sources and license details](docs/source/getting_started/ovxr.rst#optional-packages-and-licenses).

The following notice applies to separately retrieved external materials. Existing
CloudXR native libraries are bundled in Isaac Capture wheels and remain subject
to their separate NVIDIA license and EULA acceptance.

NOTICE AND DISCLAIMER: This software automatically retrieves, accesses or interacts with external
materials.  Those retrieved materials are not distributed with this software and are governed solely
by separate terms, conditions and licenses.  You are solely responsible for finding, reviewing and
complying with all applicable terms, conditions, and licenses, and for verifying the security,
integrity and suitability of any retrieved materials for your specific use case. This software is
provided "AS IS", without warranty of any kind. The author makes no representations or warranties
regarding any retrieved materials, and assumes no liability for any losses, damages, liabilities or
legal consequences from your use or inability to use this software or any retrieved materials. Use
this software and the retrieved materials at your own risk.
