<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Isaac Teleop Device Plugins - Haptikos
Use the Haptikos Exoskeletons with the Isaac Teleop framework. Currently only `Linux` is supported. Tested on `Meta Quest` headsets. Other headsets with controller may work as well.

## Overview
Reads the controllers' position and the hand tracking data from the Haptikos Core App, combines them and pushes them into the OpenXR runtime. To inject the hand tracking data, the controllers, the Haptikos Core App and the exoskeletons need to be active.

## Quick Start

### Get the Haptikos ROBOTICS API

The Haptikos ROBOTICS API must be downloaded seperately due to licencing. Check our [website](https://haptikos.tech).

1. Obtain a Haptikos account.
2. Clone the [repository](https://github.com/The-Magos/HaptikosAPI).
3. Copy the `HaptikosCpp_API_Shared` folder from our HaptikosAPI repository into the `src/plugins/haptikos` folder. The folder structure should be the following:
```
src/plugins/haptikos/
├── CMakeLists.txt
├── HaptikosCpp_API_Shared
│   ├── include
│   └── lib
├── haptikos_hands_plugin.cpp
├── inc
│   └── haptikos
├── main.cpp
├── plugin.yaml
└── README.md
```
### Build

####  Build the plugin

```bash
cd ../../..  # Navigate to TeleopCore root
cmake -S . -B build
cd build
make haptikos_hands_plugin
```

####  Build the entire project
```bash
cmake -B build -DENABLE_CLANG_FORMAT_CHECK=OFF #From Project root
cmake --build build --parallel 4
cmake --install build
```

We use the `-DENABLE_CLANG_FORMAT_CHECH=OFF` because the headers are HaptikosAPI library are not clang formatted. You can clang format the headers to not include the flag by running:
```bash
clang-format -i src/plugins/haptikos/HaptikosCpp_API_Shared/include/*
```

### Run

The executable will be located in the `build/src/plugins/haptikos`. If you installed the project by running `cmake --install build` the executable will also be found in the `/install/plugins/haptikos` folder.

### Using the plugin

To use the plugin properly it is necessary to attach the controllers on our exoskeletons using the included mount. The Haptikos Core App, the Haptikos Exoskeletons and the controllers, all need to be active while the plugin is running.

#### Orientation Calibration
The orientation calibration defines the directional the Haptikos Gloves define as [forward](https://doc-haptikos.tech/HaptikosApp.html#hand-direction-calibration). This needs to be aligned with the Head Mounted Display's. For example, when connecting to a `Meta Quest` using the [Nvidia Teleop Quick Start Guide](https://nvidia.github.io/IsaacCapture/main/getting_started/quick_start.html), the forward direction is towards the control panel that appears after Step 5 (Connect to an XR Headset).
