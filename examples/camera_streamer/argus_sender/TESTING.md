<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Testing argus_sender

## Unit and loopback tests

Run these commands on a Jetson with the build dependencies installed. The tests
use the Jetson SDK even when they do not open a camera.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

The default suite does not open a camera. It covers YAML capture-mode parsing and validation,
output geometry, mixed-rate scheduling, capture-priority selection and release
timing, stereo alignment policy, AVS1 serialization/parsing, asynchronous bundle
assembly and ordering, encoder input-slot lease release, plus TCP
send/dump/disconnect and bounded stalled-receiver drain behavior.

The matching receiver parser tests run from `camera_receiver/`:

```bash
python3 -m pytest tests -q
```

The deployment snapshot/restore checks run from `argus_sender/`:

```bash
python3 -m pytest tests/test_deployment.py -q
```

## 120-frame hardware smoke test

The opt-in test opens the devices in
`configs/hardware_stereo_smoke.yaml`, creates two HEVC encoders, and runs a local
AVS1 receiver:

```bash
cmake -S . -B build \
  -DARGUS_SENDER_HARDWARE_TESTS=ON \
  -DARGUS_SENDER_HARDWARE_CONFIG="$PWD/configs/hardware_stereo_smoke.yaml"
cmake --build build -j$(nproc)
ctest --test-dir build -R '^avs1_hardware_smoke$' --output-on-failure -V
```

The test receives exactly 120 bundles and verifies:

- contiguous AVS1 bundle sequence numbers;
- a present stereo HEVC stream with nonempty left/right payloads;
- monotonically increasing per-eye SOF TSC timestamps when available;
- VPS, SPS, PPS, and an IRAP NAL in each first access unit;
- clean sender exit with no reconnecting process left behind.

Override the loopback port with `-DARGUS_SENDER_TEST_PORT=PORT`. To use other
devices, modes, dimensions, or rates, copy the hardware YAML and pass its path
through `ARGUS_SENDER_HARDWARE_CONFIG`.
