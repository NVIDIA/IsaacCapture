<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Isaac Camera Streamer — GMSL / Argus

This component streams Argus camera data using AVS1/TCP. The protocol and
configuration tests run without camera hardware.

## Protocol and configuration tests

```bash
cmake -S examples/camera_streamer/argus_sender -B /tmp/camera-protocol-build -DARGUS_SENDER_CPU_TESTS_ONLY=ON
cmake --build /tmp/camera-protocol-build --parallel 2
ctest --test-dir /tmp/camera-protocol-build --output-on-failure
```

See [the wire specification](argus_sender/WIRE_FORMAT.md). Shared camera
profiles are in `configs/`; select sensor modes using the installed camera driver.

## Jetson sender

Install the camera vendor driver and Jetson Multimedia API on the Orin. Stop any
preview using the sensors and configure the vendor PWM for the selected frame rate.
From `examples/camera_streamer` (or a copy of that directory on the robot):

```bash
cmake -S argus_sender -B argus_sender/build -DCMAKE_BUILD_TYPE=Release
cmake --build argus_sender/build --parallel 2
ctest --test-dir argus_sender/build --output-on-failure
argus_sender/build/argus_sender --list-modes
PROFILE=s56-stereo-2560x1984-30 # Choose your resolution
argus_sender/build/argus_sender --config "configs/${PROFILE}.yaml" --check-config
argus_sender/build/argus_sender --config "configs/${PROFILE}.yaml"
```

The sender listens for a TCP receiver on port 5000 by default.

## Workstation NVDEC decoder

On a Linux workstation with an NVIDIA GPU, CUDA toolkit, CMake and `uv`, run from
`examples/camera_streamer`:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python pybind11 numpy
cmake -S camera_receiver/codec -B camera_receiver/codec/build \
  -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build camera_receiver/codec/build --parallel 2
```

The build downloads the pinned NVIDIA Video Codec SDK. CPU-side Python tests:

```bash
uv run --project tests --extra dev python -m pytest tests/python -q
```

## Workstation receiver

Run `ip -brief -4 address` on the robot and choose the address reachable from the
workstation. From `examples/camera_streamer`, with the decoder already built:

```bash
export ROBOT_IP="<robot IPv4 address>"
uv pip install --python .venv/bin/python -r camera_receiver/requirements.txt
PROFILE=s56-stereo-2560x1984-30 # Choose your resolution
.venv/bin/python camera_receiver/camera_receiver.py \
  --config "configs/${PROFILE}.yaml" --host "${ROBOT_IP:?Set ROBOT_IP first}"
```

Use the same profile on both computers. Decoded GPU frames are published through
local CUDA IPC; the receiver does not open a window. Stop it with Ctrl+C.

## Licensing

This example is Apache-2.0 licensed. Existing upstream notices and external SDK
licenses remain in effect. The Argus-derived `EGLGlobal.h`, `EGLGlobal.cpp` and
`Error.h` retain their BSD-3-Clause license and notices.

## Rebuilding after a source-directory move

CMake caches absolute source paths. After updating an existing deployment, configure
and build the native sender and decoder again at the paths shown above. Do not move
or reuse an old build directory from a different source location.

For opt-in hardware smoke tests, install PyYAML for the Python interpreter selected
by CMake. Set `ARGUS_SENDER_HARDWARE_CONFIG` to your stereo profile; the smoke receiver
reads `network.port` from that same file (default 5000), so there is no separate
receiver-port override.

CUDA IPC protocol 3 requires consumers to hold each slot's shared file lock and
verify `slot_generation` before reading pixels, then finish GPU reads before
unlocking. The receiver holds an exclusive lock while writing. Delayed notifications
are discarded; a slow active read backpressures slot reuse rather than corrupting
pixels. Update receiver and consumer together; protocol 2 consumers are rejected.
