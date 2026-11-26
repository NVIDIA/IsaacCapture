<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Camera streamer

Standalone camera streaming for a trusted local LAN. Capture runs on an NVIDIA
Jetson AGX Orin or Thor with an Argus-compatible camera driver. The reference
setup has been used with Sensing SHW5G GMSL cameras; other cameras need their own
driver, sensor-mode selection, and calibration. The receiver and CUDA IPC
consumer run on a Linux host with an NVIDIA GPU, using Ubuntu 22.04 or 24.04.

```text
AGX Orin / Thor                      Linux GPU host
Argus cameras → argus_sender ─TCP─→ camera_receiver ─CUDA IPC─→ cuda_ipc_consumer
              HEVC or raw NV12       NVDEC / GPU rings         display or metadata
```

This transport is designed for a local LAN and is **not suitable for the public
internet**. AVS1/TCP has no authentication, encryption, NAT traversal, or adaptive
congestion control. Keep the sender port on a trusted camera network. CUDA IPC
works between processes on the receiving host and requires compatible access to
the same physical GPU; its handles cannot be sent to another machine.

## Layout

| Folder | Purpose |
| --- | --- |
| [argus_sender](argus_sender/README.md) | Argus capture, stereo SOF pairing, mixed-rate streams, HEVC/NV12, optional sender remapping, TCP streaming and wire dumps |
| [camera_receiver](camera_receiver/README.md) | Reconnecting TCP receiver, parallel stream processing, NVDEC decode, presentation rotation and GPU ring publication |
| [cuda_ipc_consumer](cuda_ipc_consumer/README.md) | Stream subscription, NV12/RGBA preview, finite headless runs and latency CSV output |
| [mock_sender](mock_sender/README.md) | Camera-free AVS1 HEVC sender for running the receiver and downstream consumers without a Jetson; synthetic clock, pause and disconnect scripts |
| [argus_sender/remaps](argus_sender/remaps/README.md) | Bundled example wrist remaps, calibration, manifest and a standalone generator |
| [tests](tests/) | Checks spanning sender and receiver configurations; component tests live inside their own folders |

These tools run without the Isaac Capture Python package or robot control
software. They are built and launched separately from the main repository.

## Environment

**Sender:** the board's supported JetPack/L4T image, working Argus camera driver,
Jetson Multimedia API, CUDA headers, EGL/DRM development packages, CMake and a
C++14 compiler. Install the camera vendor's matching BSP and initialize its
serializer/deserializer as required before capture. Confirm the camera works in
the vendor's Argus sample first. A sensor can have only one capture owner.

**Host:** Ubuntu 22.04 or 24.04, an NVIDIA GPU supporting HEVC NVDEC and CUDA IPC,
a compatible NVIDIA driver and CUDA 12 toolkit including `nvcc`, CMake, a C++17
compiler, and Python 3.10–3.12 with `venv` support. Install the driver and CUDA
toolkit for your GPU using NVIDIA's installation instructions. The component
requirements retain NumPy 1.x and CuPy 13.x; use Python 3.10–3.12 for those pins.

The receiver build downloads the public NVIDIA HoloHub Video Codec SDK bundle
used by the source implementation. It requires internet access during the first
build; its sources and license remain in the build directory.

Use a wired LAN with enough capacity for the selected profiles. Raw NV12 streams
can consume substantial bandwidth. Sender and receiver configs must agree on
stream order, mono/stereo type, dimensions, and port. Synchronize both machines
with chrony or another common time source for cross-host wall-clock latency.
Stereo SOF TSC deltas and local processing durations use their own clock domains.

## Transfer the example

Fetch Git LFS assets before copying; the remaps must be binary RMAP files rather
than pointer text:

```bash
# From the repository root on the machine that holds this checkout:
git lfs pull --include='examples/camera_streamer/argus_sender/remaps/*.remap'
export SENDER_SSH=user@camera-board  # Replace with your SSH target.
rsync -az --exclude='build/' --exclude='.venv/' --exclude='__pycache__/' \
    examples/camera_streamer/ "$SENDER_SSH:camera_streamer/"
```

Remaps are bundled inside `argus_sender/remaps/`. Run the commands below
from the example directory on each machine. The supplied four-camera example
uses Argus device IDs 0, 1, 4, and 5, sensor mode 0, a full-resolution HEVC stereo
stream, and two remapped 640x480 raw NV12 streams at 60 FPS. Those device IDs and
mode numbers are installation-specific: inspect your board and edit the configs.

## Build and send on the board

```bash
cd ~/camera_streamer
./argus_sender/build.sh

# The build wrapper selects single-process Argus. Release any other camera owner.
sudo systemctl stop nvargus-daemon.service
./argus_sender/build/argus_sender --list-modes
./argus_sender/build/argus_sender --check-config \
    --config argus_sender/configs/cameras.yaml
./argus_sender/run.sh
```

`build.sh` installs missing Ubuntu build dependencies using sudo and accepts extra
CMake options. `run.sh` checks daemon ownership and starts the default config;
pass `--config /path/to/config.yaml` to select another config. Direct CMake builds
also support multiprocess Argus; in that mode use the binary directly and keep
`nvargus-daemon` running. See the sender README for capture and codec options.

The example maps are approximate viewing calibration. Replace them for your
camera unit before metric use, and preserve the declared source and output
geometry. Full-resolution head frames are not undistorted by this receiver.

## Receive on the Linux host

```bash
# Install basic build/Python support; install the NVIDIA driver and CUDA 12 first.
sudo apt-get update
sudo apt-get install -y build-essential cmake python3-dev python3-venv
cd /path/to/IsaacTeleop/examples/camera_streamer
./camera_receiver/setup.sh
export SENDER_IP=camera-board  # Replace with the board's LAN IP or hostname.
./camera_receiver/run.sh --host "$SENDER_IP" --port 5000
```

The default receiver endpoint is loopback so the example contains no deployment
address. Use `--host`, `CAMERA_RECEIVER_HOST`, or edit
`camera_receiver/configs/receiver.yaml`. The receiver creates
`/tmp/camera_receiver_avs1_cuda_ipc.sock`; leave it running for consumers.

## Preview or consume on the same host

In another terminal on the receiving host:

```bash
cd /path/to/IsaacTeleop/examples/camera_streamer
./cuda_ipc_consumer/setup.sh
./cuda_ipc_consumer/run.sh --streams head.left,head.right

# Finite headless check, or dump latency metadata:
./cuda_ipc_consumer/run.sh --streams wrist1 --no-display --frames 120
./cuda_ipc_consumer/.venv/bin/python cuda_ipc_consumer/latency_check.py \
    --streams head.left --frames 120 --output latency.csv
```

OpenCV preview requires a graphical desktop. Use `--no-display` for headless
operation. Consult the consumer README for CSV columns and GPU pixel checks.

For optional Orin/Thor service deployment, persistent journals, clock pinning,
time synchronization, and rollback, see [deployment](argus_sender/deploy/README.md).
The target must already have a working Argus camera driver and camera initialization.

## Validation

See [sender testing](argus_sender/TESTING.md) for Jetson and hardware
commands. Component tests stay under `argus_sender/tests/` and
`camera_receiver/tests/`; only checks spanning components live under `tests/`.
From the example root, run the Python checks with pytest and PyYAML installed:

```bash
python3 -m pytest argus_sender/tests camera_receiver/tests tests -q
```

These tests cover configuration and protocol behavior; live capture,
NVENC/NVDEC, and CUDA IPC still require the target NVIDIA hardware and driver.

## Licensing

Project code uses Apache-2.0. The imported Argus sample helpers `Error.h` and
`EGLGlobal.h` retain their BSD-3-Clause notices. Jetson Multimedia API and Video
Codec SDK dependencies retain their own licenses and are supplied separately.
