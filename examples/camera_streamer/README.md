<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Isaac Camera Streamer — GMSL / Argus

Stream Argus-compatible GMSL stereo video from a Jetson Orin robot computer to an NVIDIA
workstation: **Argus → hardware HEVC → AVS1/TCP → NVDEC → CUDA IPC viewer**.
This standalone example runs independently of robot control software.

| Camera | Robot Install | Robot Preview | Robot Send | Host Receive | Host Viewer |
|---|---|---|---|---|---|
| Sensing SHW5G / S56 (GMSL) | Required: driver + native build | Optional: external Argus app | Jetson HEVC/TCP | NVDEC receiver | Stereo desktop view |

Complete common setup once, then follow the five GMSL sections below.
Network streaming uses three terminals: Robot Send, Host Receive, Host Viewer.

## Common setup — once per computer

### Identify the two network addresses

On **both computers**, run:

```bash
ip -brief -4 address
```

Choose the IPv4 address of the Ethernet, Wi-Fi, or VPN interface that connects
these two sites. Use the address without its `/prefix` suffix. Do not select
loopback (`127.0.0.1`) or a Docker-only interface. For a shared LAN, choose the
addresses on that LAN; for routed sites, choose mutually reachable addresses.

Replace the placeholders below and export both values in each terminal where
you run the network commands:

```bash
export HOST_IP="<Distributed Cloud workstation IPv4 address>"
export ROBOT_IP="<Robot Site Orin IPv4 address>"
```

`HOST_IP` always means the workstation and `ROBOT_IP` always means the Orin,
regardless of which computer runs the command. Exports do not carry into a new
terminal or SSH session; repeat them there. You can check routing from each side:

```bash
# On the Distributed Cloud workstation:
ip route get "${ROBOT_IP:?Set ROBOT_IP first}"
# On the Robot Site:
ip route get "${HOST_IP:?Set HOST_IP first}"
```

The receiver connects to the sender at `ROBOT_IP:5000`. The sender listens on
all local interfaces by default, so it does not need `HOST_IP` as an argument.

### Put the source on both computers

`~/isaac-camera-streamer` is a deployment directory. The copy commands below
create it explicitly; the robot does not need a separate repository there.
Choose **one** of the following approaches.

**Recommended for a disk-constrained robot: clone on the host, copy the example.**
The robot does not need a full Isaac Capture clone. Start with a host
checkout containing `examples/camera_streamer` (Git and Git LFS required).
The commands use `main` for the merged example. Before merge, reviewers must
substitute the pull request's head branch for `main`.

```bash
# On the Distributed Cloud workstation; skip cloning if you have this checkout.
export CAMERA_REPO_URL="https://github.com/NVIDIA/IsaacCapture.git"
export CAMERA_BRANCH="main"
GIT_LFS_SKIP_SMUDGE=1 git clone --single-branch --branch "$CAMERA_BRANCH" \
  "$CAMERA_REPO_URL" "$HOME/IsaacCapture"
cd ~/IsaacCapture
git lfs pull --include='examples/camera_streamer/**' --exclude=''
```

With SSH access configured and `rsync` installed on both computers, run this on
the **host**. Replace the username with the account you use on the robot:

```bash
export ROBOT_USER="<robot SSH username>"
ssh "${ROBOT_USER}@${ROBOT_IP:?Set ROBOT_IP first}" \
  'mkdir -p "$HOME/isaac-camera-streamer/camera_streamer"'
rsync -av --exclude='.venv*/' --exclude='build/' \
  --exclude='__pycache__/' --exclude='.pytest_cache/' \
  --exclude='*.egg-info/' --exclude='*.so' \
  "$HOME/IsaacCapture/examples/camera_streamer/" \
  "${ROBOT_USER}@${ROBOT_IP}:isaac-camera-streamer/camera_streamer/"
```

The trailing source slash copies the example's contents into the destination.
Run this transfer again to deploy source updates; it overwrites matching files,
so keep robot-specific profiles outside that directory or copy edits back first.
Virtual environments and native builds stay on their own machine: workstation
binaries cannot be reused on the ARM64 Orin.

**Alternative: clone directly on both computers.** On the robot, run the same
clone and LFS commands above using the same branch/revision as the host. No SSH
file transfer or `isaac-camera-streamer` directory is then needed. Use
`~/IsaacCapture/examples/camera_streamer` instead of the deployment path in the
Robot Site commands below. This approach uses more disk space for Git history
and the rest of the checkout.

### Test a clean robot deployment

Stop the sender and any local preview, and save custom profiles or captures outside
`~/isaac-camera-streamer` before deleting that directory. Repeating the source
transfer and Robot Install steps then builds the sender from scratch. This resets
the example's source, build products and any environments inside the directory;
it does not uninstall JetPack, camera drivers, system packages or PWM setup.

For a fuller check, use fresh terminals, enumerate cameras with `--list-modes`,
validate the shared profile with `--check-config`, run the native tests, then
start sender, receiver and viewer. Check both eyes, stop/restart the receiver to
exercise reconnect, and switch profiles after stopping all three processes.
A reboot followed by the documented vendor driver/PWM setup checks cold-start
behavior. A clean OS installation is a separate test of system prerequisites.

### Install the shared host receiver and viewer

Do this once, for any supported GMSL mode. The host needs Linux, an NVIDIA NVDEC-capable
GPU, CUDA development tools, CMake, a C++ compiler and `uv`. Argus and the camera vendor drivers are needed only on the robot. The native build downloads a pinned NVIDIA Video
Codec SDK archive from the source-configured NVIDIA artifact server.

```bash
cd ~/IsaacCapture/examples/camera_streamer
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e '.[receiver,test]'
cmake -S camera_receiver/codec -B camera_receiver/codec/build \
  -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build camera_receiver/codec/build --parallel 2
```

Create the environment only once. Reinstall dependencies/rebuild after relevant
source updates. Keep host and robot environments separate even when both are
named `.venv`.

### How to select and switch modes

All profiles live directly in `configs/`, with model-prefixed filenames.
Each device/resolution has one shared YAML. Set `PROFILE` to the filename without
`.yaml` (for example, `s56-stereo-2560x1984-30`), so `configs/${PROFILE}.yaml`
selects the file. Copy the same file to both computers and select the **same PROFILE on both computers**.
`network.port` and `streams[].id/type/output` define the common wire format.
`cameras`, `capture`, `runtime` and `alignment` control capture; `receiver`, `ipc`
and `streams[].receiver` control decoding and local IPC. Stream order determines
the AVS1 index. The receiver derives dimensions from `output` without a second copy.
Set `PROFILE` again in each new terminal; shell variables are not shared.
No Python copy/modify script is needed for the supplied profiles.

Before switching mode, stop the viewer, receiver and sender with
Ctrl+C. Then choose a new profile and restart those three processes. Stop local
previews before sending: they compete for camera ownership. All profiles use
TCP port 5000 and host IPC socket `/tmp/isaac-camera-streamer.sock`, so run one
pipeline at a time. TCP is unauthenticated; use a trusted LAN or VPN.

A sensor's maximum rate is not a guarantee of the complete capture/encode/network/display
rate. Link bandwidth, lighting, driver modes and hardware sync can limit it.

## GMSL cameras: Sensing

Stereo GMSL2 → Argus → Jetson hardware HEVC → AVS1/TCP. Dimensions are per eye.
The sender uses Argus device IDs and sensor modes for Sensing cameras, including
SHW5G and S56. Device models are not hard-coded in the sender.

### S56 stereo profiles

The supplied stereo profiles use sensor indices 0/1 on the S56 adapter.
Resolution and frame rate below apply to each eye.

| PROFILE | Output resolution | Frame rate | Capture mode / notes |
|---|---|---|---|
| `s56-stereo-2560x1984-30` | 2560×1984 | 30 fps | Mode 1, native dimensions; highest resolution exposed by this driver |
| `s56-stereo-1080p60` | 1920×1080 | 60 fps | Mode 0, native dimensions; highest frame rate exposed by this driver |

### S56 mono profiles

Use one S56 sensor to exercise the mono capture, HEVC, network and viewer path.
Here, mono means one color camera stream, not grayscale. The supplied profiles
select device 0; select device 1 instead to use the other eye.

| PROFILE | Output resolution | Frame rate | Capture mode / notes |
|---|---|---|---|
| `s56-mono-2560x1984-30` | 2560×1984 | 30 fps | Mode 1, native dimensions |
| `s56-mono-1080p30` | 1920×1080 | 30 fps | Mode 0, native dimensions |
| `s56-mono-1080p60` | 1920×1080 | 60 fps | Mode 0, native dimensions |
| `s56-mono-720p30` | 1280×720 | 30 fps | Mode 0, scaled from 1920×1080 |
| `s56-mono-720p60` | 1280×720 | 60 fps | Mode 0, scaled from 1920×1080 |
| `s56-mono-2400x1440-30` | 2400×1440 | 30 fps | Mode 1, center-cropped/scaled from 2560×1984 |

Set `PROFILE` to the same mono profile in the Robot Send and Host Receive commands
below. The viewer automatically displays the single `head` stream. No runtime
code or separate receiver configuration is needed. All mono profiles use 30 Mbps.

### SHW5G mono profiles

SENSING lists SHW5G as a monocular camera with 2560×1984 at 30 fps and
2400×1440 at 60 fps ([SHW5G specifications](https://www.sgwrd.com/products/3mp-global-shutter-monocular-camera-shw5g)).
The [S56 specifications](https://wiki.sensing-world.com/docs/6_1_Camera/Binocular_Camera/S56)
list 2560×1984 at 30 fps per eye. Argus modes depend on the installed driver;
vendor specifications do not define their numeric indices.

`configs/` contains one shared YAML for each published SHW5G mode:

| PROFILE | Output resolution | Frame rate | Capture mode / notes |
|---|---|---|---|
| `shw5g-mono-2560x1984-30` | 2560×1984 | 30 fps | Set mode from `--list-modes`; highest published resolution |
| `shw5g-mono-2400x1440-60` | 2400×1440 | 60 fps | Set mode from `--list-modes`; highest published frame rate |

Edit the desired profile in `configs/`: set `cameras[].device` for your rig,
and replace `mode: SET_FROM_LIST_MODES` with the matching Argus mode index.
Then copy the edited YAML to both computers and use its filename without `.yaml` as `PROFILE`
(for example, `shw5g-mono-2560x1984-30`).
The placeholder intentionally prevents starting a sender with a guessed mode.

The supplied S56 driver's 60 fps mode is 1920×1080, not 2400×1440. Its
2400×1440 mono profile uses the full-resolution sensor mode at 30 fps; it
exercises that output geometry without claiming the SHW5G 60 fps capture mode.
Smaller scaled outputs are not additional native sensor modes. Use
`--list-modes` as the source of truth for a particular camera/driver combination.

Stereo profiles use 30 Mbps per eye. At 60 fps, a hardware-triggered rig must also have its
vendor PWM/sync configured for 60 Hz; changing YAML alone does not change PWM.

### Robot Install

Complete the common source transfer first. Install the matching JetPack
Multimedia API/CUDA development headers, CMake and a C++ compiler. Install the
Sensing adapter's vendor driver and configure its PWM/sync for the selected rate.
These are rig-specific steps; the sender does not install drivers or change PWM.

```bash
cd ~/isaac-camera-streamer/camera_streamer
cmake -S argus_sender -B argus_sender/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build argus_sender/build --parallel 2
argus_sender/build/argus_sender --list-modes
```

Verify both eye devices and mode indices against the chosen shared YAML. The
build downloads pinned yaml-cpp. Build initially and after source changes; after
moving the source directory, use a fresh build directory rather than an old
CMake cache. Do not assume supplied rig calibration maps fit your camera.

### Robot Preview

Optional, with the sender stopped, in a terminal on the **Orin desktop**:

```bash
argus_camera -d 0
```

Exit it, then use `argus_camera -d 1` to inspect the other eye. This is NVIDIA's
separate Argus sample app; install/build it from the Jetson Multimedia API Argus
samples if absent. It shows one sensor locally and does not use the sender YAML
or transmit to the host. Its GUI controls select preview settings independently.
Exit the preview before sending. No host process is needed for this check.

### Robot Send

```bash
cd ~/isaac-camera-streamer/camera_streamer
PROFILE=s56-stereo-2560x1984-30 # Choose your resolution
argus_sender/build/argus_sender \
  --config "configs/${PROFILE}.yaml" --check-config
argus_sender/build/argus_sender \
  --config "configs/${PROFILE}.yaml"
```

The sender opens no preview and waits for the receiver before capturing.
`--check-config` validates configuration; it is not an end-to-end hardware test.

### Host Receive

After the shared host installation, run in the first host terminal:

```bash
cd ~/IsaacCapture/examples/camera_streamer
PROFILE=s56-stereo-2560x1984-30 # Choose your resolution
.venv/bin/python camera_receiver/camera_receiver.py \
  --config "configs/${PROFILE}.yaml" \
  --host "${ROBOT_IP:?Set ROBOT_IP first}"
```

This decodes the configured mono or stereo stream using NVDEC and publishes GPU frames through local CUDA
IPC. It does not open a display window.

### Host Viewer

In a second host terminal:

```bash
cd ~/IsaacCapture/examples/camera_streamer
.venv/bin/python cuda_ipc_consumer/consumer.py \
  --socket /tmp/isaac-camera-streamer.sock
```

Expect `head.left` and `head.right` for stereo, or `head` for mono. This command is not a
CloudXR launcher; headset display needs a compatible AVS1 CUDA IPC source in
camera_viz.
Close the window, press q/Esc with the window focused, or Ctrl+C in its terminal.

#### Display size and stream selection

`--scale` controls each image's display dimensions. The default is `1.0`, which
preserves the decoded resolution before window display. A 2560×1984 stream
produces a 2560×1984 image at `1.0`, or 1280×992 at `0.5`. This affects only the
local preview, not capture, encoding, transmission or the decoded CUDA IPC frames.
The desktop may fit the window to your screen; full-resolution stereo can exceed
its available space. Restart only the viewer to change these options.

`--streams` selects a comma-separated list of IPC stream names. Omit it to display
all streams. For the supplied profiles, stereo names are `head.left` and
`head.right`; mono uses `head`.

```bash
# Show both eyes at half size to fit a smaller screen.
.venv/bin/python cuda_ipc_consumer/consumer.py \
  --socket /tmp/isaac-camera-streamer.sock \
  --scale 0.5 --streams head.left,head.right

# Inspect one eye at full resolution (1.0 is also the default).
.venv/bin/python cuda_ipc_consumer/consumer.py \
  --socket /tmp/isaac-camera-streamer.sock \
  --scale 1.0 --streams head.left

# Display the single stream from a mono profile.
.venv/bin/python cuda_ipc_consumer/consumer.py \
  --socket /tmp/isaac-camera-streamer.sock --streams head
```

## Tuning and diagnostics

To add a camera mode, copy one shared YAML under a new device/resolution profile name.
Change `streams[].output` and choose the matching `cameras[].mode` from
`--list-modes`. Dimensions are shared; adjust the receiver FPS warning threshold and
configure hardware PWM for the capture rate. Save changes on the host and
transfer profiles before restarting the sender, receiver and viewer.

`bitrate_mbps` is Mbps per eye; `gop` is in frames. Sensor timestamps and timing
statistics are not by themselves a photon-to-photon latency measurement.

On a CUDA 13-only host, the optional CuPy pixel-mean diagnostic needs NVRTC 12:

```bash
cd ~/IsaacCapture/examples/camera_streamer
uv pip install --python .venv/bin/python nvidia-cuda-nvrtc-cu12
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
.venv/bin/python cuda_ipc_consumer/consumer.py \
  --socket /tmp/isaac-camera-streamer.sock --no-display --frames 120 --sample-mean
```

First-use kernel compilation can pause this diagnostic. Omit `--sample-mean`
when measuring steady frame delivery.

To record frame timing without opening a window, use the same consumer:

```bash
.venv/bin/python cuda_ipc_consumer/consumer.py \
  --socket /tmp/isaac-camera-streamer.sock --no-display \
  --frames 120 --latency-csv /tmp/camera_latency.csv
```

## Validation

Portable protocol/configuration tests can run on a workstation without Jetson libraries:

```bash
cd ~/IsaacCapture/examples/camera_streamer
cmake -S argus_sender -B /tmp/camera-protocol-build -DARGUS_SENDER_CPU_TESTS_ONLY=ON
cmake --build /tmp/camera-protocol-build --parallel 2
ctest --test-dir /tmp/camera-protocol-build --output-on-failure
```

```bash
cd ~/IsaacCapture/examples/camera_streamer
uv pip install --python .venv/bin/python -e ".[test]"
.venv/bin/python -m pytest tests
```

On the robot, run the native sender tests where the sender was built. The default
suite checks configuration, scheduling, alignment and AVS1/TCP behavior without
opening a camera:

```bash
cd ~/isaac-camera-streamer/camera_streamer
ctest --test-dir argus_sender/build --output-on-failure
```

For an opt-in stereo hardware test, stop the sender and local preview first.
Check the devices, sensor modes and PWM rate against the smoke-test YAML before
running it; pass a different YAML path if your camera setup differs. Install
PyYAML for the interpreter selected by CMake (`Python3_EXECUTABLE`) before this test:

```bash
cd ~/isaac-camera-streamer/camera_streamer/argus_sender
cmake -S . -B build \
  -DARGUS_SENDER_HARDWARE_TESTS=ON \
  -DARGUS_SENDER_HARDWARE_CONFIG="$PWD/configs/hardware_stereo_smoke.yaml"
cmake --build build --parallel 2
ctest --test-dir build -R '^avs1_hardware_smoke$' --output-on-failure -V
```

This receives 120 stereo bundles and checks sequence continuity, nonempty HEVC
payloads, increasing sensor timestamps when available, initial HEVC parameter
sets/keyframes, and clean sender exit. The receiver reads `network.port` from
the same YAML as the sender (default 5000); change the YAML to select another port.
Configuration checks alone do not validate a camera mode on hardware.

## Source layout and provenance

| Directory | Purpose |
|---|---|
| `argus_sender` | Native Argus capture, hardware HEVC and AVS1/TCP |
| `camera_receiver` | AVS1 parsing, NVDEC extension, GPU rings and CUDA IPC |
| `cuda_ipc_consumer` | Standalone viewer and timing diagnostics |
| `tests` | C++ unit tests, Python receiver/config tests and opt-in hardware tests |
| `configs` | Shared device/resolution profiles for sender and receiver |

The GMSL/Argus pipeline was adapted from NVIDIA camera sources at commit
`1fdb06d91ddb02ad527563cbacfc894a76d09126`. The original extraction and hardware-validation history are retained in
[PR #1358](https://github.com/NVIDIA/IsaacCapture/pull/1358).

This example is Apache-2.0 licensed. Existing upstream notices and external SDK
licenses remain in effect. The Argus-derived `EGLGlobal.h`, `EGLGlobal.cpp` and
`Error.h` retain their BSD-3-Clause license and notices. This example is not included in Isaac Capture wheels.

## Rebuilding after a source-directory move

CMake caches absolute source paths. After updating an existing deployment, configure
and build the native sender and decoder again at the paths shown above. Do not move
or reuse an old build directory from a different source location.

CUDA IPC protocol 3 requires consumers to hold each slot's shared file lock and
verify `slot_generation` before reading pixels, then finish GPU reads before
unlocking. The receiver holds an exclusive lock while writing. Delayed notifications
are discarded; a slow active read backpressures slot reuse rather than corrupting
pixels. Update receiver and consumer together; protocol 2 consumers are rejected.
