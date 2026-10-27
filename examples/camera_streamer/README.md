<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Isaac Camera Streamer — Sensing / Argus

Stream Sensing S56 stereo video from a Jetson Orin robot computer to an NVIDIA
workstation: **Argus → hardware HEVC → AVS1/TCP → NVDEC → CUDA IPC viewer**.
This standalone example runs independently of robot control software.

| Camera | Robot Install | Robot Preview | Robot Send | Host Receive | Host Viewer |
|---|---|---|---|---|---|
| Sensing S56 stereo (GMSL2) | Required: driver + native build | Optional: external Argus app | Jetson HEVC/TCP | NVDEC receiver | Stereo desktop view |

Complete common setup once, then follow the five Sensing sections below.
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

`~/isaac-camera-streamer` is a deployment directory, not a repository or a
folder created by `launch.py`. The copy commands below create it explicitly.
Choose **one** of the following approaches.

**Recommended for a disk-constrained robot: clone on the host, copy the example.**
The robot does not need a full Isaac Capture clone. Start with a host
checkout containing `examples/camera_streamer`. Use the review branch below until this example is merged into `main`
(Git and Git LFS required):

```bash
# On the Distributed Cloud workstation; skip cloning if you have this checkout.
export CAMERA_REPO_URL="https://github.com/NVIDIA/IsaacCapture.git"
export CAMERA_BRANCH="jsepulveda/sensing-camera-streamer"
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

### Install the shared host receiver and viewer

Do this once, for either Sensing mode. The host needs Linux, an NVIDIA NVDEC-capable
GPU, CUDA development tools, CMake, a C++ compiler and `uv`. Argus and the Sensing drivers are needed only on the robot. The native build downloads a pinned NVIDIA Video
Codec SDK archive from the source-configured NVIDIA artifact server.

```bash
cd ~/IsaacCapture/examples/camera_streamer
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e '.[receiver,test]'
cmake -S televiz/camera_receiver/codec -B televiz/camera_receiver/codec/build \
  -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build televiz/camera_receiver/codec/build --parallel 2
```

Create the environment only once. Reinstall dependencies/rebuild after relevant
source updates. Keep host and robot environments separate even when both are
named `.venv`.

### How to select and switch modes

Each mode has a matched pair: `configs/PROFILE-sender.yaml` and
`configs/PROFILE-receiver.yaml`. Select the **same PROFILE on both computers**.
Set `PROFILE` again in each new terminal; shell variables are not shared.
No Python copy/modify script is needed for the supplied profiles.

Before switching mode, stop the viewer, receiver and sender with
Ctrl+C. Then choose a new profile and restart those three processes. Stop local
previews before sending: they compete for camera ownership. All profiles use
TCP port 5000 and host IPC socket `/tmp/isaac-camera-streamer.sock`, so run one
pipeline at a time. TCP is unauthenticated; use a trusted LAN or VPN.

Mode tables distinguish **tested here** from **candidate** profiles. A sensor's
maximum rate is not a guarantee of the complete capture/encode/network/display
rate. USB link speed, lighting, driver modes and hardware sync can limit it.

## Sensing S56

Stereo GMSL2 → Argus → Jetson hardware HEVC → AVS1/TCP. Dimensions are per eye.
These choices match sensor indices 0/1 on the test adapter; other rigs may differ.

| Goal | PROFILE | Resolution / rate per eye | Status |
|---|---|---|---|
| Highest resolution exposed by this S56 driver | `s56-stereo-2560x1984-30` | 2560×1984, 30 fps | Tested here; mode 1 |
| Highest frame rate exposed by this S56 driver | `s56-stereo-1080p60` | 1920×1080, 60 fps | Candidate; mode 0 enumerated, full path not yet tested here |

Both use 30 Mbps per eye. At 60 fps, a hardware-triggered rig must also have its
vendor PWM/sync configured for 60 Hz; changing YAML alone does not change PWM.

### Robot Install

Complete the common source transfer first. Install the matching JetPack
Multimedia API/CUDA development headers, CMake and a C++ compiler. Install the
Sensing adapter's vendor driver and configure its PWM/sync for the selected rate.
These are rig-specific steps; the sender does not install drivers or change PWM.

```bash
cd ~/isaac-camera-streamer/camera_streamer
cmake -S televiz/argus_sender -B televiz/argus_sender/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build televiz/argus_sender/build --parallel 2
televiz/argus_sender/build/argus_sender --list-modes
```

Verify both eye devices and mode indices against the chosen sender YAML. The
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
PROFILE=s56-stereo-2560x1984-30
# Alternative: PROFILE=s56-stereo-1080p60
televiz/argus_sender/build/argus_sender \
  --config "configs/${PROFILE}-sender.yaml" --check-config
televiz/argus_sender/build/argus_sender \
  --config "configs/${PROFILE}-sender.yaml"
```

The sender opens no preview and waits for the receiver before capturing.
`--check-config` validates configuration; it is not an end-to-end hardware test.

### Host Receive

After the shared host installation, run in the first host terminal:

```bash
cd ~/IsaacCapture/examples/camera_streamer
PROFILE=s56-stereo-2560x1984-30
# Alternative: PROFILE=s56-stereo-1080p60
.venv/bin/python televiz/camera_receiver/camera_receiver.py \
  --config "configs/${PROFILE}-receiver.yaml" \
  --host "${ROBOT_IP:?Set ROBOT_IP first}"
```

This decodes both eyes using NVDEC and publishes GPU frames through local CUDA
IPC. It does not open a display window.

### Host Viewer

In a second host terminal:

```bash
cd ~/IsaacCapture/examples/camera_streamer
.venv/bin/python televiz/cuda_ipc_consumer/consumer.py \
  --socket /tmp/isaac-camera-streamer.sock
```

Expect `head.left` and `head.right` in the desktop viewer. This command is not a
CloudXR launcher; headset display needs a compatible AVS1 CUDA IPC source in
camera_viz. See [viewer options](televiz/cuda_ipc_consumer/README.md).
Close the window, press q/Esc with the window focused, or Ctrl+C in its terminal.

## Tuning and diagnostics

To add a Sensing mode, copy a sender/receiver YAML pair under a new profile name.
Change `streams[].output` and choose the matching `cameras[].mode` from
`--list-modes`. Match the receiver dimensions and FPS warning threshold, and
configure hardware PWM for the capture rate. Save changes on the host and
transfer profiles before restarting the sender, receiver and viewer.

`bitrate_mbps` is Mbps per eye; `gop` is in frames. Sensor timestamps and timing
statistics are not by themselves a photon-to-photon latency measurement.

On a CUDA 13-only host, the optional CuPy pixel-mean diagnostic needs NVRTC 12:

```bash
cd ~/IsaacCapture/examples/camera_streamer
uv pip install --python .venv/bin/python nvidia-cuda-nvrtc-cu12
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
.venv/bin/python televiz/cuda_ipc_consumer/consumer.py \
  --socket /tmp/isaac-camera-streamer.sock --no-display --frames 120 --sample-mean
```

First-use kernel compilation can pause this diagnostic. Omit `--sample-mean`
when measuring steady frame delivery.

`launch.py` is an optional Python wrapper for build-sender, build-receiver,
sender, receiver and consumer. The direct commands above are the primary workflow.

## Validation

```bash
cd ~/IsaacCapture/examples/camera_streamer
.venv/bin/python -m pytest televiz/camera_receiver/tests
ctest --test-dir televiz/argus_sender/build --output-on-failure
```

Run the native sender tests on the robot where that sender was built.
See [VALIDATION.md](VALIDATION.md) for actual hardware results; candidate profiles
are not promoted to tested modes by configuration checks alone.

## Source layout and provenance

| Directory | Purpose |
|---|---|
| `televiz/argus_sender` | Native Argus capture, hardware HEVC, AVS1/TCP and sender tests |
| `televiz/camera_receiver` | AVS1 parsing, NVDEC extension, GPU rings, CUDA IPC and receiver tests |
| `televiz/cuda_ipc_consumer` | Standalone viewer and timing diagnostics |
| `configs` | Matched Sensing sender/receiver profiles |

`UPSTREAM_FILES.json` preserves the original repository, commit and hashes for
retained imports. The receiver private header moved from `codec/inc/` to `codec/`.
This review branch contains only the Sensing/Argus pipeline.

Explicit upstream license notices are preserved. Files without a license grant
remain `LicenseRef-NVIDIA-Internal-Review` in REUSE metadata. The owners have approved
public review of this Sensing-only extraction; maintainers should finalize the
license metadata before merging. This example is not included in Isaac Capture wheels.
