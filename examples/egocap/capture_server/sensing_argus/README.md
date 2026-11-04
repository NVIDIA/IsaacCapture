<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

SPDX-License-Identifier: Apache-2.0
-->

# SENSING cameras over Argus: Orin to PC bring-up

Streams the SENSING rig's cameras from the Orin with native libargus capture and hardware HEVC, over AVS1/TCP to the
PC, where `examples/camera_streamer`'s receiver decodes them with NVDEC and publishes them on a local CUDA IPC feed
that `capture-server --gmsl-ipc` serves as JPEG. Nothing on this page has been run on the rig: the profile's device
indices and sensor modes are placeholders until `--list-modes` has been read on the Orin.

```text
Orin: cameras -> argus_sender (HEVC) -> AVS1/TCP -> PC: camera_receiver (NVDEC) -> CUDA IPC -> capture-server -> browser
```

`shw5g-dual-mono-2560x1984-30.yaml` here describes two SHW5G cameras with one mono stream each (`cam0`, `cam1`),
HEVC, native geometry, no remap. A rig with other cameras needs other `streams` entries: one per mono camera, or a
`stereo` stream with two cameras. Use the same file on both computers.

## 1. Orin: build the sender and read the modes

Follow the camera streamer's Robot Install steps (`examples/camera_streamer/README.md`): JetPack multimedia headers,
the rig's vendor driver and PWM/sync, then

```bash
cd ~/isaac-camera-streamer/camera_streamer
cmake -S argus_sender -B argus_sender/build -DCMAKE_BUILD_TYPE=Release
cmake --build argus_sender/build --parallel 2
argus_sender/build/argus_sender --list-modes
```

Argus device indices are not `/dev/videoN`, and mode indices are not published by the sensor vendor: take both from
this output. Optionally look at one sensor locally first (`argus_camera -d 0`, then `-d 1`) and close it before sending.

## 2. Set the profile

In `shw5g-dual-mono-2560x1984-30.yaml`, set `cameras[].device` to the Argus device of each camera and replace each
`mode: SET_FROM_LIST_MODES` with the mode index whose resolution and frame rate you want. Change `output` only if the
chosen mode is not 2560x1984 at 30 fps; keep the native geometry for the first run. Copy the finished file to both
computers and use its path as `PROFILE_FILE` below. For a first run with one camera, delete the other stream block:
the sender opens every stream in the file, so a missing camera would stop it.

## 3. Orin: send

```bash
cd ~/isaac-camera-streamer/camera_streamer
PROFILE_FILE=/path/to/shw5g-dual-mono-2560x1984-30.yaml
argus_sender/build/argus_sender --config "$PROFILE_FILE" --check-config
argus_sender/build/argus_sender --config "$PROFILE_FILE"
```

`--check-config` validates the file; it does not open the cameras. The sender waits for the receiver before capturing.

## 4. PC: install once, receive, then look at the pictures

The PC needs Linux, an NVDEC-capable NVIDIA GPU, CUDA development tools, CMake, a C++ compiler and `uv`. Install the
receiver once, as the camera streamer's "Install the shared host receiver and viewer" section says:

```bash
cd ~/IsaacCapture/examples/camera_streamer
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e '.[receiver,test]'
cmake -S camera_receiver/codec -B camera_receiver/codec/build \
  -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build camera_receiver/codec/build --parallel 2
```

Then start the receiver:

```bash
cd ~/IsaacCapture/examples/camera_streamer
PROFILE_FILE=/path/to/shw5g-dual-mono-2560x1984-30.yaml
.venv/bin/python camera_receiver/camera_receiver.py --config "$PROFILE_FILE" --host "${ROBOT_IP:?Set ROBOT_IP first}"
```

In a second terminal, check each camera before involving the server:

```bash
.venv/bin/python cuda_ipc_consumer/consumer.py --socket /tmp/isaac-camera-streamer.sock
```

Confirm for `cam0` and `cam1`: which physical camera it is, the picture's orientation and colors, the dimensions, and
a steady frame rate. Rename the streams to the cameras' roles in the profile once that is established.

## 5. PC: serve them to clients

```bash
cd ~/IsaacCapture/examples/egocap
uv sync --extra gmsl
uv run capture-server --gmsl-ipc /tmp/isaac-camera-streamer.sock \
    --gmsl-camera 'left=cam0' --gmsl-camera 'right=cam1'
```

`--gmsl-camera NAME=STREAM` maps the name a client requests to a stream id in the profile; use the roles you found in
step 4 for the names. See "GMSL cameras from the camera streamer" in `../README.md` for how the feed is read and what
happens when the receiver restarts.

## What to record when you run it

The rig's `--list-modes` output, the chosen device and mode for each stream, the identity, orientation and color check
from step 4, the sustained frame rate the receiver logs, and whether `capture-server` delivers frames from both
cameras to a client. HEVC dependency ordering and bounded backpressure are the sender's and receiver's own tests; a
cable pull, a sender restart and a receiver restart are worth exercising once on the rig.
