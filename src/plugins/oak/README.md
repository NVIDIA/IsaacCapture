<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OAK Camera Plugin

C++ plugin that captures H.264 video from OAK cameras and saves to raw H.264 files.

## Features

- **Hardware H.264 Encoding**: Uses OAK's built-in video encoder
- **Raw H.264 Recording**: Writes H.264 NAL units directly to file (no container overhead)
- **OpenXR Integration**: Optional CloudXR runtime integration
- **Self-contained build**: DepthAI built automatically via CMake

## Build

DepthAI v3.x is fetched and built automatically via FetchContent. Dependencies are managed
by **vcpkg**. The first build takes ~10-15 minutes (mostly vcpkg deps + DepthAI), subsequent
builds are fast.

### Prerequisites

```bash
# Install build tools for vcpkg to build libusb (Linux)
sudo apt install libudev-dev autoconf automake libtool pkg-config

# Install vcpkg (one-time)
git clone https://github.com/microsoft/vcpkg ~/.vcpkg
~/.vcpkg/bootstrap-vcpkg.sh
echo 'export VCPKG_ROOT=~/.vcpkg' >> ~/.bashrc && source ~/.bashrc
```

### Configure and Build

CMake reads `CMAKE_TOOLCHAIN_FILE` only on a build tree's **first** configure, so an
existing `build/` has to go — otherwise vcpkg is never picked up and configure fails.

```bash
cd IsaacTeleop

rm -rf build   # required if build/ was configured without CMAKE_TOOLCHAIN_FILE
cmake -B build -DBUILD_PLUGIN_OAK_CAMERA=ON \
    -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
cmake --build build --target camera_plugin_oak --parallel
```

## Usage

```bash
# Record a single color stream
./build/src/plugins/oak/camera_plugin_oak --add-stream=camera=Color,output=./color.h264

# Record multiple streams
./build/src/plugins/oak/camera_plugin_oak \
  --add-stream=camera=Color,output=./color.h264 \
  --add-stream=camera=MonoLeft,output=./left.h264 \
  --add-stream=camera=MonoRight,output=./right.h264

# Record with a live preview window
./build/src/plugins/oak/camera_plugin_oak \
  --add-stream=camera=Color,output=./color.h264 --preview

# Show help
./build/src/plugins/oak/camera_plugin_oak --help
```

Press `Ctrl+C` to stop recording.

### Record camera metadata in MCAP

Each captured frame emits metadata (sequence number, timestamps) as FlatBuffer
`FrameMetadataOak` messages. There are two ways to record it —
`--collection-prefix` and `--mcap-filename` are **mutually exclusive**.

#### 1. Local MCAP file (`--mcap-filename`)

The plugin writes per-stream metadata directly to an MCAP file. No host-side
tracker or TeleopSession is required.

```bash
./build/src/plugins/oak/camera_plugin_oak \
  --add-stream=camera=Color,output=./color.h264 \
  --add-stream=camera=MonoLeft,output=./left.h264 \
  --mcap-filename=./metadata.mcap
```

#### 2. TeleopSession (multi-device recording)

Record camera metadata in the same MCAP file as other teleop data (hands, head,
controllers, etc.):

1. **Plugin** — launch `oak_camera` via PluginManager or `TeleopSession`'s
   `PluginConfig`, passing `--collection-prefix` so the plugin pushes metadata
   via OpenXR `SchemaPusher`.
2. **Host trackers** — create one `FrameMetadataTrackerOak` **per stream**, each
   with collection id `{collection_prefix}/{StreamName}` using the **same** prefix
   the plugin was given. TeleopSession's DeviceIO layer uses the live tracker
   implementation to read the pushed tensors and write MCAP channels.
3. **MCAP config** — add every tracker to both `TeleopSessionConfig.trackers`
   and `McapRecordingConfig.tracker_names`, each under its own base name.

```python
from pathlib import Path

from isaaccapture.deviceio import FrameMetadataTrackerOak, McapRecordingConfig
from isaaccapture.teleop_session_manager import PluginConfig, TeleopSession, TeleopSessionConfig

PLUGIN_ROOT = Path("build/src/plugins")  # or your installed plugin search path
COLLECTION_PREFIX = "oak_camera"
STREAM_NAMES = ["Color", "MonoLeft"]

# One tracker per stream, each on the collection the plugin publishes it under.
oak_trackers = {
    name: FrameMetadataTrackerOak(f"{COLLECTION_PREFIX}/{name}")
    for name in STREAM_NAMES
}

config = TeleopSessionConfig(
    app_name="OakTeleop",
    pipeline=pipeline,  # your retargeting pipeline
    trackers=list(oak_trackers.values()),
    mcap_config=McapRecordingConfig(
        "recording.mcap",
        [(tracker, f"oak_metadata_{name.lower()}") for name, tracker in oak_trackers.items()],
    ),
    plugins=[
        PluginConfig(
            plugin_name="oak_camera",
            plugin_root_id="oak_camera",
            search_paths=[PLUGIN_ROOT],
            plugin_args=[
                "--add-stream=camera=Color,output=./color.h264",
                "--add-stream=camera=MonoLeft,output=./left.h264",
                f"--collection-prefix={COLLECTION_PREFIX}",
            ],
        ),
    ],
)

with TeleopSession(config) as session:
    while True:
        session.step()  # metadata is recorded each update
```

See also `examples/oxr/python/isaaccapture_examples/oxr/test_oak_camera.py --mode schema-pusher` for a
standalone PluginManager + DeviceIOSession example of the same SchemaPusher
flow.

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--add-stream=camera=<name>,output=<path>` | (at least one required) | Add a capture stream. `camera` is one of `Color`, `MonoLeft`, `MonoRight`. Repeatable. |
| `--fps=N` | 30 | Frame rate for all streams |
| `--bitrate=N` | 8000000 | H.264 bitrate (bps) |
| `--quality=N` | 80 | H.264 quality (1-100) |
| `--device-id=ID` | first available | OAK device MxId |
| `--preview` | off | Open a live SDL2 window showing the color camera feed |
| `--collection-prefix=PREFIX` | | Push per-frame metadata via OpenXR tensor extensions |
| `--mcap-filename=PATH` | | Record per-frame metadata to an MCAP file |

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────┐     ┌──────────────┐
│   OakCamera     │────>│    FrameSink     │────>│ RawDataWriter │────>│  .h264 File  │
│  (H.264 encode) │     │ (write + push)   │     │ (file writer) │     │              │
└─────────────────┘     └──────┬───────────┘     └───────────────┘     └──────────────┘
     core/                     │    core/                core/
                               v
                     ┌──────────────────┐
                     │ MetadataPusher   │
                     │ (OpenXR tensor)  │
                     └──────────────────┘
```

## Dependencies

Transitive dependencies via **vcpkg**, DepthAI and SDL2 via FetchContent:

- **DepthAI v3.x** - OAK camera interface (FetchContent)
- **SDL2** - Live preview window (FetchContent, used by `--preview`)
- **vcpkg** - nlohmann-json, spdlog, libusb, openssl, libarchive, etc.

## Output Format

The plugin writes raw H.264 NAL units (Annex B format) to `.h264` files. To play or convert:

```bash
# Play with ffplay
ffplay -f h264 recording.h264

# Convert to MP4
ffmpeg -f h264 -i recording.h264 -c copy output.mp4

# Convert with specific framerate
ffmpeg -f h264 -framerate 30 -i recording.h264 -c copy output.mp4
```

## Troubleshooting

### Device USB Connection

```bash
# Check OAK camera connection
lsusb | grep 03e7
```

For more connection troubleshooting, see the [OAK USB deployment guide](https://docs.luxonis.com/hardware/platform/deploy/usb-deployment-guide).

### Inspect the Recording

```bash
# Verify recording (convert to MP4 first)
ffmpeg -f h264 -i recording.h264 -c copy recording.mp4
ffprobe recording.mp4

# Check frame count
ffprobe -show_entries stream=nb_frames recording.mp4
```
