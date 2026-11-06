<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SENSING GMSL cameras

SENSING connects GMSL cameras to Jetson through the SG8A carrier. This
integration uses NVIDIA SIPL for camera access and provides the tooling to
provision and validate the rig on the host or in a container.

Start with the [SENSING user guide](https://nvidia.github.io/IsaacCapture/main/device/camera/sensing.html)
for supported hardware, installation, container usage, and troubleshooting.
That guide owns the operating instructions; this README maps the source tree.

- [`scripts/`](scripts/): host setup, validation, diagnostics, and the Bash
  compatibility Docker launcher. Each entry point exposes `--help`.
- [`docker/`](docker/): example image, image-side group configuration, and
  host-side container setup.
- [`configs/`](configs/): the platform configuration and its provenance.

## Platform configuration provenance

[`configs/shw5g.json`](configs/shw5g.json) is a byte-identical copy supplied
by SENSING for the two-SHW5G rig, from:

```text
SG8A_AGON_G2Y_A1_AGX_ORIN_S56C_SHW5G_SHF3L_SIPL_JP7.2.1_L4TR39.2.1
  /query/sg8a_agth_g2a/shw5g.json
```

The public driver package includes only the mixed-camera configurations,
`s56c_shf3l_mixed.json` and `shw5g_shf3l_mixed.json`, not this `SHW5G_2`
configuration. Vendoring it makes validation independent of a downloaded
vendor package.

Keep the file byte-identical to the vendor copy. Compare new vendor drops
before replacing it; do not hand-edit the board wiring. Its CSI ports, I2C
buses, deserializer addresses, GPIOs, and link modes must match the installed
SIPL drivers.

## Orin: build and run the LiveKit publisher

The application is **`camera_plugin_sensing`**. On the Orin it does one job: capture the GMSL cameras with SIPL,
encode H.264 on the hardware encoder, and publish one LiveKit video track per camera. Everything else (CaptureServer,
the egocap backend, recording, the browser) runs on the PC.

Prerequisites: an AGX Orin on JetPack 7.2.1 / L4T R39.2.1 with the SENSING drivers installed and the cameras
working, CUDA, CMake, and network access to the LiveKit server.

```bash
# 1. Once: vendor drivers (see scripts/setup.sh --help). The Multimedia API and SIPL header trees are
#    separate prerequisites that setup.sh does not fetch; CMake looks for them as described below.
src/plugins/sensing/scripts/setup.sh
# 2. Build. The LiveKit SDK v1.12.4 prebuilt is downloaded and hash-checked; use -DLIVEKIT_SDK_DIR=... for a local
#    copy (it needs LICENSE and NOTICE files in that directory, or -DLIVEKIT_SDK_LICENSE_FILE / -DLIVEKIT_SDK_NOTICE_FILE).
# The viewer, examples, tests and Python bindings are not needed (the viewer needs libxkbcommon-dev).
# SIPL_API_DIR and JETSON_MMAPI_DIR point at the unpacked SIPL and Multimedia API trees.
cmake -B build -DBUILD_PLUGIN_SENSING=ON -DBUILD_VIZ=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF \
  -DBUILD_PYTHON_BINDINGS=OFF -DSIPL_API_DIR=... -DJETSON_MMAPI_DIR=...
cmake --build build --target camera_plugin_sensing -j
# 3. Check the cameras are visible.
build/src/plugins/sensing/app/camera_plugin_sensing --list-sensors
# 4. Publish. The token file is a LiveKit token for the room and publisher identity named in the PC's streams file.
build/src/plugins/sensing/app/camera_plugin_sensing \
  --livekit-url=ws://LIVEKIT_HOST:7880 --livekit-token-file=join.token \
  --add-stream=sensor=0,livekit=sensing-left --add-stream=sensor=1,livekit=sensing-right
```

The `livekit=` names must equal the `track` names in the PC's streams file. Stop with Ctrl-C. `--help` lists every
option (`--livekit-fps`, `--livekit-bitrate`, `--livekit-scale-div`, `--platform-config`, `--link-masks`).
`--camera-backend=auto|sipl|argus` selects the capture backend once at startup: an explicit choice must match or fail
with its own reason, and `auto` picks the only matching backend (with several it names the override instead of
guessing). SIPL is built by default (`-DSENSING_WITH_SIPL=OFF` for an Argus-only rig). The Argus backend
(`-DSENSING_WITH_ARGUS=ON`, Argus device index with `--add-stream=sensor=N,...`, `--argus-mode=K`) captures NV12 into
DMA-BUFs that the hardware encoder reads without a copy. **The Argus code has never been built or run: there is no
Argus SDK or Argus camera available to the author.** Argus DMA-BUF input cannot be combined with
`--livekit-scale-div` above 1.

## Management channel and camera faults

A camera that cannot start (no power cable, no GMSL link) does not end the producer. It reports the problem in plain
words, retries every few seconds (`--retry-seconds`), and starts streaming once the cable is fixed. The same process
can offer a WebSocket management channel, off unless `--control-token-file=PATH` names a file holding a shared secret:

```bash
camera_plugin_sensing ... --control-token-file=/etc/isaaccapture/control.token   # listens on 0.0.0.0:5610
```

Clients send one JSON object per message and get one reply; the producer also pushes `state_changed` and `error`
events. `get_status` reports the state (`starting`, `streaming`, `camera_fault`, `error`, `stopping`), the fault with
a code and what to check, each stream's frame counts, and where its direct link goes. `set_direct_target` sends a
stream's encoded video to a new PC address and port from now on (an empty host turns the direct link off); the choice
is remembered across camera restarts but not across a producer restart, where the command line applies. Every bad
request gets an error reply with a code and a sentence. The token goes in `Authorization: Bearer ...` (or `?token=`).

```bash
src/sensing_control/tools/orin_control.py ORIN_HOST --token-file control.token status
src/sensing_control/tools/orin_control.py ORIN_HOST --token-file control.token set-target sensing-left 192.168.1.205 5600
```

`sensing_mock_producer` runs the same channel, retry loop and direct sender over looped H.264 files, with
`--simulate-fault-seconds=N` to show the unplugged-camera path, so the PC side can be tried without an Orin.

`SENSING_TSC_PROBE=1` logs the SIPL capture counter beside the host clocks twice a second, as separate uncalibrated
samples. The counter's units are unconfirmed, so do not subtract or fit them.

Status: built and run on an AGX Orin by the author (both sensors, 2560x1984 at 30 fps, H.264; not reproduced in CI).
A failed build or run reports the file or dependency check that failed.

### Header trees the build needs

CMake searches for them when the `-D` variables are not given:

- `JETSON_MMAPI_DIR`: the Jetson Multimedia API tree, containing `include/NvVideoEncoder.h` and
  `samples/common/classes/`. Searched in `/usr/src/jetson_multimedia_api`.
- `SIPL_API_DIR`: the Jetson SIPL API tree, containing `sipl/include/NvSIPLCamera.hpp` (the
  `Jetson_SIPL_API_R<release>_aarch64.tbz2` package published with the L4T release). Searched in
  `/usr/src/jetson_sipl_api`, `/opt/nvidia/jetson_sipl_api` and `$SIPL_API_DIR`.

The steps to download and unpack both trees are not documented in this repository yet; the user guide is the
place for them.
