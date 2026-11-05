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
the egocap backend, recording, the browser) runs on the PC; see `examples/egocap/LIVEKIT_PREVIEW.md`.

Prerequisites: an AGX Orin on JetPack 7.2.1 / L4T R39.2.1 with the SENSING drivers installed and the cameras
working, CUDA, CMake, and network access to the LiveKit server.

```bash
# 1. Once: vendor drivers, the Multimedia API and the SIPL headers (see scripts/setup.sh --help).
src/plugins/sensing/scripts/setup.sh
# 2. Build. The LiveKit SDK v1.12.4 prebuilt is downloaded and hash-checked; use -DLIVEKIT_SDK_DIR=... for a local copy.
# Flags that worked on the Orin: the viewer, examples, tests and Python bindings are not needed (the viewer
# needs libxkbcommon-dev), nvcc's headers are not on the host compiler's path, and libcuda must be linked
# explicitly. SIPL_API_DIR and JETSON_MMAPI_DIR point at the unpacked SIPL and Multimedia API trees.
cmake -B build -DBUILD_PLUGIN_SENSING=ON -DBUILD_VIZ=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF \
  -DBUILD_PYTHON_BINDINGS=OFF -DSIPL_API_DIR=... -DJETSON_MMAPI_DIR=... \
  -DCMAKE_CXX_FLAGS=-I/usr/local/cuda/include \
  -DCMAKE_EXE_LINKER_FLAGS="-Wl,--no-as-needed /opt/nvidia/l4t-gpu-libs/nvgpu/libcuda.so.1"
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
`--camera-backend=auto|sipl|argus` selects the capture backend once at startup. Only SIPL is built in today:
`argus` fails with a message saying so, and `auto` picks the only matching backend (and, with several, names the
override instead of guessing). Argus capture is not implemented yet and has never been run: there is no Argus SDK
or Argus camera available to the author.

`SENSING_TSC_PROBE=1` logs the SIPL capture counter beside the host clocks twice a second, as separate uncalibrated
samples. The counter's units are unconfirmed, so do not subtract or fit them.

Status: built and run on an AGX Orin by the author (both sensors, 2560x1984 at 30 fps, H.264; not reproduced in CI). If a build or run fails, fix the
failure at the file it names. `examples/egocap/LIVEKIT_PREVIEW.md` section 4 covers the PC side, and what recording from this
publisher still needs.
