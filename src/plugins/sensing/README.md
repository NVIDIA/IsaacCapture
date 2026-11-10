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

- [`scripts/`](scripts/): host setup, validation, diagnostics, the Bash
  compatibility Docker launcher, and `setup_orin_crosscompile.py` for building on a PC. Each entry point exposes `--help`.
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

## Orin: camera producer

`camera_plugin_sensing` (built from [`app/`](app/) and [`core/`](core/)) captures the cameras with SIPL, encodes H.264
on the hardware encoder and sends each camera's video over a direct TCP link to the PC video stage (`src/video_stage`),
which publishes it to LiveKit. The producer has no LiveKit client of its own. The
[SENSING user guide](https://nvidia.github.io/IsaacCapture/main/device/camera/sensing.html)
has the build and run steps.

`--camera-backend=auto|sipl|argus` selects the capture backend once at startup: an explicit choice must match or fail
with its own reason, and `auto` picks the only matching backend (with several it names the override instead of
guessing). SIPL is built by default (`-DSENSING_WITH_SIPL=OFF` for an Argus-only rig). The Argus backend
(`-DSENSING_WITH_ARGUS=ON`, Argus device index with `--add-stream=sensor=N,...`, `--argus-mode=K`) captures NV12 into
DMA-BUFs that the hardware encoder reads without a copy. **The Argus code has never been built or run: there is no
Argus SDK or Argus camera available to the author.** Argus DMA-BUF input cannot be combined with
`--scale-div` above 1. An Argus-only build (`-DSENSING_WITH_SIPL=OFF -DSENSING_WITH_ARGUS=ON`) also builds on JetPack 6.2 / L4T R36.4.3.

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

`sensing_mock_producer` (on a PC, configure with `-DBUILD_SENSING_CONTROL=ON`) runs the same channel, retry loop and direct sender over looped H.264 files, with
`--simulate-fault-seconds=N` to show the unplugged-camera path, so the PC side can be tried without an Orin.
