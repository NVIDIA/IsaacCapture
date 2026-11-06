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
  `scripts/deploy_orin.py` builds the producer on an Orin from this checkout and runs it as a systemd service.
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

## Orin: LiveKit publisher

`camera_plugin_sensing` (built from [`app/`](app/), [`core/`](core/) and
[`livekit/`](livekit/)) captures the cameras with SIPL, encodes H.264 on the
hardware encoder and publishes one LiveKit video track per camera. The
[SENSING user guide](https://nvidia.github.io/IsaacCapture/main/device/camera/sensing.html)
has the build and run steps.

## Orin: deploy from the desktop with one script

`scripts/deploy_orin.py` builds the producer on an Orin from the checkout you run it in and runs it as the
`isaaccapture-producer.service` systemd service, so you do not build by hand on the Orin. It needs `ssh`, `scp` and
`git` on the desktop, and `python3`, CMake, g++, CUDA, `uv` and the SDK headers (see `scripts/setup.sh`) on the Orin.

Write a config naming the producer's own options (the LiveKit token file stays on the Orin; only its path is used):

```json
{"livekit_url": "ws://127.0.0.1:7880",
 "token_file": "/home/ido/livekit/join.token",
 "args": ["--camera-backend=sipl", "--link-masks=0x0000 0x1100",
          "--add-stream=sensor=0,livekit=sensing-left", "--add-stream=sensor=1,livekit=sensing-right"]}
```

Then run it from the repository root. The ssh user and password can come from the environment, which keeps the
password out of the command line and the output:

```bash
export ORIN_USER=ido
read -rs ORIN_PASSWORD && export ORIN_PASSWORD   # optional; without it ssh and sudo prompt
python3 src/plugins/sensing/scripts/deploy_orin.py 192.168.1.176 --backend sipl --config orin.json \
  --sdk-root /home/ido/sensing-sdk/root --replace-unmanaged
```

- `--backend sipl|argus` selects which capture SDK is built in. `--sdk-root DIR` is the Orin directory holding
  `usr/src/jetson_multimedia_api` and `usr/src/jetson_sipl_api` when they are not under `/usr/src`.
- `--source DIR` deploys another checkout (for example a git worktree of a feature branch) instead of the one holding
  the script; `--branch NAME` deploys that branch's committed files from it, ignoring the working tree. Without them
  the script deploys its own checkout, local edits included.
- `--replace-unmanaged` stops a `camera_plugin_sensing` that someone started by hand (its exact PIDs are shown);
  without it the deployment refuses and changes nothing. `--no-activate` builds and packages only, `--fetch-archive
  DIR` copies the package back, `--jobs N` bounds compile jobs, `--prefix DIR` changes the install root
  (default `/opt/isaaccapture`).
- The script checks the ssh connection first, then preflights the Orin, uploads only the source files that differ
  from what the Orin already holds, builds, installs a versioned release, replaces the running producer and waits for
  it to join the room and publish frames. Steps are numbered and the upload and packing show progress. The sudo steps
  prompt for the Orin password.
- A failed build or transfer leaves the running producer untouched. There is no automatic rollback; earlier releases
  stay under `/opt/isaaccapture/releases/`. Logs: `journalctl -u isaaccapture-producer.service -f` on the Orin. Stop
  it with `sudo systemctl stop isaaccapture-producer.service` before starting a producer by hand: the cameras are
  exclusive.
