<!--
SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

SPDX-License-Identifier: Apache-2.0
-->

# Fast SENSING demo: Orin cameras, laptop server

Camera capture and H.264 encoding run on the **Orin** (the existing SIPL producer, `camera_plugin_sensing`). The
H.264 crosses SSH into a file on the **laptop**, where `capture-server` decodes it on the laptop's NVIDIA GPU and
serves JPEG to clients. The full explanation and limits are in
[`capture_server/sensing_sipl/README.md`](capture_server/sensing_sipl/README.md); this page is the run sequence.

**This demo uses SIPL, not Argus.** The producer is `camera_plugin_sensing` (SIPL) on the Orin and the laptop decodes
with NVDEC through `--sensing-config`. Nothing on this page uses Argus, the camera streamer or the `gmsl_ipc` feed.

**Not yet run on the rig.** Expect to debug the first attempt. Start with one sensor.

**One-command run:** after the SSH setup and the station config below,
`capture_server/sensing_sipl/run_fast_demo.sh` performs steps 3 to 6 (FIFO, pump, producer, server) and stops them all
on Ctrl+C. It reads `ORIN`, `FIFO`, `SPOOL`, `PRODUCER` and `STATION` from the environment. The manual steps below
are the same sequence.

## Before you start

| Where | Needs |
|---|---|
| Laptop | NVIDIA GPU + driver; `gst-inspect-1.0 nvh264dec h264parse videoconvert` finds all three; `cd examples/egocap && uv sync` |
| Orin | `camera_plugin_sensing` built and working; its platform config, config name and link masks |
| Both | `ssh -o BatchMode=yes ORIN true` works from the laptop with no prompt |

`ORIN` is the Orin's SSH host. Replace `PLATFORM.json`, `CONFIG_NAME` and `MASKS` with the values you capture with.

### SSH setup (laptop, once)

The pump runs `ssh` non-interactively, so key login must work without a prompt.

```sh
# Skip if ~/.ssh/id_ed25519 exists.
ssh-keygen -t ed25519
# Prompts once for the Orin user's password.
ssh-copy-id -i ~/.ssh/id_ed25519.pub USER@ORIN_IP
```

Add a short name to `~/.ssh/config` on the laptop, then use `orin` for `ORIN` everywhere below:

```text
Host orin
    HostName ORIN_IP
    User USER
```

Check: `ssh -o BatchMode=yes orin true` must return silently with exit status 0. A password prompt or
`Permission denied` means the key is not installed yet.

## 1. Orin: list sensors (once)

```sh
camera_plugin_sensing --platform-config=PLATFORM.json --config-name=CONFIG_NAME --link-masks='MASKS' --list-sensors
```

Note each `sensor=N`, its size and its fps.

## 2. Laptop: write the station config (once)

Copy `capture_server/sensing_sipl/sipl_remote_station.example.yaml` to `my_station.yaml`. Set `pipeline_id`, `width`,
`height`, `fps` from step 1 and a `spool` path. For one sensor, delete the second `external_sensors` entry and the
`cam1` camera.

## 3. Orin: make the FIFO

```sh
mkdir -p /dev/shm/sensing-demo && mkfifo /dev/shm/sensing-demo/sensor0.fifo
```

## 4. Laptop: start the pump (before the producer)

```sh
mkdir -p /var/tmp/sensing && : > /var/tmp/sensing/sensor0.h264
ssh -T -o BatchMode=yes -o Compression=no ORIN 'cat /dev/shm/sensing-demo/sensor0.fifo' >> /var/tmp/sensing/sensor0.h264
```

It waits until the producer opens the FIFO. Leave it running.

## 5. Orin: start the producer

```sh
camera_plugin_sensing --platform-config=PLATFORM.json --config-name=CONFIG_NAME --link-masks='MASKS' \
    --gop=15 --bitrate=20000000 \
    --add-stream=sensor=0,output=/dev/shm/sensing-demo/sensor0.fifo
```

## 6. Laptop: start the server

```sh
cd examples/egocap
uv run capture-server --sensing-config /path/to/my_station.yaml
```

Connect a client and request `cam0`. The first picture appears at the next keyframe (up to the 15-frame GOP).

## Stop

1. Orin: Ctrl+C the producer. The pump ends by itself; confirm it exited.
2. Laptop: Ctrl+C the server.
3. Cleanup: `rm -f /var/tmp/sensing/sensor0.h264` (laptop), `rm -f /dev/shm/sensing-demo/sensor0.fifo` (Orin).

The spool grows at the full encoder bitrate (about 150 MB/min at 20 Mbps) while no client is connected, so start
the pump and producer only when you are ready to connect.

## If nothing shows up

- Server reports the camera failed at once naming a file: the spool is missing or not a regular file (step 4).
- Camera fails after a few seconds: the pump or producer stopped, or the producer never opened the FIFO.
- Hardware decoder unavailable: re-check the three `gst-inspect-1.0` elements on the laptop.
- No picture but no error: wait one GOP; confirm the spool file size is growing (`ls -l`).
