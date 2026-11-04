<!--
SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

SPDX-License-Identifier: Apache-2.0
-->

# SENSING cameras on an Orin, Isaac Capture on a PC (SIPL)

The SENSING cameras are captured and hardware-encoded on the Orin by the existing SIPL producer
(`camera_plugin_sensing`, unchanged). Its H.264 crosses the network over SSH, one byte stream per sensor, into a
local file on the PC, which the server decodes on the PC's NVIDIA GPU and serves as JPEG like any other camera.

```text
Orin: camera_plugin_sensing -> one FIFO per sensor
        -> ssh (one session per sensor) ->
PC:   spool file per sensor -> capture-server (decode on the PC GPU) -> JPEG -> clients
```

**Nothing here has been run on the rig.** The pieces that are new and unverified on hardware are the producer writing
into FIFOs, a reader that attaches to a stream already in progress, and the PC decode pipeline. The operator owns the
producer and the SSH transports: the server never starts, inspects or stops them.

## What you need

- **PC:** Linux with an NVIDIA GPU and driver, and GStreamer with the nvcodec plugin. Check
  `gst-inspect-1.0 nvh264dec`, `gst-inspect-1.0 h264parse` and `gst-inspect-1.0 videoconvert` all find their element.
  The server's Python environment is the usual one: `cd examples/egocap && uv sync`.
- **Orin:** the SIPL producer already built and working, and the platform config, config name and link masks it is
  started with.
- **SSH** from the PC to the Orin that needs no prompt: key authentication, checked with `ssh -o BatchMode=yes ORIN true`.
- **Bandwidth:** each sensor sends its encoder bitrate; two sensors at the 40 Mbps default are about 80 Mbps before
  transport overhead. Start with one sensor and a lower `--bitrate`.

`ORIN` below is the Orin's SSH host. Use a trusted network: the video is encrypted by SSH, but the server's own
WebSocket is not (see the capture server README).

## 1. Read the sensor list on the Orin

Run the producer with the same platform arguments you will capture with, plus `--list-sensors`; it never starts
capture:

```sh
camera_plugin_sensing --platform-config=PLATFORM.json --config-name=CONFIG_NAME --link-masks='MASKS' --list-sensors
```

Each line gives a `sensor=N`, the sensor's size and its frame rate. Put those into
`sipl_remote_station.example.yaml` (copy it first), together with a spool path per sensor on the PC. For a first run,
keep one sensor.

## 2. PC: create the spools, then start the pumps first

The pump must be reading the FIFO before the producer opens it, so start the pumps before the producer. Each
spool is a regular file that the pump appends to; create it empty so the server finds it, then start its pump.

On the Orin, once per sensor, make its FIFO:

```sh
mkdir -p /dev/shm/sensing-demo
mkfifo /dev/shm/sensing-demo/sensor0.fifo
```

On the PC, one terminal per sensor, as one step each:

```sh
mkdir -p /var/tmp/sensing && : > /var/tmp/sensing/sensor0.h264
ssh -T -o BatchMode=yes -o Compression=no ORIN 'cat /dev/shm/sensing-demo/sensor0.fifo' >> /var/tmp/sensing/sensor0.h264
```

`-T` keeps the byte stream clean (no terminal), compression stays off, and the pump's own diagnostics go to its
terminal. The command waits until the producer opens the FIFO.

## 3. Orin: start the producer

One producer opens all the sensors you selected, one `--add-stream` per sensor, with the pipeline id from step 1:

```sh
camera_plugin_sensing --platform-config=PLATFORM.json --config-name=CONFIG_NAME --link-masks='MASKS' \
    --gop=15 --bitrate=20000000 \
    --add-stream=sensor=0,output=/dev/shm/sensing-demo/sensor0.fifo
```

Keep the 15-frame GOP: a reader that attaches mid-stream starts decoding at the next keyframe, so the GOP is also
how long a new session waits for its first picture.

## 4. PC: start the server

```sh
cd examples/egocap
uv run capture-server --sensing-config /path/to/your_sipl_remote_station.yaml
```

Then connect a client and request the camera names from the YAML. A session attaches at the end of each spool, so it
shows live video, not what was recorded while nobody was connected. If a spool is missing or is not a regular file, the
camera is reported failed at once with the file's name. If a pump or the producer stops, the picture stops advancing
and the camera is reported failed after a few seconds; a required camera then ends the session.

## 5. Stop, in this order

1. Stop the producer on the Orin (Ctrl+C). The FIFO closes, so each pump ends by itself.
2. Check that each pump has exited; stop any that has not (Ctrl+C in its terminal).
3. Remove the spools on the PC and the FIFOs on the Orin when you are done:

```sh
rm -f /var/tmp/sensing/sensor0.h264        # on the PC
rm -f /dev/shm/sensing-demo/sensor0.fifo   # on the Orin
```

**The spool grows with every byte the pump delivers, whether or not a client is connected.** Disk blocks are handed
back only while a session is reading, so with no client attached a spool grows at the full encoder bitrate (about
150 MB per minute at 20 Mbps). Start the pumps and the producer when you are ready to connect, and stop them after.

## Limits of this slice

- No reconnect: a pump or producer failure fails the camera, and the next session needs a fresh producer and pumps.
- Timestamps are the PC's, taken when a frame is selected; the SIPL sensor time and camera synchronisation are not
  carried by bare H.264.
- The producer is not supervised from the server, so a crashed SSH session or producer is noticed only as a stream
  that stops advancing.
- No Windows support and no plain-TCP transport; SSH is the only one.
