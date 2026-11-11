<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# camera_livekit_bridge

A downstream consumer of `camera_streamer`'s CUDA IPC output that publishes the
camera streams to LiveKit as pre-encoded H.264 tracks. It talks to
`camera_receiver` only through its IPC socket and imports nothing from egocap or
the Isaac Capture runtime; egocap joins the LiveKit room on its own.

```text
camera_receiver ──CUDA IPC──> bridge ──H.264──> LiveKit room ──> egocap page / recorder
                 (GPU NV12)    GPU copy or shrink, NVENC
```

Each stream runs on its own thread. A frame is copied (or box-averaged down) on
the GPU from the IPC slot into an NVENC input surface, encoded, and submitted as
a pre-encoded access unit. Pixels never reach host memory. The frame's source
sequence is the track's `frame_id`, the sensor start time (when the receiver
derived one) is the `user_timestamp` in Unix microseconds, and the sensor
counter is the 8-byte `user_data`. The newest picture is encoded and older
queued ones are skipped. The session also answers clock-sync requests that
arrive as LiveKit data messages.

## Build

```bash
cmake -S . -B build && cmake --build build -j
```

Needs the CUDA toolkit and nlohmann_json. The NVENC headers and the LiveKit C++
SDK are fetched at configure time; pass `-DLIVEKIT_SDK_DIR=<unpacked SDK>` and
`-DFETCHCONTENT_SOURCE_DIR_NV_CODEC_HEADERS=<unpacked headers>` to build
offline. The build is standalone and is not part of the repository's top-level
build.

## Run

With `camera_receiver` running (for example fed by `camera_streamer/mock_sender`)
and a LiveKit server reachable:

```bash
build/bridge/camera_livekit_bridge \
  --livekit-url ws://127.0.0.1:7880 --livekit-token-file publisher-token.txt \
  --socket /tmp/camera_receiver_avs1_cuda_ipc.sock \
  --stream head=sensing-head:2 --stream wrist1=sensing-wrist1 \
  --fps 30 --bitrate-mbps 8
```

`--stream NAME=TRACK[:DIV]` publishes the receiver's stream `NAME` as the
LiveKit track `TRACK`, shrunk by the integer `DIV` on the GPU (the size must
divide to even dimensions). Track names, the join token's identity and the
published sizes must match the streams file egocap is given. The token needs
publish permission. Every 5 s it prints each track's frame rate, keyframes,
skipped and rejected frames and mean encode time.

`--timestamp-basis` sets what each frame's `user_timestamp` means, and the consumer's streams file must name
the same basis:

| Basis | Value |
| --- | --- |
| `capture_unix_us` (default) | The sensor start-of-frame as Unix time on the sender's clock. |
| `monotonic_us` | This host's monotonic clock at the instant the picture is ready to encode (before encoding): frame-ready time, later than the sensor capture. |

`capture_unix_us` is used only when the receiver reports a realtime sender clock; a frame with no usable sensor
start-of-frame (missing, in the future, older than 5 s, or on another clock domain) carries no `user_timestamp`, so
a consumer that needs a capture time says so instead of using a guess. `capture_monotonic_us` is refused: it would
need a measured sender-to-host clock mapping with a stated uncertainty, which does not exist yet.

Measured with the mock sender on an RTX PRO 5000 against egocap's receiver and
recording-feed checks (30 fps, head and one wrist): the head at 2560x1984 and
8 Mbps and at 1280x992 both ran with no missing frames and a median
capture-to-receive time of 65-85 ms. The head at 20 Mbps built a backlog of
about 1.2 s, so keep the bitrate near 8 Mbps. The first frames after a consumer
subscribes can be 1-2 s old.

## Probe

```bash
build/probe/camera_ipc_probe --socket /tmp/camera_receiver_avs1_cuda_ipc.sock \
  --stream head --frames 30
```

It prints each stream's plane layout, then per frame the slot, generation,
stream sequence, session id, sensor start time and a checksum of the first
pixels. Slots are overwritten without notice when the publisher's ring wraps.
