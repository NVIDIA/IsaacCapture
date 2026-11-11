<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# camera_livekit_bridge

A downstream consumer of `camera_streamer`'s CUDA IPC output that publishes the
camera streams to LiveKit. It talks to `camera_receiver` only through its IPC
socket and imports nothing from egocap or the Isaac Capture runtime; egocap
joins the LiveKit room on its own.

Status: the IPC client and a probe exist. Encoding and publishing are not
implemented yet.

## Build

```bash
cmake -S . -B build && cmake --build build -j
```

Needs the CUDA toolkit and nlohmann_json. The build is standalone and is not
part of the repository's top-level build.

## Probe

With `camera_receiver` running (for example fed by `camera_streamer/mock_sender`):

```bash
build/probe/camera_ipc_probe --socket /tmp/camera_receiver_avs1_cuda_ipc.sock \
  --stream head --frames 30
```

It prints each stream's plane layout, then per frame the slot, generation,
stream sequence, session id, sensor start time and a checksum of the first
pixels. Slots are overwritten without notice when the publisher's ring wraps.
