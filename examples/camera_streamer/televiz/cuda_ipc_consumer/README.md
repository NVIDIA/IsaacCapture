<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sensing CUDA IPC viewer

Use the [host viewer instructions](../../README.md) after starting the receiver.
The viewer shares the host environment with the receiver. It uses CuPy to open
CUDA IPC rings and copies frames to host memory only for OpenCV display.

From `examples/camera_streamer`, useful variants are:

```bash
.venv/bin/python televiz/cuda_ipc_consumer/consumer.py \
  --socket /tmp/isaac-camera-streamer.sock --streams head.left,head.right --scale 0.35
.venv/bin/python televiz/cuda_ipc_consumer/consumer.py \
  --socket /tmp/isaac-camera-streamer.sock --no-display --frames 120
.venv/bin/python televiz/cuda_ipc_consumer/consumer.py \
  --socket /tmp/isaac-camera-streamer.sock --no-display --latency-csv /tmp/camera_latency.csv
```

Close the window, press q/Esc with it focused, or Ctrl+C in the terminal.
`--sample-mean` additionally checks pixels but needs matching NVRTC libraries;
see the top-level diagnostic instructions. First-use JIT is not transport latency.
Frame metadata includes slot generations and sender/receiver timing. CUDA handles
are local-machine resources, not network handles. This viewer is not a CloudXR
launcher; a headset needs a compatible CUDA IPC visualization application.
