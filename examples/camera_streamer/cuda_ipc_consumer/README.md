<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# camera_receiver CUDA IPC consumer

This example connects to `camera_receiver`'s AVS1 CUDA IPC socket, subscribes to
selected streams, opens the shared GPU buffers with CuPy, and optionally
displays them with OpenCV.

## Setup

Set up `camera_receiver` separately from its directory:

```bash
cd ../camera_receiver
./setup.sh
```

Set up this consumer from this directory:

```bash
cd ../cuda_ipc_consumer
./setup.sh
```

Each directory has its own `.venv`; the consumer does not use the
`camera_receiver` Python environment.

## Run

Start the receiver:

```bash
cd ../camera_receiver
./run.sh --host <sender-ip> --port 5000
```

Run the consumer from this directory:

```bash
./run.sh
```

Subscribe to only a subset of streams:

```bash
./run.sh --streams head.left,head.right
```

All subscribed streams share one window, one tile per stream in the receiver's
order. `--columns N` sets the tiles per row; the default is a near-square grid.
Without `--scale`, streams are shown at native size when every stream is at most
1280 pixels wide, and at 0.35 otherwise. `--no-labels` hides the stream names.

A stream with no frame for `--stale-after` seconds (default 1) is drawn as a
`NO SIGNAL` tile of its own size instead of its last image, so a stopped camera
cannot be mistaken for live video. A stream the receiver announced that never
delivered gets the same tile. The image returns when frames resume.

With a receiver in [discovery mode](../camera_receiver/README.md#stream-discovery)
the consumer may start before the sender: it shows `WAITING FOR STREAMS`, then
follows the receiver's `streams` messages as the layout appears or changes.

For a headless smoke test:

```bash
./run.sh --no-display --frames 120
```

To collect per-frame latency metadata as CSV while still printing the normal
one-second stats:

```bash
./run.sh --no-display --latency-csv /tmp/camera_latency.csv
```

For a CSV-only latency capture:

```bash
.venv/bin/python latency_check.py --streams head.left --output /tmp/camera_latency.csv
```

The default CUDA IPC socket is:

```text
/tmp/camera_receiver_avs1_cuda_ipc.sock
```

Override it with:

```bash
./run.sh --socket /tmp/camera_receiver_avs1_cuda_ipc.sock
```

The shared buffers are GPU-resident `uint8` arrays. Supported layouts are:

```text
rgba8: (height, width, 4)
nv12:  ((height * 3) / 2, width)
```

The example copies frames to host memory only for OpenCV display. CUDA-capable
consumers can run kernels directly on the opened CuPy arrays. Frame
notifications include `slot_generation`; consumers can detect that a slot was
overwritten if a generation value skips.

Frame notifications include AVS1 metadata from the sender, including
`bundle_sequence`, `stream_sequence`, `frame_number`, SOF TSC values, sender
timing fields, receiver timing fields, and rolling latency values when the
sender and receiver clocks are comparable. `sof_delta` is signed
`right_sof_tsc_ns - left_sof_tsc_ns` for stereo streams in the sender's Tegra
TSC clock domain.

## Overrides

Use a different Python interpreter:

```bash
CUDA_IPC_CONSUMER_PYTHON=/path/to/python ./run.sh
```

Use a different bootstrap Python when creating the consumer venv:

```bash
CUDA_IPC_CONSUMER_BOOTSTRAP_PYTHON=/path/to/python3 ./setup.sh
```
