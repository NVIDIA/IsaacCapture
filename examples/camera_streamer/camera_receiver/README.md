<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# camera_receiver

Standalone AVS1 multi-stream receiver for the two-process camera path:

1. Thor `argus_sender` sends AVS1 bundles over TCP.
2. `camera_receiver` connects to the sender, decodes HEVC streams or copies raw
   NV12 streams into GPU-resident rings.
3. Same-host consumers open selected streams through CUDA IPC.

## Integration boundary

The receiver and its CUDA IPC publication are temporary integration that connects the standalone streamer to local
consumers (`cuda_ipc_consumer/` and the LiveKit bridge's `ipc_client`). The boundary is intended to converge on a
shared camera publisher: a catalog of streams, leases that say when a reader has finished with a buffer, encoded
stream profiles, and a receiver backend for remote sources. Decoding (`codec/`) and the mock sender remain useful;
the duplicate handoff is removed only after its consumers use the shared boundary.

| Current path | Intended owner or replacement | Reuse, adapt or retire | Follow-up and landing condition |
| --- | --- | --- | --- |
| `camera_receiver.py`, `cuda_ipc_consumer/`, `examples/camera_livekit_bridge/ipc_client/` | A camera publisher with a catalog, leases, encoded profiles and a receiver backend | Adapt the receiver; retire the bespoke handoff | Move discovery, frame access and completion-based release to the shared reader and receiver contracts. Lands when existing consumers read safe frames and encoded output with matching identity and timing, and the handoff has no remaining users. |
| `codec/`, `examples/camera_streamer/mock_sender/` | Decoder behind the publisher; a synthetic source for hardware-free tests | Reuse | Carry the tested decode behaviour and mock fixtures forward. No retirement planned. |

## Setup

From this directory:

```bash
./setup.sh
```

The setup script creates:

```text
.venv/
codec/_camera_receiver_codec*.so
```

It installs the Python dependencies from `requirements.txt` and builds the
local native decoder from `codec/`.

## Run

Start the receiver:

```bash
./run.sh --host <sender-lan-ip> --port 5000
```

The default endpoint is `127.0.0.1:5000`. Override `receiver.host` with
`--host` or `CAMERA_RECEIVER_HOST`. Run this process on the receiving Linux GPU
host; see the [environment and quick start](../README.md). Stop the foreground
receiver with Ctrl-C/SIGTERM. Setup does not create a receiver system service.

The default CUDA IPC socket is:

```text
/tmp/camera_receiver_avs1_cuda_ipc.sock
```

`receiver.gpu_id` is an explicit CUDA device ID. `0` means CUDA device 0. To use
the current CuPy/CUDA device instead, omit `receiver.gpu_id` from the config or
set it to `null`.

## Stream Config

The default config matches a sender with one stereo `head` stream and two mono
raw-NV12 wrist streams:

```yaml
streams:
  - index: 0
    name: head
    type: stereo
    width: 2560
    height: 1984
    output_format: nv12
    ring_depth: 6
    ipc:
      left: head.left
      right: head.right

  - index: 1
    name: wrist1
    type: mono
    width: 640
    height: 480
    output_format: nv12
    ring_depth: 6
    ipc:
      mono: wrist1
```

`index` maps to the AVS1 stream descriptor index. AVS1 descriptors do not carry
the sender YAML `id`, so the receiver config supplies user-facing names and IPC
stream names.

`output_format` can be:

```text
nv12   -> ((height * 3) / 2, width) uint8
rgba8  -> (height, width, 4) uint8
```

For raw AVS1 NV12 streams, `nv12` avoids color conversion and copies the TCP
payload into the GPU IPC ring. For HEVC streams, `nv12` asks NVDEC to write its
decoded NV12 surface into the IPC ring; `rgba8` converts to RGBA before publish.
The receiver applies the AVS1 stream descriptor's 0/180-degree presentation
rotation during that existing output write. HEVC NV12 rotation replaces the
normal NVDEC-to-ring device copy with a rotation kernel; RGBA rotation is fused
into color conversion, so neither path adds a second full-frame pass.

`ring_depth` controls how many slots each physical IPC stream owns. Frame
notifications include `slot_generation`; consumers can detect if a slot was
overwritten before they consumed it.

## Downstream consumers

Consumers subscribe to the GPU rings through the local Unix socket. The
[example consumer](../cuda_ipc_consumer/README.md) supports display, headless
checks and latency CSV output. Other applications can use its `CudaIpcClient`.

The default head frames remain distorted at 2560x1984; presentation rotation is
applied while writing the IPC output. The default wrist frames arrive at 640x480,
already remapped by the sender, so consumers must not apply the wrist remap again.
See the bundled [remap geometry and calibration](../argus_sender/remaps/README.md).

Notifications include frame sequence, slot generation, source timing and local
publish timing. Validate frame identity and generation when associating frames
with metadata. The rings are bounded live buffers, not a durable recording queue.

## Latency

The receiver prints rolling decode/copy and IPC publish stats every five
seconds by default. Configure the interval under `receiver`:

```yaml
stats_log_interval_s: 5.0
```

Set `stats_log_interval_s: 0` to disable only this recurring stats summary.
Startup, connection, warning, error, and other one-time logs are unaffected.
The equivalent command-line override is `--stats-log-interval-s`; for example,
`--stats-log-interval-s 10` prints the summary every ten seconds.

Low ingress FPS uses an independent warning-level monitor. It counts each
configured logical AVS stream after a complete bundle is received and parsed,
before decode/copy work is queued. Configure its threshold and window under
`receiver`:

```yaml
receive_fps_warning_threshold: 55.0
receive_fps_warning_interval_s: 10.0
```

Every warning interval, streams below the target are reported individually,
for example `head=52.4fps`. A connected stream that receives no complete frames
is reported as `0.0fps`. Disconnect time is excluded because the existing
connection/retry logs cover that state. Set `receive_fps_warning_threshold: 0`
to disable these warnings. This monitor remains active when
`stats_log_interval_s` is `0` because its output is warning-level, not part of
the periodic INFO summary.

The receiver processes logical AVS streams independently:

- the TCP reader must receive a complete AVS1 bundle first;
- each present logical stream is queued immediately after parse;
- different logical streams process in parallel;
- stereo eyes within one stream process in parallel;
- a stream is published as soon as its own samples are ready.

AVS1 sender wall-clock timing fields are used for capture/process/TCP latency
when sender and receiver clocks are synchronized and report the same clock
domain. AVS1 SOF TSC values are Tegra TSC-domain values, not wall-clock
timestamps; `sof_delta` is meaningful for stereo alignment, but absolute
SOF-to-consumer latency needs an additional sender TSC-to-wall-clock
calibration path.

## Overrides

Use another config:

```bash
./run.sh configs/receiver.yaml
```

Pass receiver flags after the optional config path:

```bash
./run.sh configs/receiver.yaml --log-level DEBUG --stats-log-interval-s 10
```

Override the sender endpoint without editing YAML:

```bash
CAMERA_RECEIVER_HOST=<sender-lan-ip> CAMERA_RECEIVER_PORT=5000 ./run.sh
```

Use a different Python interpreter:

```bash
CAMERA_RECEIVER_PYTHON=/path/to/python ./run.sh
```

Use a different bootstrap Python when creating the venv:

```bash
CAMERA_RECEIVER_BOOTSTRAP_PYTHON=/path/to/python3 ./setup.sh
```

## Tests

From this component directory, run the protocol, reconnect and mocked decoder
tests with pytest and PyYAML installed:

```bash
python3 -m pytest tests -q
```

These tests require neither a GPU nor the sender or consumer source folders.
