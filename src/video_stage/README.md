<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Video stage

GPU processing of camera video on the PC: decoded frames stay in GPU memory from the hardware decoder, through one
GPU operation, to the hardware encoder. This directory holds the operation interface and the example operation. The
decoder, encoder and transport sit beside it and do not change when the operation is replaced.

## The operation contract

`core/inc/video_stage/gpu_operation.hpp` defines `GpuOperation`: NV12 in, NV12 out, fixed geometry.

- `prepare(input_geometry)` runs once, outside the frame loop. Build and upload anything per-geometry here (a
  rectification map, for example). Nothing in `enqueue` allocates or uploads.
- `enqueue(in, out, stream)` only queues kernels on `stream` and returns. It does not synchronise the device or the
  stream. After it returns, the caller records a CUDA event on `stream`; whoever reads `out` waits on that event.
- `in` and `out` are different surfaces (the operation is not in-place). `out` has `output_geometry(in.geometry)` and
  keeps the input's color description. Both are `Nv12View`s: device pointers, byte pitches, size.
- Frame identity and timestamps are not the operation's concern; they travel with the frame.

## The example: `SineWarp`

`core/sine_warp.cu` is a horizontal sine ripple, `source_x = x + A*sin(2*pi*y/wavelength + phase)`,
`source_y = y`, sampled bilinearly, with the border fill for the declared range (Y=16 limited, 0 full; chroma 128).
Amplitude 0 is an identity. It is an example, not calibrated rectification: the warped picture does not keep the
source camera's projection.

The stage assumes the decoder's NV12 uses the H.264 default chroma siting described below; a stream that signals a
different siting must be rejected at the decoder, not warped with the wrong sample positions.

Chroma is not sampled at the luma coordinate divided by two. A chroma sample of H.264 NV12 sits on an even luma column
and half way between two luma rows, so the kernel maps that luma-space position, then converts back to chroma-plane
coordinates. Keep that when substituting an operation.

## Substituting the fisheye rectification

Replace the mapping, keep the rest: a class deriving from `GpuOperation` whose `prepare` generates the destination-to-
source maps once (on the CPU, uploaded once) and whose `enqueue` launches a kernel that reads `map_x/map_y` instead of
computing the sine. Sample luma at the map position and chroma at the position derived from the same map with the
siting above. `output_geometry` returns the rectified output size. Nothing else in the stage changes.

## The worker and the data path

```text
Orin producer --DATA (AVS1 records, H.264)--> PC worker: NVDEC -> GpuOperation -> NVENC --> LiveKit track
              <--CTRL (keyframe requests, clock exchange)--
```

By default the worker does no processing: `--operation=none` forwards the Orin's encoded access units to LiveKit untouched,
with no decode, encode or GPU, for the lowest latency and no quality loss. `--operation=sine-warp` turns on the GPU
stage below (decode, the example warp, re-encode); it is a test of the stage and the template for the rectification.

`video_stage_worker` serves one Orin camera per `--stream=PORT:TRACK`; all tracks are published under one LiveKit
participant, as the Orin producer does for its cameras. The recorder and the web page subscribe to the processed tracks
exactly as they subscribe to the Orin's own, so they are unchanged.

- **Direct input.** The Orin producer connects to each stream's port (`--add-stream=sensor=0,livekit=NAME,direct=HOST:PORT`)
  and sends every encoded access unit as an AVS1 record over TCP (`wire/`; codec 2 is H.264). It reconnects by itself,
  starts at a keyframe, and drops frames up to the next keyframe rather than block when the link cannot keep up. Use a
  wired link.
- **Control link.** A second connection from the Orin carries the PC's keyframe requests and a four-timestamp clock
  exchange (the same bytes the recorder uses over LiveKit). The worker converts each Orin frame-ready stamp onto its own
  monotonic clock before publishing, and answers the recorder's and the page's clock sync with that clock, so they map one
  clock. Frames are withheld while the Orin-to-PC mapping is missing or less certain than
  `--max-upstream-uncertainty-ms` (5 by default); when the Orin's clock jumps, the track is unpublished and published again
  at the next keyframe. The uncertainty the recorder and the page report covers only the worker-to-them leg: it excludes
  the Orin-to-PC mapping, which this budget bounds.
- **Fixed geometry.** A stream's picture size is fixed for the run; a change ends the stream with an error.
- **Identity.** Each picture carries a key through NVDEC and NVENC, and the source frame number and stamp are looked up by
  that key, so metadata follows its own frame through loss. A stage that cannot keep up discards its queued records and
  resumes at the next keyframe; it never drops compressed frames one by one.

Declare the processed tracks like any other camera in the CaptureServer streams file, with the worker's participant as the
publisher and the monotonic basis the worker publishes:

```yaml
livekit:
  url: ws://PC_HOST:7880
  room: sensing-preview
  publisher: stage-worker          # the identity in the worker's token
  timestamp_basis: monotonic_us
  streams:
    - {name: Head left,  track: processed-left,  width: 2560, height: 1984, fps: 30}
    - {name: Head right, track: processed-right, width: 2560, height: 1984, fps: 30}
```

```bash
video_stage_worker --stream=5600:processed-left --stream=5601:processed-right \
  --livekit-url=ws://127.0.0.1:7880 --livekit-token-file=worker.token
# a file sink instead of LiveKit, to check a run:
video_stage_worker --stream=5600:file=out.h264 --once --operation=sine-warp
```

`tests/tools/mock_orin_sender.py` stands in for the Orin: it mirrors the producer's direct sender (connection and hello order,
retry, keyframe start, bounded queue with drop-to-keyframe, keyframe requests, clock answers on its own offset clock, source
frame ids that step by 2) and feeds it Jetson-style H.264 (no AUDs, IDR every 5 s, constant bitrate). One `--port` per camera.

## Build

```bash
cmake -B build -DBUILD_VIDEO_STAGE=ON -DBUILD_PLUGINS=OFF -DBUILD_VIZ=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_PYTHON_BINDINGS=OFF
cmake --build build --target video_stage_tests && ctest --test-dir build -R video_stage
```

The tests need a CUDA GPU.
