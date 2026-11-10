<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Video stage

Moves camera video from the Orin to the PC and on to LiveKit. The Orin sends its encoded H.264 access units to the PC
over a direct link; the PC relay (`video_stage_worker`) hands them, untouched, to LiveKit or a file. By default there is no decode,
encode or GPU in the relay, and it builds without CUDA; an optional GPU stage is described below.

```text
Orin producer --DATA (AVS1 records, H.264)--> PC relay --> LiveKit track (or an Annex-B file)
              <--CTRL (keyframe requests, clock exchange)--
```

- `wire/` the AVS1 record codec, clock exchange, H.264 access-unit helpers and the frame correlator.
- `sender/` the Orin-side sender of AVS1 records.
- `app/` the relay.

`video_stage_worker` serves one Orin camera per `--stream=PORT:TRACK`; all tracks are published under one LiveKit
participant. The recorder and the web page subscribe to the relayed tracks as they would to the Orin's own.

- **Direct input.** The sender connects to each stream's port and sends every access unit as an AVS1 record over TCP
  (`wire/`; codec 2 is H.264). It reconnects by itself, starts at a keyframe, and drops frames up to the next keyframe
  rather than block when the link cannot keep up. Use a wired link.
- **Control link.** A second connection from the Orin carries the PC's keyframe requests and a four-timestamp clock
  exchange (the same bytes the recorder uses over LiveKit). The relay converts each Orin frame-ready stamp onto its own
  monotonic clock before publishing, and answers the recorder's and the page's clock sync with that clock. Video is never
  withheld because of timing: while the Orin-to-PC mapping is missing or less certain than
  `--max-upstream-uncertainty-ms` (5 by default) the frames are still published, without a capture timestamp, so a
  recording that needs one stops and says so. When the Orin's clock jumps, the track is unpublished and published again at
  the next keyframe.
- **Fixed rate and geometry.** The relay publishes what its source encodes, at that rate; a downstream bitrate request is
  logged once and not acted on. A stream's picture size is fixed for the run; a change ends the stream with an error.
- **Overload.** A relay that cannot keep up discards its queued records and resumes at the next keyframe; it never drops
  compressed frames one by one.

Declare the relayed tracks like any other camera in the CaptureServer streams file, with the relay's participant as the
publisher and the monotonic basis it publishes:

```yaml
livekit:
  url: ws://PC_HOST:7880
  room: sensing-preview
  publisher: stage-worker          # the identity in the relay's token
  timestamp_basis: monotonic_us
  streams:
    - {name: Head left,  track: processed-left,  width: 2560, height: 1984, fps: 30}
    - {name: Head right, track: processed-right, width: 2560, height: 1984, fps: 30}
```

```bash
video_stage_worker --stream=5600:processed-left --stream=5601:processed-right \
  --livekit-url=ws://127.0.0.1:7880 --livekit-token-file=worker.token
# a file sink instead of LiveKit, to check a run:
video_stage_worker --stream=5600:file=out.h264 --once
```

`tests/tools/mock_orin_sender.py` stands in for the Orin, with no Orin or camera: it mirrors the sender's connection and
hello order, retry, keyframe start, bounded queue with drop-to-keyframe, keyframe requests and clock answers (on its own
offset clock), and feeds it Jetson-style H.264 (no AUDs, IDR every 5 s, constant bitrate). One `--port` per camera. Run it
against a relay writing a file, then check the file with `ffprobe`.

## The GPU stage (optional)

Built only with `-DBUILD_VIDEO_STAGE_GPU=ON` (needs a CUDA toolkit; the tests need a CUDA GPU). `video_stage_worker --operation=sine-warp` then decodes each picture on NVDEC, keeps it in GPU memory through one GPU operation, and re-encodes it on NVENC, instead of forwarding the Orin's access units untouched. A build without it rejects `--operation=sine-warp` with a clear message and relays exactly as before.

### The operation contract

`core/inc/video_stage/gpu_operation.hpp` defines `GpuOperation`: NV12 in, NV12 out, fixed geometry.

- `prepare(input_geometry)` runs once, outside the frame loop. Build and upload anything per-geometry here (a
  rectification map, for example). Nothing in `enqueue` allocates or uploads.
- `enqueue(in, out, stream)` only queues kernels on `stream` and returns. It does not synchronise the device or the
  stream. After it returns, the caller records a CUDA event on `stream`; whoever reads `out` waits on that event.
- `in` and `out` are different surfaces (the operation is not in-place). `out` has `output_geometry(in.geometry)` and
  keeps the input's color description. Both are `Nv12View`s: device pointers, byte pitches, size.
- Frame identity and timestamps are not the operation's concern; they travel with the frame.

### The example: `SineWarp`

`core/sine_warp.cu` is a horizontal sine ripple, `source_x = x + A*sin(2*pi*y/wavelength + phase)`,
`source_y = y`, sampled bilinearly, with the border fill for the declared range (Y=16 limited, 0 full; chroma 128).
Amplitude 0 is an identity. It is an example, not calibrated rectification: the warped picture does not keep the
source camera's projection.

The stage assumes the decoder's NV12 uses the H.264 default chroma siting described below; a stream that signals a
different siting must be rejected at the decoder, not warped with the wrong sample positions.

Chroma is not sampled at the luma coordinate divided by two. A chroma sample of H.264 NV12 sits on an even luma column
and half way between two luma rows, so the kernel maps that luma-space position, then converts back to chroma-plane
coordinates. Keep that when substituting an operation.

### Substituting the fisheye rectification

Replace the mapping, keep the rest: a class deriving from `GpuOperation` whose `prepare` generates the destination-to-
source maps once (on the CPU, uploaded once) and whose `enqueue` launches a kernel that reads `map_x/map_y` instead of
computing the sine. Sample luma at the map position and chroma at the position derived from the same map with the
siting above. `output_geometry` returns the rectified output size. Nothing else in the stage changes.

### Identity and overload

Each picture carries a key through NVDEC and NVENC, and the source frame number and stamp are looked up by that key, so
metadata follows its own frame through loss. A picture arriving while every encoder input surface is busy is dropped and
counted, never queued.

## Build

```bash
cmake -B build -DBUILD_VIDEO_STAGE=ON -DBUILD_PLUGINS=OFF -DBUILD_VIZ=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_PYTHON_BINDINGS=OFF
cmake --build build && ctest --test-dir build -R 'video_stage|avs1|clock'
```

Add `-DBUILD_VIDEO_STAGE_GPU=ON` for the GPU stage and its tests.

The LiveKit SDK is fetched, or taken from `LIVEKIT_SDK_DIR`.
