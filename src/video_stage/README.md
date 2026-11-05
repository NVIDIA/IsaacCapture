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

## Build

```bash
cmake -B build -DBUILD_VIDEO_STAGE=ON -DBUILD_PLUGINS=OFF -DBUILD_VIZ=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_PYTHON_BINDINGS=OFF
cmake --build build --target video_stage_tests && ctest --test-dir build -R video_stage
```

The tests need a CUDA GPU.
