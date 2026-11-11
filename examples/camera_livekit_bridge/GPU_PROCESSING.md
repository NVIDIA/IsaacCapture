<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GPU processing in the camera bridge

Every picture the bridge publishes can pass through a series of GPU steps on its way from the receiver to the
encoder. This page describes how that works and how to add a step, for example lens rectification.

None of this runs by default. With no `--gpu-step` the picture is only copied (or shrunk, if the stream has `:DIV`).

## Where it sits

```text
camera_receiver (NVDEC)          camera_livekit_bridge, one thread per stream
  decoded NV12 in a GPU ring  --CUDA IPC-->  slot (GPU) --> [step 1] --> [step 2] --> ... --> NVENC input surface
  (slot notices on a socket)                                       all on one CUDA stream, no host copy
                                                              --> H.264 --> LiveKit
```

- The receiver decodes into a ring of GPU buffers and tells the bridge which slot holds the newest picture. The bridge
  maps the buffers once with CUDA IPC; pixels never cross the socket and never reach host memory.
- The bridge owns the code between the slot and the encoder input surface. That code is a `GpuPipeline`
  (`gpu_ops/inc/gpu_ops/gpu_step.hpp`): an ordered list of `GpuStep`s. With no steps it is one plain copy.
- Each stream has its own pipeline, so streams can have different steps and different calibrations.
- The encoder is created for the size the last step produces, so a step may change the picture size.

## Using steps

```bash
camera_livekit_bridge ... --stream head=sensing-head:2 \
  --gpu-step sine-warp=14,90 --gpu-step another-step=ARGS
```

- `--gpu-step NAME[=ARGS]` adds a step to every stream. It is repeatable. Steps run in the order given.
- Steps run at the source size. The stream's own `:DIV` downscale is added last, so it shrinks the result of the steps.
  Run corrections such as rectification at full resolution and let the downscale come after.
- A step that cannot run on a stream (an odd size, a calibration for another size, a size NV12 cannot hold) stops the
  bridge at start with the stream, the step and the reason.

Known steps:

| Step | Arguments | Purpose |
| --- | --- | --- |
| `sine-warp` | `AMPLITUDE_PX,WAVELENGTH_PX,PHASE_RAD`, all optional | An example horizontal ripple. It shows the shape of a step; it is not calibrated rectification. |

## The step contract

A step derives from `GpuStep` and implements three things:

| Method | Called | Does |
| --- | --- | --- |
| `prepare(input_size)` | once per picture size, before the first frame | Returns an empty string if the step can run on that size, otherwise the reason. Allocate and upload here: lookup tables, calibration, scratch. Remember the size; `enqueue()` receives planes only. |
| `output_size(input_size)` | at prepare time | The size this step produces. Default: unchanged. Return a different size for a crop or scale. |
| `enqueue(in, out, stream)` | every frame | Launch kernels on `stream` and return the CUDA status. |

Rules for `enqueue()`:

- Only queue work. Do not allocate, synchronise the device or the stream, or throw. The bridge waits for the stream
  once per frame, after the whole pipeline, and then checks the receiver's slot (see below).
- `in` and `out` are different buffers. Work is not in place.
- Pixels only. Frame number, timestamps and metadata travel with the frame; a step never touches them.
- NV12 layout: a full-size luma plane and one interleaved chroma plane at half size in both directions. Planes are
  pitched; use the pitch, not the width, to step between rows. Sizes must stay even.
- The output must be fully written, including positions the step cannot compute: use the video-range fill (luma 16,
  chroma 128).

The pipeline owns the buffers between steps (allocated in `GpuPipeline::prepare()`) and gives the last step the encoder
input surface directly, so a step never allocates its own output.

## Adding a step

1. Add a class deriving from `GpuStep`. Put the kernels in a `.cu` file next to it in `gpu_ops/`, add the file to
   `gpu_ops/CMakeLists.txt`, and keep the public class in `gpu_ops/inc/gpu_ops/`. `sine_warp.hpp` and `sine_warp.cu`
   are a complete small example to copy.
2. Register it with one branch in `make_gpu_step()` in `gpu_ops/gpu_pipeline.cpp`, which maps `NAME[=ARGS]` to the
   step and parses its arguments (throw `std::invalid_argument` for bad ones).
3. Add a case to `tests/gpu_pipeline/test_gpu_pipeline.cpp`. A good first test is a case where the answer is known: an
   identity setting that must equal a plain copy, and one setting that visibly changes the picture.
4. Run it: `cmake --build build && ctest --test-dir build`, then a live run with `--gpu-step` and a saved frame.

Skeleton:

```cpp
class MyStep final : public GpuStep
{
public:
    explicit MyStep(Params params) : m_params(params) {}
    const char* name() const override { return "my-step"; }

    std::string prepare(const ImageSize& input) override
    {
        m_size = input;
        if (input.width % 2 != 0 || input.height % 2 != 0)
            return "needs an even picture size";
        // upload tables here
        return {};
    }

    cudaError_t enqueue(const Nv12Planes& in, const Nv12PlanesOut& out, cudaStream_t stream) override
    {
        my_kernel_launch(in, out, m_size, m_params, stream); // luma and chroma
        return cudaGetLastError();
    }

private:
    Params m_params;
    ImageSize m_size;
};
```

## Writing lens rectification as a step

Rectification is a remap: for each pixel of the corrected picture, find where it comes from in the camera image and
sample there. The sine warp is the same operation with a made-up mapping.

- **The mapping.** For output pixel `(u, v)` build the ray of the ideal pinhole camera (the new camera matrix), distort
  it with the lens model (fisheye or radial-tangential coefficients from the camera's intrinsics), and project it with
  the real intrinsics. The result is the source position `(sx, sy)`. The sine warp uses
  `sx = x + A*sin(2*pi*y/L + phase), sy = y` in its place.
- **Compute it once.** The mapping depends on the calibration and the picture size, never on the pixels. In
  `prepare()`, compute `(sx, sy)` for every output pixel on the host, `cudaMalloc` a `float2` table of that size, and
  upload it. Free it in the destructor. The per-frame kernel then reads the table entry and bilinear-samples the
  source, which costs about the same as the sine warp.
- **Chroma.** The chroma plane is half size. Either keep a second half-size table, or evaluate the luma table at each
  chroma sample's luma position (`(2*cx, 2*cy + 0.5)` with the usual siting) and halve the result.
- **Outside the image.** A source position outside the image gets the fill (luma 16, chroma 128). Test for it with a
  positive predicate so a NaN also takes the fill path.
- **Size.** Rectification often crops or rescales the field of view. Return that from `output_size()`; the encoder is
  created for the last step's output.
- **Calibration input.** Pass a calibration file path as the step's argument, parse it in `make_gpu_step()`, and
  refuse in `prepare()` a picture size the calibration was not made for. If cameras differ, the per-stream pipelines
  allow a different calibration per stream; the step spec would need a stream name or a per-stream argument.
- **Check it.** A known calibration of a synthetic grid with straight lines is the natural test: after the step the
  lines are straight within a pixel.

## What the bridge does around the steps

- The pipeline runs on one CUDA stream and the bridge waits for that stream once per frame. That wait is required:
  the receiver's ring slot can be overwritten, so the copy out of it must be finished before the slot may be trusted.
- After the wait, the bridge checks for a newer notice for the same slot, or as many newer notices as the ring has
  other slots, and drops the picture instead of encoding it (the `slot overruns` count in the stats line). This is a
  best-effort check, not exclusive ownership of the slot; see `README.md`.
- Steps add GPU time to every frame. Keep a step's kernels light and measure with the stats line (`encode` and the
  capture-to-ready latency) before and after.
- The encoder surface is copied into by the last step, so a step that leaves part of its output unwritten publishes
  whatever that surface held before.
