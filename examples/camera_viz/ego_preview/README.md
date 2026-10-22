<!--
SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# EGO live-preview binding

This directory connects the OrbbecSDK to the local `camera_viz` EGO source.
It opens one camera's ColorLeft/ColorRight streams and returns paired host
payloads. The Python source publishes GPU images through NVIDIA's existing
`FrameSource` interface; the viewer keeps its normal rendering lifecycle.

## Responsibilities and data flow

| File or component | Responsibility |
|---|---|
| `ego_capture.cpp` | Own the SDK context, device and stereo pipeline; check pair completeness and device-time agreement; copy host payloads. |
| `__init__.py` | Export `Capture` from the native `_ego_capture` extension and report how to build it. |
| `CMakeLists.txt`, `build.sh` | Build the SDK-only extension for the active Python environment, beside the package entry point. |
| [`sources/ego.py`](../sources/ego.py) | Own the producer thread, retries, per-eye GPU conversion/decoding and stereo Frame publication. |
| [`codec/`](../codec/) | Provide the shared native NVDEC implementation used by the source for H.264/H.265. |
| [`sources/stereo_sbs.py`](../sources/stereo_sbs.py) | Optionally copy GPU eye images into one desktop image, left eye first. |

```text
EGO ColorLeft/ColorRight -> SDK Capture -> paired host payloads
  -> EgoSource -> GPU RGBA8 stereo Frame -> camera_viz
    -> window: left eye, or StereoSideBySideSource for a 2W x H image
    -> XR: separate eye images
```

MJPEG (`mjpg`) is converted to host RGB by the SDK, then staged in pinned memory
and uploaded to GPU RGBA8 images. H.264/H.265 (`h264`/`h265`) remain compressed
host bytes until the Python source passes them to one NVDEC decoder per eye.
Both paths finish GPU writes before publishing reusable triple buffers.
The binding itself does not decode video on the GPU or render a window.

OAK-D uses the external DepthAI Python binding, so it does not need an equivalent
local SDK bridge in `camera_viz`.

### GPU selection

The viewer passes `VizSession.cuda_device_id` through the source factory.
NVIDIA already matches Vulkan and CUDA devices by UUID; EGO uses that CUDA ID
for both eye buffers, upload streams, and NVDEC decoders, including after a
capture retry. A Vulkan `gpu_index` is not a CUDA device ID.

For EGO, omitted `gpu_id` or `gpu_id: 0` means automatic selection, following
the renderer. A nonzero `gpu_id` must match the renderer; a mismatch fails
before allocating EGO GPU buffers or opening the camera. This does not change
which GPU the renderer selects or enable cross-GPU copies.

Standalone `EgoSource` callers can supply `renderer_cuda_device_id` as a
keyword argument. Without it, `gpu_id: 0` follows the calling thread's current
CUDA device at each start; a nonzero value selects that CUDA device directly.

## Build and import

Install the external OrbbecSDK and its udev rules, then run from the repository
root. Setup installs Python dependencies and builds both the SDK binding and
the shared native codec:

```bash
examples/camera_viz/camera_viz.sh setup --with-ego \
  --sdk-root /path/to/OrbbecSDK
```

To rebuild only this binding with an already configured viewer environment:

```bash
source examples/camera_viz/.venv/bin/activate
examples/camera_viz/ego_preview/build.sh --sdk-root /path/to/OrbbecSDK
cd examples/camera_viz
python -c 'from ego_preview import Capture'
```

The SDK root must contain `include/` and `lib/libOrbbecSDK.so`; the active Python
environment must provide pybind11. CMake builds one `_ego_capture` target and
sets its build RPATH to the selected SDK library directory. This standalone
binding does not link the recording plugin. H.264/H.265 preview also requires
the shared codec and a working CUDA/NVDEC environment; MJPEG requires CuPy for
GPU upload. See the [viewer guide](../README.md) for setup and run commands.

## Pair contract and device ownership

`Capture(device_uid, width, height, fps, format)` opens the requested stereo
profile. An empty UID selects the first enumerated device. `next_pair(timeout_ms)`
returns a dictionary with `left`/`right` payload bytes, `timestamp_ns`,
`device_timestamp_ns`, and `sequence_left`/`sequence_right`.

- `timestamp_ns` is host monotonic time sampled after payload extraction, not
  exposure time. The source preserves it in the published Frame.
- `device_timestamp_ns` is the average raw device timestamp of the two eyes,
  in nanoseconds. Pairs whose device timestamps differ by more than 2 ms are
  skipped before conversion or copying.
- SDK wait timeouts, missing eyes and null MJPEG conversions return `None`.
  Invalid converted RGB dimensions or payload size raise an exception.
- Empty pairs check the SDK device list at most once per second. Disappearance
  of the selected UID raises a disconnect exception and uses the source's
  existing close/retry path. A confirmed removed instance releases SDK objects
  through their destructors; ordinary pipeline-stop errors still propagate.
- Payload bytes own their data after the SDK frame is released. GPU Frames
  refer to reusable source buffers and must be consumed through the viewer's
  existing FrameSource lifecycle.

H.264/H.265 preview tracks each eye's SDK sequence independently. At startup,
after reconnecting, or after a sequence gap, that eye waits for parameter sets
and an IDR (or H.265 CRA) before resuming; CRA leading RASL pictures are skipped.
Parameter sets received separately are replayed to the decoder. The other eye
continues decoding, and publication resumes when both current images are ready.
The viewer may hold its previous image until recovery. A timeout alone does not
reset the decoders; MJPEG keeps its independent-frame behavior.

The source producer serializes reads and SDK close. Open failures retry;
capture-read exceptions close/reopen the pipeline and reset encoded decoders.
Other producer failures are retained and raised by `latest()`, preserving the
original exception. NVIDIA's `VizRunner.wait()` reports them to the main thread
and the viewer runs its normal shutdown. The failure is visible before SDK
cleanup completes; a second cleanup exception cannot replace the first error.
Cleanup-only errors are logged and can reach `latest()` while it is still being
polled. Errors arriving during normal shutdown follow the runner's existing
cleanup policy. A fully stopped source resets errors and pending frames when
started again.

`stop()` requests exit and waits up to five seconds; if the producer is still
live, it retains the SDK
handle rather than closing a device still in use. Native waits, conversion and
close release the Python GIL where needed for shutdown coordination.

Preview and the [EGO recording plugin](../../../src/plugins/ego/README.md)
each open the physical camera; run one at a time. Sensor recording, controls
and restoration, OpenXR tracker publication, and MCAP output belong to the
plugin. This package only captures stereo video for live preview.
