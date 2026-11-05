<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Argus and SIPL support in the Orin LiveKit sender

## Status

Implemented but never built or run (no Argus SDK or Argus camera available): `core/argus_capture.*` (Argus buffer
streams into NvBufSurface DMA-BUFs), the encoder's DMA-BUF input mode (`JetsonEncoder::submit_dmabuf`), the output's
leased-frame path, optional builds (`SENSING_WITH_SIPL`, `SENSING_WITH_ARGUS`) and `--camera-backend`/`--argus-mode`.
Not done: dynamically loaded backend modules, a verified per-frame correlation (below), and every measurement.

## Goal

Start the same sender executable/service on Orin. It selects the camera backend
appropriate to the installed driver and configured rig, captures frames, encodes
H.264 in hardware, and publishes to LiveKit. CaptureServer, the application
backend, and recording remain on the PC.

Extend the existing FastLiveKit implementation with simple production modules.
Priorities are low latency, bounded buffering, minimal work on Orin, and readable
code. This document is an implementation plan, not a claim of hardware validation.
No Argus SDK or Argus camera has been available to test any of it.

```text
                         Argus capture: NV12 DMA-BUF
                        /                           \
Orin sender/controller                               shared hardware H.264 -> LiveKit
                        \                           /
                         SIPL capture: existing path

PC: CaptureServer -> stream descriptors -> backend / recording / video consumers
```

## Production ownership and dependencies

The sender and both backends belong under `src/plugins/sensing/`, alongside the
existing SIPL implementation. They must not live in, link against, import, or
require files from `examples/camera_streamer/` at build or runtime.

The existing Argus example is a source reference for capture and buffer ownership.
Extract/adapt only the necessary implementation into production modules, preserving
licensing. Do not bring over its TCP transport, AVS1 protocol, stereo scheduling,
priority policies, or application controller. If the example subsequently reuses
the production library, dependencies point from the example to the library.

Suggested responsibilities within the existing sensing layout:

| Component | Responsibility |
| --- | --- |
| `app/` | Parse configuration, select backend, start/stop the sender |
| `core/camera_capture.hpp` | Small capture interface and owned frame representation |
| `core/argus_camera.*` | Argus enumeration, modes, sessions, capture buffers |
| `core/sipl_camera.*` | Existing SIPL rig ownership and capture implementation |
| Existing Jetson encoder | Shared hardware H.264 encoding, with native input adapters |
| Existing LiveKit output | Session, tracks, encoded publication, feedback |

Keep vendor SDK headers out of the common capture interface. Follow the repo's
CMake/include rules when implementing the concrete target/header layout.

## Capture interface

Use one small `CameraCapture` interface for enumeration, supported modes,
start, acquire-latest, and stop. A backend instance owns the configured rig;
do not create independent SIPL instances per stream because SIPL ownership is
process-wide.

A frame carries stream identity, sequence, dimensions, timestamp information,
native buffer representation, and explicit lifetime ownership. Use a small tagged
representation for the two actual inputs needed:

- Argus: NV12 DMA-BUF surface, including plane/layout information and required
  synchronization state.
- SIPL: the existing CUDA frame representation, borrowed from the producer and
  copied out by the consumer after its pacing check, as today. An owning copy
  pool in the adapter would add a copy for every paced-out frame.

Use move-only frame leases for frames a consumer may hold (Argus buffers). The
SIPL borrowed latest-frame view is marked borrowed and must not be presented as
indefinitely owned memory. Do not normalize every backend to RGBA, add a general image-format
framework, or duplicate encoding/publication orchestration in each backend.

## Startup and backend selection

Add `--camera-backend=auto|argus|sipl`, defaulting to `auto`. Selection happens
once during startup, before backend-specific configuration becomes mandatory.
An Argus installation must not require SIPL JSON or tuning files.

1. Inspect installed backend modules and the configured camera selectors/modes.
2. Probe support for that rig using backend-specific discovery and installed
   driver information. Argus enumeration can identify exposed devices. SIPL
   configuration parsing alone does not prove that hardware is present.
3. If exactly one backend matches, select it. If a rig profile names a backend,
   use that choice. If both match without a preference, report the ambiguity and
   the explicit override rather than guessing which camera to open.
4. Release discovery resources, then open and start the chosen rig. Never run
   both backends against the same cameras concurrently.
5. Report the selected backend, resolved cameras, modes, and track names.

An explicit backend override must either succeed or explain its failure. Do not
silently fall back after camera-busy, permission, tuning, or capture errors.
Distinguish an absent/unsupported backend from a broken selected configuration.
If discovery cannot establish a match, require the rig setting or override.

Do not select solely by JetPack version, library presence, or `/dev/video0`.
Detection must not change drivers, device trees, or restart platform services.
The application cannot compensate for a missing compatible camera driver.

Keep logical stream/LiveKit track names independent of backend-specific sensor
identifiers. Argus device indices are not SIPL pipeline IDs. Extend the existing
sensor-listing command to report the backend and supported modes so configuration
can be resolved without relying on coincidental numbering.

## Optional SDK dependencies

Build the Argus and SIPL implementations as optional shared libraries, loaded on
demand through a small fixed create/destroy boundary. The sender's direct runtime
dependencies must not include an unused vendor SDK: otherwise the dynamic loader
can terminate the process before backend selection runs.

There is no plugin registry, service discovery layer, or separate capture worker
process. The loader knows the two supported modules. Keep a module loaded until
all its capture objects and frame leases have been destroyed.

Build/package against the target JetPack/BSP and compatible vendor libraries.
The same executable name, service, and configuration model do not imply a single
universal binary across incompatible BSP releases. Argus platform services remain
normal system prerequisites; this plan introduces no additional application daemon.

## Argus capture and fast encoding

The example's `CaptureManager.cpp` already demonstrates Argus buffer streams with
EGL-image-backed, client-owned NV12 surfaces. Its encoder integration demonstrates
retaining capture buffers until the encoder releases its input. Adapt those
patterns into the production Argus backend and shared encoder.

Target path:

```text
Argus -> NV12 DMA-BUF -> Jetson hardware H.264 -> encoded LiveKit publication
```

Extend the existing Jetson encoder with DMA-BUF input alongside its current CUDA
input path. Confirm the target encoder accepts the selected NV12 surface layout,
dimensions, synchronization, and plane metadata. The example uses HEVC; reusing
its ownership pattern does not mean changing the LiveKit codec to HEVC.

- Keep capture buffers in a bounded pool. Retain a submitted buffer until the
  V4L2 encoder output plane dequeues it; queue submission or arrival of an encoded
  packet alone does not prove the input is reusable.
- Observe producer completion before encoder consumption. Release dropped frames
  promptly. During shutdown, release encoder-held inputs before destroying the
  capture buffers or unloading the backend.
- Keep pending work bounded and prefer the latest unsubmitted frame. Do not build
  a FIFO backlog or discard arbitrary already-encoded reference frames.
- Keep hardware encoding single-owned, disable B-frames, and preserve existing
  bitrate/keyframe feedback handling. Capture must not block on network writes.
- Avoid CPU pixel copies and Argus-to-RGBA-to-NV12 conversion. If a target requires
  a surface conversion, use a supported hardware path and document its cost;
  do not silently claim direct DMA-BUF performance.

This is a low-overhead design, not a measured latency guarantee. SIPL's existing
conversion path can remain while adding Argus; optimizing SIPL further is a
separate measured change.

## Timestamps and PC contract

Attach timestamp provenance to the frame and preserve the association through
encoding and LiveKit publication. Argus sensor time, SIPL raw counters, host
monotonic time, and Unix time are distinct clock domains. Do not relabel a sensor
timestamp as Unix time or subtract unscaled counters from nanoseconds.

Use each backend's documented timestamp units and clock mapping. Until a sensor
mapping is qualified, retain the existing honest processing-time classification
and recording eligibility rules. Backend selection must not weaken those rules.

The PC interface remains the existing `SessionInit.streams` contract, with LiveKit
details inside the corresponding stream descriptors. Track names and media
metadata remain consistent across capture backends. CaptureServer exposes the
descriptors without joining LiveKit; consumers connect to LiveKit for actual
video. No Argus-specific discovery endpoint or parallel camera list is needed.

The mock remains a standalone process publishing LiveKit streams. It emulates
the external sender contract, not an in-process camera backend, and cannot
qualify Argus/SIPL hardware performance or sensor clock mapping.

## Open precondition: per-frame correlation

The Orin encoder does not return the timestamp it is given, so units are paired with
captures by submission order, which shifts after an encoder-internal drop. Before the
DMA-BUF work, verify against the target Multimedia API headers whether a supported
per-frame timestamp or cookie mechanism exists (for example a copy-timestamp control).
If none does, detect a lost input and stop or flag the stream.

## Implementation sequence and acceptance

1. Introduce the capture boundary and adapt SIPL without changing the wire
   contract or its current frame-lifetime guarantees.
2. Make backend dependencies optional and implement explicit/automatic startup
   selection, including clear ambiguous and unavailable cases.
3. Add the production Argus capture module and shared encoder DMA-BUF input.
4. Keep the existing PC integration and standalone mock; document backend
   configuration and supported target packages.

Step 1 and the selection part of step 2 are in the capture-backend rearchitecture branch; the Argus module and
the encoder's DMA-BUF input build on it. There is one sender stack, not a second application.

When execution is authorized, verify selection with either SDK absent, ambiguous
rig configuration, and explicit overrides. Check frame-release behavior under
encoder backpressure, dropped frames, and shutdown. On suitable hardware, verify
Argus H.264 buffer compatibility, capture-to-publish latency, resource use, and
SIPL regression behavior. Use mock timestamps for PC contract checks only.

This document does not claim that any of these checks has passed.

## Research references

- [NVIDIA camera support matrix](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/CameraDevelopment/CameraDevelopmentSupportMatrix.html):
  Argus/V4L2 and SIPL/UDDF support depends on the camera/driver stack.
- [Argus camera provider reference](https://docs.nvidia.com/jetson/l4t-multimedia/classArgus_1_1ICameraProvider.html):
  enumeration does not guarantee the camera is available for a capture session.
  Verify API details against the target BSP headers.
- Source references: `examples/camera_streamer/argus_sender/src/CaptureManager.cpp`,
  `HevcEncoder.cpp`, and `StreamProcessor.cpp`. These are implementation evidence,
  never production build/runtime dependencies.
- Existing FastLiveKit integration: `src/plugins/sensing/core/sensing_camera.*`,
  `sipl_camera.*`, `jetson_encoder.*`, and the existing LiveKit output module.
