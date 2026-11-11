<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# camera_livekit_bridge

A downstream consumer of `camera_streamer`'s CUDA IPC output that publishes the
camera streams to LiveKit as pre-encoded H.264 tracks. It talks to
`camera_receiver` only through its IPC socket and imports nothing from egocap or
the Isaac Capture runtime; egocap joins the LiveKit room on its own.

```text
camera_receiver ──CUDA IPC──> bridge ──H.264──> LiveKit room ──> egocap page / recorder
                 (GPU NV12)    GPU copy or shrink, NVENC
```

Each stream runs on its own thread. A frame is copied (or box-averaged down) on
the GPU from the IPC slot into an NVENC input surface, encoded, and submitted as
a pre-encoded access unit. Pixels never reach host memory. The frame's source
sequence is the track's `frame_id`, the sensor start time (when the receiver
derived one) is the `user_timestamp` (what it means is set by `--timestamp-basis`, below), and the sensor
counter is the 8-byte `user_data`. The newest picture is encoded and older
queued ones are skipped. The receiver overwrites a slot without notice once its ring wraps, so after the GPU copy the
bridge checks for a newer notice for the same slot, or as many newer notices as the ring has other slots, and drops the
picture instead of encoding it (counted as `slot overruns`). The session also answers clock-sync requests that
arrive as LiveKit data messages.

## Integration boundary

This example publishes receiver pictures to a browser through LiveKit. The LiveKit session, publication and feedback
wiring (`livekit_publisher/` and the session handling in `bridge/`) is temporary integration. It is intended to move
behind transport adapters: a layer that carries camera video and application messages over a selectable network
transport and owns sessions, admission limits and feedback. Camera buffer ownership and encode and decode
responsibilities belong to the camera interface. A replacement must preserve browser delivery, bounded latency and
safe GPU access. Until receiver slots have read leases, the residual risk described under "Reading the receiver's
slots" applies.

| Current path | Intended owner or replacement | Reuse, adapt or retire | Follow-up and landing condition |
| --- | --- | --- | --- |
| `livekit_publisher/` and the session, publication and feedback wiring in `bridge/` | Transport adapters | Adapt the useful implementation; retire the hard-coded transport glue | Map sessions, publications, encoder feedback and reconnect behaviour to the adapter boundary. Lands when real web clients receive streams, and feedback and recovery work, through an adapter. |
| `ipc_client/` and the slot read in `bridge/stream_worker.cpp` | Camera reader views and leases | Retire | Read frames through a lease held until the final GPU read. Lands with the receiver's lease support. |
| `chrony_clock/` and the capture-time publication in `bridge/stream_worker.cpp` | The receiver backend's published clock mapping (offset, uncertainty, discontinuity epoch) | Transitional: adapt, then retire the bridge-specific adapter | Carry mapped capture time, validity and discontinuity behaviour through the receiver contract. Lands when consumers obtain equivalent timing without the bridge-only adapter. |
| `src/core/timesync/` (used by `chrony_clock/`) | The estimator behind the receiver's clock mapping | Reuse | Keep the chrony evidence handling and qualification behind the destination interface. |

The `chrony_clock/` adapter connects the reusable timing library to the current video publication path and is intended
to converge on the receiver backend's clock-mapping contract. With `capture_monotonic_us`, a mapping that cannot be
vouched for publishes no timestamp, and the frame-ready time is never offered in its place; a capture over the
precision target keeps its timestamp. `monotonic_us` is a separate, declared basis.

## GPU steps (per-frame image operations)

Between the receiver's IPC buffer and the encoder the picture can pass through a series of GPU steps, with no host
copy. None run by default. `--gpu-step NAME[=ARGS]` adds one, repeatable; steps run in the order given, at the source
size, and a stream's own `:DIV` downscale shrinks the result afterwards.

| Step | Arguments | Purpose |
| --- | --- | --- |
| `sine-warp` | `AMPLITUDE_PX,WAVELENGTH_PX,PHASE_RAD` (all optional) | An EXAMPLE horizontal ripple. It is not calibrated rectification; it exists to show the shape of a step. |

A step is a class deriving from `GpuStep` (`gpu_ops/inc/gpu_ops/gpu_step.hpp`) with `prepare()` (once per picture size:
allocate and upload) and `enqueue()` (per frame: launch kernels on the given stream). To add one, write the class and
add a line to `make_gpu_step()` in `gpu_ops/gpu_pipeline.cpp`. `SineWarpStep` (`gpu_ops/inc/gpu_ops/sine_warp.hpp`) is
the template, and its comment explains how lens rectification would be written the same way: a remap table computed
once from the camera calibration and uploaded in `prepare()`, and a per-frame kernel that samples the source through it
(luma and chroma), with the fill for positions outside the image.

## Build

```bash
cmake -S . -B build && cmake --build build -j
```

Needs the CUDA toolkit and nlohmann_json. The NVENC headers and the LiveKit C++
SDK are fetched at configure time; pass `-DLIVEKIT_SDK_DIR=<unpacked SDK>` and
`-DFETCHCONTENT_SOURCE_DIR_NV_CODEC_HEADERS=<unpacked headers>` to build
offline. The build is standalone and is not part of the repository's top-level
build.

## Run

With `camera_receiver` running (for example fed by `camera_streamer/mock_sender`)
and a LiveKit server reachable:

```bash
build/bridge/camera_livekit_bridge \
  --livekit-url ws://127.0.0.1:7880 --livekit-token-file publisher-token.txt \
  --socket /tmp/camera_receiver_avs1_cuda_ipc.sock \
  --stream head=sensing-head:2 --stream wrist1=sensing-wrist1 \
  --fps 30 --bitrate-mbps 8
```

`--stream NAME=TRACK[:DIV]` publishes the receiver's stream `NAME` as the
LiveKit track `TRACK`, shrunk by the integer `DIV` on the GPU (the size must
divide to even dimensions). Track names, the join token's identity and the
published sizes must match the streams file egocap is given. The token needs
publish permission. Every 5 s it prints each track's frame rate, keyframes,
skipped, rejected and slot-overrun frames and mean encode time.

`--timestamp-basis` sets what each frame's `user_timestamp` means, and the consumer's streams file must name
the same basis:

| Basis | Value |
| --- | --- |
| `monotonic_us` (default) | This host's monotonic clock at the instant the picture is ready to encode (before encoding): frame-ready time, later than the sensor capture. |
| `capture_monotonic_us` | The sensor start-of-frame on this host's monotonic clock, converted from the sender's `CLOCK_REALTIME` with chrony evidence for both hosts (below). |

`capture_monotonic_us` uses the sensor start-of-frame that the receiver reads from the sender's AVS1 timing record
(`sensor_capture_realtime_ns`, `sensor_capture_uncertainty_ns`, `sensor_capture_valid`) and the time-sync library in
`src/core/timesync`. Each frame is converted from the sender's `CLOCK_REALTIME` to this host's monotonic clock, or
refused with a typed reason, using chronyd's own report for both hosts, a bracketed read of this host's clocks, and the
sender's stated uncertainty. Two outcomes are different in kind. A conversion that cannot be vouched for (a stale or
unsynchronised chrony report, another reference, an implausible age, an invalid sender record) is published without a
timestamp, and the frame-ready time is never offered in its place. A conversion that passes every one of those checks
but whose combined uncertainty is over the precision target (`over_budget`) still has a capture instant with a known
bound, so it is published with that timestamp; the target classifies the frame in the statistics and does not remove its
time. It needs:

| Option | Meaning |
| --- | --- |
| `--chrony-sender HOST[:PORT]` | The camera host's chronyd, read over chrony's monitoring interface (it must `cmdallow` this host). |
| `--chrony-sender-refid HEX` | The reference id this host's chronyd must have selected (an NTP source's reference id is its IPv4 address in hex). |
| `--chrony-pc-port PORT` | Read this host's chronyd on that command port over loopback, for a dedicated instance next to the system one. |
| `--chronyc PATH` | The `chronyc` binary (default `chronyc` on `PATH`). |

`src/core/timesync/README.md` describes the arrangement (this host measures the camera host, which serves its own clock) and how to set it up with `timesync_tool.py`. The
A realtime step of this host's wall clock is a timing fault: the frame that reveals it and every frame for the next 8 s are published without a timestamp (a longer silence than a recording take tolerates, so a take cannot continue across the step even if some of those frames are lost on the way), and chrony readings taken before the step are never reused. The
`[stats]` line gains the number of frames that met the precision target, a count per other verdict (`over_budget` frames
carry a timestamp; the rest do not), the last combined uncertainty and the clock generation the frame was converted
under. These are runtime diagnostics only and are not recorded with the video. The caller of a recording should treat a
frame without a timestamp as a timing fault.

`--source-full-range` says the receiver's pictures use the full 0-255 range, as SIPL sources do; the encoded stream
then signals full range, so players and recordings keep the black and white levels. Without it the stream signals
limited range (16-235). The receiver's IPC catalog does not carry the range, so it is set here and must match the
source: a full-range picture published as limited range has crushed blacks and clipped whites, and the reverse
looks washed out. The encoder copies levels as they are and does no range conversion. The stream signals no colour
primaries, transfer or matrix, so a player assumes its default (BT.709 for HD).

`--bitrate-mbps` is the maximum. The encoder starts at 2 Mbps and follows the target bitrate that LiveKit's
congestion control reports for each track (90% of the target, never above the maximum or below 0.5 Mbps), so a
fresh connection does not fill the sender's pacing queue. The `[stats]` line shows the current rate, the transport's
target and how many times the rate changed.

Measured with the mock sender on an RTX PRO 5000 against egocap's receiver and recording-feed checks (30 fps, head
and one wrist, local LiveKit server): frames are 60-150 ms old from the first frame and settle near 80 ms while the
rate climbs to the maximum within roughly 20 s. The head at 2560x1984 and 8 Mbps and at 1280x992 both ran with no
missing frames and a median capture-to-receive time of 65-85 ms. A 20 Mbps maximum on the head built a standing
backlog of about 1.2 s, so keep the maximum near 8 Mbps.

## Probe

```bash
build/probe/camera_ipc_probe --socket /tmp/camera_receiver_avs1_cuda_ipc.sock \
  --stream head --frames 30
```

It prints each stream's plane layout, then per frame the slot, generation,
stream sequence, session id, sensor start time and a checksum of the first
pixels. Slots are overwritten without notice when the publisher's ring wraps.

## Reading the receiver's slots

The receiver rewrites its GPU ring slots in rotation and never waits for a reader: a slot is overwritten without
notice about `ring_depth` frames after it was announced, and the only signal is a later frame notice. A consumer
therefore has to finish with a slot within that window or skip ahead. The bridge follows this: it uses only the
newest notice, starts the GPU copy at once on its own stream, waits for it to complete, and drops (never encodes)
the picture when later notices show the ring may have wrapped; those frames count as slot overruns in the `[stats]`
line. The notices arrive asynchronously, so this check cannot prove a copy was clean: a stall of most of the ring's
window between the notice and the end of the copy can still publish a picture that mixes two frames under the earlier
frame's metadata. Only a read guard in the receiver, which skips slots a reader holds, would remove that case.
