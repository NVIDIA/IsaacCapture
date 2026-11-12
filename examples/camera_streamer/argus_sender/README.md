<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# argus_sender

Standalone, headless NVIDIA Argus camera sender for Jetson Thor and Orin. One
YAML file can describe any combination of mono and stereo logical streams. The
sender captures them through Argus, optionally resizes and rotates each stream,
produces HEVC or tight NV12, and sends every stream over one TCP connection in
the AVS1 format.

The project builds only `argus_sender`; it does not configure NVIDIA's full
`argus_camera` sample and does not require the legacy `libnvmm_jpeg.so` used by
JPEG sample modules. CMake downloads the pinned yaml-cpp 0.9.0 release archive,
verifies its SHA-256 digest, and builds it statically.

## Integration boundary

The sender captures through a small backend interface (`src/CaptureBackend.h`, `src/CaptureFrame.h` and
`src/CameraCapture.h`) so that one capture, encode and send loop serves both SIPL and Argus. That interface is
temporary integration glue private to this sender. It is intended to converge on a shared camera backend and
buffer-ownership contract once such a contract covers both capture paths, and the wrappers may then be adapted or
replaced. Until then it is the supported seam, and SIPL and Argus capture and the HEVC encoder stay available.

| Current path | Intended owner or replacement | Reuse, adapt or retire | Follow-up and landing condition |
| --- | --- | --- | --- |
| `src/CaptureBackend.h`, `src/CaptureFrame.h`, `src/CameraCapture.h` | A shared camera backend contract and frame ownership | Adapt the wrappers; retire the duplicate standalone contract | Map source configuration, frame identity and buffer ownership onto the shared contract. Lands when SIPL and Argus both use it with equivalent behaviour and safe buffer lifetime. |
| Capture and encoder implementation, `src/Avs1Wire.*` | Camera backends and encoders; AVS1 over TCP remains the transport baseline | Reuse | Carry the tested behaviour forward. AVS1 over TCP stays the baseline until a replacement transport is qualified. |

## Build

```bash
cd /path/to/argus_sender
./build.sh
```

`build.sh` selects single-process Argus and is intended for AGX Orin or Thor. It installs any missing Ubuntu,
Jetson Multimedia API, CUDA, DRM, and EGL build dependencies, then configures a
Release build and builds `build/argus_sender`. Already-installed packages are
left untouched.

The first configure requires network access to GitHub. CMake caches yaml-cpp
under `build/_deps`; later builds reuse it until the build directory is removed.
For an offline build, provide a local source checkout:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DFETCHCONTENT_SOURCE_DIR_YAML_CPP=/path/to/yaml-cpp
```

If the Jetson Multimedia API is installed somewhere else:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DMMAPI_ROOT=/usr/src/jetson_multimedia_api
```

List the Argus device and mode indices visible on the current board:

```bash
build/argus_sender --list-modes
```

Validate a production config without binding TCP or opening any cameras:

```bash
build/argus_sender --check-config --config configs/cameras.yaml
```

## Run

```bash
./run.sh
```

`run.sh` uses `configs/cameras.yaml` and may be launched from any working
directory. Additional command-line arguments are passed to `argus_sender`.

By default, runtime logging is concise: it reports aggregate FPS and global TSC
span, stereo-eye TSC delta, and per-stream capture-to-TCP-sent latency. Enable
the detailed capture, FIFO-selection, stereo-repair, encoder, TCP-stage, and
per-stream diagnostics when troubleshooting. The verbose capture summary also
reports the latest selected frame's sensor exposure, sensor analog gain, ISP
digital gain, AWB mode, AWB color temperature, and per-Bayer-channel AWB gains
for each camera:

```bash
./run.sh --verbose
```

All periodic summaries use `runtime.log_interval_s` (default: 1 second). Set it
to `0` to disable periodic summaries without suppressing warnings, errors, or
one-time lifecycle messages.

The sender is the TCP server. It waits for one receiver, creates fresh Argus and
encoder pipelines after the connection is accepted, and tears the entire
pipeline down after a disconnect. It then listens again. This makes each new
receiver start with fresh HEVC parameter sets and an IDR frame while keeping the
steady-state path queue-free except for the TCP sender's depth-one handoff.

## Configuration

```yaml
version: 1

network:
  bind: 0.0.0.0
  port: 5000
  reconnect_ms: 1000
  dump_wire: ""                 # optional exact TCP-record dump

runtime:
  frames: 0                      # 0 = run until stopped/disconnected
  kpi: true
  log_interval_s: 1              # 0 disables periodic summaries
  bench_encode: false

alignment:
  warmup_frames: 60
  interval_frames: 60        # 0 disables periodic checks

streams:
  - id: front_stereo
    type: stereo                 # mono | stereo
    cameras:                     # one camera for mono, left then right for stereo
      - {device: 0, mode: 0}
      - {device: 1, mode: 0}
    output:
      fps: 60                    # 0 = native; see Mixed FPS
      codec: hevc                # hevc | raw_nv12
      width: 1280                # optional
      height: 992                # optional
      resize_mode: strict        # strict | center_crop | stretch
      rotation: 180              # 0 | 180
      bitrate_mbps: 25           # per camera, HEVC only
      gop: 60                    # 0 defaults to output FPS
```

Stream descriptors and samples always follow `streams` and `cameras` order.
Stream IDs are local configuration names; the wire uses their stable zero-based
indices. A physical Argus device may appear only once in a config.

`capture.mode` selects the sender's secondary-stream consumption policy:
`mailbox` (the default) is demand-driven, while `fifo` continuously acquires
into a two-frame application candidate window. For raw NV12 streams, it also
configures the Argus EGL stream as mailbox or FIFO; set
`capture.fifo_length: 3` or greater for EGL FIFO. For HEVC, Argus captures
directly into client-owned DMABUFs, so no EGL mailbox/FIFO mode is set;
`capture.mode` still selects the secondary consumption policy. The configured
mode and direct-buffer/EGL stream type appear separately in the startup log.
`capture.fifo_length` is not applied to a direct-buffer stream. A priority
stream must use the default mailbox policy regardless of codec; FIFO priority
capture is rejected.

### Size and aspect ratio

- Omit `width` and `height`: use the selected sensor mode's native dimensions.
- Specify one dimension: derive the other as an even number while preserving
  the source aspect ratio.
- Specify both with `strict`: reject any aspect-ratio change greater than 0.1%.
- Specify both with `center_crop`: crop equally around the source center, then
  resize to the requested dimensions.
- Specify both with `stretch`: resize the full source to the requested dimensions.

All resolved NV12 dimensions must be nonzero and even. For HEVC, Argus writes
the resolved ISP output directly into client-owned block-linear DMABUFs that are
queued unchanged to NVENC. Raw/remapped streams still use the sender's GPU/VIC
processing path. `rotation` supports 0 and 180 degrees as receiver-side
presentation metadata; the sender does not rotate pixels.

### Mixed FPS

The highest configured stream FPS is the AVS1 bundle cadence. A lower-rate
stream is marked absent in bundles where it is not due; its descriptor remains
present, but its `PRESENT` bit is clear and its samples have zero payload length.
Frames are never duplicated. For a 60 FPS base cadence, a 30 FPS stream appears
in 30 bundles per second and a 45 FPS stream appears in 45 bundles per second.

The sender leaves Argus capture at the camera/trigger rate and selects output
frames with a deterministic fractional scheduler. The priority stereo domain
is drained continuously by independent eye workers; HEVC uses direct Argus
buffers rather than an EGL stream. A mailbox-policy secondary is consumed
once per base tick through a bounded one-permit gate. A FIFO-policy secondary is
consumed continuously and retains two acquired frame handles or DMABUFs; each
base tick
selects the candidate whose SOF TSC is closest to the synchronized priority
reference. Only that selected frame is retained for processing. HEVC retains
its client-owned capture DMABUF through NVENC; raw streams copy the selection
into their processing DMABUF. Frames for a lower-rate stream are discarded on
absent ticks.

Use `fps: 0` to send every native capture. Native and explicit rates cannot be
mixed in one config. When every stream uses `fps: 0`, their resolved native rates
must match. Explicit rates may be mixed, including rates such as 60, 45, and 30.

### Codec behavior

`hevc` uses one `NvVideoEncoder` instance per camera. Stereo eyes therefore
encode independently and in parallel. Each encoder consumes the same
client-owned block-linear DMABUF that Argus filled; there is no intermediate
raw-frame copy. Payloads are Annex-B HEVC access units. The sender requests AUD
insertion and repeated SPS/PPS at IDR frames when the platform accepts those
controls.

`raw_nv12` sends tightly packed NV12: `width * height` Y bytes followed by
`width * height / 2` interleaved UV bytes. Surface pitch is removed before the
wire payload is built. Full-resolution raw streams can exceed 1 GbE; the sender
prints a bandwidth warning when the configured raw payload estimate exceeds
0.9 Gbps.

## Capture and latency model

Capture is organized into Argus domains. The priority head stereo stream owns
two independent single-device sessions and continuously acquires each eye in
its own worker. A one-frame slot per eye retains the newest acquired frame;
SOF-based pairing replaces only the older eye while retaining the newer eye.
An event worker still monitors Argus errors. A secondary stereo stream owns one
two-device session; each mono stream owns a single-device session. Mailbox-policy
secondary domains are capture-complete-driven with one pending frame-set slot;
FIFO-policy secondaries continuously fill a two-candidate frame-set window.

The coordinator chooses a maximum-rate stereo HEVC stream as the priority when
one is available; otherwise it uses the first maximum-rate stream. At startup,
only that domain runs during its warm-up and SOF-TSC synchronization check.
Secondary domains then start with their configured consumption policy and warm
up while the priority domain continues to drain. The priority domain is checked
again after secondary startup. A stereo priority domain is considered associated
after three consecutive frame sets have a TSC span no greater than half its
sensor period. Failure to associate within 120 frame sets is fatal. At runtime,
the older eye is selectively reacquired, up to four times, when a pair is more
than half a frame period apart. The retained newer frame is not processed until
the final pair is selected; direct-buffer HEVC frames are not copied into NVENC.
Runtime bundles apply the same half-period span check before computing the
priority midpoint used for FIFO selection.

On normal bundles, the coordinator consumes the priority domain first. Its HEVC
DMABUFs are already ready for NVENC because Argus captured directly into them.
Secondary-domain permits are released together after 42 percent of the priority
sensor period (7 ms at 60 Hz), and the coordinator then consumes the selected
secondary frames. After all due frames have been acquired, every HEVC stream is
queued into NVENC before raw transforms begin. Raw scaling and packing overlap
asynchronous HEVC encoding. This prioritizes head latency without adding a
frame queue or a sender-side HEVC pixel copy.

The startup synchronization check is intentionally scoped to the priority
domain. Secondary FIFO streams may select an adjacent trigger epoch without
disturbing the stereo head pair. While any continuous FIFO domain is active,
per-bundle closest-candidate selection replaces periodic global reacquisition and
`alignment.interval_frames` is reported as inactive. With mailbox-only capture,
a positive interval retains the periodic global SOF-TSC check and reacquires a
domain more than half its sensor frame interval behind the newest frame, up to
four drops. Reacquiring a stereo domain replaces both eyes together.
Periodic diagnostics report per-stream capture FPS, right-minus-left TSC delta,
per-stream processing FPS/bitrate, aggregate AVS1/TCP rates, and the average/max
`global_tsc_span_us` from the earliest to latest camera SOF in each bundle.
`argus_sender[capture:<stream-id>]` counts delivered frames, including HEVC
frames whose DMABUF ownership has already moved to the encoder path. Its
right-minus-left TSC delta appears only when both SOF timestamps are valid.
`argus_sender[sof-acquire:<stream-id>]` prints each camera's sensor-SOF-to-
acquire average/max and selected-frame gaps once per `runtime.log_interval_s`
(one second by default), even with `runtime.kpi: false`; set the interval to
zero to disable it. The SOF latency is `n/a` when comparable TSC metadata is
unavailable. With KPI enabled, `[head-handoff:<stream-id>]` also reports
priority slot overwrites and alignment drops once per interval. A priority
camera exits after 10 seconds without its first new frame or 1 second without
a subsequent new frame, including repeated acquisitions of the same frame
number; the service can then restart it.

With KPI enabled, `argus_sender[sof-to-sent]` reports each present camera's
`sensor_sof_to_tcp_sent_ms(avg/max)` for frames with valid SOF TSC. It starts
at sensor start-of-frame, not photon arrival, and ends after the complete AVS1
record has been written to the local TCP socket. It excludes network transit,
receiver processing, and display.

The TCP KPI line reports `acquire_spread_ms(avg/max)`,
`capture_to_encode_start_ms(avg/max)`,
`capture_to_encode_done_ms(avg/max)`, and `capture_to_tcp_sent_ms(avg/max)`
once per interval. The three capture-to-* measurements start when the earliest
camera worker's `acquireFrame()` or `acquireBuffer()` returns for a final frame
included in the bundle. `acquire_spread_ms` is the latest-minus-earliest
acquisition completion among all present cameras in that bundle. This is capture
acquisition start, not physical sensor SOF. Frames rejected during stereo
alignment are excluded. Encode start is the first HEVC encoder submission among
the bundle samples; for raw NV12 it is the start of payload packing. Encode
done is the latest encoder callback among the bundle samples; for raw NV12 it
is payload packing completion.
TCP sent ends when `sendAll()` has handed the complete record to the local TCP
stack. It does not include network transit, receiver decode, or display latency.

The TCP interval is followed by one `argus_sender[latency:<stream-id>]` line
for every stream present during that interval. Each line uses that stream's
earliest `acquireFrame()` completion as its capture anchor. HEVC streams report
acquisition offset from the bundle's earliest stream, within-stream acquisition
spread, copy completion, encoder submission, encoded-payload readiness, the
wait from encoded readiness through complete-bundle TCP transmission, and total
stream capture-to-send latency. For HEVC, `copy_done` denotes direct-capture
buffer readiness and should be approximately zero from the acquisition anchor;
raw streams report the corresponding copy, raw-payload readiness, ready-to-send
wait, and total latency. Copy completion is timestamped by the copy worker for
raw streams immediately after its transform returns. It is not the later time
at which the coordinator waits for that worker. These metrics are diagnostic
metadata only and do not alter the AVS1 wire format.

## Sensor capture time

A camera backend that reports its sensor's capture timestamp can have the sender state it in the sender's
`CLOCK_REALTIME`, so another host whose wall clock is synchronised can compare it with its own. `SensorClockMap` reads the
platform counter, `CLOCK_MONOTONIC` and `CLOCK_REALTIME` in one bracket once a second and converts each frame's timestamp
from the difference: counter to `CLOCK_MONOTONIC` through the anchor, then to `CLOCK_REALTIME` through the anchor's
realtime-minus-monotonic offset. Every conversion is checked: a capture time after the frame was ready, an
implausibly old one, a counter or clock that cannot be read, an unknown frequency, or a step between two consecutive
anchors (the counter against `CLOCK_MONOTONIC`, or `CLOCK_REALTIME` against `CLOCK_MONOTONIC`) gives
no capture time for that frame, never a substituted host-side stamp, and the reason is logged when it changes. A frame
that passes also carries an uncertainty (the anchor's read bracket plus 500 ppm of the time between the anchor and the
sensor event, the bound on how far `CLOCK_MONOTONIC` can be slewed against the counter) and the anchor's age, which is
reported separately as its freshness. The uncertainty bounds the mapping only; it
does not include how closely the sensor's timestamp event matches the physical exposure, or how well the two hosts'
wall clocks agree.

The SIPL backend reports its start-of-frame timestamp this way. On the board it was measured against the platform counter
(`CNTVCT_EL0`, 31.25 MHz): the raw value counts that counter, advances at the same rate as `CLOCK_MONOTONIC`, and leads
the moment the frame is dequeued by about 22 ms. That is one board and one camera, not vendor documentation.

Every AVS1 bundle carries one timing record per sample (see [WIRE_FORMAT.md](WIRE_FORMAT.md)): the capture instant in
the sender's `CLOCK_REALTIME` (clock domain 2), the uncertainty and the anchor age, valid only for frames that passed
the checks; a sample without a checked mapping carries an all-zero record. A receiver that predates the extension
rejects those bundles, so deploy the receiver before the sender. A new `session_id` invalidates every earlier mapping.

## Wire dump

Set `network.dump_wire` to save the exact length-prefixed AVS1 records that were
successfully sent over TCP:

```yaml
network:
  dump_wire: /tmp/cameras.avs1
```

The first connection truncates the file; reconnects in the same sender process
append complete records from new sessions. Dump I/O occurs in the TCP sender
thread and can apply backpressure if storage cannot sustain the stream bitrate.
See [WIRE_FORMAT.md](WIRE_FORMAT.md) for the byte-level format.

## Source layout

- `SensorClock.*`: the counter-to-`CLOCK_REALTIME` mapping (through `CLOCK_MONOTONIC`) and its checks.
- `Config.*`: YAML parsing and validation.
- `CaptureManager.*` / `CaptureDomainPlan.*` / `PriorityFrameHandoff.h`: independent priority-eye acquisition and latest-frame slots, event-driven secondary Argus domains, configurable EGL mailbox/FIFO consumers, and timing.
- `CapturePriorityPolicy.*`: priority-stream selection, startup synchronization tolerance, and secondary release timing.
- `CaptureCoordinator.*`: mixed-rate scheduling, priority-first capture, periodic global alignment, and pipelined cross-domain orchestration.
- `StreamProcessor.*`: direct Argus-DMABUF HEVC submission plus raw processing.
- `FrameBundleAssembler.*` / `Avs1Wire.*`: ordered asynchronous bundle assembly.
- `TcpTransport.*` / `SenderSession.*`: depth-one TCP sending and reconnects.
- `OutputGeometry.*` / `MixedRateScheduler.*`: pure geometry/rate policy.
- `tests/`: host unit tests and the opt-in 120-frame camera smoke test.

The matching receiver is [camera_receiver](../camera_receiver/README.md).
See the [example setup](../README.md) for sender, receiver and consumer commands.

## Remap assets

The default config uses bundled [example remaps](remaps/README.md). Paths resolve
relative to the YAML file. The maps preserve the 1280x992 capture, 1280x960 lookup
grid and 2x box downsampling to 640x480 wrist output. Replace the sample calibration
for your own camera unit.
