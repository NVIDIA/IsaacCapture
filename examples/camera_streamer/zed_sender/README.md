<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ZED USB stereo sender

This sender uses a ZED Mini USB/UVC camera on Jetson Thor and preserves the existing
HEVC → AVS1/TCP → camera_receiver → CUDA IPC preview workflow. It sends logical
stream 0 (`head`), with `head.left` and `head.right`, on TCP port 5000. No ZED SDK,
Argus camera session, calibration file, or robot runtime is required.

## Build requirements

The implementation targets Jetson Thor with L4T 39.2.1/OpenRM BSP, the installed
Jetson Multimedia API, and CUDA 13.2. It uses `NvBufSurfaceMapCudaBuffer`; older
Orin BSPs are not validated. The build defaults to CUDA architecture 110 (Thor).
CMake, a C++ compiler, and CUDA `nvcc` are required. YAML-CPP is used from an
installed package or the same pinned source archive as the Argus build; the
build script reuses a sibling Argus build's source cache when available. It installs
no system packages.

## Run

On Thor, from the Isaac Capture repository root:

```bash
cd examples/camera_streamer
./zed_sender/build.sh
./zed_sender/run.sh
```

Development is on branch `rahulsi/zed_streamer`. The tested deployment export on
the development Thor is `/home/nvidia/camera_streamer`; run the same build and
launch commands from that directory when using the export.

The sender listens for the receiver and acquires the USB camera when a receiver
connects. Ctrl+C stops it. The previous USB format and FPS are restored on normal
shutdown and after each connection; restoration is best effort if unplugged.
There is no service installation. The ZED launcher does not stop Argus services.
Only one application can stream from the selected USB video node at a time, and
only one sender can listen on a given TCP port.

On the receiving workstation, from the repository root on this branch, run the
same receiver with the ZED profile:

```bash
cd examples/camera_streamer
./camera_receiver/run.sh camera_receiver/configs/receiver_zed.yaml --host <thor-ip> --port 5000
```

In a second workstation terminal, from the repository root:

```bash
cd examples/camera_streamer
./cuda_ipc_consumer/run.sh --streams head.left,head.right
```

An existing PR #1410 receiver checkout can also be reused: copy
`camera_receiver/configs/receiver_zed.yaml` from this branch into the same relative
path in that checkout, then run these receiver and preview commands there.

The receiver's default profile has different dimensions and extra wrist streams;
select the ZED profile explicitly. Its IPC names, socket, pixel format, and decode
path remain the same. Both workstation environments can be reused. The one-line
receiver metadata adjustment in this branch labels a missing Tegra clock
as unavailable; older receivers can still display the ZED stream, but may retain
the old `sender_clock` label. AVS1 hardware timestamp validity bits remain clear.

## Configuration

All YAML widths and heights are **per eye**. The USB image contains both eyes
side by side. Capture dimensions/FPS select an exact supported camera mode;
unsupported modes fail instead of silently changing settings. Output dimensions
may be smaller, with the same aspect ratio and even dimensions. Output FPS may
be any integer up to capture FPS; the sender skips complete stereo pairs before
encoding when needed. For non-divisor rates the selected frame intervals vary.

| Profile | Capture per eye | Network output per eye | HEVC target per eye | Receiver profile |
|---|---|---|---|---|
| `configs/zed.yaml` | 1280×720 at 60 FPS | 1280×720 at 60 FPS | 50 Mbps | `receiver_zed.yaml` |
| `configs/zed_low_bandwidth.yaml` | 1280×720 at 30 FPS | 640×360 at 30 FPS | 6 Mbps | `receiver_zed_low_bandwidth.yaml` |

Select the second profile with:

```bash
./zed_sender/run.sh --config zed_sender/configs/zed_low_bandwidth.yaml
```

Copy/select its matching receiver profile in the same way as above. After editing
output width/height, update receiver width/height to match. After editing FPS,
adjust `receive_fps_warning_threshold` in the receiver profile. Bitrate is a CBR
**target per eye**, so budget roughly twice that plus transport overhead; actual
rates depend on picture content and encoder rate control. Change `output.gop` to
control the IDR interval in output frames, and `output.rotation` for 0/180-degree
receiver display rotation. The default GOP is one second.

Supported modes observed on this attached ZED Mini:

| Per-eye size | Advertised capture FPS |
|---|---|
| 1280×720 | 60, 30, 15 |
| 672×376 | 100, 60, 30, 15 |
| 1920×1080 | 30, 15 |
| 2208×1242 | 15 |

The development Thor's local Sensing profile uses 2560×1984 at 60 FPS and 50 Mbps
per eye; that deployment-specific profile is not included in this branch.
This ZED cannot match that capture resolution at 60 FPS. The default matches the
configured frame rate and bitrate target; choose a mutually supported output size
for a direct comparison. Matching settings alone does not establish equal latency.

Useful checks, from the camera_streamer directory:

```bash
./zed_sender/run.sh --list-devices
./zed_sender/run.sh --check-config
./zed_sender/run.sh --device /dev/video8 --port 5000
```

`auto` requires exactly one accessible ZED image node and excludes metadata nodes.
A `/dev/v4l/by-id/...-video-index0` path is also accepted for explicit selection.
`--frames N` exits after sending N stereo bundles; zero means continuous operation.

## Capture, encoding, and timing

V4L2 captures a single YUYV image containing both eyes. A capture thread continuously
drains at most four MMAP buffers and retains the latest complete stereo frame.
The processing thread uploads YUYV to CUDA and performs a BT.601-limited → RGBA conversion, then
NVIDIA's surface transform crops/scales each eye and converts to BT.709-limited
NV12. Two hardware HEVC encoders use no B frames and emit a complete access unit
per eye. The input surfaces are reused only after both encoders release them.

The explicit color conversion is deliberate: synthetic tests on this Thor BSP
found incorrect colors with its direct YUYV transforms. The verified CUDA conversion
avoids that issue; HEVC encoding and image scaling remain hardware accelerated.
The HEVC VUI explicitly signals BT.709 and limited range.

Both eyes share a USB frame number and one AVS1 bundle. Encoded frames are never
dropped from an active reference chain; bounded TCP backpressure instead permits
newer **unencoded** capture frames to replace older ones. A reconnect creates new
encoders, a fresh session ID, and independently decodable IDR pictures.

USB dequeue observations use the sender's realtime clock. They are capture-ready
times, **not** sensor exposure/start-of-frame times. Tegra TSC and sensor timestamp
fields are unavailable. Thus Argus SOF-to-consumer measurements are unavailable;
capture-ready-to-consumer comparisons across machines require synchronized clocks.
No stereo rectification, depth, IMU, intrinsics/extrinsics, or calibration is added.
The output is the raw camera view after color conversion and optional resizing.

## Verification

CPU tests (no camera access):

```bash
ctest --test-dir zed_sender/build --output-on-failure
python3 -m pytest -q camera_receiver/tests
```

Opt-in hardware tests require an idle camera for the live test and GStreamer
NVIDIA plugins plus Python NumPy for the color test:

```bash
cmake --build zed_sender/build --target zed_encoder_pattern
python3 zed_sender/tests/hardware_pattern.py
python3 zed_sender/tests/hardware_smoke.py --frames 300 --sessions 2
python3 zed_sender/tests/hardware_smoke.py --config zed_sender/configs/zed_low_bandwidth.yaml
```

The color test checks eight changing stereo patterns, eye order, 2× scaling,
frame reuse, BT.709 bitstream signaling, and decoded color values. The live test
uses the actual receiver AVS1 parser, validates paired frames and timestamps,
reconnects, measures received FPS/Mbps, and saves independent per-eye HEVC files.
Capture-ready-to-receive measurements are local loopback measurements excluding
sensor exposure, network transit to a workstation, decode, and display.

## Results on the development Thor (2026-11-20)

Final loopback runs used the real attached ZED Mini and both eyes:

| Profile | Received FPS | Combined wire Mbps | Capture-ready → receive, mean | p95 |
|---|---:|---:|---:|---:|
| 1280×720, target 60 FPS | 58.0–59.0 | 96.5–98.6 | 17.9–20.8 ms | 31.0–33.1 ms |
| 640×360, target 30 FPS | 29.8–30.1 | 7.0–7.5 | 13.5–13.8 ms | about 14.0 ms |
| 640×360, target 25 FPS from 60 FPS capture | 25.0 | 11.1 | 13.6 ms | about 14.0 ms |

These are short local tests (two 300-frame sessions at 720p, two 150-frame
sessions at 360p, and one 125-frame rate-selection test), not performance guarantees.
The 60 FPS profile still skips some source frames; do not assume sustained 60 FPS.
Raw V4L2-only capture also exhibited intermittent gaps in earlier runs. Encoding,
scheduling, USB delivery, picture content, and other machine load affect results.
Bitrate targets differ from actual traffic, especially for these short recordings.

Both profiles reconnect with fresh sessions and valid paired-eye AVS1 metadata.
The changing synthetic stereo pattern decoded all eight frames per eye, with
maximum central-patch error 1.25 levels out of 255 and correct BT.709 signaling.
The original Argus sender builds, all 12 Argus CTest cases pass (the socket test
requires host socket access), the ZED config test passes, and 75 Python receiver /
deployment tests pass. Both receiver profiles pass their existing dry-run validation.

The separate workstation's CUDA IPC receiver and graphical preview have **not**
been run remotely in this session. Local validation covers the actual AVS1 parser
and independent NVIDIA hardware decode. This Thor does not have the workstation
receiver's `libnvcuvid.so` dependency; the independent decode test uses its installed
NVIDIA GStreamer decoder. Use the workstation commands above for that final check.

Measurements and sender logs remain with the tested deployment export under
`/home/nvidia/camera_streamer/zed_sender/validation/2026-11-20/`; generated records
are not checked into this branch. After testing, the camera was released and
restored to its prior 1280×720-per-eye, 30 FPS USB mode, and the sender was stopped.

## Source and upstream base

Branch `rahulsi/zed_streamer` starts from upstream main commit
`84ef6e26b21188497757f78da63bfac01121b1b3`. Its base `examples/camera_streamer` tree
matches PR #1410 head `4063d5c449859efde5f5edd1982341fa91fefe63`, reviewed on
2026-11-20. The Git checkout on the development Thor is
`/home/nvidia/IsaacCapture-zed-streamer`; `/home/nvidia/camera_streamer` is the
separate deployment export used for the hardware measurements above.

The branch contains the ZED implementation and its shared-code changes. Local
Sensing launch/environment settings, calibration assets, and Thor-specific
profiles from the deployment export are excluded. ZED additions live in
`examples/camera_streamer/zed_sender/`.
AVS1 serialization and TCP transport are reused from `argus_sender/src`; the
clock helpers were extracted into `Clock.h/.cpp` so the ZED build does not depend
on Argus headers or libraries. Keep the sibling `argus_sender/src` files when
copying/building the ZED sender. The receiver changes specific to ZED are the
optional hardware-clock metadata label and the two new configuration profiles.

PR #1410 is the camera streaming work, with optional sender-side camera remaps.
Those remaps do not provide ZED calibration and are not used by this sender.
