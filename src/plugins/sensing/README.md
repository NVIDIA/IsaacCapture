<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SENSING GMSL cameras

SENSING connects GMSL cameras to Jetson through the SG8A carrier. This
integration uses NVIDIA SIPL for camera access and provides the tooling to
provision and validate the rig on the host or in a container.
The C++ plugin captures frames through SIPL and writes H.264 video or shares
RGBA frames with another process through CUDA IPC.

Start with the [SENSING user guide](https://nvidia.github.io/IsaacCapture/main/device/camera/sensing.html)
for supported hardware, installation, container usage, and troubleshooting.
That guide owns the operating instructions; this README maps the source tree.

- [`app/`](app/) and [`core/`](core/): plugin entry point, SIPL capture,
  encoding, and output sinks.
- [`cuda_ipc/`](cuda_ipc/) and [`tools/`](tools/): frame transport and a
  synthetic producer for development without a camera.
- [`scripts/`](scripts/): host setup, validation, diagnostics, and the Bash
  compatibility Docker launcher. Each entry point exposes `--help`.
- [`docker/`](docker/): example image, image-side group configuration, and
  host-side container setup.
- [`configs/`](configs/): the platform configuration and its provenance.
- [`core/mock_*`](core/): the mock backend (see below).
- [`livekit/`](livekit/): the LiveKit SDK behind a small interface, for streaming (see below).

## Platform configuration provenance

[`configs/shw5g.json`](configs/shw5g.json) is a byte-identical copy supplied
by SENSING for the two-SHW5G rig, from:

```text
SG8A_AGON_G2Y_A1_AGX_ORIN_S56C_SHW5G_SHF3L_SIPL_JP7.2.1_L4TR39.2.1
  /query/sg8a_agth_g2a/shw5g.json
```

The public driver package includes only the mixed-camera configurations,
`s56c_shf3l_mixed.json` and `shw5g_shf3l_mixed.json`, not this `SHW5G_2`
configuration. Vendoring it makes validation independent of a downloaded
vendor package.

Keep the file byte-identical to the vendor copy. Compare new vendor drops
before replacing it; do not hand-edit the board wiring. Its CSI ports, I2C
buses, deserializer addresses, GPIOs, and link modes must match the installed
SIPL drivers.

## Mock backend: run the plugin without a Jetson

`SENSING_BACKEND` selects what sits behind `SiplCamera` and `JetsonEncoder`:

| Value | Capture | Encoder | Needs |
| --- | --- | --- | --- |
| `JETSON` | SIPL | V4L2 NVENC | L4T SDKs |
| `MOCK` | animated GIF frames | NVENC through libavcodec | NVIDIA GPU, CUDA toolkit, EGL headers (`libegl-dev`), `libavcodec-dev` |
| `AUTO` (default) | `JETSON` on aarch64, `MOCK` elsewhere | | |

```bash
cmake -B build -DBUILD_PLUGIN_SENSING=ON -DSENSING_BACKEND=MOCK
cmake --build build --target camera_plugin_sensing sensing_mock_tests
./build/src/plugins/sensing/app/camera_plugin_sensing --list-sensors
```

Everything above those two classes is the code that runs on the rig: `SensingCamera`, the
sinks, the YUV-to-RGBA and RGBA-to-NV12 kernels, and the CUDA IPC publisher. The mock
resolves `--list-sensors` and the link masks from the vendor JSON, builds a synthetic NV12
surface from a GIF frame, and runs the real conversion on it. Both sensors share a capture
TSC, as the rig's fsync does.

The mock encoder produces real, decodable H.264 on the desktop GPU's NVENC (libavcodec's `h264_nvenc`),
configured as the Jetson encoder is: High profile, Level 5.2, VBR with a peak, an IDR every GOP with
SPS/PPS in front of it, no B-frames. It reproduces what callers depend on: a fixed pool of input
buffers that makes `submit()` refuse frames when the encoder falls behind, capture stamps that come
back at microsecond resolution, and the tail returned as one unit at end of stream. Bitstream bytes,
unit sizes and latency are NVENC's, not the Jetson engine's.

Every sensor shows an animated GIF chosen by its **module name** (never its pipeline index, which the
link mask renumbers), from the bundled Animated Noto Emoji in [`mock_assets/`](mock_assets/) (CC BY 4.0;
see its README): `S56C_HEAD` the robot, `SHF3L_LEFT` the waving hand, `SHF3L_RIGHT` the hand flipped left
to right. The two sensors of `S56C_HEAD` are its left and right eyes; their pictures are offset against
each other by a fixed 2% of the width (illustrative, not calibrated geometry). A module not in that
table is an error. The GIF plays on the capture clock, looping, scaled
to cover the sensor's configured size and centre-cropped so it is never stretched (bilinear when
enlarging, area-averaged when reducing). Transparent pixels show a plain gray. The GIF's own size and
aspect do not matter; the frame size always comes from the platform config. A GIF that cannot be loaded,
the bundled one included, is an error: there is no fallback.

[`mock_assets/mock_rig.json`](mock_assets/mock_rig.json) describes a three-camera rig for the mock only
(not a SIPL config): one S56C stereo head camera with two 2560x1984 @ 30 eyes, and two SHF3L wrist
cameras at 1920x1536 @ 30. Use `--platform-config=.../mock_rig.json --config-name=MOCK_S56C_2_SHF3L
--link-masks='0x0000 0x0111'`.

Environment, read once at construction:

| Variable | Effect |
| --- | --- |
| `SENSING_MOCK_FPS` | override the platform config's frame rate |
| `SENSING_MOCK_SCALE_DIV` | divide the reported width and height (kept even); useful for tests |
| `SENSING_MOCK_FAIL_AFTER` | every sensor fails after N frames, as a link drop would |

A pass on the mock says nothing about SIPL, the ISP output format, NvSci buffer lifetime, the
V4L2 encoder or real frame rates. Those are only checked on an Orin with SENSING cameras.

## LiveKit streaming

`livekit=<track>` on an `--add-stream` publishes that sensor as a LiveKit camera track, from the real
cameras or the mock alike: the code is the same on both backends. It is off unless built with
`-DSENSING_LIVEKIT=ON`, which downloads the pinned LiveKit C++ SDK package (Ubuntu 24.04, x64 or
arm64, checked against its hash) at configure time; set `LIVEKIT_SDK_DIR` to use an unpacked copy
instead.

```bash
camera_plugin_sensing --livekit-url=wss://livekit.example.org --livekit-token-file=join.token \
    --add-stream=sensor=0,livekit=sensing-left --add-stream=sensor=1,livekit=sensing-right
```

- **Hardware-encoded passthrough.** Each track carries H.264 that this machine's hardware encoder
  produced, published through the SDK's pre-encoded source (`EncodedVideoSource`, encoder backend
  `PreEncoded`): the SDK forwards each access unit and does not encode. The encoder is the plugin's
  `JetsonEncoder` class, so it is the V4L2 engine on a Jetson and NVENC through libavcodec on the
  mock. There is no software encoder and no fallback: if the hardware encoder cannot start,
  construction fails. The status line names the encoder and the passthrough mode.
- **Token.** The operator supplies it in a file, never on the command line. It names the room and
  the participant and must allow publishing; viewers use a separate subscribe-only token. It is not
  printed.
- **Tracks.** One H.264 track per sensor, no simulcast. The default size is the sensor's configured
  size. `--livekit-scale-div=N` reduces every track by an integer divisor that must divide the
  sensor exactly and leave even sizes (2 gives 1280x992 from 2560x1984); it is refused otherwise,
  never rounded or stretched, and the encoder runs at the reduced size. `--livekit-fps` caps the
  publication rate on the capture clock (default 30). `--livekit-bitrate` sets the encoder's target
  (default 8 Mbit/s; the VBR ceiling is half as much again), unlike the recording's `--bitrate`.
- **Frames.** `update()` takes a GPU snapshot of the camera frame and finishes that copy before
  returning, because the camera may rewrite its slot at the second following poll. A worker per
  sensor then reduces it if asked, hands it to the hardware encoder and publishes each encoded unit
  as one access unit, with its own stamp. A slow network drops frames before encoding and cannot
  hold up capture or the other sensor: one snapshot is being encoded, one waits, and a newer capture
  replaces the waiting one.
- **Feedback.** A viewer joining, or packet loss, makes the SDK ask for a keyframe: the next frame
  is an IDR with SPS/PPS. A frame the SDK refuses breaks the reference chain, so the next frame is
  an IDR and nothing is published in between. Bitrate targets from congestion control are clamped
  to the configured range and applied to the encoder: the Jetson engine changes rate while running;
  the mock reopens its session at the new rate, which starts with an IDR.
- **Metadata.** Each frame carries the camera's sequence as its frame ID, the capture time on the host
  clock (CLOCK_MONOTONIC microseconds) as its user timestamp, and the sensor's capture TSC as eight
  little-endian bytes of user data. Two separately published tracks are not exposure-synchronised;
  the shared TSC is what pairs them.
- **Recording.** A stream with `output=` and `livekit=` runs two encoders, one for the file at the
  recording's settings and one for the track at the live settings. `--mcap-filename` metadata is
  separate and unchanged.
- **Failure.** Losing the room, or a failed publish, ends capture with an error; there is no
  fallback to files or another transport. The SDK's own reconnection runs first; on loopback, losing
  the server was reported after 15 to 30 s.

**Verified.** On a desktop GPU against a local `livekit-server`, with a second participant decoding
what arrives: tracks, sizes and picture content, frame IDs and capture TSCs, a viewer joining late
and rejoining, keyframe requests, recovery after a refused frame, the end-of-stream drain, lower
bitrate targets, and the transport's stats naming the passthrough source. The arm64 SDK package has
the same pre-encoded API headers as the x64 one.

**Not verified.** Nothing here ran on an Orin: not real cameras, not the V4L2 encoder feeding
LiveKit, not the arm64 package, and not the `JETSON` build with LiveKit. The Jetson encoder's
keyframe, rate and drain code is written but was not compiled off the rig. That the H.264 the
Jetson engine produces (High profile, Level 5.2) is accepted by a WebRTC receiver is likewise
unproven; the mock's NVENC output, with the same settings, decoded in the tests.

**Receiver page.** [`livekit/receiver/index.html`](livekit/receiver/index.html) is a static page that
uses the LiveKit browser SDK (from a CDN, integrity-checked): enter the server URL and a subscriber
token, Connect, and each subscribed video track appears under its name with its size, in its own
aspect ratio. There is no backend or token generator; it is installed beside the plugin under
`receiver/` and can be opened from disk or any static server.

**Capacity on a Jetson (not measured).** The numbers below are calculated encoder workloads from the
module datasheets, not achieved frame rates; nothing has run on an Orin. They assume head images at
2560x1984 and wrist images at 1920x1536.

| Images | Rate | Encoded pixels/s | Assessment |
| --- | --- | --- | --- |
| Head stereo, native | 30 fps | 305 million | plausible first target |
| Head stereo and both wrists, native | 30 fps | 482 million | depends on the preset; little headroom |
| All four at half width and height | 30 fps | 120 million | more headroom |
| All four, native | 60 fps | 963 million | not a supported target |

The datasheets list H.264 ceilings of 930/530/280 MPixel/s (UHP/HP/HQ) for the AGX Orin 64GB module,
730/420/220 for the 32GB module and 680/470/220 for Orin NX; four native images at 30 fps already
exceed the NX and 32GB HP ceilings. Orin Nano has no NVENC, so it cannot meet the hardware-encoder
requirement. The encoder selects the V4L2 `FAST` hardware preset explicitly (a control separate from
max-performance clocks); four native images may need `ULTRAFAST`, to be measured. A stream with both
`output=` and `livekit=` runs two encoder instances on the one hardware engine, so budget the sum of
their pixel rates. Sustained rates also depend on power mode and cooling. A frame the encoder has no
input buffer for is dropped before any conversion work. Moving frames without the intermediate
copies (a direct NV12 or DMA-BUF path) is a separate change to make only if measurements call for it.

`cmake --install` copies `liblivekit.so` and `liblivekit_ffi.so` beside the plugin, which finds them
through `$ORIGIN`, and the SDK's own `LICENSE` and `NOTICE` (fetched at configure time, hash
checked) under `licenses/livekit-sdk/`. `liblivekit_ffi.so` bundles third-party components, WebRTC
among them, whose notices the SDK package does not ship: they need a compliance review before the
binaries are distributed.

Tests that need a LiveKit service start a local `livekit-server` when `LIVEKIT_SERVER` names its
binary, and are skipped without it; the rest need only a GPU.
