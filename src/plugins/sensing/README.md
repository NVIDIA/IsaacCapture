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
| `MOCK` | animated GIF frames | stand-in | NVIDIA GPU, CUDA toolkit, EGL headers (`libegl-dev`) |
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

The mock encoder's output is **not decodable video**: Annex-B start codes, SPS/PPS/IDR and
non-IDR NAL types, and a payload naming the frame. It reproduces what callers depend on: a
fixed pool of input buffers that makes `submit()` refuse frames when the encoder falls behind,
a warm-up delay before the first unit, and capture stamps that come back at microsecond
resolution.

Every sensor shows an animated GIF chosen by its **module name** (never its pipeline index, which the
link mask renumbers), from the bundled Animated Noto Emoji in [`mock_assets/`](mock_assets/) (CC BY 4.0;
see its README): `S56C_HEAD` the robot, `SHF3L_LEFT` the waving hand, `SHF3L_RIGHT` the hand flipped left
to right. The two sensors of `S56C_HEAD` are its left and right eyes; their pictures are offset against
each other by a fixed 2% of the width, as a 60 mm baseline gives. A module not in that table needs
`SENSING_MOCK_GIF` or `SENSING_MOCK_GIF_<module name>`. The GIF plays on the capture clock, looping, scaled
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
| `SENSING_MOCK_GIF` | path of an animated GIF shown by every sensor, as it is (default: by module, above) |
| `SENSING_MOCK_GIF_<module name>` | a GIF for one sensor, by module name (`SENSING_MOCK_GIF_SHW5G_MIX_L3`); wins over `SENSING_MOCK_GIF` |
| `SENSING_MOCK_FLIP_<module name>` | `1` or `0`: flip that module's picture left to right, or not, whatever its GIF |
| `SENSING_MOCK_FPS` | override the platform config's frame rate |
| `SENSING_MOCK_SCALE_DIV` | divide the reported width and height (kept even); useful for tests |
| `SENSING_MOCK_FAIL_AFTER` | every sensor fails after N frames, as a link drop would |
| `SENSING_MOCK_ENCODER_LATENCY_MS` | per-frame encode time |
| `SENSING_MOCK_ENCODER_DEPTH` | frames held back before the first unit (default 2) |

A pass on the mock says nothing about SIPL, the ISP output format, NvSci buffer lifetime, the
V4L2 encoder or real frame rates. Those are only checked on an Orin with SENSING cameras.
