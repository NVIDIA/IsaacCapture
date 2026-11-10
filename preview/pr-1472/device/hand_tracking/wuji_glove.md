<!-- SPDX-FileCopyrightText: Copyright (c) 2026 Wuji Technology. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Wuji Glove

A Linux-only plugin for integrating [Wuji](https://www.wuji.tech/)
data gloves into the Isaac Capture framework. It reads the glove’s 21-joint hand
skeleton over the network via the wuji_sdk C API and injects the resulting
poses into the OpenXR hand-tracking layer, so any downstream consumer can read
them as standard hand tracking.

> ##### On this page
> 
> * [Data flow](#data-flow)
> * [Prerequisites](#prerequisites)
> * [Installation](#installation)
> * [Running the Plugin](#running-the-plugin)
> * [Wrist pose](#wrist-pose)
> * [Troubleshooting](#troubleshooting)

## Data flow

The plugin combines the glove’s wrist-relative skeleton with an absolute wrist
pose and publishes the result through OpenXR:

```text
Wuji glove (21-joint shape) ─┐
                             ├─► wuji_glove plugin ──► CloudXR OpenXR runtime
Optical hand / controller ───┘   (wrist-pose fusion)   (spatial hand joints)
```

#### SEE ALSO
[Retargeter: Wuji Glove to Wuji Hand](../../references/retargeting/wuji.md) explains how to map OpenXR hand tracking
to the 20 joint commands of a Wuji Hand or Wuji Hand 2.

## Prerequisites

- **Linux** — x86_64 or aarch64 (tested on Ubuntu 22.04 / 24.04).
- **Source-build prerequisites** — the toolchain and system packages in
  [Build from Source](../../development/build_from_source/index.md), for `install.sh` and the manual
  CMake build alike.
- **wuji_sdk C SDK v2026.7.14** — the
  `wuji-sdk-c-2026.7.14-<arch>-linux-gnu.tar.gz` GitHub release asset, which
  extracts to `include/wuji_sdk.h` plus `lib/libwuji_sdk_c.so` and its
  companion `lib/libwujihandcpp.so`. This is the C FFI, *not* the
  `wuji-sdk` pip package; the install script downloads it automatically.
- **Wuji glove(s)** reachable on the same LAN as the plugin host.
- **CloudXR runtime** running on the plugin host.

## Installation

`install.sh` downloads the pinned C SDK release, verifies it against a
per-architecture SHA-256, extracts it next to the script (re-runs reuse the
extracted copy), and configures, builds, and installs the plugin:

```bash
./src/plugins/wuji_glove/install.sh
```

Pass `--build-dir <path>` to reuse a non-default top-level CMake build
directory.

To build offline, point `WUJI_SDK_C_DIR` at an already-extracted SDK copy —
or drive CMake directly:

```bash
cmake -B build -DBUILD_PLUGINS=ON -DBUILD_PLUGIN_WUJI_GLOVE=ON \
    -DWUJI_SDK_INCLUDE_DIR=/path/to/wuji-sdk-c/include \
    -DWUJI_SDK_LIB=/path/to/wuji-sdk-c/lib/libwuji_sdk_c.so
cmake --build build --target wuji_glove_plugin
cmake --install build --component wuji_glove
```

## Running the Plugin

The plugin reaches the Teleop session through the CloudXR / OpenXR runtime, so
start the runtime first and source its environment in the shell that launches
the plugin:

```bash
python -m isaaccapture.cloudxr.service start   # runs in the background
source ~/.cloudxr/run/cloudxr.env
./install/plugins/wuji_glove/wuji_glove_plugin
```

See [CloudXR Service](../../references/cloudxr.md#dedicated-cloudxr-runtime) and
[Load CloudXR environment variables](../../references/cloudxr.md#load-cloudxr-environment-variables) for the full service setup. The plugin resolves each glove’s side from device
metadata and reconnects automatically after a disconnect; applications can also
launch it through its `plugin.yaml` using `TeleopSession`.

## Wrist pose

The glove stream contains wrist-relative joints but no absolute 6DoF wrist
pose. Before injection, the plugin places that skeleton at a wrist pose from
optical hand tracking or from a controller aim pose combined with a calibrated
per-hand rigid offset.

Set `WUJI_GLOVE_WRIST_SOURCE` to `auto` (the default), `hand_tracking`, or
`controller`. In `auto` mode, optical tracking is preferred; the controller
is used when optical tracking provides no valid pose. Override the built-in
controller offsets when needed:

```bash
export WUJI_GLOVE_AIM_TO_WRIST_LEFT="px,py,pz,qx,qy,qz,qw"
export WUJI_GLOVE_AIM_TO_WRIST_RIGHT="px,py,pz,qx,qy,qz,qw"
```

Position values are in meters and quaternion values use `(x, y, z, w)`.
The built-in offsets assume mirrored left/right mounts.

## Troubleshooting

| Symptom                                               | Fix                                                                                                                                                                                       |
|-------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `install.sh` fails with a SHA-256 mismatch            | The downloaded tarball does not match the pinned release. Re-run to<br/>rule out a corrupted download; if it persists, do not bypass the check<br/>— verify the release assets upstream.  |
| No glove is discovered                                | Confirm the glove is powered and on the same LAN/subnet as the plugin<br/>host, then check the plugin log for `wuji_scan` errors.                                                         |
| Log says a second same-side glove is ignored          | The plugin binds one glove per side (first discovered wins) and ignores<br/>later same-side gloves until restart. Power off the spare, or restart<br/>the plugin to rebind.               |
| Plugin runs but a downstream client sees no hand data | The plugin and the client must talk to the same CloudXR runtime; check<br/>that the runtime was up before both, and that hands are streaming<br/>(plugin prints per-glove connect lines). |
