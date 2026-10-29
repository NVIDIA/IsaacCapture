<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Isaac Camera Streamer — GMSL / Argus

This component streams Argus camera data using AVS1/TCP. The protocol and
configuration tests run without camera hardware.

## Protocol and configuration tests

```bash
cmake -S examples/camera_streamer/argus_sender -B /tmp/camera-protocol-build
cmake --build /tmp/camera-protocol-build --parallel 2
ctest --test-dir /tmp/camera-protocol-build --output-on-failure
```

See [the wire specification](argus_sender/WIRE_FORMAT.md). Shared camera
profiles are in `configs/`; select sensor modes using the installed camera driver.

## Licensing

This example is Apache-2.0 licensed. Existing upstream notices and external SDK
licenses remain in effect.

## Rebuilding after a source-directory move

CMake caches absolute source paths. After updating an existing deployment, configure
and build the native sender and decoder again at the paths shown above. Do not move
or reuse an old build directory from a different source location.
