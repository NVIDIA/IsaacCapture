<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Jetson Multimedia API encoder classes

The SENSING producer drives the Jetson hardware H.264 encoder through NVIDIA's `NvVideoEncoder` class and the
V4L2 element classes it is built on. The Jetson Multimedia API documents these classes and ships their headers
in `include/`, but ships their implementation only as source, in the package's `samples/common/classes/`
directory: the package contains no prebuilt library for them, and NVIDIA's own samples compile them in. So they
are copied here and compiled into `sensing_core`, instead of being compiled from the installed `/usr/src` tree.

- **Source:** the `nvidia-l4t-jetson-multimedia-api` package, version 39.2.1-20260806224157 (L4T R39.2.1),
  from `https://repo.download.nvidia.com/jetson/common/pool/main/n/nvidia-l4t-jetson-multimedia-api/`. The files
  were taken unchanged from `/usr/src/jetson_multimedia_api/samples/common/classes/` (`classes/`) and
  `/usr/src/jetson_multimedia_api/include/` (`include/`), except for the change below.
- **Files:** `classes/` holds `NvVideoEncoder`, `NvV4l2Element`, `NvV4l2ElementPlane`, `NvElement`,
  `NvElementProfiler`, `NvBuffer` and `NvLogging` (`.cpp`); `include/` holds the matching headers. This is the
  complete set the encoder needs; the producer calls only the encoder and its planes.
- **License:** BSD-3-Clause. Every file carries its own copy of the license text in its header and keeps it;
  [`LICENSE`](LICENSE) repeats the text, and the build installs it beside the plugin binary under
  `licenses/jetson-multimedia-api`. The package's own copyright file names the NVIDIA Driver License Agreement as
  the default for files that carry no license of their own; none of the files here is in that category.
- **Local change:** `classes/NvVideoEncoder.cpp` has the four `cout` lines in `setBitrate()` and `setPeakBitrate()`
  removed (they printed the plane pixel-format codes to stdout on every encoder start). The two places are marked
  with a comment.

## Not copied

These headers stay in the installed tree (`JETSON_MMAPI_DIR`, default `/usr/src/jetson_multimedia_api`), because
their license does not allow copying them here or they must match the installed driver:

- `nvbufsurface.h`: NVIDIA proprietary license.
- `libv4l2.h`: LGPL-2.1.
- `v4l2_nv_extensions.h`: BSD-3-Clause, but the driver's own interface, which has to match the installed L4T.

The Argus headers and the `libnvbufsurface` and V4L2 libraries also come from the installed JetPack.

## Updating

Extract the package for the target release (`dpkg-deb -x`), copy the same fourteen files over these, compare the
result against the previous copy, and reapply the local change above. Check that the license headers are unchanged.
