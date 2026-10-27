<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Standalone import validation — 2026-10-26

Original source hashes for retained files are in `UPSTREAM_FILES.json`.
The review scope retains the native Sensing/Argus sender, NVDEC receiver and
CUDA IPC consumer. The complete multi-camera snapshot is preserved on a separate local branch.

| Check | Result |
|---|---|
| Independent editable install | Passed in a new workstation Python 3.12 environment |
| Receiver tests | 28 passed (wire format, reconnect, decoder argument forwarding) |
| Orin sender build | Passed with installed JetPack development headers |
| Native sender tests | 12 passed |
| Workstation NVDEC extension | Built with local CUDA 13.3 toolchain |
| Sensing camera enumeration | Sensors 0/1 expose 2560×1984 at 30 fps as mode 1 |
| Live AVS1 HEVC stereo | Orin → workstation, both streams decoded at 30 fps after startup |
| CUDA IPC consumer | Opened both 2560×1984 NV12 rings; sampled means 106.55 left, 103.97 right |

Live testing used the isolated Orin directory
`~/isaac-camera-streamer/camera_streamer`, TCP port 5000 and workstation
socket `/tmp/isaac-camera-streamer.sock`. Bounded test processes were stopped.


These are functional bring-up results, not a controlled latency benchmark. The
optional CuPy mean diagnostic required NVRTC 12 alongside the host's CUDA 13;
first-use JIT introduced roughly a second of diagnostic delay. Exclude sampling
and startup windows from throughput/latency comparisons. No CloudXR headset
visual validation was performed for this copy.

The owners approved public review of the Sensing-only extraction on 2026-10-27.
Existing unresolved license identifiers are retained for maintainer review.

## Sensing-only scope check — 2026-10-27

After pruning unrelated implementations, 28 receiver tests passed, the receiver
default configuration passed `--dry-run`, and the reduced package installed
editable without camera SDK dependencies. The retained native sender code was
unchanged; its 12 tests passed on the existing Orin build. Full pre-commit checks
passed. No live hardware capture was repeated for this scope-only change.
