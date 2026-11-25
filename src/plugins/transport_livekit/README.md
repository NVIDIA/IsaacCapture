<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LiveKit SDK integration helpers

This independently installable package pins the public LiveKit SDK and provides
connection lifecycle, opaque data-channel handling, pre-encoded publication and
SDK timing helpers. The transport adapter and plugin registration are added in
the next review. No SDK binary or private download is shipped here.

Install the core and SDK helpers together:

```bash
uv pip install src/core/media_transport src/plugins/transport_livekit
```

The pre-encoded path uses internal SDK APIs pinned in `pyproject.toml`; check
those APIs when upgrading. Native receive support is an optional local build:

The native receive extension is built from the public `livekit/rust-sdks` revision
pinned by `build_livekit_cuda.py`. Its patches are checked in alongside the helper.
Build it on the **host** architecture, outside the installed Python SDK:

```bash
export IC_DIR="$HOME/IsaacCapture"
cd "$IC_DIR"
python3 src/plugins/transport_livekit/build_livekit_cuda.py \
  --build-dir "$HOME/.cache/isaac-livekit-native" \
  --output "$HOME/.local/lib/liblivekit_ffi.so" \
  --timing --playout-control
```

This requires Rust/Cargo, Git with submodules, and the C/C++ toolchain required by
the pinned LiveKit SDK. Use a compiler compatible with its prebuilt WebRTC libc++.
The helper checks the source revision and refuses incompatible reuse of a build
tree. Set `livekit_cuda_library` to the result before starting the receiver. A
native extension must be loaded before any other LiveKit session in that process.
It is an opt-in local build, not a binary bundled in the Python wheel.

`livekit_playout_mode: immediate` bypasses native playout scheduling for latency
experiments. It requires `--playout-control`; `default` preserves SDK scheduling.
Immediate mode can expose network jitter and does not eliminate stereo waiting.
The CUDA path is currently H.264, complete pictures, no B-frames.
