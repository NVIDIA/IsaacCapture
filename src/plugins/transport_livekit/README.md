<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LiveKit transport plugin

`isaac-transport-livekit` implements the media transport contract using LiveKit
rooms, scoped participant credentials, data channels, and video tracks. It is an
optional standalone package under `src/plugins`; it is not a tracked-device plugin
and does not launch through the XR plugin manager.

Install the core and this package together with
`uv pip install src/core/media_transport src/plugins/transport_livekit`.
Dependencies are pinned in `pyproject.toml`. The pre-encoded sender uses native
FFI definitions from LiveKit 1.1.20; these are internal SDK APIs and must be checked
when upgrading. No private SDK download is required.

## CPU and CUDA reception

The ordinary SDK receiver returns CPU frames. Hardware decoding can be required
with `video_decoder: hardware`; this still does not make the returned frames CUDA
allocations. `decoded_memory: cuda` instead selects a patched encoded-receive
interface followed by NVDEC and an owned CUDA RGBA allocation. CUDA IPC can then
hand those frames to a separate local process.

For CUDA reception, install the GPU packages into the receiver's interpreter
(replace the example path if your environment differs):

```bash
export RECEIVER_PYTHON="$HOME/IsaacCapture/examples/remote_assistance/.venv/bin/python"
uv pip install --python "$RECEIVER_PYTHON" PyNvVideoCodec==2.0.2 cupy-cuda13x
```

Use `cupy-cuda12x` instead for a CUDA 12 installation; install only one CuPy variant.
CPU reception does not require these packages.

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

## Deployment

The example provisioner creates a LiveKit server configuration, verified TLS,
participant tokens and separate host/client bundles. The server uses the public
LiveKit image. Tokens expire. After regeneration, deploy the matching host and
client bundles, reload each site's credentials and CA configuration, then restart
the server and clients. An old bundle or shell environment will fail authentication.
Camera presets are supplied by the application, not this package.

The generated network is direct LAN/VPN connectivity. No TURN service is
provisioned. Firewalls or Wi-Fi client isolation can prevent media even when TLS
signaling works. See the example's network setup and diagnostics.
