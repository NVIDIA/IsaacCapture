<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Media transport core

`isaac-media-transport` defines message/video contracts and discovers independently
installed plugins. It does not import a vendor SDK during discovery. Applications
select one registered name; a missing or incompatible plugin fails explicitly.
There is no fallback to a different transport.

The package is standalone: installing it does not require Isaac Capture's XR
runtime, a camera SDK, LiveKit, or CUDA. GPU helpers import their dependencies only
when selected. Build/install it directly with `uv pip install src/core/media_transport`.

## Interfaces

| Contract | Responsibility |
|---|---|
| `TransportAdapter` | Capabilities and factories for sessions, receivers and provisioning |
| `MessageSession` | Start/stop, readiness, opaque topic payloads and delivery selection |
| `VideoSession` / `VideoPublication` | Publish tracks and submit complete encoded pictures or supported RGBA frames |
| `EncoderFeedback` | Keyframe request and optional target bitrate/frame rate for the upstream encoder |
| `FrameReceiver` | Latest owned decoded mono/stereo sample; bounded delivery and observable overwrites |
| `TransportProvisioner` | Generate deployment configuration and credentials; no service startup |

```python
from isaac_media_transport.transport import get_transport

adapter = get_transport({"transport": "livekit"})
print(adapter.capabilities.encoded_codecs)
```

Applications own camera capture and encoding. `capture_encoded(payload, keyframe,
timestamp_us, index)` accepts one complete access unit; input bytes remain valid
through the call. Implementations must copy or consume the input before returning.
Its Boolean result reports acceptance, not delivery or rendering. Feedback must
reach the encoder: a keyframe flag cannot turn a delta picture into a keyframe.

The CUDA H.264 path requires one picture per access unit and no B-frames. On a
sequence gap the transport adapter must reset the decoder and discard dependent
pictures until a fresh keyframe arrives.
Do not apply a newest-frame policy to arbitrary compressed delta pictures.

Encoded submission timestamps are microseconds; `FrameMetadata.user_timestamp`
is nanoseconds in the sender's clock domain. Preserve source IDs and timestamps;
a host arrival timestamp is not a sensor capture time. Cross-host latency requires
a qualified clock mapping. Stereo matching compares source timestamps only, with
bounded skew/wait; it does not establish hardware exposure synchronization.

A `VideoFrame` owns its pixels. Retaining it keeps the allocation alive; CUDA
producer work is complete before publication. `to_bytes()` performs an explicit
GPU readback for diagnostics. `IpcServer` exports acknowledged GPU leases;
`IpcReceiver` copies on the GPU before acknowledging, so consumers retain their
own frames. Both endpoints must use the same GPU and ordered track names.

The core has no controller safety policy. `reliable=True` does not guarantee
cross-topic ordering, application acceptance, or suitability of a stale command.
Applications must define freshness and recovery semantics.

## External plugin registration

A separate package registers the versioned entry-point group:

```toml
[project.entry-points."isaac_media_bridge.transports.v1"]
example = "example_transport.plugin:PLUGIN"
```

```python
from isaac_media_transport.transport_plugins import TransportPlugin


def create():
    from .adapter import ExampleTransport

    return ExampleTransport()


PLUGIN = TransportPlugin(name="example", api_version=1, create=create)
```

The descriptor is SDK-free. The factory imports the selected implementation and
returns the adapter. Duplicate names, incompatible API versions and incomplete
adapters fail during selection. Keep dependency declarations in each plugin's
package; do not make the core depend on all supported SDKs.

Use the same entry-point group in out-of-tree packages. The example installer
accepts `--plugin-project /path/to/project --transport example`; alternatively,
install an external wheel in the application's environment. No core edit is needed.

The `livekit` settings section in the example is a compatibility name. The selected
backend is its `transport` value. Adapters receive this settings dictionary; they
must not import the example or camera implementation.
