<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
SPDX-License-Identifier: Apache-2.0
-->

# AgiLink OmniGlove plugin

This plugin converts the OmniGlove SDK's wrist-relative 21-landmark
`OMNIGLOVE_STREAM_POST_DATA` stream to OpenXR hand joints and injects it into
the Isaac Capture runtime. It supports one left and one right USB glove.

## Install

Install the host USB rule once, outside a container:

```bash
./src/plugins/omniglove/install_udev_rules.sh
```

Configure and build from the Isaac Capture repository root:

```bash
cmake -S . -B out -DBUILD_PLUGIN_OMNIGLOVE=ON
cmake --build out
cmake --install out --component omniglove
```

CMake retrieves a pinned SDK release into the build directory, verifies its
checksums, and selects the package for the target architecture. It installs
runtime libraries and license notices with the plugin. Set
`-DOMNIGLOVE_SDK_ROOT=/path/to/agilink_omniglove_sdk` to explicitly use a local
release without modifying it, or `-DOMNIGLOVE_SDK_REF=<commit-or-ref>` to select
a managed release. A full commit SHA makes an override reproducible.

Missing or changed SDK APIs fail compilation or linking. At runtime, a
header/library ABI mismatch is rejected before device discovery.

## Run

Start CloudXR, load its environment, connect a calibrated glove, and run:

```bash
python -m isaaccapture.cloudxr.service start
source ~/.cloudxr/run/cloudxr.env
./install/plugins/omniglove/omniglove_plugin
```

The glove provides finger poses relative to its wrist but no world-space wrist
pose. By default the plugin obtains wrist pose from OpenXR optical hand
tracking. Set `OMNIGLOVE_WRIST_SOURCE` to `controller` or `auto` only after
measuring controller-aim-to-wrist transforms and exporting them as
`px,py,pz,qx,qy,qz,qw`:

```bash
export OMNIGLOVE_WRIST_SOURCE=controller
export OMNIGLOVE_AIM_TO_WRIST_LEFT="px,py,pz,qx,qy,qz,qw"
export OMNIGLOVE_AIM_TO_WRIST_RIGHT="px,py,pz,qx,qy,qz,qw"
```

Replace these placeholders with measured offsets for the current mount.
Controller and Auto modes reject missing transforms. Unknown wrist modes and
malformed offsets are errors; optical mode does not read the offsets.

## Data contract

- SDK positions are metres and quaternions are `(x, y, z, w)`.
- Finger orientations are converted from SDK distal +Z to OpenXR distal
  -Z by a local Y half-turn, preserving dorsal +Y. The wrist retains its
  proximal +Z; the right-hand wrist basis is converted separately.
- The 21 supplied wrist/finger landmarks map to their OpenXR equivalents.
- OpenXR palm and the index, middle, ring, and little metacarpals remain
  invalid because the SDK stream does not supply them.
- Unmapped, unconverged, invalid-bone-model, non-finite, zero-quaternion, and
  stale frames are not injected.

## Device status and CTS

The plugin publishes both hand states through the shared device status
publisher under the supplied `--plugin-root-id`. A stale SDK frame makes a
hand disconnected; a fresh frame without a usable wrist pose makes it degraded.
A glove that is discovered but fails stream configuration or startup stops the
plugin with an error.

Follow the official [hand CTS procedure](https://nvidia.github.io/IsaacCapture/main/conformance/hand.html)
and [submission procedure](https://nvidia.github.io/IsaacCapture/main/conformance/submit.html).
Keep take results and hardware evidence in the integration PR description.
