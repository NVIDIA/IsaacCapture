<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeviceIO Live View

Draws live DeviceIO human tracking — hands, head, controllers, full body — in the
browser with [viser](https://viser.studio). Trackers that are inactive or absent
are hidden rather than drawn in an error color. `CloudXRLauncher` starts the
CloudXR runtime and WSS proxy itself, so there is nothing to launch separately.

```bash
uv pip install -e ./examples/deviceio_live_view
uv run --no-sync python -m isaaccapture_examples.deviceio_live_view --accept-eula
```

Open the URL it prints (default <http://localhost:8080>). It binds every
interface, so another machine on the network can reach it at
`http://<this-host>:8080`; pass `--host 127.0.0.1` to keep it local, `--port` to
move it. Ctrl+C stops it.

## SOMA body POC

The unreleased SOMA POC uses upstream FK and renders all 77 transported joints
in SOMA's native public order. It uses a neutral identity, not subject
calibration. Separate hand tracking is unchanged.

```text
SomaBodyJointRotations -> tracker -> upstream FK --+
                                                    +-> SomaBodyInput -> viewer
SomaBodyJointPoses ----> tracker -> direct map ----+
```

Use SOMA-X 0.3.1. For an uninstalled source build, set `PYTHONPATH` to the built
IsaacCapture package and this example's `python/` directory.

```bash
uv pip install -e ./examples/deviceio_live_view

uv run --no-sync python -m isaaccapture_examples.deviceio_live_view --accept-eula \
  --body-schema soma
```

Use the independent [SOMA body publisher](../soma_body_publisher/README.md) to
send the bundled motion, or connect another publisher using the checked-in SOMA
body schema.

Select the body input at launch with `--body-schema full-body-pose` (the default)
or `--body-schema soma`. Switch sources by restarting the viewer with the other
option. Hands, head, and controllers remain unchanged.

The SOMA viewer reads the `soma_body_demo` collection by default. Use
`--soma-body-collection-id <id>` to match another publisher's collection.
Joint rotations are the default SOMA profile. Pass
`--soma-body-representation joint-poses` when the publisher sends evaluated
joint poses instead. The publisher and viewer options must match.

`SomaBodySource` follows the normal DeviceIO source contract: the session discovers
the generated tracker selected for `SomaBodyJointRotations` or
`SomaBodyJointPoses`. The raw tracker remains the transport and recording
boundary. The source evaluates rotations once per graph step against the neutral
SOMA identity resolved by SOMA-X; evaluated poses map directly. Both modes emit
the same global positions, orientations, and validity as `SomaBodyInput`. The
viewer uses the pinned SOMA joint names and hierarchy. It does not convert SOMA
into `FullBodyPose`.

SOMA-X and PyTorch are source-evaluation dependencies, not tracker or transport
dependencies.

Missing data hides the body. For joint rotations, invalid translation invalidates
the complete skeleton. An invalid joint control also invalidates its descendants
because their global poses depend on that ancestor. Evaluated joint poses carry
independent per-joint validity. Other valid branches remain visible.
Body and hand sources are selected independently. Subject calibration, body-hand
fusion, a `FullBodyPose` compatibility retargeter, and recording/replay examples
are outside this live-view POC.

## SOMA hand POC

OpenXR `HandPose` remains the default hand input. Select independent left and
right SOMA hand collections with `--hand-schema soma`:

```text
SomaHandJointRotations -> tracker -> upstream FK --+
                                                        +-> SomaHandInput -> viewer
SomaHandJointPoses ----> tracker -> direct map -------+
```

```bash
uv run --no-sync python -m isaaccapture_examples.deviceio_live_view --accept-eula \
  --hand-schema soma
```

The rotation profile uses the neutral hand identity that SOMA-X resolves for FK.
Use `--soma-hand-representation joint-poses` for evaluated poses, which do not
require SOMA-X during per-frame source evaluation.
The default collections are `soma_hand_left_demo` and
`soma_hand_right_demo`; override them with
`--soma-left-hand-collection-id` and `--soma-right-hand-collection-id`.

The independent [SOMA hand publisher](../soma_hand_publisher/README.md) sends
both hands from the bundled animation. The viewer renders the 25-joint SOMA
topology directly. It does not merge these streams with the hand joints embedded
in a SOMA body payload.

The focused tests cover upstream FK agreement, body-source selection, native
SOMA topology, and the shared renderer. Real-asset checks are optional:

```bash
python -m pytest tests/python/examples/deviceio_live_view \
  --soma-assets /path/to/SOMA-X/assets
```
