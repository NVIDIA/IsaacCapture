<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SOMA Hand Publisher

Publishes the left and right hand subsets of the motion bundled with SOMA-X.
This is a deterministic input generator for testing SOMA hand consumers. It is
not a vendor plugin or vendor network protocol.

The Python process expresses both hands through the pinned SOMA-X v0.3.1
`SOMAHandLayer`, then writes paired FlatBuffers to `soma_hand_pusher`. The
native process validates each payload and publishes independent
`soma_hand_left_demo` and `soma_hand_right_demo` collections on one OpenXR
session.

```text
SOMA-X example_animation.npy
    -> left/right SOMAHandLayer controls and evaluated poses
    -> soma_hand_pusher
    -> left/right SchemaPusher collections
```

Install the example and build the native pusher:

```bash
uv pip install -e ./examples/soma_hand_publisher
cmake --build ./build --target soma_hand_pusher
```

Start the DeviceIO viewer with SOMA hands, connect its client, then run the
publisher in another terminal:

```bash
uv run --no-sync python -m isaaccapture_examples.deviceio_live_view \
  --accept-eula \
  --hand-schema soma \
  --soma-data-root /path/to/SOMA-X-v0.3.1-assets

source ~/.cloudxr/run/cloudxr.env
uv run --no-sync python -m isaaccapture_examples.soma_hand_publisher \
  --data-root /path/to/SOMA-X-v0.3.1-assets \
  --pusher ./build/examples/soma_hand_publisher/cpp/soma_hand_pusher/soma_hand_pusher \
  --loop
```

Joint rotations are the default. To publish evaluated joint poses, add
`--hand-representation joint-poses` to the publisher and
`--soma-hand-representation joint-poses` to the viewer. The two modes publish
`soma_hand_joint_rotations_v0` and `soma_hand_joint_poses_v0`, respectively.

The assets must include `SOMAHand.npz`, `example_animation.npy`, and the neutral
SOMA identity assets from the v0.3.1 asset release. Add `--validate-only` to
process every paired frame through the native FlatBuffer verifier without
starting OpenXR.
