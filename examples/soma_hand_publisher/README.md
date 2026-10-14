<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SOMA Hand Publisher

Publishes the left and right hand subsets of the motion bundled with SOMA-X.
This is a deterministic input generator for testing SOMA hand consumers. It is
not a vendor plugin or vendor network protocol.

The Python process expresses both hands through the pinned SOMA-X v0.3.1
`SOMAHandLayer`, then publishes through `TensorPushTracker`, which wraps
`SchemaPusher`. Independent `soma_hand_left_demo` and `soma_hand_right_demo`
collections share one OpenXR session.

Each payload uses explicitly keyed SOMA joints. The bundled motion provides all
25 joints per hand; a partial producer can omit joints it does not provide.

```text
SOMA-X example_animation.npy
    -> left/right SOMAHandLayer controls and evaluated poses
    -> left/right TensorPushTracker
    -> left/right SchemaPusher collections
```

Install the example:

```bash
uv pip install -e ./examples/soma_hand_publisher
```

Start the DeviceIO viewer with SOMA hands, connect its client, then run the
publisher in another terminal:

```bash
uv run --no-sync python -m isaaccapture_examples.deviceio_live_view \
  --accept-eula \
  --hand-schema soma

source ~/.cloudxr/run/cloudxr.env
uv run --no-sync python -m isaaccapture_examples.soma_hand_publisher \
  --loop
```

Joint rotations are the default. To publish evaluated joint poses, add
`--hand-representation joint-poses` to the publisher and
`--soma-hand-representation joint-poses` to the viewer. The two modes publish
`soma_hand_joint_rotations` and `soma_hand_joint_poses`, respectively.

Add `--validate-only` to encode every paired frame and check transport capacity
without starting OpenXR.
