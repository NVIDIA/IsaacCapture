<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SOMA Body Publisher

Publishes the motion bundled with SOMA-X through the Isaac Teleop SOMA body
FlatBuffer and `SchemaPusher`. This is a deterministic input generator for
testing SOMA consumers. It is not a vendor plugin or vendor network protocol.

The Python process converts the bundled motion into the pinned SOMA-X v0.3.1
pose contract and writes framed FlatBuffers to the native `soma_body_pusher`.
It can publish joint rotations for downstream FK or pre-evaluated joint poses.
The native process validates the selected payload and publishes the `soma_demo`
tensor collection.

```text
SOMA-X example_animation.npy
    -> Python SOMA body encoder
    -> soma_body_pusher
    -> SchemaPusher collection soma_demo
```

Install the example and build the native pusher:

```bash
uv pip install -e ./examples/soma_body_publisher
cmake --build ./build --target soma_body_pusher
```

Start a SOMA consumer such as `deviceio_live_view`, connect its client, then run:

```bash
source ~/.cloudxr/run/cloudxr.env
uv run --no-sync python -m isaaccapture_examples.soma_body_publisher \
  --data-root /path/to/SOMA-X/assets \
  --pusher ./build/examples/soma_body_publisher/cpp/soma_body_pusher/soma_body_pusher \
  --loop
```

Joint rotations are the default. To evaluate the animation before transport,
add `--body-representation joint-poses`. Configure the consumer with the same
representation. The two modes publish `soma_body_joint_rotations_v0` and
`soma_body_joint_poses_v0`, respectively.

The assets directory must contain `example_animation.npy` and the neutral SOMA
identity assets. Add `--validate-only` to process every bundled frame through
the FlatBuffer verifier without starting OpenXR or publishing data.
