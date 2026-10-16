<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SOMA Body Publisher

Publishes the motion bundled with SOMA-X through the Isaac Teleop SOMA body
FlatBuffer and `SchemaPusher`. This is a deterministic input generator for
testing SOMA consumers. It is not a vendor plugin or vendor network protocol.

The Python process converts the bundled motion into the pinned SOMA-X v0.3.1
pose contract and publishes through `TensorPushTracker`, which wraps
`SchemaPusher`. It can publish joint rotations for downstream FK or
pre-evaluated joint poses on the `soma_body_demo` tensor collection.

Each payload uses explicitly keyed SOMA joints. The bundled motion provides all
77 joints; a partial producer can omit joints it does not provide.

```text
SOMA-X example_animation.npy
    -> Python SOMA body encoder
    -> TensorPushTracker
    -> SchemaPusher collection soma_body_demo
```

Install the example:

```bash
uv pip install -e ./examples/soma_body_publisher
```

Start a SOMA consumer such as `deviceio_live_view`, connect its client, then run:

```bash
source ~/.cloudxr/run/cloudxr.env
uv run --no-sync python -m isaaccapture_examples.soma_body_publisher \
  --loop
```

Joint rotations are the default. To evaluate the animation before transport,
add `--body-representation joint-poses`. Configure the consumer with the same
representation. The two modes publish `soma_body_joint_rotations` and
`soma_body_joint_poses`, respectively.

Add `--validate-only` to encode every bundled frame and check transport capacity
without starting OpenXR or publishing data.
