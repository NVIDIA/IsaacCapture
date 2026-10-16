<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Tactile

A mock tactile device and a consumer for exercising the tactile tracker without hardware.

- `feeder` publishes a 26-joint hand with 200 taxels by default. Each frame carries one
  `TactileData` and the next chunk of the `TactileShape` metadata. It publishes through
  `TensorPushTracker` on a `DeviceIOSession`, as a tactile plugin would through `SchemaPusher`
  and `SchemaMetadataPusher`.
- `printer` reads one `TactileTracker` directly. It discovers the joint and taxel counts from
  `TactileShape`, prints the first and last taxel forces, and records the session to MCAP.

```text
feeder -> "tactile"          (TactileData, every frame)
       -> "tactile/metadata" (TactileShape chunks, in a loop)
            -> TactileTracker -> printer, and tactile/* MCAP channels
```

Install the example:

```bash
uv pip install -e ./examples/tactile
```

Start the printer first; it starts the CloudXR runtime, and no headset needs to connect:

```bash
python -m isaaccapture_examples.tactile.printer --accept-eula --duration 0
```

Then, in another terminal, start the feeder. It attaches to the same runtime:

```bash
python -m isaaccapture_examples.tactile.feeder
```

Stop the printer with `Ctrl+C`; it writes `./recordings/tactile_<timestamp>.mcap`. Print a
recording with no runtime at all:

```bash
python -m isaaccapture_examples.tactile.printer --replay recordings/tactile_<timestamp>.mcap
```

Use feeder options `--joints` and `--taxels` to change the shape; the printer discovers both
counts from metadata. The two processes need only agree on `--collection-id`.
