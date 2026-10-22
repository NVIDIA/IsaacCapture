<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Tactile

A mock tactile device and a consumer for exercising the tactile tracker without tactile hardware.

- `feeder` publishes a 26-joint hand with 200 taxels by default. Each frame carries one
  `TactileDataArray`; it publishes `TactileMetadata` when it starts and every second after that.
  It publishes through `TensorPushTracker` on a `DeviceIOSession`, as a tactile plugin would
  through `SchemaPusher`.
- `printer` reads one `TactileTracker` directly. It discovers the joint and taxel counts from
  `TactileMetadata`, prints the first and last taxel forces, and records the session to MCAP.

```text
feeder -> "tactile"          (TactileDataArray, every frame)
       -> "tactile/metadata" (TactileMetadata, at start and every --metadata-interval s)
            -> TactileTracker -> printer, and tactile/* MCAP channels
```

Install the example:

```bash
uv pip install -e ./examples/tactile
```

Start the printer first; it starts the CloudXR runtime. Live initialization requires an
OpenXR HMD system; if none is available, it waits for a headset client to connect:

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
counts from metadata. The `TactileMetadata` must fit `TRACKER_METADATA_MAX_SIZE`, which holds 26
joints and 1000 taxels. The two processes need only agree on `--collection-id`.

Metadata contains an ordered `HandSkeletonRef` and an `LBS` binding. Segment indices address
that joint list, not OpenXR ordinals. Python pose arrays have seven columns: position XYZ in
meters followed by unit quaternion XYZW; per-frame force components are in taxel-local axes
and measured in newtons.
