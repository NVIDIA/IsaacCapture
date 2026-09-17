<!--
SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# P7 and RealHand Robot Assets

This directory receives the P7 + L6, O6, and L20 URDF assemblies and their meshes. The binary
assets are published separately in the public
[RealHand Teleop repository](https://huggingface.co/realhandinc/realhand-teleop) and are not stored
in Isaac Teleop.

From the Isaac Teleop repository root, download all three assemblies with:

```bash
python3 examples/teleop/python/scripts/fetch_realhand_assets.py
```

Download one assembly or choose another destination with:

```bash
python3 examples/teleop/python/scripts/fetch_realhand_assets.py \
  --hand-model l20 \
  --output-dir ~/.cache/isaaccapture/realhand
```

The fetcher pins the immutable commit behind RealHand Teleop v0.2.0, validates Hugging Face LFS
objects and the main URDF, and checks that every mesh referenced by the URDF exists. It downloads
robot assets and license notices only; personal glove calibrations, STEP files, controller mounts,
scenes, and demonstrations are excluded.
