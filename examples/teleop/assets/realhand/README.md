<!--
SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# P7 and RealHand Robot Assets

The P7 + L6, O6, and L20 URDF assemblies and their meshes are downloaded to
`~/.cache/isaaccapture/realhand` (or `$XDG_CACHE_HOME/isaaccapture/realhand`). The binary
assets are published separately in the public
[RealHand Teleop repository](https://huggingface.co/realhandinc/realhand-teleop) and are not stored
in Isaac Capture.

From the Isaac Capture repository root, install the examples and download all three assemblies:

```bash
pip install -e ./examples/teleop
python3 -m isaaccapture_examples.teleop.fetch_realhand_assets
```

Download one assembly or choose another destination with:

```bash
python3 -m isaaccapture_examples.teleop.fetch_realhand_assets \
  --hand-model l20 \
  --output-dir ~/.cache/isaaccapture/realhand
```

The fetcher pins the immutable commit behind RealHand Teleop v0.2.0, validates Hugging Face LFS
objects, Git blobs, and the main URDF, and checks that every referenced mesh exists. It downloads
robot assets and license notices only; personal glove calibrations, STEP files, controller mounts,
scenes, and demonstrations are excluded.
