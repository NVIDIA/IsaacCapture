<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sharpa Wave Standalone URDFs

This directory stores the official Sharpa Wave standalone hand URDFs used by
`hand_teleop` with `hand_retargeter:=dexpilot`:

- `left_sharpa_wave.urdf`
- `right_sharpa_wave.urdf`

Docker builds fetch the pinned official URDFs into the image at build time.
Source-tree users can populate this directory from the repo root:

```bash
python3 examples/teleop_ros2/scripts/fetch_sharpa_wave_urdfs.py
```

These robot model assets are not fetched at runtime by the node.
