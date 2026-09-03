<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sharpa Avatar sample

One-command demo of the Sharpa Avatar glove plugin through public Isaac Teleop
APIs (`TeleopSession`, `PluginConfig`, `HandsSource`, `HapticSink`). Starts
CloudXR, the plugin, a viser browser hand view, and pinch vibration.

Build the plugin first (Avatar SDK stays off-repo):

```bash
./src/plugins/sharpa_avatar/install.sh
```

Then from the Isaac Teleop root:

```bash
uv pip install viser numpy
.venv/bin/python examples/sharpa_avatar/python/sharpa_avatar_sample.py
```

Open the printed URL (default http://127.0.0.1:8080). `--no-viz` is terminal
and haptic only.

Install, data paths, and troubleshooting:
[`src/plugins/sharpa_avatar/README.md`](../../src/plugins/sharpa_avatar/README.md).
