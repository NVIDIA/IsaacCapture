<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Tactile feeder and printer

1. In the Python environment that has Isaac Capture installed, install the example from the
   repository root:

   ```bash
   uv pip install -e ./examples/tactile
   ```

2. Start the printer. It starts the CloudXR runtime if none is running:

   ```bash
   python -m isaaccapture_examples.tactile.printer --accept-eula
   ```

3. In a second terminal with the same environment, start the feeder:

   ```bash
   python -m isaaccapture_examples.tactile.feeder
   ```

4. In the printer terminal, check that a `TactileMetadata` block is printed once, followed by a
   `TactileData` block with the values of taxel 0 and taxel 15 every second.

5. Press `Ctrl+C` in the feeder terminal, then in the printer terminal.
