<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Run the tactile example

1. From the IsaacCapture repository root, install the tactile example.

   ```bash
   uv pip install -e ./examples/tactile
   ```

2. In the first terminal, start the printer and the local CloudXR runtime.

   ```bash
   python -m isaaccapture_examples.tactile.printer --accept-eula
   ```

3. In the second terminal, start the feeder.

   ```bash
   python -m isaaccapture_examples.tactile.feeder
   ```

4. Confirm that the first terminal prints `TactileMetadata`, followed by the first and last taxel values once per second.

5. Press `Ctrl+C` in both terminals.
