# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""python -m isaaccapture_examples.agentic_remote.agent --config PATH.

The agent program. The GPU is chosen here, before anything that could load
CUDA is imported: index 0 in PCI order.
"""

from __future__ import annotations

import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Nothing is written beside the modules of APAI or of the experiment.
sys.dont_write_bytecode = True

from .launcher import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
