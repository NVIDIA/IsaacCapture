# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""python -m isaaccapture_examples.agentic_remote.console: the operator console."""

from __future__ import annotations

import sys

from .console import main

if __name__ == "__main__":
    sys.exit(main())
