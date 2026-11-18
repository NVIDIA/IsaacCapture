# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys

CONSUMER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONSUMER_ROOT))
sys.path.insert(0, str(CONSUMER_ROOT.parent))
