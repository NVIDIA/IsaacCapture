# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys

STREAMER_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STREAMER_ROOT))
