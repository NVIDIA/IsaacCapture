# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The CUDA_PATH setup records for a toolkit CuPy cannot locate on its own.

CuPy finds the headers of a toolkit without a version file, such as the distro
nvidia-cuda-toolkit at /usr, only through CUDA_PATH. scripts/_install_deps.sh
writes that root into the venv; every entry point applies it before CuPy runs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

RECORD = "camera_viz_cuda_path"


def apply_recorded_cuda_path() -> None:
    # An explicit CUDA_PATH or CUDA_HOME wins over the record.
    if os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME"):
        return
    record = Path(sys.prefix) / RECORD
    if record.is_file():
        os.environ["CUDA_PATH"] = record.read_text().strip()
