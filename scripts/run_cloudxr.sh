#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR" || exit 1
source "./setup_cloudxr_env.sh"

echo "Starting WSS proxy..."
if ! python -c "import isaaccapture.cloudxr" >/dev/null 2>&1; then
    echo "Error: isaaccapture[cloudxr] is not installed. Install it before running this script (e.g. pip install 'isaaccapture[cloudxr]')."
    echo "Follow the Quick Start guide to install the package via pypi: https://nvidia.github.io/IsaacCapture/main/getting_started/quick_start.html"
    echo "Or build and install the package from source: https://nvidia.github.io/IsaacCapture/main/getting_started/build_from_source.html"
    exit 1
fi

exec python -m isaaccapture.cloudxr.service run
