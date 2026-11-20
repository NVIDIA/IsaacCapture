#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "${here}/build/zed_sender" ]]; then
    echo "Run ${here}/build.sh first." >&2
    exit 1
fi
exec "${here}/build/zed_sender" --config "${here}/configs/zed.yaml" "$@"
