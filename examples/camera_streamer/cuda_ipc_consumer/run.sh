#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${CUDA_IPC_CONSUMER_PYTHON:-}" ]]; then
  PYTHON="$CUDA_IPC_CONSUMER_PYTHON"
elif [[ -x "${HERE}/.venv/bin/python" ]]; then
  PYTHON="${HERE}/.venv/bin/python"
else
  echo "run.sh: no cuda_ipc_consumer venv found. Run ./setup.sh first." >&2
  echo "run.sh: or set CUDA_IPC_CONSUMER_PYTHON=/path/to/python" >&2
  exit 1
fi

exec "$PYTHON" "${HERE}/consumer.py" "$@"
