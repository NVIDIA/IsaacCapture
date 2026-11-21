#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${CAMERA_RECEIVER_PYTHON:-}" ]]; then
  PYTHON="$CAMERA_RECEIVER_PYTHON"
elif [[ -x "${HERE}/.venv/bin/python" ]]; then
  PYTHON="${HERE}/.venv/bin/python"
else
  echo "run.sh: no camera_receiver venv found. Run ./setup.sh first." >&2
  echo "run.sh: or set CAMERA_RECEIVER_PYTHON=/path/to/python" >&2
  exit 1
fi

if [[ $# -gt 0 && "$1" != -* ]]; then
  CONFIG="$1"
  shift
else
  CONFIG="${HERE}/configs/receiver.yaml"
fi

exec "$PYTHON" "${HERE}/camera_receiver.py" --config "$CONFIG" "$@"
