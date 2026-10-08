#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

EXAMPLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${EXAMPLE_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Missing example virtual environment: ${EXAMPLE_ROOT}/.venv" >&2
    echo "Create it with: uv venv --python 3.11 .venv" >&2
    exit 1
fi

export PYTHONPATH="${EXAMPLE_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON}" -m vehicle_teleop.isaac_remote_steering_worker \
    --rate-hz 50 \
    "$@"
