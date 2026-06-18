#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

EXAMPLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${EXAMPLE_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

exec uv run --project "${EXAMPLE_ROOT}/python" \
    python -m vehicle_teleop.replay_command_mcap \
    "$@"
