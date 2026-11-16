#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Start the operator console on the workstation, in a terminal of its own:
#   ./run_console.sh [--socket PATH] [--name NAME] [--log-level L]
# It execs
#   python -m isaaccapture_examples.agentic_remote.console "$@"
#
# The console needs Python 3.10 or newer and nothing else: no APAI, no torch
# and no Isaac Capture, so it starts when the agent program cannot.
#
# Optional in the environment:
#   AGENTIC_REMOTE_CONSOLE_PYTHON  the Python to run it with (default: python3)
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python="${AGENTIC_REMOTE_CONSOLE_PYTHON:-python3}"
if ! command -v "${python}" >/dev/null 2>&1; then
  echo "run_console.sh: no Python at '${python}'; set AGENTIC_REMOTE_CONSOLE_PYTHON" >&2
  exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${here}/python${PYTHONPATH:+:${PYTHONPATH}}"

exec "${python}" -m isaaccapture_examples.agentic_remote.console "$@"
