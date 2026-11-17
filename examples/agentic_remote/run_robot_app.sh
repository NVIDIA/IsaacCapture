#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Start the robot app on the Thor, at the Thor terminal: set its environment, then
# exec
#   python -m isaaccapture_examples.agentic_remote.robot_app "$@"
# The working folder stays as it is, so a relative --config is taken from it.
#
# --env FILE, as the first argument only, reads the variables below from FILE,
# a shell file of KEY=value lines; the robot app does not get it. Without it
# they are taken from the environment:
#   AGENTIC_REMOTE_THOR_PYTHON       the Python of APAI's environment on the Thor
#   AGENTIC_REMOTE_APAI_SRC          the src folder of the APAI checkout
#   AGENTIC_REMOTE_ISAAC_CAPTURE_PY  the folder of the Isaac Capture Python package
#                                    built on the Thor for that Python
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "--env" ]]; then
  if [[ $# -lt 2 || ! -f "$2" ]]; then
    echo "run_robot_app.sh: --env needs a file of KEY=value lines (it is '${2:-}')" >&2
    exit 2
  fi
  set -a
  # shellcheck disable=SC1090
  source "$2"
  set +a
  shift 2
fi

python="${AGENTIC_REMOTE_THOR_PYTHON:-}"
if [[ -z "${python}" || ! -x "${python}" ]]; then
  echo "run_robot_app.sh: AGENTIC_REMOTE_THOR_PYTHON must name the Python of APAI's environment on the Thor (it is '${python}')" >&2
  exit 2
fi
if [[ -z "${AGENTIC_REMOTE_APAI_SRC:-}" || ! -d "${AGENTIC_REMOTE_APAI_SRC}" ]]; then
  echo "run_robot_app.sh: AGENTIC_REMOTE_APAI_SRC must name the src folder of the APAI checkout (it is '${AGENTIC_REMOTE_APAI_SRC:-}')" >&2
  exit 2
fi
if [[ -z "${AGENTIC_REMOTE_ISAAC_CAPTURE_PY:-}" || ! -d "${AGENTIC_REMOTE_ISAAC_CAPTURE_PY}" ]]; then
  echo "run_robot_app.sh: AGENTIC_REMOTE_ISAAC_CAPTURE_PY must name the folder of the Isaac Capture Python package built on the Thor (it is '${AGENTIC_REMOTE_ISAAC_CAPTURE_PY:-}')" >&2
  exit 2
fi
export AGENTIC_REMOTE_APAI_SRC

# The robot app refuses these two: each selects another APAI configuration.
unset APAI_ROBOT APAI_BASE
# The robot app reads the hold markers where the experiment's launcher keeps
# them: APAI's default folder, ~/.apai/arm_holds.
unset APAI_ARM_HOLD_DIR
# Nothing is written into APAI's checkout.
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${here}/python:${AGENTIC_REMOTE_ISAAC_CAPTURE_PY}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${python}" -m isaaccapture_examples.agentic_remote.robot_app "$@"
