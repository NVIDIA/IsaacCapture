#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Start the human program on the workstation: set its environment, then exec
#   python -m isaaccapture_examples.agentic_remote.human "$@"
# for example
#   ./run_human.sh --config configs/human.yaml
# The human program is the program through which a person drives the arm with
# the headset's right controller. The working folder stays as it is, so a
# relative --config is taken from it.
#
# Needs in the environment:
#   AGENTIC_REMOTE_HUMAN_PYTHON      a Python with numpy, PyYAML and pinocchio;
#                                    without it AGENTIC_REMOTE_AGENT_PYTHON is
#                                    taken
#   AGENTIC_REMOTE_APAI_SRC          the src folder of the APAI checkout
#   AGENTIC_REMOTE_ISAAC_CAPTURE_PY  the folder of the Isaac Capture Python
#                                    package built for that Python
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python="${AGENTIC_REMOTE_HUMAN_PYTHON:-${AGENTIC_REMOTE_AGENT_PYTHON:-}}"
if [[ -z "${python}" || ! -x "${python}" ]]; then
  echo "run_human.sh: AGENTIC_REMOTE_HUMAN_PYTHON (or AGENTIC_REMOTE_AGENT_PYTHON) must name a Python with numpy, PyYAML and pinocchio (it is '${python}')" >&2
  exit 2
fi
if [[ -z "${AGENTIC_REMOTE_APAI_SRC:-}" || ! -d "${AGENTIC_REMOTE_APAI_SRC}" ]]; then
  echo "run_human.sh: AGENTIC_REMOTE_APAI_SRC must name the src folder of the APAI checkout (it is '${AGENTIC_REMOTE_APAI_SRC:-}')" >&2
  exit 2
fi
if [[ -z "${AGENTIC_REMOTE_ISAAC_CAPTURE_PY:-}" || ! -d "${AGENTIC_REMOTE_ISAAC_CAPTURE_PY}" ]]; then
  echo "run_human.sh: AGENTIC_REMOTE_ISAAC_CAPTURE_PY must name the folder of the Isaac Capture Python package (it is '${AGENTIC_REMOTE_ISAAC_CAPTURE_PY:-}')" >&2
  exit 2
fi
export AGENTIC_REMOTE_APAI_SRC

# The arm model's loader refuses these two: each selects another APAI
# configuration.
unset APAI_ROBOT APAI_BASE
# The program needs no GPU. Should a package it imports look for one, it finds
# the card of the other programs: index 0 in PCI order.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
# Nothing is written into APAI's checkout.
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${here}/python:${AGENTIC_REMOTE_ISAAC_CAPTURE_PY}:${AGENTIC_REMOTE_APAI_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${python}" -m isaaccapture_examples.agentic_remote.human "$@"
