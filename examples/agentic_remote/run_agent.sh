#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Start the agent program on the workstation: set its environment, then exec
#   python -m isaaccapture_examples.agentic_remote.agent "$@"
#
# Needs in the environment:
#   AGENTIC_REMOTE_AGENT_PYTHON    a Python with APAI's packages, PyYAML and, for the
#                                  full stack, torch with CUDA and Warp
#   AGENTIC_REMOTE_APAI_SRC        the src folder of the APAI checkout
#   AGENTIC_REMOTE_EXPERIMENT_DIR  the experiment folder (experiment.stack: full)
#   AGENTIC_REMOTE_CAMERA_PR       the camera_streamer example of the camera pull
#                                  request (--camera link)
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python="${AGENTIC_REMOTE_AGENT_PYTHON:-}"
if [[ -z "${python}" || ! -x "${python}" ]]; then
  echo "run_agent.sh: AGENTIC_REMOTE_AGENT_PYTHON must name a Python with APAI's packages, torch with CUDA, Warp and PyYAML (it is '${python}')" >&2
  exit 2
fi
if [[ -z "${AGENTIC_REMOTE_APAI_SRC:-}" || ! -d "${AGENTIC_REMOTE_APAI_SRC}" ]]; then
  echo "run_agent.sh: AGENTIC_REMOTE_APAI_SRC must name the src folder of the APAI checkout (it is '${AGENTIC_REMOTE_APAI_SRC:-}')" >&2
  exit 2
fi
export AGENTIC_REMOTE_APAI_SRC
if [[ -n "${AGENTIC_REMOTE_EXPERIMENT_DIR:-}" ]]; then
  export AGENTIC_REMOTE_EXPERIMENT_DIR
fi
if [[ -n "${AGENTIC_REMOTE_CAMERA_PR:-}" ]]; then
  export AGENTIC_REMOTE_CAMERA_PR
fi

# One card for this program and its children: index 0 in PCI order. The camera
# receiver and the camera frame bridge must sit on the same card.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
# A missing GPU is an error, not a silent fall back to the CPU. A dry run on the
# mock scene may set APAI_REQUIRE_CUDA=0 before it calls this script.
export APAI_REQUIRE_CUDA="${APAI_REQUIRE_CUDA:-1}"
# Nothing is written into APAI's checkout or the experiment folder.
export PYTHONDONTWRITEBYTECODE=1
# torch brings its own CUDA libraries.
unset LD_LIBRARY_PATH
export PYTHONPATH="${here}/python:${AGENTIC_REMOTE_APAI_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${python}" -m isaaccapture_examples.agentic_remote.agent "$@"
