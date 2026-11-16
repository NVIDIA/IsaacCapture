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
#   AGENTIC_REMOTE_ENV_DIR         the workstation's APAI Python environment
#                                  (setup_workstation_env.sh), as an absolute
#                                  path; optional
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
# The experiment's code keeps these caches inside the experiment folder unless
# the variables name another place. One that is set and not empty is kept,
# unless it names a place inside the experiment folder.
if [[ -n "${AGENTIC_REMOTE_ENV_DIR:-}" ]]; then
  if [[ "${AGENTIC_REMOTE_ENV_DIR}" != /* ]]; then
    echo "run_agent.sh: AGENTIC_REMOTE_ENV_DIR must be an absolute path (it is '${AGENTIC_REMOTE_ENV_DIR}')" >&2
    exit 2
  fi
  experiment="${AGENTIC_REMOTE_EXPERIMENT_DIR:-}"
  while [[ "${experiment}" == */ && "${experiment}" != "/" ]]; do
    experiment="${experiment%/}"
  done
  if [[ -n "${experiment}" ]]; then
    for name in TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR YOLO_CONFIG_DIR; do
      value="${!name:-}"
      if [[ "${value}" == "${experiment}" || "${value}" == "${experiment}/"* ]]; then
        echo "run_agent.sh: ${name} names a place inside the experiment folder (${value}); unset it, or name a folder outside ${experiment}" >&2
        exit 2
      fi
    done
  fi
  export AGENTIC_REMOTE_ENV_DIR
  caches="${AGENTIC_REMOTE_ENV_DIR}/caches"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${caches}/triton}"
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${caches}/torchinductor}"
  export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-${caches}/ultralytics}"
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
# A git status the experiment runs on its tools must not rewrite their index.
export GIT_OPTIONAL_LOCKS=0
# torch brings its own CUDA libraries.
unset LD_LIBRARY_PATH
export PYTHONPATH="${here}/python:${AGENTIC_REMOTE_APAI_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${python}" -m isaaccapture_examples.agentic_remote.agent "$@"
