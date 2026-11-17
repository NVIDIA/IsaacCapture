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
#                                  path without a "." or ".." component;
#                                  optional. Without it the agent program keeps
#                                  the three caches below in its state folder.
#
# Exit code 2, and nothing is started, when one of these cannot be used, and
# when TRITON_CACHE_DIR, TORCHINDUCTOR_CACHE_DIR or YOLO_CONFIG_DIR is not an
# absolute path, holds a "." or ".." component, or names a place inside the
# experiment folder (symbolic links are followed).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

refuse() {
  echo "run_agent.sh: $1" >&2
  exit 2
}

# resolved <path>: prints the path with the symbolic links of its part that
# exists followed, and an "x" after it, which keeps a newline at the end of a
# name. What does not exist yet is kept as it is written.
resolved() {
  local path="$1" tail=""
  while [[ "${path}" == *//* ]]; do
    path="${path//\/\///}"
  done
  while [[ "${path}" == */ && "${path}" != "/" ]]; do
    path="${path%/}"
  done
  while [[ ! -d "${path}" ]]; do
    tail="/${path##*/}${tail}"
    if [[ "${path}" != */* ]]; then
      path="."
      break
    fi
    path="${path%/*}"
    path="${path:-/}"
  done
  path="$(cd "${path}" && pwd -P && printf x)"
  path="${path%$'\n'x}"
  if [[ "${path}" == "/" ]]; then
    path=""
  fi
  printf '%sx' "${path}${tail}"
}

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
# the variables name another place. One that is set and not empty is kept.
if [[ -n "${AGENTIC_REMOTE_ENV_DIR:-}" ]]; then
  if [[ "${AGENTIC_REMOTE_ENV_DIR}" != /* ]]; then
    refuse "AGENTIC_REMOTE_ENV_DIR must be an absolute path (it is '${AGENTIC_REMOTE_ENV_DIR}')"
  fi
  if [[ "${AGENTIC_REMOTE_ENV_DIR}/" == */./* || "${AGENTIC_REMOTE_ENV_DIR}/" == */../* ]]; then
    refuse "AGENTIC_REMOTE_ENV_DIR must not hold a '.' or '..' component (it is '${AGENTIC_REMOTE_ENV_DIR}')"
  fi
  export AGENTIC_REMOTE_ENV_DIR
  caches="${AGENTIC_REMOTE_ENV_DIR}/caches"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${caches}/triton}"
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${caches}/torchinductor}"
  export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-${caches}/ultralytics}"
fi
# None of the three may name a place inside the experiment folder, however it
# is spelled: what the program will get is compared, with the links followed.
experiment="${AGENTIC_REMOTE_EXPERIMENT_DIR:-}"
inside=""
if [[ -n "${experiment}" ]]; then
  inside="$(resolved "${experiment}")"
  inside="${inside%x}"
fi
for name in TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR YOLO_CONFIG_DIR; do
  value="${!name:-}"
  if [[ -z "${value}" ]]; then
    continue
  fi
  if [[ "${value}" != /* ]]; then
    refuse "${name} must be an absolute path (it is '${value}')"
  fi
  if [[ "${value}/" == */./* || "${value}/" == */../* ]]; then
    refuse "${name} must not hold a '.' or '..' component (it is '${value}')"
  fi
  if [[ -z "${experiment}" ]]; then
    continue
  fi
  place="$(resolved "${value}")"
  place="${place%x}"
  if [[ "${place}" == "${inside}" || "${place}" == "${inside}/"* ]]; then
    refuse "${name} names a place inside the experiment folder (${value}); unset it, or name a folder outside ${experiment}"
  fi
done

# One card for this program and its children: index 0 in PCI order. The camera
# receiver and the camera frame bridge must sit on the same card.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
# APAI_REQUIRE_CUDA (APAI's strict CUDA mode) is not set here, as the
# experiment's own launcher (thor_runtime.sh) never sets it: with 1 APAI keeps
# an object's points as CUDA tensors and its grasp code fails on them. Depth,
# the segmenter and the obstacle grid run on the GPU either way.
# Nothing is written into APAI's checkout or the experiment folder.
export PYTHONDONTWRITEBYTECODE=1
# A git status the experiment runs on its tools must not rewrite their index.
export GIT_OPTIONAL_LOCKS=0
# torch brings its own CUDA libraries.
unset LD_LIBRARY_PATH
export PYTHONPATH="${here}/python:${AGENTIC_REMOTE_APAI_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${python}" -m isaaccapture_examples.agentic_remote.agent "$@"
