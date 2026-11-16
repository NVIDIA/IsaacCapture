#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build the workstation's APAI Python environment: the Python the agent program
# runs under (run_agent.sh), with torch for CUDA, Warp and the packages of
# requirements-workstation.txt. Run it once. Steps 2 and 3 download from the
# internet (several GB).
#
#   AGENTIC_REMOTE_ENV_DIR=<folder> ./setup_workstation_env.sh [options]
#
# AGENTIC_REMOTE_ENV_DIR names the environment folder: an absolute path outside
# every git repository, and not under /tmp, which a reboot clears. The script
# makes in it:
#   venv/        a Python 3.12 environment
#   caches/      triton, torchinductor, ultralytics (run_agent.sh points there)
#   freeze.txt   the versions that were installed
#   codex/       the model client, only with --with-model-client
#
# Options:
#   --dry-run            print the commands of every step and run none
#   --with-model-client  also install the model client (the Codex command-line
#                        program) with npm. This needs the owner's yes: without
#                        this option the script neither downloads nor installs
#                        it. The script never reads or writes the login of that
#                        program.
#   -h, --help           print this text
#
# Exit codes: 0 done; 1 a step failed (the step is named); 2 refused, nothing
# was run. The script refuses when the path is not absolute or holds a "." or
# ".." component, when the folder lies inside a git repository, when
# <folder>/venv exists, and when uv (or, with --with-model-client, npm) is
# missing.
set -euo pipefail

me="setup_workstation_env.sh"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
requirements="${here}/requirements-workstation.txt"
torch_index="https://download.pytorch.org/whl/cu130"
model_client="@openai/codex@0.167.0"

refuse() {
  echo "${me}: $1" >&2
  exit 2
}

dry_run=0
with_model_client=0
for argument in "$@"; do
  case "${argument}" in
    --dry-run) dry_run=1 ;;
    --with-model-client) with_model_client=1 ;;
    -h | --help)
      while IFS= read -r line; do
        [[ "${line}" == "set -euo pipefail" ]] && break
        [[ "${line}" == "#!"* || "${line}" == "# SPDX-"* ]] && continue
        line="${line#\#}"
        echo "${line# }"
      done < "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *) refuse "unknown argument '${argument}' (--dry-run, --with-model-client, --help)" ;;
  esac
done

env_dir="${AGENTIC_REMOTE_ENV_DIR:-}"
if [[ -z "${env_dir}" ]]; then
  refuse "AGENTIC_REMOTE_ENV_DIR must name the environment folder (it is not set)"
fi
if [[ "${env_dir}" != /* ]]; then
  refuse "AGENTIC_REMOTE_ENV_DIR must be an absolute path (it is '${env_dir}')"
fi
# The check below walks up the path as it is written; mkdir would follow a
# ".." out of a folder that does not exist yet.
if [[ "${env_dir}/" == */./* || "${env_dir}/" == */../* ]]; then
  refuse "AGENTIC_REMOTE_ENV_DIR must not hold a '.' or '..' component (it is '${env_dir}')"
fi
while [[ "${env_dir}" == */ && "${env_dir}" != "/" ]]; do
  env_dir="${env_dir%/}"
done

# The folder, or its nearest ancestor that exists, and every folder above it.
above="${env_dir}"
while [[ ! -d "${above}" ]]; do
  above="${above%/*}"
  above="${above:-/}"
done
above="$(cd "${above}" && pwd -P)"
while true; do
  if [[ -e "${above}/.git" ]]; then
    refuse "${env_dir} lies inside the git repository ${above}; name a folder outside every repository"
  fi
  [[ "${above}" == "/" ]] && break
  above="${above%/*}"
  above="${above:-/}"
done

if [[ -e "${env_dir}/venv" ]]; then
  refuse "${env_dir}/venv exists; this script only builds a new environment"
fi
if [[ ! -f "${requirements}" ]]; then
  refuse "${requirements} is missing"
fi
if ! command -v uv > /dev/null; then
  refuse "uv is not on PATH; it makes the environment and installs the packages"
fi
if ((with_model_client)) && ! command -v npm > /dev/null; then
  refuse "npm is not on PATH; --with-model-client installs the model client with it"
fi

python="${env_dir}/venv/bin/python"
excludes="${env_dir}/exclude-gui-opencv.txt"

# run_step <number> <what> <file for the output, or ""> <command...>
# Prints the step and its command, then runs it; exits 1 when it fails.
run_step() {
  local number="$1" what="$2" output="$3" code=0
  shift 3
  echo "step ${number}: ${what}"
  printf '   '
  printf ' %q' "$@"
  if [[ -n "${output}" ]]; then
    printf ' > %q' "${output}"
  fi
  printf '\n'
  if ((dry_run)); then
    return 0
  fi
  if [[ -n "${output}" ]]; then
    "$@" > "${output}" || code=$?
  else
    "$@" || code=$?
  fi
  if ((code != 0)); then
    echo "${me}: step ${number} (${what}) failed with exit code ${code}; the later steps were not run" >&2
    exit 1
  fi
}

run_step 0 "the folders" "" \
  mkdir -p "${env_dir}/caches/triton" "${env_dir}/caches/torchinductor" "${env_dir}/caches/ultralytics"
run_step 1 "the Python 3.12 environment" "" \
  uv venv "${env_dir}/venv" --python 3.12
run_step 2 "torch for CUDA (downloads several GB)" "" \
  uv pip install --python "${python}" --index-url "${torch_index}" torch==2.10.0 torchvision==0.25.0
# The headless OpenCV of the requirements already provides cv2; ultralytics
# would add the other build over it.
run_step 3a "the list of packages left out" "${excludes}" \
  printf 'opencv-python\n'
run_step 3b "the packages of requirements-workstation.txt (downloads)" "" \
  uv pip install --python "${python}" --excludes "${excludes}" -r "${requirements}"
if ((with_model_client)); then
  run_step 4 "the model client (downloads; needs the owner's yes)" "" \
    npm install --prefix "${env_dir}/codex" --no-audit --no-fund "${model_client}"
else
  echo "step 4: the model client: left out (it needs the owner's yes; --with-model-client)"
fi
run_step 5 "the list of what was installed" "${env_dir}/freeze.txt" \
  uv pip freeze --python "${python}"

if ((dry_run)); then
  echo "dry run: no command was run"
else
  echo "done: ${env_dir}; set AGENTIC_REMOTE_AGENT_PYTHON=${python}"
fi
