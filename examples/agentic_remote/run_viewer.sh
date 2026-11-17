#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Start the experiment's viewer on the workstation: set its environment, then exec
#   python <experiment folder>/thor_viewer.py --host 127.0.0.1 --runs RUNS \
#     --token-file FILE "$@"
# The experiment is the folder of the rig's earlier, single machine program. Its
# viewer is a web page for a browser on the runs the agent program records with
# --record-debug: the obstacle map, the measured joints, refused poses, the
# collision volumes, the clearance points and the timing graph, while a run goes
# on and afterwards. It reads files and connects to no robot and no camera.
#   ./run_viewer.sh                    # the runs in .state/runs, on 127.0.0.1
#   ./run_viewer.sh --runs FOLDER --port 8767 --no-auth
#
# Arguments, all optional; they and any other go to the viewer as they are:
#   --runs FOLDER      the folder that holds the run folders. Default: .state/runs
#                      beside this script, where the agent program puts its runs
#                      with the shipped files; it is made when it is not there
#   --host ADDRESS     the address to listen on. Default: 127.0.0.1, this machine
#   --port NUMBER      default: the viewer's own, 8766
#   --token-file FILE  where the viewer keeps its access token. Default:
#                      .state/viewer.token beside this script
#   --no-auth          serve without a token
#
# Needs in the environment:
#   AGENTIC_REMOTE_AGENT_PYTHON    a Python with APAI's packages, numpy, OpenCV
#                                  and PyYAML
#   AGENTIC_REMOTE_EXPERIMENT_DIR  the experiment folder, which holds thor_viewer.py
#   AGENTIC_REMOTE_APAI_SRC        the src folder of the APAI checkout
#
# Exit code 2 and nothing started: a variable that cannot be used, --runs or
# --token-file without a value, a --runs that names no folder, or a state folder
# that cannot be made.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
state="${here}/.state"

refuse() {
  echo "run_viewer.sh: $1" >&2
  exit 2
}

runs=""
runs_given=0
token=""
token_given=0
auth=1
previous=""
for argument in "$@"; do
  case "${previous}" in
    --runs) runs="${argument}"; runs_given=1 ;;
    --token-file) token="${argument}"; token_given=1 ;;
  esac
  case "${argument}" in
    --runs=*) runs="${argument#--runs=}"; runs_given=1 ;;
    --token-file=*) token="${argument#--token-file=}"; token_given=1 ;;
    --no-auth) auth=0 ;;
    --help | -h) sed -n '5,34p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
  esac
  previous="${argument}"
done
if [[ "${previous}" == "--runs" || "${previous}" == "--token-file" ]]; then
  refuse "${previous} needs a value"
fi

python="${AGENTIC_REMOTE_AGENT_PYTHON:-}"
if [[ -z "${python}" || ! -x "${python}" ]]; then
  refuse "AGENTIC_REMOTE_AGENT_PYTHON must name a Python with APAI's packages, numpy, OpenCV and PyYAML (it is '${python}')"
fi
experiment="${AGENTIC_REMOTE_EXPERIMENT_DIR:-}"
if [[ -z "${experiment}" || ! -d "${experiment}" ]]; then
  refuse "AGENTIC_REMOTE_EXPERIMENT_DIR must name the experiment folder (it is '${experiment}')"
fi
viewer="${experiment}/thor_viewer.py"
if [[ ! -f "${viewer}" ]]; then
  refuse "AGENTIC_REMOTE_EXPERIMENT_DIR names a folder without thor_viewer.py: ${experiment}"
fi
apai="${AGENTIC_REMOTE_APAI_SRC:-}"
if [[ -z "${apai}" || ! -d "${apai}" ]]; then
  refuse "AGENTIC_REMOTE_APAI_SRC must name the src folder of the APAI checkout (it is '${apai}')"
fi

# The viewer's own defaults for the runs and the token lie inside the experiment
# folder, which this example only reads: both get a place in the state folder.
# An argument of the caller comes later on the command line and wins.
defaults=(--host 127.0.0.1)
own_state=0
if ((runs_given)); then
  [[ -d "${runs}" ]] || refuse "--runs names no folder: '${runs}'"
else
  runs="${state}/runs"
  defaults+=(--runs "${runs}")
  own_state=1
fi
if ((auth)) && ! ((token_given)); then
  token="${state}/viewer.token"
  defaults+=(--token-file "${token}")
  own_state=1
fi
if ((own_state)); then
  mkdir -p "${state}/runs" 2> /dev/null \
    || refuse "the state folder cannot be made: ${state}; name the runs with --runs, and the token with --token-file or --no-auth"
  # As the agent program leaves it: git ignores everything in the state folder.
  if [[ ! -e "${state}/.gitignore" ]]; then
    printf '*\n' > "${state}/.gitignore"
  fi
fi

echo "run_viewer.sh: the runs in ${runs}; a run is listed when it was recorded with --record-debug"
if ((auth)); then
  # The viewer answers a browser only after it was opened once with the token.
  if [[ -r "${token}" ]]; then
    echo "run_viewer.sh: once per browser, open the address printed below with this behind it: ?token=$(< "${token}")"
  else
    echo "run_viewer.sh: once per browser, open the address printed below with ?token= and the text of ${token} behind it; the viewer makes that file now"
  fi
fi

# The viewer needs no GPU. Should a package it imports look for one, it finds
# the card of the other programs: index 0 in PCI order.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
# Nothing is written into APAI's checkout or the experiment folder.
export PYTHONDONTWRITEBYTECODE=1
unset LD_LIBRARY_PATH
export PYTHONPATH="${apai}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${python}" "${viewer}" "${defaults[@]}" "$@"
