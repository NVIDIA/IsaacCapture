#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Start the experiment's viewer on the workstation: set its environment, then exec
#   python <experiment folder>/thor_viewer.py --host 127.0.0.1 --runs RUNS \
#     --token-file FILE "$@"
# or, with the live picture, this example's launcher, which runs that viewer as
# it is in its own process:
#   python -m isaaccapture_examples.agentic_remote.viewer_live \
#     --experiment-dir <experiment folder> --host 127.0.0.1 --runs RUNS \
#     --token-file FILE "$@"
# The experiment is the folder of the rig's earlier, single machine program. Its
# viewer is a web page for a browser on the runs the agent program records with
# --record-debug: the obstacle map, the measured joints, refused poses, the
# collision volumes, the clearance points and the timing graph, while a run goes
# on and afterwards. It reads files and connects to no robot.
# The live picture is the camera's picture as it arrives, on the same page,
# through a camera frame bridge of the launcher's own. The live arm is the arm
# of each ROBOT_STATE as it arrives, in the page's 3D view. Neither is stored.
# The viewer's page with its live picture showing is the live view: the
# launcher attaches to the robot link service and tells the robot, with
# VIEW_STATUS, how fresh the picture on that page is. A person's takeover needs
# it where the robot app's file says human.require_live_view: true. Without a
# robot link service the viewer serves everything and attaches once one runs.
#   ./run_viewer.sh                    # the runs in .state/runs, on 127.0.0.1
#   ./run_viewer.sh --runs FOLDER --port 8767 --no-auth
#   ./run_viewer.sh --no-live          # the experiment's viewer alone: nothing live
#
# Arguments, all optional; any other goes to the viewer as it is:
#   --runs FOLDER      the folder that holds the run folders. Default: .state/runs
#                      beside this script, where the agent program puts its runs
#                      with the shipped files; it is made when it is not there
#   --host ADDRESS     the address to listen on. Default: 127.0.0.1, this machine
#   --port NUMBER      default: the viewer's own, 8766
#   --token-file FILE  where the viewer keeps its access token. Default:
#                      .state/viewer.token beside this script
#   --no-auth          serve without a token
#   --no-live          no live picture, no live arm and no live view: the
#                      experiment's viewer is started alone
#   --live-fps N       the most live pictures a second, 1 to 60. Default: 30
#   --live-width PIXELS
#                      the live picture's greatest width, 64 to 4096. Default: 960
#   --receiver-socket PATH
#                      the camera receiver's socket. Default: the one
#                      configs/camera_receiver_rebot.yaml names
#   --link-socket PATH the robot link service's local socket, for the live view.
#                      Default: the one the console attaches to
#
# Needs in the environment:
#   AGENTIC_REMOTE_AGENT_PYTHON    a Python with APAI's packages, numpy, OpenCV
#                                  and PyYAML
#   AGENTIC_REMOTE_EXPERIMENT_DIR  the experiment folder, which holds thor_viewer.py
#   AGENTIC_REMOTE_APAI_SRC        the src folder of the APAI checkout
# For everything live (the picture, the arm, the live view), off without it:
#   AGENTIC_REMOTE_CAMERA_PR       the camera_streamer example of the camera
#                                  streamer's pull request
#
# Exit code 2 and nothing started: a variable that cannot be used, an argument
# of the above without its value, a --runs that names no folder, or a state
# folder that cannot be made.
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
live=1
live_off="--no-live"
# The launcher's own arguments, and every other one: the viewer's.
live_arguments=()
passed=()
previous=""
for argument in "$@"; do
  case "${previous}" in
    --runs) runs="${argument}"; runs_given=1 ;;
    --token-file) token="${argument}"; token_given=1 ;;
    --live-fps | --live-width | --receiver-socket | --link-socket)
      live_arguments+=("${previous}" "${argument}")
      previous=""
      continue
      ;;
  esac
  previous="${argument}"
  case "${argument}" in
    --runs=*) runs="${argument#--runs=}"; runs_given=1 ;;
    --token-file=*) token="${argument#--token-file=}"; token_given=1 ;;
    --no-auth) auth=0 ;;
    --no-live) live=0; continue ;;
    --live-fps | --live-width | --receiver-socket | --link-socket) continue ;;
    --live-fps=* | --live-width=* | --receiver-socket=* | --link-socket=*)
      live_arguments+=("${argument}")
      continue
      ;;
    --help | -h) sed -n '5,61p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
  esac
  passed+=("${argument}")
done
case "${previous}" in
  --runs | --token-file | --live-fps | --live-width | --receiver-socket | --link-socket)
    refuse "${previous} needs a value"
    ;;
esac
if ((live)) && [[ -z "${AGENTIC_REMOTE_CAMERA_PR:-}" ]]; then
  live=0
  live_off="AGENTIC_REMOTE_CAMERA_PR is not set"
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

if ((live)); then
  echo "run_viewer.sh: live picture: on; the page shows the camera's pictures and the arm of each ROBOT_STATE as they arrive, and stores neither (--no-live turns it off, and the live arm with it)"
else
  echo "run_viewer.sh: live picture: off (${live_off}); no live arm either: the page shows the stored pictures and the recorded arm of the runs only, and the robot is told of no live view"
fi

# The viewer needs no GPU. Should a package it imports look for one, it finds
# the card of the other programs: index 0 in PCI order. The live picture's
# camera frame bridge copies the pictures off that card.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
# Nothing is written into APAI's checkout or the experiment folder.
export PYTHONDONTWRITEBYTECODE=1
unset LD_LIBRARY_PATH
if ((live)); then
  export PYTHONPATH="${here}/python:${apai}${PYTHONPATH:+:${PYTHONPATH}}"
  exec "${python}" -m isaaccapture_examples.agentic_remote.viewer_live \
    --experiment-dir "${experiment}" "${live_arguments[@]}" "${defaults[@]}" "${passed[@]}"
fi
export PYTHONPATH="${apai}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${python}" "${viewer}" "${defaults[@]}" "${passed[@]}"
