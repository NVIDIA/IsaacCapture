#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Start the workstation's programs of demo 1 on the rig in one tmux window with
# five panes, all visible together: the operator console, the agent program, the
# robot link service, the live view and the camera receiver.
#   ./run_workstation.sh --env FILE --thor ADDRESS
#   ./run_workstation.sh --env FILE --thor ADDRESS --print   # only print the commands
#
# The service, the console and the view start at once. The camera receiver starts
# at once when --thor is given. The agent's command is typed into its pane and
# waits for Enter: start it when the console shows the robot connection up.
#
# FILE is a shell file that sets the variables below (KEY=value lines). They may
# also be set in the environment, and --env left out:
#   AGENTIC_REMOTE_AGENT_PYTHON      the Python of the workstation environment
#   AGENTIC_REMOTE_ENV_DIR           that environment's folder
#   AGENTIC_REMOTE_ISAAC_CAPTURE_PY  the Isaac Capture Python package built for it
#   AGENTIC_REMOTE_APAI_SRC          the src folder of the APAI checkout
#   AGENTIC_REMOTE_EXPERIMENT_DIR    the experiment folder
#   AGENTIC_REMOTE_CAMERA_PR         the camera_streamer example of the camera
#                                    pull request
#
# Exit code 2 and nothing started: an argument or a variable that cannot be used,
# no tmux, or a session of this name that exists already.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
session="agentic_remote"
env_file=""
thor=""
print_only=0
service_config="configs/link_service_rebot.yaml"
view_config="configs/view.yaml"
agent_config="configs/agent_rebot.yaml"

refuse() {
  echo "run_workstation.sh: $1" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) [[ $# -ge 2 ]] || refuse "--env needs a file"; env_file="$2"; shift 2 ;;
    --thor) [[ $# -ge 2 ]] || refuse "--thor needs an address"; thor="$2"; shift 2 ;;
    --session) [[ $# -ge 2 ]] || refuse "--session needs a name"; session="$2"; shift 2 ;;
    --print) print_only=1; shift ;;
    --help | -h) sed -n '5,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) refuse "unknown argument '$1' (--env FILE, --thor ADDRESS, --session NAME, --print)" ;;
  esac
done

if [[ -n "${env_file}" ]]; then
  [[ -f "${env_file}" ]] || refuse "--env names no file: ${env_file}"
  set -a
  # shellcheck disable=SC1090
  source "${env_file}"
  set +a
fi

names=(
  AGENTIC_REMOTE_AGENT_PYTHON
  AGENTIC_REMOTE_ENV_DIR
  AGENTIC_REMOTE_ISAAC_CAPTURE_PY
  AGENTIC_REMOTE_APAI_SRC
  AGENTIC_REMOTE_EXPERIMENT_DIR
  AGENTIC_REMOTE_CAMERA_PR
)
for name in "${names[@]}"; do
  [[ -n "${!name:-}" ]] || refuse "${name} is not set (see the head of this script)"
done
python="${AGENTIC_REMOTE_AGENT_PYTHON}"
[[ -x "${python}" ]] || refuse "AGENTIC_REMOTE_AGENT_PYTHON names no Python: ${python}"
for folder in AGENTIC_REMOTE_ISAAC_CAPTURE_PY AGENTIC_REMOTE_APAI_SRC \
  AGENTIC_REMOTE_EXPERIMENT_DIR AGENTIC_REMOTE_CAMERA_PR; do
  [[ -d "${!folder}" ]] || refuse "${folder} names no folder: ${!folder}"
done

# One prefix for every window: a tmux server that runs already does not hand
# this script's environment to a new window.
prefix="env PYTHONDONTWRITEBYTECODE=1"
prefix+=" AGENTIC_REMOTE_CONSOLE_PYTHON=$(printf '%q' "${python}")"
prefix+=" AGENTIC_REMOTE_VIEW_PYTHON=$(printf '%q' "${python}")"
for name in "${names[@]}"; do
  prefix+=" ${name}=$(printf '%q' "${!name}")"
done
module_path="${here}/python:${AGENTIC_REMOTE_ISAAC_CAPTURE_PY}"

service="${prefix} PYTHONPATH=$(printf '%q' "${module_path}") $(printf '%q' "${python}")"
service+=" -m isaaccapture_examples.agentic_remote.link_service --config ${service_config}"
console="${prefix} ./run_console.sh"
view="${prefix} ./run_view.sh --config ${view_config}"
camera="${prefix} ./run_camera_receiver.sh --host ${thor:-THOR_ADDRESS}"
agent="${prefix} ./run_agent.sh --config ${agent_config} --interactive"

if ((print_only)); then
  echo "cd $(printf '%q' "${here}")"
  printf '%s\n' "# service" "${service}" "# console" "${console}" "# view" "${view}" \
    "# camera receiver" "${camera}" "# agent (when the robot connection is up)" "${agent}"
  exit 0
fi

command -v tmux > /dev/null || refuse "tmux is not on PATH; --print gives the commands"
if tmux has-session -t "=${session}" 2> /dev/null; then
  refuse "a tmux session '${session}' exists: tmux attach -t ${session}, or end it first"
fi

# One window with five panes, all visible together, each with its name on its
# border. A pane keeps its last lines when its program ends. A click selects a
# pane.
pane() { # pane NAME: a new pane of that name; prints its id
  local id
  id="$(tmux split-window -t "${session}:0" -c "${here}" -P -F '#{pane_id}')"
  tmux select-layout -t "${session}:0" tiled > /dev/null
  tmux select-pane -t "${id}" -T "$1"
  echo "${id}"
}
first="$(tmux new-session -d -s "${session}" -n demo1 -c "${here}" -P -F '#{pane_id}')"
tmux set-option -t "${session}" remain-on-exit on > /dev/null
tmux set-option -t "${session}" mouse on > /dev/null
tmux set-option -t "${session}" pane-border-status top > /dev/null
tmux set-option -t "${session}" pane-border-format ' #{pane_title} ' > /dev/null
tmux select-pane -t "${first}" -T console
console_pane="${first}"
agent_pane="$(pane agent)"
service_pane="$(pane service)"
view_pane="$(pane view)"
camera_pane="$(pane camera)"
tmux send-keys -t "${service_pane}" "${service}" Enter
tmux send-keys -t "${console_pane}" "${console}" Enter
tmux send-keys -t "${view_pane}" "${view}" Enter
if [[ -n "${thor}" ]]; then
  tmux send-keys -t "${camera_pane}" "${camera}" Enter
else
  tmux send-keys -t "${camera_pane}" "${camera}"
fi
tmux send-keys -t "${agent_pane}" "${agent}"
tmux select-pane -t "${console_pane}"

echo "run_workstation.sh: tmux session '${session}': panes console, agent, service, view, camera"
echo "  attach with: tmux attach -t ${session}   (click a pane to type in it)"
echo "  the agent's command waits for Enter in its pane"
