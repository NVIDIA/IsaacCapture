#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Start the workstation's programs of demo 1 on the rig in one tmux window with
# five panes, all visible together: the operator console, the agent program, the
# robot link service, the camera receiver and the experiment's viewer
# (run_viewer.sh: a web page on the runs recorded with --record-debug). The
# console and the agent lie side by side in the first row, the service and the
# camera receiver in the second, the viewer in the third.
#   ./run_workstation.sh --env FILE --thor ADDRESS
#   ./run_workstation.sh --env FILE --thor ADDRESS --print   # only print the commands
#   ./run_workstation.sh --env FILE --thor ADDRESS --live-view   # a sixth pane
#
# The live view (run_view.sh: a web page with the camera picture and the robot's
# state) is not started: the person at the controller stands at the arm and sees
# it. --live-view adds its pane, view, after the service's: three rows of two
# panes. It is for a person who drives the arm from where they cannot see it,
# with human.require_live_view true in the robot app's file on the Thor.
#
# The service and the viewer start at once. The camera receiver starts at once
# when --thor is given. The console, and the view with --live-view, attach once
# to the service's local socket and end when nothing listens there: their
# commands are sent once the service takes a connection on that socket, waited
# for at most 30 s (--socket-wait SECONDS). When that wait runs out, their
# commands are typed into their panes and wait for Enter. The agent's command is
# typed into its pane and waits for Enter: start it when the console shows the
# robot connection up. It carries the arguments of the experiment's own
# launcher, --record-debug among them, with which the viewer lists the run.
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
# Optional, from FILE or the environment:
#   AGENTIC_REMOTE_VIEWER_HOST       the address the viewer listens on, for
#                                    example this machine's address on the
#                                    internal network, or 0.0.0.0 for every
#                                    interface. Without it: 127.0.0.1 and the
#                                    viewer's token. With it anyone who can
#                                    reach that address can see the camera
#                                    pictures, the maps and the timings of the
#                                    runs, without a token
#
# Exit code 2 and nothing started: an argument or a variable that cannot be used,
# no tmux, or a session of this name that exists already.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
session="agentic_remote"
env_file=""
thor=""
print_only=0
live_view=0
socket_wait_s=30
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
    --socket-wait)
      [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]] || refuse "--socket-wait needs whole seconds"
      socket_wait_s="$2"
      shift 2
      ;;
    --print) print_only=1; shift ;;
    --live-view) live_view=1; shift ;;
    --help | -h) sed -n '5,51p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)
      known="--env FILE, --thor ADDRESS, --session NAME, --socket-wait SECONDS"
      refuse "unknown argument '$1' (${known}, --live-view, --print)"
      ;;
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
viewer_host="${AGENTIC_REMOTE_VIEWER_HOST:-}"
if [[ -n "${viewer_host}" && ! "${viewer_host}" =~ ^[A-Za-z0-9][A-Za-z0-9.-]*$ ]]; then
  plain="an IPv4 address or a host name (letters, digits, dots, hyphens)"
  refuse "AGENTIC_REMOTE_VIEWER_HOST must be ${plain}"
fi

# One prefix for every window: a tmux server that runs already does not hand
# this script's environment to a new window.
prefix="env PYTHONDONTWRITEBYTECODE=1"
prefix+=" AGENTIC_REMOTE_CONSOLE_PYTHON=$(printf '%q' "${python}")"
if ((live_view)); then
  prefix+=" AGENTIC_REMOTE_VIEW_PYTHON=$(printf '%q' "${python}")"
fi
for name in "${names[@]}"; do
  prefix+=" ${name}=$(printf '%q' "${!name}")"
done
module_path="${here}/python:${AGENTIC_REMOTE_ISAAC_CAPTURE_PY}"

service="${prefix} PYTHONPATH=$(printf '%q' "${module_path}") $(printf '%q' "${python}")"
service+=" -m isaaccapture_examples.agentic_remote.link_service --config ${service_config}"
console="${prefix} ./run_console.sh"
view="${prefix} ./run_view.sh --config ${view_config}"
camera="${prefix} ./run_camera_receiver.sh --host ${thor:-THOR_ADDRESS}"
# The arguments the experiment's own launcher is started with on the rig.
agent="${prefix} ./run_agent.sh --config ${agent_config} --operate --interactive"
agent+=" --record-debug --depth-backend ffs --motion-approval task --batch-milestones"
viewer="${prefix} ./run_viewer.sh"
# Where the viewer's page is, on the viewer's own port.
page="http://127.0.0.1:8766/ ; it asks for the token in .state/viewer.token:"
page+=" open http://127.0.0.1:8766/?token=<the token>"
if [[ -n "${viewer_host}" ]]; then
  viewer+=" --host ${viewer_host} --no-auth"
  shown="${viewer_host}"
  if [[ "${viewer_host}" == "0.0.0.0" ]]; then
    shown="<this machine's address>"
  fi
  page="http://${shown}:8766/ (no token; open to everyone who can reach that address)"
fi

# The service's local socket, as link_service/local.py finds it: socket_path of
# the service's file, else the default path.
socket=""
while IFS= read -r line; do
  if [[ "${line}" =~ ^socket_path:[[:space:]]*(.*)$ ]]; then
    socket="${BASH_REMATCH[1]}"
    break
  fi
done < "${here}/${service_config}"
socket="${socket%%#*}"
socket="${socket//[\"\'[:space:]]/}"
if [[ -z "${socket}" ]]; then
  if [[ -n "${XDG_RUNTIME_DIR:-}" ]]; then
    socket="${XDG_RUNTIME_DIR}/agentic_remote/robot_link.sock"
  else
    socket="/tmp/agentic_remote-$(id -u)/robot_link.sock"
  fi
fi

if ((print_only)); then
  once="once the service takes a connection on ${socket}"
  echo "cd $(printf '%q' "${here}")"
  printf '%s\n' "# service" "${service}" "# console (${once})" "${console}"
  if ((live_view)); then
    printf '%s\n' "# view (${once})" "${view}"
  fi
  printf '%s\n' "# camera receiver" "${camera}" "# viewer: ${page}" "${viewer}" \
    "# agent (when the robot connection is up)" "${agent}"
  exit 0
fi

# Whether the robot link service listens on its local socket. A socket file a
# dead service left takes no connection.
listening() {
  [[ -S "${socket}" ]] || return 1
  "${python}" -c '
import socket, sys
probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
probe.settimeout(1.0)
probe.connect(sys.argv[1])
probe.close()' "${socket}" 2> /dev/null
}

command -v tmux > /dev/null || refuse "tmux is not on PATH; --print gives the commands"
if tmux has-session -t "=${session}" 2> /dev/null; then
  refuse "a tmux session '${session}' exists: tmux attach -t ${session}, or end it first"
fi

# One window with five panes, six with --live-view, all visible together, each
# with its name on its border. tmux's tiled layout sets them in rows of two, in
# the order they are made; a fifth pane fills the third row. A pane keeps its
# last lines when its program ends. A click selects a pane.
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
view_pane=""
if ((live_view)); then
  view_pane="$(pane view)"
fi
camera_pane="$(pane camera)"
viewer_pane="$(pane viewer)"
tmux send-keys -t "${service_pane}" "${service}" Enter
tmux send-keys -t "${viewer_pane}" "${viewer}" Enter
if [[ -n "${thor}" ]]; then
  tmux send-keys -t "${camera_pane}" "${camera}" Enter
else
  tmux send-keys -t "${camera_pane}" "${camera}"
fi
tmux send-keys -t "${agent_pane}" "${agent}"
# The console, and the view when it is asked for, attach once: not before the
# service listens.
service_up=1
looks_left=$((socket_wait_s * 5))
until listening; do
  if ((looks_left <= 0)); then
    service_up=0
    break
  fi
  looks_left=$((looks_left - 1))
  sleep 0.2
done
attach() { # attach PANE COMMAND: started when the service listens, else typed
  if ((service_up)); then
    tmux send-keys -t "$1" "$2" Enter
  else
    tmux send-keys -t "$1" "$2"
  fi
}
attach "${console_pane}" "${console}"
if ((live_view)); then
  attach "${view_pane}" "${view}"
fi
tmux select-pane -t "${console_pane}"

panes="console, agent, service, camera, viewer"
waiting="The console's command waits for Enter in its pane"
if ((live_view)); then
  panes="console, agent, service, view, camera, viewer"
  waiting="The console's and the view's commands wait for Enter in their panes"
fi
echo "run_workstation.sh: tmux session '${session}': panes ${panes}"
if ((!live_view)); then
  echo "  the live view is not started (--live-view adds it): the person at the"
  echo "  controller must see the arm itself"
fi
if ((!service_up)); then
  echo "  the robot link service took no connection on ${socket} within"
  echo "  ${socket_wait_s} s: look at the pane service."
  echo "  ${waiting}: press it once the service runs"
fi
echo "  attach with: tmux attach -t ${session}   (click a pane to type in it)"
echo "  the agent's command waits for Enter in its pane; the pane viewer shows its"
echo "  run in a browser"
echo "  viewer: ${page}"
