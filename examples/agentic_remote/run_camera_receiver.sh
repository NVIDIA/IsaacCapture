#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Start the camera receiver of the camera pull request on the workstation with
# this rig's configuration: set its environment, then exec its own run.sh.
#   ./run_camera_receiver.sh --host <thor address>
#
# Needs in the environment:
#   AGENTIC_REMOTE_CAMERA_PR  the camera_streamer example of the camera pull request
# A first argument that does not begin with "-" is another configuration file.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pr="${AGENTIC_REMOTE_CAMERA_PR:-}"
if [[ -z "${pr}" || ! -x "${pr}/camera_receiver/run.sh" ]]; then
  echo "run_camera_receiver.sh: AGENTIC_REMOTE_CAMERA_PR must name the camera_streamer example of the camera pull request (it is '${pr}')" >&2
  exit 2
fi

if [[ $# -gt 0 && "$1" != -* ]]; then
  config="$1"
  shift
else
  config="${here}/configs/camera_receiver_rebot.yaml"
fi

# The same card as the agent program and the camera frame bridge: index 0 in
# PCI order.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
unset LD_LIBRARY_PATH

exec "${pr}/camera_receiver/run.sh" "${config}" "$@"
