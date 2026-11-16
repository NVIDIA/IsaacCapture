#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Start the live view on the workstation: set its environment, then exec
#   python -m isaaccapture_examples.agentic_remote.view "$@"
# for example
#   ./run_view.sh --config configs/view.yaml
#
# Needs in the environment:
#   AGENTIC_REMOTE_VIEW_PYTHON   a Python with numpy, OpenCV and PyYAML; without
#                                it AGENTIC_REMOTE_AGENT_PYTHON is taken
#   AGENTIC_REMOTE_CAMERA_PR     the camera_streamer example of the camera pull
#                                request (--source link)
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python="${AGENTIC_REMOTE_VIEW_PYTHON:-${AGENTIC_REMOTE_AGENT_PYTHON:-}}"
if [[ -z "${python}" || ! -x "${python}" ]]; then
  echo "run_view.sh: AGENTIC_REMOTE_VIEW_PYTHON (or AGENTIC_REMOTE_AGENT_PYTHON) must name a Python with numpy, OpenCV and PyYAML (it is '${python}')" >&2
  exit 2
fi
if [[ -n "${AGENTIC_REMOTE_CAMERA_PR:-}" ]]; then
  export AGENTIC_REMOTE_CAMERA_PR
fi

# The view itself needs no GPU. Its camera frame bridge does, and must sit on
# the card of the camera receiver: index 0 in PCI order.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
unset LD_LIBRARY_PATH
export PYTHONPATH="${here}/python${PYTHONPATH:+:${PYTHONPATH}}"

exec "${python}" -m isaaccapture_examples.agentic_remote.view "$@"
