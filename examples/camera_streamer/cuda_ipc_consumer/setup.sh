#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${CUDA_IPC_CONSUMER_VENV:-${HERE}/.venv}"
BOOTSTRAP_PYTHON="${CUDA_IPC_CONSUMER_BOOTSTRAP_PYTHON:-${PYTHON:-python3}}"

if ! command -v "$BOOTSTRAP_PYTHON" >/dev/null 2>&1; then
  echo "setup.sh: Python not found: $BOOTSTRAP_PYTHON" >&2
  echo "setup.sh: set CUDA_IPC_CONSUMER_BOOTSTRAP_PYTHON=/path/to/python3" >&2
  exit 1
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "setup.sh: creating venv at ${VENV}"
  if command -v uv >/dev/null 2>&1; then
    uv venv "${VENV}" --python "$BOOTSTRAP_PYTHON"
  else
    "$BOOTSTRAP_PYTHON" -m venv "${VENV}"
  fi
fi

PYTHON_EXE="${VENV}/bin/python"
export VIRTUAL_ENV="$VENV"
export PATH="${VENV}/bin:${PATH}"

echo "setup.sh: installing Python dependencies"
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$PYTHON_EXE" -r "${HERE}/requirements.txt"
  echo "setup.sh: checking Python dependency compatibility"
  uv pip check --python "$PYTHON_EXE"
else
  if ! "$PYTHON_EXE" -m pip --version >/dev/null 2>&1; then
    "$PYTHON_EXE" -m ensurepip --upgrade
  fi
  "$PYTHON_EXE" -m pip install --upgrade pip
  "$PYTHON_EXE" -m pip install -r "${HERE}/requirements.txt"
  echo "setup.sh: checking Python dependency compatibility"
  "$PYTHON_EXE" -m pip check
fi

echo "setup.sh: verifying imports"
"$PYTHON_EXE" - <<'PY'
import cupy
import cv2
import numpy

print(f"numpy={numpy.__version__}")
print(f"cupy={cupy.__version__}")
print(f"opencv={cv2.__version__}")
PY

echo
echo "setup.sh: ready"
echo "setup.sh: run with ./run.sh"
