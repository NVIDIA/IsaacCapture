#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Create camera_receiver's local Python environment and build its local decoder.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${CAMERA_RECEIVER_VENV:-${HERE}/.venv}"
BOOTSTRAP_PYTHON="${CAMERA_RECEIVER_BOOTSTRAP_PYTHON:-${PYTHON:-python3}}"

if ! command -v "$BOOTSTRAP_PYTHON" >/dev/null 2>&1; then
  echo "setup.sh: Python not found: $BOOTSTRAP_PYTHON" >&2
  echo "setup.sh: set CAMERA_RECEIVER_BOOTSTRAP_PYTHON=/path/to/python3" >&2
  exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "setup.sh: cmake is required to build codec/" >&2
  exit 1
fi

if [[ -z "${CUDACXX:-}" && ! -x /usr/local/cuda/bin/nvcc ]] && ! command -v nvcc >/dev/null 2>&1; then
  echo "setup.sh: nvcc not found. Install CUDA toolkit or set CUDACXX=/path/to/nvcc" >&2
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
  uv pip check --python "$PYTHON_EXE"
else
  "$PYTHON_EXE" -m pip install --upgrade pip
  "$PYTHON_EXE" -m pip install -r "${HERE}/requirements.txt"
  "$PYTHON_EXE" -m pip check
fi

echo "setup.sh: verifying NumPy/CuPy runtime"
"$PYTHON_EXE" - <<'PY'
import cupy as cp
import numpy as np

device_count = cp.cuda.runtime.getDeviceCount()
if device_count < 1:
    raise RuntimeError("camera_receiver requires at least one CUDA device")
probe = cp.cuda.alloc(1)
cp.cuda.runtime.memset(probe.ptr, 0, 1)
cp.cuda.runtime.deviceSynchronize()
print(
    f"setup.sh: NumPy {np.__version__}; CuPy {cp.__version__}; "
    f"CUDA devices={device_count}"
)
PY

export CAMERA_RECEIVER_VENV="$VENV"
export CAMERA_RECEIVER_PYTHON="$PYTHON_EXE"

if [[ "${CAMERA_RECEIVER_CLEAN_BUILD:-1}" != "0" ]]; then
  echo "setup.sh: cleaning codec/build"
  rm -rf "${HERE}/codec/build"
fi

echo "setup.sh: building native decoder"
"${HERE}/codec/build.sh"

echo "setup.sh: verifying receiver config"
"$PYTHON_EXE" "${HERE}/camera_receiver.py" \
  --config "${HERE}/configs/receiver.yaml" \
  --dry-run

echo
echo "setup.sh: ready"
echo "setup.sh: run with ./run.sh"
