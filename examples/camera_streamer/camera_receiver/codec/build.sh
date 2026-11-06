#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Builds the native NVDEC decoder. The .so lands next to
# __init__.py so the package imports straight from the source tree.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
project_dir="$(cd "${here}/.." && pwd)"
build_dir="${here}/build"
venv_dir="${CAMERA_RECEIVER_VENV:-${project_dir}/.venv}"
python_exe="${CAMERA_RECEIVER_PYTHON:-${venv_dir}/bin/python}"

if [[ ! -x "$python_exe" ]]; then
    echo "build.sh: no Python at $python_exe" >&2
    echo "build.sh: override with CAMERA_RECEIVER_VENV=/path/to/venv or CAMERA_RECEIVER_PYTHON=/path/to/python" >&2
    exit 1
fi

export VIRTUAL_ENV="$venv_dir"
export PATH="${venv_dir}/bin:${PATH}"

# Prefer /usr/local/cuda's nvcc — distro nvcc is often too old for
# modern compute capabilities.
if [[ -z "${CUDACXX:-}" && -x /usr/local/cuda/bin/nvcc ]]; then
    export CUDACXX=/usr/local/cuda/bin/nvcc
    export CUDA_PATH=/usr/local/cuda
    export PATH="/usr/local/cuda/bin:${PATH}"
fi

if ! "$python_exe" -c "import pybind11" 2>/dev/null; then
    echo "build.sh: installing pybind11 into venv..." >&2
    "$python_exe" -m pip install pybind11
fi

cuda_architectures="${CMAKE_CUDA_ARCHITECTURES:-}"
if [[ -z "$cuda_architectures" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    cuda_architectures="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | head -n1 | tr -d '[:space:].')"
fi
if [[ -z "$cuda_architectures" ]]; then
    cuda_architectures="80;86;87;89;90"
fi
echo "build.sh: Python=${python_exe}"
echo "build.sh: CMAKE_CUDA_ARCHITECTURES=${cuda_architectures}"

generator=()
if command -v ninja >/dev/null 2>&1; then
    generator=(-G Ninja)
fi

cmake -S "${here}" -B "${build_dir}" "${generator[@]}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_architectures}" \
    -DPython3_EXECUTABLE="${python_exe}"
cmake --build "${build_dir}" --parallel

echo
echo "build.sh: decoder built → ${here}/_camera_receiver_codec*.so"
