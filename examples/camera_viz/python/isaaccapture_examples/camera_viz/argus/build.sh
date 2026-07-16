#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Builds the native Argus camera source. The .so lands next to
# __init__.py so the package imports straight from the source tree.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
build_dir="${here}/build"

if [[ -z "${CUDACXX:-}" && -x /usr/local/cuda/bin/nvcc ]]; then
    export CUDACXX=/usr/local/cuda/bin/nvcc
    export CUDA_PATH=/usr/local/cuda
    export PATH="/usr/local/cuda/bin:${PATH}"
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "build.sh: no active venv - activate examples/camera_viz/.venv first" >&2
    exit 1
fi
if ! python -c "import pybind11" 2>/dev/null; then
    echo "build.sh: installing pybind11 into venv..." >&2
    uv pip install pybind11
fi

generator=()
if command -v ninja >/dev/null 2>&1; then
    generator=(-G Ninja)
fi

cmake -S "${here}" -B "${build_dir}" "${generator[@]}" \
    "-UCUDA_*" \
    "-UCUDAToolkit_*" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDAToolkit_ROOT="${CUDA_PATH:-/usr/local/cuda}" \
    -DPython3_EXECUTABLE="$(command -v python)"
cmake --build "${build_dir}" --parallel

echo
echo "build.sh: argus source built -> ${here}/_camera_viz_argus*.so"
