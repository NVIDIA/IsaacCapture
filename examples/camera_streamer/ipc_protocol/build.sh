#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Builds the slot marker module. The .so lands next to __init__.py so the package imports straight from the source
# tree. Needs only a C++17 compiler and pybind11; no CUDA.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
python_exe="${IPC_PROTOCOL_PYTHON:-${CAMERA_RECEIVER_PYTHON:-python3}}"

if ! command -v "$python_exe" >/dev/null 2>&1 && [[ ! -x "$python_exe" ]]; then
    echo "build.sh: no Python at $python_exe" >&2
    echo "build.sh: override with IPC_PROTOCOL_PYTHON=/path/to/python" >&2
    exit 1
fi

if ! "$python_exe" -c "import pybind11" 2>/dev/null; then
    echo "build.sh: installing pybind11 into the Python environment..." >&2
    "$python_exe" -m pip install pybind11
fi

# The module's file name carries the interpreter's ABI tag, so each interpreter builds in its own tree.
build_dir="${here}/build/$("$python_exe" -c 'import sys; print(f"py{sys.version_info[0]}{sys.version_info[1]}")')"

generator=()
if command -v ninja >/dev/null 2>&1; then
    generator=(-G Ninja)
fi

cmake -S "${here}" -B "${build_dir}" "${generator[@]}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE="$(command -v "$python_exe")"
cmake --build "${build_dir}" --parallel

echo
echo "build.sh: built → ${here}/_slot_markers*.so"
