#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
args=()
cached_yaml="${here}/../argus_sender/build/_deps/yaml_cpp-src"
if [[ -f "${cached_yaml}/CMakeLists.txt" ]]; then
    args+=("-DFETCHCONTENT_SOURCE_DIR_YAML_CPP=${cached_yaml}")
fi
cmake -S "${here}" -B "${here}/build" -DCMAKE_BUILD_TYPE=Release "${args[@]}" "$@"
cmake --build "${here}/build" --parallel "${BUILD_JOBS:-4}"
