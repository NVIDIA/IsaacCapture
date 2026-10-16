#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build the SDK-only preview binding for the active Python environment.
# GPU decoding is built separately in codec/; recording uses the EGO plugin.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
camera_viz_dir="$(dirname "$here")"
sdk_root="${ORBBEC_SDK_ROOT:-}"
while (( $# )); do
    case $1 in
        --sdk-root)
            (( $# >= 2 )) || { echo "build.sh: --sdk-root requires a directory" >&2; exit 1; }
            sdk_root=$2; shift 2;;
        *) echo "build.sh: unknown argument: $1" >&2; exit 1;;
    esac
done
if [[ -z "$sdk_root" ]]; then
    echo "build.sh: --sdk-root PATH (or ORBBEC_SDK_ROOT) is required" >&2
    exit 1
fi
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "build.sh: activate examples/camera_viz/.venv first" >&2
    exit 1
fi
if [[ ! -f "$sdk_root/lib/libOrbbecSDK.so" ]]; then
    echo "build.sh: libOrbbecSDK.so was not found under: $sdk_root/lib" >&2
    exit 1
fi

generator=()
if command -v ninja >/dev/null 2>&1; then
    generator=(-G Ninja)
fi
cmake -S "$here" -B "$here/build" "${generator[@]}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE="$(command -v python)" \
    -UORBBEC_SDK_LIBRARY \
    -DORBBEC_SDK_ROOT="$sdk_root"
cmake --build "$here/build" --parallel
(cd "$camera_viz_dir" && python -c 'from ego_preview import Capture')
