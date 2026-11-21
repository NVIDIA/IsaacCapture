#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${project_dir}/build"

if [[ "$(uname -m)" != "aarch64" || ! -f /etc/nv_tegra_release ]]; then
    echo "error: build.sh must run on an NVIDIA Jetson (AGX Orin or Thor)." >&2
    exit 1
fi

if ! command -v apt-get >/dev/null 2>&1 || ! command -v dpkg-query >/dev/null 2>&1; then
    echo "error: this script requires the Ubuntu apt package manager." >&2
    exit 1
fi

packages=(
    build-essential
    ca-certificates
    cmake
    libdrm-dev
    libegl-dev
    nvidia-l4t-jetson-multimedia-api
)

missing_packages=()
for package in "${packages[@]}"; do
    if [[ "$(dpkg-query -W -f='${Status}' "${package}" 2>/dev/null || true)" != \
          "install ok installed" ]]; then
        missing_packages+=("${package}")
    fi
done

if ((${#missing_packages[@]})); then
    if ((EUID == 0)); then
        apt_prefix=()
    elif command -v sudo >/dev/null 2>&1; then
        apt_prefix=(sudo)
    else
        echo "error: installing dependencies requires root or sudo." >&2
        exit 1
    fi

    echo "Installing missing build dependencies: ${missing_packages[*]}"
    "${apt_prefix[@]}" apt-get update
    "${apt_prefix[@]}" apt-get install -y --no-install-recommends \
        "${missing_packages[@]}"
else
    echo "All build dependencies are already installed."
fi

cmake -S "${project_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release -DDISABLE_MULTIPROCESS=ON "$@"
cmake --build "${build_dir}" --target argus_sender -j "$(nproc)"

echo "Built ${build_dir}/argus_sender"
