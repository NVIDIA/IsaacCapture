#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

build_dir=""
sdk_root="${OMNIGLOVE_SDK_ROOT:-}"
isaac_root=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)
            build_dir="${2:?--build-dir requires a path}"
            shift 2
            ;;
        --sdk-root)
            sdk_root="${2:?--sdk-root requires a path}"
            shift 2
            ;;
        -h | --help)
            cat <<EOF
Usage: $(basename "$0") [--build-dir PATH] [--sdk-root PATH] [ISAAC_CAPTURE_ROOT]

Uses the current local OmniGlove SDK checkout, or updates a managed checkout,
then verifies it and builds and installs the Isaac Capture plugin.
EOF
            exit 0
            ;;
        -*)
            echo "error: unknown option $1" >&2
            exit 1
            ;;
        *)
            if [[ -n "$isaac_root" ]]; then
                echo "error: only one Isaac Capture root may be supplied" >&2
                exit 1
            fi
            isaac_root="$1"
            shift
            ;;
    esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
isaac_root="${isaac_root:-$(cd "$script_dir/../../.." && pwd)}"
build_dir="${build_dir:-$isaac_root/build}"
if [[ "$build_dir" != /* ]]; then
    build_dir="$PWD/$build_dir"
fi

workspace_sdk="$isaac_root/../agilink_omniglove_sdk"
if [[ -z "$sdk_root" ]] && [[ -d "$workspace_sdk/.git" ]]; then
    sdk_root="$workspace_sdk"
fi

if [[ -z "$sdk_root" ]]; then
    sdk_root="$script_dir/.sdk/agilink_omniglove_sdk"
    if [[ -d "$sdk_root/.git" ]]; then
        echo "==> Updating managed OmniGlove SDK checkout"
        git -C "$sdk_root" pull --ff-only
    else
        echo "==> Cloning latest OmniGlove SDK release repository"
        mkdir -p "$(dirname "$sdk_root")"
        git clone --depth 1 https://github.com/AgibotTech/agilink_omniglove_sdk.git "$sdk_root"
    fi
fi

sdk_root="$(cd "$sdk_root" && pwd)"
if [[ ! -f "$sdk_root/VERSION" ]] || [[ ! -f "$sdk_root/SHA256SUMS" ]]; then
    echo "error: $sdk_root is not a complete OmniGlove SDK release checkout" >&2
    exit 1
fi

echo "==> Verifying OmniGlove SDK $(tr -d '[:space:]' < "$sdk_root/VERSION")"
(cd "$sdk_root" && sha256sum --check --quiet SHA256SUMS)

case "$(uname -m)" in
    x86_64) platform="linux-x86_64" ;;
    aarch64 | arm64) platform="linux-aarch64" ;;
    *)
        echo "error: unsupported architecture $(uname -m)" >&2
        exit 1
        ;;
esac

platform_root="$sdk_root/$platform"
for required in \
    "$platform_root/cmake/OmnigloveSDKConfig.cmake" \
    "$platform_root/include/omniglove/omniglove.h"; do
    if [[ ! -f "$required" ]]; then
        echo "error: SDK artifact is missing: $required" >&2
        exit 1
    fi
done

# Follow the selected checkout's package config instead of pinning an ABI-named
# library here. CMake and the compatibility probe reject missing required APIs.

sdk_commit="unknown"
if [[ -d "$sdk_root/.git" ]]; then
    sdk_commit="$(git -C "$sdk_root" rev-parse HEAD)"
fi

echo "==> Configuring OmniGlove plugin against $platform ($sdk_commit)"
cmake -S "$isaac_root" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=ON \
    -DBUILD_PLUGINS=ON \
    -DBUILD_PLUGIN_OMNIGLOVE=ON \
    -DOmnigloveSDK_DIR="$platform_root/cmake" \
    -DOMNIGLOVE_SDK_RELEASE_ROOT="$sdk_root" \
    -DOMNIGLOVE_SDK_GIT_COMMIT="$sdk_commit"

cmake --build "$build_dir" --target \
    omniglove_api_compat_probe \
    omniglove_hand_conversion_contract_test \
    omniglove_plugin --parallel

ctest --test-dir "$build_dir" --output-on-failure -R '^omniglove_'
cmake --install "$build_dir" --component omniglove

install_prefix="$(awk '/^CMAKE_INSTALL_PREFIX:PATH=/{sub(/^[^=]*=/, ""); print; exit}' "$build_dir/CMakeCache.txt")"
echo "==> Installed $install_prefix/plugins/omniglove/omniglove_plugin"
echo "Install USB permissions once on the host with:"
echo "    $script_dir/install_udev_rules.sh"
