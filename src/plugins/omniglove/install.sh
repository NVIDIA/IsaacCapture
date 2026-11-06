#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

build_dir=""
sdk_root="${OMNIGLOVE_SDK_ROOT:-}"
sdk_ref="${OMNIGLOVE_SDK_REF:-}"
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
        --sdk-ref)
            sdk_ref="${2:?--sdk-ref requires a commit, tag, or branch}"
            shift 2
            ;;
        -h | --help)
            cat <<EOF
Usage: $(basename "$0") [--build-dir PATH] [--sdk-root PATH | --sdk-ref REF] [ISAAC_CAPTURE_ROOT]

Uses a local OmniGlove SDK checkout, or a pinned managed checkout,
then verifies it and builds and installs the Isaac Capture plugin.
--sdk-ref (OMNIGLOVE_SDK_REF) selects a managed SDK commit, tag, or branch
and bypasses sibling checkout discovery. It cannot be combined with
--sdk-root (OMNIGLOVE_SDK_ROOT), which uses a local checkout unchanged.
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

if [[ -n "$sdk_root" && -n "$sdk_ref" ]]; then
    echo "error: SDK root and managed SDK ref cannot be supplied together" >&2
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
isaac_root="${isaac_root:-$(cd "$script_dir/../../.." && pwd)}"
build_dir="${build_dir:-$isaac_root/build}"
if [[ "$build_dir" != /* ]]; then
    build_dir="$PWD/$build_dir"
fi

workspace_sdk="$isaac_root/../agilink_omniglove_sdk"
if [[ -z "$sdk_root" && -z "$sdk_ref" ]] && [[ -e "$workspace_sdk/.git" ]]; then
    sdk_root="$workspace_sdk"
fi

if [[ -z "$sdk_root" ]]; then
    sdk_root="$script_dir/.sdk/agilink_omniglove_sdk"
    # Pin managed dependencies: their own SHA256SUMS detects corruption, not
    # authenticity. An upstream branch update must not silently change the SDK.
    sdk_ref="${sdk_ref:-a2f899cd45c51ae01c9bbca9b893e4afba41c0d7}"
    if [[ ! -e "$sdk_root/.git" ]]; then
        if [[ -e "$sdk_root" ]]; then
            echo "error: managed SDK path exists but is not a Git checkout: $sdk_root" >&2
            exit 1
        fi
        mkdir -p "$(dirname "$sdk_root")"
        git init --quiet "$sdk_root"
        git -C "$sdk_root" remote add origin https://github.com/AgibotTech/agilink_omniglove_sdk.git
    fi
    if [[ -n "$(git -C "$sdk_root" status --porcelain)" ]]; then
        echo "error: managed SDK checkout has local changes; preserve them before installing: $sdk_root" >&2
        exit 1
    fi
    if [[ "$sdk_ref" =~ ^[0-9a-fA-F]{40}$ ]] && git -C "$sdk_root" cat-file -e "$sdk_ref^{commit}" 2>/dev/null; then
        sdk_commit="$(git -C "$sdk_root" rev-parse "$sdk_ref^{commit}")"
    else
        git -C "$sdk_root" fetch --depth 1 -- origin "$sdk_ref"
        sdk_commit="$(git -C "$sdk_root" rev-parse 'FETCH_HEAD^{commit}')"
    fi
    echo "==> Using managed OmniGlove SDK $sdk_ref ($sdk_commit)"
    git -C "$sdk_root" checkout --detach "$sdk_commit" --
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
if [[ -e "$sdk_root/.git" ]]; then
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
