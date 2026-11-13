#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install the RevoHuman glove plugin: download the pinned RevoHuman C SDK release
# (or use the one given with --sdk-root), verify it against a pinned per-arch
# SHA-256, then configure, build, and install the plugin with the SDK's shared
# library next to it. Run with --help for the options.

set -euo pipefail

REVOHUMAN_SDK_C_VERSION="0.1.3"
REVOHUMAN_SDK_C_REPO="BrainCoTech/brainco-revohuman-sdk"
REVOHUMAN_SDK_C_TAG="c-sdk-v0.1.3"
# Pinned SHA-256 per release asset; change it with the version and tag above.
declare -A REVOHUMAN_SDK_C_SHA256=(
    ["x86_64"]="1e60d934375470f970805b7299e9f67600963183c5b83a3cb45049bb92e82fac"
    ["aarch64"]="9d628c90907aab34e54d5a1acfd96781f0d7a33ecfba4ced4ce53793b7c6e8b8"
)

usage() {
    cat <<EOF
Usage: $(basename "$0") [--sdk-root <dir>] [--build-dir <path>]
                  [--install-prefix <dir>] [<isaacteleop-root>]
                  [-- <cmake configure args>...]

Builds and installs the RevoHuman glove plugin with the RevoHuman C SDK library
next to it. Without --sdk-root (or REVOHUMAN_SDK_ROOT), downloads the pinned
C SDK release instead.

Options:
  --sdk-root <dir>    Already installed C SDK prefix with include/revohuman.h
                      and lib/librevohuman.so (env: REVOHUMAN_SDK_ROOT).
  --build-dir <path>  CMake build directory (default: <isaacteleop-root>/build).
  --install-prefix <dir>
                      Install prefix for the plugin, passed to
                      \`cmake --install --prefix\` (default: the build's
                      CMAKE_INSTALL_PREFIX, i.e. <isaacteleop-root>/install).
  -h, --help          Show this help.
  -- <args>...        Pass the remaining arguments to the CMake configure step,
                      e.g. \`-- -DBUILD_VIZ=OFF\` when the Televiz build tools
                      (glslangValidator, wayland-scanner) are not installed.

The option takes precedence over the environment variable.

Needs CMake 3.24 or newer first on PATH (the top-level CMakeLists.txt requires
it). The build runs \`cmake --build --parallel\` without a job count; set
CMAKE_BUILD_PARALLEL_LEVEL to limit the number of jobs.
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

# Absolute form of a path; the directory need not exist yet.
abspath() {
    if [[ "$1" == /* ]]; then
        echo "$1"
    else
        echo "$PWD/$1"
    fi
}

build_dir=""
isaac_root=""
sdk_root="${REVOHUMAN_SDK_ROOT:-}"
root_from="REVOHUMAN_SDK_ROOT"
install_prefix=""
cmake_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)
            build_dir="${2:?--build-dir requires a non-empty path}"
            shift 2
            ;;
        --sdk-root)
            sdk_root="${2:?--sdk-root requires a non-empty path}"
            root_from="--sdk-root"
            shift 2
            ;;
        --install-prefix)
            install_prefix="${2:?--install-prefix requires a non-empty path}"
            shift 2
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        --)
            shift
            cmake_args=("$@")
            break
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            if [[ -n "$isaac_root" ]]; then
                echo "Only one Isaac Teleop root may be specified" >&2
                exit 1
            fi
            isaac_root="$1"
            shift
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_ROOT="${isaac_root:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
BUILD_DIR="$(abspath "${build_dir:-$ISAAC_ROOT/build}")"

download_work=""
trap '[[ -z "$download_work" ]] || rm -rf "$download_work"' EXIT

# 1. RevoHuman C SDK (include/revohuman.h + lib/librevohuman.so).
if [[ -n "$sdk_root" ]]; then
    [[ -d "$sdk_root" ]] || die "$root_from: no such directory: $sdk_root"
    sdk_root="$(cd "$sdk_root" && pwd)"
    echo "==> Using installed RevoHuman C SDK: $sdk_root"
else
    # No SDK root given: download the pinned release and use it as the SDK root.
    command -v curl >/dev/null 2>&1 || die "curl is required to download the pinned RevoHuman C SDK release"
    case "$(uname -m)" in
        aarch64 | arm64) rh_arch="aarch64" ;;
        x86_64) rh_arch="x86_64" ;;
        *) die "unsupported arch: $(uname -m)" ;;
    esac
    expected_sha256="${REVOHUMAN_SDK_C_SHA256[$rh_arch]:-}"
    [[ -n "$expected_sha256" ]] \
        || die "no pinned RevoHuman C SDK release asset for arch '$rh_arch' yet" \
            "(see the SHA-256 map at the top of this script); pass --sdk-root instead"
    asset_base="revohuman-c-sdk-${REVOHUMAN_SDK_C_VERSION}-linux-${rh_arch}"
    asset_name="${asset_base}.tar.gz"
    sdk_root="$SCRIPT_DIR/$asset_base"
    # Keyed on the verified SHA-256, not the directory name, so a release
    # re-published under the same version is downloaded again.
    stamp_file="$sdk_root/.archive-sha256"
    if [[ -f "$sdk_root/include/revohuman.h" && -e "$sdk_root/lib/librevohuman.so" \
        && "$(cat "$stamp_file" 2>/dev/null)" == "$expected_sha256" ]]; then
        echo "==> Reusing extracted RevoHuman C SDK: $sdk_root"
    else
        download_work="$(mktemp -d)"
        archive="$download_work/$asset_name"
        echo "==> Downloading $asset_name (${REVOHUMAN_SDK_C_REPO}@${REVOHUMAN_SDK_C_TAG})"
        curl -fL --retry 3 \
            "https://github.com/${REVOHUMAN_SDK_C_REPO}/releases/download/${REVOHUMAN_SDK_C_TAG}/${asset_name}" \
            -o "$archive" \
            || die "download failed: $asset_name from ${REVOHUMAN_SDK_C_REPO}@${REVOHUMAN_SDK_C_TAG}"
        actual_sha256="$(sha256sum "$archive" | awk '{print $1}')"
        if [[ "$actual_sha256" != "$expected_sha256" ]]; then
            echo "SHA-256 mismatch for $asset_name" >&2
            echo "  expected: $expected_sha256" >&2
            echo "  actual:   $actual_sha256" >&2
            exit 1
        fi
        rm -rf "$sdk_root"
        tar -xzf "$archive" -C "$SCRIPT_DIR"
        echo "$expected_sha256" >"$stamp_file"
    fi
fi
[[ -f "$sdk_root/include/revohuman.h" && -e "$sdk_root/lib/librevohuman.so" ]] || {
    echo "RevoHuman C SDK not found under $sdk_root (need include/revohuman.h and lib/librevohuman.so)" >&2
    exit 1
}

# 2. Configure, build, and install the plugin target.
echo "==> Building revohuman_plugin"
# The arguments after -- come after the build type, so they can change it, and before the
# plugin options, so they cannot turn the plugin off or point it at another SDK.
cmake -B "$BUILD_DIR" -S "$ISAAC_ROOT" -DCMAKE_BUILD_TYPE=Release \
    "${cmake_args[@]}" \
    -DBUILD_PLUGINS=ON \
    -DBUILD_PLUGIN_REVOHUMAN=ON \
    -DREVOHUMAN_SDK_ROOT="$sdk_root" \
    || die "configuring $ISAAC_ROOT failed, see the output above; configure options such as" \
        "-DBUILD_VIZ=OFF go after -- on this script's command line"

# The plugin CMake fails on an incomplete SDK and records whether it added the plugin, which
# it does not on an unsupported platform.
plugin_available="$(awk '/^REVOHUMAN_PLUGIN_AVAILABLE:INTERNAL=/{sub(/^[^=]*=/, ""); print; exit}' \
    "$BUILD_DIR/CMakeCache.txt")"
if [[ "$plugin_available" != "ON" ]]; then
    die "the RevoHuman plugin was not configured with the C SDK in $sdk_root; see the configure" \
        "output above"
fi

cmake --build "$BUILD_DIR" --target revohuman_plugin --parallel

# The plugin, plugin.yaml and every librevohuman.so name land in plugins/revohuman/; the
# plugin's $ORIGIN RPATH finds the library there, so any prefix works.
if [[ -n "$install_prefix" ]]; then
    install_prefix="$(abspath "$install_prefix")"
    cmake --install "$BUILD_DIR" --component revohuman --prefix "$install_prefix"
else
    cmake --install "$BUILD_DIR" --component revohuman
    install_prefix="$(awk '/^CMAKE_INSTALL_PREFIX:PATH=/{sub(/^[^=]*=/, ""); print; exit}' "$BUILD_DIR/CMakeCache.txt")"
fi
echo "==> Done: $install_prefix/plugins/revohuman/revohuman_plugin"
