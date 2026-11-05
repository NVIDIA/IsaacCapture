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
# Pinned SHA-256 per release asset; update together with the version/tag above
# (every SDK version bump needs a matching commit here, same tradeoff as
# wuji_glove's install.sh).
declare -A REVOHUMAN_SDK_C_SHA256=(
    ["x86_64"]="773dc63fd5daf62e4acceb90278e7bca493dc7fada3aad4febe45fb8934759dc"
    ["aarch64"]="bdd4a9ae3c1777f4c2dccf6c2be35d3a6ccba2098e289ff1677527042cc0f642"
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
    # Keyed on the verified SHA-256, not just the directory name: a re-published
    # release that keeps the same version must not silently reuse a tree
    # downloaded from the earlier one.
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

# The plugin CMake records whether the live source links against this SDK; without it there
# is no plugin executable to install.
live_available="$(awk '/^REVOHUMAN_LIVE_AVAILABLE:INTERNAL=/{sub(/^[^=]*=/, ""); print; exit}' \
    "$BUILD_DIR/CMakeCache.txt")"
if [[ "$live_available" != "ON" ]]; then
    echo "RevoHuman live source unavailable with the C SDK in $sdk_root" \
        "(REVOHUMAN_LIVE_AVAILABLE=${live_available:-unset}): the library is missing or lacks" \
        "functions the plugin calls; see the configure output above. There is no plugin to install." >&2
    exit 1
fi

# Hand poses come only from the SDK hand points, and every glove takes its hand model from the
# SDK's product table, so an SDK that links but lacks the hand-point or product-model API would
# still pass the check above and only fail later, at runtime. Catch that here instead.
for cache_var in REVOHUMAN_HAS_SDK_POINTS REVOHUMAN_HAS_SDK_VARIANT_MODEL; do
    case "$cache_var" in
        REVOHUMAN_HAS_SDK_POINTS) needed_by="live hand poses come only from rh_hand_points" ;;
        REVOHUMAN_HAS_SDK_VARIANT_MODEL) needed_by="no glove has a hand model without it" ;;
    esac
    value="$(awk -v v="$cache_var" '
        $0 ~ "^" v ":INTERNAL=" { sub(/^[^=]*=/, ""); print; found=1; exit }
        END { if (!found) print "__MISSING__" }
    ' "$BUILD_DIR/CMakeCache.txt")"
    if [[ "$value" != "1" ]]; then
        if [[ "$value" == "__MISSING__" ]]; then
            detail="not present in CMakeCache.txt"
        else
            detail="configure check failed"
        fi
        echo "RevoHuman C SDK in $sdk_root does not provide $cache_var" \
            "($detail): $needed_by" \
            "and live would fail at runtime instead. See kinematics/CMakeLists.txt for the exact" \
            "symbols checked. Refusing the install." >&2
        exit 1
    fi
done

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
