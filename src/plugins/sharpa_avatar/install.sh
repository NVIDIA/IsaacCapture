#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Configure, build, and install the Sharpa Avatar plugin.

set -euo pipefail

build_dir=""
isaac_root=""

usage() {
  cat <<EOF
Usage: $0 [--build-dir DIR] [isaac-teleop-root]

Options:
  --build-dir DIR  CMake build directory (default: <isaac-root>/build).
  -h, --help       Show this help.

Environment:
  AVATAR_SDK_ROOT  Avatar SDK installation to build against (default: /opt/avatar-sdk).
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      build_dir="${2:?--build-dir requires a path}"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    -*)
      die "Unknown option: $1"
      ;;
    *)
      [[ -z "$isaac_root" ]] || die "Only one Isaac Teleop root may be specified."
      isaac_root="$1"
      shift
      ;;
  esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
isaac_root="${isaac_root:-$(cd "$script_dir/../../.." && pwd)}"
build_dir="${build_dir:-$isaac_root/build}"
avatar_sdk_root="${AVATAR_SDK_ROOT:-/opt/avatar-sdk}"
avatar_sdk_root="${avatar_sdk_root%/}"

if [[ ! -f "$avatar_sdk_root/include/avatar_sdk/AvatarSDK.h" ]]; then
  if [[ "$avatar_sdk_root" != "/opt/avatar-sdk" ]]; then
    die "Avatar SDK not found under $avatar_sdk_root. Install it there or fix AVATAR_SDK_ROOT."
  fi
  if [[ -e "$avatar_sdk_root" ]]; then
    die "Avatar SDK installation under $avatar_sdk_root is incomplete. Re-run install_avatar_sdk.sh."
  fi
  echo "Avatar SDK is not installed; installing the official package."
  "$script_dir/install_avatar_sdk.sh"
fi

# Layout, build type, and version pin are validated by the installer's --check
# (the single owner of the pinned version).
echo "==> Verifying the Avatar SDK at $avatar_sdk_root"
"$script_dir/install_avatar_sdk.sh" --check "$avatar_sdk_root"

cmake_bin="${CMAKE:-cmake}"
command -v "$cmake_bin" >/dev/null 2>&1 || die "CMake was not found: $cmake_bin"

echo "==> Configuring Sharpa Avatar plugin"
"$cmake_bin" -B "$build_dir" -S "$isaac_root" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_PLUGINS=ON \
  -DBUILD_PLUGIN_SHARPA_AVATAR=ON \
  -DAVATAR_SDK_ROOT="$avatar_sdk_root"

echo "==> Building Sharpa Avatar plugin"
"$cmake_bin" --build "$build_dir" \
  --target avatar_hand_plugin avatar_hand_tracker_printer \
  --parallel

install_prefix="$(awk -F= '/^CMAKE_INSTALL_PREFIX:PATH=/{print $2; exit}' "$build_dir/CMakeCache.txt")"
[[ -n "$install_prefix" ]] || die "Could not read CMAKE_INSTALL_PREFIX from $build_dir/CMakeCache.txt."

echo "==> Installing Sharpa Avatar plugin"
"$cmake_bin" --install "$build_dir" --component avatar

echo "==> Installed: $install_prefix/plugins/sharpa_avatar/avatar_hand_plugin"
