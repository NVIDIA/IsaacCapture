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

[[ -f /opt/avatar-sdk/include/avatar_sdk/AvatarSDK.h ]] || {
  if [[ ! -e /opt/avatar-sdk ]]; then
    echo "Avatar SDK is not installed; installing the official package."
    "$script_dir/install_avatar_sdk.sh"
  else
    die "Avatar SDK installation under /opt/avatar-sdk is incomplete. Re-run install_avatar_sdk.sh."
  fi
}

[[ -f /opt/avatar-sdk/include/avatar_sdk/AvatarSDK.h ]] \
  || die "Avatar SDK header not found after installation."
[[ -f /opt/avatar-sdk/lib/libavatar_sdk.so ]] \
  || die "Avatar SDK library libavatar_sdk.so not found under /opt/avatar-sdk/lib."
[[ -f /opt/avatar-sdk/share/sdk_config.json ]] \
  || die "Avatar SDK configuration not found at /opt/avatar-sdk/share/sdk_config.json."

if [[ -f /opt/avatar-sdk/share/Version ]]; then
  sdk_version="$(awk -F= '$1 == "VERSION" { print $2 }' /opt/avatar-sdk/share/Version)"
  sdk_build_type="$(awk -F= '$1 == "BUILD_TYPE" { print $2 }' /opt/avatar-sdk/share/Version)"
  echo "==> Found Avatar SDK ${sdk_version:-unknown} (${sdk_build_type:-unknown})"
  if [[ "$sdk_build_type" != "Production" ]]; then
    echo "WARNING: using an existing non-production SDK; install_avatar_sdk.sh only installs the production channel." >&2
  fi
fi

cmake_bin="${CMAKE:-cmake}"
command -v "$cmake_bin" >/dev/null 2>&1 || die "CMake was not found: $cmake_bin"

echo "==> Configuring Sharpa Avatar plugin"
"$cmake_bin" -B "$build_dir" -S "$isaac_root" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_PLUGINS=ON \
  -DBUILD_PLUGIN_SHARPA_AVATAR=ON

echo "==> Building Sharpa Avatar plugin"
"$cmake_bin" --build "$build_dir" \
  --target avatar_hand_plugin avatar_hand_tracker_printer \
  --parallel

install_prefix="$(awk -F= '/^CMAKE_INSTALL_PREFIX:PATH=/{print $2; exit}' "$build_dir/CMakeCache.txt")"
legacy_lib_dir="$install_prefix/lib"
if [[ -e "$legacy_lib_dir/libavatar_sdk.so" ]]; then
  echo "==> Removing SDK libraries installed by the legacy vendor flow"
  shopt -s nullglob
  legacy_sdk_files=(
    "$legacy_lib_dir"/libavatar_sdk*.so*
    "$legacy_lib_dir"/libcasadi*.so*
    "$legacy_lib_dir"/libipopt*.so*
    "$legacy_lib_dir"/libsipopt*.so*
    "$legacy_lib_dir"/libcoinmumps*.so*
    "$legacy_lib_dir"/libcoinmetis*.so*
  )
  shopt -u nullglob
  rm -f "${legacy_sdk_files[@]}"
fi

echo "==> Installing Sharpa Avatar plugin"
"$cmake_bin" --install "$build_dir" --component avatar

echo "==> Installed: $install_prefix/plugins/sharpa_avatar/avatar_hand_plugin"
