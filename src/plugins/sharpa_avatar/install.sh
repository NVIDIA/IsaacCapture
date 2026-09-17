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

version_file="/opt/avatar-sdk/share/Version"
[[ -f "$version_file" ]] || die "Installed SDK is missing $version_file."
production_version="$(awk -F= '/^production_version=/{ gsub(/"/, "", $2); print $2; exit }' \
  "$script_dir/install_avatar_sdk.sh")"
[[ -n "$production_version" ]] || die "Could not read production_version from install_avatar_sdk.sh."
sdk_version="$(awk -F= '$1 == "VERSION" { print $2 }' "$version_file")"
sdk_build_type="$(awk -F= '$1 == "BUILD_TYPE" { print $2 }' "$version_file")"
installed_package_version="$(dpkg-query -W -f='${Version}' avatar-sdk 2>/dev/null || true)"
echo "==> Found Avatar SDK ${sdk_version:-unknown} (${sdk_build_type:-unknown})"
[[ -n "$sdk_version" && "$sdk_build_type" == "Production" ]] \
  || die "Expected a production Avatar SDK, found ${sdk_version:-unknown} ${sdk_build_type:-unknown}. Run install_avatar_sdk.sh."
[[ "$installed_package_version" == "$production_version" ]] \
  || die "Expected avatar-sdk $production_version, but ${installed_package_version:-none} is installed. Run install_avatar_sdk.sh."

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
[[ -n "$install_prefix" ]] || die "Could not read CMAKE_INSTALL_PREFIX from $build_dir/CMakeCache.txt."

echo "==> Installing Sharpa Avatar plugin"
"$cmake_bin" --install "$build_dir" --component avatar

echo "==> Installed: $install_prefix/plugins/sharpa_avatar/avatar_hand_plugin"
