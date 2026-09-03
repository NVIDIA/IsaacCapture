#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install the Sharpa Avatar glove plugin.
#
# Two steps: (1) stage the Avatar SDK the plugin links against (from
# AVATAR_SDK_ROOT, a path argument, or /opt/avatar-sdk), (2) configure, build,
# and install the plugin target. The SDK is not committed; re-runs reuse the
# extracted vendor copy.
#
# Usage:
#   ./install.sh [--build-dir DIR] [isaac-teleop-root] [avatar-sdk-root]
#
# Env:
#   AVATAR_SDK_ROOT  skip discovery and use this installed SDK tree
#                    (must contain include/avatar_sdk/AvatarSDK.h + lib/).
#   CMAKE            cmake executable (must be 3.24+; Ubuntu 22.04 apt is 3.22).

set -euo pipefail

build_dir=""
isaac_root=""
sdk_root=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      build_dir="${2:?--build-dir requires a non-empty path}"
      shift 2
      ;;
    -h | --help)
      cat <<EOF
Usage: $0 [--build-dir DIR] [isaac-teleop-root] [avatar-sdk-root]

Stages the Avatar SDK (gitignored), then builds and installs the glove plugin.

Options:
  --build-dir DIR   CMake build directory (default: <isaac-root>/build).
  -h, --help        Show this help.
EOF
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
    *)
      if [[ -z "$isaac_root" ]]; then
        isaac_root="$1"
      elif [[ -z "$sdk_root" ]]; then
        sdk_root="$1"
      else
        echo "Too many arguments" >&2
        exit 1
      fi
      shift
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_ROOT="${isaac_root:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
BUILD_DIR="${build_dir:-$ISAAC_ROOT/build}"
if [[ "$BUILD_DIR" != /* ]]; then
  BUILD_DIR="$PWD/$BUILD_DIR"
fi

if [[ -z "$sdk_root" ]]; then
  sdk_root="${AVATAR_SDK_ROOT:-}"
fi

cmake_major_minor() {
  "$1" --version 2>/dev/null | awk 'NR==1 {
    for (i = 1; i <= NF; i++) {
      if ($i ~ /^[0-9]+\.[0-9]+/) {
        split($i, p, ".")
        printf "%d %d", p[1], p[2]
        exit
      }
    }
  }'
}

cmake_at_least_3_24() {
  local ver
  ver="$(cmake_major_minor "$1")"
  [[ -n "$ver" ]] || return 1
  local major minor
  read -r major minor <<<"$ver"
  ((major > 3)) || ((major == 3 && minor >= 24))
}

resolve_cmake() {
  local candidate
  for candidate in \
    "${CMAKE:-}" \
    "$ISAAC_ROOT/.venv/bin/cmake" \
    "$(command -v cmake || true)"
  do
    if [[ -n "$candidate" && -x "$candidate" ]] && cmake_at_least_3_24 "$candidate"; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

VENDOR_SCRIPT="$SCRIPT_DIR/vendor_avatar_sdk.sh"
VENDOR_ROOT="$SCRIPT_DIR/vendor/avatar-sdk"

echo "==> Staging Avatar SDK"
if [[ -x "$VENDOR_SCRIPT" ]]; then
  if [[ -n "$sdk_root" ]]; then
    "$VENDOR_SCRIPT" "$sdk_root"
  else
    "$VENDOR_SCRIPT"
  fi
else
  echo "vendor_avatar_sdk.sh not found next to install.sh" >&2
  exit 1
fi

[[ -f "$VENDOR_ROOT/include/avatar_sdk/AvatarSDK.h" ]] || {
  echo "Avatar SDK not staged at $VENDOR_ROOT" >&2
  echo "Install the SDK (typically /opt/avatar-sdk) or set AVATAR_SDK_ROOT." >&2
  exit 1
}

CMAKE_BIN="$(resolve_cmake || true)"
if [[ -z "$CMAKE_BIN" ]]; then
  echo "ERROR: Isaac Teleop needs CMake 3.24+ (this host has $(cmake --version 2>/dev/null | head -1 || echo 'no cmake'))." >&2
  echo "Ubuntu 22.04 apt cmake is 3.22. From the Isaac Teleop venv:" >&2
  echo "  source \"$ISAAC_ROOT/.venv/bin/activate\"" >&2
  echo "  pip install 'cmake>=3.24'" >&2
  echo "Then re-run:  $0" >&2
  exit 1
fi

echo "==> Building avatar_hand_plugin (cmake=$CMAKE_BIN)"
"$CMAKE_BIN" -B "$BUILD_DIR" -S "$ISAAC_ROOT" -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_PLUGINS=ON \
  -DBUILD_PLUGIN_SHARPA_AVATAR=ON \
  -DAVATAR_SDK_ROOT="$VENDOR_ROOT"
"$CMAKE_BIN" --build "$BUILD_DIR" --target avatar_hand_plugin avatar_hand_tracker_printer --parallel
"$CMAKE_BIN" --install "$BUILD_DIR" --component avatar

install_prefix="$(awk '/^CMAKE_INSTALL_PREFIX:PATH=/{sub(/^[^=]*=/, ""); print; exit}' "$BUILD_DIR/CMakeCache.txt")"
echo "==> Done: $install_prefix/plugins/sharpa_avatar/avatar_hand_plugin"
