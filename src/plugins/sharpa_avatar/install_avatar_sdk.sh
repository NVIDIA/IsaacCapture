#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install the official Sharpa Avatar SDK package.

set -euo pipefail

# Match the production host application's repository. APT verifies its signed
# metadata, and this installer additionally pins the key, channel, and exact version.
apt_base_url="http://118.196.115.252:8081/repository"
key_url="$apt_base_url/raw-releases/gpg-keys/apt-releases.gpg"
expected_fingerprints=$'80D634617D407A87CF54136D1594113827B5B686\nF9A50A81FE8797F953DB24E1938548788D899BCC'
keyring="/etc/apt/keyrings/sharpa-avatar-sdk.gpg"
source_list="/etc/apt/sources.list.d/sharpa-avatar-sdk.list"
# Pinned production SDK package version for reproducible installs.
production_version="1.7.3-17"
# Default host install path of the SDK package (fixed by the dpkg layout).
sdk_default_root="/opt/avatar-sdk"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Required command '$1' was not found. Install '$2' first."
}

# Read-only SDK tree validation shared by --check and the post-install self-check.
# The default host root is package-managed and held to the full pin (layout,
# BUILD_TYPE=Production, dpkg package version). Any other root is an explicit
# operator choice: the layout must be complete, a non-production build only
# warns, and the dpkg pin does not apply.
check_avatar_sdk() {
  local root="${1%/}"
  [[ -n "$root" ]] || die "SDK root must not be empty."

  local header="$root/include/avatar_sdk/AvatarSDK.h"
  local library="$root/lib/libavatar_sdk.so"
  local config="$root/share/sdk_config.json"
  local version_file="$root/share/Version"
  for required in "$header" "$library" "$config" "$version_file"; do
    [[ -f "$required" ]] || die "Avatar SDK tree is incomplete: missing $required."
  done

  local build_version build_type
  build_version="$(awk -F= '$1 == "VERSION" { print $2 }' "$version_file")"
  build_type="$(awk -F= '$1 == "BUILD_TYPE" { print $2 }' "$version_file")"

  if [[ "$root" == "$sdk_default_root" ]]; then
    [[ -n "$build_version" && "$build_type" == "Production" ]] \
      || die "Expected a production Avatar SDK, found ${build_version:-unknown} ${build_type:-unknown}."
    local package_version
    package_version="$(dpkg-query -W -f='${Version}' avatar-sdk 2>/dev/null || true)"
    [[ "$package_version" == "$production_version" ]] \
      || die "Expected avatar-sdk $production_version, but ${package_version:-none} is installed."
  elif [[ "$build_type" != "Production" ]]; then
    echo "WARNING: SDK root $root is not a production build (${build_version:-unknown} ${build_type:-unknown}); continuing because the root was chosen explicitly." >&2
  fi

  echo "Avatar SDK check passed: $root ${build_version:-unknown} (${build_type:-unknown})"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: $0"
  echo "Installs the pinned production avatar-sdk ($production_version) from Sharpa's signed APT repository."
  echo "       $0 --check [sdk-root]"
  echo "Validates an installed SDK tree without changing the system (default root: $sdk_default_root)."
  exit 0
fi
if [[ "${1:-}" == "--check" ]]; then
  [[ $# -le 2 ]] || die "Unknown argument: ${3:-}"
  require_command awk gawk
  check_avatar_sdk "${2:-${AVATAR_SDK_ROOT:-$sdk_default_root}}"
  exit 0
fi
[[ $# -eq 0 ]] || die "Unknown argument: $1"

require_command apt-get apt
require_command awk gawk
require_command curl curl
require_command dpkg-query dpkg
require_command gpg gnupg
require_command grep grep
require_command install coreutils

for non_production_package in avatar-sdk-dev avatar-sdk-beta; do
  if dpkg-query -W -f='${db:Status-Abbrev}' "$non_production_package" 2>/dev/null | grep -q '^ii'; then
    die "$non_production_package is installed. Remove it explicitly before installing the production SDK."
  fi
done

sudo_cmd=()
if [[ "$EUID" -ne 0 ]]; then
  require_command sudo sudo
  echo "This installer needs sudo to configure APT and install avatar-sdk."
  sudo -v || die "sudo authentication failed."
  sudo_cmd=(sudo)
fi

key_download="$(mktemp)"
key_dearmored="$(mktemp)"
trap 'rm -f "$key_download" "$key_dearmored"' EXIT

echo "==> Downloading Sharpa Avatar SDK signing key"
curl -fsSL "$key_url" -o "$key_download"
actual_fingerprints="$(
  gpg --show-keys --with-colons "$key_download" 2>/dev/null \
    | awk -F: '$1 == "fpr" { print $10 }' \
    | LC_ALL=C sort -u
)"
[[ "$actual_fingerprints" == "$expected_fingerprints" ]] \
  || die "Signing-key fingerprint set did not match the pinned Sharpa key."
gpg --batch --yes --dearmor --output "$key_dearmored" "$key_download"

echo "==> Configuring Sharpa Avatar SDK APT repository"
"${sudo_cmd[@]}" install -d -m 0755 /etc/apt/keyrings
"${sudo_cmd[@]}" install -m 0644 "$key_dearmored" "$keyring"
printf 'deb [signed-by=%s] %s/apt-releases/ stable main\n' "$keyring" "$apt_base_url" \
  | "${sudo_cmd[@]}" tee "$source_list" >/dev/null

apt_source_options=(
  -o "Dir::Etc::sourcelist=$source_list"
  -o Dir::Etc::sourceparts="-"
)

echo "==> Installing avatar-sdk $production_version"
"${sudo_cmd[@]}" apt-get update "${apt_source_options[@]}" -o APT::Get::List-Cleanup="0"

# --allow-downgrades so re-running always converges on the pinned version even
# if a newer avatar-sdk is already installed.
"${sudo_cmd[@]}" apt-get install -y --allow-downgrades "${apt_source_options[@]}" "avatar-sdk=$production_version"

check_avatar_sdk "$sdk_default_root"

echo "==> Avatar SDK production installed at $sdk_default_root"
