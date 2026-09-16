#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install the Sharpa Avatar glove udev rules on the host.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rule_name="70-sharpa-avatar-glove.rules"
rule_source="$script_dir/$rule_name"
rule_destination="/etc/udev/rules.d/$rule_name"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Required command '$1' was not found. Install '$2' first."
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: $0"
  echo "Installs and reloads the Sharpa Avatar glove udev rules on the host."
  exit 0
fi
[[ $# -eq 0 ]] || die "Unknown argument: $1"

[[ -f "$rule_source" ]] || die "udev rules file not found: $rule_source"
if [[ -f /.dockerenv ]] || grep -qE '(docker|containerd|kubepods)' /proc/1/cgroup 2>/dev/null; then
  die "Run this script on the host, not inside a container."
fi

require_command udevadm udev
require_command install coreutils

sudo_cmd=()
if [[ "$EUID" -ne 0 ]]; then
  require_command sudo sudo
  if ! sudo -n true 2>/dev/null; then
    echo "This installer needs sudo to write $rule_destination."
    sudo -v || die "sudo authentication failed."
  fi
  sudo_cmd=(sudo)
fi

echo "==> Installing Sharpa Avatar glove udev rules"
"${sudo_cmd[@]}" install -m 0644 "$rule_source" "$rule_destination"
"${sudo_cmd[@]}" cmp -s "$rule_source" "$rule_destination" || die "Installed rules differ from $rule_source."

echo "==> Reloading udev rules"
"${sudo_cmd[@]}" udevadm control --reload-rules
"${sudo_cmd[@]}" udevadm trigger --subsystem-match=usb --action=change
"${sudo_cmd[@]}" udevadm settle

echo "==> Installed $rule_destination"
echo "Unplug and reconnect the glove or dongle if it was already attached."
