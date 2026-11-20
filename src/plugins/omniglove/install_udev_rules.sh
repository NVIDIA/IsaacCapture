#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rule_source="$script_dir/60-omniglove-usb.rules"
rule_destination="/etc/udev/rules.d/60-omniglove-usb.rules"

if [[ -f /.dockerenv ]] || grep -Eq '(docker|containerd|kubepods)' /proc/1/cgroup 2>/dev/null; then
    echo "error: install udev rules on the host, not inside a container" >&2
    exit 1
fi
if [[ ! -f "$rule_source" ]]; then
    echo "error: missing $rule_source" >&2
    exit 1
fi
if ! command -v udevadm >/dev/null 2>&1; then
    echo "error: udevadm is required" >&2
    exit 1
fi

if [[ -f "$rule_destination" ]] && sudo cmp -s "$rule_source" "$rule_destination"; then
    echo "OmniGlove udev rule is already installed"
    exit 0
fi

sudo install -m 0644 "$rule_source" "$rule_destination"
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=usb --action=change
sudo udevadm settle

echo "Installed $rule_destination"
echo "Unplug and reconnect the gloves, then run applications as a normal user."
