#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install the SpaceMouse udev rule on the HOST machine, so /dev/hidraw* nodes of SpaceMouse devices
# come up readable by the spacemouse plugin. udev does not run inside containers: install on the
# host, then unplug and replug the SpaceMouse.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RULES_SRC="$SCRIPT_DIR/70-spacemouse.rules"
RULES_DST="/etc/udev/rules.d/70-spacemouse.rules"

die() {
    echo "Error: $1" >&2
    if [[ "${2:-}" != "" ]]; then
        echo "Action: $2" >&2
    fi
    exit 1
}

if [[ ! -f "$RULES_SRC" ]]; then
    die "rules file not found at $RULES_SRC" "Run this script from a complete IsaacTeleop checkout or install tree."
fi

if [[ -f /.dockerenv ]] || grep -qE '(docker|containerd|kubepods)' /proc/1/cgroup 2>/dev/null; then
    die "this script must be run on the HOST, not inside a container." \
        "Open a host terminal and run: $SCRIPT_DIR/install_udev_rules.sh"
fi

command -v udevadm >/dev/null 2>&1 \
    || die "required command 'udevadm' was not found." "Install it first, for example: sudo apt-get install -y udev"

if ! sudo -n true 2>/dev/null; then
    echo "This script needs sudo to write to /etc/udev/rules.d/."
    sudo -v || die "sudo authentication failed." "Re-run from a user with sudo privileges."
fi

echo "Installing SpaceMouse udev rule to $RULES_DST..."
sudo install -m 0644 "$RULES_SRC" "$RULES_DST"
sudo cmp -s "$RULES_SRC" "$RULES_DST" || die "installed rule differs from $RULES_SRC" "Inspect $RULES_DST and re-run."

echo "Reloading udev rules..."
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=hidraw --action=change
sudo udevadm settle

echo "Checking SpaceMouse hidraw permissions..."
found=0
for node in /dev/hidraw*; do
    [[ -e "$node" ]] || continue
    vendor="$(udevadm info -q property -n "$node" 2>/dev/null | sed -n 's/^ID_VENDOR_ID=//p')"
    model="$(udevadm info -q property -n "$node" 2>/dev/null | sed -n 's/^ID_MODEL_ID=//p')"
    if [[ "$vendor" == "256f" || ( "$vendor" == "046d" && ( "$model" == "c626" || "$model" == "c628" ) ) ]]; then
        found=1
        if [[ -r "$node" ]]; then
            echo "  OK: $node is readable."
        else
            echo "  Warning: $node is not readable yet. Unplug and replug the SpaceMouse."
        fi
    fi
done
if [[ "$found" -eq 0 ]]; then
    echo "  No SpaceMouse connected. Plug it in; the rule applies when it appears."
fi
echo "Done."
