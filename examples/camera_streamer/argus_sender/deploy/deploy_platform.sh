#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Deploy only the clock helper needed by the camera-streamer service setup.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
host="${1:?An SSH target is required}"
shift
check=0
for arg in "$@"; do
    case "$arg" in
        --check) check=1 ;;
        --jetson-clocks|--verbose) ;;
        *) echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done
ssh_args=(-o BatchMode=yes -o ConnectTimeout=10)
if [[ "$check" -eq 1 ]]; then
    ssh "${ssh_args[@]}" "$host" 'sudo -n bash -s' <<'REMOTE'
set -euo pipefail
command -v jetson_clocks >/dev/null
sudo -n jetson_clocks --show
for name in vi isp nvcsi vic emc; do
    base="/sys/kernel/debug/bpmp/debug/clk/$name"
    [[ -d "$base" ]] || continue
    [[ "$(cat "$base/mrq_rate_locked")" == 1 ]]
    [[ "$(cat "$base/rate")" == "$(cat "$base/max_rate")" ]]
done
REMOTE
    exit 0
fi
ssh "${ssh_args[@]}" "$host" 'sudo -n mkdir -p /usr/local/lib/camera'
ssh "${ssh_args[@]}" "$host" 'sudo -n tee /usr/local/lib/camera/lock_jetson_camera_clocks.sh >/dev/null' \
    < "$here/lock_jetson_camera_clocks.sh"
ssh "${ssh_args[@]}" "$host" 'sudo -n bash /usr/local/lib/camera/lock_jetson_camera_clocks.sh'
