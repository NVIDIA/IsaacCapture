#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Lock the Jetson clocks the Argus capture path depends on.
#
# This matches the Sensing SHW5G/S56 latency-test condition:
#   nvpmodel MAXN (ceiling) + jetson_clocks (pin at ceiling)
#   + VI/ISP/NVCSI/VIC/EMC mrq_rate_locked + max_rate.
#
# The sender service runs this helper before capture. It can also be run
# manually with sudo. Clock pins are runtime state and do not survive a reboot.

set -euo pipefail

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    exec sudo -E "$0" "$@"
fi

# ── power mode ────────────────────────────────────────────────────────
# nvpmodel sets the frequency CEILING that jetson_clocks then pins against, and
# unlike jetson_clocks it IS persistent. If it has drifted off MAXN, pinning
# still "succeeds" but clamps everything to a reduced ceiling — the failure mode
# is a robot that looks healthy and quietly runs slow, so say so explicitly.
#
# MAXN's mode id is not guaranteed to be 0 on every board/BSP, so resolve it
# from nvpmodel.conf rather than hardcoding.
maxn_id() {
    sed -n 's/^[[:space:]]*<[[:space:]]*POWER_MODEL[[:space:]]\+ID=\([0-9]\+\)[[:space:]]\+NAME=MAXN.*/\1/p' \
        /etc/nvpmodel.conf 2>/dev/null | head -1
}

if command -v nvpmodel >/dev/null 2>&1; then
    cur_name="$(nvpmodel -q 2>/dev/null | sed -n 's/^NV Power Mode: *//p' | head -1 || true)"
    cur_id="$(nvpmodel -q 2>/dev/null | sed -n '/^NV Power Mode/{n;s/^[[:space:]]*\([0-9]\+\).*/\1/p;}' | head -1 || true)"
    want_id="$(maxn_id)"
    printf "nvpmodel: mode=%s id=%s (MAXN id=%s)\n" \
        "${cur_name:-?}" "${cur_id:-?}" "${want_id:-?}"
    if [[ "$cur_name" != MAXN* ]]; then
        echo "WARNING: power mode is '${cur_name:-unknown}', not MAXN — clocks will pin" >&2
        echo "         to a REDUCED ceiling. Fix persistently with:" >&2
        echo "           sudo nvpmodel -m ${want_id:-0}   # then reboot, or re-run this script" >&2
    fi
else
    echo "nvpmodel not found; skipping power-mode check" >&2
fi

# ── pin at the ceiling ────────────────────────────────────────────────
# Runtime sysfs only: per-CPU min frequency raised to max, CPU idle states
# disabled, GPU rail and EMC pinned. Undone by a reboot or any `nvpmodel -m`.
if command -v jetson_clocks >/dev/null 2>&1; then
    jetson_clocks
    echo "jetson_clocks applied"
else
    echo "jetson_clocks not found; skipping CPU/GPU/EMC pinning" >&2
fi

# ── camera-specific clocks ────────────────────────────────────────────
lock_clk() {
    local name="$1"
    local base="/sys/kernel/debug/bpmp/debug/clk/$name"
    if [[ ! -d "$base" ]]; then
        echo "missing $base; skipping $name" >&2
        return 0
    fi

    echo 1 > "$base/mrq_rate_locked"
    cat "$base/max_rate" > "$base/rate"

    printf "%-5s locked=%s rate=%s max=%s\n" \
        "$name" \
        "$(cat "$base/mrq_rate_locked")" \
        "$(cat "$base/rate")" \
        "$(cat "$base/max_rate")"
}

lock_clk vi
lock_clk isp
lock_clk nvcsi
lock_clk vic
lock_clk emc
