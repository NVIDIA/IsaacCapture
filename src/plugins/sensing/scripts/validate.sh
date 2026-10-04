#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Stream both sensors with the vendor's nvsipl_camera. This is the ground truth
# for "is the rig alive": same SIPL API as any other client, so if it cannot
# stream, nothing else will either.
#
# Runs unchanged on the host and inside a container -- in a container it is what
# proves the device access, the driver mounts and the group membership are all
# right, before any of our code is involved.
#
# Usage:
#   ./validate.sh [options] [-- extra nvsipl_camera args]
#
# Options:
#   -t, --seconds N   stream for N seconds (default 10; 0 runs until 'q')
#   -q, --query       only ask the config what exists; needs no hardware
#   -h, --help        this text

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "$SCRIPT_DIR/common.sh"

SECONDS_ARG=10
QUERY_ONLY=0
EXTRA=()

usage() { sed -n '4,19p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; }

while (( $# )); do
    case "$1" in
        -t|--seconds) SECONDS_ARG="$2"; shift 2 ;;
        -q|--query)   QUERY_ONLY=1; shift ;;
        -h|--help)    usage; exit 0 ;;
        --)           shift; EXTRA=("$@"); break ;;
        *) die "unknown argument: $1" "Run $0 --help" ;;
    esac
done

have nvsipl_camera || die "nvsipl_camera is not on PATH." \
    "On the host, run $SCRIPT_DIR/setup.sh.
         In a container, the host's /usr/sbin is not visible -- use
         $SCRIPT_DIR/run_docker_example.sh, which mounts the package."

PKG_DIR="$(find_sensing_pkg)"
[[ -n "$PKG_DIR" ]] || die "vendor package not found." "Run $SCRIPT_DIR/setup.sh first."

# nvsipl_camera resolves -t against the working directory, and the config lives
# in the package.
cd "$PKG_DIR"
[[ -f "$SENSING_CONFIG_JSON_REL" ]] || die \
    "$SENSING_CONFIG_JSON_REL missing from $PKG_DIR." \
    "The published package has no $SENSING_PLATFORM_CONFIG config;
         $SCRIPT_DIR/setup.sh drops the vendored copy in."

if [[ "$QUERY_ONLY" -eq 1 ]]; then
    step "Querying $SENSING_PLATFORM_CONFIG (no hardware needed)"
    exec nvsipl_query -t "$SENSING_CONFIG_JSON_REL" -c "$SENSING_PLATFORM_CONFIG"
fi

# -Z: these modules carry no auth keys. --enable-camera-hal: the legacy DevBlk
# path does not support this carrier.
ARGS=(-t "$SENSING_CONFIG_JSON_REL" -c "$SENSING_PLATFORM_CONFIG"
      -m "$SENSING_LINK_MASKS" --enable-camera-hal -s -Z)
[[ "$SECONDS_ARG" -eq 0 ]] || ARGS+=(-r "$SECONDS_ARG")

step "Streaming $SENSING_PLATFORM_CONFIG"
info "$PKG_DIR"
warn "SIPL is exclusive: stop any other client first."
nvsipl_camera "${ARGS[@]}" "${EXTRA[@]}"
ok "streamed without error"
