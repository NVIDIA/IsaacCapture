#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build and start a container that can reach the cameras, then drop you at a
# shell at the repository root. Run the same nvsipl_camera command as on the host.
#
# Run this on the HOST: it reads the device nodes to derive what to mount, and
# passes host paths to `docker run`.
#
# The image side is Dockerfile.example plus image_setup.sh. The run side is the
# part no Dockerfile can express -- bind mounts, --device, and the NVIDIA
# runtime capabilities that make the BSP libraries visible.
#
# Usage:
#   ./setup_container.sh                 # build, then an interactive shell
#   ./setup_container.sh -- COMMAND  # build, run that, exit
#
# Options:
#   --build-only    build the image and stop
#   --no-build      reuse an existing image
#   --print-args    print the `docker run` flags for your own container and stop
#   --json          the same flags as devcontainer.json runArgs elements
#   --print-gids    print "GID:NAME ..." for image_setup.sh and stop
#   --tag NAME      image tag (default: sensing-example)
#   -h, --help      this text

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../scripts/common.sh
source "$SCRIPT_DIR/../scripts/common.sh"

PLUGIN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PLUGIN_DIR/../../.." && pwd)"
TAG=sensing-example
MODE=run
BUILD=1
CMD=()

usage() { sed -n '4,27p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; }

while (( $# )); do
    case "$1" in
        --build-only) MODE=build; shift ;;
        --no-build)   BUILD=0; shift ;;
        --print-args) MODE=args; shift ;;
        --json)       MODE=json; shift ;;
        --print-gids) MODE=gids; shift ;;
        --tag)        [[ $# -ge 2 && -n "$2" ]] || die "--tag requires an image name"; TAG="$2"; shift 2 ;;
        -h|--help)    usage; exit 0 ;;
        --)           shift; CMD=("$@"); break ;;
        *) die "unknown argument: $1" "Run $0 --help" ;;
    esac
done

in_container && die "run this on the HOST." \
    "It reads host device nodes and hands host paths to docker run."

# GIDs owning the nodes SIPL opens. Only group-gated nodes matter: a
# world-readable one needs no supplementary group. Deduplicated, in first-seen
# order so the output is stable across runs.
node_gids() {
    local entry path gid seen=""
    while IFS= read -r entry; do
        path="${entry%%:*}"
        [[ -e "$path" ]] || continue
        # o+r set means any uid can open it; no --group-add needed.
        [[ "$(stat -c %A "$path")" == ???????r?? ]] && continue
        gid="$(stat -c %g "$path" 2>/dev/null)" || continue
        [[ " $seen " == *" $gid "* ]] && continue
        seen="$seen $gid"
        printf '%s\n' "$gid"
    done < <(sensing_required_nodes)
}

mapfile -t GIDS < <(node_gids)

if [[ "$MODE" == gids ]]; then
    out=()
    for gid in "${GIDS[@]}"; do
        out+=("$gid:$(getent group "$gid" | cut -d: -f1 || echo "grp$gid")")
    done
    printf '%s\n' "${out[*]}"
    exit 0
fi

ARGS=()
# Vendor additions are absent from the NVIDIA runtime's BSP file list.
for dir in "$SIPL_DRV_DIR" "$SIPL_NITO_DIR"; do
    [[ -d "$dir" ]] || die "$dir missing on this host" "Run scripts/setup.sh first."
    ARGS+=(-v "$dir:$dir:ro")
done
for file in "$SIPL_SBIN_DIR/nvsipl_camera" "$SIPL_SBIN_DIR/nvsipl_query" "$SIPL_CAMERAHAL" /etc/nv_tegra_release; do
    [[ -r "$file" ]] || die "$file missing on this host" "Run scripts/setup.sh first."
    ARGS+=(-v "$file:$file:ro")
done

ARGS+=(--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all)
# Device cgroup access is required in addition to Unix group permissions.
ARGS+=(--privileged -v /dev:/dev)
# Keep the config's repository-relative path identical on host and container.
ARGS+=(-v "$PLUGIN_DIR:$PLUGIN_DIR:ro" -w "$REPO_DIR")

# Xauthority cookies are keyed by the host name for local X11 connections.
if [[ "$MODE" != build ]]; then
    [[ "${DISPLAY:-}" =~ ^:[0-9]+(\.[0-9]+)?$ ]] || die "a local desktop DISPLAY is required" \
        "Run setup_container.sh from a terminal on the Jetson desktop."
    display_number="${DISPLAY#:}"; display_number="${display_number%%.*}"
    [[ -S "/tmp/.X11-unix/X$display_number" ]] || die "no X11 socket for $DISPLAY"
    auth_file="${XAUTHORITY:-$HOME/.Xauthority}"
    # SSH does not inherit GDM's XAUTHORITY, and GDM does not use ~/.Xauthority.
    if [[ -z "${XAUTHORITY:-}" && ! -r "$auth_file" ]]; then
        auth_file="/run/user/$(id -u)/gdm/Xauthority"
    fi
    [[ -r "$auth_file" ]] || die "Xauthority file not readable: $auth_file" \
        "Run from the desktop session, with its DISPLAY and XAUTHORITY."
    ARGS+=(--hostname "$(hostname)" -e "DISPLAY=$DISPLAY" -e XAUTHORITY=/tmp/sensing.Xauthority
           -v /tmp/.X11-unix:/tmp/.X11-unix:ro
           -v "$auth_file:/tmp/sensing.Xauthority:ro")
fi

if [[ "$MODE" == args ]]; then
    printf '%q ' "${ARGS[@]}"; printf '\n'; exit 0
elif [[ "$MODE" == json ]]; then
    python3 -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "${ARGS[@]}"; exit 0
fi

have docker || die "docker is not installed on this host."
"$PLUGIN_DIR/scripts/verify.sh" || die "host setup is incomplete" "Complete scripts/setup.sh first."

if [[ "$BUILD" -eq 1 ]]; then
    step "Building $TAG"
    docker build --network=host -t "$TAG" \
        --build-arg "SENSING_GIDS=$("$0" --print-gids)" \
        -f "$SCRIPT_DIR/Dockerfile.example" "$SCRIPT_DIR" \
        || die "build failed"
    ok "$TAG"
fi
[[ "$MODE" != build ]] || exit 0

step "Starting $TAG"
if [[ "${#CMD[@]}" -eq 0 ]]; then
    hint 'nvsipl_camera -t src/plugins/sensing/configs/shw5g.json -c SHW5G_2 -m "0x0000 0x1100" --enable-camera-hal -s -Z --egl-display'
    CMD=(bash)
    ARGS+=(-it)
fi
exec docker run --rm "${ARGS[@]}" "$TAG" "${CMD[@]}"
