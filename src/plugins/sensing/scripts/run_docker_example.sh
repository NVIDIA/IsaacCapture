#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build and start a container that can reach the cameras, then drop you at a
# shell in it. Run validate.sh from there to stream; it is the same script that
# runs on the host.
#
# Run this on the HOST: it reads the device nodes to derive what to mount, and
# passes host paths to `docker run`.
#
# The image side is Dockerfile.example plus image_setup.sh. The run side is the
# part no Dockerfile can express -- bind mounts, --device, and the NVIDIA
# runtime capabilities that make the BSP libraries visible.
#
# Usage:
#   ./run_docker_example.sh                 # build, then an interactive shell
#   ./run_docker_example.sh -- validate.sh  # build, run that, exit
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
# shellcheck source=./common.sh
source "$SCRIPT_DIR/common.sh"

PLUGIN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
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
        --tag)        TAG="$2"; shift 2 ;;
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
# The NVIDIA container runtime mounts a fixed CSV file list, so it only carries
# what shipped with the BSP. The vendor install.sh ADDS files -- the
# libnvuddf_*_cameramodule_library.so drivers and the per-module .nito -- which
# appear on no CSV and are therefore invisible inside the container. Without
# them SIPL cannot instantiate the camera module and every client, NVIDIA's own
# nvsipl_camera included, fails at SetPlatformCfg with a bare "status: 10".
# Mount the directories, not the files, so a later vendor drop needs no change.
for dir in "$SIPL_DRV_DIR" "$SIPL_NITO_DIR"; do
    [[ -d "$dir" ]] || { warn "$dir missing on this host — run setup.sh"; continue; }
    ARGS+=(-v "$dir:$dir:ro")
done

# The nvidia runtime injects the BSP libraries -- libnvsipl.so among them -- only
# when the capabilities are actually requested. Without these the binary loads
# and then dies on a missing shared object, naming nothing about cameras.
ARGS+=(--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all)
# --privileged opens the device cgroup. Group membership is necessary but not
# sufficient: with the cgroup closed every node is unreadable regardless.
ARGS+=(--privileged -v /dev:/dev)
# The plugin directory, at the same path, so validate.sh finds .cache/ and the
# vendored config exactly as it does on the host.
ARGS+=(-v "$PLUGIN_DIR:$PLUGIN_DIR:ro" -w "$PLUGIN_DIR")

if [[ "$MODE" == args ]]; then
    printf '%s\n' "${ARGS[*]}"; exit 0
elif [[ "$MODE" == json ]]; then
    for arg in "${ARGS[@]}"; do printf '"%s",\n' "$arg"; done; exit 0
fi

if [[ "$BUILD" -eq 1 ]]; then
    step "Building $TAG"
    docker build --network=host -t "$TAG" \
        --build-arg "SENSING_GIDS=$("$0" --print-gids)" \
        -f "$SCRIPT_DIR/../docker/Dockerfile.example" "$SCRIPT_DIR/../docker" \
        || die "build failed"
    ok "$TAG"
fi
[[ "$MODE" != build ]] || exit 0

step "Starting $TAG"
if [[ "${#CMD[@]}" -eq 0 ]]; then
    info "run ./validate.sh to stream both sensors"
    CMD=(bash)
    ARGS+=(-it)
fi
exec docker run --rm "${ARGS[@]}" "$TAG" "${CMD[@]}"
