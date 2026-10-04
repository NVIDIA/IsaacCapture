#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Image-side setup for SIPL capture: put the container user in the groups that
# own the host's camera device nodes. Run as root from a Dockerfile RUN.
#
# This is the whole of it -- no --group-add needed on `docker run`. Docker
# resolves supplementary groups for the container user from the image's own
# /etc/group, so membership set here reaches PID 1 and `docker exec` alike.
# The reverse is not true: --group-add only reaches PID 1, because `docker exec`
# ignores HostConfig.GroupAdd.
#
# Usage:
#   image_setup.sh USER GID:NAME [GID:NAME ...]
#
# Get the gids for a specific host with: ./run_docker_example.sh --print-gids

set -euo pipefail

[[ $# -ge 2 ]] || { echo "usage: $0 USER GID:NAME [GID:NAME ...]" >&2; exit 2; }
user="$1"; shift

id "$user" >/dev/null 2>&1 || { echo "$0: no such user: $user" >&2; exit 1; }

for spec in "$@"; do
    gid="${spec%%:*}"; name="${spec##*:}"
    [[ "$gid" =~ ^[0-9]+$ ]] || { echo "$0: bad spec: $spec" >&2; exit 2; }
    # Adopt whatever group already holds the gid rather than assuming the name
    # is free -- base images vary, and the kernel checks the number anyway.
    if getent group "$gid" >/dev/null; then
        name="$(getent group "$gid" | cut -d: -f1)"
    else
        groupadd --gid "$gid" "$name"
    fi
    usermod -aG "$name" "$user"
    echo "  $user -> $name($gid)"
done
