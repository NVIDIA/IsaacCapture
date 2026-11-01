#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
binary="${project_dir}/build/argus_sender"
config="${project_dir}/configs/cameras.yaml"

if [[ ! -x "${binary}" ]]; then
    echo "error: ${binary} does not exist; run ./build.sh first." >&2
    exit 1
fi

if command -v systemctl >/dev/null 2>&1 \
    && systemctl is-active --quiet nvargus-daemon.service; then
    echo "stopping nvargus-daemon before starting single-process argus_sender"
    if ! sudo -n systemctl stop nvargus-daemon.service; then
        echo "error: could not stop nvargus-daemon; passwordless sudo is required." >&2
        exit 1
    fi
    sudo -n systemctl reset-failed nvargus-daemon.service 2>/dev/null || true
    sleep 2
fi

if command -v systemctl >/dev/null 2>&1 \
    && systemctl is-active --quiet nvargus-daemon.service; then
    echo "error: nvargus-daemon is still active; refusing competing Argus ownership." >&2
    exit 1
fi

exec "${binary}" --config "${config}" "$@"
