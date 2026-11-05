#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Orin side of the SIPL fast demo over plain TCP: make the FIFO, connect to the laptop's listener, then run the SIPL
# producer. Start run_fast_demo.sh on the laptop first. Needs only bash; the connection is bash's /dev/tcp.
# Ctrl+C stops the producer and the sender.
#
#   LAPTOP_IP=<laptop address> ./run_fast_demo_orin.sh
set -euo pipefail

LAPTOP_IP="${LAPTOP_IP:?set LAPTOP_IP to the laptop's address}"
PORT="${PORT:-5600}"
FIFO="${FIFO:-/dev/shm/sensing-demo/sensor1.fifo}"
PRODUCER="${PRODUCER:-${HOME}/start_producer.sh}"

SENDER_PID=""

# The producer is stopped by Ctrl+C itself; this ends the sender if it is still waiting on the FIFO.
cleanup() {
    trap - EXIT INT TERM
    echo "[orin] stopping"
    [[ -n "${SENDER_PID}" ]] && kill "${SENDER_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

die() { echo "[orin] ERROR: $*" >&2; exit 1; }

[[ -x "${PRODUCER}" ]] || die "producer script not found or not executable: ${PRODUCER}"

mkdir -p "$(dirname "${FIFO}")"
[[ -p "${FIFO}" ]] || mkfifo "${FIFO}"

# Connecting first and reading the FIFO second: the redirect opens the TCP connection, then cat blocks on the FIFO
# until the producer opens it. A refused connection means the laptop listener is not up yet.
echo "[orin] connecting to ${LAPTOP_IP}:${PORT}"
bash -c "cat '${FIFO}' > /dev/tcp/${LAPTOP_IP}/${PORT}" &
SENDER_PID=$!
sleep 1
kill -0 "${SENDER_PID}" 2>/dev/null || die "could not connect to ${LAPTOP_IP}:${PORT}; start run_fast_demo.sh on the laptop first"

echo "[orin] starting the producer"
"${PRODUCER}"
