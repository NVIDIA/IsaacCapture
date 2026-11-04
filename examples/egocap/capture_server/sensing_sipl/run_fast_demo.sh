#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Run the SIPL fast demo from the laptop: SSH pump, Orin producer, then capture-server in the foreground.
# Ctrl+C stops all three. Defaults match FAST_DEMO.md; override with the environment variables below.
set -euo pipefail

ORIN="${ORIN:-ido@192.168.1.176}"
FIFO="${FIFO:-/dev/shm/sensing-demo/sensor1.fifo}"
SPOOL="${SPOOL:-/var/tmp/sensing/sensor1.h264}"
PRODUCER="${PRODUCER:-~/start_producer.sh}"
EGOCAP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATION="${STATION:-${EGOCAP_DIR}/tmp/my_station.yaml}"
LOG_DIR="${LOG_DIR:-${EGOCAP_DIR}/tmp/fast-demo}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8)

PUMP_PID=""
PRODUCER_PID=""

# Stop the producer first so the FIFO closes and the pump ends by itself, then the pump if it lingers.
cleanup() {
    trap - EXIT INT TERM
    echo "[demo] stopping"
    [[ -n "${PRODUCER_PID}" ]] && kill "${PRODUCER_PID}" 2>/dev/null || true
    sleep 2
    [[ -n "${PUMP_PID}" ]] && kill "${PUMP_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

die() { echo "[demo] ERROR: $*" >&2; exit 1; }

[[ -f "${STATION}" ]] || die "station config not found: ${STATION}"
mkdir -p "${LOG_DIR}" "$(dirname "${SPOOL}")"

echo "[demo] checking laptop decoder elements"
for element in nvh264dec h264parse videoconvert; do
    gst-inspect-1.0 --exists "${element}" || die "GStreamer element missing: ${element}"
done

echo "[demo] checking key login to ${ORIN}"
ssh "${SSH_OPTS[@]}" "${ORIN}" true || die "no key login to ${ORIN}; run: ssh-copy-id ${ORIN}"

echo "[demo] making the FIFO on the Orin if it does not exist"
ssh "${SSH_OPTS[@]}" "${ORIN}" "mkdir -p '$(dirname "${FIFO}")' && { [[ -p '${FIFO}' ]] || mkfifo '${FIFO}'; }"

# A fresh empty spool so the server attaches to this run only.
: > "${SPOOL}"

echo "[demo] starting the pump (must be reading before the producer opens the FIFO)"
ssh -T "${SSH_OPTS[@]}" -o Compression=no "${ORIN}" "cat '${FIFO}'" >> "${SPOOL}" 2> "${LOG_DIR}/pump.err" &
PUMP_PID=$!
sleep 1
kill -0 "${PUMP_PID}" 2>/dev/null || die "pump exited at once; see ${LOG_DIR}/pump.err"

echo "[demo] starting the producer on the Orin (-tt so stopping this script stops it there)"
ssh -tt "${SSH_OPTS[@]}" "${ORIN}" "${PRODUCER}" > "${LOG_DIR}/producer.log" 2>&1 &
PRODUCER_PID=$!

echo "[demo] waiting up to 30 s for H.264 to reach ${SPOOL}"
for _ in $(seq 1 30); do
    [[ -s "${SPOOL}" ]] && break
    kill -0 "${PRODUCER_PID}" 2>/dev/null || die "producer exited; see ${LOG_DIR}/producer.log"
    sleep 1
done
[[ -s "${SPOOL}" ]] || die "no data arrived; see ${LOG_DIR}/producer.log and ${LOG_DIR}/pump.err"
echo "[demo] data is flowing ($(stat -c %s "${SPOOL}") bytes so far)"

echo "[demo] starting capture-server; connect a client and request the camera named in ${STATION}"
cd "${EGOCAP_DIR}"
uv run capture-server --sensing-config "${STATION}"
