#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Laptop side of the SIPL fast demo over plain TCP: listen for the Orin's H.264, append it to the spool, then run
# capture-server in the foreground. Start this first, then run run_fast_demo_orin.sh on the Orin (it prints the
# command). Ctrl+C stops both local processes. No SSH is involved.
#
# The stream is unencrypted and unauthenticated; the listener accepts one connection, from ORIN_IP only. Use a
# trusted network.
set -euo pipefail

ORIN_IP="${ORIN_IP:-192.168.1.176}"
PORT="${PORT:-5600}"
SPOOL="${SPOOL:-/var/tmp/sensing/sensor1.h264}"
EGOCAP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATION="${STATION:-${EGOCAP_DIR}/tmp/my_station.yaml}"
LOG_DIR="${LOG_DIR:-${EGOCAP_DIR}/tmp/fast-demo}"

LISTENER_PID=""

# Stop the listener; the Orin side ends its own sender when its producer stops.
cleanup() {
    trap - EXIT INT TERM
    echo "[demo] stopping"
    [[ -n "${LISTENER_PID}" ]] && kill "${LISTENER_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

die() { echo "[demo] ERROR: $*" >&2; exit 1; }

[[ -f "${STATION}" ]] || die "station config not found: ${STATION}"
command -v socat >/dev/null || die "socat is required (sudo apt install socat)"
mkdir -p "${LOG_DIR}" "$(dirname "${SPOOL}")"

echo "[demo] checking laptop decoder elements"
for element in nvh264dec h264parse videoconvert; do
    gst-inspect-1.0 --exists "${element}" || die "GStreamer element missing: ${element}"
done

# A fresh empty spool so the server attaches to this run only.
: > "${SPOOL}"

# Listening must start before the Orin connects. socat -u copies one direction, TCP to stdout, and exits when the
# Orin closes the connection.
echo "[demo] listening on TCP ${PORT} for ${ORIN_IP}"
socat -u "TCP-LISTEN:${PORT},reuseaddr,range=${ORIN_IP}/32" - >> "${SPOOL}" 2> "${LOG_DIR}/listener.err" &
LISTENER_PID=$!
sleep 1
kill -0 "${LISTENER_PID}" 2>/dev/null || die "listener exited at once; see ${LOG_DIR}/listener.err"

LAPTOP_IP="${LAPTOP_IP:-$(hostname -I | awk '{print $1}')}"
echo "[demo] now run on the Orin:"
echo "       LAPTOP_IP=${LAPTOP_IP} PORT=${PORT} ./run_fast_demo_orin.sh"

echo "[demo] waiting up to 120 s for H.264 to reach ${SPOOL}"
for _ in $(seq 1 120); do
    [[ -s "${SPOOL}" ]] && break
    kill -0 "${LISTENER_PID}" 2>/dev/null || die "listener exited; see ${LOG_DIR}/listener.err"
    sleep 1
done
[[ -s "${SPOOL}" ]] || die "no data arrived; see ${LOG_DIR}/listener.err"
echo "[demo] data is flowing ($(stat -c %s "${SPOOL}") bytes so far)"

echo "[demo] starting capture-server; connect a client and request the camera named in ${STATION}"
cd "${EGOCAP_DIR}"
uv run capture-server --sensing-config "${STATION}"
