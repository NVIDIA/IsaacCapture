#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build and deploy argus_sender to an Orin/Thor with a working Argus camera driver.
# Usage: ./deploy_argus_sender.sh host [--host-lan-ip IP] [--user USER]
#                                    [--check | --rollback | --restore] [--verbose]
# --check is read-only. --rollback/--restore restores the preceding deployment
# snapshot, or removes a first installation. Driver setup is owned by the BSP.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SENDER_SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
HOST=""
AGX_USER="nvidia"
HOST_LAN_IP=""
CHECK_ONLY=0
ROLLBACK=0
VERBOSE=0
DEPLOYMENT_STARTED=0
STAGE_DIR=""

usage() { sed -n '5,9s/^# //p' "${BASH_SOURCE[0]}"; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --check) CHECK_ONLY=1; shift ;;
        --rollback|--restore) ROLLBACK=1; shift ;;
        --verbose|-v) VERBOSE=1; shift ;;
        --user|--host-lan-ip)
            [[ $# -ge 2 && -n "$2" ]] || { echo "$1 requires a value" >&2; exit 2; }
            if [[ "$1" == --user ]]; then AGX_USER="$2"; else HOST_LAN_IP="$2"; fi
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "Unknown option: $1" >&2; exit 2 ;;
        *) [[ -z "$HOST" ]] || { echo "Unexpected argument: $1" >&2; exit 2; }; HOST="$1"; shift ;;
    esac
done
[[ -n "$HOST" && "$HOST" != -* ]] || { echo "A target SSH host is required" >&2; exit 2; }
[[ $CHECK_ONLY -eq 0 || $ROLLBACK -eq 0 ]] || { echo "--check and --rollback are mutually exclusive" >&2; exit 2; }
[[ "$AGX_USER" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "Invalid SSH user" >&2; exit 2; }
if [[ "$HOST" != *@* && "$HOST" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then HOST="${AGX_USER}@${HOST}"; fi
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10)
RSYNC_SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
RSYNC_ARGS=(-az)
[[ $VERBOSE -eq 0 ]] || RSYNC_ARGS+=(-v)
info() { printf '  [ .. ] %s\n' "$*"; }
ok() { printf '  [ OK ] %s\n' "$*"; }
fail() { printf '  [FAIL] %s\n' "$*" >&2; }
die() { fail "$*"; exit 1; }
remote() { "${SSH[@]}" "$HOST" "$@"; }
deployment_state() {
    remote "sudo -n bash -s -- '$1' '$INSTALL_DIR' '${2:-}'" < "$SCRIPT_DIR/deployment_state.sh"
}
install_remote_file() {
    local content="$1" destination="$2" current tmp
    current="$(remote "sudo -n cat '$destination' 2>/dev/null" || true)"
    [[ "$current" != "$content" ]] || return 0
    tmp="/tmp/camera_avs1_$$_$(basename "$destination")"
    printf '%s\n' "$content" | remote "cat > '$tmp' && sudo -n install -D -m 0644 '$tmp' '$destination' && rm -f '$tmp'"
}
render() {
    local file="$1"; shift
    local -a replacements=()
    while [[ $# -gt 0 ]]; do replacements+=(-e "s|@$1@|$2|g"); shift 2; done
    sed "${replacements[@]}" "$file"
}
on_exit() {
    local rc=$?
    trap - EXIT
    if [[ $rc -ne 0 && $DEPLOYMENT_STARTED -eq 1 ]]; then
        fail "deployment failed; restoring the camera-streamer snapshot"
        deployment_state restore || fail "automatic restore failed; retry with --rollback"
    fi
    if [[ -n "$STAGE_DIR" ]]; then remote "rm -rf -- '$STAGE_DIR'" || true; fi
    exit "$rc"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

remote true >/dev/null 2>&1 || die "configure key-based SSH access to $HOST before deployment"
remote 'sudo -n true' >/dev/null 2>&1 || die "sudo access is required on $HOST"
FACTS="$(remote 'bash -s' <<'REMOTE'
printf 'USER=%s\nHOME=%s\n' "$(id -un)" "$HOME"
REMOTE
)"
fact() { awk -F= -v key="$1" '$1 == key {sub(/^[^=]*=/, ""); print; exit}' <<< "$FACTS"; }
AGX_HOME="$(fact HOME)"
REMOTE_USER="$(fact USER)"
[[ "$AGX_HOME" =~ ^/[a-zA-Z0-9_./-]+$ && "$REMOTE_USER" =~ ^[a-zA-Z0-9_-]+$ ]] || die "unsupported target home/user"
INSTALL_DIR="$AGX_HOME/camera_streamer"
SENDER_DIR="$INSTALL_DIR/argus_sender"
if [[ $ROLLBACK -eq 1 ]]; then
    deployment_state restore
    exit 0
fi
if [[ -z "$HOST_LAN_IP" ]]; then
    ROBOT_IP="$(ssh -G "$HOST" 2>/dev/null | awk '/^hostname / {print $2; exit}' || true)"
    HOST_LAN_IP="$(ip -o route get "${ROBOT_IP:-${HOST#*@}}" 2>/dev/null | sed -n 's/.*src \([0-9.]*\).*/\1/p' | head -1 || true)"
fi
[[ "$HOST_LAN_IP" =~ ^[a-zA-Z0-9._:-]+$ ]] || die "pass this host's camera-LAN address with --host-lan-ip"

check_deployment() {
    remote "test -x '$SENDER_DIR/build/argus_sender'"
    remote "'$SENDER_DIR/build/argus_sender' --check-config --config '$SENDER_DIR/configs/cameras.yaml'"
    remote "ldd '$SENDER_DIR/build/argus_sender' | grep -Eq 'libnvargus\\.so([[:space:]]|$)' && ! ldd '$SENDER_DIR/build/argus_sender' | grep -q libnvargus_socketclient"
    remote 'systemctl is-enabled --quiet camera-argus-sender.service && systemctl is-active --quiet camera-argus-sender.service'
    remote "systemctl show -p Conflicts --value camera-argus-sender.service | grep -qw nvargus-daemon.service"
    remote "systemctl show -p After --value camera-argus-sender.service | grep -qw camera-time-sync.service"
    remote 'systemctl is-enabled --quiet camera-time-sync.service'
    remote 'if systemctl is-active --quiet nvargus-daemon.service; then exit 1; fi'
    remote "ss -tan | grep -Eq '[:.]5000[[:space:]]'"
    remote "test -d /var/log/journal && grep -qs '^Storage=persistent' /etc/systemd/journald.conf.d/10-camera-persistent-journal.conf"
    remote "grep -Fqs 'server $HOST_LAN_IP ' /etc/chrony/conf.d/camera-agx.conf"
    remote 'chronyc waitsync 1 0.05 0.0 1'
    remote 'if test -f /sys/kernel/debug/tracing/events/tegra_rtcpu/rtcpu_vinotify_error/format; then systemctl is-active --quiet camera-vi-error-trace.service && test "$(sudo -n cat /sys/kernel/debug/tracing/instances/camera_vi_errors/events/tegra_rtcpu/rtcpu_vinotify_error/enable)" = 1; fi'
    bash "$SCRIPT_DIR/deploy_platform.sh" "$HOST" --jetson-clocks --check
    ok "sender, configuration, time synchronization and supported diagnostics are healthy"
}
if [[ $CHECK_ONLY -eq 1 ]]; then
    check_deployment
    exit 0
fi

remote 'bash -s' <<'REMOTE'
set -euo pipefail
test -d /usr/src/jetson_multimedia_api
command -v cmake >/dev/null
command -v g++ >/dev/null
command -v chronyc >/dev/null
grep -Eqs '^[[:space:]]*(confdir|include).*conf\.d' /etc/chrony/chrony.conf || {
    echo "Enable /etc/chrony/conf.d in chrony.conf before deploying" >&2
    exit 1
}
main_pid="$(systemctl show -p MainPID --value camera-argus-sender.service 2>/dev/null || true)"
for pid in $(pgrep -x argus_sender 2>/dev/null || true); do
    [[ "$pid" == "${main_pid:-0}" ]] || { echo "Unmanaged argus_sender PID $pid" >&2; exit 1; }
done
REMOTE

candidate="$(remote "mktemp -d '$AGX_HOME/.camera-streamer-stage.XXXXXX'")"
[[ "$candidate" == "$AGX_HOME"/.camera-streamer-stage.* && "$candidate" =~ ^/[a-zA-Z0-9_./-]+$ ]] || die "invalid staging directory"
STAGE_DIR="$candidate"
info "staging sender, remaps and tests on $HOST"
rsync "${RSYNC_ARGS[@]}" --exclude build/ --exclude .git/ --exclude __pycache__/ --exclude .DS_Store \
    -e "$RSYNC_SSH" "$SENDER_SRC_DIR/" "$HOST:$STAGE_DIR/argus_sender/"
info "building and validating before changing the running deployment"
remote "cmake -S '$STAGE_DIR/argus_sender' -B '$STAGE_DIR/argus_sender/build' -DCMAKE_BUILD_TYPE=Release -DDISABLE_MULTIPROCESS=ON && cmake --build '$STAGE_DIR/argus_sender/build' -j2 && ctest --test-dir '$STAGE_DIR/argus_sender/build' --output-on-failure && '$STAGE_DIR/argus_sender/build/argus_sender' --check-config --config '$STAGE_DIR/argus_sender/configs/cameras.yaml'"
deployment_state snapshot
DEPLOYMENT_STARTED=1
deployment_state activate "$STAGE_DIR"

info "installing camera-streamer configuration and services"
install_remote_file "$(cat "$SCRIPT_DIR/10-camera-persistent-journal.conf")" /etc/systemd/journald.conf.d/10-camera-persistent-journal.conf
remote 'sudo -n mkdir -p /var/log/journal && sudo -n systemd-tmpfiles --create --prefix /var/log/journal && sudo -n systemctl restart systemd-journald && sudo -n journalctl --flush'
install_remote_file "$(render "$SCRIPT_DIR/camera-chrony-agx.conf.in" HOST_LAN_IP "$HOST_LAN_IP")" /etc/chrony/conf.d/camera-agx.conf
remote 'sudo -n systemctl restart chrony.service'
install_remote_file "$(cat "$SCRIPT_DIR/lock_jetson_camera_clocks.sh")" /usr/local/lib/camera/lock_jetson_camera_clocks.sh
install_remote_file "$(cat "$SCRIPT_DIR/camera-time-sync.service.in")" /etc/systemd/system/camera-time-sync.service
install_remote_file "$(cat "$SCRIPT_DIR/camera-vi-error-trace.service.in")" /etc/systemd/system/camera-vi-error-trace.service
install_remote_file "$(render "$SCRIPT_DIR/camera-argus-sender.service.in" AGX_USER "$REMOTE_USER" SENDER_DIR "$SENDER_DIR")" /etc/systemd/system/camera-argus-sender.service
remote 'sudo -n systemctl daemon-reload'
remote 'sudo -n systemctl enable camera-time-sync.service camera-vi-error-trace.service camera-argus-sender.service'
remote 'if test "$(systemctl show -p LoadState --value nvargus-daemon.service)" != not-found; then sudo -n systemctl disable --now nvargus-daemon.service; fi'
remote 'sudo -n systemctl reset-failed camera-argus-sender.service && sudo -n systemctl restart camera-argus-sender.service'

# Argus capture startup can take tens of seconds on a cold board.
ready=0
for _ in $(seq 1 90); do
    if remote "systemctl is-active --quiet camera-argus-sender.service && ss -tan | grep -Eq '[:.]5000[[:space:]]'"; then
        ready=1
        break
    fi
    sleep 1
done
if [[ $ready -ne 1 ]]; then
    remote 'journalctl -u camera-argus-sender -n 40 --no-pager' || true
    die "sender did not become ready"
fi
DEPLOYMENT_STARTED=0
ok "camera-argus-sender is deployed; use --rollback to restore the preceding snapshot"
