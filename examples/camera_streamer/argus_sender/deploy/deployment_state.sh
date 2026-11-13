#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Snapshot and restore only the camera-streamer installation and its service state.
set -euo pipefail

action="${1:?snapshot, activate or restore is required}"
root="${2:?The camera_streamer installation directory is required}"
[[ "$root" == /*/camera_streamer && "$root" != *'/../'* ]] || {
    echo "Invalid camera-streamer installation directory: $root" >&2
    exit 2
}
backup="$root/.deployment-backup"
# Prefix system paths for isolated filesystem tests; deployment runs with sudo.
system_root="${CAMERA_DEPLOY_SYSTEM_ROOT:-}"
owned_directories=(argus_sender)
owned_units=(camera-argus-sender camera-time-sync camera-vi-error-trace)
saved_units=(nvargus-daemon chrony camera-time-sync camera-vi-error-trace camera-argus-sender)
owned_files=(
    /etc/systemd/system/camera-argus-sender.service
    /etc/systemd/system/camera-time-sync.service
    /etc/systemd/system/camera-vi-error-trace.service
    /etc/chrony/conf.d/camera-agx.conf
    /etc/systemd/journald.conf.d/10-camera-persistent-journal.conf
    /usr/local/lib/camera/lock_jetson_camera_clocks.sh
)

unit_exists() {
    local load_state
    load_state="$(systemctl show -p LoadState --value "$1.service" 2>/dev/null || true)"
    [[ -n "$load_state" && "$load_state" != not-found ]]
}

stop_owned_units() {
    local unit
    for unit in "${owned_units[@]}"; do
        if unit_exists "$unit"; then
            systemctl stop "$unit.service"
            systemctl disable "$unit.service"
        fi
    done
}

case "$action" in
    snapshot)
        mkdir -p "$root"
        pending="$(mktemp -d "$root/.deployment-backup.XXXXXX")"
        chmod 700 "$pending"
        trap 'rm -rf -- "$pending"' EXIT
        for directory in "${owned_directories[@]}"; do
            if [[ -e "$root/$directory" ]]; then
                [[ ! -L "$root/$directory" ]] || {
                    echo "Refusing to replace a symlink: $root/$directory" >&2
                    exit 1
                }
                cp -a -- "$root/$directory" "$pending/$directory"
            fi
        done
        for file in "${owned_files[@]}"; do
            if [[ -e "$system_root$file" || -L "$system_root$file" ]]; then
                mkdir -p "$(dirname "$pending/files$file")"
                cp -a -- "$system_root$file" "$pending/files$file"
            fi
        done
        for unit in "${saved_units[@]}"; do
            enabled="$(systemctl show -p UnitFileState --value "$unit.service" 2>/dev/null || true)"
            active="$(systemctl show -p ActiveState --value "$unit.service" 2>/dev/null || true)"
            printf '%s|%s|%s\n' "$unit" "${enabled:-not-found}" "${active:-inactive}" >> "$pending/services"
        done
        touch "$pending/complete"
        rm -rf -- "$backup"
        mv -- "$pending" "$backup"
        trap - EXIT
        ;;
    activate)
        stage="${3:?A staged build directory is required}"
        # Bash 3 does not apply errexit to [[ ... ]]; activation guards must exit explicitly.
        [[ "$stage" == "${root%/camera_streamer}"/.camera-streamer-stage.* ]] || {
            echo "Invalid staged deployment path: $stage" >&2
            exit 1
        }
        [[ -f "$backup/complete" && -x "$stage/argus_sender/build/argus_sender" && -d "$stage/argus_sender/tests" ]] || {
            echo "Activation requires a completed snapshot and a validated staged build" >&2
            exit 1
        }
        stop_owned_units
        for directory in "${owned_directories[@]}"; do
            rm -rf -- "$root/$directory"
            mv -- "$stage/$directory" "$root/$directory"
        done
        ;;
    restore)
        [[ -f "$backup/complete" ]] || {
            echo "No completed camera-streamer deployment snapshot at $backup" >&2
            exit 1
        }
        if [[ -f "$backup/restored" ]]; then
            echo "Camera-streamer deployment snapshot is already restored"
            exit 0
        fi
        stop_owned_units
        for file in "${owned_files[@]}"; do
            rm -f -- "$system_root$file"
            if [[ -e "$backup/files$file" || -L "$backup/files$file" ]]; then
                mkdir -p "$(dirname "$system_root$file")"
                cp -a -- "$backup/files$file" "$system_root$file"
            fi
        done
        for directory in "${owned_directories[@]}"; do
            rm -rf -- "$root/$directory"
            if [[ -d "$backup/$directory" ]]; then
                cp -a -- "$backup/$directory" "$root/$directory"
            fi
        done
        systemctl daemon-reload
        while IFS='|' read -r unit enabled active; do
            if ! unit_exists "$unit"; then
                continue
            fi
            case "$enabled" in
                enabled) systemctl enable "$unit.service" ;;
                enabled-runtime) systemctl enable --runtime "$unit.service" ;;
                disabled) systemctl disable "$unit.service" ;;
            esac
            systemctl reset-failed "$unit.service" || true
            if [[ "$active" == active ]]; then
                systemctl restart "$unit.service"
            else
                systemctl stop "$unit.service"
            fi
        done < "$backup/services"
        systemctl try-restart systemd-journald.service
        touch "$backup/restored"
        echo "Restored the camera-streamer deployment snapshot"
        ;;
    *) echo "Unknown deployment state action: $action" >&2; exit 2 ;;
esac
