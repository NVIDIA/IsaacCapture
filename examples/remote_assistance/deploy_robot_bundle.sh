#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ $# -gt 1 || ${1:-} == --help ]]; then
  echo "Usage: ROBOT_USER=user ROBOT_IP=address bash $0 [network-bundle-directory]"
  exit 0
fi
: "${ROBOT_USER:?Set ROBOT_USER to the Linux username on Robot Site}"
: "${ROBOT_IP:?Set ROBOT_IP to the Robot Site address}"
if [[ ! $ROBOT_USER =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]*$ || ! $ROBOT_IP =~ ^[a-zA-Z0-9][a-zA-Z0-9.-]*$ ]]; then
  echo "Use a Linux username and an IPv4 address or hostname." >&2
  exit 1
fi
bundle=${1:-"$HOME/remote-assistance-network"}
files=(session.yaml camera.yaml credentials.env ca.crt)
# Allow only the known camera profiles; never include Edge keys or arbitrary files.
profiles=(camera-oak-720p60.yaml camera-oak-1080p60.yaml camera-oak-4k30.yaml camera-oak-2016x1520-60.yaml
  camera-s56-720p60.yaml camera-s56-1080p30.yaml camera-s56-1080p60.yaml camera-s56-2560x1984-30.yaml)
for file in "${profiles[@]}"; do
  if [[ -e "$bundle/robot/$file" || -L "$bundle/robot/$file" ]]; then
    files+=("$file")
  fi
done
for file in "${files[@]}"; do
  if [[ ! -f "$bundle/robot/$file" || -L "$bundle/robot/$file" ]]; then
    echo "Missing regular Robot bundle file: $bundle/robot/$file" >&2
    exit 1
  fi
done

# Stream only Robot credentials, never the Edge server keys, over one SSH connection.
tar -C "$bundle/robot" -cf - -- "${files[@]}" |
  ssh "${ROBOT_USER}@${ROBOT_IP}" 'bash -c '\''
set -euo pipefail
umask 077
staging=$(mktemp -d "$HOME/.remote-assistance-upload.XXXXXXXX")
report_failure() {
  echo "Transfer failed; staging files retained at $staging" >&2
}
trap report_failure ERR
# Extraction must finish before the current bundle is moved.
tar -xf - -C "$staging" --no-same-owner
for file in session.yaml camera.yaml credentials.env ca.crt; do
  [[ -f "$staging/$file" && ! -L "$staging/$file" ]]
  chmod 600 "$staging/$file"
done
for file in "$staging"/camera-oak-*.yaml "$staging"/camera-s56-*.yaml; do
  [[ -e "$file" ]] || continue
  [[ -f "$file" && ! -L "$file" ]]
  chmod 600 "$file"
done
bundle="$HOME/remote-assistance-robot"
backup=""
if [[ -e "$bundle" || -L "$bundle" ]]; then
  backup=$(mktemp -d "$HOME/remote-assistance-robot.backup.XXXXXXXX")
  mv -- "$bundle" "$backup/robot"
fi
if ! mv -T -- "$staging" "$bundle"; then
  if [[ -n "$backup" ]]; then
    mv -T -- "$backup/robot" "$bundle"
  fi
  exit 1
fi
trap - ERR
if [[ -n "$backup" ]]; then
  echo "Previous Robot bundle preserved at $backup/robot"
fi
echo "Installed $bundle. Source its credentials.env before restarting clients."
'\'''
