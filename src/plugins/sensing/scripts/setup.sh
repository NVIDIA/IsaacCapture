#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# HOST-side setup for the SENSING SG8A GMSL rig (two SHW5G modules) on an
# AGX Orin running JetPack 7.2.1 / L4T R39.2.1.
#
# Installing the vendor drivers and selecting the device-tree overlay are host
# operations, and the only ones needed to run the cameras. Containers need
# device access on top; see docker/setup_container.sh.
#
# There is NO per-boot step. The vendor install.sh copies userspace SIPL
# drivers, a DTBO and ISP tuning files, and that is all -- there are no kernel
# modules to insmod and no /dev/video* nodes to configure.
#
# Usage:
#   ./setup.sh [options]
#
# Options:
#   --pkg DIR          vendor driver package dir (default: fetch from SENSING's repo)
#   --no-fetch         do not download the vendor package; fail if it is not on disk
#   --install-drivers  reinstall the vendor artifacts even if unchanged
#   --groups           add $USER to the i2c and gpio groups
#   --smoke-test       stream both sensors for a few seconds via nvsipl_camera
#   -y, --yes          assume yes for every prompt
#   -h, --help         this text

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "$SCRIPT_DIR/common.sh"

PKG_DIR=""
NO_FETCH=0
INSTALL_DRIVERS=0
DO_GROUPS=ask
SMOKE_TEST=0
REBOOT_REQUIRED=0
LOGIN_REQUIRED=0
ASSUME_YES=0

usage() { sed -n '4,27p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; }

while (( $# )); do
    case "$1" in
        --pkg)             [[ $# -ge 2 && -n "$2" ]] || die "--pkg requires a directory"; PKG_DIR="$2"; shift 2 ;;
        --no-fetch)        NO_FETCH=1; shift ;;
        --install-drivers) INSTALL_DRIVERS=1; shift ;;
        --groups)          DO_GROUPS=yes; shift ;;
        --smoke-test)      SMOKE_TEST=1; shift ;;
        -y|--yes)          ASSUME_YES=1; shift ;;
        -h|--help)         usage; exit 0 ;;
        *) die "unknown argument: $1" "Run $0 --help" ;;
    esac
done
export ASSUME_YES

# --- Preconditions ----------------------------------------------------------
if in_container; then
    die "this script must run on the HOST, not inside a container." \
        "Open a host terminal and run: $SCRIPT_DIR/setup.sh"
fi
[[ "$(uname -m)" == "aarch64" ]] || die "this rig is Jetson-only (found $(uname -m) )."

# Refresh unconditionally unless told otherwise. SENSING reuses the package
# name across drops, so a copy already on disk says nothing about whether it is
# current -- skipping the fetch when one exists is how you end up installing
# last month's ISP tuning.
if [[ -n "$PKG_DIR" ]]; then
    info "using $PKG_DIR (--pkg)"
elif [[ "$NO_FETCH" -eq 1 ]]; then
    PKG_DIR="$(find_sensing_pkg)"
else
    step "Vendor driver package"
    PKG_DIR="$(fetch_sensing_pkg)" || {
        warn "fetch failed — falling back to a copy already on disk"
        PKG_DIR="$(find_sensing_pkg)"
    }
fi
[[ -n "$PKG_DIR" && -f "$PKG_DIR/install.sh" ]] || die \
    "SENSING driver package not found." \
    "Check network access to $SENSING_PKG_REPO,
         or obtain the package by hand and pass --pkg /path/to/package."
PKG_DIR="$(cd "$PKG_DIR" && pwd)"

printf '%sSENSING host setup (SIPL)%s\n' "$C_BOLD" "$C_RESET"
info "package: $PKG_DIR"
info "config:  $SENSING_PLATFORM_CONFIG  ($SENSING_CONFIGS_DIR/shw5g.json)"
info "L4T:     R$(l4t_release || echo unknown)"

[[ -r "$SENSING_CONFIGS_DIR/shw5g.json" ]] || die "configs/shw5g.json missing."

# --- Installed vs package ---------------------------------------------------
step "Installed vs package  (modules: ${SENSING_MODULES[*]})"
PKG_COMMIT="$(sensing_pkg_commit "$PKG_DIR")"
INSTALLED_COMMIT="$(sensing_read_stamp commit || true)"
if [[ -z "$INSTALLED_COMMIT" ]]; then
    info "package $SENSING_PKG_NAME @ $PKG_COMMIT; nothing recorded as installed"
elif [[ "$INSTALLED_COMMIT" == "$PKG_COMMIT" ]]; then
    info "package $SENSING_PKG_NAME @ $PKG_COMMIT (installed)"
else
    warn "installed @ $INSTALLED_COMMIT, package is @ $PKG_COMMIT"
fi
sensing_check_installer "$PKG_DIR" || \
    die "the vendor installer contains unsupported artifacts" "Review the new package before installing it."

STALE=()
while IFS=$'\t' read -r src dst module; do
    base="$(basename "$dst")"
    if ! sensing_module_used "$module"; then
        printf '  %s-%s %-46s [%s] not on this rig\n' "$C_DIM" "$C_RESET" "$base" "$module"
    elif cmp -s "$src" "$dst"; then
        printf '  %s✓%s %-46s [%s]\n' "$C_GREEN" "$C_RESET" "$base" "$module"
    else
        printf '  %s!%s %-46s [%s] %sstale%s\n' "$C_YELLOW$C_BOLD" "$C_RESET" "$base" "$module" "$C_YELLOW" "$C_RESET"
        STALE+=("$base")
    fi
done < <(sensing_payload "$PKG_DIR")

NEED_INSTALL=0
if [[ "$INSTALL_DRIVERS" -eq 1 ]]; then
    NEED_INSTALL=1
elif [[ "${#STALE[@]}" -gt 0 ]]; then
    NEED_INSTALL=1
fi
[[ "$NEED_INSTALL" -eq 1 ]] || ok "up to date"

SUDO_REASONS=()
[[ "$NEED_INSTALL" -eq 0 ]] || SUDO_REASONS+=(
    "installing the vendor SIPL drivers, device-tree overlay and ISP tuning files")
[[ "$DO_GROUPS" == no ]] || SUDO_REASONS+=(
    "adding $USER to the i2c and gpio groups (you will be asked first)")
[[ "${#SUDO_REASONS[@]}" -eq 0 ]] || require_sudo "${SUDO_REASONS[@]}"

# --- 1. Install the vendor artifacts ----------------------------------------
# Not the vendor install.sh: that wipes $SIPL_DRV_DIR and copies every module's
# driver, so a rig with one module carries three. sensing_check_installer above
# is what keeps this copy in step with theirs.
if [[ "$NEED_INSTALL" -eq 1 ]]; then
    step "Installing SIPL drivers, overlay and ISP tuning"
    warn "This overwrites $SIPL_CAMERAHAL and the ${SENSING_MODULES[*]} files in $SIPL_DRV_DIR."
    if confirm "Proceed?"; then
        sensing_install "$PKG_DIR" || die "install failed" "Re-run with sudo available."
        sensing_write_stamp "$PKG_DIR" "$SENSING_PKG_NAME" "$(sensing_pkg_commit "$PKG_DIR")" \
            && ok "recorded in $SENSING_STAMP"
        # Only the overlay is read at boot; userspace updates need no reboot.
        if [[ " ${STALE[*]} " == *" $SENSING_DTBO "* ]]; then
            REBOOT_REQUIRED=1
        fi
    else
        die "driver installation declined; host setup is incomplete."
    fi
fi

# --- 2. Vendor artifacts in place -------------------------------------------
step "Vendor artifacts"
[[ -f /boot/tegra234-camera-sipl-camera-overlay.dtbo ]] \
    && ok "/boot/tegra234-camera-sipl-camera-overlay.dtbo" \
    || { bad "device-tree overlay not installed"; hint "Re-run with --install-drivers"; }
if compgen -G "$SIPL_DRV_DIR/libnvuddf_*" >/dev/null; then
    ok "$SIPL_DRV_DIR ($(find "$SIPL_DRV_DIR" -name 'libnv*' | wc -l) driver libs)"
else
    bad "$SIPL_DRV_DIR is empty"; hint "Re-run with --install-drivers"
fi
[[ -r "$SIPL_NITO_DIR/$SENSING_NITO" ]] \
    && ok "$SIPL_NITO_DIR/$SENSING_NITO" \
    || { bad "$SENSING_NITO missing — the ISP has no tuning to load"; hint "Re-run with --install-drivers"; }

# --- 3. Group membership ----------------------------------------------------
# SIPL needs no root, but it does need the groups owning the deserializer I2C
# buses and the GPIO chips that gate deserializer power.
step "Group membership"
mapfile -t MISSING_GIDS < <(
    sensing_missing_gids
    # I2C nodes may not exist until the first boot with the overlay.
    for group in i2c gpio video; do
        gid="$(getent group "$group" | cut -d: -f3)" || continue
        [[ " $(id -G) " == *" $gid "* ]] || printf '%s\n' "$gid"
    done
)
if [[ "${#MISSING_GIDS[@]}" -eq 0 ]]; then
    ok "$USER can already reach every node SIPL opens"
else
    names=()
    for gid in "${MISSING_GIDS[@]}"; do
        names+=("$(getent group "$gid" | cut -d: -f1 || echo "$gid")")
    done
    info "missing: ${names[*]}"
    add_groups=0
    case "$DO_GROUPS" in
        yes) add_groups=1 ;;
        no)  info "skipped" ;;
        ask) confirm "Add $USER to ${names[*]}?" && add_groups=1 ;;
    esac
    if [[ "$add_groups" -eq 1 ]]; then
        sudo usermod -aG "$(IFS=,; echo "${names[*]}")" "$USER"
        LOGIN_REQUIRED=1
        ok "added — the next login or reboot activates these groups"
    else
        die "required group membership declined; host setup is incomplete."
    fi
fi

# --- 4. Performance mode ----------------------------------------------------
step "Performance mode"
if have nvpmodel; then
    mode="$(nvpmodel -q 2>/dev/null | sed -n 's/^NV Power Mode: //p' | head -1)"
    info "power mode: ${mode:-unknown}"
    hint "Two sensors at 2560x1984@60 want: sudo nvpmodel -m 0 && sudo jetson_clocks"
else
    info "nvpmodel not found"
fi

# Complete first-install setup before offering a capture command.
if [[ "$REBOOT_REQUIRED" -eq 1 ]]; then
    step "Activate the device-tree overlay"
    hint "Configure Jetson AGX CSI Connector -> Configure for compatible hardware"
    hint "$SENSING_OVERLAY_NAME"
    hint "Save pin changes -> Save and reboot to reconfigure pins"
    warn "After reboot, return to the repository root and run the camera command below."
    hint 'nvsipl_camera -t src/plugins/sensing/configs/shw5g.json -c SHW5G_2 -m "0x0000 0x1100" --enable-camera-hal -s -Z --egl-display'
    if confirm "Open Jetson-IO now?"; then
        sudo /opt/nvidia/jetson-io/jetson-io.py
    fi
    warn "Host setup requires the selected overlay and a reboot before capture."
    exit 2
fi
if [[ "$LOGIN_REQUIRED" -eq 1 ]]; then
    warn "Host setup requires a new login to activate group membership before capture."
    exit 2
fi

step "Verifying"
"$SCRIPT_DIR/verify.sh" || die "host setup is incomplete — see the failures above."

if [[ "$SMOKE_TEST" -eq 1 ]]; then
    nvsipl_camera -t "$SENSING_CONFIGS_DIR/shw5g.json" -c "$SENSING_PLATFORM_CONFIG" \
        -m "$SENSING_LINK_MASKS" --enable-camera-hal -s -Z -r 5 \
        || die "smoke test failed — check cabling and the selected overlay."
fi

printf '\n%s%sHost setup complete.%s\n' "$C_GREEN" "$C_BOLD" "$C_RESET"
printf '\nFrom the repository root, check both cameras:\n'
hint 'nvsipl_camera -t src/plugins/sensing/configs/shw5g.json -c SHW5G_2 -m "0x0000 0x1100" --enable-camera-hal -s -Z --egl-display'
printf '\nFor a container, run on the host:\n'
hint 'src/plugins/sensing/docker/setup_container.sh'
