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
#   --install-drivers  run the vendor install.sh unattended (otherwise it is offered)
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
ASSUME_YES=0

usage() { sed -n '4,27p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; }

while (( $# )); do
    case "$1" in
        --pkg)             PKG_DIR="$2"; shift 2 ;;
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
        "Open a host terminal and run: $SCRIPT_DIR/setup.sh
         Inside a container, run $SCRIPT_DIR/../docker/setup_container.sh instead."
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
info "config:  $SENSING_PLATFORM_CONFIG  ($SENSING_CONFIG_JSON_REL)"
info "L4T:     R$(l4t_release || echo unknown)"

# --- Platform config --------------------------------------------------------
# The published package has no SHW5G_2 config: SENSING ships shw5g.json only to
# customers with a 2x SHW5G rig. Without it nvsipl_camera has nothing to load
# for this population, so drop the vendored copy in.
step "Platform config"
CFG_DST="$PKG_DIR/$SENSING_CONFIG_JSON_REL"
CFG_SRC="$SENSING_CONFIGS_DIR/$(basename "$SENSING_CONFIG_JSON_REL")"
if [[ -f "$CFG_DST" ]]; then
    ok "$SENSING_CONFIG_JSON_REL (shipped in this package)"
elif [[ -r "$CFG_SRC" ]]; then
    mkdir -p "$(dirname "$CFG_DST")" 2>/dev/null || true
    if cp "$CFG_SRC" "$CFG_DST" 2>/dev/null; then
        ok "installed $SENSING_CONFIG_JSON_REL from configs/"
    else
        warn "could not write $CFG_DST"
        hint "nvsipl_camera needs -t pointing at a config that defines $SENSING_PLATFORM_CONFIG"
    fi
else
    bad "no $SENSING_PLATFORM_CONFIG config available"
    hint "Expected $CFG_SRC"
fi

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
    warn "the package copies files this script does not know about; run install.sh by hand"

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
        # install.sh only copies files. The overlay is the sole artifact the
        # kernel reads at boot, so a drop that leaves it alone -- a driver
        # rebuild or an ISP retune -- takes effect on the next process start.
        if [[ " ${STALE[*]} " == *" $SENSING_DTBO "* ]]; then
            printf '\n%s%sReboot required.%s Select the overlay first:\n' "$C_YELLOW" "$C_BOLD" "$C_RESET"
            hint "sudo /opt/nvidia/jetson-io/jetson-io.py"
            hint "  Configure Jetson AGX CSI Connector"
            hint "  -> Configure for compatible hardware"
            hint "  -> $SENSING_OVERLAY_NAME"
            hint "  -> Save pin changes -> Save and reboot to reconfigure pins"
            printf '\nAfter the reboot, set maximum performance and re-run this script:\n'
            hint "sudo nvpmodel -m 0 && sudo jetson_clocks"
            exit 0
        fi
        ok "overlay unchanged — no reboot needed"
        STALE=()
    fi
    warn "skipped vendor install.sh"
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
mapfile -t MISSING_GIDS < <(sensing_missing_gids)
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
        ok "added — log out and back in for it to take effect"
        warn "Group changes do not apply to this shell. Re-login before the smoke test."
    else
        warn "without these, SIPL fails with: Master SetPlatformConfig (Camera HAL) failed. status: 10"
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

# --- 5. Optional smoke test -------------------------------------------------
# The honest overlay check. If the DTBO is not selected, or the modules are on
# the wrong ports, this is what says so.
if [[ "$SMOKE_TEST" -eq 1 ]]; then
    step "Vendor smoke test"
    if ! have nvsipl_camera; then
        bad "nvsipl_camera not on PATH — installed by the vendor install.sh"
    else
        info "streaming $SENSING_PLATFORM_CONFIG for 5s"
        ( cd "$PKG_DIR" && nvsipl_camera \
            -t "$SENSING_CONFIG_JSON_REL" -c "$SENSING_PLATFORM_CONFIG" \
            -m "$SENSING_LINK_MASKS" --enable-camera-hal -s -Z -r 5 ) \
            && ok "both sensors streamed" \
            || bad "smoke test failed — check cabling on CN1 CAM2/CAM3 and the selected overlay"
    fi
fi

step "Verifying"
"$SCRIPT_DIR/verify.sh" || true

printf '\n%s%sHost setup complete.%s\n' "$C_GREEN" "$C_BOLD" "$C_RESET"
# Validation is the point of the whole exercise, so hand over the exact command
# rather than a pointer to the docs.
printf '\nValidate the rig:\n'
hint "$SCRIPT_DIR/validate.sh"
printf '\nTo build against this rig, run %sdocker/setup_container.sh%s in your container,\n' \
    "$C_BOLD" "$C_RESET"
printf 'or %srun_docker_example.sh%s for a ready-made one.\n' "$C_BOLD" "$C_RESET"
