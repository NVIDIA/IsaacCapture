#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Read-only health report for the SENSING SG8A SIPL rig. Takes no action, needs
# no sudo, and runs identically on the host and in a container.
#
# Exit status: 0 if the rig looks capturable from here, 1 otherwise.
#
# Usage:
#   ./verify.sh [-q|--quiet]

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "$SCRIPT_DIR/common.sh"

QUIET=0
case "${1:-}" in
    -q|--quiet) QUIET=1 ;;
    -h|--help)  sed -n '4,12p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
esac
[[ "$QUIET" -eq 0 ]] || { ok() { :; }; info() { :; }; step() { :; }; }

FAIL=0
note_fail() { FAIL=1; }

printf '%sSENSING rig report%s  %s(%s)%s\n' "$C_BOLD" "$C_RESET" \
    "$C_DIM" "$(in_container && echo container || echo host)" "$C_RESET"

# --- Platform ---------------------------------------------------------------
step "Platform"
rel="$(l4t_release || true)"
if [[ -n "$rel" ]]; then
    ok "L4T R$rel"
    [[ "$rel" == 39.* ]] || { warn "this plugin targets L4T R39.x (JetPack 7.x)"; }
else
    bad "cannot read /etc/nv_tegra_release"; note_fail
fi
[[ -r /proc/device-tree/model ]] \
    && info "$(tr -d '\0' < /proc/device-tree/model)"

# --- Device nodes -----------------------------------------------------------
step "Device nodes"
while IFS= read -r entry; do
    path="${entry%%:*}"; desc="${entry#*:}"
    if [[ ! -e "$path" ]]; then
        bad "$path absent — $desc"; note_fail
    elif [[ ! -r "$path" ]]; then
        bad "$path unreadable (gid $(stat -c %g "$path")) — $desc"; note_fail
    else
        ok "$path"
    fi
done < <(sensing_required_nodes)

mapfile -t MISSING_GIDS < <(sensing_missing_gids)
if [[ "${#MISSING_GIDS[@]}" -gt 0 ]]; then
    bad "missing supplementary group(s): ${MISSING_GIDS[*]}"
    hint "SIPL will fail with: Master SetPlatformConfig (Camera HAL) failed. status: 10"
    if in_container; then
        # Group membership comes from the image: Docker resolves supplementary
        # groups for the container user from its own /etc/group, which covers
        # PID 1 and `docker exec` alike. --group-add on the run adds nothing.
        hint "Add the gids to the IMAGE and rebuild — a restart will not do it:"
        hint "     RUN groupadd -g <gid> <name> && usermod -aG <name> \$USERNAME"
        hint "docker/image_setup.sh does exactly that, and another Dockerfile"
        hint "can bind-mount and run it. Gids for this host:"
        for gid in "${MISSING_GIDS[@]}"; do hint "     $gid ($(getent group "$gid" | cut -d: -f1))"; done
    else
        hint "sudo usermod -aG i2c,gpio \$USER   (then log out and back in)"
    fi
fi

# --- SIPL runtime -----------------------------------------------------------
step "SIPL runtime"
# Snapshot once: `grep -q` exits at the first match, which SIGPIPEs ldconfig,
# and under `set -o pipefail` that makes a *successful* match look like failure.
LDCACHE="$(ldconfig -p 2>/dev/null || true)"
for lib in nvsipl nvsipl_query nvscibuf nvscisync nvbufsurface; do
    if grep -q "lib${lib}\.so" <<<"$LDCACHE"; then ok "lib${lib}.so"
    else bad "lib${lib}.so not resolvable"; note_fail; fi
done
if compgen -G "$SIPL_DRV_DIR/libnvuddf_*" >/dev/null; then
    ok "$SIPL_DRV_DIR ($(find "$SIPL_DRV_DIR" -name 'libnv*' 2>/dev/null | wc -l) driver libs)"
else
    bad "$SIPL_DRV_DIR empty or absent — run the vendor install.sh on the host"; note_fail
fi
[[ -r "$SIPL_NITO_DIR/$SENSING_NITO" ]] \
    && ok "$SIPL_NITO_DIR/$SENSING_NITO" \
    || { bad "$SENSING_NITO missing — the ISP has no tuning"; note_fail; }

# --- Overlay ----------------------------------------------------------------
step "Device-tree overlay"
if [[ -f /boot/tegra234-camera-sipl-camera-overlay.dtbo ]]; then
    ok "dtbo installed"
    info "whether it is *selected* only shows up when capture is attempted"
    hint "validate.sh"
else
    if in_container; then
        info "/boot not visible from here"
    else
        bad "overlay not installed — setup.sh --install-drivers"; note_fail
    fi
fi

# --- Build SDKs -------------------------------------------------------------
step "Build SDKs"
if mmapi="$(find_mmapi)"; then ok "multimedia api: $mmapi"
else bad "jetson_multimedia_api not found — docker/setup_container.sh fetches it"; note_fail; fi
if sipl="$(find_sipl_api)"; then ok "sipl api: $sipl"
else bad "jetson_sipl_api not found — docker/setup_container.sh fetches it"; note_fail; fi

# --- Vendor package ---------------------------------------------------------
step "Vendor package"
pkg="$(find_sensing_pkg)"
if [[ -n "$pkg" ]]; then
    ok "$pkg"
    [[ -f "$pkg/$SENSING_CONFIG_JSON_REL" ]] \
        && ok "$SENSING_CONFIG_JSON_REL" \
        || { bad "platform config missing from the package"; note_fail; }
else
    info "not found (only needed for the vendor smoke test)"
fi
have nvsipl_camera && ok "nvsipl_camera on PATH" || info "nvsipl_camera not on PATH"

printf '\n'
if [[ "$FAIL" -eq 0 ]]; then
    printf '%s%sRig looks capturable from here.%s\n' "$C_GREEN" "$C_BOLD" "$C_RESET"
else
    printf '%s%sRig is not ready — see the failures above.%s\n' "$C_YELLOW" "$C_BOLD" "$C_RESET"
fi
exit "$FAIL"
