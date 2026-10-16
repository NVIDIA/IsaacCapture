# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shared helpers for the SENSING setup scripts. Sourced, never executed.

# Colour is opt-out (NO_COLOR) and auto-disabled when stdout is not a terminal,
# so piped/CI output stays greppable.
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'
    C_RED=$'\033[31m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'
    C_BLUE=$'\033[34m'; C_CYAN=$'\033[36m'
else
    C_RESET=''; C_BOLD=''; C_DIM=''
    C_RED=''; C_GREEN=''; C_YELLOW=''; C_BLUE=''; C_CYAN=''
fi

step() { printf '\n%s==>%s %s%s%s\n' "$C_BLUE$C_BOLD" "$C_RESET" "$C_BOLD" "$1" "$C_RESET"; }
ok()   { printf '  %s✓%s %s\n' "$C_GREEN" "$C_RESET" "$1"; }
info() { printf '  %s·%s %s\n' "$C_DIM" "$C_RESET" "$1"; }
warn() { printf '  %s!%s %s\n' "$C_YELLOW$C_BOLD" "$C_RESET" "$1" >&2; }
bad()  { printf '  %s✗%s %s\n' "$C_RED$C_BOLD" "$C_RESET" "$1" >&2; }
hint() { printf '    %s%s%s\n' "$C_CYAN" "$1" "$C_RESET"; }

die() {
    printf '\n%sError:%s %s\n' "$C_RED$C_BOLD" "$C_RESET" "$1" >&2
    [[ -z "${2:-}" ]] || printf '%sAction:%s %s\n' "$C_CYAN$C_BOLD" "$C_RESET" "$2" >&2
    exit 1
}

have() { command -v "$1" >/dev/null 2>&1; }

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SENSING_CACHE_DIR="${SENSING_CACHE_DIR:-$COMMON_DIR/../.cache}"

in_container() {
    [[ -f /.dockerenv ]] || grep -qE '(docker|containerd|kubepods)' /proc/1/cgroup 2>/dev/null
}

# Loud, coloured notice naming every privileged action before the first prompt,
# then pre-authenticate so the password prompt lands here rather than midway
# through an install. `sudo -n true` first: on NOPASSWD hosts `sudo -v` still
# demands a password and would break unattended runs.
require_sudo() {
    local width=72 line reason

    # Full banner once per run; later privileged actions get a one-liner so the
    # notice stays visible without turning into wallpaper.
    if [[ "${_SUDO_BANNER_SHOWN:-0}" == "1" ]]; then
        for reason in "$@"; do
            printf '  %ssudo:%s %s\n' "$C_YELLOW$C_BOLD" "$C_RESET" "$reason"
        done
        sudo -n true 2>/dev/null && return 0
        sudo -v || die "sudo authentication failed." "Re-run as a user with sudo privileges."
        return 0
    fi
    _SUDO_BANNER_SHOWN=1

    printf -v line '%*s' "$width" ''; line=${line// /═}

    printf '\n%s%s╔%s╗%s\n' "$C_YELLOW" "$C_BOLD" "$line" "$C_RESET"
    printf '%s%s║%s SUDO REQUIRED %*s║%s\n' \
        "$C_YELLOW" "$C_BOLD" "$C_RED" "$((width - 15))" '' "$C_RESET"
    printf '%s%s╚%s╝%s\n' "$C_YELLOW" "$C_BOLD" "$line" "$C_RESET"
    printf '%sThis script needs root privileges for:%s\n' "$C_BOLD" "$C_RESET"
    local reason
    for reason in "$@"; do
        printf '  %s•%s %s\n' "$C_YELLOW$C_BOLD" "$C_RESET" "$reason"
    done
    printf '%sYou will be asked again before each optional action. Ctrl-C now to abort.%s\n\n' \
        "$C_DIM" "$C_RESET"

    if sudo -n true 2>/dev/null; then
        ok "sudo already authenticated"
        return 0
    fi
    printf '%sEnter your password to continue.%s\n' "$C_CYAN" "$C_RESET"
    sudo -v || die "sudo authentication failed." "Re-run as a user with sudo privileges."
}

# Yes/no prompt. ASSUME_YES=1 auto-accepts; a non-interactive stdin declines,
# so unattended runs never silently take a privileged optional action.
confirm() {
    local prompt="$1" reply
    if [[ "${ASSUME_YES:-0}" == "1" ]]; then
        info "$prompt ${C_DIM}[auto-yes]${C_RESET}"
        return 0
    fi
    if [[ ! -t 0 ]]; then
        warn "$prompt — declined (non-interactive; pass --yes to accept)"
        return 1
    fi
    printf '%s%s%s [y/N] ' "$C_YELLOW$C_BOLD" "$prompt" "$C_RESET"
    read -r reply
    [[ "$reply" =~ ^[Yy]$ ]]
}

# ---------------------------------------------------------------------------
# Rig facts
#
# SENSING SG8A-AGON-G2Y-A1 carrier, two SHW5G modules on CN1 CAM2/CAM3, driven
# through SIPL. There are no kernel sensor drivers and no /dev/video* nodes:
# SIPL owns the sensors from userspace. Everything below is read off the vendor
# platform config, not guessed -- run `nvsipl_query -t <json> -c <config>` to
# re-derive it.
# ---------------------------------------------------------------------------
SENSING_PKG_GLOB='SG8A_AGON_G2Y_A1_AGX_ORIN_S56C_SHW5G_SHF3L_SIPL_JP7.2.1_L4TR39.2.1'
SENSING_PLATFORM_CONFIG='SHW5G_2'
# One mask per deserializer, in transport order: nothing on CSI-GH, links 2 and
# 3 on CSI-CD.
SENSING_LINK_MASKS='0x0000 0x1100'
SENSING_NITO='SHW5G.nito'
SENSING_OVERLAY_NAME='Jetson Sensing SG8A-AGON-G2Y-A1 SIPL GMSL2x8'

# SENSING publishes the driver packages as directories in a public git repo --
# no releases, no tarballs, so there is nothing to curl.
SENSING_PKG_REPO='https://github.com/SENSING-Technology/nvidia-jetson-camera-drivers.git'
SENSING_PKG_REPO_DIR='Jetson AGX Orin Devkit/SG8A-AGON-G2Y-A1/JetPack7.2.1'
# The clone must keep the repo's own layout, so it lives out of the way and the
# package is reached through a flat symlink. Board names are unique across the
# repo's four platform directories, so the platform level carries nothing the
# board name does not.
SENSING_PKG_REPO_NAME='.drivers-repo'
SENSING_PKG_NAME='SG8A-AGON-G2Y-A1-JetPack7.2.1'

# SENSING supplies this two-camera config separately from the public package.
SENSING_CONFIGS_DIR="$COMMON_DIR/../configs"

# The two MAX96712 deserializer buses, from `i2cDevice` in the platform config.
SENSING_I2C_BUSES=(9 10)

SIPL_DRV_DIR=/usr/lib/nvsipl_drv
SIPL_NITO_DIR=/var/nvidia/nvcam/settings/sipl
SIPL_CAMERAHAL=/usr/lib/aarch64-linux-gnu/nvidia/libnvcamerahal.so
SIPL_SBIN_DIR=/usr/sbin
SIPL_BOOT_DIR=/boot
SENSING_DTBO=tegra234-camera-sipl-camera-overlay.dtbo

# BSP version from /etc/nv_tegra_release.
l4t_release() {
    local rel
    [[ -r /etc/nv_tegra_release ]] || return 1
    rel="$(sed -n '1s/^# R\([0-9]*\).*REVISION: \([0-9.]*\).*/\1.\2/p' /etc/nv_tegra_release)"
    [[ -n "$rel" ]] || return 1
    printf '%s\n' "$rel"
}

# Locate the vendor driver package. $SENSING_PKG_DIR wins; otherwise search the
# usual drop points. Prints the path, empty if not found.
#
# Identified by install.sh + the query/ tree, since the SIPL package has no
# load_modules.sh -- there is nothing to load.
find_sensing_pkg() {
    if [[ -n "${SENSING_PKG_DIR:-}" ]]; then
        printf '%s\n' "$SENSING_PKG_DIR"
        return 0
    fi
    local d
    for d in "$SENSING_CACHE_DIR/$SENSING_PKG_NAME" \
             "$HOME/$SENSING_PKG_GLOB" \
             "$HOME/Sensing/$SENSING_PKG_GLOB" \
             /home/*/"$SENSING_PKG_GLOB" \
             /home/*/Sensing/"$SENSING_PKG_GLOB" \
             /opt/sensing/"$SENSING_PKG_GLOB"; do
        [[ -f "$d/install.sh" && -d "$d/query" ]] && { printf '%s\n' "$d"; return 0; }
    done
    printf '\n'
}

# Sparse-clone the vendor package into the download cache. Prints its path.
#
# Blobless and sparse because the repo carries every board and JetPack
# combination -- ~700 MB, of which this rig needs 3.6 MB. Progress goes to
# stderr so the path is the only thing on stdout.
fetch_sensing_pkg() {
    local root="$SENSING_CACHE_DIR/$SENSING_PKG_REPO_NAME" pkg
    have git || { warn "git is required to fetch the vendor package"; return 1; }

    if [[ -d "$root/.git" ]]; then
        info "refreshing $SENSING_PKG_REPO_NAME" >&2
        git -C "$root" fetch -q --depth 1 origin HEAD && git -C "$root" checkout -q FETCH_HEAD \
            || warn "refresh failed, using the cached copy"
    else
        info "cloning $SENSING_PKG_REPO_NAME (sparse)" >&2
        rm -rf "$root"
        mkdir -p "$SENSING_CACHE_DIR"
        git clone --filter=blob:none --no-checkout --depth 1 -q "$SENSING_PKG_REPO" "$root" \
            || { rm -rf "$root"; return 1; }
        # --no-cone: the path has spaces in it, which cone mode cannot express.
        git -C "$root" sparse-checkout set --no-cone "/$SENSING_PKG_REPO_DIR/*" || return 1
        git -C "$root" checkout -q || return 1
    fi

    # Find the package by its install.sh rather than by name, so a renamed or
    # re-versioned vendor drop still resolves.
    pkg="$(find "$root/$SENSING_PKG_REPO_DIR" -maxdepth 2 -name install.sh -printf '%h\n' 2>/dev/null | head -1)"
    [[ -n "$pkg" ]] || { warn "no install.sh under $SENSING_PKG_REPO_DIR"; return 1; }

    # Flat entry point. The repo nests the package four deep behind a directory
    # with spaces in it; nobody should have to type that.
    ln -sfn "${pkg#"$SENSING_CACHE_DIR/"}" "$SENSING_CACHE_DIR/$SENSING_PKG_NAME"
    printf '%s\n' "$SENSING_CACHE_DIR/$SENSING_PKG_NAME"
}

# Camera modules this rig actually populates. Everything else in the package --
# S56C and SHF3L drivers and their NITOs -- is dead weight here, so it is
# neither compared nor installed.
SENSING_MODULES=(SHW5G)

# Where the installed package is recorded, so a run can name the upstream commit
# it is on. SENSING reuses the package name across drops and does not bump
# Version.md, so this is the only place a version exists.
SENSING_STAMP=/var/lib/sensing/installed.json

# Classify a package file by camera module: prints the module name, or "shared"
# for anything every module needs. Naming is the vendor's: one
# libnvuddf_<module>_cameramodule_library.so and one <MODULE>.nito per module.
sensing_file_module() {
    local base module
    base="$(basename "$1")"
    case "$base" in
        libnvuddf_*_cameramodule_library.so)
            module="${base#libnvuddf_}"; module="${module%_cameramodule_library.so}"
            printf '%s\n' "${module^^}" ;;
        *.nito) printf '%s\n' "${base%.nito}" ;;
        *) printf 'shared\n' ;;
    esac
}

# True if this rig uses that module, or if the file is shared.
sensing_module_used() {
    local m
    [[ "$1" == shared ]] && return 0
    for m in "${SENSING_MODULES[@]}"; do
        [[ "$1" == "$m" ]] && return 0
    done
    return 1
}

# What the vendor install.sh copies, as "source<TAB>destination<TAB>module"
# triples. Tab separated because the package path contains spaces.
#
# Mirrors install.sh, which globs -- keep sensing_check_installer honest about
# that rather than trusting this list to stay in step by itself.
sensing_payload() {
    local pkg="$1" f
    for f in "$pkg"/driver/libnvuddf* "$pkg"/driver/libnvsipl* "$pkg"/driver/libnvcamerahal.so; do
        [[ -e "$f" ]] || continue
        if [[ "$(basename "$f")" == libnvcamerahal.so ]]; then
            printf '%s\t%s\t%s\n' "$f" "$SIPL_CAMERAHAL" shared
        else
            printf '%s\t%s\t%s\n' "$f" "$SIPL_DRV_DIR/$(basename "$f")" "$(sensing_file_module "$f")"
        fi
    done
    for f in "$pkg"/nito/*; do
        [[ -e "$f" ]] && printf '%s\t%s\t%s\n' \
            "$f" "$SIPL_NITO_DIR/$(basename "$f")" "$(sensing_file_module "$f")"
    done
    for f in nvsipl_camera nvsipl_query; do
        [[ -e "$pkg/$f" ]] && printf '%s\t%s\t%s\n' "$pkg/$f" "$SIPL_SBIN_DIR/$f" shared
    done
    [[ -e "$pkg/dts/$SENSING_DTBO" ]] && printf '%s\t%s\t%s\n' \
        "$pkg/dts/$SENSING_DTBO" "$SIPL_BOOT_DIR/$SENSING_DTBO" shared
    return 0
}

# Payload restricted to the modules this rig uses.
sensing_used_payload() {
    local src dst module
    while IFS=$'\t' read -r src dst module; do
        sensing_module_used "$module" && printf '%s\t%s\t%s\n' "$src" "$dst" "$module"
    done < <(sensing_payload "$1")
    return 0
}

# Basenames of used files that differ from what is installed. Empty means the
# host already matches the package.
#
# Compared by content, not by version: a drop that only rebuilds drivers and
# retunes the ISP is invisible to anything that looks at names or timestamps.
sensing_stale_files() {
    local src dst module
    while IFS=$'\t' read -r src dst module; do
        cmp -s "$src" "$dst" || printf '%s\n' "$(basename "$dst")"
    done < <(sensing_used_payload "$1")
    return 0
}

# Digest of the used payload, per module. Prints "module<TAB>sha256" lines.
# This is the version, since the vendor supplies none.
sensing_module_digests() {
    local pkg="$1" module
    for module in shared "${SENSING_MODULES[@]}"; do
        local -a files=()
        local src dst m
        while IFS=$'\t' read -r src dst m; do
            [[ "$m" == "$module" ]] && files+=("$src")
        done < <(sensing_used_payload "$pkg")
        [[ "${#files[@]}" -gt 0 ]] || continue
        # Sorted so the digest depends on content, not on glob order.
        printf '%s\t%s\n' "$module" \
            "$(printf '%s\n' "${files[@]}" | sort | tr '\n' '\0' | xargs -0 cat | sha256sum | cut -d' ' -f1)"
    done
}

# Every source install.sh copies, one per line, so we can prove our payload map
# covers it. Reimplementing the copy is what buys per-module scope; this is the
# check that stops a new vendor artifact from being silently dropped.
sensing_installer_sources() {
    # Every argument of every `cp` except the flags and the trailing destination.
    awk '/^[[:space:]]*sudo[[:space:]]+cp[[:space:]]/ {
             for (i = 3; i < NF; i++) if ($i !~ /^-/) print $i
         }' "$1/install.sh"
}

# Fail if install.sh copies something sensing_payload does not model. We do our
# own copying to get per-module scope, so a new vendor artifact would otherwise
# be dropped in silence -- which is how a missing camera-module driver turns
# into a bare "SetPlatformCfg status: 10".
sensing_check_installer() {
    local pkg="$1" src f missing=0
    local -A modelled=()
    while IFS=$'\t' read -r f _ _; do modelled["$(basename "$f")"]=1; done < <(sensing_payload "$pkg")
    while IFS= read -r src; do
        # Expand the vendor's globs against the package to compare like for like.
        for f in "$pkg"/$src; do
            [[ -e "$f" ]] || continue
            [[ -n "${modelled[$(basename "$f")]:-}" ]] && continue
            warn "install.sh copies $(basename "$f"), which setup does not model"
            missing=1
        done
    done < <(sensing_installer_sources "$pkg")
    return "$missing"
}

# Copy the used payload into place. Replaces the vendor install.sh so that
# modules this rig does not have stay untouched; install.sh wipes
# /usr/lib/nvsipl_drv wholesale and copies all of them.
sensing_install() {
    local pkg="$1" src dst module
    while IFS=$'\t' read -r src dst module; do
        sudo install -D -m "$(stat -c %a "$src")" "$src" "$dst" || return 1
        printf '      %s  [%s]\n' "$(basename "$dst")" "$module"
    done < <(sensing_used_payload "$pkg")
}

# Record what was installed, per module. The commit is what to quote at SENSING,
# since the package name and Version.md are the same across drops.
sensing_write_stamp() {
    local pkg="$1" name="$2" commit="$3" module digest
    local -a lines=()
    while IFS=$'\t' read -r module digest; do
        lines+=("    \"$module\": \"sha256:$digest\"")
    done < <(sensing_module_digests "$pkg")
    local body
    body="$(printf '%s,\n' "${lines[@]}")"; body="${body%,}"
    sudo mkdir -p "$(dirname "$SENSING_STAMP")" || return 1
    {
        printf '{\n  "package": "%s",\n  "commit": "%s",\n  "installed": "%s",\n' \
            "$name" "$commit" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf '  "modules": {\n%s\n  }\n}\n' "$body"
    } | sudo tee "$SENSING_STAMP" >/dev/null
}

# Read one top-level string from the stamp. Empty if absent -- a host set up by
# hand has no stamp, which is why content comparison stays authoritative.
sensing_read_stamp() {
    [[ -r "$SENSING_STAMP" ]] || return 1
    sed -n "s/.*\"$1\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*/\1/p" "$SENSING_STAMP" | head -1
}

# Commit the cached clone is on, short form. Empty if the package did not come
# from a clone (--pkg pointing at a hand-unpacked copy).
sensing_pkg_commit() {
    git -C "$1" rev-parse --short HEAD 2>/dev/null || printf 'unknown\n'
}

# Device nodes SIPL opens, as "path:description" pairs. Readability of each is
# the whole container-side health check: everything here is 0660 root:<group>,
# so a missing supplementary group is the only realistic failure.
sensing_required_nodes() {
    local bus
    for bus in "${SENSING_I2C_BUSES[@]}"; do
        printf '/dev/i2c-%s:MAX96712 deserializer bus\n' "$bus"
    done
    printf '/dev/gpiochip0:deserializer power enable\n'
    printf '/dev/gpiochip1:deserializer power enable\n'
    printf '/dev/capture-vi-channel0:VI capture channel\n'
    printf '/dev/capture-isp-channel0:ISP channel\n'
    printf '/dev/nvhost-ctrl-isp:ISP control\n'
    printf '/dev/nvmap:GPU memory\n'
    printf '/dev/v4l2-nvenc:H.264 encoder\n'
}

# GIDs of nodes this user cannot read, deduplicated. Empty output means the
# process can reach everything SIPL needs.
sensing_missing_gids() {
    local entry path gid seen=""
    while IFS= read -r entry; do
        path="${entry%%:*}"
        [[ -e "$path" ]] || continue
        [[ -r "$path" ]] && continue
        gid="$(stat -c %g "$path" 2>/dev/null)" || continue
        [[ " $seen " == *" $gid "* ]] && continue
        seen="$seen $gid"
        printf '%s\n' "$gid"
    done < <(sensing_required_nodes)
}
