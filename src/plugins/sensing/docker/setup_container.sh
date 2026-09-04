#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# CONTAINER-side setup for the SENSING SG8A SIPL rig.
#
# SIPL has no daemon: the plugin opens the hardware itself, in-process. So this
# half is about two things only -- can this container reach the device nodes,
# and are the SDK headers here to build against.
#
# Usage:
#   ./setup_container.sh [options]
#
# Options:
#   --sdk-dir DIR   where to unpack the Jetson Multimedia API (default /usr/src)
#   --skip-sdk      do not fetch anything, just report
#   --purge-cache   drop the cached SDK downloads first, then fetch fresh
#   -y, --yes       assume yes for every prompt
#   -h, --help      this text

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../scripts/common.sh
source "$SCRIPT_DIR/../scripts/common.sh"

SDK_ROOT=/usr/src
SKIP_SDK=0
ASSUME_YES=0

usage() { sed -n '4,19p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; }

while (( $# )); do
    case "$1" in
        --sdk-dir)  SDK_ROOT="$2"; MMAPI_DIR="$2/jetson_multimedia_api"; shift 2 ;;
        --skip-sdk) SKIP_SDK=1; shift ;;
        --purge-cache) [[ -n "$SENSING_CACHE_DIR" ]] && rm -rf "${SENSING_CACHE_DIR:?}"; shift ;;
        -y|--yes)   ASSUME_YES=1; shift ;;
        -h|--help)  usage; exit 0 ;;
        *) die "unknown argument: $1" "Run $0 --help" ;;
    esac
done
export ASSUME_YES

in_container || die "this script is for the CONTAINER side." \
    "You appear to be on the host — run ../scripts/setup.sh instead."

printf '%sSENSING container setup (SIPL)%s\n' "$C_BOLD" "$C_RESET"

BLOCKED=0

# --- 1. Device nodes and groups --------------------------------------------
# Everything SIPL touches is 0660 root:<group>. Root is NOT required -- the
# plugin runs as the ordinary container user -- but the supplementary groups
# are, and a missing one surfaces only as
#   Master SetPlatformConfig (Camera HAL) failed. status: 10
# which names neither the node nor the permission. Hence this check.
step "Device nodes"
missing_nodes=()
while IFS= read -r entry; do
    path="${entry%%:*}"; desc="${entry#*:}"
    [[ -e "$path" ]] || { missing_nodes+=("$path"); bad "$path absent — $desc"; continue; }
    [[ -r "$path" ]] && ok "$path" || bad "$path present but not readable — $desc"
done < <(sensing_required_nodes)

if [[ "${#missing_nodes[@]}" -gt 0 ]]; then
    hint "Nodes absent entirely: this container needs the host /dev."
    hint '  docker run: -v /dev:/dev   (or --device for each node)'
    BLOCKED=1
fi

mapfile -t MISSING_GIDS < <(sensing_missing_gids)
if [[ "${#MISSING_GIDS[@]}" -gt 0 ]]; then
    bad "this user is missing ${#MISSING_GIDS[@]} supplementary group(s)"
    # --group-add reaches PID 1 only; `docker exec` resolves supplementary groups
    # from the container's own /etc/group, so both halves are required.
    printf '\n    %sBoth of these, then REBUILD (a restart will not re-apply runArgs):%s\n' "$C_BOLD" "$C_RESET"
    printf '    %s1. devcontainer.json runArgs%s\n' "$C_BOLD" "$C_RESET"
    for gid in "${MISSING_GIDS[@]}"; do
        hint "\"--group-add\", \"$gid\","
    done
    printf '    %s2. Dockerfile - the same gids must exist in the image%s\n' "$C_BOLD" "$C_RESET"
    for gid in "${MISSING_GIDS[@]}"; do
        hint "RUN groupadd -g $gid <name> && usermod -aG <name> \$USERNAME"
    done
    BLOCKED=1
else
    ok "all required groups present (uid $(id -u), groups: $(id -G | tr ' ' ','))"
fi

# --- 2. SIPL runtime --------------------------------------------------------
# Supplied by the NVIDIA container runtime's CSV mounts plus the vendor
# install.sh on the host. Nothing in here can install them.
step "SIPL runtime"
# Snapshot once: `grep -q` SIGPIPEs ldconfig, and `set -o pipefail` then reports
# a successful match as a failure.
LDCACHE="$(ldconfig -p 2>/dev/null || true)"
if grep -q 'libnvsipl\.so' <<<"$LDCACHE"; then
    ok "libnvsipl.so resolvable"
else
    bad "libnvsipl.so not found — the NVIDIA container runtime should provide it"
    BLOCKED=1
fi
if [[ -d "$SIPL_DRV_DIR" ]] && compgen -G "$SIPL_DRV_DIR/libnvuddf_*" >/dev/null; then
    ok "$SIPL_DRV_DIR ($(find "$SIPL_DRV_DIR" -name 'libnv*' | wc -l) driver libs)"
else
    bad "$SIPL_DRV_DIR is empty or absent"
    hint "On the HOST: run the vendor install.sh — see ../scripts/setup.sh"
    BLOCKED=1
fi
if [[ -r "$SIPL_NITO_DIR/$SENSING_NITO" ]]; then
    ok "$SIPL_NITO_DIR/$SENSING_NITO"
else
    bad "$SENSING_NITO missing from $SIPL_NITO_DIR — the ISP has no tuning to load"
    hint "On the HOST: run the vendor install.sh"
    BLOCKED=1
fi

# --- 3. Jetson Multimedia API ----------------------------------------------
# NvVideoEncoder sources and nvbufsurface.h. Fetchable: the deb is in the public
# pool, and the version is parsed out of /etc/nv_tegra_release so it matches the
# running BSP by construction.
step "Jetson Multimedia API"
if mmapi="$(find_mmapi)"; then
    ok "$mmapi"
elif [[ "$SKIP_SDK" -eq 1 ]]; then
    bad "not found (--skip-sdk given, not fetching)"
    BLOCKED=1
else
    rel="$(l4t_release || true)"
    [[ -n "$rel" ]] || die "cannot read the L4T release from /etc/nv_tegra_release."
    info "L4T R$rel — resolving nvidia-l4t-jetson-multimedia-api from the public pool"
    have curl || die "curl is required to fetch the SDK." "apt-get install -y curl"

    # The flat pool index carries absolute URLs for every deb. Note the package
    # lives under common/, not t234/ -- the t234 path 404s.
    pool_index="$(mktemp)"; trap 'rm -f "$pool_index"' EXIT
    curl -fsSL --retry 3 https://repo.download.nvidia.com/jetson/ -o "$pool_index" \
        || die "could not reach repo.download.nvidia.com."
    url="$(grep -oE "https://[^\"]*nvidia-l4t-jetson-multimedia-api_${rel}-[0-9]+_arm64\.deb" \
           "$pool_index" | sort -u | tail -1)"
    [[ -n "$url" ]] || die "no nvidia-l4t-jetson-multimedia-api build for L4T R$rel in the pool." \
        "Install it by hand and re-run, or pass --skip-sdk."

    info "$(basename "$url")"
    require_sudo "unpack the Jetson Multimedia API into $SDK_ROOT"
    if confirm "Fetch and unpack it now?"; then
        fetch_cached "$url" valid_deb || die "could not fetch $(basename "$url")"
        stage="$(mktemp -d)"
        # dpkg-deb -x unpacks files without registering the package, which is
        # what a container wants: no dependency resolution, no dpkg database
        # churn, and it is idempotent. Stage first so --sdk-dir can redirect it
        # -- the deb's own layout is ./usr/src/jetson_multimedia_api.
        dpkg-deb -x "$FETCHED_PATH" "$stage" || die "could not unpack $(basename "$url")"
        sudo mkdir -p "$SDK_ROOT"
        sudo cp -a "$stage/usr/src/jetson_multimedia_api" "$SDK_ROOT/"
        rm -rf "$stage"
        if mmapi="$(find_mmapi)"; then ok "$mmapi"; else bad "unpacked but headers still not found"; BLOCKED=1; fi
    else
        warn "declined — the plugin cannot build without it"
        BLOCKED=1
    fi
fi

# --- 4. SIPL API headers ----------------------------------------------------
# Jetson_SIPL_API_<rel>_aarch64.tbz2, published alongside the L4T release. Not
# in any apt pool, so it is fetched by URL rather than by package -- but it is
# a public download, no developer login.
step "SIPL API headers"
if sipl="$(find_sipl_api)"; then
    ok "$sipl"
elif [[ "$SKIP_SDK" -eq 1 ]]; then
    bad "not found (--skip-sdk given, not fetching)"
    BLOCKED=1
else
    rel="$(l4t_release || true)"
    [[ -n "$rel" ]] || die "cannot read the L4T release from /etc/nv_tegra_release."
    have curl || die "curl is required to fetch the SDK." "apt-get install -y curl"

    # 39.2.1 -> r39_Release_v2.1 / Jetson_SIPL_API_R39.2.1_aarch64.tbz2
    major="${rel%%.*}"; rest="${rel#*.}"
    url="https://developer.nvidia.com/downloads/embedded/L4T/r${major}_Release_v${rest}/release/Jetson_SIPL_API_R${rel}_aarch64.tbz2"
    if ! curl -fsIL "$url" >/dev/null 2>&1; then
        # Naming has changed before; fall back to whatever the release page links.
        info "derived URL not reachable, checking the Jetson Linux release page"
        url="$(curl -fsSL https://developer.nvidia.com/embedded/jetson-linux 2>/dev/null \
               | grep -oiE 'https://[^"'"'"'<> ]*SIPL_API[^"'"'"'<> ]*\.tbz2' | sort -u | tail -1)"
    fi

    if [[ -z "$url" ]]; then
        bad "could not locate the SIPL API tarball for L4T R$rel"
        hint "Find Jetson_SIPL_API_*.tbz2 on https://developer.nvidia.com/embedded/jetson-linux"
        hint "then: sudo tar xf <file> -C /usr/src/"
        BLOCKED=1
    else
        info "$(basename "$url")"
        require_sudo "unpack the SIPL API headers into /usr/src"
        if confirm "Fetch and unpack them now?"; then
            fetch_cached "$url" valid_tbz2 || die "could not fetch $(basename "$url")"
            # The tarball's own layout is ./usr/src/jetson_sipl_api.
            sudo tar xf "$FETCHED_PATH" -C /
            if sipl="$(find_sipl_api)"; then ok "$sipl"; else bad "unpacked but headers still not found"; BLOCKED=1; fi
        else
            warn "declined — the plugin cannot build without it"
            BLOCKED=1
        fi
    fi
fi

if [[ -n "${sipl:-}" ]]; then
    for sub in sipl/include sipl/include/query/include sipl/include/nvsci; do
        [[ -d "$sipl/$sub" ]] && ok "  $sub" || { bad "  $sub missing"; BLOCKED=1; }
    done
fi

# --- 5. Build prerequisites -------------------------------------------------
step "Build prerequisites"
if have nvcc || [[ -x /usr/local/cuda/bin/nvcc ]]; then
    ok "nvcc: $("${CUDA_PATH:-/usr/local/cuda}/bin/nvcc" --version 2>/dev/null \
        | sed -n 's/.*release \([0-9.]*\).*/CUDA \1/p' | tail -1)"
else
    bad "nvcc not found"; hint "Expected at /usr/local/cuda/bin/nvcc"; BLOCKED=1
fi
for lib in nvscibuf nvscisync nvbufsurface; do
    if grep -q "lib${lib}\.so" <<<"$LDCACHE"; then ok "lib${lib}.so"
    else bad "lib${lib}.so not resolvable"; BLOCKED=1; fi
done

# --- 6. Vendor package and smoke-test tool ----------------------------------
step "Vendor package"
if pkg="$(find_sensing_pkg)" && [[ -n "$pkg" ]]; then
    ok "$pkg"
    [[ -f "$pkg/$SENSING_CONFIG_JSON_REL" ]] \
        && ok "platform config: $SENSING_CONFIG_JSON_REL" \
        || { bad "$SENSING_CONFIG_JSON_REL missing from the package"; BLOCKED=1; }
else
    info "not visible in this container — only needed for the vendor smoke test"
    hint "Mount it, or set SENSING_PKG_DIR"
fi

if have nvsipl_camera; then
    ok "nvsipl_camera on PATH (vendor smoke test)"
else
    info "nvsipl_camera not on PATH — installed by the vendor install.sh on the host"
fi

step "Verifying"
"$SCRIPT_DIR/verify.sh" || true

printf '\n'
if [[ "$BLOCKED" -eq 0 ]]; then
    printf '%s%sContainer setup complete.%s\n' "$C_GREEN" "$C_BOLD" "$C_RESET"
else
    printf '%s%sContainer setup incomplete — see the actions above.%s\n' "$C_YELLOW" "$C_BOLD" "$C_RESET"
fi
