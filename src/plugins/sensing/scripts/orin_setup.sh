#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Prepare an Orin for the camera producer, then check the cameras work. Run it on the
# Orin. It ships with deploy_orin.py, both ahead of the first build and in each release.
#
# 1. Prerequisites: the build toolchain, the JetPack CUDA and Multimedia API packages and
#    uv, then the SENSING drivers and device-tree overlay (setup.sh), then a check of what
#    the deploy preflight checks. The SIPL headers come from the L4T SIPL API package.
# 2. Camera check: verify.sh (read-only report), then validate.sh (streams both sensors). When the
#    stream fails, or with --scan, all 8 camera inputs are tried one at a time to show where cameras are.
#
# Exit status: 0 if everything requested passed, 2 if a reboot or new login is needed before
# going on, 1 otherwise.
#
# Usage:
#   ./orin_setup.sh [--prereqs-only | --check-only] [--no-apt] [--no-drivers] [--sipl-api FILE] [--sdk-root DIR] [--sudo-password-stdin] [--scan | --detect DIR [--rescan]] [-t SECONDS]
#
# Options:
#   --prereqs-only   install and check prerequisites; skip the camera check
#   --check-only     run the camera check only
#   --no-apt         skip apt and uv installation; only check what is present
#   --no-drivers     skip setup.sh (the SENSING drivers and overlay)
#   --sipl-api FILE  unpack this Jetson_SIPL_API_R<release>_aarch64.tbz2 if the SIPL headers are missing
#   --sdk-root DIR   where usr/src/jetson_multimedia_api and jetson_sipl_api live (default /)
#   --sudo-password-stdin  read the sudo password from stdin, so nothing prompts
#   --scan           try all 8 camera inputs alone and report where cameras are (stops the producer service)
#   --detect DIR     the same probe, then write DIR/platform.json for the cameras found and print
#                    {"config", "masks", "count", "used", "positions"} as JSON on stdout. The inputs found are
#                    saved in DIR/known-cameras.json and tried first next time; all 8 are scanned only when that
#                    file is missing or a saved camera does not answer
#   --rescan         with --detect: ignore the saved cameras and scan all 8 inputs
#   -t, --seconds N  how long validate.sh streams (default 10)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "$SCRIPT_DIR/common.sh"

SERVICE=isaaccapture-producer.service
DO_PREREQS=1
DO_CHECK=1
DO_APT=1
DO_DRIVERS=1
SIPL_API=
SDK_ROOT=
REBOOT=0
SUDO_PASSWORD_STDIN=0
DO_SCAN=0
DETECT_DIR=
RESCAN=0
SECONDS_ARG=10

usage() { sed -n '4,35p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; }

while (( $# )); do
    case "$1" in
        --prereqs-only) DO_CHECK=0; shift ;;
        --check-only)   DO_PREREQS=0; shift ;;
        --no-apt)       DO_APT=0; shift ;;
        --no-drivers)   DO_DRIVERS=0; shift ;;
        --sipl-api)     SIPL_API="${2:?--sipl-api needs a file}"; shift 2 ;;
        --sdk-root)     SDK_ROOT="${2:?--sdk-root needs a directory}"; shift 2 ;;
        --sudo-password-stdin) SUDO_PASSWORD_STDIN=1; shift ;;
        --scan)         DO_SCAN=1; shift ;;
        --detect)       DETECT_DIR="${2:?--detect needs a directory}"; shift 2 ;;
        --rescan)       RESCAN=1; shift ;;
        -t|--seconds)   SECONDS_ARG="${2:?--seconds needs a value}"; shift 2 ;;
        -h|--help)      usage; exit 0 ;;
        *) die "unknown argument: $1" "Run $0 --help" ;;
    esac
done

if [[ "$SUDO_PASSWORD_STDIN" -eq 1 ]]; then
    # Every sudo below, including the ones in setup.sh and common.sh, answers from a private tmpfs file that is
    # removed on exit: the exported function adds -A, and the askpass program prints the file.
    umask 077
    SUDO_PWFILE="$(mktemp -p "${XDG_RUNTIME_DIR:-/dev/shm}" orin-sudo.XXXXXX)"
    SUDO_ASKPASS="$(mktemp -p "${XDG_RUNTIME_DIR:-/dev/shm}" orin-askpass.XXXXXX)"
    trap 'rm -f "$SUDO_PWFILE" "$SUDO_ASKPASS"' EXIT
    cat > "$SUDO_PWFILE"
    printf '#!/bin/sh\ncat "$SUDO_PWFILE"\n' > "$SUDO_ASKPASS"
    chmod 700 "$SUDO_ASKPASS"
    export SUDO_PWFILE SUDO_ASKPASS
    sudo() { command sudo -A "$@"; }
    export -f sudo
fi

APT_PACKAGES=(
    build-essential cmake pkg-config patchelf git curl rsync
    python3 python3-dev python3-venv
    nvidia-jetpack nvidia-l4t-jetson-multimedia-api
)

FAIL=0
check() {
    local name="$1"; shift
    if "$@" >/dev/null 2>&1; then ok "$name"; else warn "missing: $name"; FAIL=1; fi
}

prerequisites() {
    [[ "$(uname -m)" == aarch64 && -f /etc/nv_tegra_release ]] \
        || die "this is not a Jetson (aarch64 L4T) system."

    if [[ "$DO_APT" -eq 1 ]]; then
        step "Installing apt packages"
        sudo apt-get update
        sudo apt-get install -y "${APT_PACKAGES[@]}"
        if ! PATH="$HOME/.local/bin:$PATH" have uv; then
            step "Installing uv"
            curl -LsSf https://astral.sh/uv/install.sh | sh
        fi
    fi

    if [[ "$DO_DRIVERS" -eq 1 ]]; then
        # Idempotent: it reinstalls only what differs. It exits nonzero while a reboot or a new login is still
        # needed (the Jetson-IO overlay choice, group membership), so stop there and re-run afterwards.
        step "Installing the SENSING drivers (setup.sh)"
        # -y: no prompts. Exit 3: the overlay is selected and a reboot is scheduled. Exit 2: only a new login is
        # needed for group membership, and every new ssh session has it.
        local code=0
        "$SCRIPT_DIR/setup.sh" -y || code=$?
        case "$code" in
            0|2) ;;
            3) warn "rebooting to load the device-tree overlay; run this again afterwards."
               FAIL=1; REBOOT=1; return ;;
            *) warn "setup.sh failed (exit $code)."; FAIL=1; return ;;
        esac
    fi

    if [[ ! -d "${SDK_ROOT%/}/usr/src/jetson_sipl_api" && -n "$SIPL_API" ]]; then
        step "Unpacking the SIPL API headers"
        sudo tar -xjf "$SIPL_API" -C "${SDK_ROOT:-/}"
    fi

    step "Checking prerequisites"
    export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
    local tool
    for tool in cmake g++ systemctl uv nvcc; do check "$tool" have "$tool"; done
    check "patchelf" have patchelf
    local sdk="${SDK_ROOT%/}/usr/src"
    check "Jetson Multimedia API ($sdk/jetson_multimedia_api)" test -d "$sdk/jetson_multimedia_api"
    check "SIPL headers ($sdk/jetson_sipl_api)" test -d "$sdk/jetson_sipl_api"
    check "SENSING drivers (nvsipl_camera)" have nvsipl_camera
    check "ISP tuning ($SIPL_NITO_DIR/$SENSING_NITO)" test -r "$SIPL_NITO_DIR/$SENSING_NITO"
    check "device-tree overlay ($SIPL_BOOT_DIR/$SENSING_DTBO)" test -f "$SIPL_BOOT_DIR/$SENSING_DTBO"
    check "overlay active: deserializer I2C bus (/dev/i2c-9)" test -e /dev/i2c-9
    check "4 GB free in $HOME" test "$(df --output=avail -B1 "$HOME" | tail -1)" -ge $((4 << 30))
    if [[ "$FAIL" -ne 0 ]]; then
        warn "SIPL headers: unpack Jetson_SIPL_API_R<release>_aarch64.tbz2 into / (it holds usr/src/jetson_sipl_api)."
        warn "SENSING drivers and overlay: rerun without --no-drivers, then reboot if asked."
    fi
}

# Try every camera input alone: four links on each of the two deserializers. The platform config places its modules
# on links 2 and 3 of deserializer CD (transport 1), so each probe uses a temporary copy whose one module sits on the
# input under test. The vendor file itself is never changed. A link whose camera does not lock fails SIPL
# initialisation (status 10). Fills FOUND (transport:link) and FOUND_LABEL.
FOUND=()
FOUND_LABEL=()

input_label() {
    if [[ "$1" -eq 1 ]]; then
        printf 'CN1 COAX%s (deserializer CD, link %s)' "$2" "$2"
    else
        printf 'CN2 COAX%s (deserializer AB, link %s)' "$((4 + $2))" "$2"
    fi
}

# Does a camera answer on this input (transport, link)? Records it in FOUND on success.
probe_input() {
    local transport="$1" link="$2" label config mask slots
    label="$(input_label "$transport" "$link")"
    config="$PROBE_TMP/shw5g-$transport-$link.json"
    python3 - "$SENSING_CONFIGS_DIR/shw5g.json" "$config" "$transport" "$link" <<'PY'
import json, sys
src, out, transport, link = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
data = json.load(open(src))
module = data["cameraConfigs"][0]["cameraModules"][0]
module.update(linkIndex=link, transportIndex=transport)
data["cameraConfigs"][0]["cameraModules"] = [module]
json.dump(data, open(out, "w"))
PY
    slots=("0x0000" "0x0000")
    slots[transport]="$(printf '0x%04x' $((1 << (4 * link))))"
    mask="${slots[0]} ${slots[1]}"
    if nvsipl_camera -t "$config" -c "$SENSING_PLATFORM_CONFIG" -m "$mask" \
            --enable-camera-hal -s -Z -r 3 </dev/null >"$PROBE_TMP/log" 2>&1; then
        ok "$label: camera streaming"
        FOUND+=("$transport:$link")
        FOUND_LABEL+=("$label")
        return 0
    fi
    info "$label: no camera"
    return 1
}

probe_inputs() {
    local transport link
    PROBE_TMP="$(mktemp -d)"
    step "Scanning all 8 camera inputs (one at a time, about 5 s each)"
    for transport in 1 0; do
        for link in 0 1 2 3; do
            probe_input "$transport" "$link" || true
        done
    done
    rm -rf "$PROBE_TMP"
}

scan_links() {
    probe_inputs
    if [[ "${#FOUND[@]}" -eq 0 ]]; then
        warn "no camera answers on any input. Check the carrier's 12 V supply and the power switching harness, that each"
        warn "camera is clicked into its cable, and that the cable is clicked into its harness lead."
        FAIL=1
        return
    fi
    ok "cameras found on ${#FOUND[@]} input(s); the deploy can use any of them (cameras: auto)"
}

# Write DIR/platform.json for the cameras found (the SHW5G config with its modules moved onto those inputs, at most
# the two the config names) and print what the producer needs on stdout. Everything else goes to stderr.
detect_cameras() {
    local dir="$1" known="$1/known-cameras.json" pair transport link stale=0 saved=()
    mkdir -p "$dir"
    if [[ "$RESCAN" -eq 0 ]]; then
        # The cameras saved last time, as "transport:link". A missing or unreadable file means none.
        mapfile -t saved < <(python3 - "$known" <<'PY'
import json, sys
try:
    for camera in json.load(open(sys.argv[1]))["cameras"]:
        print("%d:%d" % (camera["transport"], camera["link"]))
except (OSError, ValueError, KeyError, TypeError):
    pass
PY
)
    fi
    if [[ "${#saved[@]}" -gt 0 ]]; then
        # One short probe per saved camera instead of all 8 inputs.
        PROBE_TMP="$(mktemp -d)"
        step "Trying the cameras saved last time" >&2
        for pair in "${saved[@]}"; do
            transport="${pair%%:*}"
            link="${pair##*:}"
            probe_input "$transport" "$link" >&2 || stale=1
        done
        rm -rf "$PROBE_TMP"
        if [[ "$stale" -eq 1 ]]; then
            warn "a saved camera did not answer; scanning all inputs."
            FOUND=()
            FOUND_LABEL=()
        fi
    fi
    [[ "${#FOUND[@]}" -gt 0 ]] || probe_inputs >&2
    [[ "${#FOUND[@]}" -gt 0 ]] || { warn "no camera answers on any input."; return 1; }
    python3 - "$known" "${FOUND[@]}" -- "${FOUND_LABEL[@]}" <<'PY'
import json, sys
out, rest = sys.argv[1], sys.argv[2:]
split = rest.index("--")
cameras = []
for item, label in zip(rest[:split], rest[split + 1:]):
    transport, link = map(int, item.split(":"))
    cameras.append({"transport": transport, "link": link, "label": label})
json.dump({"cameras": cameras}, open(out, "w"), indent=1)
PY
    python3 - "$SENSING_CONFIGS_DIR/shw5g.json" "$dir/platform.json" "${FOUND[@]}" -- "${FOUND_LABEL[@]}" <<'PY'
import json, sys
src, out = sys.argv[1], sys.argv[2]
rest = sys.argv[3:]
split = rest.index("--")
found = [tuple(map(int, item.split(":"))) for item in rest[:split]]
labels = rest[split + 1:]
data = json.load(open(src))
config = data["cameraConfigs"][0]
templates = config["cameraModules"]  # the two modules the vendor config names
used = found[: len(templates)]
modules, slots = [], [0, 0]
for template, (transport, link) in zip(templates, used):
    modules.append(dict(template, linkIndex=link, transportIndex=transport))
    slots[transport] |= 1 << (4 * link)
config["cameraModules"] = modules
json.dump(data, open(out, "w"), indent=1)
print(json.dumps({
    "config": out,
    "masks": " ".join("0x%04x" % s for s in slots),
    "count": len(found),
    "used": len(used),
    "positions": labels,
}))
PY
}

camera_check() {
    # SIPL has one client at a time; a running producer would make the checks fail confusingly.
    if systemctl is-active --quiet "$SERVICE" 2>/dev/null; then
        if [[ "$DO_SCAN" -eq 0 ]]; then
            warn "$SERVICE is running and holds the cameras."
            warn "Stop it first: sudo systemctl stop $SERVICE"
            FAIL=1
            return
        fi
        step "Stopping $SERVICE so the scan can open the cameras"
        sudo systemctl stop "$SERVICE"
    fi
    if [[ "$DO_SCAN" -eq 1 ]]; then
        scan_links
        return
    fi
    step "Camera report (verify.sh)"
    "$SCRIPT_DIR/verify.sh" || FAIL=1
    step "Camera stream (validate.sh)"
    if ! "$SCRIPT_DIR/validate.sh" -t "$SECONDS_ARG"; then
        FAIL=1
        scan_links
    fi
}

if [[ -n "$DETECT_DIR" ]]; then
    have nvsipl_camera || die "nvsipl_camera is not on PATH." "Run orin_setup.sh --prereqs-only first."
    if systemctl is-active --quiet "$SERVICE" 2>/dev/null; then
        warn "$SERVICE is running and holds the cameras; stop it first (sudo systemctl stop $SERVICE)." >&2
        exit 1
    fi
    detect_cameras "$DETECT_DIR"
    exit $?
fi

[[ "$DO_PREREQS" -eq 0 ]] || prerequisites
if [[ "$DO_CHECK" -eq 1 ]]; then
    if [[ "$FAIL" -ne 0 ]]; then
        warn "skipping the camera check until the prerequisites above are fixed"
    else
        camera_check
    fi
fi

[[ "$FAIL" -eq 0 ]] || exit $(( REBOOT ? 2 : 1 ))
ok "all requested checks passed"
