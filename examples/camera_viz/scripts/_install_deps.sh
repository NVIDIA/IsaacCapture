#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Provisions the camera_viz venv + native codec. Invoked by
# ``camera_viz.sh setup`` (local) and ``camera_viz.sh deploy`` (over SSH).
#
# Modes:
#   --full         viewer + sender (workstation). Obtains isaaccapture from the
#                  package index, preferring a final release over a release
#                  candidate, and offers a source build if neither is available
#                  (or a local --wheel).
#   --sender-only  sender path only. No isaaccapture, no vulkan deps.
#
# Flags: --venv, --wheel, --python, --no-v4l2, --no-oakd, --with-rtp,
#        --with-zed, --zed-sdk, --build-from-source.

set -euo pipefail

# Output helpers. Same palette as camera_viz.sh so ``setup`` reads as one tool
# whether it runs locally or over ssh during ``deploy``. Color is dropped when
# stdout isn't a terminal, so piped logs and systemd journals stay clean.
if [[ -t 1 ]]; then
    _C_STEP=$'\033[1m'; _C_DIM=$'\033[2m'; _C_WARN=$'\033[33m'; _C_ERR=$'\033[31m'; _C_RESET=$'\033[0m'
else
    _C_STEP=""; _C_DIM=""; _C_WARN=""; _C_ERR=""; _C_RESET=""
fi
# step: one line per unit of work. note: indented detail under a step.
# Errors say "error:", not the script's filename — callers run camera_viz.sh,
# and _install_deps.sh is an implementation detail to them.
step() { echo "${_C_STEP}▸ $*${_C_RESET}"; }
note() { echo "${_C_DIM}  $*${_C_RESET}"; }
warn() { echo "${_C_WARN}warning:${_C_RESET} $*" >&2; }
die()  { echo "${_C_ERR}error:${_C_RESET} $*" >&2; exit 1; }

HERE="$(cd "$(dirname "$0")" && pwd)"
CAMERA_VIZ_DIR="$(cd "$HERE/.." && pwd)"

# Only resolved when this lives inside the IsaacTeleop tree (local flow).
# Empty on rsync'd robot deploys, which use --sender-only and don't need
# the wheel anyway.
REPO_ROOT=""
if [[ -d "$CAMERA_VIZ_DIR/../.." ]]; then
    REPO_ROOT="$(cd "$CAMERA_VIZ_DIR/../.." && pwd)"
fi

MODE=full
VENV_DIR="$CAMERA_VIZ_DIR/.venv"
PYTHON_VERSION=3.12
WHEEL=
WITH_V4L2=true
WITH_OAKD=true
# Opt-in: the RTP path pulls GStreamer system packages, a PyGObject source
# build, and the native codec — none of which direct mode needs.
# --sender-only forces it on below (streaming is the sender's whole job).
WITH_RTP=false
WITH_ZED=false
ZED_SDK_DIR=/usr/local/zed
# Skip the index probes and build isaaccapture from this checkout. Also the
# non-interactive answer to the tier-3 prompt below.
BUILD_FROM_SOURCE=false
# Jetson-specific provisioning: apt-install cuda-nvrtc and create the
# unversioned CUDA lib symlinks + ld.so cache entry that JetPack skips.
# Off on desktop where the normal CUDA installer covers both.
JETSON=false

while (( $# )); do
    case $1 in
        --full)         MODE=full; shift;;
        --sender-only)  MODE=sender; shift;;
        --jetson)       JETSON=true; shift;;
        --venv)         VENV_DIR=$2; shift 2;;
        --wheel)        WHEEL=$2; shift 2;;
        --python)       PYTHON_VERSION=$2; shift 2;;
        --no-v4l2)      WITH_V4L2=false; shift;;
        --no-oakd)      WITH_OAKD=false; shift;;
        --with-rtp)     WITH_RTP=true; shift;;
        --with-zed)     WITH_ZED=true; shift;;
        --zed-sdk)      ZED_SDK_DIR=$2; shift 2;;
        --build-from-source) BUILD_FROM_SOURCE=true; shift;;
        *) die "unknown arg: $1";;
    esac
done

# A sender-only host exists to stream RTP, so --with-rtp is implied there.
# Without this a bare ``--sender-only`` would install a sender that can't send.
if [[ "$MODE" == sender ]]; then
    WITH_RTP=true
fi

# major picks the cupy wheel (cupy-cuda12x / cupy-cuda13x).
# major.minor picks the apt nvrtc package (cuda-nvrtc-12-6 on Orin/JP6,
# cuda-nvrtc-13-0 on Thor/JP7); JetPack only publishes the exact-minor.
cuda_major=12
cuda_minor=0
if [[ -e /usr/local/cuda ]]; then
    cuda_resolved=$(readlink -f /usr/local/cuda 2>/dev/null)
    full=$(echo "$cuda_resolved" | grep -oE 'cuda-[0-9]+\.[0-9]+' | head -1 | sed 's/cuda-//')
    if [[ -n "$full" ]]; then
        cuda_major=$(echo "$full" | cut -d. -f1)
        cuda_minor=$(echo "$full" | cut -d. -f2)
    fi
fi

# System-dep check. apt-installable bits are NOT auto-installed — we
# only probe what's present and, if anything is missing, print the
# exact ``apt-get install`` command for the user to run and exit. The
# venv side (uv pip) is fully automated; the system side is opt-in by
# the user so setup never escalates privileges on their behalf.
check_system_deps() {
    if ! $WITH_RTP; then
        return 0
    fi
    if ! command -v apt-get >/dev/null 2>&1; then
        return 0  # not Debian/Ubuntu — user is on their own
    fi

    local pkgs=()
    # PyGObject lives in the venv (installed via uv below). What apt
    # owns here is:
    #   * Python extension build metadata/headers when uv resolves to a
    #     system Python (pycairo's Meson build asks for dependency('python')).
    #   * C build deps for the source build (libcairo / libgirepository /
    #     pkg-config).
    #   * Runtime Gst typelib (PyGObject loads it via gobject-introspection).
    #   * gst-inspect-1.0 for the plugin-presence probe below.
    command -v pkg-config >/dev/null 2>&1                       || pkgs+=(pkg-config)
    local py_dev_ver="$PYTHON_VERSION"
    if [[ "$MODE" == sender ]]; then
        local sys_py
        sys_py="$(command -v python3 || true)"
        if [[ -x "$sys_py" ]]; then
            py_dev_ver="$("$sys_py" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "$PYTHON_VERSION")"
        fi
    fi
    if ! pkg-config --exists "python-$py_dev_ver" 2>/dev/null \
            || [[ ! -f "/usr/include/python$py_dev_ver/Python.h" ]]; then
        pkgs+=("python${py_dev_ver}-dev")
    fi
    pkg-config --exists cairo 2>/dev/null                       || pkgs+=(libcairo2-dev)
    # Debian's libgirepository1.0-dev publishes the .pc file as
    # gobject-introspection-1.0, NOT girepository-1.0 — probe accordingly.
    pkg-config --exists gobject-introspection-1.0 2>/dev/null   || pkgs+=(libgirepository1.0-dev)
    ls /usr/lib/*-linux-gnu/girepository-1.0/Gst-1.0.typelib >/dev/null 2>&1 \
                                                                || pkgs+=(gir1.2-gstreamer-1.0)
    command -v gst-inspect-1.0 >/dev/null 2>&1                  || pkgs+=(gstreamer1.0-tools)
    # GStreamer elements RtpH264Sender / RtpH264Receiver need at runtime.
    # Checked per-element via gst-inspect-1.0 so partially-provisioned
    # hosts (typelib present, plugins missing — a real failure mode) still
    # get flagged correctly.
    local need_base=false need_good=false need_bad=false need_ugly=false
    if command -v gst-inspect-1.0 >/dev/null 2>&1; then
        gst-inspect-1.0 videoconvert >/dev/null 2>&1 || need_base=true
        gst-inspect-1.0 rtph264pay   >/dev/null 2>&1 || need_good=true
        gst-inspect-1.0 udpsink      >/dev/null 2>&1 || need_good=true
        gst-inspect-1.0 h264parse    >/dev/null 2>&1 || need_bad=true
        # x264enc is the CPU fallback in GstNvH264Encoder's candidate list.
        gst-inspect-1.0 x264enc      >/dev/null 2>&1 || need_ugly=true
    else
        # No gst-inspect → flag everything; user installs gst-tools and re-runs.
        need_base=true; need_good=true; need_bad=true; need_ugly=true
    fi
    $need_base && pkgs+=(gstreamer1.0-plugins-base)
    $need_good && pkgs+=(gstreamer1.0-plugins-good)
    $need_bad  && pkgs+=(gstreamer1.0-plugins-bad gstreamer1.0-libav)
    $need_ugly && pkgs+=(gstreamer1.0-plugins-ugly)

    # cuda-nvrtc on Jetson — JetPack ships partial CUDA without it.
    # Desktop CUDA installer drops libnvrtc into /usr/local/cuda; if it's
    # missing there, the user needs to fix their CUDA install.
    # capture, not `find | grep -q`: find's non-zero on an unreadable /usr dir trips pipefail.
    if $JETSON && [[ -z "$(find /usr -name 'libnvrtc.so*' -print -quit 2>/dev/null)" ]]; then
        pkgs+=("cuda-nvrtc-${cuda_major}-${cuda_minor}")
    fi

    if [[ ${#pkgs[@]} -eq 0 ]]; then
        return 0
    fi

    cat >&2 <<EOF
Missing system packages for the RTP path:
  ${pkgs[*]}

The exact command:
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends ${pkgs[*]}

(Only the RTP path needs these — it is opt-in via --with-rtp, and always on
for --sender-only. Direct mode works without them.)
EOF

    local ans=""
    if [[ -e /dev/tty ]]; then
        # NOTE: do NOT redirect stderr — ``read -p`` writes the prompt to
        # stderr, and we want the user to actually see it.
        read -r -p "Run those apt-get commands now? [y/N] " ans </dev/tty || ans=""
    fi
    case "${ans,,}" in
        y|yes)
            if ! sudo -n true 2>/dev/null; then
                echo "    sudo password required (one-time)"
            fi
            sudo apt-get update -qq
            sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${pkgs[@]}"
            ;;
        *)
            die "aborted — install the packages above and re-run"
            exit 1
            ;;
    esac
}
check_system_deps

# JetPack ships versioned libs (libnvrtc.so.13) without the unversioned
# symlink + ld.so cache entry that desktop CUDA creates. cupy looks up
# ``libnvrtc.so`` and fails to resolve without these. Skipped on desktop
# where the CUDA installer already lays down the right symlinks.
check_cuda_symlinks() {
    if ! $JETSON; then
        return 0
    fi
    if [[ ! -d /usr/local/cuda/lib64 ]]; then
        return 0
    fi
    local lib64=/usr/local/cuda/lib64
    local cmds=()
    for stem in libnvrtc.so libnvrtc-builtins.so libcudart.so; do
        if [[ ! -e "$lib64/$stem" ]]; then
            local versioned
            versioned=$(ls "$lib64/$stem".[0-9]* 2>/dev/null | sort -V | tail -1)
            if [[ -n "$versioned" ]]; then
                cmds+=("sudo ln -sf $(basename "$versioned") $lib64/$stem")
            fi
        fi
    done
    if ! ldconfig -p 2>/dev/null | grep -q "$lib64"; then
        cmds+=("echo $lib64 | sudo tee /etc/ld.so.conf.d/zz-camera-viz-cuda.conf >/dev/null"
               "sudo ldconfig")
    fi
    if [[ ${#cmds[@]} -eq 0 ]]; then
        return 0
    fi

    {
        echo "Jetson CUDA libs aren't wired into ld.so and the unversioned symlinks"
        echo "are missing; cupy cannot dlopen libnvrtc.so without them. Exact commands:"
        for c in "${cmds[@]}"; do
            echo "  $c"
        done
    } >&2

    local ans=""
    if [[ -e /dev/tty ]]; then
        read -r -p "Run those now? [y/N] " ans </dev/tty || ans=""
    fi
    case "${ans,,}" in
        y|yes)
            if ! sudo -n true 2>/dev/null; then
                echo "    sudo password required (one-time)"
            fi
            for stem in libnvrtc.so libnvrtc-builtins.so libcudart.so; do
                if [[ ! -e "$lib64/$stem" ]]; then
                    local versioned
                    versioned=$(ls "$lib64/$stem".[0-9]* 2>/dev/null | sort -V | tail -1)
                    if [[ -n "$versioned" ]]; then
                        sudo ln -sf "$(basename "$versioned")" "$lib64/$stem"
                        echo "    $lib64/$stem -> $(basename "$versioned")"
                    fi
                fi
            done
            if ! ldconfig -p 2>/dev/null | grep -q "$lib64"; then
                echo "$lib64" | sudo tee /etc/ld.so.conf.d/zz-camera-viz-cuda.conf >/dev/null
                sudo ldconfig
                echo "    registered $lib64 with ldconfig"
            fi
            ;;
        *)
            die "aborted — run the commands above and re-run setup"
            exit 1
            ;;
    esac
}
check_cuda_symlinks

# An OAK-D enumerates as a Movidius USB device (vendor 03e7) that udev leaves
# root-only, so depthai fails with "Insufficient permissions to communicate with
# X_LINK_UNBOOTED device". Only prompt when one is actually attached and nothing
# already grants access. Existing rules are matched by content, not file name:
# the vendor rule ships as 50-movidius.rules, 80-movidius.rules and others
# depending on who installed it, and writing our own would just duplicate it.
OAKD_UDEV_RULE='SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"'
OAKD_UDEV_PATH=/etc/udev/rules.d/80-movidius.rules

check_oakd_udev() {
    $WITH_OAKD || return 0
    command -v udevadm >/dev/null 2>&1 || return 0
    # shellcheck disable=SC2144  # the glob is the point: any attached 03e7 device
    grep -l '^03e7$' /sys/bus/usb/devices/*/idVendor >/dev/null 2>&1 || return 0
    if grep -rlq '03e7' /etc/udev/rules.d/ /lib/udev/rules.d/ 2>/dev/null; then
        return 0
    fi

    cat >&2 <<EOF
An OAK-D is attached but no udev rule grants access to it, so depthai will fail
with "Insufficient permissions to communicate with X_LINK_UNBOOTED device".

The exact commands:
  echo '$OAKD_UDEV_RULE' | sudo tee $OAKD_UDEV_PATH
  sudo udevadm control --reload-rules && sudo udevadm trigger
EOF

    local ans=""
    if [[ -e /dev/tty ]]; then
        read -r -p "Install the OAK-D udev rule now? [y/N] " ans </dev/tty || ans=""
    fi
    case "${ans,,}" in
        y|yes)
            step "installing the OAK-D udev rule"
            if ! sudo -n true 2>/dev/null; then
                echo "    sudo password required (one-time)"
            fi
            echo "$OAKD_UDEV_RULE" | sudo tee "$OAKD_UDEV_PATH" >/dev/null
            sudo udevadm control --reload-rules
            sudo udevadm trigger
            note "installed $OAKD_UDEV_PATH — replug the camera for it to take effect"
            ;;
        *)
            # Unlike the apt deps, this doesn't block the install: the venv is
            # still valid and the rule can be added later.
            warn "skipped the OAK-D udev rule; the camera will not be detected until it is installed"
            ;;
    esac
}
check_oakd_udev

# Bootstrap uv from astral.sh if missing (Jetson images don't ship it).
# The PATH export below only reaches this process, so record when the caller's
# shell will still be missing uv and say so at the end.
UV_OFF_PATH=false
if ! command -v uv >/dev/null 2>&1; then
    UV_OFF_PATH=true
    if [[ -x "$HOME/.local/bin/uv" ]]; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        step "installing uv (none found on PATH)"
        if ! command -v curl >/dev/null 2>&1; then
            if ! command -v apt-get >/dev/null 2>&1; then
                die "curl is required to bootstrap uv — install it and re-run setup"
            fi
            cat >&2 <<EOF
curl is required to bootstrap uv.

The exact command:
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends curl
EOF
            curl_ans=""
            if [[ -e /dev/tty ]]; then
                read -r -p "Run those apt-get commands now? [y/N] " curl_ans </dev/tty || curl_ans=""
            fi
            case "${curl_ans,,}" in
                y|yes)
                    step "installing curl (needed to bootstrap uv)"
                    if ! sudo -n true 2>/dev/null; then
                        echo "    sudo password required (one-time)"
                    fi
                    sudo apt-get update -qq
                    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl
                    ;;
                *)
                    die "curl is required to bootstrap uv — install it and re-run setup"
                    ;;
            esac
        fi
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
        command -v uv >/dev/null || {
            die "uv install failed — check ~/.local/bin/uv"
        }
    fi
fi

# isaaccapture: newest final release meeting the floor below, else newest
# release candidate, else a source build of this checkout. --wheel skips the
# ladder; sender-only deploys skip isaaccapture entirely (README has the detail).
#
# The series comes from this checkout's VERSION, so a release branch cannot
# resolve to a newer line. The cloudxr extra is required, not optional: XR is
# the default mode. uv accepts <path>[extra], so the extra rides along on the
# wheel and source-build tiers.
ISAACCAPTURE_EXTRAS="[cloudxr]"
ISAACCAPTURE_SERIES="$(cut -d. -f1,2 "$REPO_ROOT/VERSION" 2>/dev/null || true)"
ISAACCAPTURE_REQ="isaaccapture${ISAACCAPTURE_EXTRAS}==${ISAACCAPTURE_SERIES}.*"
ISAACCAPTURE_PKG="$ISAACCAPTURE_REQ"

# Prints the version the index would install, empty when nothing matches. Both
# tiers use the same requirement; pass --pre for the release-candidate tier.
# Resolves against the index without consulting the venv, so an already-installed
# copy cannot make a tier look satisfiable. --no-deps keeps it to one fetch.
isaaccapture_available() {
    echo "$1" | uv pip compile - --no-deps --quiet \
        --python-version "$PYTHON_VERSION" "${@:2}" 2>/dev/null |
        sed -n 's/^isaaccapture==//p' || true
}

resolve_isaaccapture_pkg() {
    if [[ -n "$WHEEL" ]]; then
        [[ -f "$WHEEL" ]] || die "--wheel '$WHEEL' not found"
        ISAACCAPTURE_PKG="${WHEEL}${ISAACCAPTURE_EXTRAS}"
        step "isaaccapture: local wheel"
        note "$WHEEL"
        return 0
    fi

    if ! $BUILD_FROM_SOURCE; then
        step "isaaccapture: resolving from the package index"
        local rc
        if [[ -n "$(isaaccapture_available "$ISAACCAPTURE_REQ")" ]]; then
            ISAACCAPTURE_PKG="$ISAACCAPTURE_REQ"
            note "final release ($ISAACCAPTURE_REQ)"
            return 0
        fi
        # Pin the rc exactly so only isaaccapture resolves pre-release.
        rc="$(isaaccapture_available "$ISAACCAPTURE_REQ" --pre)"
        if [[ -n "$rc" ]]; then
            ISAACCAPTURE_PKG="isaaccapture${ISAACCAPTURE_EXTRAS}==${rc}"
            note "no final release for $ISAACCAPTURE_REQ — using a release candidate ($rc)"
            return 0
        fi
        note "nothing matching $ISAACCAPTURE_REQ is installable from the index"
    fi

    # An rsync'd robot tree and a standalone copy of examples/camera_viz/ have
    # no repo root to build from.
    if [[ -z "$REPO_ROOT" || ! -f "$REPO_ROOT/pyproject.toml" ]]; then
        cat >&2 <<EOF
No way to obtain $ISAACCAPTURE_REQ.

Nothing suitable is published to the configured package index, and this copy
of camera_viz is not inside an IsaacTeleop checkout, so there is nothing to
build from either. Either:
  * clone https://github.com/NVIDIA/IsaacTeleop and re-run setup from its
    examples/camera_viz/ directory, or
  * build a wheel elsewhere and pass --wheel <path>.
EOF
        exit 1
    fi

    cat >&2 <<EOF

isaaccapture can be built from this checkout instead:
  $REPO_ROOT

That is a full C++ / CUDA / Vulkan build and takes a while. It needs CMake,
the CUDA toolkit, Vulkan headers, glslangValidator, and clang-format-14 (the
format gate runs as part of the build).
EOF

    if ! $BUILD_FROM_SOURCE; then
        local ans=""
        if [[ -e /dev/tty ]]; then
            # As with the apt prompt: do NOT redirect stderr, ``read -p`` writes
            # the prompt there.
            read -r -p "Build isaaccapture from source now? [y/N] " ans </dev/tty || ans=""
        fi
        case "${ans,,}" in
            y|yes) ;;
            *)
                die "aborted — pass --build-from-source to skip this prompt";;
        esac
    fi

    # Same mechanism as --wheel: uv builds the sdist/tree and installs the result.
    ISAACCAPTURE_PKG="${REPO_ROOT}${ISAACCAPTURE_EXTRAS}"
}

# The configuration banner: what this run is going to do, before it does any
# of it. Everything below is one ``step`` per unit of work.
EXTRAS=()
$WITH_V4L2 && EXTRAS+=(v4l2)
$WITH_OAKD && EXTRAS+=(oakd)
$WITH_RTP  && EXTRAS+=(rtp)
$WITH_ZED  && EXTRAS+=(zed)
step "camera_viz setup — ${MODE} mode"
note "venv    $VENV_DIR"
note "python  $PYTHON_VERSION"
note "cuda    ${cuda_major}.${cuda_minor} → cupy-cuda${cuda_major}x"
EXTRAS_LIST=none
if (( ${#EXTRAS[@]} )); then
    EXTRAS_LIST=$(printf '%s, ' "${EXTRAS[@]}"); EXTRAS_LIST=${EXTRAS_LIST%, }
fi
note "extras  $EXTRAS_LIST"

if [[ "$MODE" == full ]]; then
    resolve_isaaccapture_pkg
fi

if [[ ! -d "$VENV_DIR" ]]; then
    step "creating venv"
    # Strict venv isolation: no --system-site-packages. PyGObject + every
    # other Python dep is installed into the venv via uv below. Sender mode
    # still defaults to system python3 because Jetson images sometimes
    # don't have a uv-managed Python build for the JetPack arch+libc combo
    # — but the venv itself stays isolated.
    if [[ "$MODE" == sender ]]; then
        sys_py="$(command -v python3 || true)"
        [[ -x "$sys_py" ]] || die "system python3 required in --sender-only mode"
        uv venv "$VENV_DIR" --python "$sys_py"
    else
        uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
    fi
fi
PY="$VENV_DIR/bin/python"

# cupy ships separate packages per CUDA major (cupy-cuda12x, cupy-cuda13x...);
# they coexist on disk and CuPy warns about "multiple CuPy packages installed."
# If a prior setup picked a different major, uninstall the stale variant now.
target_cupy="cupy-cuda${cuda_major}x"
for v in cupy-cuda11x cupy-cuda12x cupy-cuda13x; do
    if [[ "$v" != "$target_cupy" ]] && uv pip show --python "$PY" "$v" >/dev/null 2>&1; then
        step "removing stale $v (target is $target_cupy)"
        uv pip uninstall --python "$PY" "$v" >/dev/null
    fi
done

# Broken-install guard: dist-info present but the package unusable (interrupted
# setup, manual `rm -rf cupy/`) reads as installed to uv, so it never reinstalls.
# Probe an attribute, not just the import: a directory left without __init__.py
# imports as an empty namespace package, where `import cupy` succeeds and
# `cupy.zeros` does not.
if uv pip show --python "$PY" "$target_cupy" >/dev/null 2>&1 \
        && ! "$PY" -c "import cupy; cupy.zeros" >/dev/null 2>&1; then
    step "reinstalling $target_cupy (installed but not usable)"
    uv pip uninstall --python "$PY" "$target_cupy" >/dev/null
fi

# Mirrors pyproject.toml. PyGObject is pinned <3.52: 3.52 dropped the
# girepository-1.0 build path, and Ubuntu 22.04 only ships 1.0
# (libgirepository1.0-dev). 3.50.x supports both. Source-builds against
# the C deps installed in ensure_apt_deps(); pycairo is a transitive dep.
# pillow renders the XR controls HUD (examples/camera_viz/hud.py).
PKGS=("pyyaml>=6.0" "$target_cupy" "numpy>=1.23" "scipy>=1.15" "pillow>=10.0")
[[ "$MODE" == full ]] && PKGS=("$ISAACCAPTURE_PKG" "${PKGS[@]}")
$WITH_V4L2 && PKGS+=("opencv-python>=4.5")
$WITH_OAKD && PKGS+=("depthai>=3.0")
$WITH_RTP  && PKGS+=("pybind11>=2.11" "PyGObject>=3.42,<3.52")

# Local wheels keep version ``1.3+local`` across rebuilds; uv's --upgrade
# no-ops on them. mtime probe forces a reinstall when the wheel's newer.
EXTRA_UV=()
if [[ "$MODE" == full && -f "$WHEEL" ]]; then
    wheel_mtime=$(stat -c %Y "$WHEEL" 2>/dev/null || echo 0)
    # Empty on a fresh venv; `|| true` keeps the no-match from aborting under pipefail+set -e.
    installed_dist=$(ls -d "$VENV_DIR"/lib/python*/site-packages/isaaccapture-*.dist-info 2>/dev/null | head -1 || true)
    if [[ -n "$installed_dist" ]]; then
        installed_mtime=$(stat -c %Y "$installed_dist" 2>/dev/null || echo 0)
        if (( wheel_mtime > installed_mtime )); then
            note "wheel is newer than the installed copy — forcing reinstall"
            EXTRA_UV+=(--reinstall-package isaaccapture)
        fi
    fi
fi

step "installing ${#PKGS[@]} packages"
note "${PKGS[*]}"
if (( ${#EXTRA_UV[@]} > 0 )); then
    uv pip install --python "$PY" --upgrade "${EXTRA_UV[@]}" "${PKGS[@]}"
else
    uv pip install --python "$PY" --upgrade "${PKGS[@]}"
fi

# ZED SDK ships get_python_api.py which downloads a matching pyzed wheel
# and then tries ``pip install`` it (which fails in uv venvs, no pip).
# We let that fail and install the wheel ourselves.
if $WITH_ZED; then
    [[ -f "$ZED_SDK_DIR/get_python_api.py" ]] || die \
        "--with-zed given but $ZED_SDK_DIR/get_python_api.py is missing.
       Install the ZED SDK, or point at it with --zed-sdk <dir>."
    step "fetching pyzed from $ZED_SDK_DIR"
    uv pip install --python "$PY" --quiet requests
    tmp=$(mktemp -d)
    pushd "$tmp" >/dev/null
    "$PY" "$ZED_SDK_DIR/get_python_api.py" || true
    pyzed_whl=$(ls -1 pyzed-*.whl 2>/dev/null | head -1 || true)
    if [[ -z "$pyzed_whl" ]]; then
        popd >/dev/null
        rm -rf "$tmp"
        die "get_python_api.py did not produce a wheel"
    fi
    uv pip install --python "$PY" --upgrade "$tmp/$pyzed_whl"
    popd >/dev/null
    rm -rf "$tmp"
fi

# Native NVENC/NVDEC codec. Failures are non-fatal: the runtime falls
# back to the GStreamer encoder when the native ``.so`` isn't importable.
if $WITH_RTP; then
    CODEC_DIR="$CAMERA_VIZ_DIR/codec"
    if [[ -d "$CODEC_DIR" ]]; then
        step "building native NVENC/NVDEC codec"
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
        if ! "$CODEC_DIR/build.sh"; then
            warn "codec build failed — falling back to the GStreamer encoder at runtime"
        fi
        deactivate
    fi
fi

# Smoke imports. ``gi`` is in the list under RTP to confirm PyGObject
# built and installed cleanly into the venv.
SMOKE_MODS="cupy yaml scipy.spatial.transform"
# ``websockets`` comes from the cloudxr extra and is what the XR default mode
# needs to launch the runtime; check it here so a missing extra fails setup
# rather than the first ``run --mode xr``.
[[ "$MODE" == full ]] && SMOKE_MODS="isaaccapture.viz websockets $SMOKE_MODS"
$WITH_RTP && SMOKE_MODS="$SMOKE_MODS gi"
step "verifying imports"
note "$SMOKE_MODS"
# Prints only failures; the step line above already said what is being checked.
"$PY" - <<PY || die "the venv is incomplete — see the failed imports above"
import importlib, sys
mods = "$SMOKE_MODS".split()
fail = []
for m in mods:
    try:
        mod = importlib.import_module(m)
    except Exception as e:
        fail.append((m, f"{type(e).__name__}: {e}"))
        continue
    # A partially-installed package leaves its directory without __init__.py,
    # and Python imports that as an empty namespace package: the import
    # succeeds and every attribute is missing. __file__ is None only for
    # namespace packages, and none of these are one, so it flags exactly that.
    if getattr(mod, "__file__", None) is None:
        fail.append((m, "imported as an empty namespace package — install is incomplete"))
for m, why in fail:
    print(f"  {m}: {why}", file=sys.stderr)
sys.exit(0 if not fail else 1)
PY

# Close on what the user can do next, not on the fact that the script ended.
step "setup complete"
if [[ "$MODE" == full ]]; then
    # Absolute paths: setup may be invoked from anywhere (repo root, the sample
    # dir, or over ssh), and a bare ``./camera_viz.sh`` only resolves in the
    # sample dir. Don't hand out a command that depends on the reader's cwd.
    note "cd $CAMERA_VIZ_DIR"
    note "./camera_viz.sh run configs/synthetic.yaml --mode window"
else
    note "sender ready — start it with camera_streamer.py <config>"
fi

# camera_viz itself calls .venv/bin/python directly, but the docs' uv commands
# won't work until the caller's shell can see the uv we bootstrapped.
if $UV_OFF_PATH; then
    warn "uv is at ~/.local/bin/uv, which is not on your PATH. To use it directly:"
    echo '         export PATH="$HOME/.local/bin:$PATH"   # add to ~/.bashrc to persist' >&2
fi
