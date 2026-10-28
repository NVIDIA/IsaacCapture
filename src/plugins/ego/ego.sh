#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# One entry point for building, inspecting, recording, and verifying the
# capability-compatible EGO camera plugin. Use install.sh to download the
# pinned SDK and prepare system dependencies and host udev rules.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DEFAULT_PRESET="py3.11"

die() {
    echo "ego.sh: $*" >&2
    exit 1
}

warn() {
    echo "ego.sh: warning: $*" >&2
}

usage() {
    cat <<'EOF'
Usage:
  src/plugins/ego/ego.sh doctor [--sdk-root PATH] [--preset py3.11]
  src/plugins/ego/ego.sh build --sdk-root PATH [--preview] [--preset py3.11] [--jobs N] [--clean] [--install-prefix PATH]
  src/plugins/ego/ego.sh capabilities [--preset py3.11] [--plugin PATH] [--device-uid UID]
  src/plugins/ego/ego.sh record [--config FILE.toml] [options] [-- PLUGIN_OPTIONS...]
  src/plugins/ego/ego.sh verify RUN_DIRECTORY [--require-calibration] [--preset py3.11] [--plugin PATH]
  src/plugins/ego/ego.sh export-media RUN_DIRECTORY [--output DIRECTORY] [--preset py3.11]

Common record options:
  --config FILE.toml              Load reusable record settings; later CLI options override them.
  --duration SECONDS              Stop cleanly after SECONDS; omit for Ctrl-C.
  --output DIRECTORY              Default: recordings/ego_<timestamp>.
  --format mjpg|h264|h265         Default: h264.
  --width N --height N --fps N    Default: 1600 1300 30.
  --device-uid UID                Select one enumerated device.
  --reconnect-timeout SECONDS     Wait for the same physical device after disconnect (default: 30; 0 disables).
  --reconnect-interval-ms MS      Device re-enumeration interval (default: 1000).
  --no-imu                       Disable IMU (requested by default when available).
  --audio                        Request optional audio (default: off; synchronization remains unverified).
  --no-audio                     Disable audio, overriding an audio-enabled configuration.
  --preview                       Enable the plugin's SDL preview (requires a --preview build).
  --mcap-media MODE               sidecar (default) or embedded.
  --record-device-diagnostics=on|off  Save detailed diagnostics in MCAP (default on).
  --keep-media-sidecars           Retain H.264/H.265/MJPEG and WAV beside embedded MCAP.
  --require-calibration           Require complete structured calibration during verification.
  --preset NAME --plugin PATH     Select a build tree or override the executable.

Examples:
  ./src/plugins/ego/ego.sh doctor --sdk-root /opt/OrbbecSDK
  ./src/plugins/ego/ego.sh build --sdk-root /opt/OrbbecSDK --jobs 8
  ./src/plugins/ego/ego.sh build --sdk-root /opt/OrbbecSDK --preview --jobs 8
  ./src/plugins/ego/ego.sh capabilities
  ./src/plugins/ego/ego.sh record --duration 30
  # Capabilities also reports advertised profiles that remain uncertified.
  ./src/plugins/ego/ego.sh verify recordings/ego_20260810_120000
  ./src/plugins/ego/ego.sh export-media recordings/embedded_demo

All options after -- are passed unchanged to ego_camera_plugin.  For example:
  ... record --duration 30 -- --bitrate=8 --dynamic-bitrate=on

The plugin validates the exact requested stream combination against the SDK;
unsupported combinations fail instead of falling back to another profile.
Encoded profiles above 30 FPS are currently rejected after SDK resolution
until sustained recording and strict-decode certification passes.
EOF
}

preset_python_version() {
    case "$1" in
        py3.11) echo "3.11" ;;
        py3.12) echo "3.12" ;;
        py3.13) echo "3.13" ;;
        *) die "Unknown preset '$1'. Expected py3.11, py3.12, or py3.13." ;;
    esac
}

build_dir_for_preset() {
    local version
    version="$(preset_python_version "$1")"
    echo "$SOURCE_ROOT/build-ego-py$version"
}

validate_sdk_root() {
    local sdk_root="$1"
    [[ -n "$sdk_root" ]] || die "--sdk-root PATH (or ORBBEC_SDK_ROOT) is required."
    [[ -f "$sdk_root/include/libobsensor/ObSensor.hpp" ]] || die "Invalid SDK root '$sdk_root': missing include/libobsensor/ObSensor.hpp"
    [[ -f "$sdk_root/lib/OrbbecSDKConfig.cmake" ]] || die "Invalid SDK root '$sdk_root': missing lib/OrbbecSDKConfig.cmake"
    [[ -f "$sdk_root/lib/libOrbbecSDK.so.2" ]] || die "Invalid SDK root '$sdk_root': missing lib/libOrbbecSDK.so.2"
    [[ -d "$sdk_root/lib/extensions" ]] || die "Invalid SDK root '$sdk_root': missing lib/extensions"
    command -v c++ >/dev/null 2>&1 || die "Missing C++ compiler. Install build-essential."
    if ! c++ -std=c++17 -fsyntax-only -I "$sdk_root/include" "$SCRIPT_DIR/tools/sdk_check.cpp"; then
        die "SDK lacks the required EGO APIs. Use https://github.com/orbbec/OrbbecSDK_v2-EGO/releases/tag/v2.10.6-EGO-20261019"
    fi
}

resolve_plugin() {
    local preset="$1"
    local requested="${2:-}"
    local candidate
    if [[ -n "$requested" ]]; then
        candidate="$requested"
    elif [[ -n "${EGO_PLUGIN:-}" ]]; then
        candidate="$EGO_PLUGIN"
    elif [[ -x "$SCRIPT_DIR/ego_camera_plugin" ]]; then
        candidate="$SCRIPT_DIR/ego_camera_plugin"
    else
        candidate="$(build_dir_for_preset "$preset")/src/plugins/ego_camera/ego_camera_plugin"
    fi
    [[ -x "$candidate" ]] || die "Plugin is not executable: $candidate. Run 'build' first or pass --plugin PATH."
    echo "$candidate"
}

resolve_exporter() {
    local preset="$1"
    local requested_plugin="${2:-}"
    local candidate
    # A verify run must never mix a camera plugin and exporter from different build trees.
    if [[ -n "$requested_plugin" ]]; then
        [[ -x "$requested_plugin" ]] || die "Plugin is not executable: $requested_plugin"
        local plugin_dir
        plugin_dir="$(cd "$(dirname "$requested_plugin")" && pwd -P)" \
            || die "Cannot resolve plugin directory: $requested_plugin"
        for candidate in \
                "$plugin_dir/ego_mcap_export_media" \
                "$plugin_dir/../mcap/export_media_tool/ego_mcap_export_media"; do
            if [[ -x "$candidate" ]]; then
                echo "$(cd "$(dirname "$candidate")" && pwd -P)/$(basename "$candidate")"
                return 0
            fi
        done
        die "No embedded-media exporter exists beside plugin or in its build tree: $requested_plugin"
    fi
    if [[ -x "$SCRIPT_DIR/ego_mcap_export_media" ]]; then
        candidate="$SCRIPT_DIR/ego_mcap_export_media"
    else
        candidate="$(build_dir_for_preset "$preset")/src/plugins/ego_camera/ego_mcap_export_media"
    fi
    [[ -x "$candidate" ]] || die "Embedded-media exporter is not executable: $candidate. Run 'build' first."
    echo "$candidate"
}

require_command() {
    local command_name="$1"
    local remediation="$2"
    if ! command -v "$command_name" >/dev/null 2>&1; then
        warn "Missing command '$command_name'. $remediation"
        return 1
    fi
}

# The verify tools read MCAP. Use python3 when it already has a reader (for example an
# activated venv); otherwise uv runs them with mcap from its cache.
python_has_mcap() {
    python3 -c 'from mcap.reader import make_reader' >/dev/null 2>&1
}

run_mcap_tool() {
    if python_has_mcap; then
        python3 "$@"
    else
        uv run --quiet --no-project --with 'mcap>=1.2,<2' python3 "$@"
    fi
}

require_cmake_version() {
    local version
    require_command cmake "Install CMake 3.24 or newer." || return 1
    version="$(cmake --version | awk 'NR == 1 { print $3 }')"
    if [[ "$(printf '%s\n%s\n' "3.24" "$version" | sort -V | head -n 1)" != "3.24" ]]; then
        warn "CMake $version is too old; Isaac Teleop requires CMake 3.24 or newer."
        return 1
    fi
}

command_doctor() {
    local sdk_root="${ORBBEC_SDK_ROOT:-}"
    local preset="$DEFAULT_PRESET"
    local arg
    while (( $# )); do
        arg="$1"
        case "$arg" in
            --sdk-root) sdk_root="${2:-}"; shift 2 ;;
            --preset) preset="${2:-}"; shift 2 ;;
            --help|-h) usage; return 0 ;;
            *) die "Unknown doctor option: $arg" ;;
        esac
    done
    preset_python_version "$preset" >/dev/null

    local failed=0
    echo "== EGO environment diagnosis =="
    echo "Architecture: $(uname -m)"
    if [[ "$(uname -m)" != "x86_64" ]]; then
        warn "The validated host architecture is x86_64."
    fi
    if [[ -r /etc/os-release ]]; then
        . /etc/os-release
        echo "OS: ${PRETTY_NAME:-unknown}"
    fi
    require_command python3 "Install it with: sudo apt update && sudo apt install -y python3" || failed=1
    require_command ffmpeg "Install it with: sudo apt update && sudo apt install -y ffmpeg" || failed=1
    require_command ffprobe "Install it with: sudo apt update && sudo apt install -y ffmpeg" || failed=1
    if command -v python3 >/dev/null 2>&1 && ! python_has_mcap && ! command -v uv >/dev/null 2>&1; then
        warn "verify needs the Python MCAP reader. Install uv, or install mcap into the active Python environment."
        failed=1
    fi

    local source_checkout=0
    [[ -f "$SOURCE_ROOT/CMakeLists.txt" ]] && source_checkout=1
    if (( source_checkout )); then
        require_cmake_version || failed=1
        require_command c++ "Install it with: sudo apt update && sudo apt install -y build-essential" || failed=1
        require_command uv "Install uv before configuring Isaac Teleop." || failed=1
        if [[ -z "$sdk_root" ]]; then
            warn "No SDK root supplied. Source builds require --sdk-root PATH or ORBBEC_SDK_ROOT."
            failed=1
        fi
    fi

    if [[ -n "$sdk_root" ]]; then
        if validate_sdk_root "$sdk_root"; then
            echo "OrbbecSDK: $sdk_root"
            if [[ -x "$sdk_root/shared/install_udev_rules.sh" ]]; then
                echo "udev rules installer: sudo $sdk_root/shared/install_udev_rules.sh"
            else
                warn "Could not find this SDK release's shared/install_udev_rules.sh. Consult its package documentation."
            fi
        fi
    fi

    local plugin=""
    if plugin="$(resolve_plugin "$preset" "" 2>/dev/null)"; then
        echo "Plugin: $plugin"
    else
        if (( source_checkout )); then
            warn "Plugin is not built yet. Run: $SCRIPT_DIR/ego.sh build --sdk-root PATH"
        else
            warn "Plugin is unavailable. Install a complete prebuilt package."
            failed=1
        fi
    fi

    if (( !source_checkout )); then
        local package_file
        for package_file in "$SCRIPT_DIR/ego_mcap_export_media" "$SCRIPT_DIR/libOrbbecSDK.so.2" \
                "$SCRIPT_DIR/OrbbecSDKConfig.xml" "$SCRIPT_DIR/extensions"; do
            if [[ ! -e "$package_file" ]]; then
                warn "Prebuilt package is incomplete: missing $package_file"
                failed=1
            fi
        done
        if [[ -n "$plugin" ]] && ! "$plugin" --help >/dev/null 2>&1; then
            warn "Plugin cannot start. Check the packaged SDK libraries and host runtime dependencies."
            failed=1
        fi
    fi

    if command -v lsusb >/dev/null 2>&1; then
        local usb_line bus device node
        usb_line="$(lsusb -d 2bc5:1201 2>/dev/null || true)"
        if [[ -z "$usb_line" ]]; then
            warn "No EGO USB device (2bc5:1201) found. Connect the camera before capabilities or record."
        else
            echo "Detected EGO USB device(s) (2bc5:1201):"
            echo "$usb_line"
            while IFS= read -r usb_line; do
                bus="$(awk '{print $2}' <<< "$usb_line")"
                device="$(awk '{gsub(/:/, "", $4); print $4}' <<< "$usb_line")"
                node="/dev/bus/usb/$bus/$device"
                if [[ -r "$node" && -w "$node" ]]; then
                    echo "USB access: OK ($node)"
                else
                    warn "USB access is unavailable for $node. Install this SDK's udev rules, reconnect the camera, then run doctor again."
                fi
            done <<< "$usb_line"
        fi
    else
        warn "lsusb is unavailable; install usbutils to check camera discovery."
    fi

    if (( failed )); then
        return 1
    fi
    echo "Doctor completed. Connect the camera and run capabilities before recording."
}

command_build() {
    local sdk_root="${ORBBEC_SDK_ROOT:-}"
    local preset="$DEFAULT_PRESET"
    local jobs=""
    local clean=0
    local preview=0
    local install_prefix=""
    local -a cmake_args=()
    while (( $# )); do
        case "$1" in
            --sdk-root) sdk_root="${2:-}"; shift 2 ;;
            --preset) preset="${2:-}"; shift 2 ;;
            --jobs) jobs="${2:-}"; shift 2 ;;
            --clean) clean=1; shift ;;
            --preview) preview=1; shift ;;
            --install-prefix) install_prefix="${2:-}"; shift 2 ;;
            --help|-h) usage; return 0 ;;
            *) die "Unknown build option: $1" ;;
        esac
    done
    [[ -f "$SOURCE_ROOT/CMakeLists.txt" ]] || die "The build command is only available from a source checkout."
    preset_python_version "$preset" >/dev/null
    validate_sdk_root "$sdk_root"
    require_cmake_version || die "Install CMake 3.24 or newer, then rerun build."
    local build_dir
    build_dir="$(build_dir_for_preset "$preset")"
    if (( clean )); then
        echo "Removing generated build directory: $build_dir"
        cmake -E rm -rf "$build_dir"
    fi
    local preview_option=OFF
    if (( preview )); then
        preview_option=ON
    fi
    (
        cd "$SOURCE_ROOT"
        cmake -S "$SOURCE_ROOT" -B "$build_dir" \
            -DISAAC_TELEOP_PYTHON_VERSION="$(preset_python_version "$preset")" \
            -DBUILD_VIZ=OFF \
            -DBUILD_PLUGIN_EGO_CAMERA=ON \
            -DORBBEC_SDK_ROOT="$sdk_root" \
            -DEGO_ENABLE_PREVIEW="$preview_option"
        local build_args=(--build "$build_dir" --target ego_camera_plugin ego_mcap_export_media ego_mcap_merge)
        if [[ -n "$jobs" ]]; then
            [[ "$jobs" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer."
            build_args+=(--parallel "$jobs")
        else
            build_args+=(--parallel)
        fi
        cmake "${build_args[@]}"
        if [[ -n "$install_prefix" ]]; then
            cmake --install "$build_dir" --prefix "$install_prefix" --component ego
        fi
    )
    echo "Build completed."
    echo "Plugin: $build_dir/src/plugins/ego_camera/ego_camera_plugin"
    echo "Next: $SCRIPT_DIR/ego.sh capabilities --preset $preset"
}

print_build_provenance() {
    local plugin="$1"
    local sdk_library=""
    echo "== Capture software provenance =="
    echo "Plugin path: $plugin"
    if command -v sha256sum >/dev/null 2>&1; then
        echo "Plugin SHA-256: $(sha256sum "$plugin" | awk '{print $1}')"
    else
        warn "sha256sum is unavailable; the plugin build hash was not recorded."
    fi
    if command -v ldd >/dev/null 2>&1; then
        sdk_library="$({ ldd "$plugin" 2>/dev/null || true; } \
            | awk '$1 ~ /^libOrbbecSDK[.]so/ && $2 == "=>" {print $3; exit}')"
    fi
    if [[ -n "$sdk_library" && -f "$sdk_library" ]]; then
        echo "Orbbec SDK library: $sdk_library"
        if command -v sha256sum >/dev/null 2>&1; then
            echo "Orbbec SDK library SHA-256: $(sha256sum "$sdk_library" | awk '{print $1}')"
        fi
    else
        echo "Orbbec SDK library: unresolved"
        warn "The runtime Orbbec SDK library path could not be resolved with ldd."
    fi
    echo "== Device-advertised capabilities =="
}

emit_capabilities() {
    local plugin="$1"
    local device_uid="${2:-}"
    local -a args=(--list-capabilities)
    [[ -n "$device_uid" ]] && args+=("--device-uid=$device_uid")
    print_build_provenance "$plugin"
    "$plugin" "${args[@]}"
}

command_capabilities() {
    local preset="$DEFAULT_PRESET"
    local plugin_path=""
    local device_uid=""
    while (( $# )); do
        case "$1" in
            --preset) preset="${2:-}"; shift 2 ;;
            --plugin) plugin_path="${2:-}"; shift 2 ;;
            --device-uid) device_uid="${2:-}"; shift 2 ;;
            --help|-h) usage; return 0 ;;
            *) die "Unknown capabilities option: $1" ;;
        esac
    done
    preset_python_version "$preset" >/dev/null
    local plugin
    plugin="$(resolve_plugin "$preset" "$plugin_path")"
    emit_capabilities "$plugin" "$device_uid"
}

record_usage_error() {
    die "record option error: $*. Run '$SCRIPT_DIR/ego.sh record --help' for usage."
}

stream_extension() {
    case "$1" in
        mjpg) echo "mjpg" ;;
        h264) echo "h264" ;;
        h265) echo "h265" ;;
        *) die "Unsupported format '$1'." ;;
    esac
}

write_requested_profile_manifest() {
    local output="$1"
    local format="$2"
    local width="$3"
    local height="$4"
    local fps="$5"
    local diagnostics="$6"
    printf '%s\n' \
        '{' \
        '  "version": 1,' \
        "  \"record_device_diagnostics\": \"$diagnostics\"," \
        '  "streams": {' \
        "    \"ColorLeft\": {\"format\": \"$format\", \"width\": $width, \"height\": $height, \"fps\": $fps}," \
        "    \"ColorRight\": {\"format\": \"$format\", \"width\": $width, \"height\": $height, \"fps\": $fps}" \
        '  }' \
        '}' > "$output"
}

record_config_python() {
    local candidate
    for candidate in python3 python3.13 python3.12 python3.11; do
        if command -v "$candidate" >/dev/null 2>&1 \
                && "$candidate" -c 'import tomllib' >/dev/null 2>&1; then
            command -v "$candidate"
            return 0
        fi
    done
    die "--config requires Python 3.11 or newer for TOML support."
}

read_record_config() {
    local config_file="$1"
    [[ -f "$config_file" ]] || record_usage_error "configuration file does not exist: $config_file"
    local config_python
    config_python="$(record_config_python)"
    "$config_python" "$SCRIPT_DIR/tools/record_config.py" "$config_file"
}

command_record() {
    local preset="$DEFAULT_PRESET"
    local plugin_path=""
    local duration=""
    local run_dir=""
    local format="h264"
    local width=1600
    local height=1300
    local fps=30
    local device_uid=""
    local reconnect_timeout=30
    local reconnect_interval_ms=1000
    local request_imu=1
    local require_calibration=0
    # Optional audio must be requested explicitly, not enabled by sensor discovery.
    local request_audio=0
    local preview=0
    local media_mode="sidecar"
    local keep_sidecars=0
    local record_diagnostics=on
    local -a passthrough=()
    local -a original_args=("$@")
    local config_file=""
    local index
    for (( index = 0; index < ${#original_args[@]}; ++index )); do
        [[ "${original_args[index]}" != "--" ]] || break
        if [[ "${original_args[index]}" == "--config" ]]; then
            (( index + 1 < ${#original_args[@]} )) || record_usage_error "--config requires a FILE.toml"
            [[ -z "$config_file" ]] || record_usage_error "--config may be specified only once"
            config_file="${original_args[index + 1]}"
            (( ++index ))
        fi
    done
    local -a config_args=()
    if [[ -n "$config_file" ]]; then
        local config_args_file
        config_args_file="$(mktemp)"
        if ! read_record_config "$config_file" > "$config_args_file"; then
            rm -f -- "$config_args_file"
            return 1
        fi
        mapfile -d '' -t config_args < "$config_args_file"
        rm -f -- "$config_args_file"
    fi
    set -- "${config_args[@]}" "${original_args[@]}"
    while (( $# )); do
        case "$1" in
            --config|--duration|--output|--format|--width|--height|--fps|--device-uid|--reconnect-timeout|--reconnect-interval-ms|--record-device-diagnostics|--mcap-media|--preset|--plugin|--plugin-option)
                (( $# >= 2 )) && [[ -n "$2" ]] || record_usage_error "$1 requires a value"
                ;;
        esac
        case "$1" in
            --) shift; passthrough+=("$@"); break ;;
            --config) shift 2 ;;
            --duration) duration="${2:-}"; shift 2 ;;
            --output) run_dir="${2:-}"; shift 2 ;;
            --format) format="${2:-}"; shift 2 ;;
            --width) width="${2:-}"; shift 2 ;;
            --height) height="${2:-}"; shift 2 ;;
            --fps) fps="${2:-}"; shift 2 ;;
            --device-uid) device_uid="${2:-}"; shift 2 ;;
            --reconnect-timeout) reconnect_timeout="${2:-}"; shift 2 ;;
            --reconnect-interval-ms) reconnect_interval_ms="${2:-}"; shift 2 ;;
            --imu) request_imu=1; shift ;;
            --no-imu) request_imu=0; shift ;;
            --require-calibration) require_calibration=1; shift ;;
            --audio) request_audio=1; shift ;;
            --no-audio) request_audio=0; shift ;;
            --preview) preview=1; shift ;;
            --no-preview) preview=0; shift ;;
            --record-device-diagnostics=*) record_diagnostics="${1#*=}"; shift ;;
            --record-device-diagnostics) record_diagnostics="${2:-}"; shift 2 ;;
            --mcap-media)
                media_mode="${2:-}"
                [[ "$media_mode" == "sidecar" || "$media_mode" == "embedded" ]] || record_usage_error "--mcap-media must be sidecar or embedded"
                shift 2
                ;;
            --keep-media-sidecars) keep_sidecars=1; shift ;;
            --no-keep-media-sidecars) keep_sidecars=0; shift ;;
            --preset) preset="${2:-}"; shift 2 ;;
            --plugin) plugin_path="${2:-}"; shift 2 ;;
            --plugin-option) passthrough+=("${2:-}"); shift 2 ;;
            --help|-h) usage; return 0 ;;
            *) record_usage_error "unknown option '$1'" ;;
        esac
    done
    preset_python_version "$preset" >/dev/null
    [[ "$format" == "mjpg" || "$format" == "h264" || "$format" == "h265" ]] || record_usage_error "--format must be mjpg, h264, or h265"
    [[ "$media_mode" == "sidecar" || "$media_mode" == "embedded" ]] || record_usage_error "--mcap-media must be sidecar or embedded"
    [[ "$width" =~ ^[1-9][0-9]*$ && "$height" =~ ^[1-9][0-9]*$ && "$fps" =~ ^[1-9][0-9]*$ ]] || record_usage_error "width, height, and fps must be positive integers"
    [[ "$reconnect_timeout" =~ ^[0-9]+$ ]] || record_usage_error "--reconnect-timeout must be a non-negative integer"
    [[ "$reconnect_interval_ms" =~ ^[1-9][0-9]*$ ]] || record_usage_error "--reconnect-interval-ms must be a positive integer"
    if [[ -n "$duration" ]]; then
        [[ "$duration" =~ ^[1-9][0-9]*$ ]] || record_usage_error "--duration must be a positive integer"
    fi
    if (( keep_sidecars )) && [[ "$media_mode" == "sidecar" ]]; then
        record_usage_error "--keep-media-sidecars is only valid with embedded media"
    fi

    # Native passthrough arguments win; record the effective policy, not the wrapper default.
    local option
    for option in "${passthrough[@]}"; do
        [[ "$option" != --record-device-diagnostics=* ]] || record_diagnostics="${option#*=}"
    done
    case "$record_diagnostics" in
        on|true|1) record_diagnostics=on ;;
        off|false|0) record_diagnostics=off ;;
        *) record_usage_error "--record-device-diagnostics must be on or off" ;;
    esac

    local plugin
    plugin="$(resolve_plugin "$preset" "$plugin_path")"
    local output_base="$SOURCE_ROOT"
    [[ -f "$SOURCE_ROOT/CMakeLists.txt" ]] || output_base="$PWD"
    if [[ -z "$run_dir" ]]; then
        run_dir="$output_base/recordings/ego_$(date +%Y%m%d_%H%M%S)"
    elif [[ "$run_dir" != /* ]]; then
        run_dir="$PWD/$run_dir"
    fi
    [[ ! -e "$run_dir" ]] || die "Output directory already exists: $run_dir"
    mkdir -p "$run_dir/logs"
    if [[ "$media_mode" == "sidecar" ]] || (( keep_sidecars )); then
        mkdir -p "$run_dir/raw"
    fi
    write_requested_profile_manifest "$run_dir/requested_profile.json" "$format" "$width" "$height" "$fps" "$record_diagnostics"

    local capabilities_file="$run_dir/capabilities.txt"
    echo "Inspecting connected device before recording..."
    emit_capabilities "$plugin" "$device_uid" | tee "$capabilities_file"
    grep -q '^Sensor LeftColor$' "$capabilities_file" || die "Connected device has no ColorLeft sensor."
    grep -q '^Sensor RightColor$' "$capabilities_file" || die "Connected device has no ColorRight sensor."

    local enable_imu=0
    local enable_audio=0
    if (( request_imu )); then
        if grep -q '^Sensor Accel$' "$capabilities_file" && grep -q '^Sensor Gyro$' "$capabilities_file"; then
            enable_imu=1
        else
            warn "Accel/Gyro are unavailable; continuing without IMU. Use --no-imu to silence this warning."
        fi
    fi
    if (( request_audio )); then
        if grep -q '^Sensor Audio$' "$capabilities_file"; then
            enable_audio=1
        else
            warn "Audio is unavailable; continuing without audio. Use --no-audio to silence this warning."
        fi
    fi

    local extension
    extension="$(stream_extension "$format")"
    local -a plugin_args=(
        "--mcap-filename=$run_dir/metadata.mcap"
        "--mcap-media=$media_mode"
        "--record-device-diagnostics=$record_diagnostics"
        "--calibration-output=$run_dir/calibration.json"
        "--reconnect-timeout=$reconnect_timeout"
        "--reconnect-interval-ms=$reconnect_interval_ms"
    )
    if [[ "$media_mode" == "sidecar" ]] || (( keep_sidecars )); then
        plugin_args+=(
            "--add-stream=camera=ColorLeft,output=$run_dir/raw/ColorLeft.$extension,format=$format,width=$width,height=$height,fps=$fps"
            "--add-stream=camera=ColorRight,output=$run_dir/raw/ColorRight.$extension,format=$format,width=$width,height=$height,fps=$fps"
        )
    else
        plugin_args+=(
            "--add-stream=camera=ColorLeft,format=$format,width=$width,height=$height,fps=$fps"
            "--add-stream=camera=ColorRight,format=$format,width=$width,height=$height,fps=$fps"
        )
    fi
    [[ -n "$device_uid" ]] && plugin_args+=("--device-uid=$device_uid")
    (( preview )) && plugin_args+=(--preview)
    (( enable_imu )) && plugin_args+=(--enable-imu --imu-rate=1000)
    if (( enable_audio )); then
        if [[ "$media_mode" == "sidecar" ]] || (( keep_sidecars )); then
            plugin_args+=("--audio-output=$run_dir/Audio.wav")
        else
            plugin_args+=(--enable-audio)
        fi
    fi
    (( keep_sidecars )) && plugin_args+=(--keep-media-sidecars)
    plugin_args+=("${passthrough[@]}")

    echo "Recording directory: $run_dir"
    echo "Requested exact stereo profile: ${width}x${height}@${fps} ${format}; no profile fallback will be attempted."
    echo "Stop with Ctrl-C once; the plugin will finalize media and MCAP files."
    local capture_status
    local -a pipeline_statuses=()
    # Catch INT without propagating SIG_IGN to the plugin. Keep tee alive
    # until shutdown logs and MCAP finalization have completed.
    trap ':' INT
    set +e
    if [[ -n "$duration" ]]; then
        timeout --foreground --preserve-status --signal=INT --kill-after=30s "$duration" \
            "$plugin" "${plugin_args[@]}" 2>&1 | tee -i "$run_dir/logs/capture.log"
    else
        "$plugin" "${plugin_args[@]}" 2>&1 | tee -i "$run_dir/logs/capture.log"
    fi
    pipeline_statuses=("${PIPESTATUS[@]}")
    trap - INT
    set -e
    capture_status=${pipeline_statuses[0]}
    [[ "${pipeline_statuses[1]}" -eq 0 ]] || die "Unable to write capture log: $run_dir/logs/capture.log"
    if [[ "$capture_status" -ne 0 ]]; then
        die "Capture failed with status $capture_status. See $run_dir/logs/capture.log"
    fi
    if [[ -n "$duration" ]]; then
        echo "Timed capture command completed; SIGINT was scheduled after $duration seconds."
    fi
    local -a verify_args=("$run_dir" --preset "$preset" --plugin "$plugin" --media-mode "$media_mode")
    (( enable_imu )) && verify_args+=(--expect-imu)
    (( enable_audio )) && verify_args+=(--expect-audio)
    (( require_calibration )) && verify_args+=(--require-calibration)
    command_verify "${verify_args[@]}"
}

find_media_file() {
    local directory="$1"
    local stream="$2"
    local candidate
    for candidate in "$directory/$stream.h264" "$directory/$stream.h265" "$directory/$stream.mjpg"; do
        if [[ -s "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

check_capture_health_log() {
    local capture_log="$1"
    if [[ ! -s "$capture_log" ]]; then
        warn "No capture log is available; sequence-gap and queue-drop checks were skipped."
        return 0
    fi
    if grep -Eq '[[:space:]][1-9][0-9]* sequence gaps' "$capture_log"; then
        die "Capture log reports a video sequence gap: $capture_log"
    fi
    if grep -Eq '(^|[[:space:]])dropped=[1-9][0-9]*([[:space:]]|$)' "$capture_log"; then
        die "Capture log reports dropped metadata events: $capture_log"
    fi
    if grep -Eq 'video_frame_sets_dropped=[1-9][0-9]*([[:space:]]|$)' "$capture_log"; then
        die "Capture log reports dropped video frame sets: $capture_log"
    fi
    if grep -Eq 'sequence gaps|video_frame_sets_dropped=' "$capture_log"; then
        echo "Capture log sequence-gap and queue-drop counters: OK"
    else
        warn "Capture log contains no statistics counters; inspect it before accepting the run."
    fi
}

checked_media_command() {
    local errors
    errors="$(mktemp "$run_dir/.verify_media.XXXXXX")" || die "Could not create media check log"
    if "$@" 2>"$errors"; then
        if [[ ! -s "$errors" ]]; then
            rm -f -- "$errors"
            return 0
        fi
    fi
    # FFmpeg can emit a decoder error yet exit zero; its stderr is part of the result.
    cat "$errors" >&2
    rm -f -- "$errors"
    die "Media check failed: $1"
}

video_signature() {
    local media_file="$1"
    local -a input_args=()
    [[ "$media_file" == *.mjpg ]] && input_args=(-f mjpeg)
    checked_media_command ffprobe -v error "${input_args[@]}" -select_streams v:0 \
        -show_entries stream=codec_name,width,height,r_frame_rate \
        -of csv=p=0 "$media_file"
}

video_frame_count() {
    local media_file="$1"
    local -a input_args=()
    [[ "$media_file" == *.mjpg ]] && input_args=(-f mjpeg)
    checked_media_command ffprobe -v error "${input_args[@]}" -count_frames -select_streams v:0 \
        -show_entries stream=nb_read_frames -of default=noprint_wrappers=1:nokey=1 "$media_file"
}

decode_video() {
    local stream_name="$1"
    local media_file="$2"
    local progress
    local -a input_args=()
    [[ "$media_file" == *.mjpg ]] && input_args=(-f mjpeg)
    progress="$(checked_media_command ffmpeg -v error -xerror -err_detect explode \
        "${input_args[@]}" -i "$media_file" -progress pipe:1 -nostats -f null -)" || return 1
    decoded_video_frame_count="$(awk -F= '$1 == "frame" { count = $2 } END { print count }' <<< "$progress")"
    [[ "$decoded_video_frame_count" =~ ^[1-9][0-9]*$ ]] || die "No complete decoded $stream_name frames"
    grep -Fx 'progress=end' <<< "$progress" >/dev/null || die "Incomplete $stream_name decode"
    echo "$stream_name complete decode: OK"
}

verify_requested_profile_manifest() {
    local manifest="$1"
    local left_signature="$2"
    local right_signature="$3"
    python3 "$SCRIPT_DIR/tools/verify_profile.py" "$manifest" "$left_signature" "$right_signature"
}

verify_mcap_video_timing() {
    local mcap="$1"
    local mode="$2"
    local left_frames="$3"
    local right_frames="$4"
    local manifest="$5"
    run_mcap_tool "$SCRIPT_DIR/tools/verify_video_timing.py" "$mcap" "$mode" "$left_frames" "$right_frames" "$manifest"
}

print_fingerprint() {
    local label="$1" media_file="$2" checksum
    checksum="$(sha256sum -- "$media_file")" || die "Could not hash $media_file"
    checksum="${checksum%% *}"
    [[ "$checksum" =~ ^[0-9a-f]{64}$ ]] || die "Invalid SHA-256 fingerprint for $media_file"
    echo "$label SHA-256: $checksum"
}

command_verify_impl() {
    local run_dir="${1:-}"
    shift || true
    [[ -n "$run_dir" ]] || die "verify requires a RUN_DIRECTORY"
    [[ -d "$run_dir" ]] || die "Run directory does not exist: $run_dir"
    local preset="$DEFAULT_PRESET"
    local plugin_path=""
    local expected_mode=""
    local expect_imu=0
    local expect_audio=0
    local require_calibration=0
    while (( $# )); do
        case "$1" in
            --preset) preset="${2:-}"; shift 2 ;;
            --plugin) plugin_path="${2:-}"; shift 2 ;;
            --media-mode)
                expected_mode="${2:-}"
                [[ "$expected_mode" == "sidecar" || "$expected_mode" == "embedded" ]] || die "Unknown MCAP media mode '$expected_mode'"
                shift 2
                ;;
            --expect-imu) expect_imu=1; shift ;;
            --expect-audio) expect_audio=1; shift ;;
            --require-calibration) require_calibration=1; shift ;;
            --help|-h) usage; return 0 ;;
            *) die "Unknown verify option: $1" ;;
        esac
    done
    preset_python_version "$preset" >/dev/null
    run_dir="$(cd "$run_dir" && pwd)"
    local mcap="$run_dir/metadata.mcap"
    [[ -s "$mcap" ]] || die "Missing or empty MCAP: $mcap"
    local partial_file
    partial_file="$(find "$run_dir" -name '*.partial' -print -quit)"
    [[ -z "$partial_file" ]] || die "Incomplete capture: found $partial_file"
    require_command ffmpeg "Install it with: sudo apt update && sudo apt install -y ffmpeg" || exit 1
    require_command ffprobe "Install it with: sudo apt update && sudo apt install -y ffmpeg" || exit 1
    require_command python3 "Install it with: sudo apt update && sudo apt install -y python3" || exit 1
    python_has_mcap || command -v uv >/dev/null 2>&1 || die "verify needs the Python MCAP reader. Install uv, or install mcap into the active Python environment."

    local mode="$expected_mode"
    if [[ -z "$mode" ]]; then
        mode="$(run_mcap_tool "$SCRIPT_DIR/tools/detect_media_mode.py" "$mcap")"
    fi
    [[ "$mode" == "sidecar" || "$mode" == "embedded" ]] || die "Unknown MCAP media mode '$mode'"

    local -a topic_check_args=("$mcap" "$mode" "--requested-profile=$run_dir/requested_profile.json")
    (( expect_imu )) && topic_check_args+=(--expect-imu)
    (( expect_audio )) && topic_check_args+=(--expect-audio)
    if (( require_calibration )); then
        topic_check_args+=(--require-calibration "--calibration-json=$run_dir/calibration.json")
    fi
    run_mcap_tool "$SCRIPT_DIR/tools/verify_sensor_topics.py" "${topic_check_args[@]}"

    check_capture_health_log "$run_dir/logs/capture.log"

    local media_dir="$run_dir/raw"
    local retained_media_dir=""
    local temporary_export=""
    if [[ "$mode" == "embedded" ]]; then
        [[ -d "$media_dir" ]] && retained_media_dir="$media_dir"
        temporary_export="$(mktemp -d "$run_dir/.verify_export.XXXXXX")"
        # Verification failures exit from deep helper calls; EXIT cleanup prevents a
        # multi-gigabyte embedded export from being stranded beside the recording.
        local cleanup_command
        printf -v cleanup_command 'rm -rf -- %q' "$temporary_export"
        trap "$cleanup_command" EXIT
        local exporter
        exporter="$(resolve_exporter "$preset" "$plugin_path")" || return 1
        "$exporter" "$mcap" "$temporary_export"
        media_dir="$temporary_export"
    fi
    local left right
    left="$(find_media_file "$media_dir" ColorLeft)" || die "Could not find exported ColorLeft media in $media_dir"
    right="$(find_media_file "$media_dir" ColorRight)" || die "Could not find exported ColorRight media in $media_dir"
    if [[ -n "$retained_media_dir" ]]; then
        local retained_left retained_right
        retained_left="$(find_media_file "$retained_media_dir" ColorLeft)" \
            || die "Could not find retained ColorLeft media in $retained_media_dir"
        retained_right="$(find_media_file "$retained_media_dir" ColorRight)" \
            || die "Could not find retained ColorRight media in $retained_media_dir"
        cmp -s "$left" "$retained_left" || die "Embedded and retained ColorLeft media differ"
        cmp -s "$right" "$retained_right" || die "Embedded and retained ColorRight media differ"
        echo "Embedded video matches retained media sidecars: OK"
    fi
    local decoded_video_frame_count left_frames right_frames
    decode_video ColorLeft "$left"
    left_frames="$decoded_video_frame_count"
    decode_video ColorRight "$right"
    right_frames="$decoded_video_frame_count"
    local left_signature right_signature left_probed_frames right_probed_frames
    left_signature="$(video_signature "$left")"
    right_signature="$(video_signature "$right")"
    [[ -n "$left_signature" ]] || die "ffprobe found no ColorLeft video stream: $left"
    [[ -n "$right_signature" ]] || die "ffprobe found no ColorRight video stream: $right"
    [[ "$left_signature" == "$right_signature" ]] \
        || die "Stereo media profiles differ: ColorLeft=$left_signature ColorRight=$right_signature"
    local requested_profile_manifest="$run_dir/requested_profile.json"
    if [[ -e "$requested_profile_manifest" ]]; then
        verify_requested_profile_manifest "$requested_profile_manifest" "$left_signature" "$right_signature"
    else
        warn "No requested_profile.json exists; exact requested-versus-recorded profile checks were skipped for this legacy run."
    fi
    left_probed_frames="$(video_frame_count "$left")"
    right_probed_frames="$(video_frame_count "$right")"
    [[ "$left_probed_frames" == "$left_frames" ]] || die "ColorLeft FFprobe count differs from complete decode: $left_probed_frames versus $left_frames"
    [[ "$right_probed_frames" == "$right_frames" ]] || die "ColorRight FFprobe count differs from complete decode: $right_probed_frames versus $right_frames"
    echo "Stereo elementary-stream profile (codec,width,height,ffprobe_rate): $left_signature"
    echo "Decoded frames: ColorLeft=$left_frames ColorRight=$right_frames"
    if [[ "$left_frames" != "$right_frames" ]]; then
        warn "Stereo decoded counts differ at their initial IDR boundaries; per-stream MCAP counts and cadence must still pass."
    fi
    verify_mcap_video_timing \
        "$mcap" "$mode" "$left_frames" "$right_frames" "$requested_profile_manifest"
    local wav=""
    if [[ "$mode" == "embedded" ]]; then
        wav="$media_dir/Audio.wav"
        if [[ -s "$run_dir/Audio.wav" ]]; then
            [[ -s "$wav" ]] || die "Embedded MCAP is missing audio retained as $run_dir/Audio.wav"
            cmp -s "$wav" "$run_dir/Audio.wav" || die "Embedded and retained PCM audio differ"
            echo "Embedded audio matches retained WAV sidecar: OK"
        fi
    else
        wav="$run_dir/Audio.wav"
    fi
    if [[ -s "$wav" ]]; then
        local wav_signature
        wav_signature="$(checked_media_command ffprobe -v error \
            -show_entries stream=codec_name,sample_rate,channels,bits_per_sample \
            -of default=noprint_wrappers=1 "$wav")"
        echo "$wav_signature"
        if (( expect_audio )); then
            grep -Fx 'codec_name=pcm_s16le' <<< "$wav_signature" >/dev/null \
                || die "Expected PCM S16_LE audio in $wav"
            grep -Fx 'sample_rate=48000' <<< "$wav_signature" >/dev/null \
                || die "Expected 48000 Hz audio in $wav"
            grep -Fx 'channels=1' <<< "$wav_signature" >/dev/null \
                || die "Expected mono audio in $wav"
            grep -Fx 'bits_per_sample=16' <<< "$wav_signature" >/dev/null \
                || die "Expected 16-bit audio in $wav"
        fi
    elif (( expect_audio )); then
        die "Audio was requested but no WAV sidecar or exported embedded audio exists."
    else
        warn "No WAV sidecar exists. This is expected only when audio was unavailable, disabled, or embedded without sidecars."
    fi
    echo "Delivery file: $mcap"
    if command -v sha256sum >/dev/null 2>&1; then
        # Hash while exported media still exists, and reject read failures.
        print_fingerprint Delivery "$mcap"
        print_fingerprint ColorLeft "$left"
        print_fingerprint ColorRight "$right"
        [[ ! -s "$wav" ]] || print_fingerprint Audio "$wav"
    else
        warn "sha256sum is unavailable; media fingerprints were not added to the report."
    fi
    if [[ -n "$temporary_export" ]]; then
        rm -rf -- "$temporary_export"
        temporary_export=""
        trap - EXIT
    fi
    echo "Verification passed: $run_dir"
}

command_verify() {
    local run_dir="${1:-}"
    if [[ ! -d "$run_dir" ]]; then
        command_verify_impl "$@"
        return
    fi

    run_dir="$(cd "$run_dir" && pwd)"
    local report="$run_dir/verification_report.txt"
    local failed_report="$run_dir/verification_report.failed.txt"
    local working_report="$run_dir/.verification_report.txt.inprogress"
    local -a pipeline_status

    rm -f -- "$report" || die "Could not invalidate the previous success report: $report"
    set +e
    {
        set -e
        echo "Ego MCAP consistency verification report"
        echo "Generated (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "Run directory: $run_dir"
        echo
        command_verify_impl "$run_dir" "${@:2}"
    } 2>&1 | tee "$working_report"
    pipeline_status=("${PIPESTATUS[@]}")
    set -e

    if (( pipeline_status[1] != 0 )); then
        die "Could not write consistency report: $working_report"
    fi
    if (( pipeline_status[0] != 0 )); then
        mv -f -- "$working_report" "$failed_report"
        echo "Failed consistency report: $failed_report" >&2
        return "${pipeline_status[0]}"
    fi

    mv -f -- "$working_report" "$report"
    echo "Consistency report: $report"
}

command_export_media() {
    local run_dir="${1:-}"
    shift || true
    [[ -n "$run_dir" ]] || die "export-media requires a RUN_DIRECTORY"
    [[ -d "$run_dir" ]] || die "Run directory does not exist: $run_dir"
    run_dir="$(cd "$run_dir" && pwd)"
    local preset="$DEFAULT_PRESET"
    local output_dir="$run_dir/exported"
    while (( $# )); do
        case "$1" in
            --output) output_dir="${2:-}"; shift 2 ;;
            --preset) preset="${2:-}"; shift 2 ;;
            --help|-h) usage; return 0 ;;
            *) die "Unknown export-media option: $1" ;;
        esac
    done
    preset_python_version "$preset" >/dev/null
    [[ -n "$output_dir" ]] || die "--output requires a DIRECTORY"
    [[ "$output_dir" = /* ]] || output_dir="$PWD/$output_dir"
    local mcap="$run_dir/metadata.mcap"
    [[ -s "$mcap" ]] || die "Missing or empty MCAP: $mcap"
    [[ ! -e "$mcap.partial" ]] || die "Incomplete capture: found $mcap.partial"
    [[ ! -e "$output_dir" ]] || die "Export output already exists: $output_dir"
    local exporter
    exporter="$(resolve_exporter "$preset")" || return 1
    "$exporter" "$mcap" "$output_dir"
    echo "Export completed: $output_dir"
}

main() {
    local command="${1:-}"
    [[ -n "$command" ]] || { usage; return 1; }
    shift
    case "$command" in
        doctor) command_doctor "$@" ;;
        build) command_build "$@" ;;
        capabilities) command_capabilities "$@" ;;
        record) command_record "$@" ;;
        verify) command_verify "$@" ;;
        export-media) command_export_media "$@" ;;
        --help|-h|help) usage ;;
        *) die "Unknown command '$command'." ;;
    esac
}

main "$@"
