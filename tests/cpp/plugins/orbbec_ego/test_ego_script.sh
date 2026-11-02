#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT="${1:?Pass the plugin shell entry point}"
TMPDIR_TEST="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_TEST"' EXIT

expect_failure() {
    if "$@" >/dev/null 2>&1; then
        echo "Expected command to fail: $*" >&2
        exit 1
    fi
}

expect_failure_message() {
    local expected="$1"
    shift
    local output
    if output="$("$@" 2>&1)"; then
        echo "Expected command to fail: $*" >&2
        exit 1
    fi
    if ! grep -F -- "$expected" <<< "$output" >/dev/null; then
        printf 'Expected failure containing %s: %s\n%s\n' "$expected" "$*" "$output" >&2
        exit 1
    fi
}

"$SCRIPT" --help >/dev/null
expect_failure "$SCRIPT" doctor --sdk-root "$TMPDIR_TEST/missing-sdk"
expect_failure "$SCRIPT" build --sdk-root "$TMPDIR_TEST/missing-sdk" --preset invalid
expect_failure "$SCRIPT" record --duration 0
expect_failure "$SCRIPT" record --format invalid
expect_failure "$SCRIPT" record --reconnect-timeout invalid
expect_failure "$SCRIPT" record --reconnect-interval-ms 0

MOCK_PLUGIN="$TMPDIR_TEST/mock_plugin"
MOCK_ARGS="$TMPDIR_TEST/plugin_args"
export MOCK_ARGS
cat > "$MOCK_PLUGIN" <<'EOF'
#!/usr/bin/env bash
if [[ "$1" == "--list-capabilities" ]]; then
    printf '%s\n' "$@" > "$MOCK_ARGS"
    cat <<'CAPABILITIES'
Sensor Accel
Sensor Gyro
Sensor LeftColor
Sensor RightColor
CAPABILITIES
    exit 0
fi
printf '%s\n' "$@" > "$MOCK_ARGS"
exit 7
EOF
chmod +x "$MOCK_PLUGIN"

CAPABILITY_OUTPUT="$($SCRIPT capabilities --plugin "$MOCK_PLUGIN" 2>/dev/null)"
grep -F "== Capture software provenance ==" <<< "$CAPABILITY_OUTPUT" >/dev/null
grep -F "== Device-advertised capabilities ==" <<< "$CAPABILITY_OUTPUT" >/dev/null
grep -Fx "Sensor LeftColor" <<< "$CAPABILITY_OUTPUT" >/dev/null
"$SCRIPT" capabilities --plugin "$MOCK_PLUGIN" --device-uid ego-test >/dev/null 2>&1
grep -Fx -- "--device-uid=ego-test" "$MOCK_ARGS" >/dev/null

RUN="$TMPDIR_TEST/recording"
expect_failure "$SCRIPT" record --plugin "$MOCK_PLUGIN" --duration 1 --output "$RUN" -- --bitrate=8
grep -Fx -- "--enable-imu" "$MOCK_ARGS" >/dev/null
grep -Fx -- "--mcap-filename=$RUN/metadata.mcap" "$MOCK_ARGS" >/dev/null
grep -F -- "output=$RUN/raw/ColorLeft.h264" "$MOCK_ARGS" >/dev/null
grep -F -- "output=$RUN/raw/ColorRight.h264" "$MOCK_ARGS" >/dev/null
grep -Fx -- "--reconnect-timeout=30" "$MOCK_ARGS" >/dev/null
grep -Fx -- "--reconnect-interval-ms=1000" "$MOCK_ARGS" >/dev/null
grep -Fx -- "--bitrate=8" "$MOCK_ARGS" >/dev/null
test -f "$RUN/capabilities.txt"
test -f "$RUN/logs/capture.log"
grep -F "Plugin SHA-256:" "$RUN/capabilities.txt" >/dev/null
python3 - "$RUN/requested_profile.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    manifest = json.load(stream)
assert manifest == {
    "version": 1,
    "record_device_diagnostics": "on",
    "streams": {
        "ColorLeft": {"format": "h264", "width": 1600, "height": 1300, "fps": 30},
        "ColorRight": {"format": "h264", "width": 1600, "height": 1300, "fps": 30},
    },
}
PY

CONFIG_RUN="$TMPDIR_TEST/config_recording"
CONFIG_FILE="$TMPDIR_TEST/record.toml"
cat > "$CONFIG_FILE" <<EOF
[record]
duration = 9
output = "$TMPDIR_TEST/config_output_from_file"
format = "h265"
width = 1600
height = 1200
fps = 30
device_uid = "ego-config"
reconnect_timeout = 45
reconnect_interval_ms = 250
imu = false
preview = true
plugin = "$MOCK_PLUGIN"
plugin_options = ["--bitrate=8", "--dynamic-bitrate=on"]
EOF
expect_failure "$SCRIPT" record --config "$CONFIG_FILE" --duration 1 \
    --output "$CONFIG_RUN" --format h264
grep -Fx -- "--device-uid=ego-config" "$MOCK_ARGS" >/dev/null
grep -Fx -- "--reconnect-timeout=45" "$MOCK_ARGS" >/dev/null
grep -Fx -- "--reconnect-interval-ms=250" "$MOCK_ARGS" >/dev/null
grep -Fx -- "--preview" "$MOCK_ARGS" >/dev/null
grep -Fx -- "--bitrate=8" "$MOCK_ARGS" >/dev/null
grep -Fx -- "--dynamic-bitrate=on" "$MOCK_ARGS" >/dev/null
grep -F "format=h264,width=1600,height=1200,fps=30" "$MOCK_ARGS" >/dev/null
if grep -Fxq -- "--enable-imu" "$MOCK_ARGS"; then
    echo "Config/CLI stream overrides were not applied" >&2
    exit 1
fi

INVALID_CONFIG="$TMPDIR_TEST/invalid.toml"
cat > "$INVALID_CONFIG" <<'EOF'
[record]
unknown_option = true
EOF
expect_failure "$SCRIPT" record --config "$INVALID_CONFIG"
expect_failure "$SCRIPT" record --config "$TMPDIR_TEST/missing.toml"

SIGNAL_PLUGIN="$TMPDIR_TEST/signal_plugin"
cat > "$SIGNAL_PLUGIN" <<'EOF'
#!/usr/bin/env bash
if [[ "$1" == "--list-capabilities" ]]; then
    printf '%s\n' 'Sensor LeftColor' 'Sensor RightColor'
    exit 0
fi
trap 'echo "controlled shutdown failed"; exit 7' INT
while true; do
    sleep 0.1
done
EOF
chmod +x "$SIGNAL_PLUGIN"
set +e
SIGNAL_OUTPUT="$($SCRIPT record --plugin "$SIGNAL_PLUGIN" --duration 1 \
    --output "$TMPDIR_TEST/signal_recording" --no-imu 2>&1)"
SIGNAL_STATUS=$?
set -e
[[ "$SIGNAL_STATUS" -ne 0 ]]
grep -F "Capture failed with status 7" <<< "$SIGNAL_OUTPUT" >/dev/null

PACKAGE_DIR="$TMPDIR_TEST/package/plugins/orbbec_ego_camera"
MOCK_BIN="$TMPDIR_TEST/bin"
mkdir -p "$PACKAGE_DIR/extensions" "$MOCK_BIN"
cp "$SCRIPT" "$PACKAGE_DIR/orbbec_ego.sh"
cp -r "$(dirname "$SCRIPT")/tools" "$PACKAGE_DIR/tools"
chmod +x "$PACKAGE_DIR/orbbec_ego.sh"
touch "$PACKAGE_DIR/libOrbbecSDK.so.2" "$PACKAGE_DIR/OrbbecSDKConfig.xml"

cat > "$PACKAGE_DIR/orbbec_ego_camera_plugin" <<'EOF'
#!/usr/bin/env bash
if [[ "$1" == "--help" ]]; then
    exit 0
fi
exit 1
EOF
chmod +x "$PACKAGE_DIR/orbbec_ego_camera_plugin"

for command_name in python3 ffmpeg ffprobe; do
    cat > "$MOCK_BIN/$command_name" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
    chmod +x "$MOCK_BIN/$command_name"
done

# Exercise both vendor callback layouts and reject a structurally valid ordinary SDK.
SDK_FIXTURE="$TMPDIR_TEST/sdk"
mkdir -p "$SDK_FIXTURE/include/libobsensor/hpp" "$SDK_FIXTURE/lib/extensions"
touch "$SDK_FIXTURE/lib/OrbbecSDKConfig.cmake" "$SDK_FIXTURE/lib/libOrbbecSDK.so.2"
for layout in legacy acquisition ordinary; do
    cat > "$SDK_FIXTURE/include/libobsensor/ObSensor.hpp" <<'EOF'
#pragma once
#include <functional>
enum OBPropertyID { property_id };
enum OBDataTranState { transfer_state };
struct OBDataChunk {};
struct OBEgoStateReport {};
namespace ob {
class Device {
public:
    void getRawData(OBPropertyID, std::function<void(OBDataTranState, OBDataChunk*)>) const;
EOF
    case "$layout" in
        legacy)
            echo 'void setEgoStateCallback(std::function<void(const OBEgoStateReport&)>);' \
                >> "$SDK_FIXTURE/include/libobsensor/ObSensor.hpp"
            ;;
        acquisition)
            cat > "$SDK_FIXTURE/include/libobsensor/hpp/DataAcquisitionDevice.hpp" <<'EOF'
#pragma once
namespace ob {
class DataAcquisitionDevice : public Device {
public:
    void setEgoStateCallback(std::function<void(const OBEgoStateReport&)>);
};
}
EOF
            ;;
        ordinary)
            rm "$SDK_FIXTURE/include/libobsensor/hpp/DataAcquisitionDevice.hpp"
            ;;
    esac
    echo '}; }' >> "$SDK_FIXTURE/include/libobsensor/ObSensor.hpp"
    if [[ "$layout" == acquisition ]]; then
        echo '#include "hpp/DataAcquisitionDevice.hpp"' >> "$SDK_FIXTURE/include/libobsensor/ObSensor.hpp"
    fi
    if [[ "$layout" == ordinary ]]; then
        if PATH="$MOCK_BIN:$PATH" "$PACKAGE_DIR/orbbec_ego.sh" doctor --sdk-root "$SDK_FIXTURE" \
                > "$TMPDIR_TEST/sdk_failure.log" 2>&1; then
            echo 'Ordinary SDK without EGO callbacks must fail doctor' >&2
            exit 1
        fi
        grep -F 'SDK lacks the required EGO APIs' "$TMPDIR_TEST/sdk_failure.log" >/dev/null
        grep -F 'v2.10.6-EGO-20261019' "$TMPDIR_TEST/sdk_failure.log" >/dev/null
    else
        CXX='c++ -O2' PATH="$MOCK_BIN:$PATH" "$PACKAGE_DIR/orbbec_ego.sh" doctor --sdk-root "$SDK_FIXTURE" >/dev/null
    fi
done

PATH="$MOCK_BIN:$PATH" "$PACKAGE_DIR/orbbec_ego.sh" doctor >/dev/null

PARTIAL_RUN="$TMPDIR_TEST/partial_run"
mkdir -p "$PARTIAL_RUN"
printf '\0' > "$PARTIAL_RUN/metadata.mcap"
touch "$PARTIAL_RUN/metadata.mcap.partial"
expect_failure "$SCRIPT" verify "$PARTIAL_RUN"

VERIFY_BIN="$TMPDIR_TEST/verify_bin"
VERIFY_RUN="$TMPDIR_TEST/verify_with_sidecars"
VERIFY_FFMPEG_ARGS="$TMPDIR_TEST/verify_ffmpeg_args"
mkdir -p "$VERIFY_BIN" "$VERIFY_RUN/raw" "$VERIFY_RUN/logs"
cp "$MOCK_BIN/python3" "$VERIFY_BIN/"
cat > "$VERIFY_BIN/ffmpeg" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$@" >> "$VERIFY_FFMPEG_ARGS"
if [[ "${VERIFY_ZERO_EXIT_ERROR:-0}" == 1 ]]; then
    echo 'decoder error with a zero exit status' >&2
fi
printf '%s\n' 'frame=1' 'progress=end'
exit 0
EOF
cat > "$VERIFY_BIN/ffprobe" <<'EOF'
#!/usr/bin/env bash
for argument in "$@"; do
    if [[ "$argument" == "-count_frames" ]]; then
        echo 1
        exit 0
    fi
done
echo 'h264,1600,1300,30/1'
EOF
chmod +x "$VERIFY_BIN/ffmpeg" "$VERIFY_BIN/ffprobe"
printf '\0' > "$VERIFY_RUN/metadata.mcap"
printf '\0' > "$VERIFY_RUN/raw/ColorLeft.h264"
printf '\0' > "$VERIFY_RUN/raw/ColorRight.h264"
printf '%s\n' \
    '  ColorLeft: 1 frames, 1 bytes, 0 sequence gaps' \
    '  ColorRight: 1 frames, 1 bytes, 0 sequence gaps' \
    '  IMU: accel=0 gyro=0 samples; queue_peak=0 dropped=0 video_frame_sets_dropped=0' \
    > "$VERIFY_RUN/logs/capture.log"
PATH="$VERIFY_BIN:$PATH" VERIFY_FFMPEG_ARGS="$VERIFY_FFMPEG_ARGS" \
    "$SCRIPT" verify "$VERIFY_RUN" >/dev/null 2>&1
VERIFY_ZERO_ERROR_RUN="$TMPDIR_TEST/verify_zero_exit_error"
cp -r "$VERIFY_RUN" "$VERIFY_ZERO_ERROR_RUN"
expect_failure env PATH="$VERIFY_BIN:$PATH" \
    VERIFY_FFMPEG_ARGS="$VERIFY_FFMPEG_ARGS" VERIFY_ZERO_EXIT_ERROR=1 \
    "$SCRIPT" verify "$VERIFY_ZERO_ERROR_RUN"
[[ ! -e "$VERIFY_ZERO_ERROR_RUN/verification_report.txt" ]]
grep -F 'decoder error with a zero exit status' "$VERIFY_ZERO_ERROR_RUN/verification_report.failed.txt" >/dev/null
test -s "$VERIFY_RUN/verification_report.txt"
grep -F "Verification passed:" "$VERIFY_RUN/verification_report.txt" >/dev/null
grep -E '^ColorLeft SHA-256: [0-9a-f]{64}$' "$VERIFY_RUN/verification_report.txt" >/dev/null
grep -E '^ColorRight SHA-256: [0-9a-f]{64}$' "$VERIFY_RUN/verification_report.txt" >/dev/null
grep -Fx -- "-xerror" "$VERIFY_FFMPEG_ARGS" >/dev/null
grep -Fx -- "-err_detect" "$VERIFY_FFMPEG_ARGS" >/dev/null
grep -Fx -- "explode" "$VERIFY_FFMPEG_ARGS" >/dev/null

VERIFY_FAIL_BIN="$TMPDIR_TEST/verify_fail_bin"
VERIFY_FAIL_RUN="$TMPDIR_TEST/verify_decode_failure"
mkdir -p "$VERIFY_FAIL_BIN" "$VERIFY_FAIL_RUN/raw" "$VERIFY_FAIL_RUN/logs"
cp "$MOCK_BIN/python3" "$VERIFY_BIN/ffprobe" "$VERIFY_FAIL_BIN/"
cat > "$VERIFY_FAIL_BIN/ffmpeg" <<'EOF'
#!/usr/bin/env bash
exit 9
EOF
chmod +x "$VERIFY_FAIL_BIN/ffmpeg"
printf '\0' > "$VERIFY_FAIL_RUN/metadata.mcap"
printf '\0' > "$VERIFY_FAIL_RUN/raw/ColorLeft.h264"
printf '\0' > "$VERIFY_FAIL_RUN/raw/ColorRight.h264"
printf '%s\n' \
    '  ColorLeft: 1 frames, 1 bytes, 0 sequence gaps' \
    '  ColorRight: 1 frames, 1 bytes, 0 sequence gaps' \
    '  IMU: accel=0 gyro=0 samples; queue_peak=0 dropped=0 video_frame_sets_dropped=0' \
    > "$VERIFY_FAIL_RUN/logs/capture.log"
expect_failure env PATH="$VERIFY_FAIL_BIN:$PATH" \
    "$SCRIPT" verify "$VERIFY_FAIL_RUN"
test -s "$VERIFY_FAIL_RUN/verification_report.failed.txt"

VERIFY_HASH_FAIL_BIN="$TMPDIR_TEST/verify_hash_fail_bin"
VERIFY_HASH_FAIL_RUN="$TMPDIR_TEST/verify_hash_failure"
mkdir -p "$VERIFY_HASH_FAIL_BIN"
cp "$VERIFY_BIN/"* "$VERIFY_HASH_FAIL_BIN/"
cp -r "$VERIFY_RUN" "$VERIFY_HASH_FAIL_RUN"
rm "$VERIFY_HASH_FAIL_RUN/verification_report.txt"
cat > "$VERIFY_HASH_FAIL_BIN/sha256sum" <<'EOF'
#!/usr/bin/env bash
exit 7
EOF
chmod +x "$VERIFY_HASH_FAIL_BIN/sha256sum"
expect_failure env PATH="$VERIFY_HASH_FAIL_BIN:$PATH" VERIFY_FFMPEG_ARGS="$VERIFY_FFMPEG_ARGS" \
    "$SCRIPT" verify "$VERIFY_HASH_FAIL_RUN"
grep -F "Could not hash" "$VERIFY_HASH_FAIL_RUN/verification_report.failed.txt" >/dev/null
[[ ! -e "$VERIFY_HASH_FAIL_RUN/verification_report.txt" ]]

AUX_MCAP_MODULES="$TMPDIR_TEST/aux_mcap_modules"
AUX_VERIFY_BIN="$TMPDIR_TEST/aux_verify_bin"
mkdir -p "$AUX_MCAP_MODULES/mcap" "$AUX_VERIFY_BIN"
touch "$AUX_MCAP_MODULES/mcap/__init__.py"
cat > "$AUX_MCAP_MODULES/mcap/reader.py" <<'PY'
import json
import struct
from types import SimpleNamespace


def _record(field_offsets, object_size, timestamp):
    vtable_size = 4 + 2 * len(field_offsets)
    vtable = 48
    table = (vtable + vtable_size + 3) & ~3
    data = bytearray(table + object_size)
    struct.pack_into("<I", data, 0, 12)
    struct.pack_into("<HHHH", data, 4, 8, 36, 4, 12)
    struct.pack_into("<i", data, 12, 8)
    struct.pack_into("<I", data, 16, table - 16)
    struct.pack_into("<qqq", data, 24, timestamp, timestamp, timestamp)
    struct.pack_into(
        "<" + "H" * (2 + len(field_offsets)),
        data,
        vtable,
        vtable_size,
        object_size,
        *field_offsets,
    )
    struct.pack_into("<i", data, table, table - vtable)
    return data, table


def _video(sequence, timestamp, capture_epoch=0, fps=30):
    data, table = _record([4, 8, 16, 20, 24, 28, 0, 32], 36, timestamp)
    struct.pack_into("<Q", data, table + 8, sequence)
    struct.pack_into("<III", data, table + 16, 1600, 1300, fps)
    struct.pack_into("<B", data, table + 28, 1)
    struct.pack_into("<I", data, table + 32, capture_epoch)
    return bytes(data)


def _imu(sensor, sequence, sample_rate, timestamps, capture_epoch=0):
    record_timestamp = timestamps[-1] if timestamps else 1_000_000_000
    data, table = _record([4, 8, 16, 20, 24, 28, 32], 36, record_timestamp)
    struct.pack_into("<B", data, table + 4, sensor)
    struct.pack_into("<Q", data, table + 8, sequence)
    struct.pack_into("<I", data, table + 16, sample_rate)
    struct.pack_into("<f", data, table + 20, 24.0 if sensor == 0 else 2000.0)
    struct.pack_into("<I", data, table + 28, capture_epoch)
    struct.pack_into("<f", data, table + 32, 25.0)
    vector = len(data)
    struct.pack_into("<I", data, table + 24, vector - (table + 24))
    data.extend(struct.pack("<I", len(timestamps)))
    for timestamp in timestamps:
        data.extend(struct.pack("<dddqq", 0.0, 0.0, 0.0, timestamp, timestamp))
    return bytes(data)


def _diagnostic(timestamp):
    data, table = _record([4, 0, 8, 16, 20], 24, timestamp)
    struct.pack_into("<qB", data, table + 8, timestamp, 4)
    uid = len(data)
    data.extend(struct.pack("<I", 3) + b"ego\0")
    struct.pack_into("<I", data, table + 4, uid - (table + 4))
    data.extend(struct.pack("<HHi", 4, 4, 4))
    struct.pack_into("<I", data, table + 20, len(data) - 4 - (table + 20))
    return bytes(data)


def calibration_value(scenario="valid"):
    camera = {
        "width": 1600, "height": 1300, "fx": 500.25, "fy": 501.5,
        "cx": 800.0, "cy": 650.0, "distortion_model": 5,
        "distortion": [0.01, -0.02, 0.0, 0.0],
    }
    extrinsics = {"rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                  "translation_mm": [10.0, 20.0, 30.0]}
    value = {
        "device_uid": "ego", "color_left": dict(camera), "color_right": dict(camera),
        "left_to_right": dict(extrinsics), "accel_to_left": dict(extrinsics),
        "gyro_to_left": dict(extrinsics), "raw_alignment_yaml": "camera calibration",
        "raw_imu_yaml": "IMU calibration",
    }
    if scenario == "calibration_missing":
        value["gyro_to_left"] = None
    elif scenario == "calibration_bad_rotation":
        value["gyro_to_left"]["rotation"] = [0.0] * 9
    elif scenario == "calibration_reflection":
        value["gyro_to_left"]["rotation"] = [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    elif scenario == "calibration_bad_translation":
        value["gyro_to_left"]["translation_mm"] = [1.0, 2.0]
    elif scenario == "calibration_nonfinite":
        value["gyro_to_left"]["translation_mm"] = [float("nan"), 0.0, 0.0]
    elif scenario == "calibration_bad_dimensions":
        value["color_left"]["width"] = 640
    elif scenario == "calibration_bad_focal":
        value["color_left"]["fx"] = 0.0
    return value


def _append_table(data, count):
    vtable = len(data)
    vtable_size = 4 + 2 * count
    table = (vtable + vtable_size + 3) & ~3
    object_size = 4 + 4 * count
    data.extend(b"\0" * (table + object_size - len(data)))
    struct.pack_into("<" + "H" * (2 + count), data, vtable,
                     vtable_size, object_size, *(4 + 4 * i for i in range(count)))
    struct.pack_into("<i", data, table, table - vtable)
    return table


def _append_vector(data, position, values, fmt):
    vector = len(data)
    struct.pack_into("<I", data, position, vector - position)
    data.extend(struct.pack("<I", len(values)))
    data.extend(struct.pack("<" + fmt * len(values), *values))


def _calibration(scenario, epoch=0):
    value = calibration_value(scenario)
    data, table = _record([4, 8, 12, 16, 20, 24, 28, 32, 36], 40, 1_000_000_000)
    struct.pack_into("<I", data, table + 36, epoch)
    for key, field in (("device_uid", 4), ("raw_alignment_yaml", 28), ("raw_imu_yaml", 32)):
        _append_vector(data, table + field, value[key].encode(), "B")
    for key, offset in (("color_left", 8), ("color_right", 12), ("left_to_right", 16),
                        ("accel_to_left", 20), ("gyro_to_left", 24)):
        item = value[key]
        if item is None:
            struct.pack_into("<H", data, table - struct.unpack_from("<i", data, table)[0] +
                             4 + 2 * {8: 1, 12: 2, 16: 3, 20: 4, 24: 5}[offset], 0)
            continue
        nested = _append_table(data, 8 if key.startswith("color_") else 2)
        struct.pack_into("<I", data, table + offset, nested - table - offset)
        if key.startswith("color_"):
            for index, name in enumerate(("width", "height", "fx", "fy", "cx", "cy", "distortion_model")):
                fmt = "I" if index < 2 else "i" if index == 6 else "f"
                struct.pack_into("<" + fmt, data, nested + 4 + 4 * index, item[name])
            _append_vector(data, nested + 32, item["distortion"], "f")
        else:
            _append_vector(data, nested + 4, item["rotation"], "f")
            _append_vector(data, nested + 8, item["translation_mm"], "f")
    return bytes(data)


class _Reader:
    def __init__(self, stream):
        self.fixture = json.load(stream)

    def iter_metadata(self):
        # Record commands enable diagnostics, requiring both the policy and an event.
        if not self.fixture.get("diagnostics"):
            return iter(())
        return iter(
            [
                SimpleNamespace(
                    name="orbbec_ego.recording",
                    metadata={
                        "record_device_diagnostics": "on",
                    },
                )
            ]
        )

    def get_summary(self):
        return None

    def iter_messages(self, log_time_order=False):
        del log_time_order
        scenario = self.fixture["scenario"]
        base = 1_000_000_000
        video_period = 16_666_667 if scenario == "bad_video_cadence" else 33_333_333
        video_fps = 60 if scenario == "bad_video_profile" else 30
        for frame_index in range(31):
            capture_epoch = 1 if scenario in ("reconnect", "calibration_missing_epoch") and frame_index >= 15 else 0
            sequence = frame_index - 15 if capture_epoch else frame_index
            timestamp = base + sequence * video_period
            log_time = base + frame_index * video_period
            yield from self._message(
                "orbbec_ego_metadata_colorleft/FrameMetadata", _video(sequence, timestamp, capture_epoch, video_fps), log_time
            )
            yield from self._message(
                "orbbec_ego_metadata_colorright/FrameMetadata", _video(sequence, timestamp, capture_epoch, video_fps), log_time
            )
        yield from self._message("orbbec_ego_calibration/Calibration", _calibration(scenario), base)
        if scenario == "reconnect":
            yield from self._message("orbbec_ego_calibration/Calibration", _calibration(scenario, 1), base + 1)
        if self.fixture.get("diagnostics"):
            yield (
                SimpleNamespace(name="core.OrbbecEgoDeviceStateRecord"),
                SimpleNamespace(topic="orbbec_ego_device/DeviceState"),
                SimpleNamespace(data=_diagnostic(base), log_time=base),
            )
        else:
            yield from self._message("orbbec_ego_device/DeviceState", b"fixture", base)

        if scenario in ("valid", "empty_imu", "bad_imu_cadence") or (scenario.startswith("calibration_") and scenario != "calibration_missing_epoch"):
            sample_index = 0
            sample_period = 2_000_000 if scenario == "bad_imu_cadence" else 1_000_000
            for sequence in range(40):
                timestamps = [base + (sample_index + offset) * sample_period for offset in range(25)]
                sample_index += len(timestamps)
                accel_timestamps = [] if scenario == "empty_imu" and sequence == 0 else timestamps
                yield from self._message(
                    "orbbec_ego_imu_accel/Imu", _imu(0, sequence, 1000, accel_timestamps), timestamps[-1]
                )
                yield from self._message("orbbec_ego_imu_gyro/Imu", _imu(1, sequence, 1000, timestamps), timestamps[-1])

    @staticmethod
    def _message(topic, data, timestamp):
        yield (
            None,
            SimpleNamespace(topic=topic),
            SimpleNamespace(data=data, log_time=timestamp),
        )


def make_reader(stream):
    return _Reader(stream)
PY

cat > "$AUX_VERIFY_BIN/ffmpeg" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' 'frame=31' 'progress=end'
exit 0
EOF
cat > "$AUX_VERIFY_BIN/ffprobe" <<'EOF'
#!/usr/bin/env bash
for argument in "$@"; do
    if [[ "$argument" == "-count_frames" ]]; then
        echo 31
        exit 0
    fi
done
echo 'h264,1600,1300,60/1'
EOF
chmod +x "$AUX_VERIFY_BIN/ffmpeg" "$AUX_VERIFY_BIN/ffprobe"

make_aux_verify_run() {
    local run_dir="$1"
    local scenario="$2"
    mkdir -p "$run_dir/raw" "$run_dir/logs"
    printf '{"scenario":"%s"}\n' "$scenario" > "$run_dir/metadata.mcap"
    printf '\0' > "$run_dir/raw/ColorLeft.h264"
    printf '\0' > "$run_dir/raw/ColorRight.h264"
    PYTHONPATH="$AUX_MCAP_MODULES" python3 - "$run_dir/calibration.json" "$scenario" <<'PY'
import json
import sys
from pathlib import Path
from mcap.reader import calibration_value

Path(sys.argv[1]).write_text(json.dumps(calibration_value(sys.argv[2])))
PY
    cat > "$run_dir/requested_profile.json" <<'EOF'
{"version":1,"streams":{"ColorLeft":{"format":"h264","width":1600,"height":1300,"fps":30},"ColorRight":{"format":"h264","width":1600,"height":1300,"fps":30}}}
EOF
    printf '%s\n' \
        '  ColorLeft: 31 frames, 1 bytes, 0 sequence gaps' \
        '  ColorRight: 31 frames, 1 bytes, 0 sequence gaps' \
        '  IMU: accel=1000 gyro=1000 samples; queue_peak=1 dropped=0 video_frame_sets_dropped=0' \
        > "$run_dir/logs/capture.log"
}

AUX_VALID_RUN="$TMPDIR_TEST/aux_valid"
make_aux_verify_run "$AUX_VALID_RUN" valid

PYTHONPATH="$AUX_MCAP_MODULES" PATH="$AUX_VERIFY_BIN:$PATH" \
    "$SCRIPT" verify "$AUX_VALID_RUN" --expect-imu \
    >/dev/null 2>&1
grep -F 'raw_device_fps=30.000' "$AUX_VALID_RUN/verification_report.txt" >/dev/null
grep -F 'ffprobe_rate): h264,1600,1300,60/1' "$AUX_VALID_RUN/verification_report.txt" >/dev/null

PYTHONPATH="$AUX_MCAP_MODULES" PATH="$AUX_VERIFY_BIN:$PATH" \
    "$SCRIPT" verify "$AUX_VALID_RUN" --require-calibration --expect-imu \
    >/dev/null 2>&1
grep -F 'Required structured calibration: OK' "$AUX_VALID_RUN/verification_report.txt" >/dev/null
grep -F 'Calibration JSON matches' "$AUX_VALID_RUN/verification_report.txt" >/dev/null
for scenario in calibration_missing calibration_bad_rotation calibration_reflection \
        calibration_bad_translation calibration_nonfinite calibration_bad_dimensions \
        calibration_bad_focal calibration_missing_epoch; do
    run="$TMPDIR_TEST/$scenario"
    make_aux_verify_run "$run" "$scenario"
    expect_failure env PYTHONPATH="$AUX_MCAP_MODULES" PATH="$AUX_VERIFY_BIN:$PATH" \
        "$SCRIPT" verify "$run" --require-calibration
    grep -F 'Calibration' "$run/verification_report.failed.txt" >/dev/null
done
CALIBRATION_OPTIONAL_RUN="$TMPDIR_TEST/calibration_missing"
PYTHONPATH="$AUX_MCAP_MODULES" PATH="$AUX_VERIFY_BIN:$PATH" \
    "$SCRIPT" verify "$CALIBRATION_OPTIONAL_RUN" >/dev/null 2>&1
grep -F 'missing=gyro_to_left' "$CALIBRATION_OPTIONAL_RUN/verification_report.txt" >/dev/null
CALIBRATION_JSON_RUN="$TMPDIR_TEST/calibration_json_mismatch"
make_aux_verify_run "$CALIBRATION_JSON_RUN" valid
sed -i 's/500.25/999.25/' "$CALIBRATION_JSON_RUN/calibration.json"
expect_failure env PYTHONPATH="$AUX_MCAP_MODULES" PATH="$AUX_VERIFY_BIN:$PATH" \
    "$SCRIPT" verify "$CALIBRATION_JSON_RUN" --require-calibration
grep -F 'calibration JSON.color_left.fx differs' "$CALIBRATION_JSON_RUN/verification_report.failed.txt" >/dev/null

FINALIZE_PLUGIN="$TMPDIR_TEST/finalize_plugin"
cat > "$FINALIZE_PLUGIN" <<'PY'
#!/usr/bin/env python3
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

if "--list-capabilities" in sys.argv:
    print("Sensor LeftColor\nSensor RightColor")
    sys.exit(0)

stopping = False


def stop_recording(_signum, _frame):
    global stopping
    stopping = True


signal.signal(signal.SIGINT, stop_recording)
signal.signal(signal.SIGPIPE, signal.SIG_DFL)
target = Path(
    next(arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("--mcap-filename="))
)
fixture = Path(os.environ["FINALIZE_FIXTURE"])
partial = target.with_suffix(".mcap.partial")
partial.write_text(
    json.dumps(
        {**json.loads((fixture / "metadata.mcap").read_text()), "diagnostics": True}
    )
)
for sidecar in (fixture / "raw").iterdir():
    shutil.copyfile(sidecar, target.parent / "raw" / sidecar.name)
shutil.copyfile(fixture / "calibration.json", target.parent / "calibration.json")
(target.parent / "plugin.pid").write_text(str(os.getpid()))
print("recording ready", flush=True)
while not stopping:
    time.sleep(0.05)
# Shutdown logging must complete before the final MCAP rename.
time.sleep(0.15)
print("controlled shutdown complete", flush=True)
print((fixture / "logs/capture.log").read_text(), end="", flush=True)
partial.replace(target)
sys.exit(int(os.environ.get("FINALIZE_EXIT_STATUS", "0")))
PY
chmod +x "$FINALIZE_PLUGIN"

TEE_FAIL_BIN="$TMPDIR_TEST/tee_fail_bin"
mkdir -p "$TEE_FAIL_BIN"
REAL_TEE="$(command -v tee)"
cat > "$TEE_FAIL_BIN/tee" <<'EOF'
#!/usr/bin/env bash
for argument in "$@"; do
    if [[ "$argument" == */logs/capture.log ]]; then
        exec "$REAL_TEE" "$@" /dev/full
    fi
done
exec "$REAL_TEE" "$@"
EOF
chmod +x "$TEE_FAIL_BIN/tee"

PYTHONPATH="$AUX_MCAP_MODULES" PATH="$AUX_VERIFY_BIN:$PATH" \
    FINALIZE_FIXTURE="$AUX_VALID_RUN" REAL_TEE="$REAL_TEE" \
    python3 - "$SCRIPT" "$FINALIZE_PLUGIN" "$TMPDIR_TEST" "$TEE_FAIL_BIN" <<'PY'
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

script, plugin, temporary, tee_fail_bin = sys.argv[1:]
# Ctrl-C reaches the whole foreground group, including the shell and tee.
for name, duration, interrupt, exit_status, tee_failure in (
    ("ctrl_c", None, True, 0, False),
    ("timed_ctrl_c", "30", True, 0, False),
    ("timed_completion", "1", False, 0, False),
    ("timed_complete_calibration", "1", False, 0, False),
    ("ctrl_c_plugin_failure", None, True, 7, False),
    ("ctrl_c_log_failure", None, True, 0, True),
):
    run = Path(temporary) / name
    capture_log = run / "logs/capture.log"
    command = [
        script,
        "record",
        "--plugin",
        plugin,
        "--output",
        str(run),
        "--no-imu",
    ]
    if duration:
        command.extend(["--duration", duration])
    if name == "timed_complete_calibration":
        command.append("--require-calibration")
    environment = dict(os.environ, FINALIZE_EXIT_STATUS=str(exit_status))
    if tee_failure:
        environment["PATH"] = f"{tee_fail_bin}:{environment['PATH']}"
    output_path = Path(temporary) / f"{name}.output"
    with output_path.open("w") as output:
        process = subprocess.Popen(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 5
            while not (
                capture_log.exists() and "recording ready" in capture_log.read_text()
            ):
                assert process.poll() is None and time.monotonic() < deadline, (
                    f"{name}: plugin did not become ready\n{output_path.read_text()}"
                )
                time.sleep(0.01)
            if interrupt:
                os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=8)
        finally:
            pid_file = run / "plugin.pid"
            if pid_file.exists():
                try:
                    os.kill(int(pid_file.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
    output = output_path.read_text()
    if name == "timed_complete_calibration":
        assert "Required structured calibration: OK" in output, output
    assert (run / "metadata.mcap").is_file(), (
        f"{name}: MCAP was not finalized\n{output}"
    )
    assert not list(run.rglob("*.partial")), (
        f"{name}: partial recording remains\n{output}"
    )
    log = capture_log.read_text()
    assert "controlled shutdown complete" in log, (
        f"{name}: shutdown log was lost\n{output}"
    )
    assert "video_frame_sets_dropped=0" in log, (
        f"{name}: capture statistics were lost\n{output}"
    )
    report = run / "verification_report.txt"
    if exit_status or tee_failure:
        assert process.returncode != 0, (
            f"{name}: capture failure was accepted\n{output}"
        )
        expected = (
            "Unable to write capture log"
            if tee_failure
            else "Capture failed with status 7"
        )
        assert expected in output, f"{name}: missing failure diagnostic\n{output}"
        assert not report.exists(), f"{name}: verification ran after capture failure"
    else:
        assert process.returncode == 0, f"{name}: capture failed\n{output}"
        assert "Verification passed:" in report.read_text(), (
            f"{name}: verification did not complete"
        )
PY

for scenario in bad_video_profile bad_video_cadence; do
    run="$TMPDIR_TEST/$scenario"
    make_aux_verify_run "$run" "$scenario"
    expect_failure env PYTHONPATH="$AUX_MCAP_MODULES" PATH="$AUX_VERIFY_BIN:$PATH" \
        "$SCRIPT" verify "$run"
    if [[ "$scenario" == bad_video_profile ]]; then
        grep -F 'differs from requested' "$run/verification_report.failed.txt" >/dev/null
    else
        grep -F 'raw-device cadence is 60.000 FPS' "$run/verification_report.failed.txt" >/dev/null
    fi
done
expect_failure python3 "$(dirname "$SCRIPT")/tools/verify_profile.py" \
    "$AUX_VALID_RUN/requested_profile.json" h264,1600,1300,0/1 h264,1600,1300,60/1
expect_failure python3 "$(dirname "$SCRIPT")/tools/verify_profile.py" \
    "$AUX_VALID_RUN/requested_profile.json" h264,1600,1200,60/1 h264,1600,1300,60/1
expect_failure python3 "$(dirname "$SCRIPT")/tools/verify_profile.py" \
    "$AUX_VALID_RUN/requested_profile.json" hevc,1600,1300,60/1 h264,1600,1300,60/1

AUX_RECONNECT_RUN="$TMPDIR_TEST/aux_reconnect"
make_aux_verify_run "$AUX_RECONNECT_RUN" reconnect
PYTHONPATH="$AUX_MCAP_MODULES" PATH="$AUX_VERIFY_BIN:$PATH" \
    "$SCRIPT" verify "$AUX_RECONNECT_RUN" >/dev/null 2>&1

AUX_EMPTY_IMU_RUN="$TMPDIR_TEST/aux_empty_imu"
make_aux_verify_run "$AUX_EMPTY_IMU_RUN" empty_imu
set +e
AUX_EMPTY_IMU_OUTPUT="$(PYTHONPATH="$AUX_MCAP_MODULES" PATH="$AUX_VERIFY_BIN:$PATH" \
    "$SCRIPT" verify "$AUX_EMPTY_IMU_RUN" --expect-imu 2>&1)"
AUX_EMPTY_IMU_STATUS=$?
set -e
[[ "$AUX_EMPTY_IMU_STATUS" -ne 0 ]]
grep -F "contains an empty IMU batch" <<< "$AUX_EMPTY_IMU_OUTPUT" >/dev/null

AUX_BAD_IMU_CADENCE_RUN="$TMPDIR_TEST/aux_bad_imu_cadence"
make_aux_verify_run "$AUX_BAD_IMU_CADENCE_RUN" bad_imu_cadence
set +e
AUX_BAD_IMU_CADENCE_OUTPUT="$(PYTHONPATH="$AUX_MCAP_MODULES" PATH="$AUX_VERIFY_BIN:$PATH" \
    "$SCRIPT" verify "$AUX_BAD_IMU_CADENCE_RUN" --expect-imu 2>&1)"
AUX_BAD_IMU_CADENCE_STATUS=$?
set -e
[[ "$AUX_BAD_IMU_CADENCE_STATUS" -ne 0 ]]
grep -F "raw-device cadence is 500.000 Hz" <<< "$AUX_BAD_IMU_CADENCE_OUTPUT" >/dev/null

rm "$PACKAGE_DIR/OrbbecSDKConfig.xml"
expect_failure env PATH="$MOCK_BIN:$PATH" "$PACKAGE_DIR/orbbec_ego.sh" doctor

DIAGNOSTIC_RUN="$TMPDIR_TEST/diagnostic_off"
expect_failure "$SCRIPT" record --plugin "$MOCK_PLUGIN" --duration 1 --output "$DIAGNOSTIC_RUN" --record-device-diagnostics=off
grep -Fx -- "--record-device-diagnostics=off" "$MOCK_ARGS" >/dev/null
grep -F '"record_device_diagnostics": "off"' "$DIAGNOSTIC_RUN/requested_profile.json" >/dev/null
expect_failure "$SCRIPT" record --record-device-diagnostics=invalid
