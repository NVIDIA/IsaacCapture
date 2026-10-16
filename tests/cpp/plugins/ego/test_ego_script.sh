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
Sensor Audio
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
if grep -Eq '^--enable-audio$|^--audio-output=' "$MOCK_ARGS"; then
    echo "Default capture must not request audio" >&2
    exit 1
fi
grep -Fx -- "--mcap-filename=$RUN/metadata.mcap" "$MOCK_ARGS" >/dev/null
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

AUDIO_RUN="$TMPDIR_TEST/audio_recording"
expect_failure "$SCRIPT" record --plugin "$MOCK_PLUGIN" --duration 1 --output "$AUDIO_RUN" --audio
grep -Fx -- "--audio-output=$AUDIO_RUN/Audio.wav" "$MOCK_ARGS" >/dev/null

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
audio = true
preview = true
mcap_media = "embedded"
keep_media_sidecars = true
plugin = "$MOCK_PLUGIN"
plugin_options = ["--bitrate=8", "--dynamic-bitrate=on"]
EOF
CONFIG_AUDIO_RUN="$TMPDIR_TEST/config_audio_recording"
expect_failure "$SCRIPT" record --config "$CONFIG_FILE" --duration 1 --output "$CONFIG_AUDIO_RUN"
grep -Fx -- "--audio-output=$CONFIG_AUDIO_RUN/Audio.wav" "$MOCK_ARGS" >/dev/null

expect_failure "$SCRIPT" record --config "$CONFIG_FILE" --duration 1 \
    --output "$CONFIG_RUN" --format h264 --no-audio
grep -Fx -- "--device-uid=ego-config" "$MOCK_ARGS" >/dev/null
grep -Fx -- "--reconnect-timeout=45" "$MOCK_ARGS" >/dev/null
grep -Fx -- "--reconnect-interval-ms=250" "$MOCK_ARGS" >/dev/null
grep -Fx -- "--preview" "$MOCK_ARGS" >/dev/null
grep -Fx -- "--bitrate=8" "$MOCK_ARGS" >/dev/null
grep -Fx -- "--dynamic-bitrate=on" "$MOCK_ARGS" >/dev/null
grep -F "format=h264,width=1600,height=1200,fps=30" "$MOCK_ARGS" >/dev/null
if grep -Eq '^--enable-(imu|audio)$|^--audio-output=' "$MOCK_ARGS"; then
    echo "Config/CLI stream overrides were not applied" >&2
    exit 1
fi

CONFIG_NO_AUDIO_FILE="$TMPDIR_TEST/record_no_audio.toml"
CONFIG_NO_AUDIO_RUN="$TMPDIR_TEST/config_no_audio_recording"
sed 's/^audio = true$/audio = false/' "$CONFIG_FILE" > "$CONFIG_NO_AUDIO_FILE"
expect_failure "$SCRIPT" record --config "$CONFIG_NO_AUDIO_FILE" --duration 1 \
    --output "$CONFIG_NO_AUDIO_RUN"
grep -Fx -- "--mcap-filename=$CONFIG_NO_AUDIO_RUN/metadata.mcap" "$MOCK_ARGS" >/dev/null
if grep -Eq '^--enable-audio$|^--audio-output=' "$MOCK_ARGS"; then
    echo "Config with audio=false must not request audio" >&2
    exit 1
fi

CONFIG_AUDIO_OVERRIDE_RUN="$TMPDIR_TEST/config_audio_override_recording"
expect_failure "$SCRIPT" record --config "$CONFIG_NO_AUDIO_FILE" --duration 1 \
    --output "$CONFIG_AUDIO_OVERRIDE_RUN" --audio
grep -Fx -- "--audio-output=$CONFIG_AUDIO_OVERRIDE_RUN/Audio.wav" "$MOCK_ARGS" >/dev/null

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
    --output "$TMPDIR_TEST/signal_recording" --no-imu --no-audio 2>&1)"
SIGNAL_STATUS=$?
set -e
[[ "$SIGNAL_STATUS" -ne 0 ]]
grep -F "Capture failed with status 7" <<< "$SIGNAL_OUTPUT" >/dev/null

RUN_EMBEDDED="$TMPDIR_TEST/embedded"
expect_failure "$SCRIPT" record --plugin "$MOCK_PLUGIN" --duration 1 --output "$RUN_EMBEDDED" --mcap-media embedded
grep -Fx -- "--mcap-media=embedded" "$MOCK_ARGS" >/dev/null
if grep -Eq '^--enable-audio$|^--audio-output=' "$MOCK_ARGS"; then
    echo "Default embedded capture must not request audio" >&2
    exit 1
fi

RUN_EMBEDDED_AUDIO="$TMPDIR_TEST/embedded_audio"
expect_failure "$SCRIPT" record --plugin "$MOCK_PLUGIN" --duration 1 \
    --output "$RUN_EMBEDDED_AUDIO" --mcap-media embedded --audio
grep -Fx -- "--enable-audio" "$MOCK_ARGS" >/dev/null
if grep -q '^--audio-output=' "$MOCK_ARGS"; then
    echo "Embedded capture must not request a WAV sidecar by default" >&2
    exit 1
fi

PACKAGE_DIR="$TMPDIR_TEST/package/plugins/ego_camera"
MOCK_BIN="$TMPDIR_TEST/bin"
mkdir -p "$PACKAGE_DIR/extensions" "$MOCK_BIN"
cp "$SCRIPT" "$PACKAGE_DIR/ego.sh"
cp -r "$(dirname "$SCRIPT")/tools" "$PACKAGE_DIR/tools"
chmod +x "$PACKAGE_DIR/ego.sh"
touch "$PACKAGE_DIR/libOrbbecSDK.so.2" "$PACKAGE_DIR/OrbbecSDKConfig.xml"

cat > "$PACKAGE_DIR/ego_camera_plugin" <<'EOF'
#!/usr/bin/env bash
if [[ "$1" == "--help" ]]; then
    exit 0
fi
exit 1
EOF
cat > "$PACKAGE_DIR/ego_mcap_export_media" <<'EOF'
#!/usr/bin/env bash
mkdir -p "$2"
printf '\0' > "$2/ColorLeft.h264"
printf '\0' > "$2/ColorRight.h264"
if [[ -n "${MOCK_EXPORTER_USED:-}" ]]; then
    printf '%s\n' "$0" > "$MOCK_EXPORTER_USED"
fi
EOF
chmod +x "$PACKAGE_DIR/ego_camera_plugin" "$PACKAGE_DIR/ego_mcap_export_media"

for command_name in python3 ffmpeg ffprobe; do
    cat > "$MOCK_BIN/$command_name" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
    chmod +x "$MOCK_BIN/$command_name"
done

PACKAGE_RUN="$TMPDIR_TEST/package_run"
mkdir -p "$PACKAGE_RUN"
printf '\0' > "$PACKAGE_RUN/metadata.mcap"
PATH="$MOCK_BIN:$PATH" "$PACKAGE_DIR/ego.sh" doctor >/dev/null
PATH="$MOCK_BIN:$PATH" "$PACKAGE_DIR/ego.sh" export-media "$PACKAGE_RUN" >/dev/null
test -f "$PACKAGE_RUN/exported/ColorLeft.h264"
expect_failure env PATH="$MOCK_BIN:$PATH" "$PACKAGE_DIR/ego.sh" export-media "$PACKAGE_RUN"

INSTALLED_VERIFY_RUN="$TMPDIR_TEST/installed_verify"
INSTALLED_EXPORTER_USED="$TMPDIR_TEST/installed_exporter_used"
mkdir -p "$INSTALLED_VERIFY_RUN"
printf '\0' > "$INSTALLED_VERIFY_RUN/metadata.mcap"
expect_failure env PATH="$MOCK_BIN:$PATH" MOCK_EXPORTER_USED="$INSTALLED_EXPORTER_USED" \
    "$SCRIPT" verify "$INSTALLED_VERIFY_RUN" --media-mode embedded \
    --plugin "$PACKAGE_DIR/ego_camera_plugin"
grep -Fx "$PACKAGE_DIR/ego_mcap_export_media" "$INSTALLED_EXPORTER_USED" >/dev/null

BUILD_PLUGIN_DIR="$TMPDIR_TEST/build/src/plugins/ego/app"
BUILD_EXPORTER_DIR="$TMPDIR_TEST/build/src/plugins/ego/mcap/export_media_tool"
BUILD_VERIFY_RUN="$TMPDIR_TEST/build_verify"
BUILD_EXPORTER_USED="$TMPDIR_TEST/build_exporter_used"
mkdir -p "$BUILD_PLUGIN_DIR" "$BUILD_EXPORTER_DIR" "$BUILD_VERIFY_RUN"
cp "$PACKAGE_DIR/ego_camera_plugin" "$BUILD_PLUGIN_DIR/ego_camera_plugin"
cp "$PACKAGE_DIR/ego_mcap_export_media" "$BUILD_EXPORTER_DIR/ego_mcap_export_media"
printf '\0' > "$BUILD_VERIFY_RUN/metadata.mcap"
expect_failure env PATH="$MOCK_BIN:$PATH" MOCK_EXPORTER_USED="$BUILD_EXPORTER_USED" \
    "$SCRIPT" verify "$BUILD_VERIFY_RUN" --media-mode embedded \
    --plugin "$BUILD_PLUGIN_DIR/ego_camera_plugin"
grep -Fx "$BUILD_EXPORTER_DIR/ego_mcap_export_media" "$BUILD_EXPORTER_USED" >/dev/null

MISSING_PLUGIN_DIR="$TMPDIR_TEST/missing_build/src/plugins/ego/app"
MISSING_VERIFY_RUN="$TMPDIR_TEST/missing_build_verify"
mkdir -p "$MISSING_PLUGIN_DIR" "$MISSING_VERIFY_RUN"
cp "$PACKAGE_DIR/ego_camera_plugin" "$MISSING_PLUGIN_DIR/ego_camera_plugin"
printf '\0' > "$MISSING_VERIFY_RUN/metadata.mcap"
set +e
MISSING_EXPORTER_OUTPUT="$(PATH="$MOCK_BIN:$PATH" "$SCRIPT" verify "$MISSING_VERIFY_RUN" \
    --media-mode embedded --plugin "$MISSING_PLUGIN_DIR/ego_camera_plugin" 2>&1)"
MISSING_EXPORTER_STATUS=$?
set -e
[[ "$MISSING_EXPORTER_STATUS" -ne 0 && "$MISSING_EXPORTER_STATUS" -ne 127 ]]
grep -F "No embedded-media exporter exists beside plugin or in its build tree" \
    <<< "$MISSING_EXPORTER_OUTPUT" >/dev/null

PARTIAL_RUN="$TMPDIR_TEST/partial_run"
mkdir -p "$PARTIAL_RUN"
printf '\0' > "$PARTIAL_RUN/metadata.mcap"
touch "$PARTIAL_RUN/ego_media_fragment.mcap.partial"
expect_failure "$SCRIPT" verify "$PARTIAL_RUN"

VERIFY_BIN="$TMPDIR_TEST/verify_bin"
VERIFY_RUN="$TMPDIR_TEST/verify_with_sidecars"
VERIFY_EXPORTER_USED="$TMPDIR_TEST/verify_sidecar_exporter_used"
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
    '  IMU: accel=0 gyro=0 samples; audio=0 samples; queue_peak=0 dropped=0 video_frame_sets_dropped=0' \
    > "$VERIFY_RUN/logs/capture.log"
PATH="$VERIFY_BIN:$PATH" MOCK_EXPORTER_USED="$VERIFY_EXPORTER_USED" \
    VERIFY_FFMPEG_ARGS="$VERIFY_FFMPEG_ARGS" \
    "$SCRIPT" verify "$VERIFY_RUN" --media-mode embedded \
    --plugin "$PACKAGE_DIR/ego_camera_plugin" >/dev/null 2>&1
grep -Fx "$PACKAGE_DIR/ego_mcap_export_media" "$VERIFY_EXPORTER_USED" >/dev/null
test -z "$(find "$VERIFY_RUN" -maxdepth 1 -name '.verify_export.*' -print -quit)"
VERIFY_ZERO_ERROR_RUN="$TMPDIR_TEST/verify_zero_exit_error"
cp -r "$VERIFY_RUN" "$VERIFY_ZERO_ERROR_RUN"
expect_failure env PATH="$VERIFY_BIN:$PATH" MOCK_EXPORTER_USED="$VERIFY_EXPORTER_USED" \
    VERIFY_FFMPEG_ARGS="$VERIFY_FFMPEG_ARGS" VERIFY_ZERO_EXIT_ERROR=1 \
    "$SCRIPT" verify "$VERIFY_ZERO_ERROR_RUN" --media-mode embedded \
    --plugin "$PACKAGE_DIR/ego_camera_plugin"
[[ ! -e "$VERIFY_ZERO_ERROR_RUN/verification_report.txt" ]]
grep -F 'decoder error with a zero exit status' "$VERIFY_ZERO_ERROR_RUN/verification_report.failed.txt" >/dev/null
[[ -z "$(find "$VERIFY_ZERO_ERROR_RUN" -maxdepth 1 -name '.verify_export.*' -print -quit)" ]]
test -s "$VERIFY_RUN/verification_report.txt"
grep -F "Verification passed:" "$VERIFY_RUN/verification_report.txt" >/dev/null
grep -E '^ColorLeft SHA-256: [0-9a-f]{64}$' "$VERIFY_RUN/verification_report.txt" >/dev/null
grep -E '^ColorRight SHA-256: [0-9a-f]{64}$' "$VERIFY_RUN/verification_report.txt" >/dev/null
grep -Fx -- "-xerror" "$VERIFY_FFMPEG_ARGS" >/dev/null
grep -Fx -- "-err_detect" "$VERIFY_FFMPEG_ARGS" >/dev/null
grep -Fx -- "explode" "$VERIFY_FFMPEG_ARGS" >/dev/null

VERIFY_FAIL_BIN="$TMPDIR_TEST/verify_fail_bin"
VERIFY_FAIL_RUN="$TMPDIR_TEST/verify_export_cleanup"
mkdir -p "$VERIFY_FAIL_BIN" "$VERIFY_FAIL_RUN/logs"
cp "$MOCK_BIN/python3" "$VERIFY_BIN/ffprobe" "$VERIFY_FAIL_BIN/"
cat > "$VERIFY_FAIL_BIN/ffmpeg" <<'EOF'
#!/usr/bin/env bash
exit 9
EOF
chmod +x "$VERIFY_FAIL_BIN/ffmpeg"
printf '\0' > "$VERIFY_FAIL_RUN/metadata.mcap"
printf '%s\n' \
    '  ColorLeft: 1 frames, 1 bytes, 0 sequence gaps' \
    '  ColorRight: 1 frames, 1 bytes, 0 sequence gaps' \
    '  IMU: accel=0 gyro=0 samples; audio=0 samples; queue_peak=0 dropped=0 video_frame_sets_dropped=0' \
    > "$VERIFY_FAIL_RUN/logs/capture.log"
expect_failure env PATH="$VERIFY_FAIL_BIN:$PATH" \
    "$SCRIPT" verify "$VERIFY_FAIL_RUN" --media-mode embedded \
    --plugin "$PACKAGE_DIR/ego_camera_plugin"
test -z "$(find "$VERIFY_FAIL_RUN" -maxdepth 1 -name '.verify_export.*' -print -quit)"
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
    "$SCRIPT" verify "$VERIFY_HASH_FAIL_RUN" --media-mode embedded \
    --plugin "$PACKAGE_DIR/ego_camera_plugin"
grep -F "Could not hash" "$VERIFY_HASH_FAIL_RUN/verification_report.failed.txt" >/dev/null
[[ ! -e "$VERIFY_HASH_FAIL_RUN/verification_report.txt" ]]
[[ -z "$(find "$VERIFY_HASH_FAIL_RUN" -maxdepth 1 -name '.verify_export.*' -print -quit)" ]]

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
    data, table = _record([4, 8, 16, 20, 24, 28, 0, 32, 36], 40, timestamp)
    struct.pack_into("<Q", data, table + 8, sequence)
    struct.pack_into("<III", data, table + 16, 1600, 1300, fps)
    struct.pack_into("<B", data, table + 28, 1)
    struct.pack_into("<II", data, table + 32, capture_epoch, capture_epoch)
    return bytes(data)


def _imu(sensor, sequence, sample_rate, timestamps, capture_epoch=0):
    record_timestamp = timestamps[-1] if timestamps else 1_000_000_000
    data, table = _record([4, 8, 16, 20, 24, 28], 32, record_timestamp)
    struct.pack_into("<B", data, table + 4, sensor)
    struct.pack_into("<Q", data, table + 8, sequence)
    struct.pack_into("<I", data, table + 16, sample_rate)
    struct.pack_into("<f", data, table + 20, 24.0 if sensor == 0 else 2000.0)
    struct.pack_into("<I", data, table + 28, capture_epoch)
    vector = len(data)
    struct.pack_into("<I", data, table + 24, vector - (table + 24))
    data.extend(struct.pack("<I", len(timestamps)))
    for timestamp in timestamps:
        data.extend(struct.pack("<ddddqq", 0.0, 0.0, 0.0, 25.0, timestamp, timestamp))
    return bytes(data)


def _audio(sequence, timestamp, sample_count, sample_format=0, capture_epoch=0):
    data, table = _record([8, 16, 20, 22, 24, 28, 32, 40, 44], 48, timestamp)
    struct.pack_into("<Q", data, table + 8, sequence)
    struct.pack_into("<I", data, table + 16, 48_000)
    struct.pack_into("<HH", data, table + 20, 1, 16)
    struct.pack_into("<b", data, table + 24, sample_format)
    struct.pack_into("<I", data, table + 28, sample_count)
    struct.pack_into("<I", data, table + 40, sample_count * 2)
    struct.pack_into("<I", data, table + 44, capture_epoch)
    return bytes(data)


def _pcm_audio(sequence, timestamp, sample_count, pcm_bytes=None, sample_format=0, capture_epoch=0):
    if pcm_bytes is None:
        pcm_bytes = sample_count * 2
    data, table = _record([8, 16, 20, 22, 24, 28, 32, 36], 40, timestamp)
    struct.pack_into("<Q", data, table + 8, sequence)
    struct.pack_into("<I", data, table + 16, 48_000)
    struct.pack_into("<HH", data, table + 20, 1, 16)
    struct.pack_into("<b", data, table + 24, sample_format)
    struct.pack_into("<I", data, table + 28, sample_count)
    vector = len(data)
    struct.pack_into("<I", data, table + 32, vector - (table + 32))
    struct.pack_into("<I", data, table + 36, capture_epoch)
    data.extend(struct.pack("<I", pcm_bytes))
    data.extend(b"\0" * pcm_bytes)
    return bytes(data)


class _Reader:
    def __init__(self, stream):
        self.fixture = json.load(stream)

    def iter_metadata(self):
        return iter(())

    def get_summary(self):
        return None

    def iter_messages(self, log_time_order=False):
        del log_time_order
        scenario = self.fixture["scenario"]
        mode = self.fixture["mode"]
        base = 1_000_000_000
        video_period = 16_666_667 if scenario == "bad_video_cadence" else 33_333_333
        video_fps = 60 if scenario == "bad_video_profile" else 30
        for frame_index in range(31):
            capture_epoch = 1 if scenario == "reconnect" and frame_index >= 15 else 0
            sequence = frame_index - 15 if capture_epoch else frame_index
            timestamp = base + sequence * video_period
            log_time = base + frame_index * video_period
            yield from self._message(
                "ego_metadata/ColorLeft", _video(sequence, timestamp, capture_epoch, video_fps), log_time
            )
            yield from self._message(
                "ego_metadata/ColorRight", _video(sequence, timestamp, capture_epoch, video_fps), log_time
            )
            if mode == "embedded":
                yield from self._message(
                    "ego_media/ColorLeft", _video(sequence, timestamp, capture_epoch, video_fps), log_time
                )
                yield from self._message(
                    "ego_media/ColorRight", _video(sequence, timestamp, capture_epoch, video_fps), log_time
                )
        yield from self._message("ego_calibration/Calibration", b"fixture", base)
        yield from self._message("ego_device/DeviceState", b"fixture", base)

        if scenario in ("valid", "empty_imu", "bad_imu_cadence"):
            sample_index = 0
            sample_period = 2_000_000 if scenario == "bad_imu_cadence" else 1_000_000
            for sequence in range(40):
                timestamps = [base + (sample_index + offset) * sample_period for offset in range(25)]
                sample_index += len(timestamps)
                accel_timestamps = [] if scenario == "empty_imu" and sequence == 0 else timestamps
                yield from self._message(
                    "ego_imu/Accel", _imu(0, sequence, 1000, accel_timestamps), timestamps[-1]
                )
                yield from self._message("ego_imu/Gyro", _imu(1, sequence, 1000, timestamps), timestamps[-1])

        if scenario in (
            "valid",
            "sparse_audio",
            "sparse_pcm",
            "empty_pcm",
            "bad_audio_cadence",
            "bad_audio_format",
        ):
            chunks = 2 if scenario == "sparse_audio" else 32
            sample_count = 1 if scenario == "sparse_audio" else 1500
            cadence_scale = 2 if scenario == "bad_audio_cadence" else 1
            sample_format = 1 if scenario == "bad_audio_format" else 0
            for sequence in range(chunks):
                timestamp = base + cadence_scale * sequence * sample_count * 1_000_000_000 // 48_000
                yield from self._message(
                    "ego_audio/Audio", _audio(sequence, timestamp, sample_count, sample_format), timestamp
                )
                if mode == "embedded" and scenario != "sparse_pcm":
                    pcm_bytes = 0 if scenario == "empty_pcm" and sequence == 0 else None
                    yield from self._message(
                        "ego_media/Audio",
                        _pcm_audio(sequence, timestamp, sample_count, pcm_bytes, sample_format),
                        timestamp,
                    )
            if mode == "embedded" and scenario == "sparse_pcm":
                for sequence in range(2):
                    timestamp = base + sequence * 1_000_000_000 // 48_000
                    yield from self._message(
                        "ego_media/Audio", _pcm_audio(sequence, timestamp, 1), timestamp
                    )

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
    if [[ "$argument" == *.wav ]]; then
        printf '%s\n' 'codec_name=pcm_s16le' 'sample_rate=48000' 'channels=1' 'bits_per_sample=16'
        exit 0
    fi
done
echo 'h264,1600,1300,60/1'
EOF
chmod +x "$AUX_VERIFY_BIN/ffmpeg" "$AUX_VERIFY_BIN/ffprobe"

make_aux_verify_run() {
    local run_dir="$1"
    local scenario="$2"
    local mode="$3"
    mkdir -p "$run_dir/raw" "$run_dir/logs"
    printf '{"scenario":"%s","mode":"%s"}\n' "$scenario" "$mode" > "$run_dir/metadata.mcap"
    printf '\0' > "$run_dir/raw/ColorLeft.h264"
    printf '\0' > "$run_dir/raw/ColorRight.h264"
    printf '\0' > "$run_dir/Audio.wav"
    cat > "$run_dir/requested_profile.json" <<'EOF'
{"version":1,"streams":{"ColorLeft":{"format":"h264","width":1600,"height":1300,"fps":30},"ColorRight":{"format":"h264","width":1600,"height":1300,"fps":30}}}
EOF
    printf '%s\n' \
        '  ColorLeft: 31 frames, 1 bytes, 0 sequence gaps' \
        '  ColorRight: 31 frames, 1 bytes, 0 sequence gaps' \
        '  IMU: accel=1000 gyro=1000 samples; audio=48000 samples; queue_peak=1 dropped=0 video_frame_sets_dropped=0' \
        > "$run_dir/logs/capture.log"
}

AUX_VALID_RUN="$TMPDIR_TEST/aux_valid"
make_aux_verify_run "$AUX_VALID_RUN" valid metadata-only
PYTHONPATH="$AUX_MCAP_MODULES" PATH="$AUX_VERIFY_BIN:$PATH" \
    "$SCRIPT" verify "$AUX_VALID_RUN" --media-mode metadata-only --expect-imu --expect-audio \
    >/dev/null 2>&1
grep -F 'raw_device_fps=30.000' "$AUX_VALID_RUN/verification_report.txt" >/dev/null
grep -F 'ffprobe_rate): h264,1600,1300,60/1' "$AUX_VALID_RUN/verification_report.txt" >/dev/null

for scenario in bad_video_profile bad_video_cadence; do
    run="$TMPDIR_TEST/$scenario"
    make_aux_verify_run "$run" "$scenario" metadata-only
    expect_failure env PYTHONPATH="$AUX_MCAP_MODULES" PATH="$AUX_VERIFY_BIN:$PATH" \
        "$SCRIPT" verify "$run" --media-mode metadata-only
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
make_aux_verify_run "$AUX_RECONNECT_RUN" reconnect metadata-only
PYTHONPATH="$AUX_MCAP_MODULES" PATH="$AUX_VERIFY_BIN:$PATH" \
    "$SCRIPT" verify "$AUX_RECONNECT_RUN" --media-mode metadata-only >/dev/null 2>&1

AUX_EMPTY_IMU_RUN="$TMPDIR_TEST/aux_empty_imu"
make_aux_verify_run "$AUX_EMPTY_IMU_RUN" empty_imu metadata-only
set +e
AUX_EMPTY_IMU_OUTPUT="$(PYTHONPATH="$AUX_MCAP_MODULES" PATH="$AUX_VERIFY_BIN:$PATH" \
    "$SCRIPT" verify "$AUX_EMPTY_IMU_RUN" --media-mode metadata-only --expect-imu 2>&1)"
AUX_EMPTY_IMU_STATUS=$?
set -e
[[ "$AUX_EMPTY_IMU_STATUS" -ne 0 ]]
grep -F "contains an empty IMU batch" <<< "$AUX_EMPTY_IMU_OUTPUT" >/dev/null

AUX_BAD_IMU_CADENCE_RUN="$TMPDIR_TEST/aux_bad_imu_cadence"
make_aux_verify_run "$AUX_BAD_IMU_CADENCE_RUN" bad_imu_cadence metadata-only
set +e
AUX_BAD_IMU_CADENCE_OUTPUT="$(PYTHONPATH="$AUX_MCAP_MODULES" PATH="$AUX_VERIFY_BIN:$PATH" \
    "$SCRIPT" verify "$AUX_BAD_IMU_CADENCE_RUN" --media-mode metadata-only --expect-imu 2>&1)"
AUX_BAD_IMU_CADENCE_STATUS=$?
set -e
[[ "$AUX_BAD_IMU_CADENCE_STATUS" -ne 0 ]]
grep -F "raw-device cadence is 500.000 Hz" <<< "$AUX_BAD_IMU_CADENCE_OUTPUT" >/dev/null

AUX_SPARSE_AUDIO_RUN="$TMPDIR_TEST/aux_sparse_audio"
make_aux_verify_run "$AUX_SPARSE_AUDIO_RUN" sparse_audio metadata-only
set +e
AUX_SPARSE_AUDIO_OUTPUT="$(PYTHONPATH="$AUX_MCAP_MODULES" PATH="$AUX_VERIFY_BIN:$PATH" \
    "$SCRIPT" verify "$AUX_SPARSE_AUDIO_RUN" --media-mode metadata-only --expect-audio 2>&1)"
AUX_SPARSE_AUDIO_STATUS=$?
set -e
[[ "$AUX_SPARSE_AUDIO_STATUS" -ne 0 ]]
grep -F "contains only" <<< "$AUX_SPARSE_AUDIO_OUTPUT" >/dev/null

AUX_BAD_AUDIO_CADENCE_RUN="$TMPDIR_TEST/aux_bad_audio_cadence"
make_aux_verify_run "$AUX_BAD_AUDIO_CADENCE_RUN" bad_audio_cadence metadata-only
set +e
AUX_BAD_AUDIO_CADENCE_OUTPUT="$(PYTHONPATH="$AUX_MCAP_MODULES" PATH="$AUX_VERIFY_BIN:$PATH" \
    "$SCRIPT" verify "$AUX_BAD_AUDIO_CADENCE_RUN" --media-mode metadata-only --expect-audio 2>&1)"
AUX_BAD_AUDIO_CADENCE_STATUS=$?
set -e
[[ "$AUX_BAD_AUDIO_CADENCE_STATUS" -ne 0 ]]
grep -F "raw-device cadence is 24000.000 Hz" <<< "$AUX_BAD_AUDIO_CADENCE_OUTPUT" >/dev/null

AUX_BAD_AUDIO_FORMAT_RUN="$TMPDIR_TEST/aux_bad_audio_format"
make_aux_verify_run "$AUX_BAD_AUDIO_FORMAT_RUN" bad_audio_format metadata-only
set +e
AUX_BAD_AUDIO_FORMAT_OUTPUT="$(PYTHONPATH="$AUX_MCAP_MODULES" PATH="$AUX_VERIFY_BIN:$PATH" \
    "$SCRIPT" verify "$AUX_BAD_AUDIO_FORMAT_RUN" --media-mode metadata-only --expect-audio 2>&1)"
AUX_BAD_AUDIO_FORMAT_STATUS=$?
set -e
[[ "$AUX_BAD_AUDIO_FORMAT_STATUS" -ne 0 ]]
grep -F "expected (48000, 1, 16, S16LE)" <<< "$AUX_BAD_AUDIO_FORMAT_OUTPUT" >/dev/null

AUX_SPARSE_PCM_RUN="$TMPDIR_TEST/aux_sparse_pcm"
make_aux_verify_run "$AUX_SPARSE_PCM_RUN" sparse_pcm embedded
set +e
AUX_SPARSE_PCM_OUTPUT="$(PYTHONPATH="$AUX_MCAP_MODULES" PATH="$AUX_VERIFY_BIN:$PATH" \
    "$SCRIPT" verify "$AUX_SPARSE_PCM_RUN" --media-mode embedded --expect-audio 2>&1)"
AUX_SPARSE_PCM_STATUS=$?
set -e
[[ "$AUX_SPARSE_PCM_STATUS" -ne 0 ]]
grep -F "ego_media/Audio epoch 0 contains only" <<< "$AUX_SPARSE_PCM_OUTPUT" >/dev/null

AUX_EMPTY_PCM_RUN="$TMPDIR_TEST/aux_empty_pcm"
make_aux_verify_run "$AUX_EMPTY_PCM_RUN" empty_pcm embedded
set +e
AUX_EMPTY_PCM_OUTPUT="$(PYTHONPATH="$AUX_MCAP_MODULES" PATH="$AUX_VERIFY_BIN:$PATH" \
    "$SCRIPT" verify "$AUX_EMPTY_PCM_RUN" --media-mode embedded --expect-audio 2>&1)"
AUX_EMPTY_PCM_STATUS=$?
set -e
[[ "$AUX_EMPTY_PCM_STATUS" -ne 0 ]]
grep -F "PCM byte count 0 does not match sample_count/profile" <<< "$AUX_EMPTY_PCM_OUTPUT" >/dev/null

rm "$PACKAGE_DIR/ego_mcap_export_media"
expect_failure env PATH="$MOCK_BIN:$PATH" "$PACKAGE_DIR/ego.sh" doctor

DIAGNOSTIC_RUN="$TMPDIR_TEST/diagnostic_off"
expect_failure "$SCRIPT" record --plugin "$MOCK_PLUGIN" --duration 1 --output "$DIAGNOSTIC_RUN" --record-device-diagnostics=off
grep -Fx -- "--record-device-diagnostics=off" "$MOCK_ARGS" >/dev/null
grep -F '"record_device_diagnostics": "off"' "$DIAGNOSTIC_RUN/requested_profile.json" >/dev/null
expect_failure "$SCRIPT" record --record-device-diagnostics=invalid
