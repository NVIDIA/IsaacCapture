#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT="${1:?Pass the EGO installer path}"
TMPDIR_TEST="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_TEST"' EXIT
SDK_RELEASE="v2.10.6-EGO-20261019"
SDK_NAME="OrbbecSDK_v2.10.6_202610191449_d49af169_linux_x86_64"
SDK_SHA256="378761b4331b7c6ce75d3c7dc61817ed27a5cacb3265a77477587c4d9e3bf11a"
SDK_URL="https://github.com/orbbec/OrbbecSDK_v2-EGO/releases/download/$SDK_RELEASE/$SDK_NAME.tar.gz"
REPO="$TMPDIR_TEST/repo with spaces"
MOCK_BIN="$TMPDIR_TEST/bin"
MOCK_LOG="$TMPDIR_TEST/commands"
BUILD_LOG="$TMPDIR_TEST/build"
MOCK_ARCHIVE="$TMPDIR_TEST/sdk.tar.gz"
REAL_TAR="$(command -v tar)"
export MOCK_LOG BUILD_LOG MOCK_ARCHIVE SDK_SHA256 REAL_TAR
unset ORBBEC_SDK_ROOT
mkdir -p "$REPO/src/plugins/ego" "$MOCK_BIN" "$TMPDIR_TEST/fixture/$SDK_NAME"
cp "$SCRIPT" "$REPO/src/plugins/ego/install.sh"
touch "$REPO/CMakeLists.txt" "$MOCK_LOG" "$BUILD_LOG"
INSTALLER="$REPO/src/plugins/ego/install.sh"
SDK_CACHE="$REPO/build/ego-sdk/$SDK_RELEASE"

make_sdk() {
    local root="$1"
    mkdir -p "$root/include/libobsensor" "$root/lib/extensions" "$root/shared"
    printf 'EGO header\n' > "$root/include/libobsensor/ObSensor.hpp"
    touch "$root/lib/OrbbecSDKConfig.cmake" "$root/lib/libOrbbecSDK.so.2"
    cat > "$root/shared/99-obsensor-libusb.rules" <<'EOF'
SUBSYSTEMS=="usb", ATTRS{idVendor}=="2bc5", ATTRS{idProduct}=="1201", MODE:="0666", OWNER:="root", GROUP:="video", SYMLINK+="Ego"
EOF
}
make_sdk "$TMPDIR_TEST/fixture/$SDK_NAME"
"$REAL_TAR" -czf "$MOCK_ARCHIVE" -C "$TMPDIR_TEST/fixture" "$SDK_NAME"

cat > "$REPO/src/plugins/ego/ego.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" > "$BUILD_LOG"
printf 'ego.sh\n' >> "$MOCK_LOG"
EOF
cat > "$MOCK_BIN/uname" <<'EOF'
#!/usr/bin/env bash
case "$1" in
    -s) echo "${MOCK_PLATFORM:-Linux}" ;;
    -m) echo "${MOCK_ARCH:-x86_64}" ;;
    *) exit 1 ;;
esac
EOF
cat > "$MOCK_BIN/cmake" <<'EOF'
#!/usr/bin/env bash
[[ "$1" == --version ]] || exit 1
echo "cmake version ${MOCK_CMAKE_VERSION:-3.30.0}"
EOF
cat > "$MOCK_BIN/uv" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
cat > "$MOCK_BIN/curl" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'curl %s\n' "$*" >> "$MOCK_LOG"
output=""
while (( $# )); do
    if [[ "$1" == --output ]]; then output="$2"; shift 2; else shift; fi
done
[[ -n "$output" ]]
if [[ "${MOCK_DOWNLOAD_FAIL:-0}" == 1 ]]; then
    printf 'partial download\n' > "$output"
    exit 22
elif [[ "${MOCK_DOWNLOAD_CORRUPT:-0}" == 1 ]]; then
    printf 'wrong archive\n' > "$output"
else
    cp "$MOCK_ARCHIVE" "$output"
fi
EOF
cat > "$MOCK_BIN/sha256sum" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'sha256sum %s\n' "$1" >> "$MOCK_LOG"
if cmp -s "$1" "$MOCK_ARCHIVE"; then
    printf '%s  %s\n' "$SDK_SHA256" "$1"
else
    printf '%064d  %s\n' 0 "$1"
fi
EOF
cat > "$MOCK_BIN/tar" <<'EOF'
#!/usr/bin/env bash
printf 'tar %s\n' "$*" >> "$MOCK_LOG"
exec "$REAL_TAR" "$@"
EOF
cat > "$MOCK_BIN/dpkg-query" <<'EOF'
#!/usr/bin/env bash
if [[ "${MOCK_MISSING_DEPS:-0}" == 1 ]]; then exit 1; fi
printf 'install ok installed\n'
EOF
cat > "$MOCK_BIN/sudo" <<'EOF'
#!/usr/bin/env bash
printf 'sudo %s\n' "$*" >> "$MOCK_LOG"
"$@"
EOF
for command_name in apt-get install udevadm; do
    cat > "$MOCK_BIN/$command_name" <<'EOF'
#!/usr/bin/env bash
printf '%s %s\n' "${0##*/}" "$*" >> "$MOCK_LOG"
if [[ "${MOCK_SETUP_FAIL:-0}" == 1 ]]; then exit 23; fi
EOF
done
chmod +x "$REPO/src/plugins/ego/"*.sh "$MOCK_BIN/"*
export PATH="$MOCK_BIN:$PATH"

expect_failure() {
    local message="$1"
    shift
    if "$@" > "$TMPDIR_TEST/failure.log" 2>&1; then
        echo "Expected command to fail: $*" >&2
        exit 1
    fi
    if ! grep -F "$message" "$TMPDIR_TEST/failure.log" >/dev/null; then
        cat "$TMPDIR_TEST/failure.log" >&2
        echo "Missing expected error: $message" >&2
        exit 1
    fi
}

"$INSTALLER" --help >/dev/null
expect_failure 'requires a value' "$INSTALLER" --sdk-root
expect_failure 'requires a value' "$INSTALLER" --jobs --preview
expect_failure 'Unknown option' "$INSTALLER" --unknown
expect_failure 'Unknown preset' "$INSTALLER" --preset invalid
expect_failure 'positive integer' "$INSTALLER" --jobs 0
expect_failure 'Only Linux x86_64' env MOCK_ARCH=aarch64 "$INSTALLER"
expect_failure 'Only Linux x86_64' env MOCK_PLATFORM=Darwin "$INSTALLER"
expect_failure 'too old' env MOCK_CMAKE_VERSION=3.23.0 "$INSTALLER" --skip-deps --skip-udev
[[ ! -s "$MOCK_LOG" ]]

LOCAL_SDK="$TMPDIR_TEST/local SDK"
make_sdk "$LOCAL_SDK"
"$INSTALLER" --sdk-root "$LOCAL_SDK" --skip-deps --skip-udev --preset py3.12 \
    --jobs 2 --preview --install-prefix "$TMPDIR_TEST/install prefix" >/dev/null
printf '%s\n' build --sdk-root "$LOCAL_SDK" --preset py3.12 --jobs 2 \
    --install-prefix "$TMPDIR_TEST/install prefix" --preview > "$TMPDIR_TEST/expected-build"
cmp "$BUILD_LOG" "$TMPDIR_TEST/expected-build"
[[ "$(cat "$MOCK_LOG")" == ego.sh ]]
env ORBBEC_SDK_ROOT="$LOCAL_SDK" "$INSTALLER" --skip-deps --skip-udev >/dev/null
grep -Fx "$LOCAL_SDK" "$BUILD_LOG" >/dev/null
env ORBBEC_SDK_ROOT="$TMPDIR_TEST/missing" "$INSTALLER" --sdk-root "$LOCAL_SDK" --skip-deps --skip-udev >/dev/null

expect_failure 'missing include/libobsensor/ObSensor.hpp' "$INSTALLER" --sdk-root "$TMPDIR_TEST/missing" --skip-deps --skip-udev
cp "$LOCAL_SDK/shared/99-obsensor-libusb.rules" "$TMPDIR_TEST/original-rules"
printf '# ATTRS{idVendor}=="2bc5", ATTRS{idProduct}=="1201"\n' > "$LOCAL_SDK/shared/99-obsensor-libusb.rules"
expect_failure 'lack the EGO USB device 2bc5:1201' "$INSTALLER" --sdk-root "$LOCAL_SDK" --skip-deps --skip-udev
cp "$TMPDIR_TEST/original-rules" "$LOCAL_SDK/shared/99-obsensor-libusb.rules"

: > "$MOCK_LOG"
env MOCK_MISSING_DEPS=1 "$INSTALLER" --sdk-root "$LOCAL_SDK" --container --preview > "$TMPDIR_TEST/container.log"
grep -F 'apt-get update' "$MOCK_LOG" >/dev/null
grep -F 'apt-get install -y build-essential' "$MOCK_LOG" >/dev/null
grep -F 'libsdl2-dev libavcodec-dev libavutil-dev libswscale-dev' "$MOCK_LOG" >/dev/null
grep -F 'Container: skipping host udev setup' "$TMPDIR_TEST/container.log" >/dev/null
grep -F 'host for EGO USB 2bc5:1201, then reconnect' "$TMPDIR_TEST/container.log" >/dev/null
if grep -Eq '^install |^udevadm |^sudo .*ego.sh|^curl ' "$MOCK_LOG"; then
    echo 'Local container setup must avoid udev, download, and privileged build commands' >&2
    exit 1
fi

: > "$MOCK_LOG"
expect_failure '' env MOCK_MISSING_DEPS=1 MOCK_SETUP_FAIL=1 "$INSTALLER" --container
if grep -Eq '^curl |^ego.sh$' "$MOCK_LOG"; then
    echo 'A dependency installation failure must stop before downloading or building' >&2
    exit 1
fi

: > "$MOCK_LOG"
"$INSTALLER" --skip-deps --skip-udev >/dev/null
grep -F "$SDK_URL" "$MOCK_LOG" >/dev/null
grep -F -- '--fail --location --show-error --retry 3 --connect-timeout 30 --max-time 600' "$MOCK_LOG" >/dev/null
[[ -f "$SDK_CACHE/$SDK_NAME.tar.gz" ]]
[[ -f "$SDK_CACHE/$SDK_NAME/include/libobsensor/ObSensor.hpp" ]]
grep -Fx "$SDK_CACHE/$SDK_NAME" "$BUILD_LOG" >/dev/null
[[ "$(sed -n '2s/ .*//p' "$MOCK_LOG")" == sha256sum ]]

# Each run verifies the archive and repairs changed extracted SDK files.
printf 'changed header\n' > "$SDK_CACHE/$SDK_NAME/include/libobsensor/ObSensor.hpp"
: > "$MOCK_LOG"
"$INSTALLER" --skip-deps --skip-udev > "$TMPDIR_TEST/reuse.log"
grep -F 'Reusing SHA-256 verified SDK archive' "$TMPDIR_TEST/reuse.log" >/dev/null
grep -Fx 'EGO header' "$SDK_CACHE/$SDK_NAME/include/libobsensor/ObSensor.hpp" >/dev/null
[[ "$(grep -c '^sha256sum ' "$MOCK_LOG")" == 1 ]]
if grep -q '^curl ' "$MOCK_LOG"; then
    echo 'An intact cached SDK archive should avoid downloading again' >&2
    exit 1
fi

printf 'corrupt cached archive\n' > "$SDK_CACHE/$SDK_NAME.tar.gz"
: > "$MOCK_LOG"
"$INSTALLER" --skip-deps --skip-udev > "$TMPDIR_TEST/repair.log"
grep -F 'Cached SDK SHA-256 mismatch' "$TMPDIR_TEST/repair.log" >/dev/null
[[ "$(grep -c '^curl ' "$MOCK_LOG")" == 1 ]]
[[ "$(grep -c '^sha256sum ' "$MOCK_LOG")" == 2 ]]
cmp "$SDK_CACHE/$SDK_NAME.tar.gz" "$MOCK_ARCHIVE"

rm "$SDK_CACHE/$SDK_NAME.tar.gz"
: > "$MOCK_LOG"
expect_failure 'SDK SHA-256 mismatch' env MOCK_DOWNLOAD_CORRUPT=1 "$INSTALLER" --skip-deps --skip-udev
[[ ! -f "$SDK_CACHE/$SDK_NAME.tar.gz" ]]
if grep -Eq '^tar |^ego.sh$' "$MOCK_LOG"; then
    echo 'A checksum failure must stop before extraction or build' >&2
    exit 1
fi
expect_failure '' env MOCK_DOWNLOAD_FAIL=1 "$INSTALLER" --skip-deps --skip-udev
[[ ! -f "$SDK_CACHE/$SDK_NAME.tar.gz" ]]
[[ -z "$(find "$SDK_CACHE" -maxdepth 1 -name '.install.*' -print -quit)" ]]
echo 'EGO installer regression checks passed.'
