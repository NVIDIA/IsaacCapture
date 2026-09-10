#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Stage Avatar SDK headers, runtime libraries, and share data into
# vendor/avatar-sdk next to this script. The vendor tree is gitignored.
#
# Usage:
#   ./vendor_avatar_sdk.sh [avatar-sdk-root]
#
# avatar-sdk-root defaults to $AVATAR_SDK_ROOT or /opt/avatar-sdk.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${1:-}" ]]; then
  SRC_ROOT="$1"
elif [[ -n "${AVATAR_SDK_ROOT:-}" ]]; then
  SRC_ROOT="${AVATAR_SDK_ROOT}"
else
  SRC_ROOT="/opt/avatar-sdk"
fi
DEST="${SCRIPT_DIR}/vendor/avatar-sdk"

if [[ ! -f "${SRC_ROOT}/include/avatar_sdk/AvatarSDK.h" ]]; then
  echo "ERROR: Avatar SDK headers not found under '${SRC_ROOT}/include/avatar_sdk'." >&2
  exit 1
fi
if [[ ! -f "${SRC_ROOT}/lib/libavatar_sdk.so" && ! -f "${SRC_ROOT}/lib/libavatar_sdk_wrapper.so" ]]; then
  echo "ERROR: libavatar_sdk[_wrapper].so not found under '${SRC_ROOT}/lib'." >&2
  exit 1
fi
if [[ -f "${SRC_ROOT}/share/sdk_config.json" ]]; then
  SDK_CONFIG_SRC="${SRC_ROOT}/share/sdk_config.json"
elif [[ -f "${SRC_ROOT}/../config/sdk_config.json" ]]; then
  SDK_CONFIG_SRC="${SRC_ROOT}/../config/sdk_config.json"
else
  echo "ERROR: sdk_config.json not found under ${SRC_ROOT}/share." >&2
  exit 1
fi
if [[ -d "${SRC_ROOT}/share/hand_fk" ]]; then
  HAND_FK_SRC="${SRC_ROOT}/share/hand_fk"
elif [[ -d "${SRC_ROOT}/../src/hand_fk/data" ]]; then
  HAND_FK_SRC="${SRC_ROOT}/../src/hand_fk/data"
else
  echo "ERROR: hand_fk data not found; cannot stage the ROBOT dataset runtime." >&2
  exit 1
fi
if [[ ! -d "${SRC_ROOT}/share/wave-sdk" ]]; then
  echo "ERROR: wave-sdk not found under ${SRC_ROOT}/share; cannot stage USB transport support." >&2
  exit 1
fi

# Prefer mode/timestamps; skip ownership (fails under some containers / sandboxes).
if cp --help 2>&1 | grep -q -- '--no-preserve'; then
  cp_sdk() { cp -a --no-preserve=ownership "$@"; }
else
  cp_sdk() { cp -R "$@"; }
fi

echo "Staging Avatar SDK: ${SRC_ROOT} -> ${DEST}"
rm -rf "${DEST}"
mkdir -p "${DEST}/include" "${DEST}/lib" "${DEST}/share"

cp_sdk "${SRC_ROOT}/include/avatar_sdk" "${DEST}/include/"

# Runtime libs: SDK + CasADi/IPOPT. Skip missing literals (no wrapper.so on some debs).
shopt -s nullglob
candidates=(
  "${SRC_ROOT}/lib/libavatar_sdk.so"
  "${SRC_ROOT}/lib/libavatar_sdk_wrapper.so"
  "${SRC_ROOT}/lib/libcasadi.so"*
  "${SRC_ROOT}/lib/libcasadi_nlpsol_ipopt.so"
  "${SRC_ROOT}/lib/libipopt.so"*
  "${SRC_ROOT}/lib/libsipopt.so"*
  "${SRC_ROOT}/lib/libcoinmumps.so"*
  "${SRC_ROOT}/lib/libcoinmetis.so"*
)
shopt -u nullglob
copied=0
for f in "${candidates[@]}"; do
  if [[ -e "$f" ]]; then
    cp_sdk "$f" "${DEST}/lib/"
    copied=$((copied + 1))
  fi
done
if [[ "${copied}" -eq 0 ]]; then
  echo "ERROR: no runtime libraries copied from ${SRC_ROOT}/lib" >&2
  exit 1
fi

cp_sdk "${SDK_CONFIG_SRC}" "${DEST}/share/sdk_config.json"
cp_sdk "${HAND_FK_SRC}" "${DEST}/share/hand_fk"
cp_sdk "${SRC_ROOT}/share/wave-sdk" "${DEST}/share/"

if [[ -f "${SRC_ROOT}/share/Version" ]]; then
  cp_sdk "${SRC_ROOT}/share/Version" "${DEST}/share/"
fi

# Relative roots resolve next to the bundled sdk_config.json (plugin dir).
# Plugin rewrites them to absolute paths if the config file is copied elsewhere.
python3 - "${DEST}/share/sdk_config.json" <<'PY'
import json, sys
path = sys.argv[1]
with open(path) as f:
    cfg = json.load(f)
cfg["hand_fk_config_root"] = "hand_fk/"
cfg["transport_link"] = "usb_serial"
cfg["wave_sdk_root"] = ["wave-sdk"]
with open(path, "w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
print(f"normalized sdk_config.json in {path}")
PY

FIX_RPATH="${SCRIPT_DIR}/fix_vendored_rpath.sh"
if [[ -f "${FIX_RPATH}" ]]; then
  chmod +x "${FIX_RPATH}" 2>/dev/null || true
  "${FIX_RPATH}" "${DEST}/lib"
fi

echo "Done. Vendored tree:"
du -sh "${DEST}" "${DEST}/lib" "${DEST}/share" 2>/dev/null || true
