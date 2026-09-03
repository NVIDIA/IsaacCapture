#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Rewrite RUNPATH/RPATH of vendored CasADi/IPOPT libs to $ORIGIN so colocated
# libcoinmumps.so.3 resolves without LD_LIBRARY_PATH or a build-machine path.
#
# Usage:
#   ./fix_vendored_rpath.sh <lib-directory>

set -euo pipefail

LIB_DIR="${1:-}"
if [[ -z "${LIB_DIR}" || ! -d "${LIB_DIR}" ]]; then
  echo "Usage: $0 <lib-directory>" >&2
  exit 1
fi
LIB_DIR="$(cd "${LIB_DIR}" && pwd)"

find_patchelf() {
  if command -v patchelf >/dev/null 2>&1; then
    command -v patchelf
    return 0
  fi
  # PyPI / uv wheel installs often land here when active venv is present.
  if command -v python3 >/dev/null 2>&1; then
    local py_bin
    py_bin="$(python3 - <<'PY'
import shutil, sys
print(shutil.which("patchelf") or "")
PY
)"
    if [[ -n "${py_bin}" ]]; then
      echo "${py_bin}"
      return 0
    fi
  fi
  return 1
}

PATCHELF="$(find_patchelf || true)"
if [[ -z "${PATCHELF}" ]]; then
  echo "WARNING: patchelf not found; skipping RPATH rewrite in ${LIB_DIR}" >&2
  echo "         Plugin binaries still prepend install/lib to LD_LIBRARY_PATH at startup." >&2
  exit 0
fi

echo "Rewriting RUNPATH -> \$ORIGIN in ${LIB_DIR} (using ${PATCHELF})"
shopt -s nullglob
changed=0
for f in \
  "${LIB_DIR}"/libavatar_sdk.so \
  "${LIB_DIR}"/libavatar_sdk_wrapper.so \
  "${LIB_DIR}"/libcasadi.so* \
  "${LIB_DIR}"/libcasadi_nlpsol_ipopt.so \
  "${LIB_DIR}"/libipopt.so* \
  "${LIB_DIR}"/libsipopt.so* \
  "${LIB_DIR}"/libcoinmumps.so* \
  "${LIB_DIR}"/libcoinmetis.so*
do
  [[ -e "$f" ]] || continue
  [[ -L "$f" ]] && continue   # only rewrite real ELF files
  if ! file -b "$f" | grep -q 'ELF'; then
    continue
  fi
  "${PATCHELF}" --force-rpath --set-rpath '$ORIGIN' "$f"
  changed=$((changed + 1))
done
shopt -u nullglob
echo "Updated RPATH on ${changed} libraries."
