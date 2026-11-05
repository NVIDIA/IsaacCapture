#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Builds the venv and the flatc-generated Python bindings this checker decodes with.
set -euo pipefail
cd "$(dirname "$0")"
REPO="${REPO:-$(cd ../../../../../.. && pwd)}"
FBS="$REPO/src/core/schema/fbs"

WITH_PANEL=0
for arg in "$@"; do
  case "$arg" in
    --panel) WITH_PANEL=1 ;;
    *) echo "usage: $0 [--panel]" >&2; exit 2 ;;
  esac
done

# Pinned to flatc v24.3.25; another version produces a different .bfbs than the repo golden.
case "$(uname -s)" in
  Darwin) FLATC_ASSET=Mac.flatc.binary.zip ;;
  Linux)  FLATC_ASSET='Linux.flatc.binary.clang++-15.zip' ;;
  *) echo "unsupported host $(uname -s)" >&2; exit 1 ;;
esac

mkdir -p toolchain
if [[ ! -x toolchain/flatc ]]; then
  curl -sSL -o "toolchain/$FLATC_ASSET" \
    "https://github.com/google/flatbuffers/releases/download/v24.3.25/$FLATC_ASSET"
  (cd toolchain && unzip -oq "$FLATC_ASSET")
fi
toolchain/flatc --version

if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.12 .venv
fi
uv pip install --python .venv/bin/python -r requirements.txt -r requirements-dev.txt
if [[ "$WITH_PANEL" == 1 ]]; then
  uv pip install --python .venv/bin/python -r requirements-panel.txt
fi

SITE_PACKAGES=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
printf '%s\n' "$PWD/src" "$REPO/src/python/isaaccapture/cts/common/src" > "$SITE_PACKAGES/hand_cts.pth"

# The generated .bfbs must be byte-identical to the repo golden.
mkdir -p generated build/bfbs
toolchain/flatc --cpp --cpp-ptr-type std::shared_ptr --gen-object-api --gen-mutable \
  --schema --bfbs-gen-embed --reflect-names --gen-name-strings -b \
  -I "$FBS" -o build/bfbs "$FBS/hand.fbs"
if cmp build/bfbs/hand.bfbs "$REPO/src/core/schema/golden/hand.bfbs"; then
  echo "bfbs matches the repo golden"
else
  echo "bfbs differs from the repo golden" >&2
  exit 1
fi

# Generated modules import `core.X`; generated/ must be on sys.path.
toolchain/flatc --python --gen-object-api --gen-mutable -I "$FBS" -o generated \
  "$FBS/hand.fbs" "$FBS/pose.fbs" "$FBS/point.fbs" "$FBS/quaternion.fbs" \
  "$FBS/timestamp.fbs"

echo "env ready"
