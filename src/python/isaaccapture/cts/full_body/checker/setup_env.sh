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

# flatc v24.3.25, as pinned by deps/third_party/CMakeLists.txt; other versions give a different .bfbs.
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

# The packages are read from src/ rather than installed.
SITE_PACKAGES=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
printf '%s\n' "$PWD/src" "$REPO/src/python/isaaccapture/cts/common/src" > "$SITE_PACKAGES/full_body_cts.pth"

# Schema match: the .bfbs must be byte-identical to the repo golden. It is not used for decoding.
mkdir -p generated build/bfbs
toolchain/flatc --cpp --cpp-ptr-type std::shared_ptr --gen-object-api --gen-mutable \
  --schema --bfbs-gen-embed --reflect-names --gen-name-strings -b \
  -I "$FBS" -o build/bfbs "$FBS/full_body.fbs"
cmp build/bfbs/full_body.bfbs "$REPO/src/core/schema/golden/full_body.bfbs" \
  && echo "bfbs matches the repo golden"

toolchain/flatc --python --gen-object-api --gen-mutable -I "$FBS" -o generated \
  "$FBS/full_body.fbs" "$FBS/pose.fbs" "$FBS/point.fbs" "$FBS/quaternion.fbs" \
  "$FBS/timestamp.fbs"

echo "env ready"
