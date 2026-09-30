#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Builds the venv the capture panel runs in: the locally built isaacteleop wheel beside
# the checker's modules, read from its src/. Run the checker's setup_env.sh first; the
# panel decodes the take it writes through the bindings that emits.
set -euo pipefail
cd "$(dirname "$0")"
REPO="${REPO:-$(cd ../../.. && pwd)}"
CHECKER="$REPO/acceptance/hand/checker"

WHEEL=$(ls -t "$REPO"/install/wheels/isaacteleop-*.whl "$REPO"/build/wheels/isaacteleop-*.whl \
  2>/dev/null | head -1 || true)
if [[ -z "$WHEEL" ]]; then
  echo "no isaacteleop wheel in $REPO/{install,build}/wheels; build the repo first" >&2
  exit 1
fi

if [[ ! -f "$CHECKER/generated/core/HandPoseRecord.py" ]]; then
  echo "run $CHECKER/setup_env.sh first; the panel decodes the take it writes" >&2
  exit 1
fi

# The wheel is cp312.
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.12 .venv
fi
uv pip install --python .venv/bin/python "$WHEEL" -r requirements.txt

SITE_PACKAGES=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
printf '%s\n' "$CHECKER/src" "$REPO/acceptance/common/src" \
  > "$SITE_PACKAGES/hand_acceptance.pth"

.venv/bin/python - <<'PY'
import isaacteleop
from isaacteleop.retargeting_engine.deviceio_source_nodes import HandsSource  # noqa: F401
from hand_acceptance.mcap_source import McapFrameSource  # noqa: F401

print(f"env ready -- isaacteleop {isaacteleop.__version__}, checker importable")
PY
