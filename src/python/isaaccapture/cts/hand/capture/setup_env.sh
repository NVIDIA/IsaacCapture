#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Builds the venv the capture panel runs in: the isaaccapture wheel beside the checker's
# modules. Run the checker's setup_env.sh first; it generates the bindings.
set -euo pipefail
cd "$(dirname "$0")"
REPO="${REPO:-$(cd ../../../../../.. && pwd)}"
CHECKER="$REPO/src/python/isaaccapture/cts/hand/checker"

WHEEL=$(ls -t "$REPO"/install/wheels/isaaccapture-*.whl "$REPO"/build/wheels/isaaccapture-*.whl \
  2>/dev/null | head -1 || true)
if [[ -z "$WHEEL" ]]; then
  echo "no isaaccapture wheel in $REPO/{install,build}/wheels; build the repo first" >&2
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
# Resolve the wheel's isaacteleop dependency from the same directory.
uv pip install --python .venv/bin/python --find-links "$(dirname "$WHEEL")" "$WHEEL" \
  -r requirements.txt

SITE_PACKAGES=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
printf '%s\n' "$CHECKER/src" "$REPO/src/python/isaaccapture/cts/common/src" \
  > "$SITE_PACKAGES/hand_cts.pth"

.venv/bin/python - <<'PY'
import isaaccapture
from isaaccapture.retargeting_engine.deviceio_source_nodes import HandsSource  # noqa: F401
from hand_cts.mcap_source import McapFrameSource  # noqa: F401

print(f"env ready -- isaaccapture {isaaccapture.__version__}, checker importable")
PY
