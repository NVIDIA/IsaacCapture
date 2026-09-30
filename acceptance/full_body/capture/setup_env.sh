#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Builds the venv the capture panel runs in: the locally built isaacteleop wheel beside
# the checker's own modules, so one interpreter can drive a device and read the
# recording back. Run the checker's setup_env.sh first -- the flatc-generated bindings
# it emits are what this venv decodes the take with.
set -euo pipefail
cd "$(dirname "$0")"
REPO="${REPO:-$(cd ../../.. && pwd)}"
CHECKER="$REPO/acceptance/full_body/checker"

WHEEL=$(ls -t "$REPO"/install/wheels/isaacteleop-*.whl "$REPO"/build/wheels/isaacteleop-*.whl \
  2>/dev/null | head -1 || true)
if [[ -z "$WHEEL" ]]; then
  echo "no isaacteleop wheel in $REPO/{install,build}/wheels; build the repo first" >&2
  exit 1
fi

if [[ ! -f "$CHECKER/generated/core/FullBodyPoseRecord.py" ]]; then
  echo "run $CHECKER/setup_env.sh first; the panel decodes the take it writes" >&2
  exit 1
fi

# 3.12 because the wheel is built cp312; a wheel for another version will not install.
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.12 .venv
fi
uv pip install --python .venv/bin/python "$WHEEL" -r requirements.txt

# The checker is read out of its src/ rather than installed, the same way its own venv
# does it, so the two halves cannot drift to different copies of one module.
SITE_PACKAGES=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
printf '%s\n' "$CHECKER/src" "$REPO/acceptance/common/src" \
  > "$SITE_PACKAGES/full_body_acceptance.pth"

# One interpreter, both halves: the panel records through TeleopSession and labels the
# result through the checker's decoder. If this line fails the panel cannot run.
.venv/bin/python - <<'PY'
import isaacteleop
import full_body_acceptance.panel.track  # noqa: F401
from full_body_acceptance.mcap_source import McapFrameSource  # noqa: F401

print(f"env ready -- isaacteleop {isaacteleop.__version__}, checker importable")
PY
