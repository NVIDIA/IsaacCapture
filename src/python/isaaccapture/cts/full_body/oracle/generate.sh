#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Builds the fixture set into fixtures/ and reads it back. Run it after a fresh clone and
# after changing a generator; the .mcap files are derived and not in git.
# Uses the checker's venv: run the checker's setup_env.sh first.
set -euo pipefail
cd "$(dirname "$0")"
CHECKER="$(cd ../checker && pwd)"

if [[ ! -x "$CHECKER/.venv/bin/python" ]]; then
  echo "missing $CHECKER/.venv; run $CHECKER/setup_env.sh first" >&2
  exit 1
fi
PY="$CHECKER/.venv/bin/python"

"$PY" generate_fixtures.py
"$PY" verify_fixtures.py
"$PY" posture_verify.py

# Fail if the committed index changed.
if ! git diff --quiet -- fixtures_index.json; then
  echo
  echo "fixtures_index.json changed -- review and commit it, or the oracle no longer" >&2
  echo "describes what this generator produces." >&2
  exit 1
fi
