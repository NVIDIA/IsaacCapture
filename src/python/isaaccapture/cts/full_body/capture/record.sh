#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Records one take of the motion script. Run it again for another take; nothing is
# ever overwritten.
#
#   ./record.sh [device]
#
#   ~/isaaccapture-captures/<device>_<date>_<time>/<device>_<date>_<time>-body.mcap
#                                                                       -body.labels.json
#                                                                       -body.log
#                                                                       -body.json
#
# One directory per take. This names the take and records what produced it; the panel,
# cues, windows and sidecar are capture_panel.py.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAPTURES="$HOME/isaaccapture-captures"

DEVICE="${1:-pico4u}"
shift || true

if [[ ! -x "$HERE/.venv/bin/python" ]]; then
    echo "missing $HERE/.venv; run $HERE/setup_env.sh" >&2
    exit 1
fi

take="${DEVICE}_$(date +%Y-%m-%d_%H%M%S)"
mkdir -p "$CAPTURES/$take"
stem="$CAPTURES/$take/$take-body"

# git -C: the caller's working directory is not this repository.
python3 - "$stem.json" "$DEVICE" "$HERE" <<'PY'
import datetime, json, pathlib, subprocess, sys
path, device, here = sys.argv[1:4]
def run(*args):
    try:
        return subprocess.run(args, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None
pathlib.Path(path).write_text(json.dumps({
    "recorded_at": datetime.datetime.now().astimezone().isoformat(),
    "device": device,
    "repo_commit": run("git", "-C", here, "rev-parse", "HEAD"),
}, indent=2) + "\n")
PY

echo "take: $stem.mcap"
echo

"$HERE/.venv/bin/python" -u "$HERE/capture_panel.py" "$stem.mcap" "$@" \
    2>&1 | tee "$stem.log"

echo
echo "run again for another take; this one is kept either way"
