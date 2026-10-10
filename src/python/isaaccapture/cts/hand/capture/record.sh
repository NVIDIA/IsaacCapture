#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Records one take of the hand script. Run it again for another take; nothing is ever
# overwritten.
#
#   ./record.sh DEVICE [capture_panel.py options]
#   ./record.sh quest3
#   ./record.sh manus --plugin manus_hand_plugin
#   WUJI_GLOVE_WRIST_SOURCE=controller ./record.sh wuji --plugin wuji_glove_plugin
#
#   ~/isaaccapture-captures/<device>_<date>_<time>/<device>_<date>_<time>-hand.mcap
#                                                                         -hand.labels.json
#                                                                         -hand.json
#                                                                         -hand.log

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAPTURES="$HOME/isaaccapture-captures"

if [[ $# -lt 1 || "$1" == -* ]]; then
    echo "usage: $0 DEVICE [capture_panel.py options]" >&2
    exit 2
fi
DEVICE="$1"
shift

if [[ ! -x "$HERE/.venv/bin/python" ]]; then
    echo "missing $HERE/.venv; run $HERE/setup_env.sh" >&2
    exit 1
fi

take="${DEVICE}_$(date +%Y-%m-%d_%H%M%S)"
mkdir -p "$CAPTURES/$take"
stem="$CAPTURES/$take/$take-hand"

echo "take: $stem.mcap"
echo

"$HERE/.venv/bin/python" -u "$HERE/capture_panel.py" "$stem.mcap" --device "$DEVICE" "$@" \
    2>&1 | tee "$stem.log"

echo
echo "check it: $HERE/../checker/.venv/bin/python -m hand_cts.cli $stem.mcap"
