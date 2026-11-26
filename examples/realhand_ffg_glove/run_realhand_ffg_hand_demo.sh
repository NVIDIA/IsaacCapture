#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
# The calibration fitter shares the CPU with RTX.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
BUILD="${REALHAND_CAPTURE_BUILD:-$REPO/build}"
DEMO_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/realhand_ffg_hand_demo"
CAPTURE_PYTHON="${REALHAND_CAPTURE_PYTHON:-$REPO/.venv/bin/python}"

if [[ ! -x "$CAPTURE_PYTHON" ]]; then
    printf 'Isaac Capture Python not found: %s\nSet REALHAND_CAPTURE_PYTHON to a matching Isaac Capture interpreter.\n' "$CAPTURE_PYTHON" >&2
    exit 1
fi
if [[ ! -f "$BUILD/CMakeCache.txt" ]]; then
    printf 'Configure Isaac Capture first: cmake -S "%s" -B "%s" -DBUILD_PLUGIN_REALHAND_FFG_GLOVE=ON\n' "$REPO" "$BUILD" >&2
    exit 1
fi
cmake -S "$REPO" -B "$BUILD" -DBUILD_PLUGIN_REALHAND_FFG_GLOVE=ON
cmake --build "$BUILD" --target realhand_ffg_glove_plugin python_package -j 4
cmake --install "$BUILD" --component realhand_ffg_glove --prefix "$DEMO_CACHE/install"
BUILD_CONFIG="${REALHAND_CAPTURE_BUILD_CONFIG:-Release}"
export PYTHONPATH="$SCRIPT_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
exec python -m isaaccapture_examples.realhand_ffg_glove.realhand_ffg_hand_demo \
    --capture-python "$CAPTURE_PYTHON" \
    --capture-package "$BUILD/python_package/$BUILD_CONFIG" \
    --plugin-path "$DEMO_CACHE/install/plugins" \
    "$@"
