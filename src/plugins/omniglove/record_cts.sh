#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../../.." && pwd)"
allow_dirty=false
capture_args=()
operator_note=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wrist-config)
            source "${2:?--wrist-config requires an environment file}"
            shift 2
            ;;
        --allow-dirty)
            allow_dirty=true
            shift
            ;;
        --note)
            operator_note="${2:?--note requires text}"
            shift 2
            ;;
        --note=*) operator_note="${1#--note=}"; shift ;;
        -h | --help)
            cat <<'EOF'
Usage: record_cts.sh [--wrist-config PATH] [--allow-dirty] [capture options]

Formal recordings require a clean, pushed plugin commit and an unchanged CTS.
--allow-dirty permits development takes; their provenance remains unchanged.
Set OMNIGLOVE_WRIST_SOURCE and measured per-hand controller transforms, or
load them from --wrist-config. Other options go to the official recorder.
EOF
            exit 0
            ;;
        --plugin | --plugin=* | --wrist-source | --wrist-source=*)
            echo "error: plugin and wrist source are selected by this script" >&2
            exit 1
            ;;
        *) capture_args+=("$1"); shift ;;
    esac
done

cd -- "$repo_dir"
if [[ -n "$(git status --porcelain -- src/python/isaaccapture/cts)" ]]; then
    echo "error: CTS has local changes; restore the official tools before recording" >&2
    exit 1
fi
if [[ "$allow_dirty" == false ]]; then
    if [[ -n "$(git status --porcelain)" ]] ||
        ! git cat-file -e HEAD:src/plugins/omniglove/plugin.yaml 2>/dev/null; then
        echo "error: commit the final plugin and keep the repository clean before a formal take" >&2
        exit 1
    fi
    if ! git rev-parse --verify '@{upstream}' >/dev/null 2>&1 ||
        ! git merge-base --is-ancestor HEAD '@{upstream}'; then
        echo "error: push this commit and set its upstream before a formal take" >&2
        exit 1
    fi
fi

wrist_source="${OMNIGLOVE_WRIST_SOURCE:-hand_tracking}"
case "$wrist_source" in
    hand_tracking) ;;
    controller | auto)
        : "${OMNIGLOVE_AIM_TO_WRIST_LEFT:?set the measured left aim-to-wrist transform}"
        : "${OMNIGLOVE_AIM_TO_WRIST_RIGHT:?set the measured right aim-to-wrist transform}"
        ;;
    *) echo "error: unsupported OMNIGLOVE_WRIST_SOURCE=$wrist_source" >&2; exit 1 ;;
esac
export OMNIGLOVE_WRIST_SOURCE="$wrist_source"
if [[ "$wrist_source" != hand_tracking ]]; then
    export OMNIGLOVE_AIM_TO_WRIST_LEFT OMNIGLOVE_AIM_TO_WRIST_RIGHT
fi
plugin_binary="$repo_dir/install/plugins/omniglove/omniglove_plugin"
if [[ ! -x "$plugin_binary" ]]; then
    echo "error: build and install the OmniGlove plugin first" >&2
    exit 1
fi
plugin_sha="$(sha256sum "$plugin_binary")"
plugin_sha="${plugin_sha%% *}"
exec src/python/isaaccapture/cts/hand/capture/record.sh omniglove \
    --plugin omniglove_plugin \
    --wrist-source "$wrist_source" \
    --note "OmniGlove; binary_sha256=$plugin_sha; wrist=$wrist_source; LEFT=${OMNIGLOVE_AIM_TO_WRIST_LEFT:-unused}; RIGHT=${OMNIGLOVE_AIM_TO_WRIST_RIGHT:-unused}; $operator_note" \
    "${capture_args[@]}"
