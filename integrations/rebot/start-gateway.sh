#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Start the motorbridge WS gateway for the reBot B601-RS arm, then point
# Motorbridge Studio (https://motorbridge.github.io/motorbridge-studio/) at it.
#
# The gateway takes exclusive ownership of the CAN bus — stop it before running
# any motorbridge-cli command.
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Loopback needs no token. The gateway refuses any wider bind unless
# MOTORBRIDGE_WS_TOKEN is set, so reaching the arm from another machine is a
# deliberate two-variable opt-in and never a default:
#   MOTORBRIDGE_WS_TOKEN=<token> BIND=0.0.0.0:9002 ./start-gateway.sh
: "${BIND:=127.0.0.1:9002}"

# Router mode starts fine without CAN, so a missing bus is a warning. Only an
# absent or explicitly-down interface is reported; CAN links often read
# "unknown" while perfectly usable.
can_state="$(cat /sys/class/net/can0/operstate 2>/dev/null || true)"
if [[ -z "$can_state" ]]; then
  echo "warning: can0 not found — no motors will respond (see SKILL.md, Step 2)" >&2
elif [[ "$can_state" == "down" ]]; then
  echo "warning: can0 is down — run: sudo ip link set can0 up" >&2
fi

echo "gateway listening on ws://$BIND"
exec uv run --project "$here" motorbridge-gateway --bind "$BIND" "$@"
