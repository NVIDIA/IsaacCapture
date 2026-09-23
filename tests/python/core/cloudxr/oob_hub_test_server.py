# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone real OOBControlHub, for a real HeadsetControlChannel to connect to.

Not a pytest test: a small asyncio script a JS/TS test spawns as a subprocess, so a
real ``HeadsetControlChannel`` (deps/cloudxr/webxr_client/helpers/controlChannel.ts) can
talk to a real ``OOBControlHub`` over an actual (plain ws://, no TLS - the proxy's TLS
termination is orthogonal to the hub's own protocol) loopback WebSocket, with neither
side mocked.

Protocol on stdout (one line per event, flushed immediately):
  READY <port>    - listening, ready for a client to connect
  SNAPSHOT <json> - hub.get_snapshot(), printed on every change to the
                    (clientId, connected, streaming, metricsByCadence) tuple of any
                    headset - i.e. on register/disconnect, sendStreamStatus, and
                    clientMetrics
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Not run under pytest (conftest.py never loads), so add tests/python to sys.path
# ourselves the same way conftest.py does, to reach the repo_paths helper.
_TESTS_PYTHON = Path(__file__).resolve().parents[2]
if str(_TESTS_PYTHON) not in sys.path:
    sys.path.insert(0, str(_TESTS_PYTHON))

from repo_paths import repo_root  # noqa: E402

# oob_teleop_hub.py has no relative imports, so it's importable flat without installing
# isaaccapture as a package (mirrors tests/python/core/cloudxr/conftest.py's approach).
_CLOUDXR_PY = repo_root() / "src" / "python" / "isaaccapture" / "cloudxr"
if str(_CLOUDXR_PY) not in sys.path:
    sys.path.insert(0, str(_CLOUDXR_PY))

# Both imports must come after the sys.path edit above, hence the noqa (E402).
import websockets  # noqa: E402

from oob_teleop_hub import OOBControlHub  # noqa: E402

POLL_INTERVAL_S = 0.05


def _fingerprint(snapshot: dict) -> tuple:
    """Reduce a snapshot to the fields whose change should trigger a new SNAPSHOT print."""
    return tuple(
        sorted(
            (
                h["clientId"],
                h["connected"],
                h["streaming"],
                # Sorted so key order in the JSON payload can't change the fingerprint.
                tuple(sorted(h["metricsByCadence"].items())),
            )
            for h in snapshot["headsets"]
        )
    )


async def main() -> None:
    """Start the hub, announce readiness, then print a snapshot on every state change."""
    hub = OOBControlHub()

    async def handler(ws) -> None:
        """Route each new WebSocket connection to the real hub's own handler."""
        await hub.handle_connection(ws)

    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    print(f"READY {port}", flush=True)

    last = _fingerprint(await hub.get_snapshot())  # empty at startup - not printed

    try:
        while True:
            await asyncio.sleep(POLL_INTERVAL_S)
            snap = await hub.get_snapshot()
            fp = _fingerprint(snap)
            if fp != last:
                last = fp
                print("SNAPSHOT " + json.dumps(snap), flush=True)
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
