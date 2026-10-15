# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Use the public browser input API; read simulation observations over loopback."""

import atexit
import json
from pathlib import Path
import time
from urllib.parse import urlsplit
from urllib.request import urlopen
import uuid

import websocket

_cdp_url = "http://127.0.0.1:9222"
_feedback_url = "http://127.0.0.1:8765/state"
_socket = None
_sequence = 0
_output = None


def configure(cdp_url, feedback_url, output=None):
    global _cdp_url, _feedback_url, _output
    for value in (cdp_url, feedback_url):
        parsed = urlsplit(value)
        if parsed.scheme != "http" or parsed.hostname not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            raise ValueError(
                "Example control and observation endpoints must use loopback HTTP"
            )
    _cdp_url, _feedback_url = cdp_url.rstrip("/"), feedback_url
    if output is not None:
        _output = Path(output) / "commands"
        _output.mkdir(parents=True, exist_ok=True)


def _read_json(url):
    with urlopen(url, timeout=5) as response:
        return json.load(response)


def close():
    global _socket
    if _socket is not None:
        _socket.close()
        _socket = None


atexit.register(close)


def command(action, timeout=20, **parameters):
    global _socket, _sequence
    if _socket is None:
        pages = [
            page
            for page in _read_json(_cdp_url + "/json/list")
            if page.get("type") == "page" and "syntheticXR=1" in page.get("url", "")
        ]
        if len(pages) != 1:
            raise RuntimeError(
                "Open exactly one Synthetic XR client tab in the debugging browser"
            )
        _socket = websocket.create_connection(
            pages[0]["webSocketDebuggerUrl"], timeout=timeout, suppress_origin=True
        )
    payload = {"action": action, **parameters}
    expression = (
        """(c => {
      const api = window.syntheticXR;
      if (!api) throw new Error('Synthetic XR API is not ready');
      if (c.action === 'pose') {
        const update = {};
        if (c.position) update.position = c.position;
        if (c.quaternion) update.quaternion = c.quaternion;
        if (c.buttons && c.buttons.trigger !== undefined) update.trigger = c.buttons.trigger;
        if (c.buttons && c.buttons.squeeze !== undefined) update.grip = c.buttons.squeeze;
        api.setController(c.hand || 'right', update);
      } else if (c.action === 'head') {
        const update = {};
        if (c.position) update.position = c.position;
        if (c.quaternion) update.quaternion = c.quaternion;
        api.setHead(update);
      } else if (c.action === 'start') {
        const button = document.getElementById('synthetic-xr-start-teleop');
        if (!button || button.disabled) throw new Error('Connect the client before starting teleoperation');
        button.click();
      } else if (c.action === 'stop') {
        api.stop();
      } else if (c.action !== 'inspect') {
        throw new Error('Unsupported synthetic input command');
      }
      if (c.action === 'inspect' && (!api.lastFrame || performance.now() - api.lastFrame.time > 1000))
        throw new Error('No fresh WebXR frame; connect the client first');
      return {state: api.getState(), frame: api.lastFrame};
    })("""
        + json.dumps(payload, allow_nan=False)
        + ")"
    )
    _sequence += 1
    _socket.send(
        json.dumps(
            {
                "id": _sequence,
                "method": "Runtime.evaluate",
                "params": {
                    "expression": expression,
                    "returnByValue": True,
                    "awaitPromise": True,
                },
            }
        )
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        message = json.loads(_socket.recv())
        if message.get("id") != _sequence:
            continue
        result = message.get("result", {})
        if "error" in message or "exceptionDetails" in result:
            raise RuntimeError(
                "Browser rejected synthetic input: " + json.dumps(message)
            )
        receipt = {
            "id": uuid.uuid4().hex,
            "timestamp": time.time(),
            "ok": True,
            "command": payload,
            "browser": result["result"]["value"],
        }
        if _output is not None:
            (_output / ("command-" + receipt["id"] + ".json")).write_text(
                json.dumps(receipt, indent=2)
            )
        return receipt
    raise TimeoutError("Browser did not acknowledge synthetic input")


def state(max_age=5):
    result = _read_json(_feedback_url)
    if time.time() - result.get("timestamp", 0) > max_age:
        raise RuntimeError("Simulator observation is stale")
    return result
