# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Use the public browser input API; read simulation observations over loopback."""

import atexit
import json
import math
from pathlib import Path
import time
from urllib.parse import parse_qsl, urlsplit
from urllib.request import ProxyHandler, build_opener
import uuid

import websocket


class BrowserClient:
    """One loopback browser connection and fresh observation stream per driver."""

    def __init__(
        self,
        cdp_url="http://127.0.0.1:9222",
        feedback_url="http://127.0.0.1:8765/state",
        output=None,
    ):
        for value in (cdp_url, feedback_url):
            parsed = urlsplit(value)
            if (
                parsed.scheme != "http"
                or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
                or parsed.username
                or parsed.password
                or parsed.fragment
            ):
                raise ValueError(
                    "Control and observation endpoints must use loopback HTTP without credentials"
                )
        self.cdp_url, self.feedback_url = cdp_url.rstrip("/"), feedback_url
        self.output = None if output is None else Path(output) / "commands"
        if self.output is not None:
            self.output.mkdir(parents=True, exist_ok=True)
        self._socket = None
        self._sequence = 0

    @staticmethod
    def _read_json(url):
        # Loopback control must never be routed through an ambient corporate proxy.
        with build_opener(ProxyHandler({})).open(url, timeout=5) as response:
            return json.load(response)

    def close(self):
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def state(self, max_age=5):
        result = self._read_json(self.feedback_url)
        age = time.time() - result.get("timestamp", 0)
        if not 0 <= age <= max_age:
            raise RuntimeError("Simulator observation is stale or from the future")
        return result

    def command(self, action, timeout=20, **parameters):
        if not math.isfinite(timeout) or not 0 < timeout <= 300:
            raise ValueError("Command timeout must be in (0, 300] wall-clock seconds")
        if self._socket is None:
            pages = [
                page
                for page in self._read_json(self.cdp_url + "/json/list")
                if page.get("type") == "page"
                and dict(parse_qsl(urlsplit(page.get("url", "")).query)).get(
                    "syntheticXR"
                )
                == "1"
            ]
            if len(pages) != 1:
                raise RuntimeError(
                    "Open exactly one Synthetic XR client tab in the debugging browser"
                )
            self._socket = websocket.create_connection(
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
        self._sequence += 1
        self._socket.send(
            json.dumps(
                {
                    "id": self._sequence,
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
            self._socket.settimeout(max(0.01, deadline - time.monotonic()))
            message = json.loads(self._socket.recv())
            if message.get("id") != self._sequence:
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
            if self.output is not None:
                (self.output / ("command-" + receipt["id"] + ".json")).write_text(
                    json.dumps(receipt, indent=2)
                )
            return receipt
        raise TimeoutError("Browser did not acknowledge synthetic input")


_default = BrowserClient()


def configure(cdp_url, feedback_url, output=None):
    global _default
    replacement = BrowserClient(cdp_url, feedback_url, output)
    _default.close()
    _default = replacement


def command(action, timeout=20, **parameters):
    return _default.command(action, timeout, **parameters)


def state(max_age=5):
    return _default.state(max_age)


def close():
    _default.close()


atexit.register(close)
