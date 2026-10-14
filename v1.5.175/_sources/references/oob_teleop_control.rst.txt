.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Out-of-Band Teleop Control
==========================

The **OOB (out-of-band) teleop control hub** lets you coordinate Isaac Teleop
from outside the headset — read streaming metrics, inspect connected clients,
and push configuration changes — over the **same TLS port** as the CloudXR
proxy.

The hub shares the proxy TLS port (default **48322**, override with
``PROXY_PORT``).

Quick start
-----------

**Step 1 — Start the streaming host with OOB enabled**

On first use, it is recommended to run once **without** ``--setup-oob`` to
confirm ``adb devices`` sees the headset, verify USB debugging is enabled, and
accept the self-signed certificate in the headset browser manually (both the
web client page and the ``https://<host>:48322`` proxy page). Once that
baseline works, add ``--setup-oob`` to automate the full flow.

Launch the CloudXR runtime with the ``--setup-oob`` flag (add ``--accept-eula``
on first run):

.. code-block:: bash

   python -m isaacteleop.cloudxr.service run --accept-eula --setup-oob

This will:

1. Verify a USB-connected headset is available via ``adb devices``
2. Start the WSS proxy with the OOB control hub
3. Open the teleop page on the headset via ``adb shell am start``
4. Accept the self-signed certificate and click CONNECT automatically
   via Chrome DevTools Protocol (CDP)

To start the hub **without** any ``adb`` interaction — useful in containers,
CI, or wireless-only environments — set ``TELEOP_OOB_HUB_ONLY=1`` and open
the teleop URL on the headset manually. This mode supports WiFi setup only
and is **not compatible with** ``--usb-local``:

.. code-block:: bash

   TELEOP_OOB_HUB_ONLY=1 python -m isaacteleop.cloudxr.service run --accept-eula --setup-oob

You should see output confirming the hub is running:

.. code-block:: text

   CloudXR WSS proxy: running, log file: /home/<user>/.cloudxr/logs/wss.2026-04-13T202133Z.log
           oob:       enabled  (hub + USB adb automation — see OOB TELEOP block)

.. note::

   The headset must be:

   - **Connected via USB cable** for adb commands (opening the teleop URL)
   - **Connected to WiFi** on the same network as the streaming host (for web
     page access and CloudXR streaming)

   Streaming and web page access use WiFi, not USB tethering.
   ``adb forward`` is used only temporarily for CDP automation.

**Step 2 — (Manual fallback) Open the web client on the headset**

If the adb automation fails (e.g. headset not paired), you can manually open
the client URL on the headset browser with **all three** required query
parameters — ``oobEnable``, ``serverIP``, and ``port``:

.. parsed-literal::

   |web_client_url|\ ?oobEnable=1&serverIP=<HOST_IP>&port=48322

Replace ``<HOST_IP>`` with the streaming host's LAN IP. The ``port`` must
match the proxy port (default 48322).

.. note::

   All three parameters are required. If ``serverIP`` or ``port`` is missing,
   the OOB control channel is silently skipped — the client will still work for
   streaming but will not register with the hub or report metrics.

**Step 3 — Verify the headset registered with the hub**

From a PC on the same network, query the hub state API (``-k`` skips the
self-signed certificate check):

.. code-block:: bash

   curl -k https://<HOST_IP>:48322/api/oob/v1/state

You should see the headset listed under ``"headsets"`` with
``"connected": true``:

.. code-block:: json

   {
     "updatedAt": 1776112022900,
     "configVersion": 0,
     "config": {"serverIP": "<HOST_IP>", "port": 48322},
     "headsets": [
       {
         "clientId": "193f3758-281e-4292-8c36-6541b58963ef",
         "connected": true,
         "deviceLabel": null,
         "registeredAt": 1776112022805,
         "metricsByCadence": {}
       }
     ]
   }

If ``"headsets"`` is empty, double-check that the URL on the headset includes
both ``serverIP`` and ``port`` and that the headset can reach the host over the
network.

**Step 4 — (Optional) Push config to the headset**

Before or after the headset connects to the CloudXR stream, you can push
configuration overrides via the HTTP config API:

.. code-block:: bash

   curl -k "https://<HOST_IP>:48322/api/oob/v1/config?serverIP=<HOST_IP>&port=48322&codec=av1"

See ``GET /api/oob/v1/config`` below for all supported keys.

**Step 5 — Stream and poll for metrics**

With ``--setup-oob``, CONNECT is clicked automatically via CDP.  If running
without it, press **CONNECT** on the headset manually.  Once streaming begins,
the headset reports metrics to the hub every 500 ms.  Poll the state endpoint
from a PC to collect them:

.. code-block:: bash

   # Poll every 2 seconds (adjust to taste)
   watch -n 2 'curl -sk https://<HOST_IP>:48322/api/oob/v1/state | python3 -m json.tool'

The ``metricsByCadence`` field on each headset entry will now contain live streaming metrics.

ADB automation
--------------

The ``--setup-oob`` flag automates headset setup via USB ``adb``:

1. **adb devices** verifies exactly one device is connected
2. **am start** opens the teleop bookmark URL in the headset browser with
   the correct ``oobEnable=1``, ``serverIP``, and ``port`` parameters
3. **CDP connect** forwards the browser's DevTools socket over ``adb``,
   accepts the self-signed certificate interstitial, and clicks CONNECT
   via Chrome DevTools Protocol (``Input.dispatchMouseEvent``)

Streaming and web page access use WiFi, not USB tethering.  The headset
reaches the streaming host directly over WiFi.  ``adb forward`` is used only
temporarily during CDP automation to reach the browser's DevTools socket.

Prerequisites:

- ``adb`` must be on ``PATH`` (Android SDK Platform Tools)
- The headset must be connected via USB with USB debugging enabled
- The headset must be on the same WiFi network as the streaming host

If any step fails, the hub still starts.  Fall back to
``chrome://inspect/#devices`` from the PC or tap CONNECT on the headset
directly.  To re-open the page later without restarting the launcher, pass the
bookmark URL above to ``python -m isaacteleop.cloudxr.webclient`` (see
:doc:`/references/cloudxr`).

Architecture
------------

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Role
     - Software
     - What it does
   * - **XR headset**
     - Isaac Teleop web client in the device browser
     - Registers with the hub via WebSocket, reports streaming metrics
       periodically (default every 500 ms), receives config pushes.
   * - **Streaming host**
     - ``python -m isaacteleop.cloudxr.service run --setup-oob``
     - Runs CloudXR runtime + WSS proxy + OOB hub on a single TLS port.
       Opens the teleop page and clicks CONNECT via USB adb + CDP.
   * - **Operator / scripts**
     - ``curl``, browser, or custom tooling
     - Reads state via HTTP, optionally pushes config via HTTP.

WebSocket protocol
------------------

Endpoint: ``wss://<host>:<port>/oob/v1/ws``

All messages are JSON text frames with ``{"type": ..., "payload": ...}``.

Registration (first message)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

   {
     "type": "register",
     "payload": {
       "role": "headset",
       "deviceLabel": "Quest 3",
       "token": "<optional CONTROL_TOKEN>"
     }
   }

``role`` must be ``"headset"``. The hub replies with:

.. code-block:: json

   {
     "type": "hello",
     "payload": {
       "clientId": "<uuid>",
       "configVersion": 0,
       "config": {"serverIP": "...", "port": 48322}
     }
   }

Headset → hub: ``clientMetrics``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

   {
     "type": "clientMetrics",
     "payload": {
       "t": 1712800000000,
       "cadence": "frame",
       "metrics": {
         "streaming.framerate": 72.0,
         "render.pose_to_render_time": 18.5
       }
     }
   }

One message is sent per cadence per tick, carrying the last known value of every metric
reported so far in that cadence. Metric names are the CloudXR.js ``MetricsName`` values,
so they match the SDK documentation. The hub stores them as an arbitrary
``{name: value}`` map per cadence, so metrics added by a future SDK need no hub change.

.. list-table:: Metrics reported per cadence (CloudXR.js 6.3.0)
   :header-rows: 1
   :widths: 12 40 48

   * - Cadence
     - Metric
     - Meaning
   * - ``render``
     - ``render.framerate``
     - Client render rate (FPS), rolling average.
   * - ``render``
     - ``pose.send_framerate``
     - Rate at which poses are sent upstream (FPS), rolling average. For teleop this is
       the rate operator intent reaches the robot.
   * - ``render``
     - ``latency.xr_pose_age_ms``
     - Age of the XR pose at send time.
   * - ``frame``
     - ``streaming.framerate``
     - Streamed video rate (FPS), rolling average.
   * - ``frame``
     - ``streaming.frame_count``
     - Monotonic count of streamed frames.
   * - ``frame``
     - ``render.pose_to_render_time``
     - Pose-to-render latency (ms), rolling average.
   * - ``frame``
     - ``latency.pose_upload_ms``
     - Time spent uploading the pose.
   * - ``frame``
     - ``latency.pose_to_frame_received_ms``
     - Time from pose send to frame received.
   * - ``frame``
     - ``frame_pipeline.compositor_skipped_percent``
     - Share of frames the compositor skipped.
   * - ``frame``
     - ``frame_pipeline.out_of_order_percent``
     - Share of frames that arrived out of order.
   * - ``frame``
     - ``frame_pipeline.mismatched_percent``
     - Share of frames whose pose did not match the rendered pixels.
   * - ``network``
     - ``network.streaming_rate_mbps``
     - Current streaming rate.
   * - ``network``
     - ``network.available_bandwidth_mbps``
     - Estimated available bandwidth.
   * - ``network``
     - ``network.rtt_ms``
     - Round-trip time.
   * - ``network``
     - ``network.packet_loss``
     - Packet loss ratio.
   * - ``network``
     - ``network.avg_decode_time_ms``
     - Average video decode time.
   * - ``network``
     - ``network.quality_score``
     - Composite network quality, 0-4.
   * - ``network``
     - ``network.bandwidth_score``
     - Bandwidth component of the quality score, 0-4.
   * - ``network``
     - ``network.network_loss_score``
     - Loss component of the quality score, 0-4.
   * - ``network``
     - ``network.latency_score``
     - Latency component of the quality score, 0-4.
   * - ``network``
     - ``session.quality``
     - Overall session quality, 0-4. Also drives the in-XR quality bars.

Score metrics use the SDK's ``QualityScore`` scale: 0 NoData, 1 Unsustainable,
2 Degraded, 3 Good, 4 Excellent.

Metrics are cleared when the stream stops, so a new session never reports the previous
session's last known values.

HTTP API
--------

All endpoints use **GET** with query parameters on the proxy TLS port.

``GET /api/oob/v1/state``
^^^^^^^^^^^^^^^^^^^^^^^^^

Returns the current hub state: connected headsets, latest metrics, and config
version.

.. code-block:: bash

   curl -k https://localhost:48322/api/oob/v1/state

Example response:

.. code-block:: json

   {
     "updatedAt": 1712800000000,
     "configVersion": 0,
     "config": {"serverIP": "10.0.0.1", "port": 48322},
     "headsets": [
       {
         "clientId": "abc-123",
         "connected": true,
         "deviceLabel": "Quest 3",
         "registeredAt": 1712799990000,
         "metricsByCadence": {
           "frame": {
             "at": 1712800000000,
             "metrics": {"streaming.framerate": 72.0}
           }
         }
       }
     ]
   }

``GET /api/oob/v1/config``
^^^^^^^^^^^^^^^^^^^^^^^^^^

Push config to connected headsets via query parameters:

.. code-block:: bash

   curl -k "https://localhost:48322/api/oob/v1/config?serverIP=10.0.0.5&port=48322"

Example response:

.. code-block:: json

   {
     "ok": true,
     "changed": true,
     "configVersion": 1,
     "targetCount": 1
   }

Supported query keys: ``serverIP``, ``port``, ``panelHiddenAtStart``, ``codec``.
Optional ``targetClientId`` restricts the push to a single headset (returns 404
if not connected).

Authentication
--------------

Set ``CONTROL_TOKEN=<secret>`` to require a token on all hub operations.
Pass it as:

- WebSocket: ``"token"`` field in the ``register`` payload
- HTTP: ``?token=<secret>`` query parameter or ``X-Control-Token`` header

Web client integration
----------------------

The web client connects to the hub when the page URL contains
``oobEnable=1`` plus ``serverIP`` and ``port``:

.. parsed-literal::

   |web_client_url|\ ?oobEnable=1&serverIP=10.0.0.1&port=48322

The client builds ``wss://{serverIP}:{port}/oob/v1/ws`` and:

1. Registers as role ``"headset"``
2. Reports ``clientMetrics`` periodically (default every 500 ms)
3. Receives ``config`` pushes from operator

URL query parameter overrides
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following URL parameters override their corresponding form fields (and
``localStorage`` values) so that bookmarked links always take priority over
previously saved settings:

- ``serverIP`` CloudXR server IP address
- ``port`` CloudXR server port
- ``codec`` video codec
- ``panelHiddenAtStart`` hide the control panel on load

When no URL override is present, form fields restore from ``localStorage``, with one
exception: a saved device profile other than ``Custom`` is re-applied from the current
profile table at startup, so profile default updates (frame rate, bitrate, codec)
reach returning clients. Only ``Custom`` keeps manually edited values.

Environment variables
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Description
   * - ``PROXY_PORT``
     - WSS proxy port (default ``48322``)
   * - ``CONTROL_TOKEN``
     - Optional auth token for hub access
   * - ``TELEOP_STREAM_SERVER_IP``
     - Override the auto-detected LAN IP in hub initial config
   * - ``TELEOP_PROXY_HOST``
     - Override the LAN IP used for headset bookmark URLs
   * - ``TELEOP_WEB_CLIENT_BASE``
     - Override the web client origin URL
   * - ``TELEOP_STREAM_PORT``
     - Override the signaling port (default same as proxy port)
   * - ``TELEOP_CLIENT_CODEC``
     - Default video codec for headset bookmarks
   * - ``TELEOP_CLIENT_PANEL_HIDDEN_AT_START``
     - Hide control panel on load (``true`` / ``false``)
   * - ``TELEOP_CLIENT_ROUTE``
     - HashRouter fragment appended to the bookmark URL. Default empty
       (no fragment — the web client picks its own landing route). Set
       to e.g. ``/real/gear/dexmate`` to force a specific route. A
       leading ``#`` is stripped automatically.
   * - ``TELEOP_OOB_HUB_ONLY``
     - Set to any non-empty value (e.g. ``1``) to start the OOB hub
       without any ``adb`` interaction. The hub starts normally and
       accepts WebSocket connections, but the launcher skips
       ``adb devices``, ``adb shell`` wakefulness checks, CDP port
       forwarding, and the automated browser open + CONNECT click.
       Useful in environments where ``adb`` is unavailable or
       unreliable (CI, containers, wireless-only setups). The operator
       must open the teleop page on the headset manually with the
       correct ``oobEnable=1&serverIP=<HOST>&port=<PORT>`` parameters.
       Only meaningful together with ``--setup-oob``; has no effect
       without it. **Not compatible with** ``--usb-local`` — hub-only
       mode supports WiFi setup only; the launcher rejects the
       combination at startup.
   * - ``ANDROID_SERIAL``
     - Pin a specific adb device when more than one is connected. The
       launcher refuses to start with multiple devices unless this is
       set; the value must match a serial currently in ``device`` state
       (per ``adb devices``). This is the standard adb env var — every
       ``adb`` subprocess inherits it, so no code path needs ``-s
       <serial>``.
   * - ``USB_UI_PORT``
     - HTTPS static web client port for ``--usb-local`` (default ``8080``).
       Binds to ``127.0.0.1:<port>`` and ``adb reverse``-maps the port to
       the headset.  ``--host-client`` uses the WSS proxy port (``PROXY_PORT``)
       instead; ``USB_UI_PORT`` has no effect on it.
   * - ``USB_BACKEND_PORT``
     - CloudXR backend port the headset reaches via ``adb reverse`` in
       ``--usb-local`` mode (default ``49100``).
   * - ``USB_TURN_PORT``
     - coturn TURN-server port for WebRTC ICE relay in ``--usb-local``
       mode (default ``3478``). ``adb reverse``-mapped to the headset.

USB-local mode (``--setup-oob --usb-local``)
--------------------------------------------

``--usb-local`` routes teleop signalling, the web client, and WebRTC media
over the USB cable on the headset's loopback via ``adb reverse``.  It requires
``--setup-oob`` because the OOB hub is the only path that delivers TURN relay
configuration to the client (without it the headset has no way to discover the
coturn endpoint).  Always use both flags together:

.. code-block:: bash

   python -m isaacteleop.cloudxr.service run --accept-eula --setup-oob --usb-local

On startup the launcher:

1. Pre-flights: ``adb`` on PATH, ``coturn`` installed, exactly one device
   connected, headset has at least one non-loopback IP (Wi-Fi up — see
   troubleshooting below for why this is required even though no packets
   traverse the network).
2. Resolves the WebXR static directory from
   ``TELEOP_WEB_CLIENT_STATIC_DIR`` (default ``~/.cloudxr/static-client``)
   and syncs missing ``index.html``, ``bundle.js``, and ``bundle.emulator.js`` from
   the published client (see :doc:`../getting_started/build_from_source/webxr`).
3. Serves that directory over HTTPS on 127.0.0.1:8080 with the same PEM
   the WSS proxy uses (Python ``http.server`` in a daemon thread).
4. ``adb reverse`` for 8080 (static UI), 48322 (WSS), 49100 (backend),
   3478 (coturn TURN).
5. Starts coturn locally on 127.0.0.1:3478 for WebRTC ICE relay.
6. Launches the teleop URL on the headset and auto-clicks CONNECT via CDP.

In ``--usb-local`` mode the launcher also wipes localStorage / IndexedDB /
cookies / HTTP cache for the teleop UI origin (``https://127.0.0.1:<usb_ui_port>``)
before the session starts — the SDK and web client both cache settings
(e.g. ``general.iceTransportPolicy`` for ICE transport policy,
``cxr.isaac.teleopPath`` for the last-used project) in localStorage, and a
stale value can silently win over a fresh URL param. The origin is owned
by the launcher so clearing it has no collateral effect; the step is a
no-op when the headset browser isn't running yet.

Required apt packages: ``adb`` (``android-tools-adb``) and ``coturn``.
No Node.js / ``npm`` is required at runtime.

Troubleshooting
---------------

Teleop client error: "No local connection candidates" (0xC0F2220F)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause:** Wi-Fi must stay associated on the headset throughout the
session, even in ``--usb-local`` mode. No teleop traffic actually flows
over Wi-Fi — every byte goes over the USB cable via ``adb reverse`` —
but Chromium's WebRTC ``rtc::NetworkManager`` excludes loopback
interfaces when enumerating networks for ICE. If the only active
network on the headset is ``lo``, ICE gathering hangs at ``gathering``
forever (no local candidates emitted, no error fires) until the CloudXR
session times out with this code.

**Fix:** Associate the headset with any Wi-Fi network and retry.
Internet is **not** required — a phone hotspot with no SIM works, an
open AP you never authenticate to works. The packets still route over
USB (the kernel short-circuits loopback regardless of source
interface); the Wi-Fi interface just needs to *exist* with an IP so
WebRTC's enumeration is non-empty.

The ``--usb-local`` launcher pre-flights this via
``adb shell ip -o -4 addr show`` and refuses to start if no non-loopback
interface is present. A runtime monitor also watches for mid-session
Wi-Fi drops and prints a yellow warning so the cause is obvious without
having to puzzle out a frozen WebRTC connection.

Web client UI has no text (OOB / ``--host-client``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause:** Production webpack emits ``bundle.js`` (main app, including UIKit MSDF text)
and ``bundle.emulator.js`` (desktop emulation only). If the static cache under
``TELEOP_WEB_CLIENT_STATIC_DIR`` is missing or outdated — especially a truncated
``bundle.js`` — in-VR text does not render.

**Fix:** Ensure ``index.html``, ``bundle.js``, and ``bundle.emulator.js`` are present
under the static dir — run the launcher once with network, copy a full ``npm run build``
output, or use the GitHub Pages URL directly. See
:doc:`../getting_started/build_from_source/webxr`.

CDP: startButton marked failed / not actionable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause:** The web client's capability check (``App.tsx``) sets the button
label to ``CONNECT (capability check failed)`` and disables it when a
required feature is missing (WebGL2, ``requestVideoFrameCallback``,
immersive-VR support). ADB automation detects this and aborts instead of
clicking a dead button.

**Fix:** Open the teleop URL on the headset manually and read the
``errorMessageBox`` — it names the specific missing capability. Common
causes: launched in WebLayer instead of Meta Quest Browser (no WebXR
support → IWER fallback silently activates); WebGL2 disabled by device
policy.

``coturn`` not found / TURN server failed to start
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause:** ``--usb-local`` requires coturn to relay WebRTC media between
the headset (TCP via adb reverse) and the CloudXR backend (UDP on
loopback).

**Fix:** ``sudo apt-get install -y coturn``. The launcher starts its own
``turnserver`` process on 127.0.0.1:3478; no systemd service is needed
(and may conflict — stop the system ``coturn.service`` if enabled).

Inspect ``/tmp/coturn-cloudxr-3478.log`` for bind errors or credential
mismatches; the launcher truncates this file on every start so only the
current session's output is present.

Tab not found within timeout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause:** The headset's default URL handler is something other than Meta
Quest Browser (e.g. WebLayer on Meta Quest) and did not open the teleop
URL in a browser with remote-debugging exposed.

**Fix:** Open ``chrome://inspect#devices`` on this PC, inspect the
headset tab manually, and click CONNECT. Or set a different default
browser on the headset.

WebXR static download fails (offline / proxy)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause:** The launcher syncs ``index.html``, ``bundle.js``, and
``bundle.emulator.js`` into the static dir on first run. Behind a proxy or with no
internet, this fails and ``--usb-local`` aborts.

**Fix:** Pre-stage the full client ``build/`` tree (or the published
``/client/<version>/`` directory) into the static dir, then re-run.  The
launcher only downloads missing or empty files.  Override the target directory
via ``TELEOP_WEB_CLIENT_STATIC_DIR``.

**Fix:** Set the SDK versions in ``deps/cloudxr/.env`` (copy from
``.env.default``) so the download script can resolve the right version,
or stage ``nvidia-cloudxr-<version>.tgz`` in ``deps/cloudxr/`` manually.
