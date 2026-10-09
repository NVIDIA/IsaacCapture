<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OOB real-browser integration test

Exercises `run_oob_connect()`'s real orchestration logic against a real
Chromium instance running the real teleop web client (real IWER-emulated
WebXR, real CDP, real client JS), as additional, slower coverage alongside
the fast, fully-scripted `mock_adb()` / fake-CDP-server unit tests already in
`test_oob_teleop_adb.py`.

## What's real, what's faked

- **ADB**: no physical device exists in CI/dev-sandbox, so this stays
  synthetic via `RealBrowserAdb` (`conftest.py`) — but `am start` and `adb
  forward` have real side effects (navigating a real local Chromium, aliasing
  the local TCP port straight to Chromium's own `--remote-debugging-port`)
  instead of returning canned output. Error-injection kwargs
  (`device_state`, `am_start_rc`, `devtools_socket`, `forward_rc`) work the
  same as on `FakeAdb`.
- **CDP**: fully real — a real Chromium (`real_chrome()`, matching
  `playwright.config.js`'s launch args: SwiftShader software WebGL,
  `--disable-features=WebXR` so IWER provides `navigator.xr`) reachable over
  its own `/json`, `/json/close/<id>`, and debugger websocket.
- **OOB control-hub** (`/oob/v1/ws`): the real `OOBControlHub`, via a real
  `wss.py` (`real_wss_proxy()`), run for real — pure Python, no GPU
  dependency.
- **CloudXR signaling**: never a real or mocked runtime backend.
  `wss.py` already treats "nothing listening on `backend_port`" as an
  anticipated configuration (`_is_backend_connection_refused()`), not an
  error to work around, so the client's own `#errorMessageBox` on a refused
  signaling connection is a genuine, free signal — not something this test
  infra needs to fake.
- **The web client**: always `MockCloudXR` (`npm run build:app-mock`), never
  the real `@nvidia/cloudxr` SDK. The real SDK genuinely tries to stream
  against whatever's on `backend_port`, which is unpredictable and
  uncontrollable to assert against. `MockCloudXR` mounts the identical
  `App.tsx`/`CloudXRComponent.tsx` UI (same `#startButton`/`#errorMessageBox`
  DOM) with a fully deterministic, externally-controllable session
  underneath — `window.__mockCloudXRFail(message?, code?)`, callable over CDP
  via `cdp_evaluate()` — and opens no socket of its own.

## Two bugs found building this infra (worth knowing if this code is touched again)

- **Webpack HMR module duplication.** A dev-server's hot-reload re-executes a
  module's top-level code on live-reload, creating a *second* instance of it
  side by side with the one React already mounted against. For
  `cloudxr-mock-alias.ts` this meant `window.__mockCloudXRFail()` silently
  bound to a fresh, never-used `activeSession` while the real,
  already-running session lived on untouched — callable, no error, just no
  effect. Fixed by never using a dev-server: `static_webxr_build()`
  (`conftest.py`) always does a real `npm run build:app-mock` production
  build and serves the static output, which has no live-reload runtime at
  all.
- **`<React.StrictMode>` double-invoking effects in a dev-mode build.**
  Independent of HMR: `webpack.app-mock.js` used `mode: 'development'`, and
  `src/index.tsx` wraps the app in `StrictMode`, which deliberately
  double-invokes effects in dev builds — calling `CloudXRComponent.tsx`'s
  `establishSession()` (and so `CloudXR.createSession()`) twice. Fixed by
  switching `webpack.app-mock.js` to `mode: 'production'`, which sets
  `NODE_ENV=production` and makes React skip StrictMode's double-invoke
  entirely.
- A third, unrelated trap hit while diagnosing the above: `webpack.common.js`'s
  persistent filesystem cache (`cache: {type: 'filesystem'}`) is keyed only
  on the webpack config file, not on every source file it compiles — a stale
  cache entry silently served an old compiled bundle that didn't include a
  real source edit. `static_webxr_build()` now deletes
  `node_modules/.cache/webpack` before every build.

## ADB/CDP command inventory (what `oob_teleop_adb.py` actually talks to)

| adb command | Function | Used for |
|---|---|---|
| `adb get-state` | `adb_device_state()` → `assert_adb_device_online()` | Preflight gate before nearly every other call; drives offline→`adb reconnect` retry |
| `adb shell cat /proc/net/unix` | `_discover_devtools_socket()` | Regex-scans for `@..._devtools_remote[_pid]` |
| `adb shell am start -a android.intent.action.VIEW -d <url>` | `open_url_on_headset()` → `run_adb_headset_bookmark()` / `run_oob_connect()` | Opens the browser to the teleop URL; retried in a poll loop until a matching tab appears over CDP |
| `adb forward tcp:<local_port> localabstract:<socket_name>` | `_adb_forward_cdp()` | Maps the discovered DevTools abstract socket to a local TCP port — what makes CDP reachable at all |
| `adb forward --remove tcp:<local_port>` | `_adb_forward_remove()` | Cleanup |

CDP itself (`_cdp_list_tabs()`, `_cdp_session_click_connect()`,
`_monitor_teleop_error_banner()`) is reached over the `adb forward`'d port
and never touches application/CloudXR traffic directly.

## Tests

Three tests in `test_oob_teleop_adb.py`, all slow (real browser + a real
webpack production build + real `wss.py`), requiring a system Chrome
(`PLAYWRIGHT_CHROME_PATH` override; defaults to `/usr/bin/google-chrome`) and
Node/npm on `PATH` — run manually for now, not yet wired into CI:

- `test_run_oob_connect_real_browser_end_to_end` — the full real
  adb→CDP→click path against `MockCloudXR`, proving `run_oob_connect()`'s own
  orchestration for real.
- `test_run_oob_connect_real_mock_cxr_crash_surfaces_error_banner` — proves
  the crash-trigger mechanism itself: a real, deterministic, DOM-visible
  error banner on demand.
- `test_run_oob_connect_real_mock_cxr_crash_triggers_real_relaunch` —
  exercises the repair-capable `_monitor_teleop_error_banner` added by
  `gmorgan/oob-error-relaunch` (#1146): a genuine crash recovered by genuine
  ADB+CDP automation (a second real `am start` and a second real tab), not a
  scripted stand-in for either.
