# OOB real-browser integration test — plan

Goal: exercise `run_oob_connect()`'s real orchestration logic against a real
Chromium instance running the real teleop web client (real IWER-emulated
WebXR, real CDP, real client JS), instead of today's fully-scripted
`mock_adb()` / fake-CDP-server unit tests (`gmorgan/oob-connect-tests`,
PR #1148). Those stay as fast unit coverage of the orchestration logic; this
is new, slower, additional integration coverage.

**Revised: a minimal mock CloudXR runtime is required after all.** §3/§5
originally concluded none was needed, reasoning that `oob_teleop_adb.py`
never talks to the runtime directly. That's still true, but it missed a
second-order effect: `_cdp_session_click_connect()` (`oob_teleop_adb.py:1584`)
raises `OobAdbError(f"Teleop connection failed: {error_text}")` if the real
client's `#errorMessageBox` shows an explicit error within 30s of the click.
With nothing listening on `backend_port`, `wss.py`'s `proxy_handler()`
accepts the client's signaling WS handshake, then immediately fails to reach
the backend and closes the connection out from under it — the real client
will very likely surface that as a real error, which means
`run_oob_connect()` would raise instead of returning a monitor task, for
real client-behavior reasons, not a gap in the OOB code itself.

So: something needs to listen on `backend_port` and respond with synthetic
data — not to make signaling "succeed" in any protocol-complete sense, just
enough that the client doesn't explicitly error out within the click-wait
window. Scope stays minimal and empirical: start with the cheapest thing
that might work (accept the WS, hold it open, respond to nothing) and only
add real SDP/ICE handling if the real client's own error/timeout behavior
requires it. Full WebRTC media (real rendering) remains explicitly out of
scope / a separate future project — this is only "enough to not error."

## 0. Confirmed architecture

Four real components, one mock:

1. **Real webxr client** (`App.tsx`, `npm run dev-server`, port 8080) in a
   real desktop Chromium — not `MockCloudXR.ts`. Uses real IWER to emulate a
   WebXR/OpenXR-backed headset client-side (`loadIWERIfNeeded`).
2. **Real `wss.py` + `OOBControlHub`**, running for real (self-signed cert,
   hub wired in), pointed at an unused `backend_port` (49100). No CloudXR
   runtime behind it — the resulting connection refusal is handled by
   existing production code, not test infra. Required so both the CloudXR
   SDK *and* the OOB control channel (`wss://{serverIP}:{port}/oob/v1/ws`)
   have a real `serverIP:port` to resolve to.
3. **Real DevTools/CDP** on that real Chromium instance — nothing to fake.
4. **Mock ADB** — the one actual test double, since no physical Android
   device exists. Must stay controllable/parametrized like today's
   `FakeAdb(device_state=, am_start_rc=, devtools_socket=, forward_rc=)`, so
   tests can still deliberately trigger the real error paths (unauthorized/
   offline device, `am start` failure, no devtools socket, forward failure)
   — but the success-path commands (`am start`, `adb forward`) now drive
   the real browser/CDP port instead of returning canned output (see §3).

## 1. Exhaustive inventory: everything the OOB code talks to

### A. ADB (device/OS layer) — driven directly by `oob_teleop_adb.py`

Read-only queries:

| adb command | Function | Used for |
|---|---|---|
| `adb get-state` | `adb_device_state()` → `assert_adb_device_online()` | Preflight gate before nearly every other call; drives offline→`adb reconnect` retry and user-facing error text |
| `adb devices` | `assert_exactly_one_adb_device()` | Pins a single device (or validates `ANDROID_SERIAL`) |
| `adb shell ip -o -4 addr show` | `headset_non_loopback_interfaces()` | USB-local WiFi-presence check (`require_headset_non_loopback_network()`); polled every 5s by `monitor_headset_wifi()` during a session |
| `adb shell dumpsys power` | `headset_wakefulness()` | `mWakefulness=` regex; polled by `assert_headset_awake()` after sending a wake key |
| `adb shell getprop ro.product.manufacturer` / `ro.product.brand` | `_adb_getprop()` → `headset_browser_package()` | Vendor string selects which browser package to force |
| `adb shell pm list packages <pkg>` | `_adb_pkg_installed()` / `_first_installed_pkg()` | Confirms the chosen vendor browser is actually installed |
| `adb shell cat /proc/net/unix` | `_discover_devtools_socket()` | Regex-scans for `@..._devtools_remote[_pid]`; priority order `weblayer` > `com.oculus.browser` > `chrome` > `webview` |
| `adb reverse --list` | `verify_adb_reverse_rules()` | Confirms USB-local reverse rules survived (rc=0 doesn't guarantee persistence) |

Actions (side-effecting):

| adb command | Function | Effect |
|---|---|---|
| `adb reconnect` | `assert_adb_device_online()` | One retry on transient `offline` |
| `adb shell input keyevent KEYCODE_WAKEUP` | `assert_headset_awake()` | Wakes the device |
| `adb shell am start -a android.intent.action.VIEW -d <url> [-p <pkg>]` | `open_url_on_headset()` → `run_adb_headset_bookmark()` / `run_oob_connect()` | **Opens the browser to the teleop URL.** Retried in a poll loop inside `run_oob_connect()` until a matching tab appears over CDP |
| `adb reverse tcp:<port> tcp:<port>` | `setup_adb_reverse_ports()`, `setup_adb_reverse_turn()` | USB-local: headset-loopback → PC port mapping (proxy port, backend port, TURN port) |
| `adb reverse --remove tcp:<port>` | `teardown_adb_reverse_ports()`, `teardown_adb_reverse_turn()` | Cleanup |
| `adb forward tcp:<local_port> localabstract:<socket_name>` | `_adb_forward_cdp()` | Maps the discovered DevTools abstract socket to a local TCP port — this is what makes CDP reachable at all |
| `adb forward --remove tcp:<local_port>` | `_adb_forward_remove()` / `teardown_adb_forward_cdp()` | Cleanup (also run at startup to clear a stale rule from a crashed prior run) |

### B. Chrome DevTools Protocol (CDP) — reached over the `adb forward`'d port, driven directly by `oob_teleop_adb.py`

- `GET http://localhost:<port>/json` (`_cdp_list_tabs()`) — list tabs. Used by
  `_close_stale_teleop_tabs()` (closes any tab whose URL contains
  `oobEnable=`, freeing XR resources a crashed prior session held) and by
  `run_oob_connect()`'s tab-matching loop after `am start`.
- `GET http://localhost:<port>/json/close/<id>` — close a specific tab.
- CDP WebSocket session on a tab's `webSocketDebuggerUrl`
  (`_cdp_session_click_connect()`): `Security.setIgnoreCertificateErrors`
  (cert bypass), then `Runtime.evaluate` polling a client-side readiness
  state machine (`loading` → `initializing` [IWER + capability checks
  running] → `ready`/`failed`), then synthesizes the CONNECT click.
- A second, independent CDP WebSocket session, opened by
  `_monitor_teleop_error_banner()` once `run_oob_connect()` returns: polls
  `document.getElementById('errorMessageBox')` once/second and logs any new
  `error`-class banner text. Runs until cancelled or the tab/socket drops;
  always tears down its own `adb forward` on exit.

CDP is purely browser automation — it never touches application/CloudXR
traffic.

### C. What the client JS itself opens, once loaded — *not* driven by
`run_oob_connect()`'s Python code, but part of the same end-to-end flow

- **CloudXR streaming signaling websocket.** The real `@nvidia/cloudxr`
  client component opens this once CONNECT is clicked and
  `requestSession()` succeeds. It terminates at `wss.py`'s real, pure-Python
  WSS proxy (TLS termination + WebSocket forwarding), which in turn forwards
  to a native CloudXR Runtime backend process
  (`CloudXRService._runtime_proc`, `service/_service.py`, launched via
  `subprocess.Popen`) — GPU-dependent, and the one piece the existing test
  suite (`test_service.py`'s `mock_service_deps()`) always mocks out rather
  than launching for real.
- **OOB control-hub websocket** (`/oob/v1/ws`, `oob_teleop_hub.py`,
  `OOBControlHub`). Pure Python, no GPU dependency. Headset registers
  (`{"type": "register", "role": "headset", ...}`), gets a `hello`, then
  sends `streamStatus`/`clientMetrics`; operator-side config pushes flow
  back down the same socket. Fully real and runnable with zero mocking.
- `MockCloudXR.ts` (`deps/cloudxr/webxr_client/tests/mock/MockCloudXR.ts`,
  behind `CloudXRComponentTest.html` / `npm run dev-server:component-mock`)
  is a *third*, unrelated option: it fakes the CloudXR session object
  entirely client-side (synthetic stereo render) and opens **no** websocket
  at all. Confirmed via grep — zero `WebSocket` usage in that file. Useful
  as today's manual dev-server page, but doesn't exercise the real signaling
  path, so it's not what we want for this integration test if the goal is
  exercising real client↔server behavior.

## 2. What "real" actually means for this test

- **ADB layer**: no real device exists in CI/dev-sandbox, so this stays
  synthetic — but with real side effects where one exists locally, per the
  session's prior discussion. Concretely:
  - `am start` → parse the `-d <url>` out of the real shell command string
    and actually navigate a real local Chromium instance to it (not a
    canned success).
  - `adb forward` / `--remove` → alias the local TCP port directly to
    Chromium's own `--remote-debugging-port` (no real abstract socket
    exists locally, so there's nothing to really forward — but the *effect*,
    a working local CDP port, is real).
  - `get-state`, `/proc/net/unix`, `getprop`, `pm list packages` → stay
    canned; there is no real device for these to reflect.
- **CDP layer**: fully real. A real Chromium (matching
  `deps/cloudxr/webxr_client/playwright.config.js`'s launch args —
  `--disable-features=WebXR`, SwiftShader software WebGL, no sandbox) is
  reachable over its own real `/json`, `/json/close/<id>`, and debugger
  websocket. Nothing to fake here at all.
- **Client-opened sockets layer**: split.
  - OOB control-hub (`/oob/v1/ws`) — run the real `OOBControlHub` for real.
    No mock needed.
  - CloudXR signaling — `wss.py`'s proxy is real Python, runs for real.
    **Resolved: no runtime backend needed, real or mocked.** `wss.py`
    already treats "nothing listening on `backend_port`" as an anticipated,
    supported configuration, not an error to work around —
    `_is_backend_connection_refused()` (`wss.py:454`) detects exactly this
    and `proxy_handler()` logs it as `"expected when running WSS+hub without
    the runtime; teleop signaling uses %s"`, then closes that one signaling
    connection and returns; nothing crashes. It also never touches the OOB
    hub — `OOB_WS_PATH` is routed to `hub.handle_connection()` *before*
    `proxy_handler()` is ever reached (`wss.py:644`), so hub
    register/`hello`/`streamStatus`/`clientMetrics` and the
    `/api/oob/v1/*` HTTP routes are unaffected either way. So: never start
    anything on `backend_port` (49100) for this test. The client will very
    likely show a real connection-error banner after CONNECT is clicked
    (its own signaling socket gets refused) — a genuine, free signal
    `_monitor_teleop_error_banner()` already knows how to read, usable as a
    bonus assertion but not required for the test's core scope.

## 3. Proposed test shape

New test double(s) in `tests/python/core/cloudxr/conftest.py`, alongside the
existing `FakeAdb`/`mock_adb()`:

- `RealBrowserAdb` (name TBD): same `subprocess.run`-patching shape as
  `FakeAdb`, but `am start` and `adb forward` have the real side effects
  described in §2, driven against a Chromium instance the test itself
  launches (Playwright, matching `playwright.config.js`'s launch args) and
  navigates to the real `webxr_client` bundle served locally (reuse
  `npm run dev-server:component-mock` or an equivalent dev-server target
  that serves the *real* CloudXR component, not `MockCloudXR.ts`).
- New integration test (separate file or a clearly-marked slow section of
  `test_oob_teleop_adb.py`) driving the real `run_oob_connect()` end to end:
  device-online → `am start` opens the real tab → real devtools socket
  reachable → real CDP tab discovery finds the real tab → real
  `_cdp_session_click_connect()` polls the *real* client's readiness state
  machine (real IWER capability checks) → real click → real monitor task
  reading the real client's error banner.
- Keep this as new, separate, explicitly slower coverage — don't fold it
  into the existing fast `mock_adb()`/fake-CDP-server suite from PR #1148.

## 4. Setup gaps (wiring/plumbing to build — not mocks)

Confirmed: nothing in §1–3 needs a protocol-level mock. `npm run dev-server`
(`webpack.dev.js`, port 8080) already serves the *real* production client
(`App.tsx`), not `MockCloudXR.ts` — this is "the CloudXRComponent" the test
needs, and it's already runnable today. What's actually missing is wiring:

1. **`wss.py` isn't started anywhere for this purpose.** Needs launching as
   a real process with the OOB hub wired in (`OOBControlHub`) and *some*
   `backend_host`/`backend_port` configured for the CloudXR-signaling
   forward target — even an unused one is fine, since `/oob/v1/ws` and the
   `/api/oob/v1/*` HTTP routes work independently of whether anything
   answers on the signaling side. It generates its own self-signed cert; no
   action needed there.
2. **No code today launches a real Chromium process directly** (as opposed
   to through `adb`). `FakeAdb` only patches `subprocess.run` for synthetic
   `adb` calls. Need a small helper that spawns Chrome with
   `--remote-debugging-port=<port>`, reusing `playwright.config.js`'s known
   working launch args and — critically — its `executablePath` override
   (`/usr/bin/google-chrome`): this sandbox cannot reach Playwright's own
   browser-download CDN, a constraint already hit once building that config.
3. **URL construction** must go through the existing `build_teleop_url()` /
   `build_headset_bookmark_url()` helpers (`oob_teleop_env.py`), pointed at
   `localhost:8080` (the client) with `serverIP`/`port` aimed at the local
   `wss.py` instance and `oobEnable=1` set, rather than hand-assembling a
   query string — this is also what makes `_close_stale_teleop_tabs()`'s
   `oobEnable=` tab filter match.
4. Cert trust is already a non-issue: `_cdp_session_click_connect()` already
   sends CDP `Security.setIgnoreCertificateErrors` before polling.

## 5. Exhaustive check: what talks to the CXR runtime directly

Repo-wide search (Python + vendored `deps/`) for any direct connection to
the runtime — its WS port (`backend_port`, default `49100`) or its
`ipc_cloudxr` Unix socket. Exactly two hits, both already covered above:

1. **`wss.py:proxy_handler()` → `ws_connect(f"ws://{backend_host}:{backend_port}{path}")`.**
   The only WS client of the runtime anywhere in the codebase. A dumb
   relay — `_pipe()` forwards every message verbatim, `async for msg in
   src: await dst.send(msg)`, never parsing SDP/ICE/JSON content. This is
   the one seam a mock runtime would need to satisfy to give the client a
   real, successful streaming session (bind `49100`, accept the WS
   connection, do real SDP/ICE negotiation, send synthetic media over a
   real `RTCPeerConnection` — confirmed via `RTCPeerConnection`/
   `createOffer`/`createAnswer`/`addIceCandidate`/`iceServers` strings in
   the vendored `@nvidia/cloudxr` bundle).
2. **`runtime.py:is_runtime_live()` → raw `AF_UNIX` `connect()` to
   `<run_dir>/ipc_cloudxr`.** A liveness probe only (no protocol messages),
   used solely by `CloudXRService._cleanup_stale_runtime` in `service.py`'s
   own launch path. Not touched by the OOB/CDP/client flow this plan covers
   — irrelevant to a mock runtime built for the goal above.

Everything else matching `49100`/`libcloudxr`/`nv_cxr_service` is either the
runtime process *becoming* the listener (`runtime.py:run()`,
`deps/cloudxr/runtime/main.py` — these load `libcloudxr.so` in-process, they
don't connect to anything) or test/comment noise.

**Conclusion: `run_oob_connect()`'s own code (`oob_teleop_adb.py`) never
touches the runtime, directly or indirectly** — it only ever does ADB and
CDP (§1.A/B). The runtime is entirely orthogonal to what this plan's
integration test exercises. A mock runtime is only relevant if the goal
shifts to proving out a real, successful, rendering CloudXR session — a
separate concern from OOB orchestration coverage, and (per the top-of-file
scope note) tracked as its own separate project, not folded into this one.

## 6. Open questions before implementation

1. ~~Does this test need the CloudXR signaling socket to succeed~~ —
   **resolved, see §3**: no runtime backend, real or mocked. Never bind
   `backend_port`; `wss.py` handles the refused connection as an
   anticipated configuration.
2. ~~Where does the local dev-server get started from~~ — **resolved, see
   §7**: a Python-side `conftest.py` fixture, reusing the pattern already
   used for `live_ipc_socket`/`mock_service_deps`.
3. CI feasibility: deliberately deferred (see §7) — this test runs manually
   for now; CI wiring (parallel-worker port collisions, ephemeral ports,
   headless-runner Chrome availability) is a future PR.

## 7. Dev-server launch: where and how (manual-run scope; CI deferred)

Surveyed two precedents before deciding:

- **This repo's `playwright.config.js`** uses Playwright's own `webServer`
  config (`command: 'npm run dev-server:component-mock'`, fixed port 8083,
  `reuseExistingServer: !process.env.CI`) — JS-test-runner machinery, not
  usable from a Python pytest test.
- **`~/gitlab/cloudxr-js`'s `tests/playwright/global-setup.js`** spawns
  `npx http-server <root> -p 0 --cors` as a child process, scrapes the
  ephemeral bound port from stdout, writes it to a side-channel file
  (`.http-server-port`) for fixtures to read — built for CI parallelism
  (many workers, no fixed-port collisions). That repo also runs a much
  heavier **real** (not mock) streaming e2e
  (`tests/docker-compose.test-e2e.yaml`): three containers
  (`cloudxr-runtime` = the actual GPU-backed CloudXR runtime image,
  `webapp`, `playwright`), requiring an NVIDIA GPU + docker +
  `nvidia-container-toolkit`. Confirms no lightweight mock runtime exists
  anywhere in either repo today — that remains genuinely separate,
  greenfield work (per the top-of-file scope note), well outside what this
  plan needs.

**Decision for this plan:** launch the dev server from a new Python-side
`@contextmanager` fixture in `tests/python/core/cloudxr/conftest.py`,
following this repo's own existing convention (`live_ipc_socket`,
`mock_service_deps`) rather than cloudxr-js's Node-side pattern:

- `subprocess.Popen(["npm", "run", "dev-server"], cwd=<repo>/deps/cloudxr/webxr_client, ...)`,
  fixed port `8080` (matches `webpack.dev.js`'s default, and what's already
  running manually in this session today) — skip ephemeral-port scraping;
  not needed without parallel CI workers.
- Mirror `playwright.config.js`'s `reuseExistingServer: !process.env.CI`
  idea: probe port 8080 first and reuse an already-running server if found;
  only spawn (and later tear down) one if nothing answers. Keeps manual runs
  fast and avoids colliding with a dev-server already running for other
  purposes.
- Ephemeral-port scraping (`cloudxr-js`'s approach) and any CI-specific
  wiring are explicitly deferred to the future CI PR.
