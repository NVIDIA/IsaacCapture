<!--
SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

SPDX-License-Identifier: Apache-2.0
-->

# egocap-web-frontend

The capture preview React page. `useCaptureSession` (`src/useCaptureSession.ts`)
owns one `WS /api/session` connection: a snapshot on bootstrap and every
lifecycle change, a heartbeat snapshot at least once a second, and sampled
tracking bytes once capture is running. Camera previews are plain `<img>`
elements pointed at `GET /stream{id}` (MJPEG), bound to the backend/session
identity advertised in the snapshot. This page never starts, stops, or
reconfigures capture itself -- a browser reconnect repairs delivery only.
`GET /api/capture-status` is a thin inspection endpoint; the page's live
updates come over the WebSocket connection instead.

Besides those snapshots, the same socket carries the recording strip's traffic:
the page sends one small `recording_command` JSON message at a time, and the
backend answers each with a correlated `recording_reply` and pushes a
`recording_snapshot` to every viewer when the shared recording state changes.

## WS /api/session snapshot fields

Each `SNAPSHOT=3`/`version=1` message (`backend/egocap_web/preview_protocol.py`,
mirrored by `frontend/src/captureProtocol.ts`) is a capture-preview
projection of real `CaptureClient` callback facts -- nothing here is
inferred or fabricated, except the optional `visualization` member, which is
web presentation metadata.

| Field | Meaning |
| --- | --- |
| `version`, `type` | Always `1`/`3`; an unsupported value is a protocol error, not coerced. |
| `backend_instance_id` | Identifies this backend process; changes across a backend restart. |
| `revision` | Advances only on a meaningful lifecycle/status/descriptor change -- heartbeats and sample arrival never advance it. |
| `station_label`, `profile_label` | Operator-facing labels from `CaptureStatusConfig`, not inferred from hardware. |
| `capture.connecting` | True only during the pre-acceptance connection attempt. |
| `capture.started` | Accepted session ID/mode/stream descriptors once `CaptureStarted` has fired, else `null`. |
| `capture.status` | Latest `CaptureStatus` projection (startup stage/message/`StatusSnapshot`), else `null`. |
| `capture.stopped` | Reason/diagnostic once the attempt has ended, else `null`. |
| `capture.binary_generation` | Nonzero once this connection's descriptors are installed; `null` before acceptance and after terminal completion. |
| `samples.tracking`, `samples.cameras` | Latest sequence (decimal string) and backend-local `age_ms` per advertised stream; `samples.cameras[].url` is the `GET /stream{id}` URL for that camera. |
| `visualization` | Optional web-presentation member, not a capture fact: the 3D view's `status` (`disabled`, `ready`, `unavailable`), the `url` to embed and a short `diagnostic`. Absent from a backend without the 3D view; the page treats absence as no 3D view. |

### Binary tracking messages

A 32-byte little-endian header (generation, kind, reserved, stream ID,
sequence, capture/receive timestamps) precedes the unchanged
`ic_tracking_v1` tracking payload -- see `preview_protocol.py`'s table and
`frontend/src/trackingDecoder.ts`, which decodes it field-by-field from the
snapshot's own descriptor (never hardcoded offsets).

## Camera rate and latency readout

Each camera tile shows a stat line under its name, e.g. `47.9 / 60 Hz · 21 ms`:
the frame rate measured by the page against the rate the session advertised for
that stream, then the newest frame's producer-side age. The measured rate is the
camera's `samples.cameras[].sequence` delta over the receipt times of the last ~4 s
of snapshots (`src/cameraRates.ts`); it reads `—` until ~1.5 s of history exists
and is hidden whenever the tile is stale or disconnected, so it never looks current
when it isn't. Snapshots arrive about once a second, so a per-frame jitter figure
cannot be derived here and is deliberately not shown; measure jitter on the backend
or with a direct `/stream{id}` client instead.

## Status presentation

`derivePresentation` (`frontend/src/App.tsx`) is the single source every
component (hero, inventories, live region) reads its tone/headline from, so
they can't disagree about what's on screen. It never reports positive
status from a stale or `terminal` response: a response older than the
polling deadline is shown as delayed/stale rather than trusted, and a
`terminal` response always renders its stop/error state rather than
whatever `running`/`status` happened to hold last.

## Setup

```bash
npm install
```

## Development

A CaptureServer must already be listening before the backend starts --
either the real one or `uv run python -m capture_server --mock` (run from
`egocap/`). The backend owns exactly one connection attempt to it, made
once at backend startup; this page never starts, stops, or restarts that
attempt, including via its own Refresh button.

```bash
# From egocap/ -- starts the backend against EGOCAP_CAPTURE_SERVER_URL
# (default ws://127.0.0.1:8082; a mock CaptureServer on that port works too)
uv run uvicorn egocap_web.app:app --app-dir backend --port 8000

# From egocap/frontend/ -- Vite's dev server, proxying /api to the
# backend above (see vite.config.ts; the proxy target is 127.0.0.1:8000,
# so the backend must use that port in development)
npm run dev
```

Fixed defaults the backend starts with, set in `backend/egocap_web/config.py`'s
`CaptureStatusConfig`:

| Setting | Default | Override |
| --- | --- | --- |
| CaptureServer URL | `ws://127.0.0.1:8082` | `EGOCAP_CAPTURE_SERVER_URL` |
| Station label | `Station 01` | the station YAML's label, see below |
| Profile label | `Headset and hands` | fixed |
| Sources | head, controllers, hands | fixed |
| Cameras | none requested | the station YAML's `video_sources`, see below (`type` `1` = `OAK_D_COLOR`, `2` = `SENSING_COLOR`) |
| Plugins | none | `EGOCAP_CAPTURE_PLUGINS`, a comma-separated list of plugin names (case-insensitive); only `manus` exists today, so `manus` requests glove hand tracking. An unknown name stops the backend at startup. |
| Tracking sources | head, controllers, hands | `EGOCAP_CAPTURE_SOURCES`, a comma-separated list of `head`, `controllers`, `hands`, or `none` for a cameras-only session that needs no headset and no IsaacCapture (it cannot be combined with `EGOCAP_CAPTURE_PLUGINS`). |
| Preview sample rates | 10 Hz cameras, 10 Hz tracking | `EGOCAP_CAMERA_SAMPLE_HZ`, `EGOCAP_TRACKING_SAMPLE_HZ`: how often the page is sent camera frames / tracking samples (positive Hz). Raise to the capture rate (e.g. `60`) to see the native feed; every viewer pays for each sample. |

Optionally, `EGOCAP_STATION_CONFIG` points at a deployment station YAML (see
`backend/egocap_web/station_config.py`, and the shipped
`recording/egocap/stations/ego_stations.example.yaml` template) that seeds the station label, the
initial task instruction, the configured operator name, the recording output root, and the
cameras requested from CaptureServer (each `video_sources` entry's `name`/`type` becomes one
client camera selection) at startup. Each `name` must match one of the names the running
CaptureServer instance actually knows about (its own `--camera` list, mock or real) -- an unknown
name rejects the whole capture session, not just that one camera. Unset, the backend keeps today's
configuration-free development workflow; a configured-but-invalid file blocks new recording AND
requests no cameras at all (capture preview goes empty too, it is not unaffected), with a reason
shown in the UI, never a silent fallback.

The configured operator name never auto-commits an active operator -- that always requires the
operator's own explicit "Set operator" click. Whether it even prefills the entry field's draft
text is controlled separately by `meta.auto_fill_operator_name` (off by default): an unconfigured
or explicit-false station always starts with an empty draft.

## 3D tracking view

The page shows a Viser 3D view of the tracked head, controllers and hands, embedded as an iframe under the episode
controls. It is on by default, and runs inside the same backend process as a separate Viser server with its own
port; `EGOCAP_VISER_ENABLED` is an advanced setting for a deployment that wants it off.

| Setting | Default | Meaning |
| --- | --- | --- |
| `EGOCAP_VISER_ENABLED` | on | `0`, `false`, `no` or `off` turns the view off; an unrecognized value is rejected, not silently treated as on |
| `EGOCAP_VISER_HOST` | `127.0.0.1` | Address Viser binds to. Viser has no authentication, so only bind a non-loopback address on a trusted network |
| `EGOCAP_VISER_PORT` | `8085` | Port Viser binds to; the page is told the port actually bound |
| `EGOCAP_VISER_PUBLIC_URL` | none | The browser-facing URL to embed (use this behind an HTTPS reverse proxy, which must forward Viser's HTTP and WebSocket traffic and allow framing) |
| `EGOCAP_VISER_DEVICE` | `pico-4-ultra` | Which shipped VR device set to draw: `pico-4-ultra` (headset and controllers), `pico-4` (its controllers, generic headset) or `generic` (generic headset, controller axes). An unknown name stops the backend at startup. When the capture status names the headset (for example "Pico 4 Ultra headset"), the view switches to that device's models on its own |
| `EGOCAP_VISER_HEAD_MODEL`, `EGOCAP_VISER_HEAD_MODEL_SCALE` | the profile's headset, `1.0` | A different head model file (used for every device) and its fixed scale |

```bash
# From egocap/ -- backend with the 3D view turned off for this run
EGOCAP_VISER_ENABLED=0 uv run uvicorn egocap_web.app:app --app-dir backend --port 8000
```

- With the default loopback bind the view is offered only to a browser on the same machine; for a remote browser set
  `EGOCAP_VISER_HOST` (for example `0.0.0.0`) or `EGOCAP_VISER_PUBLIC_URL`, otherwise the page says the view is unavailable
  and why.
- Models ship in `assets/viser/` (see `assets/viser/NOTICE.md` for sources and licenses): a Pico 4 Ultra headset
  and controllers, Pico 4 controllers and a generic headset. They are plain files in the source tree; deployments must
  provision that directory with the application. The view's `diagnostic` always says which device's models it is
  showing (`models: Pico 4 Ultra`), so a stand-in is never mistaken for the real hardware.
- The hands are drawn from the reported tracking-space joints as they arrive; controller models follow the grip pose.
- Viser draws its own control strip down the right edge of its page and 0.2.x cannot hide it, so the page makes the
  iframe wider and crops that strip (see `.pose-viser-viewport` in `src/styles.css`). The cropped controls can still be
  reached by keyboard from inside the iframe. Reset view reloads the iframe, which returns the camera to the default
  position. Recording shortcuts work only while the page itself has focus: after clicking inside the 3D view, click
  anywhere outside it first.
- A failed or slow start of the view never stops capture or the page; the page shows the 3D view as unavailable with the
  reason.

## Voice commands (optional)

The recording strip supports hands-free voice control via the browser's [SpeechRecognition
API](https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognition) -- a fixed phrase
grammar routed through the SAME command handlers, eligibility checks and task-draft editor the
Record/Stop/Discard buttons and `t` shortcut already use; there is no second recording command
path or backend endpoint for this.

- Press **Space** (outside any text field) to enable voice mode, or click the Voice toggle.
  Listening stays on across utterances and recording start/stop cycles until you press Space
  again, click the toggle, or say "stop listening".
- Supported phrases: `record` / `start recording` / `begin recording`; `stop` / `stop recording` /
  `save recording` (deliberately not "done" -- too common in ordinary conversation with an
  always-on mic); `discard` / `discard recording` (opens the browser's own native `confirm()`
  dialog -- discarding is never a second spoken command); `task <text>` / `set task <text>` (opens
  or updates the task draft); `save task`; `cancel task`; `stop listening`.
- While the task draft is open, only task-related phrases and "stop listening" are accepted;
  record/stop/discard are ignored with feedback so dictated task text can't accidentally start or
  stop an episode.
- Requires a browser with `SpeechRecognition` (or the `webkitSpeechRecognition` prefix) and
  microphone access. In Chrome, by default, the recognizer sends the microphone audio to Google's
  own server-based speech-recognition service and needs internet access -- it is not local/offline
  processing, and the station's microphone audio leaves the device. The application backend itself
  only ever receives the resulting command/task text, never raw audio. If a deployment requires the
  audio to stay on-device, verify a supported on-device recognition mode before enabling this
  feature; nothing here currently makes that choice for you.
- If this browser has no `SpeechRecognition` constructor at all, the Voice toggle is simply not
  shown -- there is no separate "unsupported" message on top of that. If voice recognition itself
  then hits a real permission denial, an unreachable recognition service, or another persistent
  error, automatic listening stops and the reason is shown next to the toggle; fix the underlying
  issue and enable voice again.
- Like most browser APIs that touch the microphone, `SpeechRecognition` generally requires a
  [secure context](https://developer.mozilla.org/en-US/docs/Web/Security/Secure_Contexts)
  (`https://` or `localhost`). On plain HTTP from another host the toggle can still appear (the
  constructor exists) and then fail once clicked with a permission-denied-looking error, rather
  than an upfront "insecure context" message -- serve this page over HTTPS if voice commands
  don't work from a remote deployment.
- A record/stop/discard command's target (which batch/episode it acts on) is captured once, at
  the browser's first *interim* recognition result for that utterance -- not at `onspeechstart`,
  which some browsers fire once per listening session rather than once per utterance. A command
  that's recognized as already-final with no interim observation first (e.g. a very short
  utterance) has no target to bind and is rejected with "please repeat," the same as a target that
  changed before the command's final result arrived.

## Production build

```bash
npm run build
```

Produces `dist/`, which `backend/egocap_web/app.py` serves directly from the
same FastAPI process (`uvicorn egocap_web.app:app --app-dir backend`) once
present -- no separate frontend server in production.

## Tests and type checking

```bash
npm run lint   # tsc --noEmit
npm run test   # vitest run
```
