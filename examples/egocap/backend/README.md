<!--
SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

SPDX-License-Identifier: Apache-2.0
-->

# backend

The FastAPI application (`egocap_web`) tying CaptureServer, the recording service and both
frontends together. See the root [`README.md`](../README.md) for install/quickstart and
[`../DESIGN.md`](../DESIGN.md) for architecture; this file covers the backend's own routes,
configuration and development workflow.

## Starting it

```bash
# From egocap/ -- a CaptureServer (mock or real) must already be listening
uv run uvicorn egocap_web.app:app --app-dir backend --port 8000
```

The backend makes exactly one CaptureServer connection attempt at startup and owns it for the
whole process lifetime; it is never per-browser-tab, and a browser reconnect never starts, stops
or restarts it.

## Routes

| Route | Purpose |
| --- | --- |
| `WS /api/session` | The main live connection: capture-status snapshots, sampled binary tracking frames, and recording commands/replies/snapshots. One socket per browser tab; see [`frontend/README.md`](../frontend/README.md) for the full message shapes. |
| `GET /stream{id}` | MJPEG camera preview, bound to the backend/session identity advertised in the `/api/session` snapshot. |
| `GET /api/capture-status` | Thin HTTP inspection endpoint mirroring the latest snapshot; the frontend does not poll this during normal operation. |
| `GET /api/settings` | The backend's effective configuration (CaptureServer URL, sources, station file, recording root, 3D view, origin policy), read-only and gated by the `X-Control-Token` header; the page's Advanced settings tab shows it. Secrets are reported as present or absent, never by value. |
| `GET`/`PUT /api/operator-settings` | The operator's station setup: the camera type (`sensing_gmsl` or `oak_d`), camera roles (`ego`, `wrist_left`, `wrist_right`), the task catalog the recording panel's picker offers (`task_file`), the task it preselects (`preselected_task`) and whether calibration files are required before recording (`require_calibration`, a boolean, `false` by default; see [Start-of-day flow](#start-of-day-flow)). Kept in `.egocap/operator_settings.json` under the recording output (hidden from the recordings browser) (`EGOCAP_OPERATOR_SETTINGS_FILE` overrides the path) so every browser sees the same values. `PUT` names only the fields it changes and uses `X-Control-Token`; legacy JSON `control_token` remains accepted. A wrong token is 403 and bodies above 16 KiB are rejected with 413 before parsing. |
| `GET`/`POST /api/stations`, `PUT`/`DELETE /api/stations/{id}`, `POST /api/stations/select` | The station registry: the stations an operator chooses between and the one in use. A station is a name, a camera type, the camera filling each role and the capture host address last given to its headset. Kept in `stations.json` beside the operator settings. Every change uses `X-Control-Token` and answers with the new listing; a refusal is a 400 with the specific reason (a duplicate name, a camera in two roles, an unknown field), and an unknown id is 404. The station in use cannot be selected, edited or deleted while a take is being written, and a change that would start a new batch (selecting another station or none, renaming or deleting the station in use) is refused with `recover or discard the interrupted take first` while an interrupted take waits. A station's headset address is listed only when the `GET` carries the token. See [Stations](#stations). |
| `GET`/`POST /api/station-uptime` | What the station in use is doing now (`ready`, `recording`, or `down` with the reason) and its active and down time between today's first and last take, with each down period as a blocker (`end` is `null` for one that has not ended). It is observed by the backend, never set: `POST` (`X-Control-Token`) only saves the day's notes, with the body `{"station": <the station the notes were written for>, "notes": <text>}`; when that is no longer the station in use the reply is a 409 and nothing is written. `GET` is open, and returns `notes` as an empty string unless it carries `X-Control-Token`: the notes are the operator's free text, and the end-of-day report, which needs the token, is where the page reads them. See [Stations](#stations). |
| `GET /api/setup` | The start-of-day flow: the four steps' states, the live station checklist (`checks`, each with a `status` of `pass`, `fail`, `warn` or `skip`), `enforced` (whether the checklist gates recording or only advises), `blocked_reason` (why station setup holds a take; only a failed item does) and, when the request carries `X-Control-Token`, `headset`: the capture host address to enter in the headset, the one remembered for the station in use, and whether that one stopped being valid. Everything in one reply describes one station: if the station in use keeps changing while it is gathered the reply is a 503 (`the station in use is changing; try again`). Read-only. See [Start-of-day flow](#start-of-day-flow). |
| `GET /api/day-report`, `POST /api/day-report/save`, `POST /api/day-report/correct` | The end-of-day report, all with `X-Control-Token`. `GET` returns today's takes for the station in use by session, with a count and duration per disposition (kept, flagged, rejected, discarded, pending, recording) read from the recording service's manifests, plus the station's time (`uptime`). `save` writes `<station>_<date>_report.json` beside the daily log and refreshes that log's recorded time. `correct` rewrites the task of one take or a range of a session's takes (`session`, `episodes`, `task`). Each take carries `location`, where its data is under the recording output. See [End-of-day report](#end-of-day-report). |
| `POST /api/task-files/import`, `DELETE /api/task-files/{name}` | Import a task catalog the operator supplies — by `name` + `content` (text uploaded from the browser) — into the imported-catalog directory beside the operator settings, once it parses; and remove an imported one. `POST` and `DELETE` use the `X-Control-Token` header (legacy JSON tokens remain accepted for `POST`); `DELETE` refuses removal of the selected catalog (409); imports also refuse replacement of that catalog until it is deselected. Imports appear in `GET /api/task-files` (as `imported`) and shadow a shipped catalog of the same name. |
| `GET /api/tools` | The operator tools the Settings dialog's Devices tab lists (headset calibration, glove calibration, fisheye rectification for the SENSING cameras), each with whether this build can launch it. None can yet. |
| `GET /api/livekit-token?room=ROOM` | Viewer token (subscribe and data messages, no media publishing) for a LiveKit room the session offers (404 otherwise, 503 without signing credentials). Which streams exist comes from CaptureServer's session message. |
| `GET /api/task-files`, `GET /api/task-files/{name}` | Read-only task-catalog API; see [`../data/taskfiles/README.md`](../data/taskfiles/README.md). |
| `POST /recordings/api/session` | Exchanges `X-Control-Token` for the recordings browser's same-site, HTTP-only cookie, scoped to `/recordings`. |
| `GET /recordings/...` | The optional recordings browser's API and file routes, off unless `EGOCAP_RECORDINGS_ENABLED=1`. The listing and every file need the control token (header) or that cookie, else 401, and a request from another site is 403; see [`../recordings_frontend/README.md`](../recordings_frontend/README.md). |

Both frontends' built `dist/` directories are served as static assets from this same process
once present (`uv run python build_support.py` from the project root builds them).

## Configuration

The station's identity, initial operator/task, recording output location, and requested cameras
come from a deployment station YAML pointed at by `EGOCAP_STATION_CONFIG` (see
[`station_config.py`](egocap_web/station_config.py) and the shipped
[`recording/egocap/stations/ego_stations.example.yaml`](../recording/egocap/stations/ego_stations.example.yaml)
template). Unset, the backend keeps the configuration-free development workflow below; a
configured-but-invalid file blocks new recording AND requests no cameras at all (capture preview
goes empty too, it is not unaffected), with a reason shown in the UI, never a silent fallback.
Everything else is an environment variable:

### Stations

A site usually has several stations, and an operator picks the one they are at. The registry
([`stations.py`](egocap_web/stations.py)) holds them; the Settings dialog's Station tab edits it.

- **No stations** is the configuration-free default: the deployment's own station label applies and no station
  rule blocks recording.
- **Once stations exist**, a take is refused until one is in use and can record: its ego role is assigned and every
  camera it names is offered by the capture session. The reason is published as `not_ready_reason` in the recording
  snapshot and returned by a refused `start_episode`.
- **Changing the station in use applies in process.** The label shown and recorded changes at once and the
  recording batch rotates, so the next take opens a session carrying the new station's name. It is refused while a
  take is being written. Any change to the station in use, its camera assignments included, holds a take for the
  moment it is applied (`the station is being changed`), so none can open a session under the outgoing name or
  record against cameras being reassigned; the batch rotates only when the label changes. While an interrupted
  take waits to be recovered or discarded, a change that would rotate the batch is refused (`recover or discard
  the interrupted take first`): a new batch would leave that take on disk with nothing on the page to resolve
  it. Which cameras CaptureServer is asked for stays a deployment
  setting (the station file); restarting the backend is the only way to change that.
- **The station in use holds the camera setup.** `GET /api/operator-settings` returns its camera type and roles, and
  they are changed through the station routes; a `PUT` of those two fields to the operator settings is refused
  while a station is in use.
- **Names.** Two names that would share one daily log file (they differ only in case or punctuation) are one name.
  Renaming the station in use carries its day over: its timeline, its daily log and its saved report move to the
  new name, and takes recorded under the old name stay part of its report. Once another station records under
  the old name, that name's takes are that station's.
- **Headset address.** The backend remembers, per station, the capture host address its headset was given. It does
  so by itself, once a headset is connected, and replaces the remembered one only when the host no longer has it
  and the station and its saved address are still the ones that were looked at.

### Station time

Nobody marks a station active or blocked. Every two seconds the backend asks whether the station in use is
recording, ready to record, or down (a take would be refused for a station, device or capture reason) and logs the
day as segments ([`station_uptime.py`](egocap_web/station_uptime.py)).

- **Active and down time are counted between the first and last take of the day**, so a station left powered on
  outside working hours adds to neither. Each down period inside that window is a blocker, kept with the reason it
  began with. A missing task or operator name is not downtime. An interrupted take waiting to be recovered or
  discarded is: nothing records until it is resolved.
- A new state must hold for a few seconds before it is logged, so a device that flickers is not a blocker; the
  start and end of a take are logged at once.
- Time nothing was sampled for more than a minute and a half (the backend was not running, or its host was
  suspended) is logged as down. The log uses wall-clock time, so a forward clock step of more than that is
  logged the same way. A day that crosses midnight is split there, and a take in progress at midnight brings
  the old day's daily log up to it. A backend that stopped with the day still open and starts again on a later
  day finishes that day first: its open segment ends where it was last seen, the rest of it is down, and its
  daily log is brought up to its midnight.
- A daily log or a midnight split that could not be written is made good at the next save of the timeline, so
  one failed write does not leave a day's log behind or lose a take that crossed midnight.
- A station's day holds a bounded number of segments. Past that its state is no longer logged and it reports
  `down` with the reason that the day's log is full.
- Files, all under `station_status/` in the recording output, written atomically and never uploaded:
  `<date>_timeline.json` (the segments and the day's notes, written once the day has a take or notes),
  `<station>_<date>_activity.json` (the daily log: active and down time, blockers, recorded time; rewritten after
  each take) and `<station>_<date>_report.json` (the saved end-of-day report).

### Start-of-day flow

The flow is station, devices, record, wrap-up ([`setup_service.py`](egocap_web/setup_service.py),
[`setup_checks.py`](egocap_web/setup_checks.py)). With a station in use it gates recording; with none it only advises.

- **The checklist is live.** There is nothing to run: a background duty judges it every second, and
  `GET /api/setup` judges it afresh. It reads the calibration directory and the recording disk, so it is judged in
  a worker thread, one evaluation at a time; a slow mount cannot stall the event loop that carries capture frames.
- **What holds a take** is a failed item, and an item fails only for what recording readiness itself refuses, plus
  a full disk: the capture session not running, no recent tracking sample, a station camera that is not offered,
  delivers no recent frame or, for a LiveKit camera, cannot be recorded, and less than 10 GB free on the disk
  holding the recording output.
- **What only warns** (`warn`) is shown with its reason and never holds a take; the Devices step stays done and
  names it. A tracking device or a camera that reports another state than connected warns: readiness judges the
  data that arrives, not the state a device reports, and a station may leave a controller or a glove unused.
- **Calibration files are optional unless the deployment requires them.** The files of the roles in use are
  looked for under `EGOCAP_CALIBRATION_DIR` (head intrinsics and extrinsics, each wrist's intrinsics, and the
  wrist offsets once a wrist is assigned); calibration is skipped for a mock session. A missing or unreadable
  file warns. With the operator setting `require_calibration` on, the same item fails and holds recording.
- **Skipped, never failed:** a role the station leaves unassigned, and any check that does not apply.
- **Record follows the day, not the batch.** The step is done once the station in use has a take in today's
  timeline, so it stays done after a new batch or a restart and starts over at midnight; the current batch's
  saved count is only a detail.
- **The gate** decides whether a take is admitted and the readiness that is published. It never interrupts a take:
  a setup fact changing mid-take does not stop it, unlike a capture source going stale. It is asked on the event
  loop, during a take too, and answers from the last evaluation in memory. Until the station in use has been
  judged (at startup, just after it changed, or when the last result is more than ten seconds old because an
  evaluation is stuck) a take is held with `the station checks are being evaluated`. A checklist that cannot be
  evaluated holds the take and is logged; it never raises into a viewer's connection.

### End-of-day report

[`egocap.analytics.day_report`](../recording/egocap/analytics/day_report.py) reads each session's manifests, so every
take is counted with what became of it:

| Disposition | Meaning |
| --- | --- |
| kept | Finalized, and capture quality did not mark it bad. |
| flagged | Finalized and kept, but capture quality marked its data bad. |
| rejected | Failed its check; a camera failure also removed its media. |
| discarded | The operator discarded it. |
| pending | Interrupted before it finished; it waits to be recovered or discarded. |
| recording | The take being written right now. |

A take belongs to the day in its finalized episode's `recorded_at`, else the day its manifest was last checkpointed.
A session whose name says it was opened on a later day than the report's is passed over without being read; one
opened on that day or earlier, or with no date in its name, is always read, since it may hold the day's takes.
A manifest that cannot be read, or that holds values of the wrong shape, is reported with its session, never dropped.

A correction ([`day_report_routes.py`](egocap_web/day_report_routes.py)) accepts only a recorded session inside the
recording output: a recording-service session directly under it, or a studio's group one level below that carries
the recording service's route stamp. The path is resolved first, so a traversal, an absolute path or a link out is
refused, as is a time-aligned copy that leads out. A time-aligned copy or a single episode is never a target of its
own: the copy is corrected with its session. A value the filesystem cannot look up is refused like any other
directory that is not a session, and a range covers at most 10,000 takes.
It is refused while a take is being written and for the session still open for recording, including that session's
routed groups; an interrupted take holds only its own session. It is also refused, to be tried again, while a take
of that session is being time-aligned, since the copy being built already has the old label, and while another
correction of that session is being written. One lock per session ([`session_locks.py`](egocap_web/session_locks.py))
is held by the correction across every file it writes and by the aligner across a whole alignment, so an episode
waiting to be aligned does not start in the middle of a correction. Each metadata file is
replaced whole, both task labels of a take are kept in step, and the session's own task label changes only when
every take was corrected. Nothing is ever moved.

| Variable | Default | Meaning |
| --- | --- | --- |
| `EGOCAP_STATION_CONFIG` | unset (development default) | Path to the deployment station YAML described above. |
| `EGOCAP_CAPTURE_SERVER_URL` | `ws://127.0.0.1:8082` | Where to connect for tracking/camera data. |
| `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | unset | The LiveKit server's key pair. The backend signs two viewer tokens with it (subscribe and data messages, no media publishing): viewer tokens for `GET /api/livekit-token`, and the recorder's own token. LiveKit recording will not start without both variables; see [`LIVEKIT_PREVIEW.md`](../LIVEKIT_PREVIEW.md). |
| `EGOCAP_TASK_FILES_DIR` | the shipped `data/taskfiles/` | Which task-catalog directory the task-file API serves. |
| `EGOCAP_RECORDINGS_ENABLED` | off | Turns on the recordings browser's routes. |
| `EGOCAP_ALLOWED_ORIGINS` | none (same-origin only) | Comma-separated extra origins allowed to open `WS /api/session` — needed when the frontend is hosted separately or behind a reverse proxy. |
| `EGOCAP_TRUSTED_HOSTNAMES` | unset (dev default) | Comma-separated hostnames trusted for the same-origin fallback on `WS /api/session`. Leaving this unset derives the expected origin from the request's own `Host` header, which is fine for local development but not a safe default for a deployment reachable from an untrusted network. |
| `EGOCAP_VISER_ENABLED` and related `EGOCAP_VISER_*` | on | The embedded 3D tracking view; see [`frontend/README.md`](../frontend/README.md). |

## Optional LiveKit support

Recording LiveKit camera video needs the `livekit` extra (`uv sync --extra livekit`). Without it the backend
starts normally and the browser preview works; only recording a LiveKit stream is refused, with a message that
names the extra. See [`LIVEKIT_PREVIEW.md`](../LIVEKIT_PREVIEW.md).

## Development

```bash
uv sync --extra dev
uv run pytest tests/backend -q
```

`tests/backend` runs against the real app (via `TestClient`) and a real mock CaptureServer
process over an actual WebSocket connection — not a hand-rolled fake transport. No real hardware
or GPU is required.

## Troubleshooting

- **Capture preview never comes up, page shows a connection error** — no CaptureServer was
  listening at `EGOCAP_CAPTURE_SERVER_URL` when the backend started; the backend doesn't exit or
  retry, so start CaptureServer first, then restart the backend.
- **`/api/task-files` returns 503** — the configured (or default) task-file directory is
  missing; capture/recording still work, only the task catalog is unavailable.
- **A browser tab never shows "Connected"** — check `EGOCAP_ALLOWED_ORIGINS` includes the
  frontend's actual origin if it's not being served from the same origin as the backend.

Mutation request limits are enforced while streaming, including requests without Content-Length. CSV imports allow a bounded JSON envelope for escaping, then enforce the 2 MiB CSV limit. Filesystem failures stay in backend logs; responses contain only deliberate public messages.
