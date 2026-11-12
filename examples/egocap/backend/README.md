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
| `GET`/`PUT /api/operator-settings` | The operator's station setup: the camera type (`sensing_gmsl` or `oak_d`), camera roles (`ego`, `wrist_left`, `wrist_right`), the task catalog the recording panel's picker offers (`task_file`) and the task it preselects (`preselected_task`). Kept in `.egocap/operator_settings.json` under the recording output (hidden from the recordings browser) (`EGOCAP_OPERATOR_SETTINGS_FILE` overrides the path) so every browser sees the same values. `PUT` names only the fields it changes and uses `X-Control-Token`; legacy JSON `control_token` remains accepted. A wrong token is 403 and bodies above 16 KiB are rejected with 413 before parsing. |
| `POST /api/task-files/import`, `DELETE /api/task-files/{name}` | Import a task catalog the operator supplies — by `name` + `content` (text uploaded from the browser) — into the imported-catalog directory beside the operator settings, once it parses; and remove an imported one. `POST` and `DELETE` use the `X-Control-Token` header (legacy JSON tokens remain accepted for `POST`); `DELETE` refuses removal of the selected catalog (409); imports also refuse replacement of that catalog until it is deselected. Imports appear in `GET /api/task-files` (as `imported`) and shadow a shipped catalog of the same name. |
| `GET /api/tools` | The operator tools the Settings drawer offers (headset calibration, glove calibration, fisheye rectification for the SENSING cameras), each with whether this build can launch it. None can yet. |
| `GET /api/livekit-token?room=ROOM` | Viewer token (subscribe and data messages, no media publishing) for a LiveKit room the session offers (404 otherwise, 503 without signing credentials). Which streams exist comes from CaptureServer's session message. |
| `GET /api/task-files`, `GET /api/task-files/{name}` | Read-only task-catalog API; see [`../data/taskfiles/README.md`](../data/taskfiles/README.md). |
| `GET /recordings/...` | The optional recordings browser's API and file routes, off unless `EGOCAP_RECORDINGS_ENABLED=1`; see [`../recordings_frontend/README.md`](../recordings_frontend/README.md). |

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
