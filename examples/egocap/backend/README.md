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
| `EGOCAP_TASK_FILES_DIR` | the shipped `data/taskfiles/` | Which task-catalog directory the task-file API serves. |
| `EGOCAP_RECORDINGS_ENABLED` | off | Turns on the recordings browser's routes. |
| `EGOCAP_ALLOWED_ORIGINS` | none (same-origin only) | Comma-separated extra origins allowed to open `WS /api/session` — needed when the frontend is hosted separately or behind a reverse proxy. |
| `EGOCAP_TRUSTED_HOSTNAMES` | unset (dev default) | Comma-separated hostnames trusted for the same-origin fallback on `WS /api/session`. Leaving this unset derives the expected origin from the request's own `Host` header, which is fine for local development but not a safe default for a deployment reachable from an untrusted network. |
| `EGOCAP_VISER_ENABLED` and related `EGOCAP_VISER_*` | on | The embedded 3D tracking view; see [`frontend/README.md`](../frontend/README.md). |

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
