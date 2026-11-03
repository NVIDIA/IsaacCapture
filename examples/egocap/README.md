<!--
SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

SPDX-License-Identifier: Apache-2.0
-->

# Ego Capture UI

Ego Capture UI is an IsaacCapture example for recording egocentric camera video and headset/
controller/hand tracking: a headset and OAK-D cameras stream through **CaptureServer** to a
Python **backend**, which previews the feed, records episodes to disk, and serves two small
browser pages — a live capture/recording view (React) and an optional recordings browser
(TypeScript). Camera and tracking timestamps mark each source's own host-sampling step, not a
hardware exposure time, and are not frame-exact synchronized across sensors. Commands below run
from this example's own `egocap/` directory inside an IsaacCapture checkout.

`egocap/` is this example's own Python project (one `pyproject.toml`, one `uv.lock`, one `.venv`)
plus two independent Vite/TypeScript frontends:

- **`capture_server/`** — a standalone WebSocket server exposing tracking and camera data in a
  fixed wire protocol, with a mock source for development without hardware. See
  [`capture_server/README.md`](capture_server/README.md) for setup, the Python client API, and
  the full protocol reference.
- **`recording/`** — the recording module library: configuration I/O, episode storage/layout,
  video writers, QC and studio/task-library lookups. See [`recording/README.md`](recording/README.md).
- **`recording/recording_service/`** — the recording service built on that library: validates,
  writes and finalizes episodes. See [`recording/recording_service/README.md`](recording/recording_service/README.md).
- **`backend/`** — the FastAPI app tying CaptureServer, the recording service and both frontends
  together. See [`backend/README.md`](backend/README.md).
- **`frontend/`** — the live capture/recording page. See [`frontend/README.md`](frontend/README.md).
- **`recordings_frontend/`** — an optional recordings browser/downloader. See
  [`recordings_frontend/README.md`](recordings_frontend/README.md).

See [`DESIGN.md`](DESIGN.md) for architecture, data flow and design decisions, and
[`AGENTS.md`](AGENTS.md) for the project layout and contributor/test conventions.

## Install

```bash
cd egocap
uv sync --extra dev
```

This resolves every Python dependency from PyPI, including the 3D tracking view (Viser); no
external checkout is needed to install or import the Python side. One optional extra adds real
hardware support: `--extra oak` (DepthAI, for real OAK-D cameras).

**System prerequisites beyond `uv sync`:** a real `ffmpeg` binary on `PATH` (the default video
encoder shells out to it — recording, and `tests/recording`, need it; capture preview alone does
not), and Node/npm to build or test either frontend (`frontend/`, `recordings_frontend/`).
`tests/backend`'s real-app page-serving test builds `recordings_frontend/` automatically via a
pytest fixture; the rest of `tests/backend` needs no Node/npm.

## Quickstart (mock data, no hardware)

Three processes, each in its own terminal, from `egocap/`:

```bash
# 1. A standalone mock CaptureServer -- synthetic tracking and camera frames, no hardware
uv run capture-server --mock

# 2. The backend, connecting to CaptureServer and serving both frontends -- note the
#    "Recording control token for this run: ..." line it prints to stderr at startup.
#    EGOCAP_STATION_CONFIG points at the shipped example station file, which requests
#    the mock head/wrist cameras. The template ships with an empty operator name.
EGOCAP_STATION_CONFIG=recording/egocap/stations/ego_stations.example.yaml \
  uv run uvicorn egocap_web.app:app --app-dir backend --port 8000

# 3. The frontend's dev server (first: cd frontend && npm install)
cd frontend && npm run dev
```

Open the URL `npm run dev` prints. Enter an operator name and paste in the control token the
backend printed at startup — both are required before any recording command is accepted. See
[Configuration](#configuration) and
[`frontend/README.md`](frontend/README.md) for the full station-file and environment-variable
reference.

## Real hardware

CaptureServer's real mode needs its own Python bindings built in an
[IsaacCapture](https://github.com/NVIDIA/IsaacCapture) checkout (`ISAAC_ROOT`, the checkout root)
for tracking, and the `oak` extra plus a camera's device ID for OAK-D cameras. See
[`capture_server/README.md`](capture_server/README.md) for exact setup and validated hardware.
The backend and frontends are unchanged between mock and real mode — CaptureServer is the only
process that differs.

## Both UIs

- The **capture/recording page** (`frontend/`) is always served once built; see
  [`frontend/README.md`](frontend/README.md) for its development setup and the live protocol it
  consumes.
- The **recordings browser** (`recordings_frontend/`) is an optional, unauthenticated file
  browser over the recording output root — off by default (`EGOCAP_RECORDINGS_ENABLED=1` to
  enable). See [`recordings_frontend/README.md`](recordings_frontend/README.md).

## Configuration

A station's identity, initial operator/task, recording output location, and requested cameras come
from a deployment station file (`EGOCAP_STATION_CONFIG`), never something the browser supplies —
see [`backend/README.md`](backend/README.md) and the shipped
[`recording/egocap/stations/ego_stations.example.yaml`](recording/egocap/stations/ego_stations.example.yaml)
template. Unset, the backend keeps a configuration-free development workflow. Everything else is
an environment variable:

| Variable | Affects |
| --- | --- |
| `EGOCAP_STATION_CONFIG` | Path to the deployment station file described above. See [`backend/README.md`](backend/README.md). |
| `EGOCAP_CAPTURE_SERVER_URL` | Where to connect for tracking/camera data. See [`frontend/README.md`](frontend/README.md). |
| `EGOCAP_DATA_DIR`, `EGOCAP_CALIBRATION_DIR` | Where configuration/calibration defaults live. See [`recording/README.md`](recording/README.md). |
| `EGOCAP_TASK_FILES_DIR` | Which task-catalog directory the task-file API serves. See [`data/taskfiles/README.md`](data/taskfiles/README.md). |
| `EGOCAP_RECORDINGS_ENABLED` | Turns on the recordings browser. See [`recordings_frontend/README.md`](recordings_frontend/README.md). |
| `EGOCAP_ALLOWED_ORIGINS`, `EGOCAP_TRUSTED_HOSTNAMES` | Origin admission for the live `WS /api/session` connection. See [`backend/README.md`](backend/README.md). |
| `EGOCAP_VISER_ENABLED` and the other `EGOCAP_VISER_*` variables | The embedded 3D tracking view, on by default. See [`frontend/README.md`](frontend/README.md). |

## Building both frontends

```bash
uv run python build_support.py
```

Builds `frontend/dist/` and `recordings_frontend/dist/`, which the backend serves directly once
present. `uv build` produces a Python-only wheel; it does not bundle either frontend, and the
backend locates `frontend/dist/`, `recordings_frontend/dist/`, `data/taskfiles/` and
`assets/viser/` by checkout-relative paths, not as installed-wheel resources -- the backend is run
from this checkout, not from an installed wheel. Packaging them as proper installed-wheel
resources is separate, not-yet-done work.

## Tests

```bash
uv sync --extra dev
uv run pytest tests/recording -q --cov=recording/egocap --cov-branch --cov-fail-under=100
uv run pytest tests/capture_server -q --cov=capture_server/capture_server --cov-branch
uv run pytest tests/recording_service -q --cov=recording/recording_service --cov-branch
uv run pytest tests/backend -q
(cd frontend && npm install && npm run lint && npm run test)
(cd recordings_frontend && npm install && npm run lint)
```

See [`AGENTS.md`](AGENTS.md) for coverage gates, module boundaries enforced by tests, and when
to run the real-episode acceptance tool. None of these suites need real hardware; CaptureServer
is mocked throughout. `tests/recording`'s default video encoder needs a real `ffmpeg` binary on
`PATH`.

## Troubleshooting

- **No camera preview** — `EGOCAP_STATION_CONFIG` is unset, its station file requests no cameras,
  or its station file is invalid (an invalid file also requests no cameras, never a silent
  fallback); see [`backend/README.md`](backend/README.md).
- **Backend can't reach CaptureServer** — confirm a CaptureServer (mock or real) is already
  listening at `EGOCAP_CAPTURE_SERVER_URL` *before* starting the backend; the backend makes one
  connection attempt at startup and does not retry a server that wasn't there yet.
- **Frontend dev server shows nothing / proxy errors** — the backend must already be running on
  the port `frontend/vite.config.ts` proxies to (127.0.0.1:8000 by default).
- **Recordings browser 404s** — it's off by default; set `EGOCAP_RECORDINGS_ENABLED=1`.
