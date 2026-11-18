<!--
SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

SPDX-License-Identifier: Apache-2.0
-->

# egocap-recordings-frontend

PoC demo page for browsing and downloading recorded files -- not part of
the final system design, and not a `RecordingService` responsibility of its
own. See `../backend/egocap_web/recordings.py`'s module docstring for the
backend side.

A separate Vite/TypeScript project from `../frontend` -- its own
`package.json`, `tsconfig.json`, `vite.config.ts`, and output directory --
so this feature never touches the main capture-preview app's toolchain.
`src/main.ts` polls `GET /recordings/api/list[/<path>]` every 4s and
renders a breadcrumb/grid view; clicking a file hits
`GET /recordings/files/<path>` to download it.

Once `EGOCAP_RECORDINGS_ENABLED=1` is set, this page serves every file under
the recording root, unauthenticated, to anyone who can reach the backend --
there is no login, token, or Origin check of its own. Only enable it on a
deployment where that's an acceptable exposure.

## Setup

```bash
npm install
```

## Development

The backend must be running with the feature enabled (off by default --
see `recordings.py`'s module docstring):

```bash
# From egocap/
EGOCAP_RECORDINGS_ENABLED=1 uv run uvicorn egocap_web.app:app --app-dir backend --port 8000
```

```bash
# From egocap/recordings_frontend/ -- Vite's dev server, proxying
# /recordings/api and /recordings/files to the backend above (see
# vite.config.ts; the proxy target is 127.0.0.1:8000)
npm run dev
```

## Production build

```bash
npm run build
```

Produces `dist/`, which `recordings.py` serves directly under `/recordings/`
from the same FastAPI process once present -- a source checkout before
`npm run build` here can still run the backend standalone, just without the
page mounted.

## Type checking

```bash
npm run lint   # tsc -b --noEmit
```

No test suite yet -- this is a PoC; see `recordings.py`'s own docstring and
the backend's `tests/backend/test_recordings.py` for what's actually
covered.
