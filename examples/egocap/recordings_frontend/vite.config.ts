// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

import { defineConfig } from "vite";

// A separate Vite project from ../frontend -- its own build, own dev
// server, own output directory -- so the recordings feature never touches
// the main app's toolchain. Dev-time proxy mirrors ../frontend's for the
// same reason: hit a locally running backend
// (`EGOCAP_RECORDINGS_ENABLED=1 uv run uvicorn egocap_web.app:app --app-dir
// backend --port 8000` -- the feature is off by default, see
// recordings.py's module docstring) without a separate CORS setup.
export default defineConfig({
  // The built page is served under /recordings/ (recordings.py mounts
  // dist/ at that prefix), so asset URLs must be prefixed too -- the
  // default root-absolute base would point at the main app's own mount.
  base: "/recordings/",
  server: {
    proxy: {
      "/recordings/api": "http://127.0.0.1:8000",
      "/recordings/files": "http://127.0.0.1:8000",
    },
  },
  build: {
    outDir: "dist",
  },
});
