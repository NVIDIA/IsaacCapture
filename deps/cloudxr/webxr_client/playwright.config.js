/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// @ts-check
const { defineConfig } = require('@playwright/test');

/**
 * Chromium launch args, matching ~/gitlab/cloudxr-js's tests/playwright/helpers/fixtures.js
 * SHARED_LAUNCH_ARGS: SwiftShader software WebGL (this sandbox has no GPU) and
 * --disable-features=WebXR so native/real XR is off and IWER (loaded by the page itself)
 * provides navigator.xr.
 */
const CHROMIUM_ARGS = [
  '--no-sandbox',
  '--disable-setuid-sandbox',
  '--ignore-gpu-blocklist',
  '--enable-webgl',
  '--use-angle=swiftshader',
  '--use-gl=angle',
  '--disable-features=WebXR',
];

// This sandbox cannot reach Playwright's own browser-download CDN (`npx playwright install`
// fails). Point at the system-installed Google Chrome instead; override via
// PLAYWRIGHT_CHROME_PATH if that path differs elsewhere (e.g. CI).
const CHROME_EXECUTABLE_PATH = process.env.PLAYWRIGHT_CHROME_PATH || '/usr/bin/google-chrome';

module.exports = defineConfig({
  testDir: './tests/playwright',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  reporter: [['line']],
  timeout: 60000,

  // Builds and serves both mock targets before tests run, tearing them down after:
  //  - CloudXRComponentTest.html on :8083 (dev-server:component-mock) - the purpose-built
  //    scripted test harness page.
  //  - the real App.tsx on :8082 (dev-server:app-mock) - the actual production UI against
  //    MockCloudXR via the build-time @nvidia/cloudxr alias, for tests that need the real
  //    settings panel / CloudXRUI, not a scripted harness.
  webServer: [
    {
      command: 'npm run dev-server:component-mock',
      url: 'http://localhost:8083/CloudXRComponentTest.html',
      reuseExistingServer: !process.env.CI,
      timeout: 60000,
    },
    {
      command: 'npm run dev-server:app-mock',
      url: 'http://localhost:8082/',
      reuseExistingServer: !process.env.CI,
      timeout: 60000,
    },
  ],

  use: {
    baseURL: 'http://localhost:8083',
    trace: 'on-first-retry',
    launchOptions: {
      executablePath: CHROME_EXECUTABLE_PATH,
      args: CHROMIUM_ARGS,
      ignoreDefaultArgs: ['--hide-scrollbars'],
    },
  },

  projects: [
    {
      name: 'chromium',
      use: { ...require('@playwright/test').devices['Desktop Chrome'] },
    },
  ],
});
