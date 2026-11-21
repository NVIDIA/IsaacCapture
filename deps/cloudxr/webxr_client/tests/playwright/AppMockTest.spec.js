/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// @ts-check
const { test, expect } = require('@playwright/test');

/**
 * Minimal smoke test: the real production App.tsx (not the scripted CloudXRComponentTest.tsx
 * harness), running against MockCloudXR via the build-time @nvidia/cloudxr alias
 * (webpack.app-mock.js, :8082), under IWER emulation (no headset). Just proves the app loads
 * and a session can be connected - the foundation the client-UI-state tests (passthrough-only,
 * missing-panel, stale-tab, certificate, client-not-loaded) build on top of.
 */

test('opens the real app against MockCloudXR and connects', async ({ page }) => {
  const consoleLines = [];
  page.on('console', msg => consoleLines.push(msg.text()));

  await page.goto('http://localhost:8082/');

  await page.click('#startButton', { timeout: 15000 });

  // CloudXRComponent itself logs this unconditionally in onStreamStarted, before calling
  // onStatusChange(true, 'Connected') - the same signal-of-record used throughout this
  // repo's other Playwright coverage (CloudXRComponentTest.spec.js).
  await expect
    .poll(() => consoleLines.some(l => l.includes('CloudXR stream started')), {
      timeout: 15000,
      message: () => `never saw a stream start; console so far:\n${consoleLines.join('\n')}`,
    })
    .toBe(true);
});
