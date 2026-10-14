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

  // startButton can report enabled before IWER's device.installRuntime() actually finishes
  // (a real race in App.tsx's capability-check gating, not a test artifact - see LoadIWER.ts),
  // so clicking on "IWER loaded as fallback." (the first script's onload) is unreliable: it can
  // land before navigator.xr is truly usable and permanently wedge the UI with no retry path.
  // "IWER DevUI initialized with XR device." isn't a safe wait signal either: it logs BEFORE
  // installRuntime() (not after), and it's skipped entirely on the supported no-DevUI path,
  // which would falsely fail this test even against a perfectly usable app. Wait for
  // LoadIWER.ts's own "IWER runtime installed." instead - unconditional, and only logged once
  // installRuntime() has actually succeeded.
  await expect
    .poll(() => consoleLines.some(l => l.includes('IWER runtime installed.')), {
      timeout: 15000,
      message: () => `IWER never finished installing; console so far:\n${consoleLines.join('\n')}`,
    })
    .toBe(true);

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
