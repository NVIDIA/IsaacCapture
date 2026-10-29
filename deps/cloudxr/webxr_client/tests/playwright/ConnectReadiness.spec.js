/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// @ts-check
const { test, expect } = require('@playwright/test');
const { EMULATOR_BUNDLE_PATTERN } = require('./emulator');

test('Connect stays disabled until the delayed emulator is ready', async ({ page }) => {
  let releaseEmulator;
  const emulatorGate = new Promise(resolve => {
    releaseEmulator = resolve;
  });
  let emulatorRequested = false;
  const consoleLines = [];
  page.on('console', msg => consoleLines.push(msg.text()));
  await page.route(EMULATOR_BUNDLE_PATTERN, async route => {
    emulatorRequested = true;
    await emulatorGate;
    await route.continue();
  });

  await page.addInitScript(() => {
    window.__earlyXRSessionRequests = 0;
    Object.defineProperty(navigator, 'xr', {
      configurable: true,
      value: Object.assign(new EventTarget(), {
        isSessionSupported: async () => false,
        requestSession: () => {
          window.__earlyXRSessionRequests++;
          return Promise.reject(new Error('XR requested before emulator readiness'));
        },
      }),
    });
  });

  try {
    await page.goto('http://localhost:8082/', { waitUntil: 'domcontentloaded' });
    await expect.poll(() => emulatorRequested).toBe(true);
    const connect = page.locator('#startButton');
    await expect(connect).toBeDisabled();
    await page.locator('#serverIpInput').fill('localhost');
    await expect(connect).toBeDisabled();
    expect(await page.evaluate(() => !!window.xrDevice)).toBe(false);

    // A trusted pointer click must not reach requestSession while startup is pending.
    const bounds = await connect.boundingBox();
    if (!bounds) throw new Error('Connect button is not visible');
    await page.mouse.click(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
    // Let click-triggered async work run while the emulator is still blocked.
    await page.evaluate(
      () => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))
    );
    expect(await page.evaluate(() => window.__earlyXRSessionRequests)).toBe(0);

    releaseEmulator();
    await connect.click();
    await expect
      .poll(() => consoleLines.some(line => line.includes('CloudXR stream started')), {
        timeout: 15000,
      })
      .toBe(true);
    await expect(page.locator('#errorMessageText')).not.toContainText('No XR hardware found');
  } finally {
    releaseEmulator();
  }
});
