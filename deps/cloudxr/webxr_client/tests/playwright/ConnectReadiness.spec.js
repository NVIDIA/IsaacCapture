/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// @ts-check
const { test, expect } = require('@playwright/test');

test('Connect stays disabled until the delayed emulator is ready', async ({ page }) => {
  let releaseEmulator;
  const emulatorGate = new Promise(resolve => {
    releaseEmulator = resolve;
  });
  let emulatorRequested = false;
  const consoleLines = [];
  page.on('console', msg => consoleLines.push(msg.text()));
  await page.route(/(?:emulator|iwer.*index_js).*\.js/, async route => {
    emulatorRequested = true;
    await emulatorGate;
    await route.continue();
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
    await expect(page.locator('#errorMessageText')).not.toContainText('Failed to start XR session');

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
