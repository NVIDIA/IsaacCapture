/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
const { test, expect } = require('@playwright/test');

// Real IWER and production UI; CloudXR transport is mocked in this browser-only suite.
test('synthetic input reaches real WebXR poses and buttons; stop releases input', async ({
  page,
}) => {
  await page.goto('http://localhost:8082/?syntheticXR=1&immersiveMode=vr');
  await page.waitForFunction(() => Boolean(window.syntheticXR));
  await page.click('#startButton');
  await page.waitForFunction(() => window.syntheticXR.lastFrame?.sources.length === 2);
  await page.getByRole('button', { name: 'Start teleoperation', exact: true }).click();
  await expect(
    page.getByRole('button', { name: 'Teleoperation started', exact: true })
  ).toBeDisabled();
  await page.getByLabel('X position').fill('0.4');
  await page.getByLabel('Grip / clutch').check();
  await page.evaluate(() => window.syntheticXR.setController('right', { trigger: 0.8 }));
  await expect
    .poll(() =>
      page.evaluate(() => {
        const right = window.syntheticXR.lastFrame.sources.find(s => s.hand === 'right');
        return {
          x: Math.round(right.aim.position[0] * 100),
          trigger: right.buttons[0],
          grip: right.buttons[1],
        };
      })
    )
    .toEqual({ x: 40, trigger: 0.8, grip: 1 });
  await page.getByRole('button', { name: 'Stop input', exact: true }).click();
  await expect
    .poll(() =>
      page.evaluate(() =>
        window.syntheticXR.lastFrame.sources.find(s => s.hand === 'right').buttons.slice(0, 2)
      )
    )
    .toEqual([0, 0]);
});

test('generated sweep completes once and an exported trace can be imported', async ({ page }) => {
  await page.goto('http://localhost:8082/?syntheticXR=1&immersiveMode=vr');
  await page.waitForFunction(() => Boolean(window.syntheticXR));
  await page.click('#startButton');
  await page.waitForFunction(() => window.syntheticXR.lastFrame?.sources.length === 2);
  // This long-running input test needs a decoded mock frame, or the real startup
  // watchdog correctly disconnects the transport and releases synthetic input.
  await page.evaluate(() => window.__mockCloudXRVideoFrame(1));
  const run = page.getByRole('button', { name: 'Run controller sweep' });
  await expect(run).toBeEnabled();
  await run.click();
  await expect
    .poll(() => page.evaluate(() => window.syntheticXR.status), { timeout: 15000 })
    .toContain('Playback complete');
  await page.getByRole('button', { name: 'Record input', exact: true }).click();
  await page.getByLabel('X position').fill('0.45');
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Save input trace' }).click();
  const download = await downloadPromise;
  const path = await download.path();
  await page.getByLabel('Load synthetic trace').setInputFiles(path);
  await page.getByRole('button', { name: 'Replay once', exact: true }).click();
  await expect
    .poll(() => page.evaluate(() => window.syntheticXR.status))
    .toContain('Playback complete');
});

test('ordinary desktop load does not install the synthetic control API', async ({ page }) => {
  await page.goto('http://localhost:8082/');
  await page.getByRole('button', { name: 'Open Synthetic XR controls' }).waitFor();
  expect(await page.evaluate(() => window.syntheticXR)).toBeUndefined();
});

test('native immersive capability never installs or overrides synthetic XR', async ({ page }) => {
  await page.addInitScript(() => {
    window.nativeXRMarker = Object.assign(new EventTarget(), {
      isSessionSupported: async () => true,
    });
    Object.defineProperty(navigator, 'xr', { value: window.nativeXRMarker, configurable: true });
  });
  await page.goto('http://localhost:8082/?syntheticXR=1');
  await expect(page.getByText('Native XR is active.', { exact: false })).toBeVisible();
  expect(
    await page.evaluate(
      () => navigator.xr === window.nativeXRMarker && !window.syntheticXR && !window.xrDevice
    )
  ).toBe(true);
});
