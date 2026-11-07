/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// @ts-check
const { test, expect } = require('@playwright/test');

test('video HUD shows connection progress, hides on stream start, and reopens on loss', async ({
  page,
}) => {
  const lines = [];
  page.on('console', msg => lines.push(msg.text()));
  const waitForLog = text =>
    expect.poll(() => lines.some(line => line.includes(text)), { timeout: 15000 }).toBe(true);
  const lastVisibility = () =>
    lines.filter(line => line.startsWith('[CloudXRUI] panel visibility:')).at(-1);

  await page.addInitScript(() => {
    window.__mockCloudXRConnectDelayMs = 3000;
    localStorage.setItem('cxr.isaac.teleopPath', 'sim/isaacsim');
  });
  // Even a saved/explicit hide-on-entry choice must not hide video connection progress.
  await page.goto(
    'http://localhost:8082/?panelHiddenAtStart=true&autoRefreshMode=never&reconnectEnabled=true&reconnectMaxAttempts=2&controlPanelPosition=center#/real/gear/video'
  );
  await waitForLog('IWER runtime installed.');
  await page.click('#startButton');
  await waitForLog('Mock connecting to');
  await expect.poll(lastVisibility).toBe('[CloudXRUI] panel visibility: visible');
  expect(lines.some(line => line === '[CloudXRUI] panel visibility: hidden')).toBe(false);

  await waitForLog('CloudXR stream started');
  await expect.poll(lastVisibility).toBe('[CloudXRUI] panel visibility: hidden');

  // The existing panel-reset shortcut reopens it; ongoing rendering must not collapse it again.
  await page.keyboard.press('r');
  await expect.poll(lastVisibility).toBe('[CloudXRUI] panel visibility: visible');
  await page.waitForTimeout(500);
  expect(lastVisibility()).toBe('[CloudXRUI] panel visibility: visible');

  await page.evaluate(() => window.__mockCloudXRFail('Mock video stream interrupted'));
  await waitForLog('Reconnecting');
  expect(lastVisibility()).toBe('[CloudXRUI] panel visibility: visible');
  await expect
    .poll(lastVisibility, { timeout: 15000 })
    .toBe('[CloudXRUI] panel visibility: hidden');

  // This time lose the stream while collapsed, proving the error/progress details reopen.
  lines.length = 0;
  await page.evaluate(() => window.__mockCloudXRFail('Mock video stream interrupted again'));
  await expect.poll(lastVisibility).toBe('[CloudXRUI] panel visibility: visible');
  await waitForLog('CloudXR stream started');
  await expect.poll(lastVisibility).toBe('[CloudXRUI] panel visibility: hidden');
});
