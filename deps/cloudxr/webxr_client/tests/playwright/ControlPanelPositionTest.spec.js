/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// @ts-check
const { test, expect } = require('@playwright/test');

/**
 * Coverage for the in-XR control panel's head-relative positioning settings (branch
 * gmorgan/reset-panel-key, not yet merged as of this writing - these tests are expected to fail
 * against the plain App.tsx/CloudXRUI.tsx on this branch until that work lands): the
 * controlPanelPosition/Distance/Height/AngleDegrees settings, the R-key/CDP reset, and the
 * "Track Headset" mode. All assertions go through the "[CloudXRUI] headset=... relative=...
 * world=..." console.debug line - the only host-observable signal for where the panel actually
 * ended up, since there is no other way to read a Three.js object's world position from outside
 * the page.
 */

/** Parses the last "[CloudXRUI] headset=(x, y, z) relative=(x, y, z) world=(x, y, z)" line. */
function parseLastPanelPoseLog(consoleLines) {
  const poseLines = consoleLines.filter(l => l.startsWith('[CloudXRUI] headset='));
  if (poseLines.length === 0) return null;
  const last = poseLines[poseLines.length - 1];
  const m = last.match(
    /headset=\(([-\d.]+), ([-\d.]+), ([-\d.]+)\) relative=\(([-\d.]+), ([-\d.]+), ([-\d.]+)\) world=\(([-\d.]+), ([-\d.]+), ([-\d.]+)\)/
  );
  if (!m) return null;
  const n = m.slice(1).map(Number);
  return {
    headset: { x: n[0], y: n[1], z: n[2] },
    relative: { x: n[3], y: n[4], z: n[5] },
    world: { x: n[6], y: n[7], z: n[8] },
    count: poseLines.length,
  };
}

async function waitForConsoleText(lines, text, timeoutMs = 15000) {
  await expect
    .poll(() => lines.some(l => l.includes(text)), {
      timeout: timeoutMs,
      message: () => `never saw "${text}"; console so far:\n${lines.join('\n')}`,
    })
    .toBe(true);
}

/** Sets a form element's value/checked and fires input+change - the settings panel's fields
 * live inside a collapsed <details> group, so a normal Playwright .fill()/.check() (which
 * requires visibility) doesn't apply here without first expanding it. */
async function setFormValue(page, id, value) {
  await page.evaluate(
    ({ id, value }) => {
      const el = document.getElementById(id);
      if (!el) throw new Error(`no element with id ${id}`);
      if (el instanceof HTMLInputElement && el.type === 'checkbox') {
        el.checked = Boolean(value);
      } else {
        el.value = String(value);
      }
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.dispatchEvent(new Event('change', { bubbles: true }));
    },
    { id, value }
  );
}

/** Moves the IWER-emulated headset by a world-space delta. window.xrDevice is IWER's own
 * global (see LoadIWER.ts) - the only way to move the emulated headset from outside the page. */
async function moveHeadsetBy(page, dx, dy, dz) {
  await page.evaluate(
    ({ dx, dy, dz }) => {
      const device = window.xrDevice;
      if (!device) throw new Error('window.xrDevice not available - IWER not loaded?');
      device.position.set(device.position.x + dx, device.position.y + dy, device.position.z + dz);
    },
    { dx, dy, dz }
  );
}

/** Camera yaw is 0 in this mock environment (headset faces -Z, untouched), so
 * worldPositionFromHeadOffset's yaw rotation is a no-op: world.x/z = headset.x/z + relative.x/z,
 * and world.y is the absolute configured height (not headset-relative). See CloudXRUI.tsx. */
function expectWorldMatchesHeadsetPlusRelative(pose) {
  expect(pose.world.x).toBeCloseTo(pose.headset.x + pose.relative.x, 1);
  expect(pose.world.z).toBeCloseTo(pose.headset.z + pose.relative.z, 1);
  expect(pose.world.y).toBeCloseTo(pose.relative.y, 1);
}

async function connectAndCapture(
  page,
  { position, distance, height, angleDegrees, trackHeadset } = {}
) {
  const consoleLines = [];
  page.on('console', msg => consoleLines.push(msg.text()));

  await page.goto('http://localhost:8082/');
  // See AppMockTest.spec.js: "IWER DevUI initialized with XR device." logs before
  // installRuntime() and is skipped on the supported no-DevUI path - "IWER runtime installed."
  // is the reliable, unconditional signal that navigator.xr is actually usable.
  await waitForConsoleText(consoleLines, 'IWER runtime installed.');

  if (position !== undefined) await setFormValue(page, 'controlPanelPosition', position);
  if (distance !== undefined) await setFormValue(page, 'controlPanelDistance', distance);
  if (height !== undefined) await setFormValue(page, 'controlPanelHeight', height);
  if (angleDegrees !== undefined)
    await setFormValue(page, 'controlPanelAngleDegrees', angleDegrees);
  if (trackHeadset !== undefined)
    await setFormValue(page, 'controlPanelTrackHeadset', trackHeadset);

  await page.click('#startButton', { timeout: 15000 });
  await waitForConsoleText(consoleLines, 'CloudXR stream started');

  await expect
    .poll(() => consoleLines.some(l => l.startsWith('[CloudXRUI] headset=')), {
      timeout: 15000,
      message: () => `never saw a panel pose log; console so far:\n${consoleLines.join('\n')}`,
    })
    .toBe(true);

  return { page, consoleLines };
}

test.describe('control panel positioning settings', () => {
  test('default position is offset to the right, not dead center', async ({ page }) => {
    test.setTimeout(30000);
    const { consoleLines } = await connectAndCapture(page);
    const pose = parseLastPanelPoseLog(consoleLines);
    // distance=1.8, angle=70deg, right: x = 1.8*sin(70deg), z = -1.8*cos(70deg)
    expect(pose.relative.x).toBeCloseTo(1.69, 1);
    expect(pose.relative.z).toBeCloseTo(-0.62, 1);
    expect(pose.relative.y).toBeCloseTo(1.85, 1);
    expectWorldMatchesHeadsetPlusRelative(pose);
  });

  test('center position has no lateral offset', async ({ page }) => {
    test.setTimeout(30000);
    const { consoleLines } = await connectAndCapture(page, { position: 'center' });
    const pose = parseLastPanelPoseLog(consoleLines);
    expect(pose.relative.x).toBeCloseTo(0, 1);
    expect(pose.relative.z).toBeCloseTo(-1.8, 1);
  });

  test('left position mirrors right', async ({ page }) => {
    test.setTimeout(30000);
    const { consoleLines } = await connectAndCapture(page, { position: 'left' });
    const pose = parseLastPanelPoseLog(consoleLines);
    expect(pose.relative.x).toBeCloseTo(-1.69, 1);
    expect(pose.relative.z).toBeCloseTo(-0.62, 1);
  });

  test('distance/height/angle overrides feed the resulting relative offset', async ({ page }) => {
    test.setTimeout(30000);
    const { consoleLines } = await connectAndCapture(page, {
      position: 'right',
      distance: 2.5,
      height: 1.5,
      angleDegrees: 45,
    });
    const pose = parseLastPanelPoseLog(consoleLines);
    // distance=2.5, angle=45deg: x = 2.5*sin(45deg) = z = -2.5*cos(45deg) (symmetric at 45deg)
    expect(pose.relative.x).toBeCloseTo(1.77, 1);
    expect(pose.relative.z).toBeCloseTo(-1.77, 1);
    expect(pose.relative.y).toBeCloseTo(1.5, 1);
    expectWorldMatchesHeadsetPlusRelative(pose);
  });

  test('the reset key (R) restores the configured pose after the headset moves', async ({
    page,
  }) => {
    test.setTimeout(30000);
    const { consoleLines } = await connectAndCapture(page);
    const before = parseLastPanelPoseLog(consoleLines);
    expect(before.count).toBe(1);

    // Move the headset first - a reset that merely re-logged the stale pose from connect,
    // instead of recomputing relative to the *current* head position, would be caught here.
    await moveHeadsetBy(page, 2, 0, -1.5);

    await page.click('body');
    await page.keyboard.press('r');

    await expect
      .poll(() => parseLastPanelPoseLog(consoleLines).count, {
        timeout: 5000,
        message: () =>
          `R key never produced a second pose log; console so far:\n${consoleLines.join('\n')}`,
      })
      .toBe(2);

    const after = parseLastPanelPoseLog(consoleLines);
    // The configured offset is unchanged by reset, but the headset moved, so the reset pose must
    // be re-anchored to the new head position, not a replay of the one logged on connect.
    expect(after.relative.x).toBeCloseTo(before.relative.x, 2);
    expect(after.relative.z).toBeCloseTo(before.relative.z, 2);
    expect(after.headset.x).toBeCloseTo(before.headset.x + 2, 1);
    expect(after.headset.z).toBeCloseTo(before.headset.z - 1.5, 1);
    expectWorldMatchesHeadsetPlusRelative(after);
  });

  test('typing "r" into a settings text input does not trigger a reset', async ({ page }) => {
    test.setTimeout(30000);
    const { consoleLines } = await connectAndCapture(page);
    expect(parseLastPanelPoseLog(consoleLines).count).toBe(1);

    await page.fill('#serverIpInput', 'r');
    await page.waitForTimeout(500);

    expect(parseLastPanelPoseLog(consoleLines).count).toBe(1);
  });

  test('Track Headset mode logs exactly once on connect, not every frame', async ({ page }) => {
    test.setTimeout(30000);
    const { consoleLines } = await connectAndCapture(page, { trackHeadset: true });
    const countAfterConnect = parseLastPanelPoseLog(consoleLines).count;
    expect(countAfterConnect).toBe(1);

    // Tracking recomputes position every frame under the hood, but shouldn't log every frame -
    // only a real reposition beyond CloudXRUI's TRACKING_LOG_EPSILON_M does (see below).
    await page.waitForTimeout(3000);
    expect(parseLastPanelPoseLog(consoleLines).count).toBe(countAfterConnect);
  });

  test('Track Headset mode follows the headset when it moves', async ({ page }) => {
    test.setTimeout(30000);
    const { consoleLines } = await connectAndCapture(page, { trackHeadset: true });
    const before = parseLastPanelPoseLog(consoleLines);
    expect(before.count).toBe(1);

    await moveHeadsetBy(page, 1.5, 0, -1);

    await expect
      .poll(() => parseLastPanelPoseLog(consoleLines).count, {
        timeout: 5000,
        message: () =>
          `headset move never produced a new pose log; console so far:\n${consoleLines.join('\n')}`,
      })
      .toBeGreaterThan(before.count);

    const after = parseLastPanelPoseLog(consoleLines);
    // The configured offset is unchanged - tracking just re-anchors it to the new head pose,
    // same relationship a reset keeps (world = headset + relative).
    expect(after.relative.x).toBeCloseTo(before.relative.x, 2);
    expect(after.relative.z).toBeCloseTo(before.relative.z, 2);
    expect(after.headset.x).toBeCloseTo(before.headset.x + 1.5, 1);
    expect(after.headset.z).toBeCloseTo(before.headset.z - 1, 1);
    expectWorldMatchesHeadsetPlusRelative(after);
  });

  test('panel does not follow the headset when Track Headset is off', async ({ page }) => {
    test.setTimeout(30000);
    const { consoleLines } = await connectAndCapture(page);
    const before = parseLastPanelPoseLog(consoleLines);
    expect(before.count).toBe(1);

    await moveHeadsetBy(page, 1.5, 0, -1);
    await page.waitForTimeout(1000);

    // Without Track Headset, nothing recomputes the panel's position on a per-frame basis, so
    // a headset move alone produces no new pose log.
    expect(parseLastPanelPoseLog(consoleLines).count).toBe(before.count);
  });
});
