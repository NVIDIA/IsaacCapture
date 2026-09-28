/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// @ts-check
const { test, expect } = require('@playwright/test');

/**
 * Client-side-only coverage for the real App.tsx (via webpack.app-mock.js, :8082) of the
 * "missing client UI state" failure modes that are detectable/reproducible from
 * App.tsx/CloudXR2DUI.tsx alone. Host-side states (stale-tab, certificate interstitial) live in
 * oob_teleop_adb.py's CDP orchestration and are covered by a separate Python test suite.
 */

/** Waits for a console message containing `text`, polling `lines` (already being appended to by a `page.on('console')` listener). */
async function waitForConsoleText(lines, text, timeoutMs = 15000) {
  await expect
    .poll(() => lines.some(l => l.includes(text)), {
      timeout: timeoutMs,
      message: () => `never saw "${text}"; console so far:\n${lines.join('\n')}`,
    })
    .toBe(true);
}

test.describe('client UI states', () => {
  test('browser-launched-but-client-not-loaded: IWER load failure is surfaced, not silent', async ({
    page,
  }) => {
    test.setTimeout(30000);

    // Block the IWER CDN script the real page fetches (helpers/LoadIWER.ts) to force the
    // script.onerror path - the same failure mode as a headset browser that launched but never
    // finished loading the client page (here: never finished loading its WebXR emulation dep).
    await page.route('**/iwer*.min.js', route => route.abort());

    const consoleLines = [];
    page.on('console', msg => consoleLines.push(msg.text()));

    await page.goto('http://localhost:8082/');
    await waitForConsoleText(consoleLines, 'Failed to load IWER.');

    // App.tsx:263-269 reacts to loadIWERIfNeeded() reporting !supportsImmersive by calling
    // showError(), which reliably reaches #errorMessageText (this is the "Detected at launch"
    // case from the reliability doc). #startButton's disabled state is deliberately NOT asserted
    // here: it races an unrelated bug in updateConnectButtonState() (CloudXR2DUI.tsx:961), which
    // unconditionally re-enables the button from resolution/grid validity alone whenever its
    // text is exactly 'CONNECT', with no awareness of capabilities/IWER state - so the button can
    // end up enabled even after a capability failure that IS otherwise correctly surfaced.
    await expect(page.locator('#errorMessageText')).toHaveText('Immersive mode not supported');
  });

  test('passthrough-only: a session that enters but never streams looks identical to a slow connect', async ({
    page,
  }) => {
    test.setTimeout(30000);

    // window.__mockCloudXRConnectDelayMs (tests/mock/cloudxr-mock-alias.ts) holds the mock
    // session in SessionState.Connecting indefinitely - set before any app code runs so it's in
    // place the moment CloudXRComponent creates the session.
    await page.addInitScript(() => {
      window.__mockCloudXRConnectDelayMs = 24 * 60 * 60 * 1000;
    });

    const consoleLines = [];
    page.on('console', msg => consoleLines.push(msg.text()));

    await page.goto('http://localhost:8082/');
    await waitForConsoleText(consoleLines, 'IWER DevUI initialized with XR device.');
    await page.click('#startButton', { timeout: 15000 });

    // The session enters immersive-ar and MockCloudXR.connect() is called (proving this isn't
    // just "never tried"), but it never reaches Connected.
    await waitForConsoleText(consoleLines, 'CloudXR session connect initiated');
    await waitForConsoleText(consoleLines, 'Mock connecting to');

    // Reliability doc's characterization of the gap: "the app only tracks isXRMode true/false" -
    // give it a few seconds and confirm neither a stream-started signal nor any error/warning
    // ever distinguishes this from a session that's connecting normally.
    await page.waitForTimeout(3000);
    expect(consoleLines.some(l => l.includes('CloudXR stream started'))).toBe(false);
    expect(consoleLines.some(l => l.includes('Mock stream started'))).toBe(false);
    // errorMessageBox may legitimately show an unrelated "info" notice (e.g. the HEVC capability
    // warning) - only the "error" type token (not the base "error-message-box" class) would
    // indicate the app itself noticed a problem.
    await expect(page.locator('#errorMessageBox')).not.toHaveClass(/(^|\s)error(\s|$)/);
  });

  test('missing-panel: panelHiddenAtStart is reflected in the panel-visibility log', async ({
    page,
  }) => {
    test.setTimeout(30000);

    const consoleLines = [];
    page.on('console', msg => consoleLines.push(msg.text()));

    // The in-headset panel (CloudXRUI.tsx) is a world-anchored WebXR scene object, not a DOM
    // element, so there was no host-observable signal for panelHidden either way (reliability
    // doc: "not detected today"). CloudXRUI.tsx now logs panel-visibility transitions on every
    // change - this only proves that signal exists and reflects panelHiddenAtStart correctly;
    // a real watchdog still needs a *recovery* path (this remains "not detected" in the sense
    // that nothing reacts if the panel becomes unreachable mid-session).
    await page.goto('http://localhost:8082/?panelHiddenAtStart=true');
    await waitForConsoleText(consoleLines, 'IWER DevUI initialized with XR device.');
    await page.click('#startButton', { timeout: 15000 });
    await waitForConsoleText(consoleLines, 'CloudXR stream started');
    // panelHidden starts false (logged "visible" at mount) and only picks up
    // panelHiddenAtStart once the XR session actually enters - the last logged transition is
    // therefore the one that matters here.
    await waitForConsoleText(consoleLines, '[CloudXRUI] panel visibility: hidden');
    const visibilityLogs = consoleLines.filter(l => l.includes('[CloudXRUI] panel visibility:'));
    expect(visibilityLogs.at(-1)).toContain('hidden');
  });
});
