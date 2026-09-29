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
 * missing-panel lives in ControlPanelPositionTest.spec.js instead of here: it needs the actual
 * fix (head-relative reset/tracking, gmorgan/reset-panel-key), not just a reproduction of the
 * gap - a panelHiddenAtStart-only check doesn't touch the real failure (panel/handle out of
 * reach after the operator moves), so keeping it under this file's "state reproduction" framing
 * was misleading about what it covered.
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

  test('passthrough-only: a session that enters but never streams is now detected via streamAttachTimeoutMs', async ({
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

    // streamAttachTimeoutMs is a URL-configurable param (params.ts) read straight into
    // CloudXRComponent's prop of the same name (App.tsx) - overridden here so this test doesn't
    // have to wait out the real 2-minute default to observe the timeout firing.
    //
    // reconnectEnabled=false is explicit, not incidental: its checkbox (cloudxrReconnectEnabled)
    // can come in checked (index.html default, or persisted localStorage from an earlier test in
    // this same browser context), and if it is, the synthetic attach-timeout error goes through
    // 3 retries - each with a doubling attach budget - before finally giving up, which both
    // changes what this test is actually exercising and can outlast a 15s console-wait timeout.
    // This test wants the no-retry give-up path specifically.
    await page.goto('http://localhost:8082/?streamAttachTimeoutMs=500&reconnectEnabled=false');
    // See AppMockTest.spec.js: "IWER DevUI initialized with XR device." logs before
    // installRuntime() and is skipped on the supported no-DevUI path - "IWER runtime installed."
    // is the reliable, unconditional signal that navigator.xr is actually usable.
    await waitForConsoleText(consoleLines, 'IWER runtime installed.');
    await page.click('#startButton', { timeout: 15000 });

    // The session enters immersive-ar and MockCloudXR.connect() is called (proving this isn't
    // just "never tried"), but it never reaches Connected.
    await waitForConsoleText(consoleLines, 'CloudXR session connect initiated');
    await waitForConsoleText(consoleLines, 'Mock connecting to');

    // Reliability doc's characterization of the gap ("the app only tracks isXRMode true/false")
    // no longer holds: CloudXRComponent.tsx's armStreamAttachTimer now fires a synthetic,
    // recoverable error once streamAttachTimeoutMs elapses with no onStreamStarted. reconnect
    // isn't enabled here (no reconnectEnabled=true param), so App.tsx's onError -> showError path
    // is what surfaces it, the same as any other CloudXR error.
    //
    // Checked via console output, not the live #errorMessageBox class/text: that box is a single
    // shared slot (CloudXR2DUI.tsx's own showStatus() doc comment) that an unrelated capability/
    // performance "info" notice can legitimately overwrite shortly after our error renders - the
    // original version of this test hit exactly that race. showStatus() always mirrors its
    // message to console[type](message) too, which console output doesn't get overwritten.
    await waitForConsoleText(consoleLines, 'CloudXR stream did not attach within 500ms');
    await waitForConsoleText(
      consoleLines,
      'CloudXR session stopped: Stream did not attach within 500ms'
    );
  });
});
