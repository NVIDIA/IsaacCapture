/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// @ts-check
const { test, expect } = require('@playwright/test');

/**
 * Client-side-only coverage for the real App.tsx (via webpack.app-mock.js, :8082) of two
 * "missing client UI state" failure modes detectable from App.tsx/CloudXR2DUI.tsx alone:
 * browser-launched-but-client-not-loaded, and passthrough-only. Host-side states (stale-tab,
 * certificate interstitial) live in oob_teleop_adb.py's CDP orchestration, covered by a separate
 * Python test suite. missing-panel is deferred: it needs the actual fix (head-relative
 * reset/tracking, see PR #1140) before it can be tested meaningfully, not just a reproduction of
 * the gap - a panelHiddenAtStart-only check doesn't touch the real failure (panel/handle out of
 * reach after the operator moves). CloudXRUI.tsx's panel-visibility console logging (added here)
 * is unused by any test in this file yet - it's the signal that deferred coverage will consume.
 */

/** Waits for a console message containing `text`, polling `lines` (already being appended to by a `page.on('console')` listener). */
async function waitForConsoleText(lines, text, timeoutMs = 15000) {
  try {
    await expect.poll(() => lines.some(l => l.includes(text)), { timeout: timeoutMs }).toBe(true);
  } catch (error) {
    // expect.poll's `message` option is a string, not a callback (it would otherwise print the
    // function's source instead of evaluating it) - attach the console dump here instead, only
    // once there's actually a failure to explain.
    throw new Error(`never saw "${text}"; console so far:\n${lines.join('\n')}\n\n${error}`);
  }
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
    // Text content alone doesn't prove the operator can see it - showStatus() can set
    // #errorMessageText without the box being visible if it regresses. toBeVisible() respects
    // the actual .show CSS rule (display: none by default).
    await expect(page.locator('#errorMessageBox')).toBeVisible();
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
    // Waited for via console first, not the live #errorMessageBox: that box is a single shared
    // slot (CloudXR2DUI.tsx's showStatus() doc comment) that an unrelated capability/performance
    // "info" notice can legitimately overwrite shortly after. showStatus() sets the DOM and
    // mirrors to console[type](message) in the same synchronous call, so checking the DOM right
    // after this console wait resolves - before any later notice gets a chance to run - is safe.
    await waitForConsoleText(consoleLines, 'CloudXR stream did not attach within 500ms');
    await waitForConsoleText(
      consoleLines,
      'CloudXR session stopped: Stream did not attach within 500ms'
    );
    await expect(page.locator('#errorMessageBox')).toBeVisible();
    await expect(page.locator('#errorMessageText')).toHaveText(
      'CloudXR session stopped: Stream did not attach within 500ms'
    );
  });
});
