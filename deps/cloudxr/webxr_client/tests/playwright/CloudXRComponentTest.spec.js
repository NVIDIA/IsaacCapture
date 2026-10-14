/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// @ts-check
const { test, expect } = require('@playwright/test');

/**
 * Drives tests/mock/CloudXRComponentTest.tsx's real CloudXRComponent + MockCloudXR scripted
 * 6-step sequence in a real browser, and asserts on its console output (every line the page logs
 * is prefixed "[CloudXRComponentTest] " via appendLog() - see that file) rather than the DOM log
 * panel, so this test exercises exactly what a real integrator's console would show. Steps 2-5
 * cover CloudXRComponent's bounded streaming-session retry (give-up on an unrecoverable error,
 * retry+reconnect+counter-reset on a recoverable one, exhausting maxAttempts, and cancelling a
 * pending retry on session end); step 6 covers streamTest; the trailing assertions cover the rest
 * of the CloudXRComponentProps callback surface.
 */

const LOG_PREFIX = '[CloudXRComponentTest] ';
/** Steps 3-5 alone are ~15s of retry/reconnect pacing; the full 6-step run is comfortably under this. */
const SEQUENCE_TIMEOUT_MS = 60000;

/** Collects every "[CloudXRComponentTest] "-prefixed console line, with the prefix stripped. */
function collectLogLines(page) {
  const lines = [];
  page.on('console', msg => {
    const text = msg.text();
    if (text.startsWith(LOG_PREFIX)) {
      lines.push(text.slice(LOG_PREFIX.length));
    }
  });
  return lines;
}

/** Lines strictly between two markers (exclusive), in collection order. */
function between(lines, startMarker, endMarker) {
  const start = lines.indexOf(startMarker);
  const end = endMarker ? lines.indexOf(endMarker, start + 1) : lines.length;
  if (start === -1) return [];
  return lines.slice(start + 1, end === -1 ? lines.length : end);
}

const STEP_MARKERS = [
  '=== Step 1: start, then close cleanly - expect no errors ===',
  '=== Step 2: fail while still connecting (unrecoverable) - expect no retry ===',
  '=== Step 3: fail once connected (recoverable) - expect automatic retry, then counter reset ===',
  '=== Step 4: exhaust retry attempts - expect give-up after 3 ===',
  '=== Step 5: end session while a retry is pending - expect the retry to be cancelled ===',
  '=== Step 6: stream test (remounts with streamTest enabled) ===',
  '=== Test sequence complete ===',
];

test('CloudXRComponentTest scripted sequence: every callback and retry path fires as expected', async ({
  page,
}) => {
  test.setTimeout(SEQUENCE_TIMEOUT_MS + 20000);

  const lines = collectLogLines(page);
  await page.goto('/CloudXRComponentTest.html');
  await page.click('#startButton');

  await expect
    .poll(() => lines.includes('=== Test sequence complete ==='), {
      timeout: SEQUENCE_TIMEOUT_MS,
      message: () => `sequence did not complete; captured so far:\n${lines.join('\n')}`,
    })
    .toBe(true);

  // No step's own internal PASS/FAIL assertion (see CloudXRComponentTest.tsx's runStepN
  // functions) reported FAIL - catches any regression across all six steps in one assertion.
  expect(lines.some(l => l.includes('FAIL'))).toBe(false);

  const [step1Body, step2Body, step3Body, step4Body, step5Body, step6Body] = STEP_MARKERS.slice(
    0,
    -1
  ).map((marker, i) => between(lines, marker, STEP_MARKERS[i + 1]));

  // --- Step 1: normal start/stop, no errors ---
  expect(step1Body).toContain('[step1] PASS: no errors');
  expect(step1Body.some(l => l.startsWith('[error]'))).toBe(false);

  // --- Step 2: unrecoverable failure while still connecting - gives up immediately, no retry ---
  expect(step2Body).toContain('[step2] PASS: no retry attempted');
  const step2Errors = step2Body.filter(l => l.startsWith('[error]'));
  expect(step2Errors).toHaveLength(1);
  expect(step2Errors[0]).toContain('Mock unrecoverable failure (server-disconnect range)');
  expect(step2Body.some(l => l.startsWith('[status]') && l.includes('Reconnecting'))).toBe(false);

  // --- Step 3: recoverable failure once connected - retries, reconnects, counter resets ---
  // onError is only called on the give-up path (see CloudXRComponent.tsx's onStreamStopped) -
  // a retry that's scheduled, not given up on, never calls it, so zero [error] lines here.
  expect(step3Body).toContain(
    '[step3] PASS: retried, reconnected, and the attempt counter reset for a later failure'
  );
  expect(step3Body.some(l => l.startsWith('[error]'))).toBe(false);
  // Both retries report attempt 1 - proves the counter actually reset between them.
  const step3ReconnectingStatuses = step3Body.filter(l => l.includes('Reconnecting ('));
  expect(step3ReconnectingStatuses).toEqual([
    '[status] connected=false Reconnecting (1/3)',
    '[status] connected=false Reconnecting (1/3)',
  ]);

  // --- Step 4: exhausts all 3 attempts (no onError for those), then gives up on the 4th ---
  expect(step4Body).toContain('[step4] PASS: retried 3 times then gave up');
  const step4Errors = step4Body.filter(l => l.startsWith('[error]'));
  expect(step4Errors).toHaveLength(1); // only the 4th, exhausted failure gives up
  expect(step4Errors[0]).toContain('Mock recoverable failure #4 (network interrupted)');
  expect(step4Body.filter(l => l.includes('Reconnecting ('))).toEqual([
    '[status] connected=false Reconnecting (1/3)',
    '[status] connected=false Reconnecting (2/3)',
    '[status] connected=false Reconnecting (3/3)',
  ]);
  expect(step4Body.filter(l => l === '[event] onExitImmersiveXR')).toHaveLength(1);

  // --- Step 5: a pending retry is cancelled by ending the session before its delay elapses ---
  // The one failure here schedules a retry (no onError) that then gets cancelled - zero
  // [error] lines and zero onExitImmersiveXR, since give-up never happens.
  expect(step5Body).toContain('[step5] PASS: pending retry was cancelled');
  expect(step5Body.some(l => l.startsWith('[error]'))).toBe(false);
  expect(step5Body.filter(l => l.includes('Reconnecting ('))).toHaveLength(1);
  expect(step5Body.some(l => l === '[event] onExitImmersiveXR')).toBe(false);
  // Cancelling a pending retry has no live session to disconnect(), so nothing else would
  // emit this - handleSessionEnd must emit it directly or the app is stuck seeing
  // "Reconnecting (n/maxAttempts)" forever.
  expect(step5Body).toContain('[status] connected=false Disconnected');

  // --- Step 6: stream test runs to completion ---
  const step6StartedIdx = step6Body.indexOf('[event] onStreamTestStarted');
  const step6StoppedIdx = step6Body.indexOf('[event] onStreamTestStopped passed=true');
  expect(step6StartedIdx).toBeGreaterThanOrEqual(0);
  expect(step6StoppedIdx).toBeGreaterThan(step6StartedIdx);

  // --- Full CloudXRComponentProps callback surface fired at least once across the whole run ---
  expect(lines).toContain('[event] onSessionReady session');
  expect(lines).toContain('[event] onSessionReady null');
  expect(lines.some(l => l.startsWith('[event] onServerAddress '))).toBe(true);
  expect(lines.some(l => l.startsWith('[event] onLog '))).toBe(true);
  expect(lines).toContain('[event] onRenderPerformanceMetrics (first occurrence)');
  expect(lines).toContain('[event] onStreamingPerformanceMetrics (first occurrence)');
  expect(lines).toContain('[event] onNetworkPerformanceMetrics (first occurrence)');
  expect(lines).toContain('[prop] trackingFrameAdapter called');
  expect(lines.some(l => l.startsWith('[status] connected='))).toBe(true);

  // --- No stray errors: onError only fires on the two give-up paths (step2's unrecoverable
  // failure, step4's exhausted retry) - never on a merely-scheduled retry (steps 3, 4's first
  // 3 attempts, step 5). ---
  const totalErrors = lines.filter(l => l.startsWith('[error]'));
  expect(totalErrors).toHaveLength(2); // step2 + step4's 4th attempt
});
