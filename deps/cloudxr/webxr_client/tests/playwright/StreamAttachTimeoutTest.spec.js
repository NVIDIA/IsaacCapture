/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// @ts-check
const { test, expect } = require('@playwright/test');

/**
 * Drives tests/mock/StreamAttachTimeoutTest.tsx's real CloudXRComponent + MockCloudXR 3-attempt
 * sequence (streamAttachTimeoutMs and warmupTimeoutMs, in isolation and combination - see that
 * file's header for the full scenario) in a real browser, and asserts on its console output
 * (every line the page logs is prefixed "[StreamAttachTimeoutTest] " via appendLog()).
 */

const LOG_PREFIX = '[StreamAttachTimeoutTest] ';
const SEQUENCE_TIMEOUT_MS = 15000;

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

test('StreamAttachTimeoutTest: attach timeout, warmup timeout, then success', async ({ page }) => {
  test.setTimeout(SEQUENCE_TIMEOUT_MS + 20000);

  const lines = collectLogLines(page);
  await page.goto('/StreamAttachTimeoutTest.html');
  await page.click('#startButton');

  await expect
    .poll(() => lines.some(l => l.startsWith('[result]')), {
      timeout: SEQUENCE_TIMEOUT_MS,
      message: () => `sequence did not complete; captured so far:\n${lines.join('\n')}`,
    })
    .toBe(true);

  const resultLine = lines.find(l => l.startsWith('[result]'));
  expect(resultLine, `captured lines:\n${lines.join('\n')}`).toContain('[result] PASS');
});
