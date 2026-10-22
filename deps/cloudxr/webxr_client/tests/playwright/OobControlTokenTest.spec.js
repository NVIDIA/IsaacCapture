/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// @ts-check
const { spawn } = require('child_process');
const path = require('path');
const { createInterface } = require('readline');
const { test, expect } = require('@playwright/test');

/**
 * The real App.tsx (app-mock build, :8082) delivering the OOB control token to a real
 * OOBControlHub (tests/python/core/cloudxr/oob_hub_test_server.py): token in the URL fragment ->
 * authenticated register -> token stripped from the URL -> reload -> authenticated register again.
 * The hub speaks plain ws:// on loopback while the app dials wss://, so the page's WebSocket is
 * bridged to the hub with page.routeWebSocket. Does not cover adb launch or a real CloudXR stream.
 */

const HUB_SCRIPT = path.resolve(
  __dirname,
  '../../../../../tests/python/core/cloudxr/oob_hub_test_server.py'
);
// Must match TEST_CONTROL_TOKEN in oob_hub_test_server.py.
const HUB_TOKEN = 'oob-hub-test-token';
const ROUTE = 'sim/isaaclab';
const HUB_QUERY = 'oobEnable=1&serverIP=127.0.0.1&port=48322';

/** Starts the hub; `connectedCount()` is how many snapshots so far showed a connected headset. */
async function startHub() {
  const proc = spawn('python3', [HUB_SCRIPT], { stdio: ['ignore', 'pipe', 'pipe'] });
  let connected = 0;
  const port = await new Promise((resolve, reject) => {
    proc.once('error', reject);
    proc.once('exit', code => reject(new Error(`hub exited early (${code})`)));
    createInterface({ input: proc.stdout }).on('line', line => {
      const ready = /^READY (\d+)$/.exec(line);
      if (ready) resolve(Number(ready[1]));
      if (
        line.startsWith('SNAPSHOT ') &&
        JSON.parse(line.slice(9)).headsets.some(h => h.connected)
      ) {
        connected += 1;
      }
    });
  });
  return { proc, port, connectedCount: () => connected };
}

/** Bridges the page's wss://…/oob/v1/ws to the plain-ws hub. */
async function bridgeHub(page, hubPort) {
  await page.routeWebSocket(/\/oob\/v1\/ws/, ws => {
    const upstream = new WebSocket(`ws://127.0.0.1:${hubPort}`);
    const queued = [];
    upstream.onopen = () => queued.splice(0).forEach(m => upstream.send(m));
    upstream.onmessage = event => ws.send(event.data);
    upstream.onclose = () => ws.close();
    ws.onMessage(message =>
      upstream.readyState === 1 ? upstream.send(message) : queued.push(message)
    );
    ws.onClose(() => upstream.close());
  });
}

let hub;
test.beforeEach(async () => {
  hub = await startHub();
});
test.afterEach(() => {
  hub?.proc.kill('SIGKILL');
});

test('fragment token authenticates with the hub, is stripped, and survives a reload', async ({
  page,
}) => {
  const requests = [];
  page.on('request', request =>
    requests.push(`${request.url()} ${request.headers().referer ?? ''}`)
  );
  await bridgeHub(page, hub.port);

  await page.goto(`http://localhost:8082/?${HUB_QUERY}#/${ROUTE}?controlToken=${HUB_TOKEN}`);
  await expect.poll(() => hub.connectedCount(), { timeout: 20000 }).toBe(1);

  const afterLoad = new URL(page.url());
  expect(afterLoad.hash).toBe(`#/${ROUTE}`);
  expect(afterLoad.search).toBe(`?${HUB_QUERY}`);

  await page.reload();
  await expect.poll(() => hub.connectedCount(), { timeout: 20000 }).toBe(2);
  expect(new URL(page.url()).hash).toBe(`#/${ROUTE}`);

  expect(requests.filter(r => r.includes(HUB_TOKEN))).toEqual([]);
});

test('a legacy query token authenticates and is stripped', async ({ page }) => {
  await bridgeHub(page, hub.port);
  await page.goto(`http://localhost:8082/?${HUB_QUERY}&controlToken=${HUB_TOKEN}#/${ROUTE}`);
  await expect.poll(() => hub.connectedCount(), { timeout: 20000 }).toBe(1);
  expect(page.url()).not.toContain('controlToken');
});

test('without a token the hub rejects the headset', async ({ page }) => {
  await bridgeHub(page, hub.port);
  await page.goto(`http://localhost:8082/?${HUB_QUERY}#/${ROUTE}`);
  await page.waitForTimeout(5000);
  expect(hub.connectedCount()).toBe(0);
});
