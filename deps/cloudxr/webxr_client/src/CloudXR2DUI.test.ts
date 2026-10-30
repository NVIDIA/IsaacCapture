/** @jest-environment jsdom */

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { CloudXR2DUI } from './CloudXR2DUI';

describe('device profile refresh settings', () => {
  let ui: CloudXR2DUI;
  function loadUI() {
    document.documentElement.innerHTML = readFileSync(join(__dirname, 'index.html'), 'utf8');
    ui = new CloudXR2DUI();
    ui.initialize();
  }
  function change(id: string, value: string) {
    const select = document.getElementById(id) as HTMLSelectElement;
    select.value = value;
    select.dispatchEvent(new Event('change', { bubbles: true }));
  }

  beforeEach(() => {
    jest.spyOn(console, 'info').mockImplementation(() => {});
    loadUI();
  });
  afterEach(() => {
    ui.cleanup();
    localStorage.clear();
    jest.restoreAllMocks();
  });

  test.each(['any', 'clean'] as const)(
    'selecting Custom preserves %s refresh mode and restores it on reload',
    mode => {
      change('deviceProfile', 'pico4ultra');
      if (mode === 'clean') change('cloudxrAutoRefreshMode', mode);
      change('deviceProfile', 'custom');
      expect(ui.getConfiguration().autoRefreshMode).toBe(mode);
      ui.cleanup();
      loadUI();
      expect(ui.getConfiguration().autoRefreshMode).toBe(mode);
    }
  );

  test.each([
    ['quest2', 'never'],
    ['quest3', 'never'],
    ['pico4ultra', 'any'],
  ])('selecting named profile %s applies its %s default', (profile, mode) => {
    change('cloudxrAutoRefreshMode', mode === 'any' ? 'never' : 'any');
    change('deviceProfile', profile);
    expect(ui.getConfiguration().autoRefreshMode).toBe(mode);
  });
});
