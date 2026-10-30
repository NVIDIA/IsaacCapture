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

describe('Connect capability readiness', () => {
  let ui: CloudXR2DUI;
  let button: HTMLButtonElement;

  beforeEach(() => {
    document.documentElement.innerHTML = readFileSync(join(__dirname, 'index.html'), 'utf8');
    ui = new CloudXR2DUI();
    ui.initialize();
    button = document.getElementById('startButton') as HTMLButtonElement;
  });

  afterEach(() => {
    ui.cleanup();
    localStorage.clear();
    jest.restoreAllMocks();
  });

  it('blocks form updates, session notifications and clicks until capabilities pass', async () => {
    const onConnect = jest.fn().mockResolvedValue(undefined);
    const onError = jest.fn();
    ui.setupConnectButtonHandler(onConnect, onError);
    expect(button.disabled).toBe(true);

    ui.setStartButtonState(false, 'CONNECT');
    const width = document.getElementById('perEyeWidth') as HTMLInputElement;
    width.dispatchEvent(new Event('blur', { bubbles: true }));
    expect(button.disabled).toBe(true);
    button.dispatchEvent(new MouseEvent('click'));
    expect(onConnect).not.toHaveBeenCalled();

    ui.setCapabilitiesReady(true);
    expect(button.disabled).toBe(false);
    button.click();
    await Promise.resolve();
    expect(onConnect).toHaveBeenCalledTimes(1);
    expect(onError).not.toHaveBeenCalled();
  });

  it('keeps failed capability checks disabled after returning to idle', () => {
    ui.setCapabilitiesReady(true);
    ui.setCapabilitiesReady(false);
    ui.setStartButtonState(false, 'CONNECT');
    ui.updateConnectButtonState();
    expect(button.disabled).toBe(true);
  });

  it('preserves resolution validation after capabilities pass', () => {
    const width = document.getElementById('perEyeWidth') as HTMLInputElement;
    width.value = '1';
    width.dispatchEvent(new Event('blur', { bubbles: true }));
    ui.setCapabilitiesReady(true);
    expect(button.disabled).toBe(true);
    width.value = '2048';
    width.dispatchEvent(new Event('blur', { bubbles: true }));
    expect(button.disabled).toBe(false);
  });
});
