/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @jest-environment jsdom
 */

import type { IWERLoadResult } from './LoadIWER';

type IWERTestWindow = Window &
  typeof globalThis & {
    IWER?: {
      XRDevice: jest.Mock;
      metaQuest3: object;
    };
    IWER_DevUI?: {
      DevUI: object;
    };
  };

describe('loadIWERIfNeeded', () => {
  const testWindow = window as IWERTestWindow;
  let originalXRDescriptor: PropertyDescriptor | undefined;
  let loadIWERIfNeeded: () => Promise<IWERLoadResult>;

  beforeEach(() => {
    originalXRDescriptor = Object.getOwnPropertyDescriptor(navigator, 'xr');
    jest.resetModules();
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    ({ loadIWERIfNeeded } = require('./LoadIWER'));
  });

  afterEach(() => {
    if (originalXRDescriptor) {
      Object.defineProperty(navigator, 'xr', originalXRDescriptor);
    } else {
      delete (navigator as Navigator & { xr?: XRSystem }).xr;
    }
    delete testWindow.xrDevice;
    delete testWindow.IWER;
    delete testWindow.IWER_DevUI;
    jest.restoreAllMocks();
  });

  it('returns native immersive support without loading IWER', async () => {
    const isSessionSupported = jest.fn().mockResolvedValue(true);
    Object.defineProperty(navigator, 'xr', {
      configurable: true,
      value: { isSessionSupported },
    });

    const appendChild = jest.spyOn(document.head, 'appendChild');

    await expect(loadIWERIfNeeded()).resolves.toEqual({
      supportsImmersive: true,
      iwerLoaded: false,
    });

    expect(appendChild).not.toHaveBeenCalled();
  });

  it('forces the fallback runtime over a native XRSystem without immersive support', async () => {
    const isSessionSupported = jest.fn().mockResolvedValue(false);
    Object.defineProperty(navigator, 'xr', {
      configurable: true,
      value: { isSessionSupported },
    });

    const device = {
      installDevUI: jest.fn(),
      installRuntime: jest.fn(() => {
        isSessionSupported.mockResolvedValue(true);
      }),
    };
    testWindow.IWER = {
      XRDevice: jest.fn(() => device),
      metaQuest3: {},
    };
    testWindow.IWER_DevUI = { DevUI: {} };

    const appendChild = jest
      .spyOn(document.head, 'appendChild')
      .mockImplementation(<T extends Node>(node: T): T => {
        if (node instanceof HTMLScriptElement) {
          queueMicrotask(() => node.onload?.(new Event('load')));
        }
        return node;
      });

    await expect(loadIWERIfNeeded()).resolves.toEqual({
      supportsImmersive: true,
      iwerLoaded: true,
    });

    expect(isSessionSupported).toHaveBeenNthCalledWith(1, 'immersive-vr');
    expect(isSessionSupported).toHaveBeenNthCalledWith(2, 'immersive-ar');
    expect(device.installRuntime).toHaveBeenCalledWith({ forceInstall: true });
    expect(appendChild).toHaveBeenCalledTimes(2);
  });

  it('rejects a skipped runtime installation and allows a later retry', async () => {
    const nativeXR = { isSessionSupported: jest.fn().mockResolvedValue(false) };
    Object.defineProperty(navigator, 'xr', { configurable: true, value: nativeXR });
    const device = { installRuntime: jest.fn() };
    testWindow.IWER = { XRDevice: jest.fn(() => device), metaQuest3: {} };
    jest.spyOn(document.head, 'appendChild').mockImplementation(<T extends Node>(node: T): T => {
      if (node instanceof HTMLScriptElement) {
        queueMicrotask(() => node.onload?.(new Event('load')));
      }
      return node;
    });

    await expect(loadIWERIfNeeded()).resolves.toEqual({
      supportsImmersive: false,
      iwerLoaded: false,
    });
    expect(window.xrDevice).toBeUndefined();
    expect(navigator.xr).toBe(nativeXR);

    device.installRuntime.mockImplementation(() => {
      Object.defineProperty(navigator, 'xr', {
        configurable: true,
        value: { isSessionSupported: jest.fn().mockResolvedValue(true) },
      });
    });
    await expect(loadIWERIfNeeded()).resolves.toEqual({
      supportsImmersive: true,
      iwerLoaded: true,
    });
    expect(window.xrDevice).toBe(device);
  });

  it('shares one IWER installation across concurrent callers', async () => {
    let appendedScripts = 0;
    let constructedDevices = 0;
    let runtimeInstalls = 0;

    const device = {
      installDevUI: jest.fn(),
      installRuntime: jest.fn(() => {
        runtimeInstalls++;
        Object.defineProperty(navigator, 'xr', {
          configurable: true,
          value: { isSessionSupported: jest.fn().mockResolvedValue(true) },
        });
      }),
    };
    testWindow.IWER = {
      XRDevice: jest.fn(() => {
        constructedDevices++;
        return device;
      }),
      metaQuest3: {},
    };
    testWindow.IWER_DevUI = { DevUI: {} };

    jest.spyOn(document.head, 'appendChild').mockImplementation(<T extends Node>(node: T): T => {
      appendedScripts++;
      if (node instanceof HTMLScriptElement) {
        queueMicrotask(() => node.onload?.(new Event('load')));
      }
      return node;
    });

    const [first, second] = await Promise.all([loadIWERIfNeeded(), loadIWERIfNeeded()]);

    expect(first).toEqual({ supportsImmersive: true, iwerLoaded: true });
    expect(second).toEqual(first);
    expect(appendedScripts).toBe(2);
    expect(constructedDevices).toBe(1);
    expect(runtimeInstalls).toBe(1);
  });
});
