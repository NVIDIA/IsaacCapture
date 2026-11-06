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

jest.mock('iwer', () => ({ XRDevice: jest.fn(), metaQuest3: {} }));
jest.mock('@iwer/devui', () => ({ DevUI: jest.fn() }));

describe('loadIWERIfNeeded', () => {
  let originalXRDescriptor: PropertyDescriptor | undefined;
  let loadIWERIfNeeded: () => Promise<IWERLoadResult>;
  let XRDevice: jest.Mock;
  let DevUI: jest.Mock;
  let device: { installDevUI: jest.Mock; installRuntime: jest.Mock };

  beforeEach(() => {
    originalXRDescriptor = Object.getOwnPropertyDescriptor(navigator, 'xr');
    jest.resetModules();
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    ({ loadIWERIfNeeded } = require('./LoadIWER'));
    ({ XRDevice } = jest.requireMock('iwer'));
    ({ DevUI } = jest.requireMock('@iwer/devui'));
    device = { installDevUI: jest.fn(), installRuntime: jest.fn() };
    XRDevice.mockReturnValue(device);
  });

  afterEach(() => {
    if (originalXRDescriptor) {
      Object.defineProperty(navigator, 'xr', originalXRDescriptor);
    } else {
      delete (navigator as Navigator & { xr?: XRSystem }).xr;
    }
    delete window.xrDevice;
    jest.restoreAllMocks();
  });

  it('uses native immersive support without installing the emulator', async () => {
    Object.defineProperty(navigator, 'xr', {
      configurable: true,
      value: { isSessionSupported: jest.fn().mockResolvedValue(true) },
    });

    await expect(loadIWERIfNeeded()).resolves.toEqual({
      supportsImmersive: true,
      iwerLoaded: false,
    });
    expect(XRDevice).not.toHaveBeenCalled();
  });

  it('installs bundled DevUI over a native XRSystem without immersive support', async () => {
    const isSessionSupported = jest.fn().mockResolvedValue(false);
    Object.defineProperty(navigator, 'xr', {
      configurable: true,
      value: { isSessionSupported },
    });
    const appendChild = jest.spyOn(document.head, 'appendChild');

    await expect(loadIWERIfNeeded()).resolves.toEqual({
      supportsImmersive: true,
      iwerLoaded: true,
    });
    expect(isSessionSupported).toHaveBeenNthCalledWith(1, 'immersive-vr');
    expect(isSessionSupported).toHaveBeenNthCalledWith(2, 'immersive-ar');
    expect(device.installDevUI).toHaveBeenCalledWith(DevUI);
    expect(device.installRuntime).toHaveBeenCalledWith({ forceInstall: true });
    expect(window.xrDevice).toBe(device);
    expect(appendChild).not.toHaveBeenCalled();
  });

  it('shares one installation across concurrent callers', async () => {
    const [first, second] = await Promise.all([loadIWERIfNeeded(), loadIWERIfNeeded()]);
    expect(first).toEqual({ supportsImmersive: true, iwerLoaded: true });
    expect(second).toEqual(first);
    expect(XRDevice).toHaveBeenCalledTimes(1);
    expect(device.installRuntime).toHaveBeenCalledTimes(1);
  });

  it('retries a failed runtime installation', async () => {
    device.installRuntime.mockRejectedValueOnce(new Error('transient failure'));
    await expect(loadIWERIfNeeded()).resolves.toEqual({
      supportsImmersive: false,
      iwerLoaded: false,
    });
    expect(window.xrDevice).toBeUndefined();
    await expect(loadIWERIfNeeded()).resolves.toEqual({
      supportsImmersive: true,
      iwerLoaded: true,
    });
    expect(device.installRuntime).toHaveBeenCalledTimes(2);
  });
});
