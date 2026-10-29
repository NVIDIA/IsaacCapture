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
 */

import type { XRDevice } from 'iwer';

declare global {
  interface Window {
    xrDevice?: XRDevice | null;
  }
}

let pendingIWERLoad: Promise<IWERLoadResult> | null = null;

export interface IWERLoadResult {
  supportsImmersive: boolean;
  iwerLoaded: boolean;
}

export async function loadIWERIfNeeded(): Promise<IWERLoadResult> {
  if (pendingIWERLoad) {
    return pendingIWERLoad;
  }
  pendingIWERLoad = loadIWER();
  const result = await pendingIWERLoad;
  if (!result.supportsImmersive) {
    // Clear so a later call can retry after a transient failure.
    pendingIWERLoad = null;
  }
  return result;
}

async function loadIWER(): Promise<IWERLoadResult> {
  let supportsImmersive = false;
  let iwerLoaded = false;

  if ('xr' in navigator) {
    try {
      const vr = await (navigator.xr as XRSystem).isSessionSupported?.('immersive-vr');
      const ar = await (navigator.xr as XRSystem).isSessionSupported?.('immersive-ar');
      supportsImmersive = Boolean(vr || ar);
    } catch (_) {}
  }

  if (!supportsImmersive) {
    console.info('Immersive mode not supported, loading IWER as fallback.');

    try {
      // Share the client's pointer-events module. A standalone DevUI bundle has
      // its own pointer ID counter but shares the global capture map, so mouse
      // captures can collide with XR controller pointers.
      const [{ XRDevice, metaQuest3 }, { DevUI }] = await Promise.all([
        import(/* webpackMode: "lazy" */ 'iwer'),
        import(/* webpackMode: "lazy" */ '@iwer/devui'),
      ]);
      const device = new XRDevice(metaQuest3);
      device.installDevUI(DevUI);

      // The native XRSystem has already reported no immersive support.
      await device.installRuntime({ forceInstall: true });
      window.xrDevice = device;
      supportsImmersive = true;
      iwerLoaded = true;
      console.info('IWER runtime installed.');
    } catch (error) {
      console.warn('IWER runtime install failed:', error);
    }
  }

  return { supportsImmersive, iwerLoaded };
}
