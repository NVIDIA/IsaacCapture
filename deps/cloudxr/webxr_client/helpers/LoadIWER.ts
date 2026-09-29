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

// Both versions are loaded from CDN; bump them together when upgrading IWER.
const IWER_version = '2.3.0';
const IWER_DEVUI_version = '2.3.0';
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

    // Load IWER first
    const script = document.createElement('script');
    script.src = `https://unpkg.com/iwer@${IWER_version}/build/iwer.min.js`;
    script.async = true;
    script.integrity = 'sha384-m8Xcl9WwdP6j/5Fv7MAs7IvvW3PmWCbNZlbF8IsJAnujyeMZi0/xgUE4mM7Tbm/j';
    script.crossOrigin = 'anonymous';

    await new Promise<void>(resolve => {
      script.onload = async () => {
        console.info('IWER loaded as fallback.');
        const IWERGlobal = (window as any).IWER || (globalThis as any).IWER;
        if (!IWERGlobal) {
          console.warn('IWER global not found after script load.');
          supportsImmersive = false;
          resolve();
          return;
        }

        // Load iwer-devui after IWER
        const devUIScript = document.createElement('script');
        devUIScript.src = `https://unpkg.com/@iwer/devui@${IWER_DEVUI_version}/build/iwer-devui.min.js`;
        devUIScript.async = true;
        devUIScript.integrity =
          'sha384-TAu5gWlv92O6gkVclV8HYctTEswyA/0eoS6ULHjjLCShUVts4kDlUg6iLEKmL1r8';
        devUIScript.crossOrigin = 'anonymous';

        await new Promise<void>(devUIResolve => {
          devUIScript.onload = () => {
            console.info('IWER DevUI loaded.');
            devUIResolve();
          };
          devUIScript.onerror = error => {
            console.warn('Failed to load IWER DevUI:', error);
            devUIResolve();
          };
          document.head.appendChild(devUIScript);
        });

        try {
          const device: XRDevice = new IWERGlobal.XRDevice(IWERGlobal.metaQuest3);

          const IWER_DevUI = (window as any).IWER_DevUI || (globalThis as any).IWER_DevUI;
          if (IWER_DevUI?.DevUI) {
            device.installDevUI(IWER_DevUI.DevUI);
            console.info('IWER DevUI initialized with XR device.');
          } else {
            console.warn('IWER DevUI not found after script load, continuing without DevUI.');
          }

          // This fallback runs only after the native XRSystem reports no immersive
          // support. IWER 2.3 otherwise preserves that unusable navigator.xr.
          await device.installRuntime({ forceInstall: true });
          window.xrDevice = device;
          supportsImmersive = true;
          iwerLoaded = true;
          // The only log that's both unconditional (fires whether or not DevUI loaded, unlike
          // "IWER DevUI initialized...") and guaranteed to follow installRuntime() actually
          // succeeding (unlike "IWER DevUI initialized...", which logs before this await) - the
          // one a caller (e.g. a test) should wait on to know navigator.xr is truly usable.
          console.info('IWER runtime installed.');
        } catch (e) {
          console.warn('IWER runtime install failed:', e);
          supportsImmersive = false;
        }
        resolve();
      };
      script.onerror = () => {
        console.warn('Failed to load IWER.');
        supportsImmersive = false;
        resolve();
      };
      document.head.appendChild(script);
    });
  }

  return { supportsImmersive, iwerLoaded };
}
