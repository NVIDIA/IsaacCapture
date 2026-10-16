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

/**
 * Drop-in replacement for `@nvidia/cloudxr`, aliased in via webpack.app-mock.js: re-exports
 * everything from the real package except `createSession`, which is redirected to
 * {@link createMockCloudXRSession}. This lets the *real* App.tsx/CloudXRComponent.tsx run
 * against MockCloudXR without any source change - same idea as cloudxr-js's
 * examples/webpack-nvidia-cloudxr-alias.cjs (OVERRIDE_CLOUDXR_FILENAME), one level up: that
 * repo aliases an internal SDK dependency; we don't control @nvidia/cloudxr's internals, so we
 * alias the whole package entry instead.
 *
 * The most-recently-created mock session is exposed on `window.__mockCloudXRFail`, mirroring
 * cloudxr-js's `window.__cloudxrMockFail` (see src/mocks/ragnarok-mock.ts there), so a
 * Playwright test can force a mid-stream failure from outside the page. `window.__mockCloudXRConnectDelayMs`,
 * read once per `createSession()` call (before the real App.tsx ever calls `connect()`), extends
 * that pattern to delay connection indefinitely, holding a session in `Connecting` to exercise
 * CloudXRComponent's streamAttachTimeoutMs deadline from the outside.
 *
 * Imports/re-exports the real SDK's concrete entry file (`@nvidia/cloudxr/build/cloudxr.js`),
 * not the bare `'@nvidia/cloudxr'` specifier: webpack.app-mock.js aliases that bare specifier to
 * this very file, so referencing it here would self-import circularly instead of reaching the
 * real package.
 */

import * as CloudXR from '@nvidia/cloudxr/build/cloudxr.js';

import { createMockCloudXRSession, MockCloudXR } from './MockCloudXR';

export * from '@nvidia/cloudxr/build/cloudxr.js';

let activeSession: MockCloudXR | null = null;

export function createSession(
  options: CloudXR.SessionOptions,
  delegates: CloudXR.SessionDelegates
): MockCloudXR {
  activeSession = createMockCloudXRSession(options, delegates);
  if (typeof window !== 'undefined' && window.__mockCloudXRConnectDelayMs !== undefined) {
    activeSession.connectWait(window.__mockCloudXRConnectDelayMs);
  }
  return activeSession;
}

declare global {
  interface Window {
    __mockCloudXRFail?: (message?: string) => void;
    __mockCloudXRConnectDelayMs?: number;
  }
}

if (typeof window !== 'undefined') {
  window.__mockCloudXRFail = (message = 'Mock transient server/tunnel outage') => {
    activeSession?.triggerFailure({ name: 'StreamingError', message });
  };
}
