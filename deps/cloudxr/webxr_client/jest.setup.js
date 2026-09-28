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
 * Jest setup shared by every test environment.
 *
 * jsdom implements `performance.now()` but not the User Timing entries, which the CloudXR
 * SDK bundle calls during module initialization. Suites that import the SDK therefore need
 * these stubs to get as far as their first assertion. No-op under the default `node`
 * environment, where the real implementations already exist.
 */
if (typeof performance !== 'undefined' && typeof performance.mark !== 'function') {
  performance.mark = () => {};
  performance.measure = () => {};
  performance.clearMarks = () => {};
  performance.clearMeasures = () => {};
  performance.getEntriesByName = () => [];
  performance.getEntriesByType = () => [];
}

/**
 * `WebSocket` is a native global on Node 22+ but not on older Node (e.g. Node 20, which
 * CI's web-client build job currently pins). Suites that talk to a real WebSocket server
 * (controlChannel.test.ts) need a real client either way, so fill the gap with `ws` rather
 * than only working on whichever Node happens to be running locally. No-op wherever the
 * native global already exists.
 */
if (typeof WebSocket === 'undefined') {
  global.WebSocket = require('ws');
}
