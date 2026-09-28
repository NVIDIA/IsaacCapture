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

import type * as CloudXR from '@nvidia/cloudxr';

import { isRecoverable } from './streamingErrorClassification';

function error(code?: number): CloudXR.StreamingError {
  return { name: 'StreamingError', message: 'mock', code } as CloudXR.StreamingError;
}

describe('isRecoverable', () => {
  test('undefined error is recoverable', () => {
    expect(isRecoverable(undefined)).toBe(true);
  });

  test('error with no code is recoverable', () => {
    expect(isRecoverable(error(undefined))).toBe(true);
  });

  test('code below the server-disconnect range is recoverable', () => {
    expect(isRecoverable(error(0xc0f22204))).toBe(true);
    expect(isRecoverable(error(0xc0f222ff))).toBe(true);
  });

  test('code above the server-disconnect range is recoverable', () => {
    expect(isRecoverable(error(0xc0f22400))).toBe(true);
  });

  test('code inside the server-disconnect range is not recoverable', () => {
    expect(isRecoverable(error(0xc0f22300))).toBe(false);
    expect(isRecoverable(error(0xc0f22350))).toBe(false);
    expect(isRecoverable(error(0xc0f223ff))).toBe(false);
  });
});
