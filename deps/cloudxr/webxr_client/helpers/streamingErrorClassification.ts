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

/**
 * Server-disconnect codes (0xC0F223xx, per the CloudXR SDK's error-code convention) mean the far
 * end actively ended the session - the closest CloudXR equivalent to WebSocket close code 1008,
 * which HeadsetControlChannel's own reconnect (controlChannel.ts) treats as terminal. Not a
 * guaranteed-permanent rejection (the runtime restarting also lands here), but the best available
 * signal that blindly retrying is more likely to fail the same way again.
 */
const SERVER_DISCONNECT_CODE_MIN = 0xc0f22300;
const SERVER_DISCONNECT_CODE_MAX = 0xc0f223ff;

/**
 * Whether a stream error is worth a bounded reconnect attempt. Codes with no documented meaning
 * (undefined, or outside the server-disconnect range) default to recoverable: there is no
 * positive signal that retrying is futile, and CloudXR has no dedicated auth/policy-rejection
 * code today (unlike WebSocket's 1008) to fail closed on instead.
 */
export function isRecoverable(error: CloudXR.StreamingError | undefined): boolean {
  if (!error || error.code === undefined) {
    return true;
  }
  return !(error.code >= SERVER_DISCONNECT_CODE_MIN && error.code <= SERVER_DISCONNECT_CODE_MAX);
}
