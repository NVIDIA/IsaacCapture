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
 * RecorderComponent.tsx - Null-rendering R3F component that drives the recorder each frame.
 *
 * Runs at priority -1001 so replay advances before CloudXR requests its scoped
 * tracking-frame adapter at priority -1000.
 */

import { useFrame } from '@react-three/fiber';
import { useRef } from 'react';

import { useRecorder } from './RecorderContext';

interface RecorderComponentProps {
  /** True when the CloudXR session is in the Connected state — gates replay frame advancement. */
  isConnected: boolean;
  /** Capture live input while idle so the trace can render it. */
  showTrace: boolean;
}

export function RecorderComponent({ isConnected, showTrace }: RecorderComponentProps) {
  const { recorder, onFrameRecord, onFrameState } = useRecorder();
  const tickRef = useRef(0);

  useFrame(state => {
    const xrFrame = state.gl.xr.getFrame() as XRFrame | null;
    if (!xrFrame) return;

    // Only advance replay/record when the session is visible (avoids poisoning recordings
    // with null poses on Quest sleep) AND CloudXR is Connected (avoids consuming replay
    // frames during Connecting state before the server is ready).
    const session = state.gl.xr.getSession() as XRSession | null;
    const isVisible = session?.visibilityState === 'visible';

    recorder.beginFrame(
      xrFrame,
      state.gl.xr.getReferenceSpace(),
      isConnected && isVisible,
      showTrace && isVisible
    );

    onFrameState();

    if (recorder.mode === 'recording') {
      tickRef.current++;
      if (tickRef.current % 30 === 0) {
        onFrameRecord(recorder.recordedFrameCount);
      }
    } else {
      tickRef.current = 0;
    }
  }, -1001);

  return null;
}
