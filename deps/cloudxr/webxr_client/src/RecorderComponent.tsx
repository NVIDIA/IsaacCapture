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

import { useFrame, useThree } from '@react-three/fiber';
import { useEffect, useRef, useState } from 'react';

import { useRecorder } from './RecorderContext';

/** Mirrors CloudXRComponent's 'auto' fallback order. */
const AUTO_FALLBACKS: XRReferenceSpaceType[] = ['local-floor', 'local', 'viewer', 'unbounded'];

/**
 * Resolve the same reference space CloudXR streams poses in.
 *
 * three.js pins WebXRManager to 'local-floor' (origin on the floor) while
 * CloudXR requests `config.referenceSpaceType` ('local' by default, origin at
 * the headset). Recording in one frame and streaming in the other makes hand
 * poses absolute above the floor while the scene stays anchored to the headset,
 * so entering at a different height moves the table but not the trajectory.
 */
async function resolveStreamingSpace(
  session: XRSession,
  referenceSpaceType: string
): Promise<XRReferenceSpace | null> {
  const types: XRReferenceSpaceType[] =
    referenceSpaceType === 'auto' ? AUTO_FALLBACKS : [referenceSpaceType as XRReferenceSpaceType];
  for (const type of types) {
    try {
      const space = await session.requestReferenceSpace(type);
      if (space) return space;
    } catch {
      // Try the next candidate; CloudXR tolerates the same failures.
    }
  }
  return null;
}


interface RecorderComponentProps {
  /** True when the CloudXR session is in the Connected state — gates replay frame advancement. */
  isConnected: boolean;
  /** Capture live input while idle so the trace can render it. */
  showTrace: boolean;
  /** `config.referenceSpaceType`, so recording uses the frame CloudXR streams in. */
  referenceSpaceType: string;
}

export function RecorderComponent({
  isConnected,
  showTrace,
  referenceSpaceType,
}: RecorderComponentProps) {
  const { recorder, onFrameRecord, onFrameState } = useRecorder();
  const gl = useThree(state => state.gl);
  const tickRef = useRef(0);
  const [streamingSpace, setStreamingSpace] = useState<XRReferenceSpace | null>(null);

  useEffect(() => {
    const webXRManager = gl.xr;
    if (!webXRManager) return;
    let cancelled = false;

    const onSessionStart = () => {
      const session = webXRManager.getSession() as XRSession | null;
      if (!session) return;
      void resolveStreamingSpace(session, referenceSpaceType).then(space => {
        if (!cancelled) setStreamingSpace(space);
      });
    };
    const onSessionEnd = () => {
      if (!cancelled) setStreamingSpace(null);
    };

    if (webXRManager.getSession()) onSessionStart();
    webXRManager.addEventListener('sessionstart', onSessionStart);
    webXRManager.addEventListener('sessionend', onSessionEnd);
    return () => {
      cancelled = true;
      webXRManager.removeEventListener('sessionstart', onSessionStart);
      webXRManager.removeEventListener('sessionend', onSessionEnd);
    };
  }, [gl, referenceSpaceType]);

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
      streamingSpace ?? state.gl.xr.getReferenceSpace(),
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
