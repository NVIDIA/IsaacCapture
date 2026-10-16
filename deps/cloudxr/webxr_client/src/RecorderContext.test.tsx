/** @jest-environment jsdom */

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

import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';

import { type RecorderContextValue, RecorderProvider, useRecorder } from './RecorderContext';

let current: RecorderContextValue;
let root: Root;

function Probe() {
  current = useRecorder();
  return null;
}

function frame(session: XRSession): XRFrame {
  return {
    session,
    predictedDisplayTime: 0,
    getViewerPose: () => ({
      transform: { position: { x: 0, y: 0, z: 0 }, orientation: { x: 0, y: 0, z: 0, w: 1 } },
    }),
  } as unknown as XRFrame;
}

beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  root = createRoot(document.createElement('div'));
  act(() =>
    root.render(
      <RecorderProvider>
        <Probe />
      </RecorderProvider>
    )
  );
});

afterEach(() => act(() => root.unmount()));

test('exposes cross-session calibration and resumes after the operator calibrates', () => {
  const space = new EventTarget() as XRReferenceSpace;
  const original = { inputSources: [] } as unknown as XRSession;
  act(() => current.startRecord());
  act(() => current.recorder.beginFrame(frame(original), space));
  act(() => current.stopRecord());
  expect(current.savedRecording?.frames).toHaveLength(1);
  act(() => current.startReplay());
  const next = { inputSources: [] } as unknown as XRSession;
  act(() => {
    current.recorder.beginFrame(frame(next), space);
    current.onFrameState();
  });
  expect(current.mode).toBe('replaying');
  expect(current.replayNeedsCalibration).toBe(true);
  act(() => current.calibrateReplay());
  act(() => {
    current.recorder.beginFrame(frame(next), space);
    current.onFrameState();
  });
  expect(current.replayNeedsCalibration).toBe(false);
  expect(current.recorder.currentFrame).not.toBeNull();
});

test('saves interrupted recording and updates the UI when the tracking origin changes', () => {
  const session = { inputSources: [] } as unknown as XRSession;
  const space = new EventTarget() as XRReferenceSpace;
  act(() => current.startRecord());
  act(() => current.recorder.beginFrame(frame(session), space));
  act(() => {
    space.dispatchEvent(Object.assign(new Event('reset'), { transform: null }));
    current.onFrameState();
  });
  expect(current.mode).toBe('idle');
  expect(current.recordingInterrupted).toBe(true);
  expect(current.savedRecording?.frames).toHaveLength(1);
  act(() => current.startRecord());
  expect(current.recordingInterrupted).toBe(false);
});
