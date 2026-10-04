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

import type { ReactElement } from 'react';
import { type Points, Vector3 } from 'three';

import { TraceVisualization } from './TraceVisualization';
import type { RecordedFrame } from './xrInputRecorder';

let mockFrame: () => void;
const sample: RecordedFrame = {
  timeMs: 0,
  poses: {
    leftGrip: { px: 1, py: 0, pz: 0, ox: 0, oy: 0, oz: 0, ow: 1 },
    leftAim: null,
    rightGrip: null,
    rightAim: null,
  },
  gamepads: { left: null, right: null },
  handJoints: { left: {}, right: {} },
};
const mockRecorder = {
  currentFrame: sample as RecordedFrame | null,
  replaySceneAlignment: { px: 10, py: 0, pz: 0, ox: 0, oy: 0, oz: 0, ow: 1 },
};

jest.mock('@react-three/fiber', () => ({
  useFrame: (callback: () => void) => {
    mockFrame = callback;
  },
}));
jest.mock('./RecorderContext', () => ({
  useRecorder: () => ({ recorder: mockRecorder, mode: 'replaying' }),
}));
jest.mock('react', () => ({
  ...jest.requireActual('react'),
  useRef: () => ({ current: null }),
  useEffect: jest.fn(),
}));

test('places the whole replay trail in calibrated scene space and hides it while paused', () => {
  const tree = TraceVisualization({ showTrace: true });
  const points = (tree.props.children as ReactElement<{ object: Points }>[])[0].props.object;
  const worldPoint = (index: number) => {
    points.updateMatrixWorld();
    return new Vector3()
      .fromBufferAttribute(points.geometry.getAttribute('position'), index)
      .applyMatrix4(points.matrixWorld);
  };
  mockFrame();
  expect(worldPoint(0).x).toBeCloseTo(11);

  // A reset changes the placement of earlier samples as well as the newest one.
  mockRecorder.replaySceneAlignment = {
    px: 100,
    py: 0,
    pz: 0,
    ox: 0,
    oy: Math.SQRT1_2,
    oz: 0,
    ow: Math.SQRT1_2,
  };
  mockFrame();
  for (const index of [0, 1]) {
    expect(worldPoint(index).x).toBeCloseTo(100);
    expect(worldPoint(index).z).toBeCloseTo(-1);
  }
  mockRecorder.currentFrame = null;
  mockFrame();
  expect(points.visible).toBe(false);
  for (const { props } of tree.props.children as ReactElement<{ object: Points }>[]) {
    props.object.geometry.dispose();
  }
});
