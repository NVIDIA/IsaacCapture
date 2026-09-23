/** @jest-environment jsdom */

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ReadonlySignal } from '@preact/signals-react';
import { act } from 'react';
import { createRoot, Root } from 'react-dom/client';
import { CanvasTexture } from 'three';

import { PerformanceCanvasImage } from './PerformanceCanvasImage';

const metric = <T,>(value: T): ReadonlySignal<T> => ({ value }) as ReadonlySignal<T>;

let mockFrame: (() => void) | undefined;
const mockImage: { texture: { value: CanvasTexture | undefined } } = {
  texture: { value: undefined },
};

jest.mock('@react-three/fiber', () => ({
  useFrame: (callback: () => void) => {
    mockFrame = callback;
  },
}));
jest.mock('@react-three/uikit', () => ({
  Image: require('react').forwardRef((_props: unknown, ref: React.Ref<unknown>) => {
    require('react').useImperativeHandle(ref, () => mockImage);
    return null;
  }),
}));

const ctx = {
  beginPath: jest.fn(),
  moveTo: jest.fn(),
  lineTo: jest.fn(),
  quadraticCurveTo: jest.fn(),
  closePath: jest.fn(),
  fill: jest.fn(),
  clearRect: jest.fn(),
  fillText: jest.fn(),
  measureText: jest.fn(() => ({ width: 40 })),
};

describe('PerformanceCanvasImage', () => {
  let host: HTMLDivElement;
  let root: Root;
  let getContext: jest.SpyInstance;

  beforeEach(() => {
    (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
    host = document.createElement('div');
    root = createRoot(host);
    getContext = jest
      .spyOn(HTMLCanvasElement.prototype, 'getContext')
      .mockReturnValue(ctx as never);
    mockImage.texture.value = undefined;
    mockFrame = undefined;
    jest.clearAllMocks();
  });

  afterEach(() => {
    act(() => root.unmount());
    getContext.mockRestore();
  });

  it('draws the static labels and empty values before metrics arrive', () => {
    act(() => root.render(<PerformanceCanvasImage />));
    expect(mockFrame).toBeDefined();
    mockFrame!();

    expect(mockImage.texture.value).toBeInstanceOf(CanvasTexture);
    const text = ctx.fillText.mock.calls.map(([value]) => value);
    expect(text).toEqual([
      'Render FPS',
      '  —',
      'Pose Send FPS',
      '  —',
      'Streaming FPS',
      '  —',
      'Pose-to-Render',
      '  —',
    ]);

    // The image library may resolve an empty src after our first assignment.
    mockImage.texture.value = undefined;
    mockFrame!();
    expect(mockImage.texture.value).toBeInstanceOf(CanvasTexture);
  });

  it('draws populated metric values from the latest signals', () => {
    const render = metric('72.0');
    const pose = metric('71.0');
    const stream = metric('70.0');
    const latency = metric('12.3ms');
    act(() =>
      root.render(
        <PerformanceCanvasImage
          renderFpsText={render}
          poseSendFpsText={pose}
          streamingFpsText={stream}
          poseToRenderText={latency}
          sessionQuality={metric(3)}
        />
      )
    );
    mockFrame!();
    expect(ctx.fillText.mock.calls.map(([value]) => value)).toEqual([
      'Render FPS',
      '  72.0',
      'Pose Send FPS',
      '  71.0',
      'Streaming FPS',
      '  70.0',
      'Pose-to-Render',
      '  12.3ms',
    ]);
  });
});
