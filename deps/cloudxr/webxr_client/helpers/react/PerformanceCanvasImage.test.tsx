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
  let root: Root | undefined;
  let getContext: jest.SpyInstance;
  let dispose: jest.SpyInstance;

  const renderAndFrame = (): CanvasTexture => {
    act(() => root!.render(<PerformanceCanvasImage />));
    expect(mockFrame).toBeDefined();
    mockFrame!();
    expect(mockImage.texture.value).toBeInstanceOf(CanvasTexture);
    return mockImage.texture.value!;
  };

  beforeEach(() => {
    (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
    host = document.createElement('div');
    root = createRoot(host);
    getContext = jest
      .spyOn(HTMLCanvasElement.prototype, 'getContext')
      .mockReturnValue(ctx as never);
    dispose = jest.spyOn(CanvasTexture.prototype, 'dispose');
    mockImage.texture.value = undefined;
    mockFrame = undefined;
    jest.clearAllMocks();
  });

  afterEach(() => {
    if (root) act(() => root!.unmount());
    dispose.mockRestore();
    getContext.mockRestore();
  });

  it('draws the static labels and empty values before metrics arrive', () => {
    const ownedTexture = renderAndFrame();
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
    expect(mockImage.texture.value).toBe(ownedTexture);
  });

  it('replaces a foreign texture with the texture owned by this mount', () => {
    const ownedTexture = renderAndFrame();
    const foreignTexture = new CanvasTexture(document.createElement('canvas'));

    mockImage.texture.value = foreignTexture;
    mockFrame!();

    expect(mockImage.texture.value).toBe(ownedTexture);
    foreignTexture.dispose();
  });

  it('clears an owned texture and disposes it on unmount', () => {
    const ownedTexture = renderAndFrame();

    act(() => root!.unmount());
    root = undefined;

    expect(mockImage.texture.value).toBeUndefined();
    expect(dispose.mock.instances).toContain(ownedTexture);
  });

  it('preserves a foreign texture while disposing its own on unmount', () => {
    const ownedTexture = renderAndFrame();
    const foreignTexture = new CanvasTexture(document.createElement('canvas'));
    mockImage.texture.value = foreignTexture;

    act(() => root!.unmount());
    root = undefined;

    expect(mockImage.texture.value).toBe(foreignTexture);
    expect(dispose.mock.instances).toContain(ownedTexture);
    foreignTexture.dispose();
  });

  it('creates a distinct texture after unmount and remount', () => {
    const firstTexture = renderAndFrame();
    act(() => root!.unmount());
    expect(dispose.mock.instances).toContain(firstTexture);

    host = document.createElement('div');
    root = createRoot(host);
    const secondTexture = renderAndFrame();

    expect(secondTexture).not.toBe(firstTexture);
    expect(dispose.mock.instances).not.toContain(secondTexture);
  });

  it('draws populated metric values from the latest signals', () => {
    const render = metric('72.0');
    const pose = metric('71.0');
    const stream = metric('70.0');
    const latency = metric('12.3ms');
    act(() =>
      root!.render(
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
