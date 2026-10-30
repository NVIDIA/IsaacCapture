/** @jest-environment jsdom */

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { act, StrictMode } from 'react';
import { createRoot, Root } from 'react-dom/client';

import { useControlPanelVisibility } from './useControlPanelVisibility';

type Options = Parameters<typeof useControlPanelVisibility>[0];

function Panel(props: Options) {
  const [hidden, setHidden] = useControlPanelVisibility(props);
  return <button onClick={() => setHidden(!hidden)}>{hidden ? 'Show' : 'Hide'}</button>;
}

describe('control panel visibility', () => {
  let host: HTMLDivElement;
  let root: Root;
  let options: Options;

  const render = (changes: Partial<Options> = {}) => {
    options = { ...options, ...changes };
    act(() =>
      root.render(
        <StrictMode>
          <Panel {...options} />
        </StrictMode>
      )
    );
  };
  const expectHidden = (hidden: boolean) => {
    expect(host.textContent).toBe(hidden ? 'Show' : 'Hide');
  };
  const toggle = () => act(() => host.querySelector('button')!.click());

  beforeEach(() => {
    (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
    host = document.createElement('div');
    root = createRoot(host);
    options = {
      isXRMode: false,
      panelHiddenAtStart: true,
      autoHideOnConnect: true,
      isStreaming: false,
    };
  });

  afterEach(() => act(() => root.unmount()));

  it('keeps video connection progress visible despite an old hide-on-entry preference', () => {
    render();
    expectHidden(false);
    render({ isXRMode: true });
    expectHidden(false);
    render(); // Waiting for the stream or testing the network is still not streaming.
    expectHidden(false);
    render({ isStreaming: true });
    expectHidden(true);
  });

  it('lets Show/Hide keep control after the stream connects', () => {
    render({ isXRMode: true, isStreaming: true });
    expectHidden(true);
    toggle();
    expectHidden(false);
    render(); // Repeated Connected notifications / other UI updates must not hide it again.
    expectHidden(false);
    render({ panelHiddenAtStart: false });
    expectHidden(false);
    toggle();
    expectHidden(true);
    toggle();
    expectHidden(false);
  });

  it('reopens on stream loss and hides again when reconnect succeeds', () => {
    render({ isXRMode: true, isStreaming: true });
    expectHidden(true);
    render({ isStreaming: false });
    expectHidden(false);
    render(); // Retry progress or a terminal error keeps the details open.
    expectHidden(false);
    render({ isStreaming: true });
    expectHidden(true);
  });

  it('starts the next XR session open while awaiting a new stream', () => {
    render({ isXRMode: true, isStreaming: true });
    expectHidden(true);
    render({ isXRMode: false, isStreaming: false });
    render({ isXRMode: true });
    expectHidden(false);
  });

  it.each([false, true])('preserves other presets with hide-on-entry=%s', panelHiddenAtStart => {
    render({ autoHideOnConnect: false, panelHiddenAtStart });
    render({ isXRMode: true });
    expectHidden(panelHiddenAtStart);
    render({ isStreaming: true });
    expectHidden(panelHiddenAtStart);
    toggle();
    render({ isStreaming: false });
    expectHidden(!panelHiddenAtStart);
    render({ isXRMode: false });
    render({ isXRMode: true });
    expectHidden(panelHiddenAtStart);
  });
});
