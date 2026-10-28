/** @jest-environment jsdom */

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { act, useEffect } from 'react';
import { createRoot, Root } from 'react-dom/client';

import { CloudXR2DUI } from './CloudXR2DUI';
import { useSessionError } from './useSessionError';

const storageKey = 'cxr.isaac.lastSessionError';
const error = 'CloudXR session stopped: Connection failed (Error code: 0xC0F2220F)';

describe('restored session error banner', () => {
  let ui: CloudXR2DUI;
  let root: Root;
  let updateError: (message: string) => void;
  const text = () => document.getElementById('errorMessageText')!;
  const box = () => document.getElementById('errorMessageBox')!;

  beforeEach(() => {
    (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
    jest.spyOn(console, 'info').mockImplementation(() => {});
    jest.spyOn(console, 'error').mockImplementation(() => {});
    document.documentElement.innerHTML = readFileSync(join(__dirname, 'index.html'), 'utf8');
    ui = new CloudXR2DUI();
    ui.initialize();
    root = createRoot(document.createElement('div'));
    sessionStorage.setItem(storageKey, error);
  });

  afterEach(() => {
    act(() => root.unmount());
    ui.cleanup();
    sessionStorage.clear();
    localStorage.clear();
    jest.restoreAllMocks();
  });

  test.each([
    ['CloudXR.js SDK is supported.', 'success'],
    ['Performance notice:\nAV1 codec is not supported on this device.', 'info'],
    ['CloudXR.js SDK is supported.\nUsing IWER (Immersive Web Emulator Runtime).', 'info'],
    ['Browser does not meet required capabilities:\nWebXR is required.', 'error'],
  ] as const)(
    'retains the full error after asynchronous startup notice: %s',
    async (notice, type) => {
      let finishCheck!: () => void;
      const capabilityCheck = new Promise<void>(resolve => {
        finishCheck = resolve;
      });

      function Startup() {
        updateError = useSessionError(ui);
        useEffect(() => {
          void capabilityCheck.then(() => ui.showStatus(notice, type));
        }, []);
        return null;
      }

      act(() => root.render(<Startup />));
      expect(text().textContent).toBe(error);
      await act(async () => {
        finishCheck();
        await capabilityCheck;
      });
      expect(text().textContent).toBe(error);
      expect(box().className).toBe('error-message-box show error');

      ui.showStatus('Reconnecting (1/3)', 'info');
      ui.hideStatusIfShowing(error);
      ui.hideError();
      expect(text().textContent).toBe(error);
      expect(box().classList.contains('show')).toBe(true);

      // App clears through this setter only when the stream reports Connected.
      act(() => updateError(''));
      expect(sessionStorage.getItem(storageKey)).toBeNull();
      expect(box().classList.contains('show')).toBe(false);
      ui.showStatus('A later notice', 'info');
      expect(text().textContent).toBe('A later notice');
    }
  );

  test('a new session failure replaces the pinned error and persists its code', () => {
    function Startup() {
      updateError = useSessionError(ui);
      return null;
    }
    act(() => root.render(<Startup />));
    const latest = 'CloudXR session stopped: Timeout (Error code: 0xC0F22210)';
    act(() => updateError(latest));
    expect(text().textContent).toBe(latest);
    expect(sessionStorage.getItem(storageKey)).toBe(latest);
  });

  test('restores a saved failure after the hook remounts', () => {
    function Startup() {
      updateError = useSessionError(ui);
      return null;
    }
    act(() => root.render(<Startup />));
    const latest = 'CloudXR session stopped: Timeout (Error code: 0xC0F22210)';
    act(() => updateError(latest));
    act(() => root.unmount());
    root = createRoot(document.createElement('div'));
    act(() => root.render(<Startup />));
    expect(text().textContent).toBe(latest);
    expect(box().className).toBe('error-message-box show error');
  });

  test.each(['getItem', 'setItem', 'removeItem'] as const)(
    'continues reporting and clearing errors when storage %s fails',
    method => {
      jest.spyOn(Storage.prototype, method).mockImplementation(() => {
        throw new Error('Storage disabled');
      });
      function Startup() {
        updateError = useSessionError(ui);
        return null;
      }
      act(() => root.render(<Startup />));
      act(() => updateError(error));
      expect(text().textContent).toBe(error);
      expect(box().classList.contains('show')).toBe(true);
      act(() => updateError(''));
      expect(box().classList.contains('show')).toBe(false);
    }
  );
});
