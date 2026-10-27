/** @jest-environment jsdom */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { readSessionError, saveSessionError } from './sessionError';

afterEach(() => sessionStorage.clear());

test('preserves the full error across reload reads until a connection succeeds', () => {
  const error = 'CloudXR session stopped: Failed (Error code: 0xC0DE)';
  saveSessionError(error);
  expect(readSessionError()).toBe(error);
  expect(readSessionError()).toBe(error);
  saveSessionError('');
  expect(readSessionError()).toBe('');
});

test('continues when browser storage is unavailable', () => {
  const get = jest.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
    throw new Error('Storage disabled');
  });
  const set = jest.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
    throw new Error('Storage disabled');
  });
  expect(readSessionError()).toBe('');
  expect(() => saveSessionError('Connection failed')).not.toThrow();
  get.mockRestore();
  set.mockRestore();
});
