/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
import {
  controllerSweep,
  initialInput,
  parseTrace,
  SyntheticXRController,
  type SyntheticDevice,
} from './controller';

function device(): SyntheticDevice {
  const controller = () => ({
    position: { set: jest.fn() },
    quaternion: { set: jest.fn() },
    updateButtonValue: jest.fn(),
  });
  return {
    controlMode: 'manual',
    position: { set: jest.fn() },
    quaternion: { set: jest.fn() },
    controllers: { left: controller(), right: controller() },
    notifyStateChange: jest.fn(),
  };
}

describe('Synthetic controller input contract', () => {
  beforeEach(() => jest.useFakeTimers());
  afterEach(() => jest.useRealTimers());
  test('maps grip to squeeze and supplies quaternion without changing convention', () => {
    const d = device();
    const api = new SyntheticXRController(d);
    api.setController('right', {
      position: [1, 2, -1],
      quaternion: [0, 1, 0, 0],
      trigger: 0.7,
      grip: 1,
    });
    expect(d.controllers.right.position.set).toHaveBeenLastCalledWith(1, 2, -1);
    expect(d.controllers.right.quaternion.set).toHaveBeenLastCalledWith(0, 1, 0, 0);
    expect(d.controllers.right.updateButtonValue).toHaveBeenCalledWith('squeeze', 1);
    expect(d.controllers.right.updateButtonValue).toHaveBeenCalledWith('trigger', 0.7);
    api.dispose();
    expect(d.controlMode).toBe('manual');
  });
  test.each([
    { position: [NaN, 0, 0] },
    { position: [11, 0, 0] },
    { quaternion: [0, 0, 0, 0] },
    { trigger: 2 },
    { grip: -1 },
  ])('invalid updates are atomic: %j', update => {
    const d = device();
    const api = new SyntheticXRController(d);
    const old = api.getState();
    const calls = (d.controllers.right.position.set as jest.Mock).mock.calls.length;
    expect(() => api.setController('right', update as never)).toThrow();
    expect(api.getState()).toEqual(old);
    expect(d.controllers.right.position.set).toHaveBeenCalledTimes(calls);
    api.dispose();
  });
  test('single replay completes, releases buttons and keeps final pose', async () => {
    const api = new SyntheticXRController(device());
    const trace = controllerSweep();
    trace.frames[180].state.right.trigger = 1;
    const done = api.play(trace);
    expect(() => api.setHead({ position: [0, 1, 0] })).toThrow('Stop playback');
    jest.advanceTimersByTime(9050);
    await done;
    expect(api.isPlaying).toBe(false);
    expect(api.getState().right.trigger).toBe(0);
    expect(api.getState().right.position).toEqual(trace.frames[180].state.right.position);
    api.dispose();
  });
  test('loop restarts and explicit stop cancels scheduled input', async () => {
    const api = new SyntheticXRController(device());
    const done = api.play(controllerSweep(), { loop: true });
    jest.advanceTimersByTime(9400);
    expect(api.isPlaying).toBe(true);
    api.stop();
    await done;
    const stopped = api.getState();
    jest.advanceTimersByTime(20000);
    expect(api.getState()).toEqual(stopped);
    expect(stopped.right.grip).toBe(0);
    api.dispose();
  });
  test('recording captures input and final hold; exported state cannot mutate device', () => {
    const api = new SyntheticXRController(device());
    api.record();
    jest.advanceTimersByTime(100);
    api.setController('left', { trigger: 0.5 });
    jest.advanceTimersByTime(100);
    const trace = api.finishRecording();
    expect(trace.frames.map(f => f.timeMs)).toEqual([0, 100, 200]);
    expect(trace.frames[1].state.left.trigger).toBe(0.5);
    trace.frames[1].state.left.trigger = 1;
    expect(api.getState().left.trigger).toBe(0.5);
    api.dispose();
  });
  test('rejects malformed traces before starting or changing input', () => {
    const trace = controllerSweep();
    trace.frames[1].timeMs = 0;
    const api = new SyntheticXRController(device());
    expect(() => api.play(trace)).toThrow('strictly increase');
    expect(api.getState()).toEqual(initialInput());
    expect(() => parseTrace({ version: 1, frames: [] })).toThrow();
    api.dispose();
  });
  test('browser replay suspends script input; disposal makes retained handles unusable', () => {
    const api = new SyntheticXRController(device());
    api.setSuspended(true);
    expect(() => api.setController('right', { trigger: 1 })).toThrow('owns tracking');
    api.setSuspended(false);
    api.setController('right', { trigger: 1 });
    api.dispose();
    expect(() => api.reset()).toThrow('disposed');
  });
});
