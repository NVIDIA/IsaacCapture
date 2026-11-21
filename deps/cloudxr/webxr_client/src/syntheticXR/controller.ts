/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

export type Hand = 'left' | 'right';
export type Vector3 = [number, number, number];
export type Quaternion = [number, number, number, number];
export type Pose = { position: Vector3; quaternion: Quaternion };
export type ControllerState = Pose & { trigger: number; grip: number };
export type InputState = { head: Pose; left: ControllerState; right: ControllerState };
export type InputTrace = {
  version: 1;
  referenceSpace: 'iwer-local';
  frames: Array<{ timeMs: number; state: InputState }>;
};

/** Structural interface keeps generated inputs independent of the IWER distribution. */
export interface SyntheticDevice {
  controlMode: string;
  notifyStateChange?: () => void;
  position: { set: (...v: Vector3) => unknown };
  quaternion: { set: (...q: Quaternion) => unknown };
  controllers: Record<
    Hand,
    {
      position: { set: (...v: Vector3) => unknown };
      quaternion: { set: (...q: Quaternion) => unknown };
      updateButtonValue: (button: string, value: number) => void;
    }
  >;
}

const copy = <T>(value: T): T => JSON.parse(JSON.stringify(value));
export function initialInput(): InputState {
  return {
    head: { position: [0, 1.6, 0], quaternion: [0, 0, 0, 1] },
    left: { position: [-0.25, 1.3, -0.4], quaternion: [0, 0, 0, 1], trigger: 0, grip: 0 },
    right: { position: [0.25, 1.3, -0.4], quaternion: [0, 0, 0, 1], trigger: 0, grip: 0 },
  };
}

function validatePose(pose: Pose): void {
  if (
    !pose ||
    !Array.isArray(pose.position) ||
    pose.position.length !== 3 ||
    !pose.position.every(n => Number.isFinite(n) && Math.abs(n) <= 10)
  ) {
    throw new Error('Position must contain three finite metre values within ±10 m.');
  }
  if (
    !Array.isArray(pose.quaternion) ||
    pose.quaternion.length !== 4 ||
    !pose.quaternion.every(Number.isFinite) ||
    Math.abs(Math.hypot(...pose.quaternion) - 1) > 0.001
  ) {
    throw new Error('Quaternion must be normalized, in x/y/z/w order.');
  }
}

function validateState(state: InputState): void {
  if (!state) throw new Error('Missing input state.');
  validatePose(state.head);
  for (const hand of ['left', 'right'] as const) {
    validatePose(state[hand]);
    for (const value of [state[hand].trigger, state[hand].grip]) {
      if (!Number.isFinite(value) || value < 0 || value > 1) {
        throw new Error('Trigger and grip must be in [0, 1].');
      }
    }
  }
}

export function parseTrace(value: unknown): InputTrace {
  const trace = value as InputTrace;
  if (
    !trace ||
    trace.version !== 1 ||
    trace.referenceSpace !== 'iwer-local' ||
    !Array.isArray(trace.frames) ||
    trace.frames.length < 2 ||
    trace.frames.length > 36000
  ) {
    throw new Error('Expected a version 1 iwer-local trace with 2–36000 frames.');
  }
  let previous = -1;
  for (const frame of trace.frames) {
    if (
      !frame ||
      !Number.isFinite(frame.timeMs) ||
      frame.timeMs < 0 ||
      frame.timeMs <= previous ||
      frame.timeMs > 600000
    ) {
      throw new Error('Trace times must strictly increase within ten minutes.');
    }
    validateState(frame.state);
    previous = frame.timeMs;
  }
  if (trace.frames[0].timeMs !== 0) throw new Error('Trace must start at zero.');
  return copy(trace);
}

export class SyntheticXRController {
  readonly version = 1;
  private state = initialInput();
  private timer: ReturnType<typeof setInterval> | undefined;
  private originalMode: string;
  private recording: InputTrace | undefined;
  private recordStart = 0;
  private disposed = false;
  private listeners = new Set<() => void>();
  private finishPlayback: (() => void) | undefined;
  private suspended = false;
  lastFrame: {
    time: number;
    sources: Array<{ hand: string; aim: Pose | null; grip: Pose | null; buttons: number[] }>;
  } | null = null;
  status = 'Ready';

  constructor(private device: SyntheticDevice) {
    this.originalMode = device.controlMode;
    device.controlMode = 'programmatic';
    this.apply(this.state);
  }

  getState(): InputState {
    return copy(this.state);
  }
  get isPlaying(): boolean {
    return this.timer !== undefined;
  }
  get isRecording(): boolean {
    return this.recording !== undefined;
  }
  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  private changed(): void {
    this.listeners.forEach(fn => fn());
  }
  private assertActive(): void {
    if (this.disposed) throw new Error('Synthetic XR controller has been disposed.');
  }
  private assertManual(): void {
    this.assertActive();
    if (this.suspended) throw new Error('Browser replay owns tracking input.');
    if (this.isPlaying) throw new Error('Stop playback before changing input.');
  }
  private apply(state: InputState): void {
    this.assertActive();
    validateState(state);
    this.device.position.set(...state.head.position);
    this.device.quaternion.set(...state.head.quaternion);
    for (const hand of ['left', 'right'] as const) {
      const target = this.device.controllers[hand];
      target.position.set(...state[hand].position);
      target.quaternion.set(...state[hand].quaternion);
      target.updateButtonValue('trigger', state[hand].trigger);
      target.updateButtonValue('squeeze', state[hand].grip);
    }
    this.state = copy(state);
    this.device.notifyStateChange?.();
    this.changed();
  }

  setSuspended(value: boolean): void {
    if (value && !this.suspended) this.stop();
    this.suspended = value;
  }

  observeFrame(frame: XRFrame, reference: XRReferenceSpace): void {
    const pose = (value: XRPose | null | undefined): Pose | null =>
      value
        ? {
            position: [
              value.transform.position.x,
              value.transform.position.y,
              value.transform.position.z,
            ],
            quaternion: [
              value.transform.orientation.x,
              value.transform.orientation.y,
              value.transform.orientation.z,
              value.transform.orientation.w,
            ],
          }
        : null;
    this.lastFrame = {
      time: performance.now(),
      sources: Array.from(frame.session.inputSources).map(source => ({
        hand: source.handedness,
        aim: pose(frame.getPose(source.targetRaySpace, reference)),
        grip: source.gripSpace ? pose(frame.getPose(source.gripSpace, reference)) : null,
        buttons: Array.from(source.gamepad?.buttons ?? [], button => button.value),
      })),
    };
  }

  setController(hand: Hand, update: Partial<ControllerState>): void {
    this.assertManual();
    if (hand !== 'left' && hand !== 'right') throw new Error('Choose left or right.');
    this.apply({ ...this.state, [hand]: { ...this.state[hand], ...update } });
    this.capture();
  }
  setHead(update: Partial<Pose>): void {
    this.assertManual();
    this.apply({ ...this.state, head: { ...this.state.head, ...update } });
    this.capture();
  }
  reset(): void {
    this.assertManual();
    this.apply(initialInput());
    this.capture();
  }

  record(): void {
    this.assertManual();
    if (this.isRecording) throw new Error('Already recording.');
    this.recordStart = performance.now();
    this.recording = {
      version: 1,
      referenceSpace: 'iwer-local',
      frames: [{ timeMs: 0, state: this.getState() }],
    };
    this.status = 'Recording input';
    this.changed();
  }
  private capture(): void {
    if (!this.recording) return;
    const frames = this.recording.frames;
    const timeMs = performance.now() - this.recordStart;
    if (timeMs > 600000 || frames.length >= 35999) {
      this.status = 'Recording limit reached — export trace';
      this.changed();
      return;
    }
    if (timeMs > frames[frames.length - 1].timeMs) frames.push({ timeMs, state: this.getState() });
  }
  finishRecording(): InputTrace {
    this.assertActive();
    if (!this.recording) throw new Error('No recording in progress.');
    this.capture();
    if (this.recording.frames.length < 2) throw new Error('Record for longer before exporting.');
    const result = parseTrace(this.recording);
    this.recording = undefined;
    this.status = 'Input trace saved';
    this.changed();
    return result;
  }

  play(value: unknown, options: { loop?: boolean } = {}): Promise<void> {
    this.assertManual();
    if (this.isRecording) throw new Error('Finish recording before playback.');
    const trace = parseTrace(value);
    const started = performance.now();
    const duration = trace.frames[trace.frames.length - 1].timeMs;
    this.apply(trace.frames[0].state);
    this.status = options.loop ? 'Playing loop' : 'Playing once';
    const done = new Promise<void>(resolve => {
      this.finishPlayback = resolve;
    });
    this.timer = setInterval(() => {
      const elapsed = performance.now() - started;
      if (elapsed >= duration && !options.loop) {
        this.apply(trace.frames[trace.frames.length - 1].state);
        this.stop();
        this.status = 'Playback complete — buttons released';
        this.changed();
        return;
      }
      const time = options.loop ? elapsed % duration : elapsed;
      let index = 0;
      while (index + 1 < trace.frames.length && trace.frames[index + 1].timeMs <= time) index++;
      this.apply(trace.frames[index].state);
    }, 16);
    this.changed();
    return done;
  }

  stop(): void {
    this.assertActive();
    const hadInput =
      this.isPlaying ||
      this.state.left.grip ||
      this.state.left.trigger ||
      this.state.right.grip ||
      this.state.right.trigger;
    clearInterval(this.timer);
    this.timer = undefined;
    this.apply({
      ...this.state,
      left: { ...this.state.left, grip: 0, trigger: 0 },
      right: { ...this.state.right, grip: 0, trigger: 0 },
    });
    this.capture();
    // Repeated disconnect notifications must not erase a completed replay result.
    if (hadInput) this.status = 'Stopped — buttons released';
    this.finishPlayback?.();
    this.finishPlayback = undefined;
    this.changed();
  }
  dispose(): void {
    if (this.disposed) return;
    this.stop();
    this.device.controlMode = this.originalMode;
    this.listeners.clear();
    this.disposed = true;
  }
}

/** A generated sequence, not a captured human demonstration. */
export function controllerSweep(): InputTrace {
  const frames: InputTrace['frames'] = [];
  for (let i = 0; i <= 180; i++) {
    const state = initialInput();
    const phase = (i / 180) * Math.PI * 2;
    state.right.position[0] += 0.12 * Math.sin(phase);
    state.right.position[1] += 0.08 * (1 - Math.cos(phase));
    state.left.position[2] -= 0.06 * Math.sin(phase);
    state.right.grip = i > 0 && i < 180 ? 1 : 0;
    state.right.trigger = i >= 60 && i < 120 ? 1 : 0;
    frames.push({ timeMs: i * 50, state });
  }
  return { version: 1, referenceSpace: 'iwer-local', frames };
}
