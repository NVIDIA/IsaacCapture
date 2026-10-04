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

import { type RecordedFrame, type Recording, XRInputRecorder } from './xrInputRecorder';

class FakeRigidTransform {
  readonly position: DOMPointReadOnly;
  readonly orientation: DOMPointReadOnly;

  constructor(position: DOMPointInit = {}, orientation: DOMPointInit = { w: 1 }) {
    this.position = {
      x: position.x ?? 0,
      y: position.y ?? 0,
      z: position.z ?? 0,
      w: position.w ?? 1,
    } as DOMPointReadOnly;
    this.orientation = {
      x: orientation.x ?? 0,
      y: orientation.y ?? 0,
      z: orientation.z ?? 0,
      w: orientation.w ?? 1,
    } as DOMPointReadOnly;
  }
}

const sceneSpace = {} as XRReferenceSpace;

function pose(x: number, y = 0, z = 0, orientation: DOMPointInit = { w: 1 }): XRPose {
  return {
    transform: new FakeRigidTransform({ x, y, z, w: 1 }, orientation),
    emulatedPosition: false,
    linearVelocity: null,
    angularVelocity: null,
  } as unknown as XRPose;
}

function jointPose(x: number, radius = 0.01): XRJointPose {
  return { ...pose(x), radius } as XRJointPose;
}

function gamepad(axis: number): Gamepad {
  return {
    axes: [axis],
    buttons: [{ value: axis, pressed: axis > 0, touched: true }],
  } as unknown as Gamepad;
}

function makeSession(inputSources: XRInputSource[] = []): XRSession {
  return Object.assign(new EventTarget(), { inputSources }) as unknown as XRSession;
}

// Frames in a test share a session unless the test explicitly starts another.
let defaultSession: XRSession;
beforeEach(() => {
  defaultSession = makeSession();
});

type PoseResolver = (space: XRSpace, baseSpace: XRSpace) => XRPose | null;
type JointResolver = (joint: XRJointSpace, baseSpace: XRSpace) => XRJointPose | null;

function makeFrame(
  inputSources: XRInputSource[] = [],
  getPose: PoseResolver = () => null,
  getJointPose: JointResolver = () => null,
  predictedDisplayTime = 0,
  viewerPose: XRPose | null = pose(0),
  session: XRSession = defaultSession
): XRFrame {
  Object.assign(session, { inputSources });
  return {
    session,
    predictedDisplayTime,
    getPose,
    getJointPose,
    getViewerPose: () => viewerPose,
  } as unknown as XRFrame;
}

function frameData(x = 0): RecordedFrame {
  return {
    timeMs: x,
    poses: {
      leftGrip: { px: x, py: 0, pz: 0, ox: 0, oy: 0, oz: 0, ow: 1 },
      leftAim: { px: x + 1, py: 0, pz: 0, ox: 0, oy: 0, oz: 0, ow: 1 },
      rightGrip: null,
      rightAim: null,
    },
    gamepads: {
      left: { axes: [x], buttons: [{ value: x, pressed: true, touched: true }] },
      right: null,
    },
    handJoints: {
      left: {
        wrist: { px: x + 2, py: 0, pz: 0, ox: 0, oy: 0, oz: 0, ow: 1, radius: 0.02 },
      },
      right: {},
    },
  };
}

function recording(...frames: RecordedFrame[]): Recording {
  return { version: 1, frames };
}

function calibratedRecording(viewerX: number, ...frames: RecordedFrame[]): Recording {
  return {
    version: 1,
    calibration: {
      mode: 'viewer-start-yaw',
      pose: { px: viewerX, py: 0, pz: 0, ox: 0, oy: 0, oz: 0, ow: 1 },
    },
    frames,
  };
}

function timedFrame(timeMs: number, x: number): RecordedFrame {
  return { ...frameData(x), timeMs };
}

function recordFrames(count: number): Recording {
  const recorder = new XRInputRecorder();
  recorder.startRecording();
  for (let index = 0; index < count; index++) {
    recorder.beginFrame(makeFrame(), sceneSpace);
  }
  recorder.stopRecording();
  return recorder.getRecording();
}

beforeAll(() => {
  (global as { XRRigidTransform?: unknown }).XRRigidTransform = FakeRigidTransform;
});

describe('lifecycle and frame advancement', () => {
  test('records frames and returns to idle', () => {
    const recorder = new XRInputRecorder();
    recorder.startRecording();
    recorder.beginFrame(makeFrame(), sceneSpace);
    recorder.beginFrame(makeFrame(), sceneSpace);
    expect(recorder.mode).toBe('recording');
    expect(recorder.recordedFrameCount).toBe(2);

    recorder.stopRecording();
    expect(recorder.mode).toBe('idle');
    expect(recorder.currentFrame).toBeNull();
    expect(recorder.getRecording().frames).toHaveLength(2);
  });

  test.each(['recording', 'replaying'] as const)(
    'rejects starting another operation while %s',
    mode => {
      const recorder = new XRInputRecorder();
      if (mode === 'recording') recorder.startRecording();
      else recorder.startReplay(recording(frameData()));
      expect(() => recorder.startRecording()).toThrow(/already active/);
    }
  );

  test('does not record or advance replay while disconnected', () => {
    const recorder = new XRInputRecorder();
    recorder.startRecording();
    recorder.beginFrame(makeFrame(), sceneSpace, false);
    expect(recorder.recordedFrameCount).toBe(0);
    recorder.stopRecording();

    recorder.startReplay(recording(frameData(1), frameData(2)));
    recorder.beginFrame(makeFrame(), sceneSpace, false);
    expect(recorder.replayFrameIndex).toBe(0);
    expect(recorder.currentFrame).toBeNull();
  });

  test('loops replay by frame and can clamp at the final frame', () => {
    const looped = new XRInputRecorder();
    looped.startReplay(recording(frameData(1), frameData(2)), true, 'frame');
    looped.beginFrame(makeFrame(), sceneSpace);
    looped.beginFrame(makeFrame(), sceneSpace);
    expect(looped.replayFrameIndex).toBe(0);

    const clamped = new XRInputRecorder();
    clamped.startReplay(recording(frameData(1), frameData(2)), false, 'frame');
    clamped.beginFrame(makeFrame(), sceneSpace);
    clamped.beginFrame(makeFrame(), sceneSpace);
    clamped.beginFrame(makeFrame(), sceneSpace);
    expect(clamped.currentFrame).toEqual(frameData(2));
    expect(clamped.replayFrameIndex).toBe(1);
  });

  test('time-paces replay by default and interpolates timestamped samples', () => {
    const first = timedFrame(0, 0);
    first.gamepads.left!.buttons[0].pressed = false;
    const second = timedFrame(100, 10);
    second.poses.leftGrip = {
      ...second.poses.leftGrip!,
      oz: 1,
      ow: 0,
    };

    const recorder = new XRInputRecorder();
    recorder.startReplay(recording(first, second), false);
    recorder.beginFrame(makeFrame([], undefined, undefined, 1000), sceneSpace);
    recorder.beginFrame(makeFrame([], undefined, undefined, 1050), sceneSpace);

    expect(recorder.currentFrame?.timeMs).toBe(50);
    expect(recorder.currentFrame?.poses.leftGrip?.px).toBe(5);
    expect(recorder.currentFrame?.poses.leftGrip?.oz).toBeCloseTo(Math.sqrt(0.5));
    expect(recorder.currentFrame?.poses.leftGrip?.ow).toBeCloseTo(Math.sqrt(0.5));
    expect(recorder.currentFrame?.gamepads.left?.axes).toEqual([5]);
    expect(recorder.currentFrame?.gamepads.left?.buttons[0].pressed).toBe(false);
    expect(recorder.currentFrame?.handJoints.left.wrist?.px).toBe(7);
  });

  test('holds the earlier gamepad sample when controller layouts differ', () => {
    const first = timedFrame(0, 0);
    const second = timedFrame(100, 10);
    second.gamepads.left!.axes.push(20);
    second.gamepads.left!.buttons.push({ value: 20, pressed: true, touched: true });

    const recorder = new XRInputRecorder();
    recorder.startReplay(recording(first, second), false);
    recorder.beginFrame(makeFrame([], undefined, undefined, 1000), sceneSpace);
    recorder.beginFrame(makeFrame([], undefined, undefined, 1050), sceneSpace);

    expect(recorder.currentFrame?.gamepads.left).toEqual(first.gamepads.left);
  });

  test('holds the final timed sample before looping', () => {
    const recorder = new XRInputRecorder();
    recorder.startReplay(recording(timedFrame(0, 0), timedFrame(100, 10)), true, 'time');

    recorder.beginFrame(makeFrame([], undefined, undefined, 1000), sceneSpace);
    recorder.beginFrame(makeFrame([], undefined, undefined, 1100), sceneSpace);
    expect(recorder.currentFrame).toEqual(timedFrame(100, 10));

    recorder.beginFrame(makeFrame([], undefined, undefined, 1150), sceneSpace);
    expect(recorder.currentFrame).toEqual(timedFrame(100, 10));

    recorder.beginFrame(makeFrame([], undefined, undefined, 1200), sceneSpace);
    expect(recorder.currentFrame).toEqual(timedFrame(0, 0));
  });

  test('pauses time-paced replay while advancement is gated off', () => {
    const recorder = new XRInputRecorder();
    recorder.startReplay(recording(timedFrame(0, 0), timedFrame(100, 10)), false, 'time');

    recorder.beginFrame(makeFrame([], undefined, undefined, 1000), sceneSpace);
    recorder.beginFrame(makeFrame([], undefined, undefined, 1040), sceneSpace);
    recorder.beginFrame(makeFrame([], undefined, undefined, 1100), sceneSpace, false);
    recorder.beginFrame(makeFrame([], undefined, undefined, 2000), sceneSpace);
    expect(recorder.currentFrame?.poses.leftGrip?.px).toBe(4);

    recorder.beginFrame(makeFrame([], undefined, undefined, 2020), sceneSpace);
    expect(recorder.currentFrame?.poses.leftGrip?.px).toBe(6);
  });

  test('captures live input while idle only when requested', () => {
    const recorder = new XRInputRecorder();
    recorder.beginFrame(makeFrame(), sceneSpace, true, false);
    expect(recorder.currentFrame).toBeNull();
    recorder.beginFrame(makeFrame(), sceneSpace, true, true);
    expect(recorder.currentFrame).toEqual({
      timeMs: 0,
      controllerProfiles: { left: null, right: null },
      poses: { leftGrip: null, leftAim: null, rightGrip: null, rightAim: null },
      gamepads: { left: null, right: null },
      handJoints: { left: {}, right: {} },
    });
  });
});

test.each([1, -1])('interpolates the shortest rotation for quaternion sign %s', sign => {
  const first = timedFrame(0, 0);
  const last = timedFrame(100, 4);
  last.poses.leftGrip = {
    ...last.poses.leftGrip!,
    oy: sign * Math.sin(Math.PI / 3),
    ow: sign * Math.cos(Math.PI / 3),
  };
  const recorder = new XRInputRecorder();
  recorder.startReplay(recording(first, last), false);
  recorder.beginFrame(makeFrame(), sceneSpace);
  recorder.beginFrame(makeFrame([], undefined, undefined, 50), sceneSpace);
  const result = recorder.currentFrame!.poses.leftGrip!;
  expect(result.px).toBeCloseTo(2);
  expect(result.oy).toBeCloseTo(Math.sin(Math.PI / 6));
  expect(result.ow).toBeCloseTo(Math.cos(Math.PI / 6));
  expect(Math.hypot(result.ox, result.oy, result.oz, result.ow)).toBeCloseTo(1);
});

describe('canonical scene-space capture', () => {
  test('captures grip, aim, gamepad, and joints by WebXR joint name', () => {
    const grip = {} as XRSpace;
    const aim = {} as XRSpace;
    const wrist = {} as XRJointSpace;
    const indexTip = {} as XRJointSpace;
    const source = {
      handedness: 'left',
      gripSpace: grip,
      targetRaySpace: aim,
      gamepad: gamepad(0.75),
      hand: new Map([
        ['wrist', wrist],
        ['index-finger-tip', indexTip],
      ]),
    } as unknown as XRInputSource;
    const frame = makeFrame(
      [source],
      space => (space === grip ? pose(1) : space === aim ? pose(2) : null),
      joint => (joint === wrist ? jointPose(3) : jointPose(4))
    );

    const recorder = new XRInputRecorder();
    recorder.startRecording();
    recorder.beginFrame(frame, sceneSpace);
    const captured = recorder.currentFrame!;

    expect(captured.poses.leftGrip?.px).toBe(1);
    expect(captured.poses.leftAim?.px).toBe(2);
    expect(captured.gamepads.left?.axes).toEqual([0.75]);
    expect(captured.handJoints.left.wrist?.px).toBe(3);
    expect(captured.handJoints.left['index-finger-tip']?.px).toBe(4);
  });

  test('captures a gravity-aligned viewer pose for cross-session calibration', () => {
    const quarterTurn = Math.sqrt(0.5);
    const recorder = new XRInputRecorder();
    recorder.startRecording();
    recorder.beginFrame(
      makeFrame([], undefined, undefined, 0, pose(1, 2, 3, { y: quarterTurn, w: quarterTurn })),
      sceneSpace
    );
    recorder.stopRecording();

    const calibration = recorder.getRecording().calibration;
    expect(calibration?.mode).toBe('viewer-start-yaw');
    expect(calibration?.pose).toMatchObject({ px: 1, py: 2, pz: 3, ox: 0, oz: 0 });
    expect(calibration?.pose.oy).toBeCloseTo(quarterTurn);
    expect(calibration?.pose.ow).toBeCloseTo(quarterTurn);
  });

  test('waits for a viewer pose before recording the first frame', () => {
    const recorder = new XRInputRecorder();
    recorder.startRecording();
    recorder.beginFrame(makeFrame([], undefined, undefined, 0, null), sceneSpace);
    expect(recorder.recordedFrameCount).toBe(0);

    recorder.beginFrame(makeFrame([], undefined, undefined, 1, pose(0)), sceneSpace);
    expect(recorder.recordedFrameCount).toBe(1);
  });
});

describe('scoped CloudXR replay frame', () => {
  test('preserves live tracking outside replay without changing browser frames', () => {
    const recorder = new XRInputRecorder();
    const frame = makeFrame();
    const originalGetPose = frame.getPose;
    recorder.startRecording();
    const adapted = recorder.adaptTrackingFrame(frame);
    expect(adapted.session.inputSources).toBe(frame.session.inputSources);
    expect(adapted.getViewerPose(sceneSpace)).toEqual(frame.getViewerPose(sceneSpace));
    expect(frame.getPose).toBe(originalGetPose);
    recorder.stopRecording();
  });

  test('transforms recorded grip and aim from scene space into the requested base space', () => {
    const grip = {} as XRSpace;
    const aim = {} as XRSpace;
    const cloudSpace = {} as XRSpace;
    const source = {
      handedness: 'left',
      gripSpace: grip,
      targetRaySpace: aim,
      gamepad: gamepad(99),
    } as XRInputSource;
    const quarterTurn = Math.sqrt(0.5);
    const frame = makeFrame([source], (space, base) => {
      if (space === sceneSpace && base === cloudSpace) {
        return pose(10, 0, 0, { z: quarterTurn, w: quarterTurn });
      }
      return pose(99);
    });
    const recorder = new XRInputRecorder();
    recorder.startReplay(recording(frameData(1)));
    recorder.beginFrame(frame, sceneSpace);

    const adapted = recorder.adaptTrackingFrame(frame);
    const replayedGrip = adapted.getPose(grip, cloudSpace)!;
    const replayedAim = adapted.getPose(aim, cloudSpace)!;

    expect(frame.getPose(grip, cloudSpace)?.transform.position.x).toBe(99);
    expect(replayedGrip.transform.position.x).toBeCloseTo(10);
    expect(replayedGrip.transform.position.y).toBeCloseTo(1);
    expect(replayedAim.transform.position.y).toBeCloseTo(2);
    expect(replayedGrip.transform.orientation.z).toBeCloseTo(quarterTurn);
  });

  test('replays gamepads and joints without changing the real input source', () => {
    const grip = {} as XRSpace;
    const wrist = {} as XRJointSpace;
    const cloudSpace = {} as XRSpace;
    const source = {
      handedness: 'left',
      gripSpace: grip,
      targetRaySpace: {} as XRSpace,
      gamepad: gamepad(99),
      hand: new Map([['wrist', wrist]]),
    } as unknown as XRInputSource;
    const frame = makeFrame(
      [source],
      (space, base) => (space === sceneSpace && base === cloudSpace ? pose(10) : null),
      () => jointPose(99)
    );
    const recorder = new XRInputRecorder();
    recorder.startReplay(recording(frameData(3)));
    recorder.beginFrame(frame, sceneSpace);

    const adapted = recorder.adaptTrackingFrame(frame);
    const replaySources = adapted.session.inputSources;
    expect(source.gamepad?.axes).toEqual([99]);
    expect(adapted.session.inputSources).toBe(replaySources);
    expect(replaySources[0].gamepad).toBe(replaySources[0].gamepad);
    expect(replaySources[0].gamepad?.axes).toEqual([3]);
    expect(adapted.getJointPose?.(wrist, cloudSpace)?.transform.position.x).toBe(15);
    expect(adapted.getJointPose?.(wrist, cloudSpace)?.radius).toBe(0.02);
  });

  test('delegates unknown spaces and joints to the real frame', () => {
    const unknownSpace = {} as XRSpace;
    const unknownJoint = {} as XRJointSpace;
    const frame = makeFrame(
      [],
      () => pose(7),
      () => jointPose(8)
    );
    const recorder = new XRInputRecorder();
    recorder.startReplay(recording(frameData()));
    recorder.beginFrame(frame, sceneSpace);
    const adapted = recorder.adaptTrackingFrame(frame);

    expect(adapted.getPose(unknownSpace, sceneSpace)?.transform.position.x).toBe(7);
    expect(adapted.getJointPose?.(unknownJoint, sceneSpace)?.transform.position.x).toBe(8);
  });

  test('aligns loaded poses from the recorded viewer origin to the current XR session', () => {
    const grip = {} as XRSpace;
    const source = {
      handedness: 'left',
      gripSpace: grip,
      targetRaySpace: {} as XRSpace,
    } as XRInputSource;
    const session = makeSession([source]);
    const loaded = XRInputRecorder.importJSON(JSON.stringify(calibratedRecording(1, frameData(2))));
    const frame = makeFrame([source], undefined, undefined, 0, pose(11, 5), session);
    const recorder = new XRInputRecorder();
    recorder.startReplay(loaded, true, 'frame');
    recorder.beginFrame(frame, sceneSpace);
    expect(recorder.replayNeedsCalibration).toBe(true);
    recorder.calibrateReplay();
    recorder.beginFrame(frame, sceneSpace);

    const replayed = recorder.adaptTrackingFrame(frame).getPose(grip, sceneSpace);
    expect(replayed?.transform.position.x).toBeCloseTo(12);
    expect(replayed?.transform.position.y).toBeCloseTo(5);
  });

  test('applies the calibrated heading to replayed poses', () => {
    const grip = {} as XRSpace;
    const source = {
      handedness: 'left',
      gripSpace: grip,
      targetRaySpace: {} as XRSpace,
    } as XRInputSource;
    const session = makeSession([source]);
    const sample = frameData();
    sample.poses.leftGrip = { px: 0, py: 0, pz: -1, ox: 0, oy: 0, oz: 0, ow: 1 };
    const loaded = XRInputRecorder.importJSON(JSON.stringify(calibratedRecording(0, sample)));
    const quarterTurn = Math.sqrt(0.5);
    const frame = makeFrame(
      [source],
      undefined,
      undefined,
      0,
      pose(0, 0, 0, { y: quarterTurn, w: quarterTurn }),
      session
    );
    const recorder = new XRInputRecorder();
    recorder.startReplay(loaded, true, 'frame');
    recorder.beginFrame(frame, sceneSpace);
    expect(recorder.replayNeedsCalibration).toBe(true);
    recorder.calibrateReplay();
    recorder.beginFrame(frame, sceneSpace);

    const replayed = recorder.adaptTrackingFrame(frame).getPose(grip, sceneSpace);
    expect(replayed?.transform.position.x).toBeCloseTo(-1);
    expect(replayed?.transform.position.z).toBeCloseTo(0);
    expect(replayed?.transform.orientation.y).toBeCloseTo(quarterTurn);
    expect(replayed?.transform.orientation.w).toBeCloseTo(quarterTurn);
  });

  test('freezes calibration within a session and recalibrates for a new session', () => {
    const grip = {} as XRSpace;
    const source = {
      handedness: 'left',
      gripSpace: grip,
      targetRaySpace: {} as XRSpace,
    } as XRInputSource;
    const firstSession = makeSession([source]);
    const secondSession = makeSession([source]);
    const loaded = XRInputRecorder.importJSON(JSON.stringify(calibratedRecording(1, frameData(2))));
    const recorder = new XRInputRecorder();

    const firstFrame = makeFrame([source], undefined, undefined, 0, pose(11), firstSession);
    recorder.startReplay(loaded, true, 'frame');
    recorder.beginFrame(firstFrame, sceneSpace);
    expect(recorder.replayNeedsCalibration).toBe(true);
    recorder.calibrateReplay();
    recorder.beginFrame(firstFrame, sceneSpace);
    expect(
      recorder.adaptTrackingFrame(firstFrame).getPose(grip, sceneSpace)?.transform.position.x
    ).toBeCloseTo(12);
    recorder.stopReplay();

    const movedViewerFrame = makeFrame([source], undefined, undefined, 1, pose(21), firstSession);
    recorder.startReplay(loaded, true, 'frame');
    recorder.beginFrame(movedViewerFrame, sceneSpace);
    expect(
      recorder.adaptTrackingFrame(movedViewerFrame).getPose(grip, sceneSpace)?.transform.position.x
    ).toBeCloseTo(12);
    recorder.stopReplay();

    const newSessionFrame = makeFrame([source], undefined, undefined, 2, pose(21), secondSession);
    recorder.startReplay(loaded, true, 'frame');
    recorder.beginFrame(newSessionFrame, sceneSpace);
    expect(recorder.replayNeedsCalibration).toBe(true);
    recorder.calibrateReplay();
    recorder.beginFrame(newSessionFrame, sceneSpace);
    expect(
      recorder.adaptTrackingFrame(newSessionFrame).getPose(grip, sceneSpace)?.transform.position.x
    ).toBeCloseTo(22);
  });

  test('keeps in-memory recording and replay in the same reference space unchanged', () => {
    const grip = {} as XRSpace;
    const source = {
      handedness: 'left',
      gripSpace: grip,
      targetRaySpace: {} as XRSpace,
    } as XRInputSource;
    const session = makeSession([source]);
    const recordingFrame = makeFrame(
      [source],
      space => (space === grip ? pose(2) : null),
      undefined,
      0,
      pose(1),
      session
    );
    const recorder = new XRInputRecorder();
    recorder.startRecording();
    recorder.beginFrame(recordingFrame, sceneSpace);
    recorder.stopRecording();
    const saved = recorder.getRecording();

    const replayFrame = makeFrame([source], undefined, undefined, 1, pose(11), session);
    recorder.startReplay(saved, true, 'frame');
    recorder.beginFrame(replayFrame, sceneSpace);

    const replayed = recorder.adaptTrackingFrame(replayFrame).getPose(grip, sceneSpace);
    expect(replayed?.transform.position.x).toBeCloseTo(2);
  });
});

describe('explicit replay calibration and reference-space resets', () => {
  function setup(pacing: 'frame' | 'time' = 'frame') {
    const grip = {} as XRSpace;
    const wrist = {} as XRJointSpace;
    const source = {
      handedness: 'left',
      gripSpace: grip,
      targetRaySpace: {} as XRSpace,
      gamepad: gamepad(1),
      hand: new Map([['wrist', wrist]]),
    } as unknown as XRInputSource;
    const session = makeSession([source]);
    const space = new EventTarget() as XRReferenceSpace;
    const recorder = new XRInputRecorder();
    const saved = calibratedRecording(0, timedFrame(0, 1), timedFrame(100, 3));
    const frame = (viewerX: number, time = 0, viewer: XRPose | null = pose(viewerX)) =>
      makeFrame(
        [source],
        () => pose(999),
        () => jointPose(999),
        time,
        viewer,
        session
      );
    const advance = (viewerX: number, time = 0) => {
      const current = frame(viewerX, time);
      recorder.beginFrame(current, space);
      return recorder.adaptTrackingFrame(current);
    };
    const reset = (transform: XRRigidTransform | null) => {
      space.dispatchEvent(Object.assign(new Event('reset'), { transform }));
    };
    recorder.startReplay(saved, false, pacing);
    return { recorder, grip, wrist, source, session, space, saved, frame, advance, reset };
  }

  test('suppresses live input and waits for a deliberate calibration with a valid viewer pose', () => {
    const { recorder, grip, wrist, space, frame, advance } = setup();
    const waiting = advance(10);
    expect(recorder.replayNeedsCalibration).toBe(true);
    expect(recorder.currentFrame).toBeNull();
    expect(waiting.getPose(grip, space)).toBeUndefined();
    expect(waiting.getJointPose?.(wrist, space)).toBeUndefined();
    expect(waiting.session.inputSources).toHaveLength(0);

    recorder.calibrateReplay();
    recorder.beginFrame(frame(10, 0, null), space);
    expect(recorder.replayNeedsCalibration).toBe(true);
    expect(recorder.replayFrameIndex).toBe(0);
    expect(advance(10).getPose(grip, space)?.transform.position.x).toBeCloseTo(11);
    expect(recorder.replayNeedsCalibration).toBe(false);
  });

  test.each(['frame', 'time'] as const)(
    '%s replay preserves world placement after headset motion and repeated translation resets',
    pacing => {
      const { recorder, grip, wrist, space, advance, reset } = setup(pacing);
      advance(10);
      recorder.calibrateReplay();
      expect(advance(10).getPose(grip, space)?.transform.position.x).toBeCloseTo(11);
      reset(new XRRigidTransform({ x: -100 }));
      const firstReset = advance(112, pacing === 'time' ? 50 : 1);
      expect(recorder.replayNeedsCalibration).toBe(false);
      expect(firstReset.getPose(grip, space)?.transform.position.x).toBeCloseTo(
        pacing === 'time' ? 112 : 113
      );
      expect(firstReset.getJointPose?.(wrist, space)?.transform.position.x).toBeCloseTo(
        pacing === 'time' ? 114 : 115
      );
      reset(new XRRigidTransform({ x: -20 }));
      expect(advance(132, 100).getPose(grip, space)?.transform.position.x).toBeCloseTo(133);
    }
  );

  test('uses the inverse reset rotation for position and orientation', () => {
    const { recorder, grip, space, saved, advance, reset } = setup();
    saved.frames = [saved.frames[0]];
    recorder.stopReplay();
    recorder.startReplay(saved, false, 'frame');
    advance(10);
    recorder.calibrateReplay();
    advance(10);
    const q = Math.sqrt(0.5);
    reset(new XRRigidTransform({}, { x: 0, y: q, z: 0, w: q }));
    const result = advance(999).getPose(grip, space)!;
    expect(result.transform.position.x).toBeCloseTo(0);
    expect(result.transform.position.z).toBeCloseTo(11);
    expect(result.transform.orientation.y).toBeCloseTo(-q);
    expect(result.transform.orientation.w).toBeCloseTo(q);
  });

  test('unknown reset pauses the replay clock and requires another explicit calibration', () => {
    const { recorder, grip, space, advance, reset } = setup('time');
    advance(10, 1000);
    recorder.calibrateReplay();
    advance(10, 1000);
    expect(advance(12, 1050).getPose(grip, space)?.transform.position.x).toBeCloseTo(12);
    reset(null);
    expect(advance(112, 2000).getPose(grip, space)).toBeUndefined();
    expect(recorder.replayNeedsCalibration).toBe(true);
    expect(recorder.currentFrame).toBeNull();
    recorder.calibrateReplay();
    expect(advance(110, 5000).getPose(grip, space)?.transform.position.x).toBeCloseTo(112);
    expect(recorder.currentFrame?.timeMs).toBe(50);
  });

  test('preserves cached placement when a known reset occurs while replay is stopped', () => {
    const { recorder, grip, space, saved, advance, reset } = setup();
    advance(10);
    recorder.calibrateReplay();
    advance(10);
    recorder.stopReplay();
    reset(new XRRigidTransform({ x: -100 }));
    recorder.startReplay(saved, false, 'frame');
    expect(advance(112).getPose(grip, space)?.transform.position.x).toBeCloseTo(111);
    expect(recorder.replayNeedsCalibration).toBe(false);
  });

  test('calibration after a reset is stored in the original reference frame', () => {
    const { recorder, grip, space, saved, advance, reset } = setup();
    advance(10);
    reset(new XRRigidTransform({ x: -100 }));
    recorder.calibrateReplay();
    expect(advance(110).getPose(grip, space)?.transform.position.x).toBeCloseTo(111);
    recorder.stopReplay();
    recorder.startReplay(saved, false, 'frame');
    expect(advance(115).getPose(grip, space)?.transform.position.x).toBeCloseTo(111);
  });

  test('reference-space replacement invalidates placement and pending calibration', () => {
    const { recorder, grip, frame, advance } = setup();
    advance(10);
    recorder.calibrateReplay();
    const replacement = new EventTarget() as XRReferenceSpace;
    const current = frame(20);
    recorder.beginFrame(current, replacement);
    expect(recorder.replayNeedsCalibration).toBe(true);
    expect(recorder.adaptTrackingFrame(current).getPose(grip, replacement)).toBeUndefined();
    recorder.calibrateReplay();
    recorder.beginFrame(current, replacement);
    expect(
      recorder.adaptTrackingFrame(current).getPose(grip, replacement)?.transform.position.x
    ).toBeCloseTo(21);
  });

  test.each([true, false])('stops recording on reset (known transform: %s)', known => {
    const { recorder, grip, space, frame, reset } = setup();
    recorder.stopReplay();
    recorder.startRecording();
    recorder.beginFrame(frame(0), space);
    reset(known ? new XRRigidTransform({ x: -100 }) : null);
    expect(recorder.mode).toBe('idle');
    expect(recorder.recordingInterrupted).toBe(true);
    recorder.beginFrame(frame(100), space);
    expect(recorder.getRecording().frames).toHaveLength(1);
    const saved = recorder.getRecording();
    recorder.startReplay(saved, false, 'frame');
    recorder.beginFrame(frame(100), space);
    expect(recorder.replayNeedsCalibration).toBe(!known);
    if (known) {
      expect(
        recorder.adaptTrackingFrame(frame(100)).getPose(grip, space)?.transform.position.x
      ).toBeCloseTo(1099);
    }
  });
});

describe('recorded hands without live tracking', () => {
  function benchmark(pacing: 'frame' | 'time') {
    const session = makeSession();
    const recorder = new XRInputRecorder();
    const frame = (sources: XRInputSource[] = [], time = 0) =>
      makeFrame(
        sources,
        () => null,
        () => null,
        time,
        pose(0),
        session
      );
    // CloudXR initializes its active-hand set on the first tracking frame and
    // subsequently refreshes it from inputsourceschange, not from joint poses.
    const active = new Set<XRHandedness>();
    const refresh = (current: XRSession) => {
      active.clear();
      for (const source of current.inputSources) {
        if (source.hand) active.add(source.handedness);
      }
    };
    const idle = recorder.adaptTrackingFrame(frame());
    refresh(idle.session);
    const changed = jest.fn((event: XRInputSourcesChangeEvent) => refresh(event.session));
    idle.session.addEventListener('inputsourceschange', changed);
    const read = (current: XRFrame) => {
      recorder.beginFrame(current, sceneSpace);
      const adapted = recorder.adaptTrackingFrame(current);
      const source = Array.from(adapted.session.inputSources).find(s => s.handedness === 'left');
      if (!source?.hand || !active.has(source.handedness)) return null;
      return adapted.getJointPose?.(source.hand.get('wrist')!, sceneSpace)?.transform.position.x;
    };
    recorder.startReplay(recording(timedFrame(0, 1), timedFrame(100, 3)), false, pacing);
    return { recorder, session, frame, active, changed, read, idle };
  }

  test.each(['frame', 'time'] as const)(
    '%s replay advances when hands were never detected',
    pacing => {
      const { read, frame, active, changed, session } = benchmark(pacing);
      expect(read(frame())).toBeCloseTo(3);
      expect(active.has('left')).toBe(true);
      expect(read(frame([], pacing === 'time' ? 50 : 1))).toBeCloseTo(pacing === 'time' ? 4 : 5);
      expect(changed).toHaveBeenCalledTimes(1);
      expect(session.inputSources).toHaveLength(0);
    }
  );

  test('live hands appearing and disappearing do not gate or replace recorded hands', () => {
    const { read, frame, changed, session, active, recorder } = benchmark('time');
    expect(read(frame())).toBeCloseTo(3);
    const live = {
      handedness: 'left',
      hand: new Map([['wrist', {}]]),
      targetRaySpace: {},
    } as unknown as XRInputSource;
    const visible = frame([live], 25);
    session.dispatchEvent(
      Object.assign(new Event('inputsourceschange'), {
        session,
        added: [live],
        removed: [],
      })
    );
    expect(read(visible)).toBeCloseTo(3.5);
    const absent = frame([], 50);
    session.dispatchEvent(
      Object.assign(new Event('inputsourceschange'), {
        session,
        added: [],
        removed: [live],
      })
    );
    expect(active.has('left')).toBe(true);
    expect(read(absent)).toBeCloseTo(4);
    expect(changed).toHaveBeenCalledTimes(1);
    recorder.stopReplay();
    recorder.adaptTrackingFrame(absent);
    expect(active.size).toBe(0);
    expect(changed).toHaveBeenCalledTimes(2);
  });

  test('does not expose an unrecorded live hand to the SDK during replay', () => {
    const { recorder, read, frame, active } = benchmark('frame');
    const liveRight = {
      handedness: 'right',
      hand: new Map([['wrist', {}]]),
      targetRaySpace: {},
    } as unknown as XRInputSource;
    const current = frame([liveRight]);
    expect(read(current)).toBeCloseTo(3);
    expect([...active]).toEqual(['left']);
    expect(recorder.adaptTrackingFrame(current).session.inputSources).toHaveLength(1);
    expect(current.session.inputSources).toEqual([liveRight]);
  });

  test('recorded tracking loss removes the source, and recorded recovery restores it', () => {
    const { recorder, read, frame, active } = benchmark('frame');
    recorder.stopReplay();
    const lost = timedFrame(50, 2);
    lost.handJoints.left = { wrist: null };
    recorder.startReplay(recording(timedFrame(0, 1), lost, timedFrame(100, 3)), false, 'frame');
    expect(read(frame())).toBeCloseTo(3);
    expect(read(frame())).toBeNull();
    expect(active.size).toBe(0);
    expect(read(frame())).toBeCloseTo(5);
  });

  test('forwards native events outside replay and supports listener cleanup', () => {
    const { recorder, session, frame, changed, idle } = benchmark('frame');
    recorder.stopReplay();
    recorder.adaptTrackingFrame(frame());
    session.dispatchEvent(
      Object.assign(new Event('inputsourceschange'), {
        session,
        added: [],
        removed: [],
      })
    );
    expect(changed).toHaveBeenCalledTimes(1);
    expect(changed.mock.calls[0][0].session).toBe(idle.session);
    idle.session.removeEventListener('inputsourceschange', changed);
    session.dispatchEvent(
      Object.assign(new Event('inputsourceschange'), {
        session,
        added: [],
        removed: [],
      })
    );
    expect(changed).toHaveBeenCalledTimes(1);
    recorder.dispose();
  });
});

describe('controller replay without live hardware', () => {
  function controllerFrame(timeMs: number, x: number, profiles?: string[]): RecordedFrame {
    const sample = timedFrame(timeMs, x);
    sample.handJoints.left = {};
    if (profiles) sample.controllerProfiles = { left: profiles, right: null };
    return sample;
  }

  test.each(['time', 'frame'] as const)(
    'replays poses and controls without live sources (%s)',
    pacing => {
      const recorder = new XRInputRecorder();
      const session = makeSession();
      const frame = (time: number) => makeFrame([], undefined, undefined, time, pose(0), session);
      const scoped = recorder.adaptTrackingFrame(frame(0)).session;
      const active = new Set<XRHandedness>();
      const changed = jest.fn((event: XRInputSourcesChangeEvent) => {
        active.clear();
        for (const source of event.session.inputSources) {
          if (source.gamepad && !source.hand) active.add(source.handedness);
        }
      });
      scoped.addEventListener('inputsourceschange', changed);
      recorder.startReplay(
        recording(controllerFrame(0, 0), controllerFrame(100, 1)),
        false,
        pacing
      );
      let firstSource: XRInputSource | undefined;
      for (const [time, expected] of [
        [0, 0],
        [50, pacing === 'time' ? 0.5 : 1],
        [100, 1],
      ]) {
        const current = frame(time);
        recorder.beginFrame(current, sceneSpace);
        const adapted = recorder.adaptTrackingFrame(current);
        const source = adapted.session.inputSources[0];
        firstSource ??= source;
        expect(source).toBe(firstSource);
        expect(active.has('left')).toBe(true);
        expect(source.hand).toBeUndefined();
        expect(source.profiles).toEqual(['generic-trigger-squeeze-thumbstick']);
        expect(adapted.getPose(source.gripSpace!, sceneSpace)?.transform.position.x).toBe(expected);
        expect(adapted.getPose(source.targetRaySpace, sceneSpace)?.transform.position.x).toBe(
          expected + 1
        );
        expect(source.gamepad?.axes).toEqual([expected]);
        expect(source.gamepad?.buttons[0]).toEqual({
          value: expected,
          pressed: true,
          touched: true,
        });
        expect(source.gamepad).toBe(source.gamepad);
        expect(session.inputSources).toHaveLength(0);
      }
      expect(changed).toHaveBeenCalledTimes(1);
      recorder.stopReplay();
      recorder.adaptTrackingFrame(frame(100));
      expect(active.size).toBe(0);
      expect(changed).toHaveBeenCalledTimes(2);
      recorder.dispose();
    }
  );

  test('keeps recorded profiles and controls through live controller connection changes', () => {
    const recorder = new XRInputRecorder();
    const session = makeSession();
    const changed = jest.fn();
    const initial = makeFrame([], undefined, undefined, 0, pose(0), session);
    recorder.adaptTrackingFrame(initial).session.addEventListener('inputsourceschange', changed);
    const profiles = ['meta-quest-touch-plus', 'oculus-touch-v3'];
    recorder.startReplay(
      recording(controllerFrame(0, 1, profiles), controllerFrame(100, 3, profiles)),
      false
    );
    recorder.beginFrame(initial, sceneSpace);
    const source = recorder.adaptTrackingFrame(initial).session.inputSources[0];
    const live = {
      handedness: 'left',
      profiles: ['pico-4u'],
      gamepad: gamepad(99),
    } as XRInputSource;
    for (const [time, inputs] of [
      [50, [live]],
      [100, []],
    ] as const) {
      const frame = makeFrame([...inputs], undefined, undefined, time, pose(0), session);
      session.dispatchEvent(
        Object.assign(new Event('inputsourceschange'), { session, added: inputs, removed: [] })
      );
      recorder.beginFrame(frame, sceneSpace);
      expect(recorder.adaptTrackingFrame(frame).session.inputSources).toEqual([source]);
      expect(source.profiles).toEqual(profiles);
      expect(source.gamepad?.axes[0]).toBe(time === 50 ? 2 : 3);
    }
    expect(changed).toHaveBeenCalledTimes(1);
  });

  test('round-trips profiles and replays a controller alongside a recorded hand', () => {
    const recorder = new XRInputRecorder();
    const profiles = ['pico-4u', 'oculus-touch-v2'];
    const controller = {
      handedness: 'left',
      profiles,
      gamepad: gamepad(0.75),
      gripSpace: {},
      targetRaySpace: {},
    } as XRInputSource;
    const hand = {
      handedness: 'right',
      hand: new Map([['wrist', {}]]),
      targetRaySpace: {},
    } as unknown as XRInputSource;
    recorder.startRecording();
    recorder.beginFrame(
      makeFrame(
        [controller, hand],
        () => pose(2),
        () => jointPose(3)
      ),
      sceneSpace
    );
    recorder.stopRecording();
    const saved = XRInputRecorder.importJSON(recorder.exportJSON());
    expect(saved.frames[0].controllerProfiles).toEqual({ left: profiles, right: null });
    delete saved.calibration;
    recorder.startReplay(saved);
    const frame = makeFrame();
    recorder.beginFrame(frame, sceneSpace);
    const sources = Array.from(recorder.adaptTrackingFrame(frame).session.inputSources);
    expect(sources.find(source => source.handedness === 'left')?.profiles).toEqual(profiles);
    expect(sources.find(source => source.handedness === 'right')?.hand).toBeDefined();
  });

  test('preserves gaps and device switches at recorded timestamps', () => {
    const recorder = new XRInputRecorder();
    const first = controllerFrame(0, 1, ['meta-quest-touch-plus']);
    const other = controllerFrame(100, 9, ['pico-4u']);
    const lost = controllerFrame(200, 10);
    lost.gamepads.left = null;
    lost.poses.leftGrip = lost.poses.leftAim = null;
    recorder.startReplay(
      recording(first, other, lost, timedFrame(300, 20), controllerFrame(400, 30)),
      false
    );
    const sources: XRInputSource[] = [];
    for (const time of [0, 50, 100, 200, 300, 400]) {
      const frame = makeFrame([], undefined, undefined, time);
      recorder.beginFrame(frame, sceneSpace);
      const adapted = recorder.adaptTrackingFrame(frame);
      const source = adapted.session.inputSources[0];
      if (time === 200) {
        expect(adapted.session.inputSources).toHaveLength(0);
      } else {
        sources.push(source);
        expect(adapted.session.inputSources).toHaveLength(1);
        expect(!!source.hand).toBe(time === 300);
        expect(source.gamepad?.axes[0]).toBe(
          time < 100 ? 1 : time === 100 ? 9 : time === 300 ? 20 : 30
        );
      }
    }
    expect(sources[0]).toBe(sources[1]);
    expect(sources[2]).not.toBe(sources[0]);
    expect(sources[2].profiles).toEqual(['pico-4u']);
  });

  test('waits for calibration and restores the live controller on stop', () => {
    const recorder = new XRInputRecorder();
    const live = { handedness: 'left', gamepad: gamepad(99) } as XRInputSource;
    const frame = makeFrame([live]);
    recorder.startReplay(calibratedRecording(0, controllerFrame(0, 1)));
    recorder.beginFrame(frame, sceneSpace);
    expect(recorder.adaptTrackingFrame(frame).session.inputSources).toHaveLength(0);
    recorder.calibrateReplay();
    recorder.beginFrame(frame, sceneSpace);
    expect(recorder.adaptTrackingFrame(frame).session.inputSources[0].gamepad?.axes).toEqual([1]);
    recorder.stopReplay();
    expect(recorder.adaptTrackingFrame(frame).session.inputSources[0]).toBe(live);
  });

  test.each([null, {}, { left: 'quest', right: null }, { left: [42], right: null }])(
    'rejects malformed controller profiles (%j)',
    controllerProfiles => {
      expect(() =>
        XRInputRecorder.importJSON(
          JSON.stringify({ version: 1, frames: [{ ...frameData(), controllerProfiles }] })
        )
      ).toThrow('controllerProfiles is invalid');
    }
  );
});

describe('serialization', () => {
  test('round-trips version 1 recordings', () => {
    const recorder = new XRInputRecorder();
    recorder.startRecording();
    recorder.beginFrame(makeFrame(), sceneSpace);
    recorder.stopRecording();
    const imported = XRInputRecorder.importJSON(recorder.exportJSON());
    expect(imported.frames).toHaveLength(1);
    expect(imported.calibration?.mode).toBe('viewer-start-yaw');
    expect(typeof recorder.getRecording().recordedAt).toBe('number');
  });

  test('records relative XR frame timing', () => {
    const recorder = new XRInputRecorder();
    recorder.startRecording();
    recorder.beginFrame(makeFrame([], undefined, undefined, 100), sceneSpace);
    recorder.beginFrame(makeFrame([], undefined, undefined, 116.5), sceneSpace);
    recorder.stopRecording();

    expect(recorder.getRecording().frames.map(frame => frame.timeMs)).toEqual([0, 16.5]);
  });

  test('records finite monotonic timeMs when predictedDisplayTime is missing', () => {
    // PICO leaves predictedDisplayTime undefined; without a fallback timeMs was
    // NaN -> serialized to null -> rejected on import.
    const noTime = undefined as unknown as number;
    const recorder = new XRInputRecorder();
    recorder.startRecording();
    recorder.beginFrame(makeFrame([], undefined, undefined, noTime), sceneSpace);
    recorder.beginFrame(makeFrame([], undefined, undefined, noTime), sceneSpace);
    recorder.stopRecording();

    const times = recorder.getRecording().frames.map(frame => frame.timeMs);
    expect(times.every(t => Number.isFinite(t) && t >= 0)).toBe(true);
    expect(times[1]).toBeGreaterThanOrEqual(times[0]);
    expect(() => XRInputRecorder.importJSON(recorder.exportJSON())).not.toThrow();
  });

  test.each([
    { version: 2, frames: [] },
    { version: 1, frames: null },
    { version: 1, frames: [{ ...frameData(), timeMs: undefined }] },
    { version: 1, frames: [frameData(2), frameData(1)] },
    { version: 1, calibration: { mode: 'viewer-start-yaw', pose: {} }, frames: [] },
    { version: 1, calibration: { mode: 'unknown', pose: {} }, frames: [] },
    null,
    42,
    [],
  ])('rejects incompatible or malformed recordings', value => {
    expect(() => XRInputRecorder.importJSON(JSON.stringify(value))).toThrow();
  });

  test('rejects non-JSON input with a clear message', () => {
    expect(() => XRInputRecorder.importJSON('not json {')).toThrow('File is not valid JSON');
  });

  test('returns a recording snapshot', () => {
    const recorder = new XRInputRecorder();
    recorder.startRecording();
    recorder.beginFrame(makeFrame(), sceneSpace);
    recorder.stopRecording();
    const snapshot = recorder.getRecording();
    recorder.startRecording();
    recorder.beginFrame(makeFrame(), sceneSpace);
    recorder.beginFrame(makeFrame(), sceneSpace);
    expect(snapshot.frames).toHaveLength(1);
  });

  test('helper can create recordings for replay tests', () => {
    expect(recordFrames(3).frames).toHaveLength(3);
  });
});
