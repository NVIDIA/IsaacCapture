/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
import { XRPresence } from './xrPresence';

function fixture() {
  const session = { inputSources: [{ handedness: 'left', gripSpace: {} }] };
  const frame = { session, getViewerPose: () => ({}), getPose: () => ({}) } as unknown as XRFrame;
  const presence = new XRPresence();
  presence.setSession(session, 'visible');
  return { presence, session, frame, space: {} as XRReferenceSpace };
}

test('stationary real frames refresh head and left independently; unknown right stays unknown', () => {
  const { presence, frame, space } = fixture();
  presence.observe(frame, space, 10);
  expect(presence.snapshot(20).metrics).toMatchObject({
    headAgeMs: 10,
    leftAgeMs: 10,
    rightAgeMs: -1,
    rightConfigured: 0,
  });
  frame.getViewerPose = () => null;
  presence.observe(frame, space, 30);
  expect(presence.snapshot(40).metrics).toMatchObject({
    frameAgeMs: 10,
    headAgeMs: 30,
    leftAgeMs: 10,
  });
});

test('heartbeat continues without frames and visibility changes invalidate earlier poses', () => {
  const { presence, session, frame, space } = fixture();
  presence.observe(frame, space, 10);
  expect(presence.snapshot(10000).metrics.headAgeMs).toBe(9990);
  presence.setSession(session, 'hidden');
  expect(presence.snapshot(11000).metrics).toMatchObject({
    visibility: 0,
    headAgeMs: -1,
    generation: 2,
  });
  presence.observe(frame, space, 12000);
  expect(presence.snapshot(13000).metrics.frameAgeMs).toBe(-1);
  presence.setSession(session, 'visible');
  expect(presence.snapshot(14000).metrics.headAgeMs).toBe(-1);
});

test('session reset rejects old frames and wall clock skew cannot affect monotonic ages', () => {
  const { presence, frame, space } = fixture();
  const wall = jest.spyOn(Date, 'now').mockReturnValue(-99999999999);
  presence.observe(frame, space, 100);
  expect(presence.snapshot(150).metrics.headAgeMs).toBe(50);
  presence.setSession({}, 'visible');
  presence.observe(frame, space, 200);
  expect(presence.snapshot(250).metrics.headAgeMs).toBe(-1);
  presence.setSession(undefined, undefined);
  expect(presence.snapshot(300).metrics.sessionActive).toBe(0);
  wall.mockRestore();
});
