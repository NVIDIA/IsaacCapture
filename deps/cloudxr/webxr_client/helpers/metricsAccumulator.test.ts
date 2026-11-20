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
 *
 * @jest-environment jsdom
 *
 * Jest only reads this pragma from the file's first comment block, hence its placement
 * here. The CloudXR SDK bundle touches `window` at import time, and this suite imports it
 * (transitively, via metricsAccumulator) to assert against the real MetricsName values
 * rather than a mock that could drift from them.
 */

import * as CloudXR from '@nvidia/cloudxr';

import { MetricsAccumulator } from './metricsAccumulator';

/** Finds one cadence in a snapshot list. */
const cadence = (snapshots: ReturnType<MetricsAccumulator['takeSnapshot']>, name: string) =>
  snapshots.find(s => s.cadence === name);

describe('MetricsAccumulator', () => {
  it('preserves observation time while cached metrics are polled', () => {
    const clock = jest.spyOn(performance, 'now');
    try {
      clock.mockReturnValue(100);
      const acc = new MetricsAccumulator();
      acc.recordRender({ framerate: 72 });
      clock.mockReturnValue(6100);
      acc.recordRender({ framerate: undefined });
      expect(cadence(acc.takeSnapshot(), 'render')).toMatchObject({ observedAt: 100 });

      acc.recordRender({ framerate: 72 });
      expect(cadence(acc.takeSnapshot(), 'render')).toMatchObject({ observedAt: 6100 });
      acc.reset();
      expect(acc.takeSnapshot()).toEqual([]);
    } finally {
      clock.mockRestore();
    }
  });

  it('reports nothing before any metric arrives', () => {
    expect(new MetricsAccumulator().takeSnapshot()).toEqual([]);
  });

  it('merges partial updates instead of replacing them', () => {
    const acc = new MetricsAccumulator();
    // The send-begin and render-begin paths each emit their own PerRender tick.
    acc.recordRender({ framerate: 72 });
    acc.recordRender({ sendFramerate: 60 });

    const render = cadence(acc.takeSnapshot(), 'render');
    expect(render?.metrics).toEqual({
      [CloudXR.MetricsName.RenderFramerate]: 72,
      [CloudXR.MetricsName.PoseSendFramerate]: 60,
    });
  });

  it('overwrites a metric when a later tick carries it again', () => {
    const acc = new MetricsAccumulator();
    acc.recordRender({ framerate: 72 });
    acc.recordRender({ framerate: 71 });

    expect(cadence(acc.takeSnapshot(), 'render')?.metrics).toEqual({
      [CloudXR.MetricsName.RenderFramerate]: 71,
    });
  });

  it('ignores undefined and non-finite values rather than reporting them as zero', () => {
    const acc = new MetricsAccumulator();
    acc.recordRender({ framerate: 72, sendFramerate: undefined });
    acc.recordNetwork({ rttMs: NaN, packetLoss: Infinity });

    expect(cadence(acc.takeSnapshot(), 'render')?.metrics).toEqual({
      [CloudXR.MetricsName.RenderFramerate]: 72,
    });
    // An all-invalid update leaves the cadence empty, so it is omitted entirely.
    expect(cadence(acc.takeSnapshot(), 'network')).toBeUndefined();
  });

  it('keeps a zero value, which is meaningful for loss and quality metrics', () => {
    const acc = new MetricsAccumulator();
    acc.recordNetwork({ packetLoss: 0, sessionQuality: 0 });

    expect(cadence(acc.takeSnapshot(), 'network')?.metrics).toEqual({
      [CloudXR.MetricsName.NetworkPacketLoss]: 0,
      [CloudXR.MetricsName.SessionQuality]: 0,
    });
  });

  it('separates the three cadences and omits empty ones', () => {
    const acc = new MetricsAccumulator();
    acc.recordRender({ framerate: 72 });
    acc.recordNetwork({ sessionQuality: 3 });

    const snapshots = acc.takeSnapshot();
    expect(snapshots.map(s => s.cadence).sort()).toEqual(['network', 'render']);
  });

  it('reports every field of every cadence under its SDK metric name', () => {
    const acc = new MetricsAccumulator();
    acc.recordRender({ framerate: 1, sendFramerate: 2, xrPoseAgeMs: 3 });
    acc.recordFrame({
      framerate: 4,
      frameCount: 5,
      poseToRenderTimeMs: 6,
      poseUploadMs: 7,
      poseToFrameReceivedMs: 8,
      compositorSkippedPercent: 9,
      outOfOrderPercent: 10,
      mismatchedPercent: 11,
    });
    acc.recordNetwork({
      streamingRateMbps: 12,
      availableBandwidthMbps: 13,
      rttMs: 14,
      packetLoss: 15,
      avgDecodeTimeMs: 16,
      qualityScore: 17,
      bandwidthScore: 18,
      lossScore: 19,
      latencyScore: 20,
      sessionQuality: 21,
    });

    const snapshots = acc.takeSnapshot();
    expect(Object.keys(cadence(snapshots, 'render')!.metrics)).toHaveLength(3);
    expect(Object.keys(cadence(snapshots, 'frame')!.metrics)).toHaveLength(8);
    expect(Object.keys(cadence(snapshots, 'network')!.metrics)).toHaveLength(10);
    // Pin the wire names the OOB hub and its dashboards consume, not just the enum
    // reference: renaming a key here is a breaking protocol change, not a refactor.
    expect(cadence(snapshots, 'render')!.metrics['render.framerate']).toBe(1);
    expect(cadence(snapshots, 'render')!.metrics['pose.send_framerate']).toBe(2);
    expect(cadence(snapshots, 'network')!.metrics[CloudXR.MetricsName.SessionQuality]).toBe(21);
  });

  it('snapshots are detached, so later ticks do not mutate an emitted payload', () => {
    const acc = new MetricsAccumulator();
    acc.recordRender({ framerate: 72 });
    const first = cadence(acc.takeSnapshot(), 'render')!.metrics;
    acc.recordRender({ framerate: 30 });

    expect(first[CloudXR.MetricsName.RenderFramerate]).toBe(72);
  });

  it('reset drops everything so a new session cannot inherit stale values', () => {
    const acc = new MetricsAccumulator();
    acc.recordRender({ framerate: 72 });
    acc.recordNetwork({ sessionQuality: 4 });
    acc.reset();

    expect(acc.takeSnapshot()).toEqual([]);
  });
});
