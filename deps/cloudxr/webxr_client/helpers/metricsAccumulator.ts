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

/**
 * MetricsAccumulator - last-known value per CloudXR metric, grouped by cadence.
 *
 * The in-XR HUD only needs a handful of metrics, but the OOB teleop hub wants
 * everything the SDK reports. The SDK's `onMetrics` callbacks are *partial*: one
 * tick may carry only `RenderFramerate`, the next only `PoseSendFramerate`.
 * Replacing the stored record on each tick would make metrics flicker in and out
 * of hub snapshots, so this merges instead, and `HeadsetControlChannel` polls the
 * merged state on its own timer (default 500 ms).
 *
 * Keys are the SDK's `CloudXR.MetricsName` string values, so the hub and any
 * downstream dashboards see the same names the SDK documents. The hub side
 * (`oob_teleop_hub.py`) stores metrics as an arbitrary `{str: float}` map per
 * cadence, so new metric names flow through without a hub change.
 */

import * as CloudXR from '@nvidia/cloudxr';

import type { MetricsSnapshot } from './controlChannel';
import type {
  FrameMetricsUpdate,
  NetworkMetricsUpdate,
  RenderMetricsUpdate,
} from './metricsUpdates';

/** Cadence labels used in `clientMetrics` payloads sent to the OOB hub. */
export type MetricsCadenceLabel = 'render' | 'frame' | 'network';

/** One cadence's worth of metrics, ready to place in a `clientMetrics` payload. */
export type { MetricsSnapshot } from './controlChannel';

/**
 * Copies the defined entries of `update` into `target` under their SDK metric names.
 *
 * @param target - Accumulated metrics for one cadence; mutated in place.
 * @param update - Partial callback payload; `undefined` fields are left untouched.
 * @param names - Maps each update field to the `CloudXR.MetricsName` to store it under.
 */
function mergeDefined<T extends object>(
  target: Record<string, number>,
  update: T,
  names: { [K in keyof T]-?: string }
): boolean {
  let observed = false;
  for (const key of Object.keys(names) as Array<keyof T>) {
    const value = update[key];
    // Guard on the value, not the key: absent and explicitly-undefined must behave alike.
    if (typeof value === 'number' && Number.isFinite(value)) {
      target[names[key]] = value;
      observed = true;
    }
  }
  return observed;
}

/** Field-to-MetricsName maps, one per cadence. Kept beside the interfaces they mirror. */
const RENDER_NAMES: { [K in keyof Required<RenderMetricsUpdate>]: string } = {
  framerate: CloudXR.MetricsName.RenderFramerate,
  sendFramerate: CloudXR.MetricsName.PoseSendFramerate,
  xrPoseAgeMs: CloudXR.MetricsName.LatencyXrPoseAgeMs,
};

const FRAME_NAMES: { [K in keyof Required<FrameMetricsUpdate>]: string } = {
  framerate: CloudXR.MetricsName.StreamingFramerate,
  frameCount: CloudXR.MetricsName.StreamingFrameCount,
  poseToRenderTimeMs: CloudXR.MetricsName.PoseToRenderTime,
  poseUploadMs: CloudXR.MetricsName.LatencyPoseUploadMs,
  poseToFrameReceivedMs: CloudXR.MetricsName.LatencyPoseToFrameReceivedMs,
  compositorSkippedPercent: CloudXR.MetricsName.FramePipelineCompositorSkippedPercent,
  outOfOrderPercent: CloudXR.MetricsName.FramePipelineOutOfOrderPercent,
  mismatchedPercent: CloudXR.MetricsName.FramePipelineMismatchedPercent,
};

const NETWORK_NAMES: { [K in keyof Required<NetworkMetricsUpdate>]: string } = {
  streamingRateMbps: CloudXR.MetricsName.NetworkStreamingRateMbps,
  availableBandwidthMbps: CloudXR.MetricsName.NetworkAvailableBandwidthMbps,
  rttMs: CloudXR.MetricsName.NetworkRttMs,
  packetLoss: CloudXR.MetricsName.NetworkPacketLoss,
  avgDecodeTimeMs: CloudXR.MetricsName.NetworkAvgDecodeTimeMs,
  qualityScore: CloudXR.MetricsName.NetworkQualityScore,
  bandwidthScore: CloudXR.MetricsName.NetworkBandwidthScore,
  lossScore: CloudXR.MetricsName.NetworkLossScore,
  latencyScore: CloudXR.MetricsName.NetworkLatencyScore,
  sessionQuality: CloudXR.MetricsName.SessionQuality,
};

export class MetricsAccumulator {
  private readonly byCadence: Record<MetricsCadenceLabel, Record<string, number>> = {
    render: {},
    frame: {},
    network: {},
  };
  private readonly observedAt: Partial<Record<MetricsCadenceLabel, number>> = {};

  /** Merge a PerRender callback payload. */
  recordRender(update: RenderMetricsUpdate): void {
    if (mergeDefined(this.byCadence.render, update, RENDER_NAMES)) {
      this.observedAt.render = performance.now();
    }
  }

  /** Merge a PerFrame callback payload. */
  recordFrame(update: FrameMetricsUpdate): void {
    if (mergeDefined(this.byCadence.frame, update, FRAME_NAMES)) {
      this.observedAt.frame = performance.now();
    }
  }

  /** Merge a PerNetwork callback payload. */
  recordNetwork(update: NetworkMetricsUpdate): void {
    if (mergeDefined(this.byCadence.network, update, NETWORK_NAMES)) {
      this.observedAt.network = performance.now();
    }
  }

  /**
   * Drop everything recorded so far.
   *
   * Called when a session ends so the next session's hub snapshots cannot report
   * the previous session's last-known values as if they were live.
   */
  reset(): void {
    for (const cadence of Object.keys(this.byCadence) as MetricsCadenceLabel[]) {
      this.byCadence[cadence] = {};
      delete this.observedAt[cadence];
    }
  }

  /**
   * Snapshot the accumulated metrics for transmission.
   *
   * @returns One entry per cadence that has at least one metric; empty cadences
   *   are omitted so the hub is not sent empty `clientMetrics` frames.
   */
  takeSnapshot(): MetricsSnapshot[] {
    const snapshots: MetricsSnapshot[] = [];
    for (const cadence of Object.keys(this.byCadence) as MetricsCadenceLabel[]) {
      const metrics = this.byCadence[cadence];
      if (Object.keys(metrics).length === 0) continue;
      // Copy: the caller serializes asynchronously, and callbacks keep mutating.
      snapshots.push({ cadence, metrics: { ...metrics }, observedAt: this.observedAt[cadence] });
    }
    return snapshots;
  }
}
