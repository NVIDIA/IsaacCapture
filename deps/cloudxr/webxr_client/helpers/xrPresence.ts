/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
import type { MetricsSnapshot } from './controlChannel';
type Source = 'frame' | 'head' | 'left' | 'right';
/** Observes real XR frames; unknown poses never count as fresh. */
export class XRPresence {
  private session: object | undefined;
  private visibility = -1;
  private generation = 0;
  private observations: Partial<Record<Source, number>> = {};
  private controllers = { left: false, right: false };
  setSession(session: object | undefined, visibility: string | undefined): void {
    const next =
      visibility === 'visible'
        ? 2
        : visibility === 'visible-blurred'
          ? 1
          : visibility === 'hidden'
            ? 0
            : -1;
    if (session !== this.session || next !== this.visibility) {
      this.generation++;
      this.observations = {};
      this.controllers = { left: false, right: false };
    }
    this.session = session;
    this.visibility = next;
  }
  observe(frame: XRFrame, referenceSpace: XRReferenceSpace, now = performance.now()): void {
    // Reject delayed frames from before a session/visibility transition.
    if (frame.session !== this.session || this.visibility !== 2) return;
    this.observations.frame = now;
    if (frame.getViewerPose(referenceSpace)) this.observations.head = now;
    this.controllers = { left: false, right: false };
    for (const source of frame.session.inputSources) {
      const hand = source.handedness;
      if (source.hand || (hand !== 'left' && hand !== 'right')) continue;
      this.controllers[hand] = true;
      if (source.gripSpace && frame.getPose(source.gripSpace, referenceSpace))
        this.observations[hand] = now;
    }
  }
  snapshot(now = performance.now()): MetricsSnapshot {
    const metrics: Record<string, number> = {
      sessionActive: this.session ? 1 : 0,
      visibility: this.visibility,
      generation: this.generation,
      leftConfigured: Number(this.controllers.left),
      rightConfigured: Number(this.controllers.right),
    };
    for (const source of ['frame', 'head', 'left', 'right'] as const) {
      const at = this.observations[source];
      metrics[`${source}AgeMs`] = at === undefined ? -1 : Math.max(0, now - at);
    }
    // Timer-driven heartbeat remains observable when XR RAF is paused.
    return { cadence: 'xr', metrics, observedAt: now };
  }
}
