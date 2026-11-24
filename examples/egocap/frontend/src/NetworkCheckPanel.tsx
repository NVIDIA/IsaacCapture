// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * CaptureServer's camera delivery check: whether frames reach the capture host well enough to record. Shown whether
 * or not a take is running, and only from what the server reports; a result that is not current (a stale feed, or the
 * server's own ageing out) is never drawn as a pass. It covers media delivery only, apart from recorder and disk
 * health, and the panel says which legs it did and did not test.
 */

import { useState } from "react";

import { NetworkCheckState, type NetworkCheckCamera, type NetworkCheckStatus } from "./captureStatus";
import type { RecordingCommandResult } from "./recordingTypes";

export interface NetworkCheckPanelProps {
  check: NetworkCheckStatus | null;
  /** false when the status feed is stale or finished: whatever it last said is no longer current. */
  live: boolean;
  /** Asks the server to run the check again; undefined when this page cannot send commands yet. */
  onRun?: () => Promise<RecordingCommandResult>;
  /** Why a rerun is not allowed right now (a take is in progress); the button is disabled and says so. */
  runBlockedReason?: string | null;
}

type Tone = "ok" | "warn" | "bad" | "idle";

interface Headline {
  label: string;
  tone: Tone;
}

/** The one-line verdict. A pass only reads as one while the feed is live and the server has not aged it out. */
export function headlineFor(check: NetworkCheckStatus, live: boolean): Headline {
  if (!live) return { label: "Network test: status unavailable or out of date", tone: "idle" };
  switch (check.state) {
    case NetworkCheckState.CHECKING: {
      const phase = check.phase === "measuring" ? `measuring, ${check.progress_s ?? 0} of ${check.window_s} seconds` : "connecting to camera streams";
      return { label: `Network test in progress (${phase})`, tone: "idle" };
    }
    case NetworkCheckState.READY:
      return { label: "Network test passed", tone: "ok" };
    case NetworkCheckState.UNSUITABLE:
      return { label: "Network test failed", tone: "bad" };
    case NetworkCheckState.STALE:
      return { label: "Network test expired; run it again before recording", tone: "warn" };
    default:
      return { label: "Network test has not run for this capture session", tone: "idle" };
  }
}

/** A measurement the server could not make is shown as unknown, never as zero. */
function show(value: number | null, unit: string, digits = 0): string {
  return value === null ? "unknown" : `${value.toFixed(digits)}${unit}`;
}

/** The longest silence anywhere in the window: between arrivals and at its start and end. */
function longestSilence(camera: NetworkCheckCamera): number | null {
  const known = [camera.max_arrival_gap_ms, camera.start_silence_ms, camera.end_silence_ms].filter(
    (v): v is number => v !== null,
  );
  return known.length === 0 ? null : Math.max(...known);
}

function CameraRow({ camera }: { camera: NetworkCheckCamera }) {
  const verdict = camera.passed === null ? "not judged" : camera.passed ? "pass" : "fail";
  return (
    <li className={`netcheck-camera netcheck-camera--${verdict.replace(" ", "-")}`}>
      <span className="netcheck-camera-name">{camera.name}</span>
      <span className="netcheck-camera-verdict">{camera.passed === null ? "Not measured" : camera.passed ? "Passed" : "Failed"}</span>
      <span>
        Delivery rate: {show(camera.delivered_fps, " fps", 1)} (configured: {camera.expected_fps} fps)
      </span>
      <span>Usable frames received: {show(camera.usable_frames, "")}</span>
      <span>Missing frames: {show(camera.missing_frames, "")}</span>
      <span>Longest gap without usable frames: {show(longestSilence(camera), " ms")}</span>
      <span>
        Capture-to-receive delay: median {show(camera.age_median_ms, " ms")}; maximum {show(camera.age_max_ms, " ms")}
      </span>
      {camera.reasons.length > 0 && <span className="netcheck-camera-reasons">{camera.reasons.join("; ")}</span>}
    </li>
  );
}

export function NetworkCheckPanel({ check, live, onRun, runBlockedReason = null }: NetworkCheckPanelProps) {
  const [busy, setBusy] = useState(false);
  const [refusal, setRefusal] = useState<string | null>(null);
  // The server runs no check, so there is nothing to show or to retry.
  if (check === null) return null;
  const headline = headlineFor(check, live);
  const checking = live && check.state === NetworkCheckState.CHECKING;
  const run = async () => {
    if (onRun === undefined) return;
    setBusy(true);
    setRefusal(null);
    try {
      const result = await onRun();
      if (result.outcome === "rejected") setRefusal(result.reason ?? "Network test did not start");
    } catch {
      // The outcome is unknown; the status feed says whether a check began.
      setRefusal("Could not confirm whether the network test started; check the reported status");
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className={`netcheck netcheck--${headline.tone}`} aria-label="Network check">
      <div className="netcheck-head">
        <span aria-live="polite" className="netcheck-headline">
          {headline.label}
        </span>
        <button
          type="button"
          className="netcheck-run"
          disabled={onRun === undefined || !live || checking || busy || runBlockedReason !== null}
          title={runBlockedReason ?? undefined}
          onClick={() => void run()}
        >
          {check.state === NetworkCheckState.NOT_TESTED ? "Run network test" : "Run test again"}
        </button>
      </div>
      {live && check.reason !== null && check.state === NetworkCheckState.UNSUITABLE && (
        <p className="netcheck-reason">{check.reason}</p>
      )}
      {refusal !== null && <p className="netcheck-reason">{refusal}</p>}
      <ul className="netcheck-cameras" aria-label="Camera delivery">
        {check.cameras.map((camera) => (
          <CameraRow key={camera.stream_id} camera={camera} />
        ))}
      </ul>
      {check.completed_at_unix_ns !== null && (
        <p className="netcheck-reason">
          Test completed at {new Date(check.completed_at_unix_ns / 1e6).toLocaleTimeString()}. Results expire after {check.valid_for_s} seconds.
          This test does not monitor subsequent connection changes.
        </p>
      )}
      <details className="netcheck-scope">
        <summary>Test coverage</summary>
        <p>Media endpoint: {check.endpoint}</p>
        <p>Measured path: {check.legs_tested.join(", ")}.</p>
        <p>Outside this test: {check.legs_not_tested.join("; ")}.</p>
      </details>
    </section>
  );
}
