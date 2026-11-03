/**
 * The operator's status bar: what the take is doing, what the task is, how many are saved, how the last one went,
 * and whether every device and camera is healthy. Read-only -- the recording controls themselves stay in
 * RecordingPanel, which this bar never duplicates -- and derived only from the backend's own recording snapshot and
 * capture status, so it can never claim a state those facts do not support.
 */

import { useEffect, useReducer } from "react";

import type { CaptureStatusResponse } from "./captureStatus";
import { chipsNeedingAttention, deriveHealthChips, type HealthChip } from "./deviceHealth";
import { formatElapsed } from "./RecordingPanel";
import type { RecordingSnapshot } from "./recordingTypes";
import type { RecordingPhase } from "./recordingUiReducer";

/** How the recording indicator is drawn. */
export type RecVariant = "recording" | "busy" | "ready" | "idle" | "warning" | "stale";

export interface RecIndicator {
  /** The big word: REC, READY, SAVING, ... */
  label: string;
  variant: RecVariant;
  /** A short sentence of why, when the state needs one. */
  hint: string | null;
}

/** What the operator has entered in this browser. Recording is refused until both are set, whatever the backend's
 * own readiness says, so READY must not be claimed without them. */
export interface OperatorSetup {
  operatorSet: boolean;
  controlTokenSet: boolean;
}

/** The recording indicator for one backend snapshot. ``live`` is false when the feed is stale or finished, in which
 * case the snapshot is last-known state and is labelled so rather than shown as current. */
export function deriveRecIndicator(
  snapshot: RecordingSnapshot | null,
  live: boolean,
  setup: OperatorSetup = { operatorSet: true, controlTokenSet: true },
  pendingPhase: RecordingPhase = "idle",
): RecIndicator {
  if (snapshot === null) return { label: "WAITING", variant: "idle", hint: "Waiting for the recording service" };
  if (!live) return { label: "NO FEED", variant: "stale", hint: "Showing the last known state" };
  // A command this browser has sent and not yet seen answered. The backend's phase only changes once it finishes
  // (a stop can take seconds to write the take), so without this the bar would keep saying REC after Stop.
  switch (pendingPhase) {
    case "starting":
      return { label: "STARTING", variant: "busy", hint: null };
    case "stopping":
      return { label: "STOPPING", variant: "busy", hint: "Saving the take" };
    case "discarding":
      return { label: "DISCARDING", variant: "busy", hint: null };
    case "restarting":
      return { label: "NEW BATCH", variant: "busy", hint: null };
    case "uncertain":
      return { label: "UNKNOWN", variant: "warning", hint: "The last command's outcome is unknown -- wait for the status to settle" };
    default:
      break;
  }
  switch (snapshot.episodePhase) {
    case "RECORDING":
      return { label: "REC", variant: "recording", hint: null };
    case "STARTING":
      return { label: "STARTING", variant: "busy", hint: null };
    case "STOPPING":
      return { label: "STOPPING", variant: "busy", hint: null };
    case "FINALIZING":
      return { label: "SAVING", variant: "busy", hint: "Do not unplug anything yet" };
    case "RECOVERY_REQUIRED":
      return { label: "RECOVERY", variant: "warning", hint: "An interrupted take is blocking Record -- start a new batch" };
    case "IDLE": {
      if (!snapshot.readyToRecord) {
        return { label: "NOT READY", variant: "idle", hint: snapshot.taskText ? null : "Set a task to enable Record" };
      }
      // The backend would admit a take, but this browser may not be allowed to start one yet.
      if (!setup.operatorSet || !setup.controlTokenSet) {
        const missing = [!setup.operatorSet && "an operator name", !setup.controlTokenSet && "the control token"].filter(Boolean);
        return { label: "SET UP", variant: "warning", hint: `Enter ${missing.join(" and ")} below` };
      }
      return { label: "READY", variant: "ready", hint: null };
    }
  }
}

/** The elapsed time of the take in progress, advanced locally between snapshots; null when no take is running.
 * The backend's own monotonic figure is the basis, plus the local time since this browser received it, so it stays
 * right under clock drift. */
function liveElapsedMs(snapshot: RecordingSnapshot | null, receivedAtMonotonic: number | null): number | null {
  if (snapshot?.currentEpisodeElapsedMs == null || receivedAtMonotonic === null) return null;
  return snapshot.currentEpisodeElapsedMs + (performance.now() - receivedAtMonotonic);
}

interface OperatorBarProps {
  response: CaptureStatusResponse | null;
  /** False when the feed is stale or finished: everything shown is then last-known, never current. */
  live: boolean;
  recordingSnapshot: RecordingSnapshot | null;
  recordingSnapshotReceivedAtMonotonic: number | null;
  /** The operator name this browser has entered, or null. */
  operatorName: string | null;
  /** Whether this browser has entered the recording control token. */
  hasControlToken: boolean;
  /** The recording panel's own in-flight phase (see RecordingPanel's onPhaseChange). */
  pendingPhase: RecordingPhase;
}

function HealthChipView({ chip }: { chip: HealthChip }) {
  const unhealthy = chip.tone === "warning" || chip.tone === "critical";
  return (
    <li className={`op-chip op-chip--${chip.tone}`} title={chip.detail ?? `${chip.label}: ${chip.state}`}>
      <span className="op-chip-dot" aria-hidden="true" />
      <span className="op-chip-label">{chip.label}</span>
      {/* A healthy device is just its name; only a problem spends space on the reason. */}
      {unhealthy && <span className="op-chip-state">{chip.detail ?? chip.state}</span>}
    </li>
  );
}

export function OperatorBar({
  response,
  live,
  recordingSnapshot,
  recordingSnapshotReceivedAtMonotonic,
  operatorName,
  hasControlToken,
  pendingPhase,
}: OperatorBarProps) {
  // Re-render once a second while a take is running so the timer advances; released as soon as it is not.
  const [, tick] = useReducer((n: number) => n + 1, 0);
  const running = recordingSnapshot?.currentEpisodeElapsedMs != null;
  useEffect(() => {
    if (!running) return undefined;
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [running]);

  const rec = deriveRecIndicator(recordingSnapshot, live, { operatorSet: operatorName !== null, controlTokenSet: hasControlToken }, pendingPhase);
  const elapsedMs = liveElapsedMs(recordingSnapshot, recordingSnapshotReceivedAtMonotonic);
  const chips = deriveHealthChips(response, live);
  const problems = chipsNeedingAttention(chips);
  const task = recordingSnapshot?.taskText ?? "";
  const last = recordingSnapshot?.lastResult ?? null;

  return (
    <section className="op-bar" aria-label="Operator status">
      <div className={`op-rec op-rec--${rec.variant}`}>
        <span className="op-rec-dot" aria-hidden="true" />
        <div className="op-rec-text">
          <span className="op-rec-label" aria-live="polite">
            {rec.label}
          </span>
          {rec.hint !== null && <span className="op-rec-hint">{rec.hint}</span>}
        </div>
        {elapsedMs !== null && <span className="op-rec-time">{formatElapsed(elapsedMs)}</span>}
      </div>

      <div className="op-task">
        <span className="op-kicker">Task</span>
        <p className={`op-task-text${task === "" ? " op-task-text--unset" : ""}`}>{task === "" ? "No task set" : task}</p>
      </div>

      <div className="op-counts">
        <div className="op-saved">
          <span className="op-saved-count">{recordingSnapshot?.savedCount ?? "—"}</span>
          <span className="op-kicker">saved</span>
        </div>
        {last !== null && (
          <div
            className={`op-last op-last--${last.disposition.toLowerCase()}`}
            title={last.issues.length > 0 ? last.issues.join("; ") : undefined}
          >
            <span className="op-kicker">Last take</span>
            <span className="op-last-value">
              {last.disposition === "KEPT" ? "Kept" : last.disposition === "REJECTED" ? "Rejected" : "Discarded"}
              {" · "}
              {formatElapsed(last.durationMs)}
              {last.issues.length > 0 && ` · ${last.issues.length} issue${last.issues.length === 1 ? "" : "s"}`}
            </span>
          </div>
        )}
      </div>

      <div className="op-health">
        <span className="op-kicker">
          {chips.length === 0 ? "Devices" : problems.length === 0 ? "All devices OK" : `${problems.length} need attention`}
        </span>
        {chips.length === 0 ? (
          <span className="op-health-empty">none reported yet</span>
        ) : (
          <ul className="op-chips" aria-label="Device health">
            {/* Problems first, so a fault is never buried behind a row of healthy chips. */}
            {[...problems, ...chips.filter((c) => !problems.includes(c))].map((chip) => (
              <HealthChipView key={chip.key} chip={chip} />
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}
