// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * The narrow shape the real backend command handlers (backend/egocap_web/
 * recording_command.py) publish and accept today -- in-memory batch/task/
 * episode bookkeeping, no actual RecordingService call yet, but a real
 * working round trip, not a stub.
 *
 * A recording "batch" (the group of episodes sharing one batch id) is a
 * concept scoped entirely to this module and its backend counterpart --
 * it is NOT the capture preview's own connection/session; that stays
 * entirely separate (see useCaptureSession.ts).
 *
 * This component is the live recording strip only (`s`/`w`/`a`/`t`/`r` for
 * record/stop/discard/task/new batch) -- it collects the operator name but
 * does not select instruction modes/catalogs or configure a batch; the
 * batch's own identity and current task text are supplied by the backend's
 * recording-command snapshot, not chosen here.
 */

/** Where the backend's current episode currently is, once one exists. */
export type RecordingEpisodePhase =
  | "IDLE"
  | "STARTING"
  | "RECORDING"
  | "STOPPING"
  | "FINALIZING"
  | "RECOVERY_REQUIRED";

/** The real disposition of a finished episode, once a backend reports one. */
export type EpisodeDisposition = "KEPT" | "REJECTED" | "DISCARDED";

/** Identifies one backend recording batch -- distinct from the preview
 * page's own capture session ID, which is a different resource entirely. */
export interface RecordingBatchIdentity {
  batchId: string;
}

/** Identifies one episode within a specific batch, so a command can never
 * silently apply to a different, later episode. */
export interface RecordingEpisodeIdentity {
  batchId: string;
  episodeId: string;
}

/** The outcome of the most recently finished episode, for result presentation. */
export interface EpisodeResultSummary {
  episodeId: string;
  disposition: EpisodeDisposition;
  issues: readonly string[];
  /** This episode's final duration (ms), frozen at the moment it ended --
   * so a just-finished take's duration stays visible/accounted for
   * instead of the elapsed display simply vanishing the instant
   * recording stops. Carried on this existing fact rather than a second
   * parallel completion record. */
  durationMs: number;
}

/**
 * Authoritative recording facts, published by the backend's RecordingContext
 * over the existing WS. `null` until the first one actually arrives --
 * RecordingPanel must render that honestly, never a fabricated state. Once
 * a snapshot exists, `batchId` always names a real, live batch -- there is
 * no closed/terminal batch state; `new_batch` rotates the id in place.
 * `taskText` is the currently committed next-task instruction text -- the
 * only task-related fact this component renders or edits; any other task
 * metadata (mode, catalog, library selection) belongs to the separate
 * batch-setup screen, not here.
 */
export interface RecordingSnapshot {
  batchId: string;
  episodePhase: RecordingEpisodePhase;
  currentEpisodeId: string | null;
  /** How many ms had elapsed in the current episode as of the moment the
   * backend produced THIS snapshot, or null while idle -- monotonic
   * (time.monotonic()-derived), not wall-clock, so it stays correct
   * regardless of clock drift between the backend's machine and this
   * browser. Pair with useCaptureSession's own recordingSnapshotReceivedAtMonotonic
   * (performance.now() at receipt) to keep counting locally between
   * snapshots, so a refresh/reconnect/new tab shows genuine elapsed time
   * instead of inventing a local basis from whenever that viewer happened
   * to observe "recording". */
  currentEpisodeElapsedMs: number | null;
  taskText: string | null;
  /** The station's configured default operator name (from the deployment's station
   * YAML), or null when unconfigured -- a PREFILL hint only, applied at most once to
   * the entry field when nothing is already stored for this browser; never itself
   * authorizes a command. See useCaptureSession's operatorToken. */
  defaultOperator: string | null;
  /** Whether defaultOperator may be used to prefill the entry field's DRAFT text at
   * all -- off by default (station YAML's auto_fill_operator_name). Even when true,
   * this never means auto-commit: the operator still has to press "Set operator". */
  autoFillOperatorName: boolean;
  savedCount: number;
  lastResult: EpisodeResultSummary | null;
  /** Whether the backend would currently ADMIT a start_episode command
   * (no episode already active and a task is set) -- not a real observed
   * device/capture-source readiness signal. */
  readyToRecord: boolean;
  /** The backend's count of accepted commands. Compared only within one
   * socket, so a snapshot older than one already applied can be dropped;
   * it restarts from 0 with the backend process. */
  revision: number;
}

/** The exact action names the backend's command handlers recognize --
 * matches the keys of backend/egocap_web/recording_command.py's
 * _PAYLOAD_VALIDATORS exactly. Kept narrow rather than a bare string so a typo is a compile
 * error. No `open_session`/`close_session` here -- this component never
 * opens or closes anything, it only rotates to a fresh batch. */
export type RecordingActionName = "set_task" | "start_episode" | "stop_episode" | "discard_episode" | "new_batch";

/** The real, awaited result of one recording command: the backend's own
 * handler accepted it ("ok") or rejected it with a real, specific reason
 * ("rejected") -- e.g. no matching batch, an episode already active, no
 * active episode to stop/discard. `reason` is always non-null exactly when
 * `outcome` is `"rejected"`. This is a narrow union (not a bare string) so
 * a future new outcome is added deliberately, not silently accepted. */
export interface RecordingCommandResult {
  outcome: "ok" | "rejected";
  reason: string | null;
}

/** Everything this component can ask the backend to do, each named with
 * the exact identity it would act on -- never "whatever is current by the
 * time this resolves." An absent handler disables the corresponding
 * control rather than silently doing nothing. Each returns a promise of
 * the real reply (or rejects on disconnect/timeout) so a caller can track
 * "is this still in flight" and show the real outcome, never infer success
 * from mere receipt. There is no `onCloseSession` -- `onNewBatch` replaces
 * it; this component never closes/opens anything. */
export interface RecordingActions {
  onSetTask?: (target: RecordingBatchIdentity, taskText: string) => Promise<RecordingCommandResult>;
  onStartEpisode?: (target: RecordingBatchIdentity) => Promise<RecordingCommandResult>;
  onStopEpisode?: (target: RecordingEpisodeIdentity) => Promise<RecordingCommandResult>;
  onDiscardEpisode?: (target: RecordingEpisodeIdentity) => Promise<RecordingCommandResult>;
  onNewBatch?: (target: RecordingBatchIdentity) => Promise<RecordingCommandResult>;
}
