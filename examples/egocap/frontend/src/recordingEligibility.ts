// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * One pure source of truth for whether each recording action is currently
 * allowed, and why not when it isn't. Both a control's disabled state and
 * its click/shortcut handler call the same function, so a stale click can
 * never slip past a guard the button itself was supposedly enforcing.
 */

import type { RecordingPhase } from "./recordingUiReducer";
import type { RecordingActions, RecordingSnapshot } from "./recordingTypes";

export type RecordingActionKind = "startEpisode" | "stopEpisode" | "discardEpisode" | "taskEdit" | "newBatch";

export interface Eligibility {
  allowed: boolean;
  /** Always non-null when `allowed` is false; always null when `allowed` is true. */
  reason: string | null;
}

/** Shorthand for an always-allowed result. */
function eligible(): Eligibility {
  return { allowed: true, reason: null };
}

/** Shorthand for a disallowed result carrying its own visible reason. */
function blocked(reason: string): Eligibility {
  return { allowed: false, reason };
}

/** Which action kind, if any, `phase` represents an in-flight command for. */
const BUSY_KIND: Partial<Record<RecordingPhase, RecordingActionKind>> = {
  starting: "startEpisode",
  stopping: "stopEpisode",
  discarding: "discardEpisode",
  savingTask: "taskEdit",
  restarting: "newBatch",
};

/** Computes every action's current eligibility from the real snapshot, the
 * actually-supplied handler props, and the recording interaction phase --
 * any phase other than idle/recording means one command is already in
 * flight and blocks the four take-level actions. New batch is stricter: it
 * is only available while actually idle (not even while recording). */
export function computeEligibility(
  snapshot: RecordingSnapshot | null,
  actions: RecordingActions,
  phase: RecordingPhase,
): Record<RecordingActionKind, Eligibility> {
  const connected = snapshot !== null;
  const episodePhase = snapshot?.episodePhase ?? null;
  const busyKind = BUSY_KIND[phase];
  const busy = phase !== "idle" && phase !== "recording";

  function result(kind: RecordingActionKind, compute: () => Eligibility): Eligibility {
    // A command's real outcome is unresolved -- block everything until an
    // actual observed snapshot resolves it, never a guess from local state.
    if (phase === "uncertain") return blocked("Previous command's outcome is unknown -- waiting to confirm");
    if (busy) return blocked(kind === busyKind ? "Already in progress" : "Another action is still in progress");
    return compute();
  }

  return {
    startEpisode: result("startEpisode", () => {
      if (!connected) return blocked("Recording backend not connected yet");
      if (actions.onStartEpisode === undefined) return blocked("Start is not available yet");
      // The more specific reason first: while an episode is active, "already active" is the true one.
      if (episodePhase === "RECOVERY_REQUIRED") return blocked("A take was interrupted -- use New batch to clear it");
      if (episodePhase !== "IDLE") return blocked("An episode is already active");
      if (snapshot.taskText === null || snapshot.taskText.trim() === "") return blocked("Set a task first");
      if (!snapshot.readyToRecord) return blocked("Required sources not ready");
      return eligible();
    }),
    stopEpisode: result("stopEpisode", () => {
      if (!connected) return blocked("Recording backend not connected yet");
      if (actions.onStopEpisode === undefined) return blocked("Stop/save is not available yet");
      if (episodePhase !== "RECORDING") return blocked("No active recording to stop");
      if (snapshot.currentEpisodeId === null) return blocked("No active episode to stop");
      return eligible();
    }),
    discardEpisode: result("discardEpisode", () => {
      if (!connected) return blocked("Recording backend not connected yet");
      if (actions.onDiscardEpisode === undefined) return blocked("Discard is not available yet");
      // Only the stable RECORDING state is a safe, well-defined point for an
      // operator-triggered discard; STARTING/STOPPING/FINALIZING are already
      // backend-owned transitions in flight, and RECOVERY_REQUIRED is
      // cleared by New batch, not by this control.
      if (episodePhase !== "RECORDING") return blocked("No active episode to discard");
      if (snapshot.currentEpisodeId === null) return blocked("No active episode to discard");
      return eligible();
    }),
    taskEdit: result("taskEdit", () => {
      if (!connected) return blocked("Recording backend not connected yet");
      if (actions.onSetTask === undefined) return blocked("Task editing is not available yet");
      // Editing the next task stays allowed between takes, including while
      // a prior take is checking/finalizing -- only an actively RECORDING
      // take blocks it, since its own context is already frozen.
      if (episodePhase === "RECORDING") return blocked("Cannot edit task while recording");
      return eligible();
    }),
    // Deliberately NOT routed through result()'s busy check -- new batch is
    // unavailable while actually recording too (a take must be stopped or
    // discarded first, never implicitly), not just while another command
    // is in flight.
    newBatch: (() => {
      if (!connected) return blocked("Recording backend not connected yet");
      if (actions.onNewBatch === undefined) return blocked("New batch is not available yet");
      if (phase === "uncertain") return blocked("Previous command's outcome is unknown -- waiting to confirm");
      if (phase === "recording") return blocked("Stop/save or discard the current take first");
      if (phase !== "idle") return blocked("Another action is still in progress");
      return eligible();
    })(),
  };
}
