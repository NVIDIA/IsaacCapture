/**
 * The Ego Capture UI live recording strip: s/w/a/t controls plus `r` for New
 * batch, rendered as a SIBLING of the existing generic capture
 * tree/camera/status display -- never a separate page, tab, or its own copy
 * of that display. This is the live recording screen only; it collects the
 * operator name (a "Set operator" action, required before any recording
 * command is accepted) but does not select instruction modes/catalogs. It
 * also has no knowledge of whether the backend's own RecordingService
 * integration is connected -- that distinction belongs in docs/an MR
 * description, never a branch or banner in this component.
 *
 * This component owns the recording interaction state machine
 * (`recordingUiReducer`'s `phase`), driven by real command replies and the
 * backend's own observed snapshot. Every action's eligibility is computed
 * once by `computeEligibility` and used by both a control's disabled state
 * and its handler, so a stale click or shortcut can never slip past a
 * guard the button was showing.
 */

import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { computeEligibility } from "./recordingEligibility";
import { INITIAL_RECORDING_UI_STATE, recordingUiReducer, type CommandPhase, type RecordingPhase } from "./recordingUiReducer";
import { RecordingCommandUncertainError } from "./useCaptureSession";
import { useVoiceRecognition, type VoiceRecognitionStatus } from "./useVoiceRecognition";
import { parseVoicePhrase } from "./voiceCommands";
import type {
  RecordingActions,
  RecordingBatchIdentity,
  RecordingCommandResult,
  RecordingEpisodeIdentity,
  RecordingSnapshot,
} from "./recordingTypes";

export interface RecordingPanelProps {
  /** Authoritative recording facts; null until the backend's first push arrives. */
  recordingSnapshot: RecordingSnapshot | null;
  /** performance.now() at the instant `recordingSnapshot` was received --
   * the local basis `recordingSnapshot.currentEpisodeElapsedMs` (itself
   * monotonic but backend-clock-relative) is paired with to keep the
   * elapsed display ticking locally between snapshots. Null exactly when
   * `recordingSnapshot` is null. */
  recordingSnapshotReceivedAtMonotonic?: number | null;
  actions?: RecordingActions;
  /** The operator name attached to every outgoing recording command, or
   * null before one has ever been entered. Attribution only -- the
   * backend is the one real enforcement point; this just keeps the panel
   * from sending commands it already knows will be rejected. */
  operatorToken: string | null;
  setOperatorToken: (name: string) => void;
  /** The "New operator" action -- returns to requiring entry before any
   * further recording command, for a different person taking over. */
  clearOperatorToken: () => void;
  /** The control token attached to every outgoing recording command, or
   * null before one has been entered this tab -- the real credential (see
   * app.py's lifespan); without it the backend rejects all five mutating
   * actions outright. Kept in sessionStorage for this tab's session only
   * (see useCaptureSession.ts), never localStorage. */
  controlToken: string | null;
  setControlToken: (token: string) => void;
  /** Clears the entered control token -- disables this tab's future
   * commands until a new one is entered. */
  clearControlToken: () => void;
}

const NO_CONTROL_TOKEN_REASON = "Enter the recording control token before using recording controls";

const PHASE_LABEL: Record<RecordingPhase, string> = {
  idle: "Idle",
  starting: "Starting…",
  recording: "Recording",
  stopping: "Stopping…",
  discarding: "Discarding…",
  editingTask: "Editing task",
  savingTask: "Saving task…",
  restarting: "Restarting…",
  uncertain: "Outcome unknown…",
};

const VOICE_STATUS_LABEL: Record<VoiceRecognitionStatus, string> = {
  unsupported: "Voice: unsupported",
  disabled: "Voice: off",
  starting: "Voice: starting…",
  listening: "Voice: listening",
  error: "Voice: error",
};

/** Which visual treatment the phase badge gets -- purely presentational,
 * derived from the same real `ui.phase` the rest of the component already
 * uses, never a separate fabricated status. */
const PHASE_BADGE_VARIANT: Record<RecordingPhase, "idle" | "recording" | "busy" | "warning"> = {
  idle: "idle",
  starting: "busy",
  recording: "recording",
  stopping: "busy",
  discarding: "busy",
  editingTask: "idle",
  savingTask: "busy",
  restarting: "busy",
  uncertain: "warning",
};

/** Formats a millisecond duration as `m:ss` for the elapsed-time display. */
function formatElapsed(ms: number): string {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

/** The Ego Capture UI live recording strip. Does not render or own the generic
 * capture tree/camera display -- `App` mounts that once, as a permanent
 * sibling. */
const NO_OPERATOR_REASON = "Enter an operator name before using recording controls";

export function RecordingPanel({
  recordingSnapshot,
  recordingSnapshotReceivedAtMonotonic = null,
  actions = {},
  operatorToken,
  setOperatorToken,
  clearOperatorToken,
  controlToken,
  setControlToken,
  clearControlToken,
}: RecordingPanelProps) {
  const [ui, dispatch] = useReducer(recordingUiReducer, INITIAL_RECORDING_UI_STATE);
  const [operatorDraft, setOperatorDraft] = useState(operatorToken ?? "");
  const [controlDraft, setControlDraft] = useState(controlToken ?? "");
  // controlToken can become null from outside a local Clear click too (a
  // stale-token auto-clear when the backend restarts) -- without this, the
  // reappearing entry form would silently pre-fill the now-rejected value.
  useEffect(() => {
    if (controlToken === null) setControlDraft("");
  }, [controlToken]);
  // Prefills the DRAFT (never commits) from the configured default once to an untouched entry,
  // preserving user edits/clearing -- and only when the station opted in (off by default): the
  // operator still has to press "Set operator" either way, this only saves them typing the name.
  const operatorDraftTouchedRef = useRef(false);
  const defaultOperatorAppliedRef = useRef(false);
  useEffect(() => {
    if (defaultOperatorAppliedRef.current || operatorDraftTouchedRef.current) return;
    if (!recordingSnapshot?.autoFillOperatorName) return;
    const defaultOperator = recordingSnapshot?.defaultOperator ?? null;
    if (defaultOperator === null) return;
    defaultOperatorAppliedRef.current = true;
    if (operatorToken === null) setOperatorDraft(defaultOperator);
  }, [recordingSnapshot, operatorToken]);
  const baseEligibility = computeEligibility(recordingSnapshot, actions, ui.phase);
  // Every handler below (and so every keyboard shortcut, which calls the
  // same handlers) already gates on `eligibility.X.allowed` -- overriding
  // it here once, rather than re-checking `operatorToken`/`controlToken` at each call
  // site, is what makes "including keyboard shortcuts" automatic. The
  // backend enforces this independently and is the real authority; this
  // is just not sending a command it already knows will be rejected.
  const eligibility =
    operatorToken === null
      ? {
          startEpisode: { allowed: false, reason: NO_OPERATOR_REASON },
          stopEpisode: { allowed: false, reason: NO_OPERATOR_REASON },
          discardEpisode: { allowed: false, reason: NO_OPERATOR_REASON },
          taskEdit: { allowed: false, reason: NO_OPERATOR_REASON },
          newBatch: { allowed: false, reason: NO_OPERATOR_REASON },
        }
      : controlToken === null
        ? {
            startEpisode: { allowed: false, reason: NO_CONTROL_TOKEN_REASON },
            stopEpisode: { allowed: false, reason: NO_CONTROL_TOKEN_REASON },
            discardEpisode: { allowed: false, reason: NO_CONTROL_TOKEN_REASON },
            taskEdit: { allowed: false, reason: NO_CONTROL_TOKEN_REASON },
            newBatch: { allowed: false, reason: NO_CONTROL_TOKEN_REASON },
          }
        : baseEligibility;
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  // Reconciles `phase` with the backend's own observation -- covers this
  // viewer's own just-applied command, another viewer's command, and
  // resolving "uncertain" once a real snapshot confirms current state. The
  // reducer itself ignores this while a command of THIS viewer's own is
  // genuinely still in flight (see SNAPSHOT_OBSERVED's own doc).
  useEffect(() => {
    if (recordingSnapshot === null) return;
    dispatch({
      type: "SNAPSHOT_OBSERVED",
      recording: recordingSnapshot.episodePhase === "RECORDING",
      taskText: recordingSnapshot.taskText,
    });
  }, [recordingSnapshot]);

  // Mirrors the latest prop for callbacks below, updated by the effect
  // above on every render this component actually commits -- a prop
  // change this component was mid-editingTask/mid-command for (and so
  // ignored, see SNAPSHOT_OBSERVED's own doc) must still be the fact used
  // once that phase actually settles, not whatever was true when it began.
  const recordingSnapshotRef = useRef(recordingSnapshot);
  useEffect(() => {
    recordingSnapshotRef.current = recordingSnapshot;
  }, [recordingSnapshot]);

  // Forces a re-render once a second while an episode is actually running
  // (a real backend fact, not local phase) so the elapsed-time display
  // below keeps advancing; releases its interval the moment it isn't.
  const [, forceTick] = useReducer((n: number) => n + 1, 0);
  useEffect(() => {
    if (recordingSnapshot?.currentEpisodeElapsedMs == null) return;
    const id = setInterval(forceTick, 1000);
    return () => clearInterval(id);
  }, [recordingSnapshot?.currentEpisodeElapsedMs]);

  /** The one execution path for every command this component sends: marks
   * the matching transient phase, awaits the real reply, and applies its
   * real outcome to the reducer -- a resolved accept/reject is confident,
   * while a timeout/disconnect (RecordingCommandUncertainError) is
   * genuinely unknown and never treated as a confident rejection. A
   * command only ever has one outstanding instance (eligibility blocks
   * re-entry), so a resolved promise is always the correlated reply for
   * this exact call; no post-hoc identity re-check is needed or
   * performed. */
  const runCommand = useCallback(async (phase: CommandPhase, run: () => Promise<RecordingCommandResult> | undefined) => {
    dispatch({ type: "COMMAND_STARTED", phase });
    let result: RecordingCommandResult | undefined;
    let error: unknown;
    try {
      result = await run();
    } catch (err) {
      error = err;
    }
    if (!mountedRef.current) return;
    if (error instanceof RecordingCommandUncertainError) {
      dispatch({ type: "COMMAND_UNKNOWN", feedback: error.message });
      return;
    }
    // The latest known fact at settle time -- not necessarily from this
    // command's own reply, see COMMAND_REJECTED's own doc.
    const recording = recordingSnapshotRef.current?.episodePhase === "RECORDING";
    if (error !== undefined) {
      dispatch({ type: "COMMAND_REJECTED", feedback: error instanceof Error ? error.message : "request failed", recording });
      return;
    }
    if (result === undefined) {
      dispatch({ type: "COMMAND_REJECTED", feedback: "no handler available for this action yet", recording });
      return;
    }
    if (result.outcome === "rejected") {
      dispatch({ type: "COMMAND_REJECTED", feedback: result.reason ?? "rejected", recording });
      return;
    }
    dispatch({ type: "COMMAND_SUCCEEDED" });
  }, []);

  const batchId = recordingSnapshot?.batchId ?? null;
  const episodeId = recordingSnapshot?.currentEpisodeId ?? null;

  const handleStart = useCallback(() => {
    if (!eligibility.startEpisode.allowed || batchId === null) return;
    const target: RecordingBatchIdentity = { batchId };
    void runCommand("starting", () => actions.onStartEpisode?.(target));
  }, [eligibility, batchId, actions, runCommand]);

  const handleStopSave = useCallback(() => {
    if (!eligibility.stopEpisode.allowed || batchId === null || episodeId === null) return;
    const target: RecordingEpisodeIdentity = { batchId, episodeId };
    void runCommand("stopping", () => actions.onStopEpisode?.(target));
  }, [eligibility, batchId, episodeId, actions, runCommand]);

  // The `a` shortcut and Discard button both act directly, with no confirmation dialog.
  const handleDiscard = useCallback(() => {
    if (!eligibility.discardEpisode.allowed || batchId === null || episodeId === null) return;
    const target: RecordingEpisodeIdentity = { batchId, episodeId };
    void runCommand("discarding", () => actions.onDiscardEpisode?.(target));
  }, [eligibility, batchId, episodeId, actions, runCommand]);

  // While idle only -- never implicitly discards an active take. Success
  // applies the fresh batch identity/context via the next observed
  // snapshot; rejection (e.g. a second tab's concurrent request against
  // the now-stale id) preserves the existing batch and shows the reason.
  const handleNewBatch = useCallback(() => {
    if (!eligibility.newBatch.allowed || batchId === null) return;
    const target: RecordingBatchIdentity = { batchId };
    void runCommand("restarting", () => actions.onNewBatch?.(target));
  }, [eligibility, batchId, actions, runCommand]);

  const beginTaskEdit = useCallback(() => {
    if (!eligibility.taskEdit.allowed) return;
    dispatch({ type: "BEGIN_TASK_EDIT", currentText: recordingSnapshot?.taskText ?? "", currentBatchId: batchId });
  }, [eligibility, recordingSnapshot, batchId]);

  const cancelTaskEdit = useCallback(() => {
    dispatch({ type: "CANCEL_TASK_EDIT", recording: recordingSnapshotRef.current?.episodePhase === "RECORDING" });
  }, []);

  const submitTaskEdit = useCallback(() => {
    if (ui.phase !== "editingTask") return;
    // New operator/control token can clear while the editor is still open
    // (phase stays "editingTask" either way) -- without these, Save/Enter
    // would still send the command with no operator or no credential.
    if (operatorToken === null) return;
    if (controlToken === null) return;
    if (batchId === null) {
      void runCommand("savingTask", () => Promise.resolve({ outcome: "rejected", reason: "No open batch to save to yet" }));
      return;
    }
    const target: RecordingBatchIdentity = { batchId };
    const text = ui.taskDraft;
    void runCommand("savingTask", () => actions.onSetTask?.(target, text));
  }, [ui.phase, ui.taskDraft, batchId, actions, runCommand, operatorToken, controlToken]);

  // --- Voice commands: a continuously-listening alternative to the s/w/a/t/r
  // shortcuts and the inline task editor above, never a second command path
  // -- every branch below ends by calling one of the SAME guarded handlers
  // (or dispatching the SAME reducer actions) the buttons/keyboard shortcuts
  // already use. See voiceCommands.ts for the fixed phrase grammar and
  // useVoiceRecognition.ts for the recognizer lifecycle this wraps.
  const [voiceFeedback, setVoiceFeedback] = useState<string | null>(null);
  const uiPhaseRef = useRef(ui.phase);
  useEffect(() => {
    uiPhaseRef.current = ui.phase;
  }, [ui.phase]);

  // The batch/episode a voice command should act on is bound once PER
  // UTTERANCE, at its first interim observation (see useVoiceRecognition's
  // onUtteranceStart) -- never read fresh only at final-result time, which
  // could already be a different, later batch/episode than what was live
  // when the operator actually spoke.
  // "invalid" means a SECOND utterance started before the first one's final
  // result was consumed -- overlapping utterances can't be safely
  // attributed, so this marks the binding unusable rather than silently
  // overwriting it with the second utterance's own target. The hook's own
  // pending-interim tracking now fires onUtteranceStart at most once per
  // utterance (not once per interim result index), so this branch should be
  // unreachable in practice; kept as a cheap defensive backstop.
  const utteranceTargetRef = useRef<{ batchId: string | null; episodeId: string | null } | "invalid" | null>(null);
  const handleUtteranceStart = useCallback(() => {
    utteranceTargetRef.current =
      utteranceTargetRef.current !== null
        ? "invalid"
        : { batchId: recordingSnapshotRef.current?.batchId ?? null, episodeId: recordingSnapshotRef.current?.currentEpisodeId ?? null };
  }, []);
  // A recognizer SESSION boundary (initial enable, or any restart-after-end/disable/error) always
  // clears a leftover unfinished binding -- without this, an utterance that began but never
  // reached a final result before the session ended (e.g. Space disabled voice mid-utterance)
  // would still look "unconsumed" to the NEXT session's first onUtteranceStart, which would then
  // mark ITS OWN binding "invalid" and reject the first real command after re-enabling.
  const handleSessionStart = useCallback(() => {
    utteranceTargetRef.current = null;
  }, []);

  /** For an action whose target is the current batch/episode: the live
   * batch/episode if the bound-at-utterance-start target was actually
   * captured AND still matches it, or "reject" otherwise -- a missing
   * binding (no onUtteranceStart ever fired for this utterance), an
   * overlapping/invalid one, or a genuine mismatch are all treated the
   * same way: the caller must ask the operator to repeat the command, never
   * silently fall back to whatever is current now. Always consumes
   * (clears) the binding -- one utterance, one target, never reused by a
   * later unrelated command. */
  function resolveUtteranceTarget(): { batchId: string; episodeId: string | null } | "reject" {
    const bound = utteranceTargetRef.current;
    utteranceTargetRef.current = null;
    if (bound === null || bound === "invalid") return "reject";
    const live = { batchId: recordingSnapshotRef.current?.batchId ?? null, episodeId: recordingSnapshotRef.current?.currentEpisodeId ?? null };
    if (bound.batchId !== live.batchId || bound.episodeId !== live.episodeId || live.batchId === null) return "reject";
    return { batchId: live.batchId, episodeId: live.episodeId };
  }

  const handleVoiceResult = useCallback(
    (rawText: string) => {
      const intent = parseVoicePhrase(rawText);

      if (intent?.kind === "stopListening") {
        setVoiceFeedback(null);
        voiceToggle();
        utteranceTargetRef.current = null;
        return;
      }

      const phase = uiPhaseRef.current;

      if (intent === null) {
        // Never echoes the raw transcript: with an always-on mic, this is as likely to be nearby
        // conversation picked up by the recognizer as an actual misheard command, and displaying
        // it on the station screen would leak whatever was overheard.
        setVoiceFeedback("Not recognized.");
        utteranceTargetRef.current = null;
        return;
      }

      const busyWithTaskEdit = phase === "editingTask" || phase === "savingTask";

      switch (intent.kind) {
        case "record": {
          if (busyWithTaskEdit) {
            setVoiceFeedback("Finish or cancel the task edit first.");
            utteranceTargetRef.current = null;
            return;
          }
          // Consume the binding even when ineligible, so the next utterance starts fresh; report
          // unmet prerequisites (real reasons repeating can't fix) before asking for a repeat.
          const target = resolveUtteranceTarget();
          if (!eligibility.startEpisode.allowed) {
            setVoiceFeedback(eligibility.startEpisode.reason ?? "Cannot start recording right now.");
            return;
          }
          if (target === "reject") {
            setVoiceFeedback("Voice command target is unknown or changed -- please repeat.");
            return;
          }
          setVoiceFeedback(null);
          handleStart();
          return;
        }
        case "stop": {
          if (busyWithTaskEdit) {
            setVoiceFeedback("Finish or cancel the task edit first.");
            utteranceTargetRef.current = null;
            return;
          }
          const target = resolveUtteranceTarget();
          if (!eligibility.stopEpisode.allowed) {
            setVoiceFeedback(eligibility.stopEpisode.reason ?? "Nothing is currently recording.");
            return;
          }
          if (target === "reject") {
            setVoiceFeedback("Voice command target is unknown or changed -- please repeat.");
            return;
          }
          setVoiceFeedback(null);
          handleStopSave();
          return;
        }
        case "discard": {
          if (busyWithTaskEdit) {
            setVoiceFeedback("Finish or cancel the task edit first.");
            utteranceTargetRef.current = null;
            return;
          }
          const target = resolveUtteranceTarget();
          if (!eligibility.discardEpisode.allowed) {
            setVoiceFeedback(eligibility.discardEpisode.reason ?? "Nothing is currently recording.");
            return;
          }
          if (target === "reject") {
            setVoiceFeedback("Voice command target is unknown or changed -- please repeat.");
            return;
          }
          // A plain, synchronous browser confirm -- not a second spoken command. It blocks the
          // entire page's JS until dismissed, so nothing (including recordingSnapshotRef) can
          // change while it's open -- a post-close re-check against the live snapshot here would
          // always trivially pass and isn't real protection. handleDiscard() calls the same
          // guarded handler the Discard button uses, which the backend's own exact-episode check
          // (recording_session.py) rejects outright if the target has genuinely gone stale by the
          // time the command actually reaches it.
          if (!window.confirm("Discard this take?")) {
            setVoiceFeedback("Discard cancelled.");
            return;
          }
          setVoiceFeedback(null);
          handleDiscard();
          return;
        }
        case "task":
          utteranceTargetRef.current = null;
          if (phase === "idle") {
            if (!eligibility.taskEdit.allowed) {
              setVoiceFeedback(eligibility.taskEdit.reason ?? "Task editing is not available.");
              return;
            }
            setVoiceFeedback(null);
            dispatch({ type: "BEGIN_TASK_EDIT", currentText: intent.text, currentBatchId: batchId });
          } else if (phase === "editingTask") {
            setVoiceFeedback(null);
            dispatch({ type: "UPDATE_TASK_DRAFT", value: intent.text });
          } else {
            setVoiceFeedback("Cannot edit task right now.");
          }
          return;
        case "saveTask":
          utteranceTargetRef.current = null;
          if (phase !== "editingTask") {
            setVoiceFeedback("No task edit in progress to save.");
            return;
          }
          setVoiceFeedback(null);
          submitTaskEdit();
          return;
        case "cancelTask":
          utteranceTargetRef.current = null;
          if (phase !== "editingTask") {
            setVoiceFeedback("No task edit in progress to cancel.");
            return;
          }
          setVoiceFeedback(null);
          cancelTaskEdit();
          return;
      }
    },
    // voiceToggle is intentionally omitted here: it's declared right after this callback
    // (useVoiceRecognition needs this callback to construct it), but its identity is stable
    // across renders (see useVoiceRecognition's own useCallback), so omitting it from these deps
    // never leaves a stale closure.
    [eligibility, batchId, handleStart, handleStopSave, handleDiscard, submitTaskEdit, cancelTaskEdit],
  );

  const { status: voiceStatus, enabled: voiceEnabled, toggle: voiceToggle, errorMessage: voiceErrorMessage } = useVoiceRecognition(handleVoiceResult, {
    onUtteranceStart: handleUtteranceStart,
    onSessionStart: handleSessionStart,
  });

  // Space toggles voice mode globally -- independent of ui.phase (unlike the
  // s/w/a/t/r shortcut listener below, which is deliberately disabled while
  // editingTask/savingTask): the keydown listener itself isn't gated on
  // phase, so Space can still toggle voice mode while the task editor is
  // open PROVIDED focus isn't actually in the editor's own input (which it
  // normally is right after opening it, via its own autoFocus -- in that
  // case Space types a space character into the draft instead, exactly
  // like any other text-entry control below). Skipped while focus is in a
  // text-entry control or any element that natively responds to Space
  // (a button, a `<summary>`, or anything with role="button"), so normal
  // page interaction and activation of a focused control are never
  // hijacked. Also skipped entirely when voice isn't supported in this
  // browser -- there is no toggle to flip and no reason shown, so
  // swallowing Space here would be pure interference.
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key !== " " || event.repeat || event.altKey || event.ctrlKey || event.metaKey) return;
      if (voiceStatus === "unsupported") return;
      if (focusOwnedByInput() || (document.activeElement?.closest("button, summary, [role='button']") ?? null) !== null) return;
      voiceToggle();
      event.preventDefault();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [voiceToggle, voiceStatus]);


  // Buttons and shortcuts call the same guarded handlers above; this
  // listener only decides whether a shortcut is allowed to fire at all
  // right now (focus ownership), not whether the action itself is allowed
  // (each handler re-checks that through `eligibility`). The inline task
  // editor owns Enter/Escape itself, so this listener is skipped entirely
  // while it is open.
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.repeat || event.altKey || event.ctrlKey || event.metaKey) return;
      if (ui.phase === "editingTask" || ui.phase === "savingTask" || focusOwnedByInput()) return;
      switch (event.key) {
        case "s":
          handleStart();
          break;
        case "w":
          handleStopSave();
          break;
        case "a":
          handleDiscard();
          break;
        case "t":
          beginTaskEdit();
          break;
        case "r":
          handleNewBatch();
          break;
        default:
          return;
      }
      event.preventDefault();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [ui.phase, handleStart, handleStopSave, handleDiscard, beginTaskEdit, handleNewBatch]);

  // Live elapsed = the backend's own monotonic figure as of its snapshot,
  // plus local time elapsed since THIS browser received that snapshot --
  // never a shared wall-clock basis, so it stays correct under clock
  // drift. Once idle again, the most recently finished episode's frozen
  // duration stays visible rather than the display just vanishing.
  const elapsedMs =
    recordingSnapshot?.currentEpisodeElapsedMs != null && recordingSnapshotReceivedAtMonotonic != null
      ? recordingSnapshot.currentEpisodeElapsedMs + (performance.now() - recordingSnapshotReceivedAtMonotonic)
      : (recordingSnapshot?.lastResult?.durationMs ?? null);
  const taskEditBatchChanged = ui.taskEditBatchId !== null && batchId !== null && ui.taskEditBatchId !== batchId;
  const editingTask = ui.phase === "editingTask" || ui.phase === "savingTask";
  // `ui.phase` is this component's own interaction state machine, which
  // has no RECOVERY_REQUIRED value of its own -- with no command in
  // flight it reads "idle" even while the backend has a genuinely
  // interrupted take retained and blocking Record. Without this, the
  // panel showed "Idle" with no persistent indication that New batch is
  // needed, only a one-shot feedback message from whatever command last
  // happened to be rejected.
  const statusLabel =
    ui.phase === "idle" && recordingSnapshot?.episodePhase === "RECOVERY_REQUIRED"
      ? "Recovery required"
      : PHASE_LABEL[ui.phase];

  return (
    <section className="recording-panel" aria-label="Ego Capture UI episode recording">
      <div className="recording-panel-header">
        <h2 className="recording-panel-title">Recording</h2>
        <span className={`recording-phase-badge recording-phase-badge--${PHASE_BADGE_VARIANT[ui.phase]}`}>
          {PHASE_BADGE_VARIANT[ui.phase] === "recording" && <span className="recording-phase-dot" aria-hidden="true" />}
          {PHASE_LABEL[ui.phase]}
        </span>
      </div>
      <div className="recording-controls">
        {operatorToken === null ? (
          <form
            className="recording-operator-entry"
            onSubmit={(event) => {
              event.preventDefault();
              const name = operatorDraft.trim();
              if (name !== "") setOperatorToken(name);
            }}
          >
            <label htmlFor="recording-operator-input">Operator</label>
            <div className="recording-operator-row">
              <input
                id="recording-operator-input"
                type="text"
                value={operatorDraft}
                onChange={(event) => {
                  operatorDraftTouchedRef.current = true;
                  setOperatorDraft(event.target.value);
                }}
                placeholder="Enter your name to enable recording"
                autoComplete="off"
                maxLength={128}
              />
              <button type="submit" className="recording-button recording-button--secondary" disabled={operatorDraft.trim() === ""}>
                Set operator
              </button>
            </div>
          </form>
        ) : (
          <p className="recording-operator-current">
            Operator: {operatorToken}
            <button
              type="button"
              className="recording-button recording-button--secondary recording-button--small"
              onClick={() => {
                operatorDraftTouchedRef.current = true;
                setOperatorDraft("");
                clearOperatorToken();
              }}
            >
              New operator
            </button>
          </p>
        )}
        {controlToken === null ? (
          <form
            className="recording-control-token-entry"
            onSubmit={(event) => {
              event.preventDefault();
              if (controlDraft.trim() !== "") setControlToken(controlDraft);
            }}
          >
            <label htmlFor="recording-control-token-input">Control token</label>
            <div className="recording-control-token-row">
              <input
                id="recording-control-token-input"
                type="password"
                value={controlDraft}
                onChange={(event) => setControlDraft(event.target.value)}
                placeholder="Enter control token"
                autoComplete="off"
              />
              <button type="submit" className="recording-button recording-button--secondary" disabled={controlDraft.trim() === ""}>
                Set control token
              </button>
            </div>
            <p className="recording-control-token-help">Use the token shown in the backend console.</p>
          </form>
        ) : (
          <p className="recording-control-token-current">
            Control token entered
            <button
              type="button"
              className="recording-button recording-button--secondary recording-button--small"
              onClick={() => {
                setControlDraft("");
                clearControlToken();
              }}
            >
              Clear
            </button>
          </p>
        )}
        <p className="recording-status">
          <span className="recording-status-phase">{statusLabel}</span>
          <span className="recording-status-fact">
            Episode: {recordingSnapshot === null ? "—" : (recordingSnapshot.currentEpisodeId ?? "none")}
          </span>
          <span className="recording-status-fact">Saved: {recordingSnapshot?.savedCount ?? "—"}</span>
          {elapsedMs !== null && <span className="recording-status-fact recording-status-elapsed">Elapsed: {formatElapsed(elapsedMs)}</span>}
          <span className="recording-status-task">
            Task: {recordingSnapshot === null ? "—" : recordingSnapshot.taskText || "unset"}
          </span>
        </p>
        {voiceStatus !== "unsupported" && (
          <p className="recording-voice-status">
            <button
              type="button"
              className="recording-button recording-button--secondary recording-button--small"
              onClick={voiceToggle}
              aria-pressed={voiceEnabled}
            >
              {VOICE_STATUS_LABEL[voiceStatus]} (<kbd>space</kbd>)
            </button>
            {voiceErrorMessage !== null ? (
              <span className="recording-voice-feedback recording-voice-feedback--error" role="status">
                {voiceErrorMessage}
              </span>
            ) : (
              voiceFeedback !== null && (
                <span className="recording-voice-feedback" role="status">
                  {voiceFeedback}
                </span>
              )
            )}
          </p>
        )}
        <div className="recording-actions recording-actions--primary">
          <button
            type="button"
            className="recording-button recording-button--record"
            onClick={handleStart}
            disabled={!eligibility.startEpisode.allowed}
            title={eligibility.startEpisode.reason ?? undefined}
          >
            Record (<kbd>s</kbd>)
          </button>
          <button
            type="button"
            className="recording-button recording-button--stop"
            onClick={handleStopSave}
            disabled={!eligibility.stopEpisode.allowed}
            title={eligibility.stopEpisode.reason ?? undefined}
          >
            Stop/Save (<kbd>w</kbd>)
          </button>
          <button
            type="button"
            className="recording-button recording-button--discard"
            onClick={handleDiscard}
            disabled={!eligibility.discardEpisode.allowed}
            title={eligibility.discardEpisode.reason ?? undefined}
          >
            Discard (<kbd>a</kbd>)
          </button>
        </div>
        <div className="recording-actions-divider" />
        <div className="recording-actions recording-actions--secondary">
          {!editingTask ? (
            <button
              type="button"
              className="recording-button recording-button--secondary"
              onClick={beginTaskEdit}
              disabled={!eligibility.taskEdit.allowed}
              title={eligibility.taskEdit.reason ?? undefined}
            >
              Edit task (<kbd>t</kbd>)
            </button>
          ) : (
            <InlineTaskEditor
              draft={ui.taskDraft}
              pending={ui.phase === "savingTask"}
              saveBlockedReason={
                operatorToken === null
                  ? NO_OPERATOR_REASON
                  : controlToken === null
                    ? NO_CONTROL_TOKEN_REASON
                    : recordingSnapshot?.batchId == null
                      ? "No open batch to save to yet"
                      : null
              }
              batchChanged={taskEditBatchChanged}
              onChange={(value) => dispatch({ type: "UPDATE_TASK_DRAFT", value })}
              onSubmit={submitTaskEdit}
              onCancel={cancelTaskEdit}
            />
          )}
          <button
            type="button"
            className="recording-button recording-button--secondary"
            onClick={handleNewBatch}
            disabled={!eligibility.newBatch.allowed}
            title={eligibility.newBatch.reason ?? undefined}
          >
            New batch (<kbd>r</kbd>)
          </button>
        </div>
        {ui.feedback !== null && (
          <p className="recording-feedback" role="status">
            {ui.feedback}
          </p>
        )}
        {recordingSnapshot?.lastResult != null && (
          <p className="recording-last-result">
            Last: {recordingSnapshot.lastResult.disposition}
            {recordingSnapshot.lastResult.issues.length > 0 ? ` -- ${recordingSnapshot.lastResult.issues.join("; ")}` : ""}
          </p>
        )}
      </div>
    </section>
  );
}

/** True while a text-input/select/contenteditable element currently owns keyboard focus -- shortcuts must not fire underneath it. */
function focusOwnedByInput(): boolean {
  const el = document.activeElement;
  if (el === null) return false;
  const tag = el.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || (el as HTMLElement).isContentEditable;
}

/** The inline next-task text editor -- only the committed instruction TEXT,
 * never a mode/catalog/metadata editor. Enter submits, Escape cancels; it
 * owns both keys itself rather than relying on the global shortcut
 * listener (which is skipped entirely while this is open). If the batch
 * rotated (a New batch command, from this viewer or another) while the
 * draft is still unsaved, shows that fact rather than silently retargeting
 * or committing the draft -- Save retries explicitly against the new batch. */
function InlineTaskEditor({
  draft,
  pending,
  saveBlockedReason,
  batchChanged,
  onChange,
  onSubmit,
  onCancel,
}: {
  draft: string;
  pending: boolean;
  /** Why Save is disabled, or null when it's allowed -- distinguishes "no
   * open batch" from "no control token", since both block Save but need
   * different operator actions to fix. */
  saveBlockedReason: string | null;
  batchChanged: boolean;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onCancel: () => void;
}) {
  return (
    <span className="recording-task-editor">
      <input
        type="text"
        className="recording-task-editor-input"
        value={draft}
        autoFocus
        disabled={pending}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            onSubmit();
          } else if (e.key === "Escape") {
            e.preventDefault();
            onCancel();
          }
        }}
      />
      <span className="recording-task-editor-actions">
        <button
          type="button"
          className="recording-button recording-button--stop recording-button--small"
          onClick={onSubmit}
          disabled={pending || saveBlockedReason !== null}
          title={saveBlockedReason ?? undefined}
        >
          Save
        </button>
        <button
          type="button"
          className="recording-button recording-button--secondary recording-button--small"
          onClick={onCancel}
          disabled={pending}
        >
          Cancel
        </button>
      </span>
      {batchChanged && (
        <span className="recording-feedback" role="status">
          Batch changed since you started editing -- Save will apply to the new batch.
        </span>
      )}
    </span>
  );
}
