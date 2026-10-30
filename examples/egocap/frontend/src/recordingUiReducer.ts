/**
 * The recording interaction state machine: idle -> starting -> recording ->
 * stopping -> idle (plus discarding/editingTask/savingTask/restarting/
 * uncertain). This is the one place that turns a user action, a command's
 * real success/rejection/unknown outcome, and an observed backend snapshot
 * into "what phase are we presenting right now" -- it owns no copy of
 * backend facts (batch id, task text, saved count, elapsed basis, results
 * all come straight from the RecordingSnapshot prop) and no separate
 * "pending action" flag; `phase` alone says what, if anything, is in
 * flight. There is no terminal/closed phase -- "restarting" (new_batch)
 * rotates straight back to idle.
 */

export type RecordingPhase =
  | "idle"
  | "starting"
  | "recording"
  | "stopping"
  | "discarding"
  | "editingTask"
  | "savingTask"
  | "restarting"
  // A command's real outcome is genuinely unknown (timeout/disconnect,
  // not a resolved reply) -- the backend may have actually applied it.
  // Every action stays blocked until an actual observed snapshot
  // resolves this, never a timer or a guess.
  | "uncertain";

/** The phases entered only while awaiting one command's own real reply. */
export type CommandPhase = "starting" | "stopping" | "discarding" | "savingTask" | "restarting";

/** The stable phase (idle/recording) considered "not busy" -- every other
 * phase blocks starting a different take-level command. */
type StablePhase = "idle" | "recording";

const STABLE_PHASES: ReadonlySet<RecordingPhase> = new Set<StablePhase>(["idle", "recording"]);

const SUCCESS_TARGET: Record<CommandPhase, RecordingPhase> = {
  starting: "recording",
  stopping: "idle",
  discarding: "idle",
  savingTask: "idle",
  restarting: "idle",
};

export interface RecordingUiState {
  phase: RecordingPhase;
  /** The phase to revert to if the current in-flight command is rejected; set on COMMAND_STARTED, consumed by COMMAND_REJECTED. */
  previousPhase: RecordingPhase | null;
  /** Which command was in flight when it went "uncertain" -- set on
   * COMMAND_UNKNOWN, consumed by SNAPSHOT_OBSERVED once a real fact
   * resolves it. Specifically distinguishes "it was savingTask" so a
   * save whose real outcome turned out NOT to have taken effect can
   * return the user to editingTask with their draft intact, rather than
   * silently losing their retry opportunity. */
  uncertainCommandPhase: CommandPhase | null;
  /** The staged, not-yet-committed task text -- only meaningful while phase is editingTask/savingTask. */
  taskDraft: string;
  /** The batch id live when editing began -- lets the component show a
   * "batch changed since you started editing" indication if a New batch
   * command (from this viewer or another) rotates the batch while the
   * draft is still unsaved, without silently retargeting/committing it. */
  taskEditBatchId: string | null;
  /** The real reply/error text for the most recently rejected or unknown command, or null. */
  feedback: string | null;
}

export type RecordingUiAction =
  | { type: "COMMAND_STARTED"; phase: CommandPhase }
  | { type: "COMMAND_SUCCEEDED" }
  /** `recording` is the latest known backend fact at settle time (not
   * necessarily from this command's own reply -- see the reducer case's
   * own doc for why reverting to `previousPhase` blindly isn't safe). */
  | { type: "COMMAND_REJECTED"; feedback: string; recording: boolean }
  /** The command's real outcome could not be determined (timeout/
   * disconnect) -- distinct from a resolved rejection: the backend may
   * have actually applied it. Lands on "uncertain" until an actual
   * observed snapshot resolves it (see SNAPSHOT_OBSERVED). */
  | { type: "COMMAND_UNKNOWN"; feedback: string }
  | { type: "BEGIN_TASK_EDIT"; currentText: string; currentBatchId: string | null }
  | { type: "UPDATE_TASK_DRAFT"; value: string }
  | { type: "CANCEL_TASK_EDIT"; recording: boolean }
  /** One observed fact from the backend's own snapshot -- reconciles
   * `phase` with another viewer's command, or confirms this viewer's own.
   * Ignored while a command of THIS viewer's own is in flight, so it can
   * never clobber that command's own pending COMMAND_SUCCEEDED/REJECTED
   * -- EXCEPT while "uncertain", which has no pending command of its own
   * left to clobber and exists precisely to be resolved by this.
   * `taskText` is the resolved snapshot's own committed task text, used
   * only to tell whether an uncertain savingTask actually took effect. */
  | { type: "SNAPSHOT_OBSERVED"; recording: boolean; taskText: string | null };

/** The stable phase a known `recording` fact maps to. */
function stablePhaseFor(recording: boolean): StablePhase {
  return recording ? "recording" : "idle";
}

export const INITIAL_RECORDING_UI_STATE: RecordingUiState = {
  phase: "idle",
  previousPhase: null,
  uncertainCommandPhase: null,
  taskDraft: "",
  taskEditBatchId: null,
  feedback: null,
};

/** Pure interaction-phase transition; never touches a snapshot, socket, or timer. */
export function recordingUiReducer(state: RecordingUiState, action: RecordingUiAction): RecordingUiState {
  switch (action.type) {
    case "COMMAND_STARTED":
      return { ...state, phase: action.phase, previousPhase: state.phase, feedback: null };

    case "COMMAND_SUCCEEDED": {
      if (STABLE_PHASES.has(state.phase) || state.phase === "editingTask") return state;
      const next = SUCCESS_TARGET[state.phase as CommandPhase];
      return { ...state, phase: next, previousPhase: null, taskEditBatchId: next === "idle" ? null : state.taskEditBatchId };
    }

    case "COMMAND_REJECTED": {
      if (state.previousPhase === null) return state;
      // A rejection means nothing of THIS command's own took effect, but
      // time still passed while it was in flight -- another viewer's
      // command could have changed what's actually true. Reverting to the
      // raw `previousPhase` would reapply a possibly-outdated fact, so a
      // take-level revert (previousPhase idle/recording) uses the latest
      // known fact instead. Reverting FROM savingTask always goes back to
      // editingTask regardless -- the draft stays open either way, not
      // reconciled against recording/idle.
      const target = state.previousPhase === "editingTask" ? "editingTask" : stablePhaseFor(action.recording);
      return { ...state, phase: target, previousPhase: null, feedback: action.feedback };
    }

    case "COMMAND_UNKNOWN":
      // `state.phase` here IS the CommandPhase that was in flight --
      // COMMAND_UNKNOWN is only ever dispatched while one is (see
      // RecordingPanel's runCommand).
      return {
        ...state,
        phase: "uncertain",
        previousPhase: null,
        uncertainCommandPhase: state.phase as CommandPhase,
        feedback: action.feedback,
      };

    case "BEGIN_TASK_EDIT":
      return state.phase === "idle"
        ? { ...state, phase: "editingTask", taskDraft: action.currentText, taskEditBatchId: action.currentBatchId, feedback: null }
        : state;

    case "UPDATE_TASK_DRAFT":
      return state.phase === "editingTask" ? { ...state, taskDraft: action.value } : state;

    case "CANCEL_TASK_EDIT": {
      if (state.phase !== "editingTask") return state;
      // Same reconciliation as a rejection leaving editingTask: the batch
      // could genuinely be recording by now (another viewer), and Escape
      // must land on that real fact, not blindly on "idle".
      return { ...state, phase: stablePhaseFor(action.recording), taskEditBatchId: null, feedback: null };
    }

    case "SNAPSHOT_OBSERVED": {
      if (state.phase !== "uncertain" && !STABLE_PHASES.has(state.phase)) return state;
      if (state.phase === "uncertain" && state.uncertainCommandPhase === "savingTask") {
        // Can't tell confidently that the save applied unless the
        // resolved fact matches the draft exactly -- treat anything else
        // (including a still-unset task) as "didn't take effect" and
        // hand the user back their draft to retry, rather than silently
        // discarding it.
        if (action.taskText === state.taskDraft) {
          return { ...state, phase: stablePhaseFor(action.recording), uncertainCommandPhase: null, taskEditBatchId: null };
        }
        return {
          ...state,
          phase: "editingTask",
          uncertainCommandPhase: null,
          feedback: "Previous save's outcome is now known: it did not take effect -- edit and save again.",
        };
      }
      const target = stablePhaseFor(action.recording);
      if (target === state.phase && state.uncertainCommandPhase === null) return state;
      return { ...state, phase: target, uncertainCommandPhase: null };
    }

    default:
      return state;
  }
}
