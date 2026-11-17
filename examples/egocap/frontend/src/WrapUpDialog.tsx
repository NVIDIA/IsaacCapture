// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * The end-of-day dialog: what was recorded today at the station in use and what became of each take, the
 * station's active and down time with what blocked it and the operator's notes, correcting a task after
 * recording, and saving the report on the station.
 *
 * Closing would drop notes that are not saved and a correction that is half typed, so while there are any the
 * dialog asks before it closes, whichever way the close was asked for.
 *
 * Notes are written for one station's day. They are saved for the station of the report they were typed over,
 * and the backend refuses them when another station is in use by then; the operator reloads the report, sees
 * which station it is now for, and saves again.
 *
 * Everything shown comes from the backend's report, which reads the recording service's own manifests: a
 * rejected, discarded or interrupted take is listed with its real outcome, never left out. The dialog itself
 * decides nothing about a take, and the station's time is the backend's own log, not something set here.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import {
  DISPOSITIONS,
  correctSession,
  fetchDayReport,
  saveDayReport,
  type Correction,
  type CorrectionReply,
  type DayReportReply,
  type ReportEpisode,
  type ReportSession,
  type TakeDisposition,
} from "./dayReportApi";
import { formatElapsed } from "./RecordingPanel";
import { sentence } from "./sentence";
import { SettingsHttpError } from "./settingsApi";
import { Card, errorMessage, useAction, useEscapeToClose, useModalFocus, type Loaded } from "./settingsParts";
import { saveStationNotes, type StationUptime, type UptimeBlocker } from "./stationsApi";
import { formatActivity, formatTotal } from "./stationTime";

/** The calls the dialog makes; the page uses the real ones and tests pass fakes. */
export interface WrapUpBackend {
  load: (controlToken: string, signal: AbortSignal) => Promise<DayReportReply>;
  save: (controlToken: string) => Promise<DayReportReply>;
  correct: (controlToken: string, correction: Correction) => Promise<CorrectionReply>;
  /** Saves the day's notes for `station`; rejects with a 409 `SettingsHttpError` when that is no longer the station in use. */
  saveNotes: (controlToken: string, station: string, notes: string) => Promise<StationUptime>;
}

/** Notes as typed, with the station whose day they were typed for. */
interface NotesDraft {
  station: string;
  text: string;
}

export const REAL_WRAP_UP_BACKEND: WrapUpBackend = { load: fetchDayReport, save: saveDayReport, correct: correctSession, saveNotes: saveStationNotes };

/** The longest notes the backend accepts, so the field stops where a save would be refused. */
const MAX_NOTES_CHARS = 4000;

/** Each outcome in the operator's words. */
const DISPOSITION_LABELS: Record<TakeDisposition, string> = {
  kept: "Kept",
  flagged: "Flagged",
  rejected: "Rejected",
  discarded: "Discarded",
  pending: "Pending",
  recording: "Recording",
};

/** A take's episode number where its data is: the trailing number of its location. */
function episodeNumber(episode: ReportEpisode): string {
  return episode.location.slice(episode.location.lastIndexOf("_") + 1);
}

/** The session or routed group holding a take's data: its location without the episode. */
function holderOf(episode: ReportEpisode): string {
  return episode.location.slice(0, episode.location.lastIndexOf("/"));
}

/** The clock time a session was opened, from its `YYYY-MM-DDTHH:MM:SS` start. */
function startedAt(session: ReportSession): string {
  return session.started_at === null ? "" : session.started_at.slice(11, 16);
}

/** A local clock time, in the 24-hour form the session list uses. */
function clockTime(epochSeconds: number): string {
  const at = new Date(epochSeconds * 1000);
  return `${String(at.getHours()).padStart(2, "0")}:${String(at.getMinutes()).padStart(2, "0")}`;
}

export interface WrapUpDialogProps {
  open: boolean;
  onClose: () => void;
  /** The run's control token; the report is not readable without it. */
  controlToken: string | null;
  backend?: WrapUpBackend;
}

/** Show the day's takes, the station's time and notes, and the correction and report actions. */
export function WrapUpDialog({ open, onClose, controlToken, backend = REAL_WRAP_UP_BACKEND }: WrapUpDialogProps) {
  const dialogRef = useModalFocus(open);
  const [state, setState] = useState<Loaded<DayReportReply>>({ kind: "loading" });
  // The notes as typed, kept here so that reading the report again never replaces them. Null while the operator
  // has typed nothing: the field then shows what is saved.
  const [notesDraft, setNotesDraft] = useState<NotesDraft | null>(null);
  // Whether the backend refused the notes because another station is in use than the one they were typed for.
  const [stationChanged, setStationChanged] = useState(false);
  // Set by the reload the operator asks for after that refusal: the report it brings is the one they then save
  // the notes over, so the draft moves to that report's station. No other read moves it.
  const adoptStation = useRef(false);
  // Bumped after a correction or saved notes, so the report shows what is now on disk.
  const [generation, setGeneration] = useState(0);
  // Why the latest read failed. A report already on screen stays there with this beside it: the correction or
  // notes save that asked for the read succeeded, and the notes draft is only reachable through the report.
  const [readFailure, setReadFailure] = useState<string | null>(null);
  const reload = useCallback(() => setGeneration((n) => n + 1), []);
  const { load } = backend;
  // The sessions whose correction form holds typed input. Only consulted when a close is asked for, so a ref.
  const typedCorrections = useRef(new Set<string>());
  /** Note whether a session's correction form holds typed input. */
  const noteCorrectionTyped = useCallback((sessionId: string, typed: boolean) => {
    if (typed) typedCorrections.current.add(sessionId);
    else {
      typedCorrections.current.delete(sessionId);
      // An applied or cleared correction leaves nothing of its own to discard, so the question goes with it.
      if (typedCorrections.current.size === 0) setConfirmingDiscard(false);
    }
  }, []);
  // Whether the header is asking to discard what is unsaved instead of offering Close.
  const [confirmingDiscard, setConfirmingDiscard] = useState(false);
  const closeButton = useRef<HTMLButtonElement>(null);
  const keepButton = useRef<HTMLButtonElement>(null);
  /** Close, or ask first while notes are unsaved or a correction is typed. */
  const requestClose = useCallback(() => {
    if (notesDraft !== null || typedCorrections.current.size > 0) setConfirmingDiscard(true);
    else onClose();
  }, [notesDraft, onClose]);
  // Escape asks to close like the button does; over the question it answers "keep editing", never "discard".
  useEscapeToClose(open, confirmingDiscard ? () => setConfirmingDiscard(false) : requestClose);
  useEffect(() => {
    if (!confirmingDiscard) return;
    // The question replaces Close in the header: focus goes to its safe answer, and afterwards back to where
    // the operator was, or to Close when that control is gone.
    const previous = document.activeElement;
    keepButton.current?.focus();
    return () => {
      if (previous instanceof HTMLElement && previous.isConnected && previous !== document.body) previous.focus();
      else closeButton.current?.focus();
    };
  }, [confirmingDiscard]);

  useEffect(() => {
    if (!open || controlToken === null) return;
    const controller = new AbortController();
    // A refresh keeps the last report on screen instead of flashing back to "Loading".
    setState((current) => (current.kind === "loaded" ? current : { kind: "loading" }));
    load(controlToken, controller.signal).then(
      (value) => {
        if (controller.signal.aborted) return;
        setState({ kind: "loaded", value });
        setReadFailure(null);
        if (adoptStation.current) {
          adoptStation.current = false;
          setNotesDraft((draft) => (draft === null ? null : { ...draft, station: value.station_in_use }));
        }
      },
      (error: unknown) => {
        if (controller.signal.aborted) return;
        const message = errorMessage(error);
        setState((current) => (current.kind === "loaded" ? current : { kind: "error", message }));
        setReadFailure(message);
      },
    );
    return () => controller.abort();
  }, [open, controlToken, load, generation]);

  if (!open) return null;
  const reply = state.kind === "loaded" ? state.value : null;
  /** Show the saved notes at once, then read the report again so all of it is what is on disk. */
  const notesSaved = (uptime: StationUptime) => {
    setState((current) => (current.kind === "loaded" ? { kind: "loaded", value: { ...current.value, uptime } } : current));
    setNotesDraft(null);
    setStationChanged(false);
    // Saved notes are no longer something to discard.
    if (typedCorrections.current.size === 0) setConfirmingDiscard(false);
    reload();
  };
  /** Read the report for the station now in use, keeping the notes as typed to be saved over it. */
  const reloadForStation = () => {
    adoptStation.current = true;
    setStationChanged(false);
    reload();
  };
  return (
    <>
      <div className="settings-backdrop" onClick={requestClose} aria-hidden="true" />
      <div ref={dialogRef} tabIndex={-1} className="settings-modal wrap-up-modal" role="dialog" aria-modal="true" aria-label="End of day">
        <div className="settings-content">
          <header className="settings-content-header">
            <h3>End of day{reply !== null && ` — ${reply.report.station ?? reply.station_in_use}, ${reply.report.date}`}</h3>
            {confirmingDiscard ? (
              <div className="wrap-up-discard" role="group" aria-label="Discard unsaved changes?">
                <span role="alert">Discard unsaved changes?</span>
                <button
                  type="button"
                  className="settings-danger"
                  onClick={() => {
                    setConfirmingDiscard(false);
                    onClose();
                  }}
                >
                  Discard
                </button>
                <button ref={keepButton} type="button" className="settings-quiet" onClick={() => setConfirmingDiscard(false)}>
                  Keep editing
                </button>
              </div>
            ) : (
              <button ref={closeButton} type="button" className="settings-quiet" onClick={requestClose}>
                Close
              </button>
            )}
          </header>
          <div className="settings-content-body">
            {controlToken === null ? (
              <p className="settings-note settings-note--hint">Enter the control token in the recording panel to see the day's report.</p>
            ) : (
              <>
                {state.kind === "loading" && <p className="settings-note">Loading…</p>}
                {state.kind === "error" && <p className="settings-note tone-warning">The report is unavailable: {state.message}</p>}
                {reply !== null && (
                  <>
                    {readFailure !== null && (
                      <p className="settings-note tone-warning">
                        The report could not be read again, so it is shown as last read: {readFailure}{" "}
                        <button type="button" className="settings-quiet" onClick={reload}>
                          Read again
                        </button>
                      </p>
                    )}
                    <Totals reply={reply} />
                    <StationTime
                      uptime={reply.uptime}
                      // The report counts the takes; the time figures alone can all be zero after the first one.
                      hasTake={reply.report.totals.episodes > 0}
                      notes={notesDraft?.text ?? reply.uptime.notes}
                      // A draft stays with the station it was begun for, whatever report is read meanwhile.
                      station={notesDraft?.station ?? reply.station_in_use}
                      // Typing back to what is saved leaves nothing to keep.
                      onNotes={(text) => setNotesDraft((draft) => (text === reply.uptime.notes ? null : { station: draft?.station ?? reply.station_in_use, text }))}
                      controlToken={controlToken}
                      saveNotes={backend.saveNotes}
                      onSaved={notesSaved}
                      stationChanged={stationChanged}
                      onStationChanged={() => setStationChanged(true)}
                      onReload={reloadForStation}
                    />
                    <Sessions sessions={reply.report.sessions} controlToken={controlToken} correct={backend.correct} onCorrected={reload} onTyped={noteCorrectionTyped} />
                    <SaveReport controlToken={controlToken} save={backend.save} />
                  </>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </>
  );
}

/** The day's takes by outcome, each with its recorded time. */
function Totals({ reply }: { reply: DayReportReply }) {
  const { totals } = reply.report;
  return (
    <Card title="Takes" lead={`${totals.episodes} take${totals.episodes === 1 ? "" : "s"}, ${formatActivity(totals.seconds)} recorded.`}>
      <ul className="wrap-up-tiles" aria-label="Takes by outcome">
        {DISPOSITIONS.filter((name) => name !== "recording" || totals.by_disposition.recording.count > 0).map((name) => (
          <li key={name} className={`wrap-up-tile wrap-up-tile--${name}`}>
            <span className="wrap-up-tile-count">{totals.by_disposition[name].count}</span>
            <span className="wrap-up-tile-label">{DISPOSITION_LABELS[name]}</span>
            <span className="wrap-up-tile-time">{formatActivity(totals.by_disposition[name].seconds)}</span>
          </li>
        ))}
      </ul>
    </Card>
  );
}

/** One stretch the station could not record: when, for how long, and why. */
function BlockerLine({ blocker }: { blocker: UptimeBlocker }) {
  return (
    <li>
      {clockTime(blocker.start)}–{blocker.end === null ? "now" : clockTime(blocker.end)} · {formatActivity(blocker.seconds)} · {sentence(blocker.reason)}
    </li>
  );
}

/** The station's active and down time today, what blocked it, and the operator's notes. */
function StationTime({
  uptime,
  hasTake,
  notes,
  station,
  onNotes,
  controlToken,
  saveNotes,
  onSaved,
  stationChanged,
  onStationChanged,
  onReload,
}: {
  uptime: StationUptime;
  /** Whether the day has a take; before one there is no time to show. */
  hasTake: boolean;
  /** The notes as the field shows them: the operator's draft, or what is saved. */
  notes: string;
  /** The station the notes are saved for. */
  station: string;
  onNotes: (notes: string) => void;
  controlToken: string;
  saveNotes: WrapUpBackend["saveNotes"];
  onSaved: (uptime: StationUptime) => void;
  /** Whether the last save was refused because another station is in use by now. */
  stationChanged: boolean;
  onStationChanged: () => void;
  /** Reads the report for the station now in use, keeping the notes as typed. */
  onReload: () => void;
}) {
  const [{ busy, failure }, run] = useAction();
  const dirty = notes !== uptime.notes;
  /** Save the draft; a save already on its way is not sent twice. */
  const save = () => {
    if (busy) return;
    run(() =>
      saveNotes(controlToken, station, notes).then(onSaved, (error: unknown) => {
        // That refusal has its own words and a way out, shown in place of the backend's reason.
        if (!(error instanceof SettingsHttpError) || error.status !== 409) throw error;
        onStationChanged();
      }),
    );
  };
  return (
    <Card title="Station time" lead="Logged by the backend between the first and the last take of the day: active while the station could record, down while something held it.">
      <p className="wrap-up-time">{hasTake ? `Active ${formatTotal(uptime.active_seconds)} · Down ${formatTotal(uptime.down_seconds)}` : "No take yet today."}</p>
      {uptime.blockers.length > 0 && (
        <ul className="wrap-up-blockers" aria-label="Blockers">
          {uptime.blockers.map((blocker, index) => (
            <BlockerLine key={`${blocker.start}:${index}`} blocker={blocker} />
          ))}
        </ul>
      )}
      <label className="wrap-up-notes">
        Notes
        <textarea
          className="settings-input wrap-up-notes-text"
          value={notes}
          maxLength={MAX_NOTES_CHARS}
          rows={3}
          // Read-only, not disabled, while saving: a disabled field gives up keyboard focus, which would leave
          // the operator outside the notes when the save is refused and they are still to be corrected.
          readOnly={busy}
          onChange={(e) => onNotes(e.target.value)}
        />
      </label>
      <div className="settings-actions">
        {/* Not disabled while saving, for the same reason as the field. */}
        <button type="button" className="settings-primary" disabled={!dirty} aria-busy={busy} onClick={save}>
          {busy ? "Saving…" : "Save notes"}
        </button>
        {stationChanged ? (
          <span className="settings-note tone-warning" role="alert">
            The station in use changed; reload the report.{" "}
            <button type="button" className="settings-quiet" onClick={onReload}>
              Reload
            </button>
          </span>
        ) : (
          failure !== null && <span className="settings-note tone-warning">Not saved: {failure}</span>
        )}
        {!dirty && uptime.notes !== "" && failure === null && <span className="settings-note tone-positive">✓ Saved</span>}
      </div>
    </Card>
  );
}

/** Every session of the day, each opening to its takes. */
function Sessions({
  sessions,
  controlToken,
  correct,
  onCorrected,
  onTyped,
}: {
  sessions: ReportSession[];
  controlToken: string;
  correct: WrapUpBackend["correct"];
  onCorrected: () => void;
  /** Told which session's correction form holds typed input, and when it no longer does. */
  onTyped: (sessionId: string, typed: boolean) => void;
}) {
  return (
    <Card title="Sessions" lead="Each recording session of the day and its takes. A take's task can be corrected here; its video and tracking stay as recorded.">
      {sessions.length === 0 ? (
        <p className="settings-note">Nothing was recorded today.</p>
      ) : (
        sessions.map((session, index) => (
          // The latest session opens by default: it is the one an operator is finishing.
          <SessionDetails key={session.session_id} session={session} open={index === sessions.length - 1} controlToken={controlToken} correct={correct} onCorrected={onCorrected} onTyped={onTyped} />
        ))
      )}
    </Card>
  );
}

/** One session: its counts in the summary line, its takes and the correction form beneath. */
function SessionDetails({
  session,
  open,
  controlToken,
  correct,
  onCorrected,
  onTyped,
}: {
  session: ReportSession;
  open: boolean;
  controlToken: string;
  correct: WrapUpBackend["correct"];
  onCorrected: () => void;
  onTyped: (sessionId: string, typed: boolean) => void;
}) {
  const counts = DISPOSITIONS.filter((name) => session.by_disposition[name].count > 0).map((name) => `${session.by_disposition[name].count} ${DISPOSITION_LABELS[name].toLowerCase()}`);
  // Only a take that was finalized has recorded data whose task can be relabelled.
  const holders = [...new Set(session.episodes.filter((episode) => episode.disposition === "kept" || episode.disposition === "flagged").map(holderOf))];
  const { session_id: sessionId } = session;
  // A routed take is numbered by its slot in its group and an unfinished one by its place in the session, and
  // each group numbers from zero: once any take is held outside the session, a number alone does not name a take.
  const routed = session.episodes.some((episode) => holderOf(episode) !== sessionId);
  const noteTyped = useCallback((typed: boolean) => onTyped(sessionId, typed), [onTyped, sessionId]);
  return (
    <details className="wrap-up-session" open={open}>
      <summary>
        <span className="settings-list-name">{startedAt(session) === "" ? session.session_id : `${startedAt(session)} · ${session.session_id}`}</span>
        <span className="settings-note">
          {[session.station, ...counts].filter((part) => part !== null).join(" · ")}
        </span>
      </summary>
      {session.problems.map((problem) => (
        <p key={problem} className="settings-note tone-warning">
          Unreadable take record: {problem}
        </p>
      ))}
      {session.episodes.length > 0 && (
        <div className="data-table-wrapper">
          <table className="data-table wrap-up-table">
            <thead>
              <tr>
                <th scope="col" className="num-cell">
                  Take
                </th>
                {routed && <th scope="col">Recorded under</th>}
                <th scope="col">Outcome</th>
                <th scope="col" className="num-cell">
                  Time
                </th>
                <th scope="col">Task</th>
                <th scope="col">Operator</th>
                <th scope="col">Issues</th>
              </tr>
            </thead>
            <tbody>
              {session.episodes.map((episode) => (
                <tr key={episode.episode_id}>
                  <td className="num-cell">{episodeNumber(episode)}</td>
                  {routed && <td>{holderOf(episode)}</td>}
                  <td>
                    <span className={`wrap-up-outcome wrap-up-outcome--${episode.disposition}`}>{DISPOSITION_LABELS[episode.disposition]}</span>
                  </td>
                  <td className="num-cell">{formatElapsed(episode.duration_seconds * 1000)}</td>
                  <td title={episode.task_details === "" ? undefined : episode.task_details}>{episode.task}</td>
                  <td>{episode.operator}</td>
                  <td>{episode.issues.length === 0 ? "—" : episode.issues.join("; ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {holders.length > 0 && <CorrectionForm holders={holders} controlToken={controlToken} correct={correct} onCorrected={onCorrected} onTyped={noteTyped} />}
    </details>
  );
}

/** Correct the task of one take or a range of takes of a session. */
function CorrectionForm({
  holders,
  controlToken,
  correct,
  onCorrected,
  onTyped,
}: {
  /** Where the session's finalized takes are: the session itself, or each routed group it filled. */
  holders: string[];
  controlToken: string;
  correct: WrapUpBackend["correct"];
  onCorrected: () => void;
  /** Told whether the form holds typed input, which closing the dialog would drop. */
  onTyped: (typed: boolean) => void;
}) {
  const [holder, setHolder] = useState(holders[0]);
  const [episodes, setEpisodes] = useState("");
  const [task, setTask] = useState("");
  const [done, setDone] = useState<string | null>(null);
  const [{ busy, failure }, run] = useAction();
  const ready = episodes.trim() !== "" && task.trim() !== "" && !busy;
  const typed = episodes !== "" || task !== "";
  useEffect(() => {
    onTyped(typed);
    // A form that leaves the report has nothing left to lose.
    return () => onTyped(false);
  }, [typed, onTyped]);
  /** Apply the correction, say what it changed, and have the report read again. */
  const apply = () =>
    run(async () => {
      setDone(null);
      const reply = await correct(controlToken, { session: holder, episodes: episodes.trim(), task: task.trim() });
      const skipped = reply.missing.length === 0 ? "" : ` Not found: ${reply.missing.join(", ")}.`;
      // With more than one group the take numbers repeat, so the confirmation says which group was changed.
      const where = holders.length > 1 ? ` in ${holder}` : "";
      setDone(`Corrected ${reply.updated.length === 0 ? "nothing" : reply.updated.join(", ")}${where}.${skipped}`);
      setEpisodes("");
      setTask("");
      onCorrected();
    });
  return (
    <fieldset className="settings-fieldset wrap-up-correction">
      <legend>Correct a task</legend>
      {holders.length > 1 && (
        <select className="settings-input" aria-label="Recorded under" value={holder} disabled={busy} onChange={(e) => setHolder(e.target.value)}>
          {holders.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
      )}
      <div className="settings-actions">
        <input className="settings-input wrap-up-range" aria-label="Takes to correct" placeholder="Take or range, e.g. 3 or 2-7" value={episodes} disabled={busy} onChange={(e) => setEpisodes(e.target.value)} />
        <input className="settings-input wrap-up-task" aria-label="Correct task" placeholder="The task these takes actually show" value={task} maxLength={1000} disabled={busy} onChange={(e) => setTask(e.target.value)} />
        <button type="button" className="settings-quiet" disabled={!ready} onClick={apply}>
          {busy ? "Correcting…" : "Correct"}
        </button>
      </div>
      {failure !== null && <p className="settings-note tone-warning">Not corrected: {failure}</p>}
      {done !== null && failure === null && <p className="settings-note">{done}</p>}
    </fieldset>
  );
}

/** Save the report as a file on the station and say what was written. */
function SaveReport({ controlToken, save }: { controlToken: string; save: WrapUpBackend["save"] }) {
  const [saved, setSaved] = useState<string | null>(null);
  const [{ busy, failure }, run] = useAction();
  /** Write the report and name the files, so the operator knows where the day's record is. */
  const write = () =>
    run(async () => {
      const reply = await save(controlToken);
      const files = [reply.saved?.report, reply.saved?.activity_log].filter((name): name is string => typeof name === "string");
      setSaved(`Saved ${files.join(" and ")} in station_status under the recording output.`);
    });
  return (
    <div className="settings-actions">
      <button type="button" className="settings-primary" disabled={busy} onClick={write}>
        {busy ? "Saving…" : "Save report"}
      </button>
      {failure !== null && <span className="settings-note tone-warning">Not saved: {failure}</span>}
      {saved !== null && failure === null && <span className="settings-note">{saved}</span>}
    </div>
  );
}
