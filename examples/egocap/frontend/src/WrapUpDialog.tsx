// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * The end-of-day dialog: what was recorded today and what became of each take, the station's active and blocked
 * time with the operator's notes, saving the report on the station, and correcting a task after recording.
 *
 * Everything shown comes from the backend's report, which reads the recording service's own manifests: a
 * rejected, discarded or interrupted take is listed with its real outcome, never left out. The dialog itself
 * decides nothing about a take.
 */
import { useCallback, useEffect, useState } from "react";
import {
  DISPOSITIONS,
  correctSession,
  fetchDayReport,
  saveDayReport,
  type Correction,
  type CorrectionReply,
  type DayReportQuery,
  type DayReportReply,
  type ReportEpisode,
  type ReportSession,
  type TakeDisposition,
} from "./dayReportApi";
import { formatElapsed } from "./RecordingPanel";
import { Card, errorMessage, useAction, useEscapeToClose, useModalFocus } from "./settingsParts";
import { formatActivity } from "./StationActivityToggle";
import { changeStationActivity, type StationActivity } from "./stationsApi";

/** The calls the dialog makes; the page uses the real ones and tests pass fakes. */
export interface WrapUpBackend {
  load: (controlToken: string, query: DayReportQuery, signal: AbortSignal) => Promise<DayReportReply>;
  save: (controlToken: string, query: DayReportQuery) => Promise<DayReportReply>;
  correct: (controlToken: string, correction: Correction) => Promise<CorrectionReply>;
  saveNotes: (controlToken: string, change: { notes: string }) => Promise<StationActivity>;
}

export const REAL_WRAP_UP_BACKEND: WrapUpBackend = { load: fetchDayReport, save: saveDayReport, correct: correctSession, saveNotes: changeStationActivity };

/** Each outcome in the operator's words. */
const DISPOSITION_LABELS: Record<TakeDisposition, string> = {
  kept: "Kept",
  flagged: "Flagged",
  rejected: "Rejected",
  discarded: "Discarded",
  pending: "Pending",
  recording: "Recording",
};

/** The `station` value that asks the backend for every station. */
const ALL_STATIONS = "*";

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

type Loaded = { kind: "loading" } | { kind: "error"; message: string } | { kind: "loaded"; value: DayReportReply };

export interface WrapUpDialogProps {
  open: boolean;
  onClose: () => void;
  /** The run's control token; the report is not readable without it. */
  controlToken: string | null;
  backend?: WrapUpBackend;
}

/** Show the day's takes, the station's activity and notes, and the report and correction actions. */
export function WrapUpDialog({ open, onClose, controlToken, backend = REAL_WRAP_UP_BACKEND }: WrapUpDialogProps) {
  const dialogRef = useModalFocus(open);
  useEscapeToClose(open, onClose);
  const [state, setState] = useState<Loaded>({ kind: "loading" });
  const [date, setDate] = useState("");
  const [everyStation, setEveryStation] = useState(false);
  // Bumped after a correction or saved notes, so the report shows what is now on disk.
  const [generation, setGeneration] = useState(0);
  const reload = useCallback(() => setGeneration((n) => n + 1), []);
  const { load } = backend;

  useEffect(() => {
    if (!open || controlToken === null) return;
    const controller = new AbortController();
    // A refresh keeps the last report on screen instead of flashing back to "Loading".
    setState((current) => (current.kind === "loaded" ? current : { kind: "loading" }));
    load(controlToken, { date, station: everyStation ? ALL_STATIONS : undefined }, controller.signal).then(
      (value) => setState({ kind: "loaded", value }),
      (error: unknown) => {
        if (!controller.signal.aborted) setState({ kind: "error", message: errorMessage(error) });
      },
    );
    return () => controller.abort();
  }, [open, controlToken, load, date, everyStation, generation]);

  if (!open) return null;
  const reply = state.kind === "loaded" ? state.value : null;
  return (
    <>
      <div className="settings-backdrop" onClick={onClose} aria-hidden="true" />
      <div ref={dialogRef} tabIndex={-1} className="settings-modal wrap-up-modal" role="dialog" aria-modal="true" aria-label="End of day">
        <div className="settings-content">
          <header className="settings-content-header">
            <h3>End of day{reply !== null && ` — ${reply.report.station ?? "every station"}, ${reply.report.date}`}</h3>
            <button type="button" className="settings-quiet" onClick={onClose}>
              Close
            </button>
          </header>
          <div className="settings-content-body">
            {controlToken === null ? (
              <p className="settings-note settings-note--hint">Enter the control token in the recording panel to see the day's report.</p>
            ) : (
              <>
                <div className="settings-actions">
                  <label className="wrap-up-filter">
                    Day
                    <input type="date" className="settings-input" value={date === "" ? (reply?.today ?? "") : date} max={reply?.today} onChange={(e) => setDate(e.target.value)} />
                  </label>
                  <label className="wrap-up-filter">
                    <input type="checkbox" checked={everyStation} onChange={(e) => setEveryStation(e.target.checked)} />
                    Every station
                  </label>
                </div>
                {state.kind === "loading" && <p className="settings-note">Loading…</p>}
                {state.kind === "error" && <p className="settings-note tone-warning">The report is unavailable: {state.message}</p>}
                {reply !== null && (
                  <>
                    <Totals reply={reply} />
                    {reply.activity !== null && <ActivityCard activity={reply.activity} controlToken={controlToken} saveNotes={backend.saveNotes} onSaved={reload} />}
                    <Sessions sessions={reply.report.sessions} controlToken={controlToken} correct={backend.correct} onCorrected={reload} />
                    <SaveReport controlToken={controlToken} query={{ date, station: everyStation ? ALL_STATIONS : undefined }} save={backend.save} />
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

/** The station's active and blocked time today, and the operator's notes on what blocked it. */
function ActivityCard({
  activity,
  controlToken,
  saveNotes,
  onSaved,
}: {
  activity: StationActivity;
  controlToken: string;
  saveNotes: WrapUpBackend["saveNotes"];
  onSaved: () => void;
}) {
  const [notes, setNotes] = useState(activity.notes);
  const [{ busy, failure }, run] = useAction();
  // What is saved replaces the draft only when the two were the same, so a reload never eats typing.
  useEffect(() => setNotes((draft) => (draft === "" || draft === activity.notes ? activity.notes : draft)), [activity.notes]);
  const dirty = notes !== activity.notes;
  return (
    <Card title="Station activity" lead={`${activity.station} is ${activity.state === "active" ? "active" : activity.since === null ? "not started" : "blocked"} now.`}>
      <dl className="kv-list">
        <div>
          <dt>Active</dt>
          <dd>{formatActivity(activity.active_seconds)}</dd>
        </div>
        <div>
          <dt>Blocked</dt>
          <dd>{formatActivity(activity.inactive_seconds)}</dd>
        </div>
        <div>
          <dt>Blockers</dt>
          <dd>{activity.blocker_count}</dd>
        </div>
      </dl>
      <label className="wrap-up-notes">
        Notes and blockers
        <textarea className="settings-input wrap-up-notes-text" value={notes} maxLength={4000} rows={3} disabled={busy} onChange={(e) => setNotes(e.target.value)} />
      </label>
      <div className="settings-actions">
        <button type="button" className="settings-primary" disabled={!dirty || busy} onClick={() => run(() => saveNotes(controlToken, { notes }).then(onSaved))}>
          {busy ? "Saving…" : "Save notes"}
        </button>
        {failure !== null && <span className="settings-note tone-warning">Not saved: {failure}</span>}
        {!dirty && activity.notes !== "" && failure === null && <span className="settings-note settings-ready-item--ok">✓ Saved</span>}
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
}: {
  sessions: ReportSession[];
  controlToken: string;
  correct: WrapUpBackend["correct"];
  onCorrected: () => void;
}) {
  return (
    <Card title="Sessions" lead="Each recording session of the day and its takes. A take's task can be corrected here; its video and tracking stay as recorded.">
      {sessions.length === 0 ? (
        <p className="settings-note">Nothing was recorded on this day.</p>
      ) : (
        sessions.map((session, index) => (
          // The latest session opens by default: it is the one an operator is finishing.
          <SessionDetails key={session.session_id} session={session} open={index === sessions.length - 1} controlToken={controlToken} correct={correct} onCorrected={onCorrected} />
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
}: {
  session: ReportSession;
  open: boolean;
  controlToken: string;
  correct: WrapUpBackend["correct"];
  onCorrected: () => void;
}) {
  const counts = DISPOSITIONS.filter((name) => session.by_disposition[name].count > 0).map((name) => `${session.by_disposition[name].count} ${DISPOSITION_LABELS[name].toLowerCase()}`);
  // Only a take that was finalized has recorded data whose task can be relabelled.
  const holders = [...new Set(session.episodes.filter((episode) => episode.disposition === "kept" || episode.disposition === "flagged").map(holderOf))];
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
                <th scope="col">Take</th>
                <th scope="col">Outcome</th>
                <th scope="col">Time</th>
                <th scope="col">Task</th>
                <th scope="col">Operator</th>
                <th scope="col">Issues</th>
              </tr>
            </thead>
            <tbody>
              {session.episodes.map((episode) => (
                <tr key={episode.episode_id}>
                  <td className="num-cell">{episodeNumber(episode)}</td>
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
      {holders.length > 0 && <CorrectionForm holders={holders} controlToken={controlToken} correct={correct} onCorrected={onCorrected} />}
    </details>
  );
}

/** Correct the task of one take or a range of takes of a session. */
function CorrectionForm({
  holders,
  controlToken,
  correct,
  onCorrected,
}: {
  /** Where the session's finalized takes are: the session itself, or each routed group it filled. */
  holders: string[];
  controlToken: string;
  correct: WrapUpBackend["correct"];
  onCorrected: () => void;
}) {
  const [holder, setHolder] = useState(holders[0]);
  const [episodes, setEpisodes] = useState("");
  const [task, setTask] = useState("");
  const [done, setDone] = useState<string | null>(null);
  const [{ busy, failure }, run] = useAction();
  const ready = episodes.trim() !== "" && task.trim() !== "" && !busy;
  /** Apply the correction, say what it changed, and have the report read again. */
  const apply = () =>
    run(async () => {
      setDone(null);
      const reply = await correct(controlToken, { session: holder, episodes: episodes.trim(), task: task.trim() });
      const skipped = reply.missing.length === 0 ? "" : ` Not found: ${reply.missing.join(", ")}.`;
      setDone(`Corrected ${reply.updated.length === 0 ? "nothing" : reply.updated.join(", ")}.${skipped}`);
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
function SaveReport({ controlToken, query, save }: { controlToken: string; query: DayReportQuery; save: WrapUpBackend["save"] }) {
  const [saved, setSaved] = useState<string | null>(null);
  const [{ busy, failure }, run] = useAction();
  /** Write the report and name the files, so the operator knows where the day's record is. */
  const write = () =>
    run(async () => {
      const reply = await save(controlToken, query);
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
