// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Owns WS /api/session: the browser connection/reconnect/liveness state
 * machine, the latest validated capture snapshot, the latest recording
 * snapshot (pushed proactively by the backend), and sending one small
 * bounded recording command at a time and awaiting its correlated reply.
 * Socket construction, close, and timers are effects owned by this hook;
 * the reducer itself is pure. Nothing is sent automatically during render
 * -- `sendRecordingCommand` only ever runs from an explicit caller.
 *
 * The browser transport and capture lifecycle are deliberately separate:
 * a connected socket can truthfully show "Capture stopped". A lost socket
 * never turns a remembered capture-running fact into a fabricated stop --
 * it is presented as stale by the caller (App.tsx), using `phase` and
 * `snapshot` together.
 */

import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import {
  BINARY_HEADER_SIZE,
  BINARY_KIND_TRACKING,
  CapturePreviewParseError,
  MAX_RECORDING_COMMAND_BYTES,
  parseBinaryHeader,
  parseSnapshot,
  type PreviewSnapshot,
  type TrackingStreamDescriptor,
} from "./captureProtocol";
import { decodeTracking, type DecodedTracking } from "./trackingDecoder";
import type {
  EpisodeDisposition,
  EpisodeResultSummary,
  RecordingActionName,
  RecordingCommandResult,
  RecordingSnapshot,
} from "./recordingTypes";

/** How long a command may wait for its reply before its outcome is treated as unknown. Stop retires the ingress
 * (up to ~32 s when a conversion is stuck), then seals, checks and finalizes the take, so it waits far longer than
 * the others. Discard also retires the ingress the same way. New batch instead refuses outright (CoordinatorBusyError)
 * if a previous take's ingress hasn't yet settled, so its own worst case is just the backend service's ~30 s close
 * bound, not ingress retirement -- both Discard and New batch get that backend bound plus a margin, not the same
 * budget as Start/set_task, which do far less work. These figures are budgets taken from the backend's own
 * deadlines, not measurements of a real take. */
const RECORDING_COMMAND_TIMEOUT_MS: Record<RecordingActionName, number> = {
  set_task: 5_000,
  start_episode: 30_000,
  stop_episode: 120_000,
  discard_episode: 45_000,
  new_batch: 45_000,
};

/** sessionStorage key for the entered control token -- scoped to this
 * tab's session, so it survives a reload. */
const CONTROL_TOKEN_STORAGE_KEY = "egocap.controlToken";

/** sessionStorage access can throw (site storage blocked/disabled, private
 * browsing in some browsers) -- a failure here falls back to behaving as
 * though nothing were stored, rather than crashing the whole page. */
function readStoredControlToken(): string | null {
  try {
    return sessionStorage.getItem(CONTROL_TOKEN_STORAGE_KEY);
  } catch {
    return null;
  }
}
function writeStoredControlToken(token: string): void {
  try {
    sessionStorage.setItem(CONTROL_TOKEN_STORAGE_KEY, token);
  } catch {
    // Falls back to in-memory-only for this tab; setControlTokenState still runs.
  }
}
function removeStoredControlToken(): void {
  try {
    sessionStorage.removeItem(CONTROL_TOKEN_STORAGE_KEY);
  } catch {
    // Nothing to clean up if sessionStorage itself is inaccessible.
  }
}

/** sessionStorage key for the `backend_instance_id` the current stored
 * token was last confirmed against -- lets a reload tell a still-valid
 * token apart from one left over from a backend that has since restarted
 * (see the effect that reads/writes this in useCaptureSession). */
const CONTROL_TOKEN_BACKEND_ID_STORAGE_KEY = "egocap.controlTokenBackendInstanceId";
function readStoredTokenBackendInstanceId(): string | null {
  try {
    return sessionStorage.getItem(CONTROL_TOKEN_BACKEND_ID_STORAGE_KEY);
  } catch {
    return null;
  }
}
function writeStoredTokenBackendInstanceId(id: string): void {
  try {
    sessionStorage.setItem(CONTROL_TOKEN_BACKEND_ID_STORAGE_KEY, id);
  } catch {
    // Falls back to in-memory-only for this tab.
  }
}
function removeStoredTokenBackendInstanceId(): void {
  try {
    sessionStorage.removeItem(CONTROL_TOKEN_BACKEND_ID_STORAGE_KEY);
  } catch {
    // Nothing to clean up if sessionStorage itself is inaccessible.
  }
}
const KNOWN_OUTCOMES = new Set<RecordingCommandResult["outcome"]>(["ok", "rejected"]);
const EPISODE_PHASES = new Set<RecordingSnapshot["episodePhase"]>([
  "IDLE",
  "STARTING",
  "RECORDING",
  "STOPPING",
  "FINALIZING",
  "RECOVERY_REQUIRED",
]);
const DISPOSITIONS = new Set<EpisodeDisposition>(["KEPT", "REJECTED", "DISCARDED"]);

/** Rejects a `sendRecordingCommand` promise specifically for a timeout or
 * lost connection -- a command actually sent whose real outcome is
 * genuinely unknown (the backend may have applied it before the reply was
 * lost), distinct from a confident local rejection (never sent at all,
 * e.g. not connected / already in flight / oversized). Callers use this to
 * show genuine uncertainty rather than a false "nothing happened". */
export class RecordingCommandUncertainError extends Error {}

interface PendingReply {
  requestId: string;
  resolve: (result: RecordingCommandResult) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

/** Thrown internally by parseRecordingSnapshot's field checks; never escapes it. */
class _InvalidRecordingSnapshot extends Error {}

/** Raises unless `value` is a string. */
function requireString(value: unknown): string {
  if (typeof value !== "string") throw new _InvalidRecordingSnapshot();
  return value;
}

/** Raises unless `value` is a string or null. */
function requireNullableString(value: unknown): string | null {
  return value === null ? null : requireString(value);
}

/** Raises unless `value` is one of `allowed`'s members. */
function requireEnum<T extends string>(value: unknown, allowed: ReadonlySet<T>): T {
  if (typeof value !== "string" || !allowed.has(value as T)) throw new _InvalidRecordingSnapshot();
  return value as T;
}

/** Raises unless `value` is a nonnegative safe integer. */
function requireNonNegativeSafeInt(value: unknown): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) throw new _InvalidRecordingSnapshot();
  return value;
}

/** Raises unless `value` is a boolean. */
function requireBoolean(value: unknown): boolean {
  if (typeof value !== "boolean") throw new _InvalidRecordingSnapshot();
  return value;
}

/** Raises unless `value` is a nonnegative safe integer or null. */
function requireNullableNonNegativeSafeInt(value: unknown): number | null {
  if (value === null) return null;
  return requireNonNegativeSafeInt(value);
}

/** Raises unless `value` is null or a well-shaped episode result object. */
function parseEpisodeResult(value: unknown): EpisodeResultSummary | null {
  if (value === null) return null;
  if (typeof value !== "object") throw new _InvalidRecordingSnapshot();
  const r = value as Record<string, unknown>;
  const issues = r.issues;
  if (!Array.isArray(issues) || !issues.every((issue) => typeof issue === "string")) throw new _InvalidRecordingSnapshot();
  return {
    episodeId: requireString(r.episode_id),
    disposition: requireEnum(r.disposition, DISPOSITIONS),
    issues,
    // Frozen at the moment this episode ended -- the one piece of
    // finished-episode timing data the UI actually renders, carried on
    // the existing result fact rather than a second parallel record.
    durationMs: requireNonNegativeSafeInt(r.duration_ms),
  };
}

/** Converts the backend's snake_case recording snapshot object into the
 * frontend's typed RecordingSnapshot, or null if any field is missing or
 * wrong-shaped -- this is real backend-produced data, but still untrusted
 * input to this browser process; a malformed field must never reach render. */
function parseRecordingSnapshot(raw: unknown): RecordingSnapshot | null {
  if (typeof raw !== "object" || raw === null) return null;
  const r = raw as Record<string, unknown>;
  try {
    return {
      batchId: requireString(r.batch_id),
      episodePhase: requireEnum(r.episode_phase, EPISODE_PHASES),
      currentEpisodeId: requireNullableString(r.current_episode_id),
      currentEpisodeElapsedMs: requireNullableNonNegativeSafeInt(r.current_episode_elapsed_ms),
      taskText: requireNullableString(r.task_text),
      defaultOperator: requireNullableString(r.default_operator),
      autoFillOperatorName: requireBoolean(r.auto_fill_operator_name),
      savedCount: requireNonNegativeSafeInt(r.saved_count),
      lastResult: parseEpisodeResult(r.last_result),
      readyToRecord: requireBoolean(r.ready_to_record),
      revision: requireNonNegativeSafeInt(r.revision),
    };
  } catch (err) {
    if (err instanceof _InvalidRecordingSnapshot) return null;
    throw err;
  }
}

/** True for a parsed message that is a recording-command reply -- checked
 * before ever attempting to parse the payload as a preview snapshot, so a
 * reply can never be misrouted into the snapshot path (or vice versa), and
 * never arms snapshot freshness/bootstrap -- it is a separate channel, not
 * a substitute heartbeat. */
function isRecordingReply(
  data: unknown,
): data is { type: "recording_reply"; request_id: string; outcome: string; reason: string | null; snapshot: unknown } {
  return (
    typeof data === "object" &&
    data !== null &&
    (data as { type?: unknown }).type === "recording_reply" &&
    typeof (data as { request_id?: unknown }).request_id === "string" &&
    typeof (data as { outcome?: unknown }).outcome === "string" &&
    KNOWN_OUTCOMES.has((data as { outcome: string }).outcome as RecordingCommandResult["outcome"])
  );
}

/** True for a proactively-pushed recording snapshot (not correlated to any
 * one command) -- same routing priority as a reply, checked before preview
 * snapshot parsing. */
function isRecordingSnapshotMessage(data: unknown): data is { type: "recording_snapshot"; snapshot: unknown } {
  return typeof data === "object" && data !== null && (data as { type?: unknown }).type === "recording_snapshot";
}

export type ConnectionPhase = "connecting" | "synchronizing" | "connected" | "reconnecting" | "unavailable";

export interface ConnectionState {
  phase: ConnectionPhase;
  snapshot: PreviewSnapshot | null;
  diagnostic: string | null;
}

export type Action =
  | { type: "SOCKET_OPEN" }
  | { type: "SNAPSHOT"; snapshot: PreviewSnapshot }
  | { type: "BOOTSTRAP_TIMEOUT" }
  | { type: "TRANSPORT_FAILURE"; diagnostic: string }
  | { type: "INCOMPATIBLE_PROTOCOL"; diagnostic: string }
  | { type: "STALE" }
  | { type: "RETRY_TIMER" }
  | { type: "RETRY_BUDGET_EXHAUSTED" }
  | { type: "EXPLICIT_RETRY" };

export const INITIAL_STATE: ConnectionState = { phase: "connecting", snapshot: null, diagnostic: null };

/** Pure connection-phase state transition. All socket/timer side effects
 * live in useCaptureSession's own effect, never here. */
export function reducer(state: ConnectionState, action: Action): ConnectionState {
  switch (action.type) {
    case "SOCKET_OPEN":
      // Explicit Retry can start a fresh connect from "unavailable" too --
      // not just from connecting/reconnecting.
      return state.phase !== "connected" ? { ...state, phase: "synchronizing" } : state;
    case "SNAPSHOT":
      // A valid snapshot always resets the failure streak and installs the
      // latest facts, regardless of current phase -- see useCaptureSession's
      // own effect for the reconnect-vs-same-connection distinction.
      return { phase: "connected", snapshot: action.snapshot, diagnostic: null };
    case "BOOTSTRAP_TIMEOUT":
      return state.phase === "connected" ? state : { ...state, phase: "reconnecting", diagnostic: "bootstrap timed out" };
    case "STALE":
      return state.phase === "connected" ? { ...state, phase: "reconnecting", diagnostic: "no snapshot received recently" } : state;
    case "TRANSPORT_FAILURE":
      // A fatal protocol close (unavailable) must stay manual-retry-only --
      // an ordinary transport failure never downgrades that phase. The
      // effect only dispatches this for a retryable failure in the first
      // place (see onclose below), but this guard keeps the reducer itself
      // honest about the rule even if a caller ever changes that.
      return state.phase === "unavailable" ? state : { ...state, phase: "reconnecting", diagnostic: action.diagnostic };
    case "INCOMPATIBLE_PROTOCOL":
      return { ...state, phase: "unavailable", diagnostic: action.diagnostic };
    case "RETRY_TIMER":
      return { ...state, phase: "connecting" };
    case "RETRY_BUDGET_EXHAUSTED":
      return { ...state, phase: "unavailable" };
    case "EXPLICIT_RETRY":
      return { phase: "connecting", snapshot: null, diagnostic: null };
    default:
      return state;
  }
}

const BOOTSTRAP_DEADLINE_MS = 5_000;
const STALE_AFTER_MS = 3_000;
const RECONNECT_DELAYS_MS = [500, 1_000, 2_000, 4_000, 8_000];

/** Bounded exponential-ish reconnect delay with small jitter, capped at the
 * last entry in RECONNECT_DELAYS_MS. */
function backoffDelay(attempt: number): number {
  const base = RECONNECT_DELAYS_MS[Math.min(attempt, RECONNECT_DELAYS_MS.length - 1)];
  const jitter = base * 0.2 * Math.random();
  return base + jitter;
}

/** Derives the session WebSocket URL from the page's own origin/scheme. */
function wsUrl(): string {
  const scheme = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${scheme}//${window.location.host}/api/session`;
}

/** One decoded tracking sample together with everything needed to judge
 * its own freshness -- installed and cleared as a single unit, so there is
 * no way for its pieces to drift out of sync with each other. */
export interface AcceptedTracking {
  decoded: DecodedTracking;
  /** The validated descriptor backing `decoded` -- units/quaternion
   * order/axes/hand joint order, for labeling observations from the
   * actual validated metadata rather than an assumed/hardcoded label. */
  descriptor: TrackingStreamDescriptor;
  sequence: bigint;
  /** The backend's own producer-side age (samples.tracking.age_ms) as of
   * the most recent heartbeat at the time this sample was installed -- a
   * baseline captured once per accepted sample, not the latest heartbeat's
   * ever-refreshing value. */
  snapshotAgeMs: number | null;
  /** performance.now() at receipt of THAT heartbeat -- not of this binary
   * sample itself. A sample can arrive meaningfully later than the
   * heartbeat that reported its producer-side age, so anchoring to the
   * heartbeat's own receipt time (and letting elapsed time accumulate from
   * there) correctly counts the gap between the two; anchoring to the
   * sample's own receipt time instead would silently discard it and show
   * an already-stale sample as freshly arrived. */
  snapshotReceivedAtMonotonic: number;
}

export interface CaptureSessionState {
  phase: ConnectionPhase;
  snapshot: PreviewSnapshot | null;
  /** Wall-clock time this snapshot was actually received, recorded once at
   * delivery -- never recomputed at render time, which would manufacture a
   * fresh receipt time on every re-render (including ones triggered by a
   * binary tracking update) without a new snapshot actually arriving. For
   * human-readable display only ("Updated Ns ago") -- a system clock
   * correction must not make a cached sample's computed *freshness* look
   * wrong or right; see receivedAtMonotonic for that. */
  receivedAtMs: number | null;
  /** performance.now() at the same receipt instant as receivedAtMs --
   * monotonic, immune to system clock adjustments. Use this (not
   * receivedAtMs) for any elapsed-time-since-receipt freshness/staleness
   * math, e.g. CameraGrid's age_ms-plus-elapsed calculation. */
  receivedAtMonotonic: number | null;
  diagnostic: string | null;
  /** The most recently accepted tracking binary sample, or null if none
   * has been installed yet (or the association backing one was reset) --
   * see AcceptedTracking for why this is one value, not several. */
  trackingSample: AcceptedTracking | null;
  /** The backend's real recording facts, pushed proactively on connect and
   * whenever they change (including from another viewer's command) -- null
   * until the first one actually arrives, or after this connection ends. */
  recordingSnapshot: RecordingSnapshot | null;
  /** performance.now() at the instant `recordingSnapshot` was actually
   * received -- the LOCAL basis `currentEpisodeElapsedMs` is paired with
   * to keep counting between snapshots, since that figure is monotonic
   * but backend-clock-relative, not this browser's own clock. Null
   * exactly when `recordingSnapshot` is null. */
  recordingSnapshotReceivedAtMonotonic: number | null;
  retry: () => void;
  /** Sends one small bounded recording command and resolves with its real,
   * correlated reply -- never resolves on mere receipt, never retries
   * automatically, rejects immediately (without sending anything) if not
   * currently connected OR if no control token has been entered yet. Sends
   * the current `operatorToken` and `controlToken` with every call; the
   * backend rejects any of the five mutating actions without both. */
  sendRecordingCommand: (action: RecordingActionName, payload: Record<string, unknown>) => Promise<RecordingCommandResult>;
  /** The operator name attached to every outgoing recording command, or
   * null before one has ever been entered -- persisted in localStorage
   * (not sessionStorage) so it survives across tabs/reloads/browser
   * restarts, letting an operator pick up where they left off.
   * Attribution only, never authentication: this never gates
   * anything client-side, the backend is the one real enforcement point. */
  operatorToken: string | null;
  /** Normalizes and persists a new operator name for every subsequent
   * command on this and future connections. The backend does its own
   * normalization/validation independently -- this is a convenience for
   * display, not a substitute for that check. */
  setOperatorToken: (name: string) => void;
  /** Clears the stored operator name -- the "New operator" action. The
   * panel returns to requiring entry before any further recording
   * command. */
  clearOperatorToken: () => void;
  /** The control token attached to every outgoing recording command, or
   * null before one has been entered this tab. The server is what makes
   * it per-run (see app.py's lifespan: a fresh secret each backend
   * start, rejecting a stale one regardless of what the client does with
   * its own copy) -- kept in sessionStorage, scoped to this tab's
   * session (never localStorage, which would outlive the tab, and never
   * a URL); a browser's own tab-restore/duplicate feature may carry it
   * along, same as any other sessionStorage value. A stale token from a
   * prior backend run is dropped locally as soon as a snapshot from the
   * new backend arrives (see the backend_instance_id binding below), not
   * merely rejected on the next command. */
  controlToken: string | null;
  /** Sets the entered control token for this tab. A nonempty value here
   * means "entered", not "authenticated" -- only a real server reply
   * establishes whether a given command's token was actually accepted. */
  setControlToken: (token: string) => void;
  /** Clears the entered control token -- disables this tab's future
   * commands until a new one is entered. */
  clearControlToken: () => void;
}

const OPERATOR_TOKEN_STORAGE_KEY = "egocap_operator_token";
//: Matches the backend's own bound (recording_command.py's
//: _MAX_OPERATOR_LENGTH) -- not authoritative (the backend re-validates
//: independently), just what keeps this side from storing/displaying
//: something the backend will reject outright.
const OPERATOR_TOKEN_MAX_LENGTH = 128;

/** Trims and bounds a raw operator name; empty after trimming, or over
 * length, becomes null. Used for every storage read AND write so a
 * stale/bad stored value (e.g. from an older build, or hand-edited
 * storage) is never trusted blindly. */
function normalizeOperatorName(raw: string): string | null {
  if (raw.length > OPERATOR_TOKEN_MAX_LENGTH) return null;
  const trimmed = raw.trim();
  return trimmed === "" ? null : trimmed;
}

/** The persisted operator name, re-validated (not trusted as-is -- an
 * older build or hand-edited storage could leave something invalid). */
function readStoredOperatorToken(): string | null {
  try {
    const stored = window.localStorage.getItem(OPERATOR_TOKEN_STORAGE_KEY);
    return stored === null ? null : normalizeOperatorName(stored);
  } catch {
    return null; // private-browsing/storage-disabled: usable for this tab's lifetime only
  }
}

/** What makes one tracking binary sample eligible for this socket, as
 * installed by the most recently *sent-and-processed* snapshot -- not
 * re-derived from whatever the monitor happens to report by the time a
 * binary frame arrives. Reset on bootstrap, identity change and terminal. */
interface TrackingAssociation {
  descriptor: TrackingStreamDescriptor;
  generation: number;
  running: boolean;
  lastSequence: bigint | null;
}

/** Owns the /api/session WebSocket connection, reconnect, capture snapshot, recording snapshot, and recording-command transport -- see the module docstring. */
export function useCaptureSession(): CaptureSessionState {
  const [state, dispatch] = useReducer(reducer, INITIAL_STATE);
  const [operatorToken, setOperatorTokenState] = useState<string | null>(readStoredOperatorToken);
  const setOperatorToken = useCallback((name: string) => {
    const normalized = normalizeOperatorName(name);
    if (normalized === null) return; // the entry form already disables submit on blank; a >128-char
    // paste is the only other way to reach here -- silently refuse rather than store garbage.
    try {
      window.localStorage.setItem(OPERATOR_TOKEN_STORAGE_KEY, normalized);
    } catch {
      // ignore storage failures (e.g. private browsing) -- still usable for this tab's lifetime
    }
    setOperatorTokenState(normalized);
  }, []);
  const clearOperatorToken = useCallback(() => {
    try {
      window.localStorage.removeItem(OPERATOR_TOKEN_STORAGE_KEY);
    } catch {
      // ignore
    }
    setOperatorTokenState(null);
  }, []);
  const [trackingSample, setTrackingSample] = useState<AcceptedTracking | null>(null);
  const [receivedAtMs, setReceivedAtMs] = useState<number | null>(null);
  const [receivedAtMonotonic, setReceivedAtMonotonic] = useState<number | null>(null);
  const [recordingSnapshot, setRecordingSnapshot] = useState<RecordingSnapshot | null>(null);
  const [recordingSnapshotReceivedAtMonotonic, setRecordingSnapshotReceivedAtMonotonic] = useState<number | null>(null);
  const recordingRevisionRef = useRef<number | null>(null);
  /** The single place a freshly-parsed recording snapshot is ever installed
   * -- always pairs it with its own receipt-time performance.now() basis,
   * so the two state values can never drift out of sync with each other. */
  const installRecordingSnapshot = useCallback((recording: RecordingSnapshot) => {
    // Dropped when older than one this socket already applied; a new
    // socket starts unset, since a restarted backend counts from 0 again.
    const applied = recordingRevisionRef.current;
    if (applied !== null && recording.revision < applied) return;
    recordingRevisionRef.current = recording.revision;
    setRecordingSnapshot(recording);
    setRecordingSnapshotReceivedAtMonotonic(performance.now());
  }, []);
  const [controlToken, setControlTokenState] = useState<string | null>(() => readStoredControlToken());
  // The backend_instance_id the current token was last confirmed against --
  // null means "not yet bound to a live backend" (just entered, or just
  // restored from a reload before this tab's first snapshot has arrived).
  const tokenBackendInstanceIdRef = useRef<string | null>(readStoredTokenBackendInstanceId());
  const setControlToken = useCallback((token: string) => {
    const trimmed = token.trim();
    removeStoredTokenBackendInstanceId();
    tokenBackendInstanceIdRef.current = null;
    if (trimmed === "") {
      removeStoredControlToken();
      setControlTokenState(null);
      return;
    }
    writeStoredControlToken(trimmed);
    setControlTokenState(trimmed);
  }, []);
  const clearControlToken = useCallback(() => {
    removeStoredControlToken();
    removeStoredTokenBackendInstanceId();
    tokenBackendInstanceIdRef.current = null;
    setControlTokenState(null);
  }, []);
  // Binds a freshly-entered/restored token to the first backend_instance_id
  // it's actually seen, then drops it the moment a later snapshot reports a
  // DIFFERENT id -- a backend restart (or a different backend entirely)
  // invalidates whatever was entered before, and a reload alone must not
  // let a stale token keep looking "entered" while every command silently
  // fails (see RecordingPanel's NO_CONTROL_TOKEN_REASON path).
  //
  // Only binds while actually connected: the reducer never clears
  // state.snapshot on reconnect, so right after a backend restart it can
  // still hold the OLD backend's id while phase is "reconnecting"/
  // "unavailable". Binding against that stale snapshot would pair a
  // freshly-entered token with the old id, and the new backend's first real
  // snapshot would then immediately clear it out from under the operator.
  useEffect(() => {
    if (state.phase !== "connected") return;
    const id = state.snapshot?.backend_instance_id ?? null;
    if (id === null || controlToken === null) return;
    const bound = tokenBackendInstanceIdRef.current;
    if (bound === null) {
      tokenBackendInstanceIdRef.current = id;
      writeStoredTokenBackendInstanceId(id);
      return;
    }
    if (bound !== id) {
      clearControlToken();
    }
  }, [state.phase, state.snapshot, controlToken, clearControlToken]);
  const [retryNonce, setRetryNonce] = useState(0);
  // Mirrors the effect's own `socket` local so sendRecordingCommand (called
  // from outside the effect) can reach the currently-live socket without
  // the effect re-running on every render.
  const socketRef = useRef<WebSocket | null>(null);
  // Only ONE recording command is ever in flight at a time -- a second
  // send attempt while one is outstanding is rejected locally rather than
  // letting callers queue an unbounded backlog during the timeout window.
  const pendingRequestRef = useRef<PendingReply | null>(null);
  const requestCounterRef = useRef(0);

  /** The one place a pending request is ever resolved/rejected and cleared
   * -- used on a real reply, on disconnect/teardown, and on unmount, so
   * that bookkeeping never duplicates across call sites. */
  const settlePendingRequest = useCallback((outcome: { result: RecordingCommandResult } | { error: Error }) => {
    const pending = pendingRequestRef.current;
    if (pending === null) return;
    clearTimeout(pending.timer);
    pendingRequestRef.current = null;
    if ("result" in outcome) pending.resolve(outcome.result);
    else pending.reject(outcome.error);
  }, []);

  useEffect(() => {
    let cancelled = false;
    let socket: WebSocket | null = null;
    let bootstrapTimer: ReturnType<typeof setTimeout> | null = null;
    let staleTimer: ReturnType<typeof setTimeout> | null = null;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    let attempt = 0;
    let association: TrackingAssociation | null = null;
    let lastBackendInstanceId: string | null = null;

    /** Clears every timer this effect owns. Idempotent. */
    function clearTimers(): void {
      if (bootstrapTimer !== null) clearTimeout(bootstrapTimer);
      if (staleTimer !== null) clearTimeout(staleTimer);
      bootstrapTimer = null;
      staleTimer = null;
    }

    /** Closes a socket and clears its handlers so any in-flight/buffered
     * event from it is ignored. Idempotent -- safe to call on an already-
     * invalidated or null socket. */
    function invalidateSocket(ws: WebSocket | null): void {
      if (ws === null) return;
      ws.onopen = null;
      ws.onmessage = null;
      ws.onclose = null;
      ws.onerror = null;
      try {
        ws.close();
      } catch {
        // already closed/closing
      }
    }

    function resetAssociation(): void {
      association = null;
      setTrackingSample(null);
    }

    /** The one teardown path for "this connection attempt is over, for
     * reason X" -- every transition away from a socket (stale, bootstrap
     * timeout, malformed message, fatal protocol error, ordinary close,
     * unmount) goes through this, not a scattered subset of its steps.
     * It always clears every owned timer and the current association
     * before deciding what happens next, so a fatal transition can never
     * leave a stray staleTimer/bootstrapTimer to fire later and schedule
     * an unwanted reconnect -- the single place that decides whether to
     * reconnect is the `fatal` flag here, not each call site. */
    function teardown(action: Action, ws: WebSocket, options: { fatal: boolean }): void {
      clearTimers();
      invalidateSocket(ws);
      if (socket === ws) {
        socket = null;
        socketRef.current = null;
      }
      resetAssociation();
      // This connection's own recording facts are gone with it -- shown
      // honestly as unavailable rather than a stale phase that would
      // invite an action doomed to fail; the next connection's own
      // proactive push repopulates it.
      setRecordingSnapshot(null);
      setRecordingSnapshotReceivedAtMonotonic(null);
      // A command sent on this socket can never be answered now -- reject
      // it rather than leave a caller hanging past its RECORDING_COMMAND_TIMEOUT_MS entry
      // for no reason, and never auto-retry it against whatever socket
      // reconnects next. Genuinely uncertain, not a confident rejection --
      // the backend may have already applied it.
      settlePendingRequest({ error: new RecordingCommandUncertainError("connection lost before a reply arrived") });
      dispatch(action);
      if (!options.fatal) scheduleReconnect();
    }

    function scheduleReconnect(): void {
      if (cancelled) return;
      if (attempt >= RECONNECT_DELAYS_MS.length) {
        dispatch({ type: "RETRY_BUDGET_EXHAUSTED" });
        return;
      }
      const delay = backoffDelay(attempt);
      attempt += 1;
      retryTimer = setTimeout(() => {
        if (cancelled) return;
        dispatch({ type: "RETRY_TIMER" });
        connect();
      }, delay);
    }

    function armStaleTimer(ws: WebSocket): void {
      if (staleTimer !== null) clearTimeout(staleTimer);
      staleTimer = setTimeout(() => {
        if (socket !== ws || cancelled) return;
        teardown({ type: "STALE" }, ws, { fatal: false });
      }, STALE_AFTER_MS);
    }

    function connect(): void {
      if (cancelled) return;
      resetAssociation();
      recordingRevisionRef.current = null;
      const ws = new WebSocket(wsUrl());
      ws.binaryType = "arraybuffer";
      socket = ws;
      socketRef.current = ws;
      // Scoped to THIS socket, unlike lastBackendInstanceId -- a reconnect's
      // replacement socket must enforce the snapshot-before-binary ordering
      // rule against its own first message, not inherit "already bootstrapped"
      // from a prior socket's backend identity.
      let bootstrapped = false;
      // The backend's own last-reported tracking.age_ms, and this socket's
      // own receipt time for that same heartbeat -- carried forward as the
      // producer-side freshness baseline for whichever binary sample gets
      // installed next. The monotonic reference is the HEARTBEAT's own
      // receipt instant, not whatever instant a later binary happens to
      // arrive at -- see AcceptedTracking's own doc for why.
      let lastKnownTrackingAgeMs: number | null = null;
      let lastSnapshotReceivedAtMonotonic: number | null = null;

      bootstrapTimer = setTimeout(() => {
        if (socket !== ws || cancelled) return;
        teardown({ type: "BOOTSTRAP_TIMEOUT" }, ws, { fatal: false });
      }, BOOTSTRAP_DEADLINE_MS);

      ws.onopen = () => {
        if (socket !== ws || cancelled) return;
        dispatch({ type: "SOCKET_OPEN" });
      };

      ws.onmessage = (event: MessageEvent) => {
        if (socket !== ws || cancelled) return;
        if (typeof event.data === "string") {
          // A recording-command reply is a distinct message shape from a
          // preview snapshot, checked and routed BEFORE any attempt to
          // parse it as one -- parseSnapshot must never see a reply, and a
          // reply must never be able to reset the snapshot bootstrap/
          // association state below.
          let parsed: unknown;
          try {
            parsed = JSON.parse(event.data);
          } catch {
            teardown({ type: "TRANSPORT_FAILURE", diagnostic: "malformed message" }, ws, { fatal: false });
            return;
          }
          if (isRecordingSnapshotMessage(parsed)) {
            const recording = parseRecordingSnapshot(parsed.snapshot);
            if (recording !== null) installRecordingSnapshot(recording);
            return;
          }
          if (isRecordingReply(parsed)) {
            // The embedded snapshot reflects real current backend facts
            // regardless of which request it correlates to -- applied
            // unconditionally. Only the pending PROMISE is scoped to the
            // one request currently outstanding -- a reply for a since-
            // superseded or already-timed-out request id settles nothing.
            const recording = parseRecordingSnapshot(parsed.snapshot);
            if (recording !== null) installRecordingSnapshot(recording);
            if (pendingRequestRef.current?.requestId === parsed.request_id) {
              settlePendingRequest({
                result: {
                  outcome: parsed.outcome as RecordingCommandResult["outcome"],
                  reason: parsed.reason,
                },
              });
            }
            return;
          }
          let snapshot: PreviewSnapshot;
          try {
            snapshot = parseSnapshot(parsed);
          } catch (err) {
            if (err instanceof CapturePreviewParseError) {
              teardown({ type: "INCOMPATIBLE_PROTOCOL", diagnostic: err.message }, ws, { fatal: true });
              return;
            }
            teardown({ type: "TRANSPORT_FAILURE", diagnostic: "malformed snapshot message" }, ws, { fatal: false });
            return;
          }
          if (bootstrapTimer !== null) {
            clearTimeout(bootstrapTimer);
            bootstrapTimer = null;
          }
          // A changed backend identity clears all old associations even if
          // a revision number happens to repeat -- a same-backend/
          // same-revision bootstrap still installs that socket's own
          // generation/metadata via this same dispatch.
          if (snapshot.backend_instance_id !== lastBackendInstanceId) {
            resetAssociation();
          }
          lastBackendInstanceId = snapshot.backend_instance_id;
          bootstrapped = true;
          lastKnownTrackingAgeMs = snapshot.tracking?.age_ms ?? null;
          lastSnapshotReceivedAtMonotonic = performance.now();

          // parseSnapshot (the one authoritative boundary validator) already
          // enforced the descriptor's extent/overlap/contract rules before
          // this snapshot was ever dispatched as "connected" -- re-validating
          // here would be a second copy of that algorithm, and a parse
          // failure caught locally would have silently reset the
          // association while still showing "connected", which is exactly
          // the empty-preview-while-healthy failure mode the boundary
          // validator exists to prevent.
          const found = snapshot.started?.streams.find((s) => s.kind === "tracking") ?? null;
          const running = snapshot.status?.running === true;
          if (found !== null && found.kind === "tracking" && snapshot.binary_generation !== null && running) {
            // Installing a new descriptor/generation always starts a fresh
            // sequence floor -- a sample sequence number from before this
            // snapshot was processed must never be treated as "newer" than
            // one that arrives after it.
            association = {
              descriptor: found,
              generation: snapshot.binary_generation,
              running: true,
              lastSequence: association?.generation === snapshot.binary_generation ? association.lastSequence : null,
            };
          } else {
            // Not advertised, not yet accepted, not running, or terminal
            // (binary_generation === null) -- no binary is eligible.
            resetAssociation();
          }

          attempt = 0;
          dispatch({ type: "SNAPSHOT", snapshot });
          setReceivedAtMs(Date.now());
          setReceivedAtMonotonic(performance.now());
          armStaleTimer(ws);
        } else if (event.data instanceof ArrayBuffer) {
          // Binary before any snapshot at all is a real ordering violation
          // -- this connection's protocol requires a snapshot before any
          // binary message on it. This is a framing/ordering problem, not a
          // native-invalid-pose sample: surface it visibly rather than
          // silently discarding, the same as a malformed header below.
          if (!bootstrapped) {
            teardown(
              { type: "INCOMPATIBLE_PROTOCOL", diagnostic: "binary message received before any snapshot" },
              ws,
              { fatal: true },
            );
            return;
          }
          let header;
          try {
            header = parseBinaryHeader(event.data);
          } catch (err) {
            // A genuinely malformed header (too short, nonzero reserved
            // byte) is a framing violation -- fatal, same as the server's
            // own close-on-malformed-framing behavior. This is distinct
            // from the eligibility checks below (wrong generation/stream/
            // stale sequence), which are routine and expected during
            // normal operation (races, a superseded association) and stay
            // a simple discard rather than tearing down the connection.
            teardown(
              { type: "INCOMPATIBLE_PROTOCOL", diagnostic: err instanceof Error ? err.message : "malformed binary header" },
              ws,
              { fatal: true },
            );
            return;
          }
          if (association === null) return; // not currently advertised/running -- routine, not malformed
          if (header.kind !== BINARY_KIND_TRACKING) return;
          if (header.generation !== association.generation) return;
          if (header.streamId !== association.descriptor.stream_id) return;
          if (!association.running) return;
          // Past this point the message is for the one currently-active
          // association (right generation, stream, running) -- a wrong byte
          // length here is a framing/bounds violation the server should
          // never produce for its own advertised descriptor, not a routine
          // obsolete-generation/sequence race. Fatal, same as a malformed
          // header.
          const expectedLength = BINARY_HEADER_SIZE + association.descriptor.payload_size;
          if (event.data.byteLength !== expectedLength) {
            teardown(
              {
                type: "INCOMPATIBLE_PROTOCOL",
                diagnostic: `tracking payload size mismatch: expected ${expectedLength} bytes, got ${event.data.byteLength}`,
              },
              ws,
              { fatal: true },
            );
            return;
          }
          if (association.lastSequence !== null && header.sequence <= association.lastSequence) return;
          // decodeTracking should not throw here: captureProtocol.ts's
          // parseStreamDescriptor already validated every known field's
          // dtype/shape/extent and the byteLength check above already
          // enforced the exact expected payload size. Kept as defense in
          // depth, not a documented failure path.
          const decoded = decodeTracking(association.descriptor, event.data, BINARY_HEADER_SIZE);
          association.lastSequence = header.sequence;
          setTrackingSample({
            decoded,
            descriptor: association.descriptor,
            sequence: header.sequence,
            snapshotAgeMs: lastKnownTrackingAgeMs,
            // association is only ever non-null after at least one snapshot
            // was processed, which always sets this in the same branch --
            // the fallback below is defensive, not an expected path.
            snapshotReceivedAtMonotonic: lastSnapshotReceivedAtMonotonic ?? performance.now(),
          });
        }
      };

      ws.onclose = () => {
        if (socket !== ws || cancelled) return;
        // A fatal protocol error's own teardown() call already nulled this
        // handler before calling ws.close(), so onclose only ever fires
        // here for an ordinary/unexpected close -- always retryable.
        teardown({ type: "TRANSPORT_FAILURE", diagnostic: "connection closed" }, ws, { fatal: false });
      };

      ws.onerror = () => {
        // onclose always follows onerror for a WebSocket; let onclose own
        // the state transition and reconnect scheduling.
      };
    }

    connect();

    return () => {
      cancelled = true;
      clearTimers();
      if (retryTimer !== null) clearTimeout(retryTimer);
      invalidateSocket(socket);
      socket = null;
      socketRef.current = null;
      // A reconnect-driven effect restart (or real unmount) leaves nothing
      // able to ever answer an in-flight command -- reject it the same way
      // teardown() does, rather than leaving a caller's promise hanging.
      // Genuinely uncertain, same as teardown()'s own rejection.
      settlePendingRequest({ error: new RecordingCommandUncertainError("connection closed before a reply arrived") });
    };
    // retryNonce is read only to restart this effect on explicit retry --
    // every other value used inside is effect-local, not a dependency.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [retryNonce]);

  function retry(): void {
    dispatch({ type: "EXPLICIT_RETRY" });
    setRetryNonce((n) => n + 1);
  }

  /** Sends one small recording command and resolves with its correlated
   * reply, or rejects on disconnect/timeout/local validation failure.
   * Never retried automatically -- a caller that wants another attempt
   * calls this again with a fresh request. Rejects immediately, without
   * sending anything, if the connection is not actually synchronized yet
   * (not merely socket-OPEN -- a pre-bootstrap or reconnecting socket is
   * not "connected" for this purpose either), if a request is already
   * outstanding, if no control token has been entered, or if the
   * serialized command would exceed the backend's own size bound. */
  const sendRecordingCommand = useCallback(
    (action: RecordingActionName, payload: Record<string, unknown>): Promise<RecordingCommandResult> => {
      const ws = socketRef.current;
      if (ws === null || ws.readyState !== WebSocket.OPEN || state.phase !== "connected") {
        return Promise.reject(new Error("not connected"));
      }
      if (pendingRequestRef.current !== null) {
        return Promise.reject(new Error("another recording command is already in flight"));
      }
      // RecordingPanel already gates on controlToken !== null before calling
      // this, but the backend is the only real authority either way -- this
      // just avoids sending a command it already knows will be rejected if
      // some other caller bypasses that UI gate directly.
      if (controlToken === null) {
        return Promise.reject(new Error("no control token entered"));
      }
      requestCounterRef.current += 1;
      const requestId = `req-${requestCounterRef.current}`;
      const body = JSON.stringify({
        type: "recording_command",
        request_id: requestId,
        action,
        payload,
        operator_token: operatorToken,
        control_token: controlToken,
      });
      if (new TextEncoder().encode(body).length > MAX_RECORDING_COMMAND_BYTES) {
        return Promise.reject(new Error("command exceeds the maximum allowed size"));
      }
      return new Promise<RecordingCommandResult>((resolve, reject) => {
        const timer = setTimeout(() => {
          // The command WAS sent -- its real outcome is genuinely unknown,
          // not a confident rejection.
          settlePendingRequest({ error: new RecordingCommandUncertainError("timed out waiting for a reply") });
          // Actively resolve the uncertainty by reusing the EXISTING
          // transport teardown/reconnect path (ws.onclose -> teardown(),
          // same as any ordinary connection failure) rather than passively
          // waiting for an unrelated revision bump that may never come.
          // The replacement connection's own bootstrap already delivers a
          // fresh recording_snapshot -- no retry of the mutation itself, no
          // new protocol action, no second error-swallowing path.
          ws.close();
        }, RECORDING_COMMAND_TIMEOUT_MS[action]);
        pendingRequestRef.current = { requestId, resolve, reject, timer };
        try {
          ws.send(body);
        } catch (err) {
          settlePendingRequest({ error: err instanceof Error ? err : new Error("failed to send") });
        }
      });
    },
    [state.phase, settlePendingRequest, operatorToken, controlToken],
  );

  return {
    phase: state.phase,
    snapshot: state.snapshot,
    receivedAtMs,
    receivedAtMonotonic,
    diagnostic: state.diagnostic,
    trackingSample,
    recordingSnapshot,
    recordingSnapshotReceivedAtMonotonic,
    retry,
    sendRecordingCommand,
    operatorToken,
    setOperatorToken,
    clearOperatorToken,
    controlToken,
    setControlToken,
    clearControlToken,
  };
}
