import { useEffect, useId, useState, type CSSProperties } from "react";
import { useCaptureSession, type ConnectionPhase } from "./useCaptureSession";
import type { PreviewSnapshot } from "./captureProtocol";
import {
  CaptureMode,
  CaptureStatusType,
  ProviderState,
  deviceStateLabel,
  providerStateLabel,
  startupStageLabel,
  statusReasonLabel,
  stopReasonLabel,
  type CaptureStatusResponse,
  type CameraStatus,
  type DeviceStatus,
  type ProviderStatus,
} from "./captureStatus";
import { CaptureTree } from "./CaptureTree";
import { CameraGrid } from "./CameraGrid";
import { OperatorBar } from "./OperatorBar";
import { SidebarResizer } from "./SidebarResizer";
import {
  SIDEBAR_DEFAULT_PX,
  SIDEBAR_MIN_PX,
  clampSidebarWidth,
  loadSidebarWidth,
  maxSidebarWidth,
  saveSidebarWidth,
} from "./sidebarWidth";
import { deviceTone, retainedTone, type Tone } from "./deviceHealth";
import { PoseViserPanel } from "./PoseViserPanel";
import { RecordingPanel } from "./RecordingPanel";
import { buildRecordingActions } from "./recordingActions";
import type { RecordingPhase } from "./recordingUiReducer";

/** Feed health is derived from ConnectionPhase rather than a separate
 * polling freshness signal, since the connection hook owns delivery
 * freshness directly now -- there is no HTTP polling fallback. */
type FeedHealth = "connecting" | "ok" | "delayed" | "unavailable";

/** Maps the connection hook's own phase to the coarser feed-health label
 * shown in the header. */
function feedHealthFromPhase(phase: ConnectionPhase): FeedHealth {
  switch (phase) {
    case "connecting":
    case "synchronizing":
      return "connecting";
    case "connected":
      return "ok";
    case "reconnecting":
      return "delayed";
    case "unavailable":
      return "unavailable";
  }
}

/** Adapts a WS PreviewSnapshot into the existing CaptureStatusResponse
 * shape so the status strip and session/provider details can keep reading
 * one consistent response type. `schema_version`/`stop_reason` are
 * reconstructed field-for-field from the snapshot's own capture.* facts,
 * never invented. */
function snapshotToLegacyStatus(snapshot: PreviewSnapshot): CaptureStatusResponse {
  // One coherent observation: once `stopped` is present the attempt is
  // over, so running/startup/message/status all come from the terminal
  // facts, never from an earlier (now superseded) `status` projection --
  // a terminal view must not keep internally claiming running=true or
  // showing a stale startup_stage/message, and must show the real final
  // StatusSnapshot (counters/providers/devices), not an older one.
  const terminal = snapshot.stopped !== null;
  return {
    schema_version: snapshot.version,
    backend_instance_id: snapshot.backend_instance_id,
    station_label: snapshot.station_label,
    profile_label: snapshot.profile_label,
    mode: snapshot.started?.mode ?? null,
    session_id: snapshot.started?.session_id ?? snapshot.status?.session_id ?? snapshot.stopped?.session_id ?? null,
    connecting: snapshot.connecting,
    status_type: terminal ? null : (snapshot.status?.type ?? null),
    running: terminal ? false : (snapshot.status?.running ?? false),
    startup_stage: terminal ? null : (snapshot.status?.startup_stage ?? null),
    message: terminal ? null : (snapshot.status?.message ?? null),
    terminal,
    stop_reason: snapshot.stopped?.reason ?? null,
    error_message: snapshot.error_message ?? snapshot.stopped?.error_message ?? null,
    status: terminal ? (snapshot.stopped?.status ?? null) : (snapshot.status?.status ?? null),
  };
}

interface Presentation {
  headline: string;
  detail: string | null;
  tone: Tone;
  /** false means the status strip and details below show an observation
   * that is no longer current (a stale feed) or will never change again
   * (terminal) -- callers must not present it with live/positive styling. */
  live: boolean;
  /** A short caption for sections showing retained data while !live, e.g.
   * "Last known" or "Final". null while live. */
  freshnessLabel: string | null;
}

/** The pure headline/detail/tone facts for a response, independent of feed
 * freshness -- derivePresentation() below is what actually accounts for
 * staleness/terminal state before this reaches the page. */
function deriveLifecycleFacts(r: CaptureStatusResponse): Pick<Presentation, "headline" | "detail" | "tone"> {
  if (r.terminal) {
    if (r.stop_reason === null) {
      // CaptureStopped.reason is None for unconfirmed local completion
      // (e.g. the transport dropped) -- never presented as if the server
      // had confirmed a real stop reason.
      return { headline: "Capture ended (unconfirmed)", detail: r.error_message, tone: "warning" };
    }
    const reason = stopReasonLabel(r.stop_reason);
    return {
      headline: "Capture stopped",
      detail: `${reason}${r.error_message ? ` — ${r.error_message}` : ""}`,
      tone: "neutral",
    };
  }
  if (r.connecting) {
    return { headline: "Connecting to capture server", detail: r.message, tone: "neutral" };
  }
  if (r.error_message !== null && r.session_id === null) {
    // A real pre-acceptance failure: rejected/refused/closed before a
    // session even existed -- never presented as a fabricated lifecycle.
    return { headline: "Capture unavailable", detail: r.error_message, tone: "critical" };
  }
  if (r.status_type === CaptureStatusType.STARTING) {
    const stage = r.startup_stage === null ? null : startupStageLabel(r.startup_stage);
    return {
      headline: "Capture initializing",
      detail: [stage, r.message].filter((v) => v !== null).join(" — ") || null,
      tone: "neutral",
    };
  }
  if (r.running) {
    return { headline: "Capture running", detail: null, tone: "positive" };
  }
  return { headline: "Capture initializing", detail: r.message, tone: "neutral" };
}

/** Derives the one presentation the whole page renders from actual response
 * facts plus socket currentness -- see frontend/README.md's "Status
 * presentation" section. The single source every component (status strip,
 * details, live region, CameraGrid) reads from, so they cannot disagree
 * about whether what's on screen is current.
 *
 * `connected` must be exactly `phase === "connected"` -- not a broader
 * "feed health" mapping. Every other phase (auto-retrying, exhausted
 * retry budget, fatal protocol error) means the socket is not currently
 * delivering, so any cached nonterminal facts are presented as retained/
 * stale, never live-positive, regardless of which specific phase it is. */
function derivePresentation(r: CaptureStatusResponse | null, connected: boolean): Presentation {
  if (r === null) {
    return { headline: "Connecting to capture server", detail: null, tone: "neutral", live: true, freshnessLabel: null };
  }
  const facts = deriveLifecycleFacts(r);
  if (r.terminal) {
    return { ...facts, live: false, freshnessLabel: "Final" };
  }
  if (!connected) {
    return {
      headline: facts.headline,
      detail: facts.detail,
      // Same rule retainedTone() applies to provider rows below -- a stale
      // feed must never keep showing the live-positive "running" look.
      tone: retainedTone(facts.tone, false),
      live: false,
      freshnessLabel: "Last known",
    };
  }
  return { ...facts, live: true, freshnessLabel: null };
}

/** The capture mode as shown to the operator, or "Mode pending" before
 * an attempt has been accepted. */
function modeLabel(mode: CaptureStatusResponse["mode"]): string {
  if (mode === null) return "Mode pending";
  return mode === CaptureMode.MOCK ? "Mock" : "Live";
}

/** The CSS class for one of the four semantic tones. */
function toneClass(tone: Tone): string {
  return `tone-${tone}`;
}

// DeviceState and ProviderState are different enums that happen to share
// some numeric values (e.g. ProviderState.AVAILABLE=2 == DeviceState.UNKNOWN=2)
// -- kept as a separate table, never looked up through DEVICE_TONE.
const PROVIDER_TONE: Partial<Record<number, Tone>> = {
  [ProviderState.AVAILABLE]: "positive",
  [ProviderState.FAILED]: "critical",
  [ProviderState.INITIALIZING]: "warning",
  [ProviderState.STOPPED]: "neutral",
};

/** The semantic tone for a provider's ProviderState -- looked up through
 * its own table, never DEVICE_TONE (see the comment above it). */
function providerTone(status: ProviderStatus["status"]): Tone {
  return PROVIDER_TONE[status] ?? "neutral";
}

/** The page's top bar: app title, capture mode and feed-health status.
 * There is no page/tab navigation -- the generic capture display and the
 * Ego Capture UI recording panel are always-present siblings on one page. */
function AppHeader({ response, feedHealthLabel }: { response: CaptureStatusResponse | null; feedHealthLabel: string }) {
  return (
    <header className="app-header">
      <div className="app-title">
        <span className="app-name">Ego Capture UI</span>
      </div>
      <div className="header-right">
        <span className="mode-badge">{modeLabel(response?.mode ?? null)}</span>
        {/* Announces feed-health changes (e.g. "Delayed") but not the
         * separately-rendered sample-age ticker, so visitors using a
         * screen reader hear meaningful transitions, not every tick. */}
        <span className="feed-health" aria-live="polite">
          Feed {feedHealthLabel}
        </span>
      </div>
    </header>
  );
}

/** The compact status row beneath the header: station and the real capture
 * lifecycle headline (a quiet one-liner, not a large hero). The supporting
 * session/provider details are reached by scrolling to their own collapsed
 * disclosure at the bottom of the page, not a link from here. A normal
 * running/current state stays quiet here; an error or stale feed instead
 * gets a visible banner from `StaleOrErrorBanner` below. */
function StatusLine({ response, presentation }: { response: CaptureStatusResponse | null; presentation: Presentation }) {
  return (
    <div className={`status-strip ${toneClass(presentation.tone)}`}>
      <span className="status-station">{response?.station_label ?? "Station"}</span>
      <span aria-live="polite" className="status-headline">
        {presentation.freshnessLabel !== null && <span className="freshness-badge">{presentation.freshnessLabel}</span>}
        {presentation.headline}
        {presentation.detail !== null && <span className="status-detail"> — {presentation.detail}</span>}
      </span>
    </div>
  );
}

/** A visible banner for a stale or unavailable feed; renders nothing for
 * a live or terminal (already-labeled "Final") presentation. */
function StaleOrErrorBanner({
  presentation,
  feedHealth,
  diagnostic,
  onRetry,
}: {
  presentation: Presentation;
  feedHealth: FeedHealth;
  diagnostic: string | null;
  onRetry: () => void;
}) {
  if (presentation.live || presentation.freshnessLabel === "Final") return null;
  return (
    <p className="stale-banner" role="status">
      {feedHealth === "unavailable" ? (
        <>
          Status updates unavailable{diagnostic !== null ? `: ${diagnostic}` : "."} — showing last known data.{" "}
          <button type="button" className="refresh-button" onClick={onRetry}>
            Retry
          </button>
        </>
      ) : (
        "Status updates unavailable — showing last known data."
      )}
    </p>
  );
}

/** One tracking device or camera row in the session/provider details
 * disclosure. */
function SourceRow({
  label,
  status,
  reason,
  error,
  live,
}: {
  label: string;
  status: DeviceStatus["status"] | CameraStatus["status"];
  reason: DeviceStatus["reason"];
  error: string;
  live: boolean;
}) {
  const tone = retainedTone(deviceTone(status), live);
  return (
    <li className={`source-row ${toneClass(tone)}`}>
      <span className="source-label">{label}</span>
      <span className="source-status">{deviceStateLabel(status)}</span>
      {tone !== "positive" && (error !== "" || statusReasonLabel(reason) !== "None") && (
        <span className="source-diagnostic">{error !== "" ? error : statusReasonLabel(reason)}</span>
      )}
    </li>
  );
}

/** `freshnessLabel` is appended to each group header (e.g. "Tracking —
 * Last known") rather than duplicated per row -- the staleness applies to
 * the whole snapshot, not individual entries within it. */
function SourceStatusList({
  response,
  live,
  freshnessLabel,
}: {
  response: CaptureStatusResponse | null;
  live: boolean;
  freshnessLabel: string | null;
}) {
  const devices = response?.status?.devices ?? [];
  const cameras = response?.status?.cameras ?? [];
  const suffix = freshnessLabel !== null ? ` — ${freshnessLabel}` : "";
  return (
    <div className="source-groups">
      <section>
        <h3 className="group-sublabel">Tracking devices{suffix}</h3>
        {devices.length === 0 ? (
          <p className="empty-note">No tracking devices reported yet.</p>
        ) : (
          <ul className="source-list">
            {devices.map((d) => (
              <SourceRow key={d.id} label={d.description || d.id} status={d.status} reason={d.reason} error={d.error} live={live} />
            ))}
          </ul>
        )}
      </section>
      <section>
        <h3 className="group-sublabel">Cameras{suffix}</h3>
        {cameras.length === 0 ? (
          <p className="empty-note">No cameras selected.</p>
        ) : (
          <ul className="source-list">
            {cameras.map((c) => (
              <SourceRow key={c.stream_id} label={c.name} status={c.status} reason={c.reason} error={c.error} live={live} />
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

/** The supporting disclosure for session/provider/device metadata and raw
 * diagnostics -- collapsed by default so it never competes with the
 * tracking tree for the first screen. */
function SessionDetails({
  response,
  live,
  freshnessLabel,
}: {
  response: CaptureStatusResponse | null;
  live: boolean;
  freshnessLabel: string | null;
}) {
  const detailsId = useId();
  const providers = response?.status?.providers ?? [];
  const suffix = freshnessLabel !== null ? ` — ${freshnessLabel}` : "";
  return (
    <details className="session-details">
      <summary className="group-label">Session and provider details{suffix}</summary>
      <div className="session-details-body">
        <section>
          <h3 className="group-sublabel">Session</h3>
          <dl className="kv-list">
            <div>
              <dt>Station</dt>
              <dd>{response?.station_label ?? "—"}</dd>
            </div>
            <div>
              <dt>Mode</dt>
              <dd>{modeLabel(response?.mode ?? null)}</dd>
            </div>
            <div>
              <dt>Configuration</dt>
              <dd>{response?.profile_label ?? "—"}</dd>
            </div>
            <div>
              <dt>Session ID</dt>
              <dd className="selectable">{response?.session_id ?? "—"}</dd>
            </div>
          </dl>
        </section>
        <section>
          <h3 className="group-sublabel">Providers{suffix}</h3>
          {providers.length === 0 ? (
            <p className="empty-note">No providers reported yet.</p>
          ) : (
            <dl className="kv-list">
              {providers.map((p) => (
                <div key={p.id}>
                  <dt>{p.name}</dt>
                  <dd className={toneClass(retainedTone(providerTone(p.status), live))}>{providerStateLabel(p.status)}</dd>
                </div>
              ))}
            </dl>
          )}
        </section>
        <SourceStatusList response={response} live={live} freshnessLabel={freshnessLabel} />
        <details className="technical-details">
          <summary id={detailsId}>Technical details</summary>
          <pre className="technical-json">{JSON.stringify(response, null, 2)}</pre>
        </details>
      </div>
    </details>
  );
}

const FEED_HEALTH_LABEL: Record<string, string> = {
  connecting: "connecting",
  ok: "current",
  delayed: "delayed",
  unavailable: "unavailable",
};

/** The one application shell: a single capture connection powers the
 * generic status/tree/camera display and the Ego Capture UI recording panel,
 * rendered together as permanent siblings on one page -- there is no
 * separate page/tab for either. */
export default function App() {
  const {
    phase,
    snapshot,
    receivedAtMonotonic,
    diagnostic,
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
  } = useCaptureSession();
  const feedHealth = feedHealthFromPhase(phase);
  const response = snapshot === null ? null : snapshotToLegacyStatus(snapshot);
  const presentation = derivePresentation(response, phase === "connected");
  const recordingActions = buildRecordingActions(sendRecordingCommand);
  // A live take frames the cameras in red, so the state reads from across the room; never from a stale feed.
  const recording = presentation.live && recordingSnapshot?.episodePhase === "RECORDING";
  const [panelPhase, setPanelPhase] = useState<RecordingPhase>("idle");
  // The side panel is always showing: the 3D view above the raw data. Its width is the operator's to choose by
  // dragging the divider; the cameras, status bar and controls take whatever it leaves. The choice is remembered.
  const [sidebarWidth, setSidebarWidth] = useState(() => clampSidebarWidth(loadSidebarWidth(), window.innerWidth));
  const [resizing, setResizing] = useState(false);
  useEffect(() => {
    // A smaller window can no longer fit the remembered width, and the main column must keep its minimum.
    const refit = () => setSidebarWidth((width) => clampSidebarWidth(width, window.innerWidth));
    window.addEventListener("resize", refit);
    return () => window.removeEventListener("resize", refit);
  }, []);
  useEffect(() => {
    // Remembered once a drag ends (or after a key press), not on every pointer move.
    if (!resizing) saveSidebarWidth(sidebarWidth);
  }, [sidebarWidth, resizing]);
  const unavailable = response === null && feedHealth === "unavailable";

  return (
    <div className="app-shell app-shell--operator">
      <AppHeader response={response} feedHealthLabel={FEED_HEALTH_LABEL[feedHealth]} />
      <div
        className={`op-body${resizing ? " op-body--resizing" : ""}`}
        style={{ "--op-sidebar-width": `${sidebarWidth}px` } as CSSProperties}
      >
        <main className="app-main op-main">
          {unavailable ? (
            <section className="unavailable-panel">
              <p>Status updates unavailable{diagnostic !== null ? `: ${diagnostic}` : "."}</p>
              <button type="button" className="refresh-button" onClick={retry}>
                Retry
              </button>
            </section>
          ) : (
            <>
              <OperatorBar
                response={response}
                live={presentation.live}
                recordingSnapshot={recordingSnapshot}
                recordingSnapshotReceivedAtMonotonic={recordingSnapshotReceivedAtMonotonic}
                operatorName={operatorToken}
                hasControlToken={controlToken !== null}
                pendingPhase={panelPhase}
              />
              <StatusLine response={response} presentation={presentation} />
              <StaleOrErrorBanner presentation={presentation} feedHealth={feedHealth} diagnostic={diagnostic} onRetry={retry} />
              {/* The cameras are the page: they take every pixel the status bar above and the controls below leave. */}
              <div className={`op-stage${recording ? " op-stage--recording" : ""}`}>
                <CameraGrid snapshot={snapshot} live={presentation.live} receivedAtMonotonic={receivedAtMonotonic} />
              </div>
              <div className="op-dock">
                <RecordingPanel
                  recordingSnapshot={recordingSnapshot}
                  recordingSnapshotReceivedAtMonotonic={recordingSnapshotReceivedAtMonotonic}
                  actions={recordingActions}
                  operatorToken={operatorToken}
                  setOperatorToken={setOperatorToken}
                  clearOperatorToken={clearOperatorToken}
                  controlToken={controlToken}
                  setControlToken={setControlToken}
                  clearControlToken={clearControlToken}
                  onPhaseChange={setPanelPhase}
                />
              </div>
            </>
          )}
        </main>
        {!unavailable && (
          <>
            <SidebarResizer
              width={sidebarWidth}
              min={SIDEBAR_MIN_PX}
              max={maxSidebarWidth(window.innerWidth)}
              onResize={(width) => setSidebarWidth(clampSidebarWidth(width, window.innerWidth))}
              onReset={() => setSidebarWidth(clampSidebarWidth(SIDEBAR_DEFAULT_PX, window.innerWidth))}
              onDragChange={setResizing}
            />
            <aside className="op-sidebar" aria-label="3D view and details">
              <section className="op-panel op-panel--3d" aria-label="3D view">
                <PoseViserPanel
                  visualization={snapshot?.visualization ?? null}
                  backendInstanceId={snapshot?.backend_instance_id ?? null}
                />
              </section>
              <section className="op-panel op-panel--details" aria-label="Details">
                <CaptureTree
                  trackingSample={presentation.live ? trackingSample : null}
                  diagnostic={snapshot?.tracking?.diagnostic ?? null}
                />
                <SessionDetails response={response} live={presentation.live} freshnessLabel={presentation.freshnessLabel} />
              </section>
            </aside>
          </>
        )}
      </div>
    </div>
  );
}
