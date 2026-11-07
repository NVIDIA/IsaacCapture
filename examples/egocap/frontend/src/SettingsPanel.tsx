// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * The Settings dialog: a tabbed modal over the page.
 *
 * Tabs, in the order an operator sets up a station: Station (camera type), Tasks (the catalogs the task picker
 * offers, imports, and the preselected task), Tools (calibration and rectification), Display (this browser's
 * choices and the keyboard shortcuts), Advanced (what the deployment is running with, read-only). Station, Tasks
 * and Tools are saved by the backend for every browser and need the run's control token to change, as recording
 * commands do; Display is this browser's only; Advanced values are set where the backend is launched.
 */
import { useEffect, useRef, useState, type ReactNode } from "react";
import {
  CAMERA_TYPE_LABELS,
  ROLE_LABELS,
  fetchOperatorSettings,
  fetchSettings,
  fetchTaskFile,
  fetchTaskFiles,
  fetchTools,
  importTaskFile,
  removeTaskFile,
  saveOperatorSettings,
  type EffectiveSettings,
  type OperatorSettings,
  type OperatorSettingsChange,
  type TaskFileContents,
  type TaskFileListing,
  type TaskFileSource,
  type ToolInfo,
} from "./settingsApi";

export interface StationSetup {
  operator: OperatorSettings;
  tools: ToolInfo[];
  taskFiles: TaskFileListing;
}

/** The backend calls the dialog makes; tests inject fakes, the page uses the real ones. */
export interface SettingsBackend {
  loadDeployment: (signal: AbortSignal) => Promise<EffectiveSettings>;
  loadStationSetup: (signal: AbortSignal) => Promise<StationSetup>;
  loadTaskFile: (name: string, signal: AbortSignal) => Promise<TaskFileContents>;
  saveOperator: (controlToken: string, changes: OperatorSettingsChange) => Promise<OperatorSettings>;
  importTaskFile: (controlToken: string, source: TaskFileSource) => Promise<TaskFileListing & { file: TaskFileContents }>;
  removeTaskFile: (controlToken: string, name: string) => Promise<TaskFileListing>;
}

export const REAL_SETTINGS_BACKEND: SettingsBackend = {
  loadDeployment: fetchSettings,
  loadStationSetup: async (signal) => {
    const [operator, tools, taskFiles] = await Promise.all([fetchOperatorSettings(signal), fetchTools(signal), fetchTaskFiles(signal)]);
    return { operator, tools, taskFiles };
  },
  loadTaskFile: fetchTaskFile,
  saveOperator: saveOperatorSettings,
  importTaskFile,
  removeTaskFile,
};

export type SettingsTab = "station" | "tasks" | "tools" | "display" | "advanced";

const TABS: ReadonlyArray<{ id: SettingsTab; label: string }> = [
  { id: "station", label: "Station" },
  { id: "tasks", label: "Tasks" },
  { id: "tools", label: "Tools" },
  { id: "display", label: "Display" },
  { id: "advanced", label: "Advanced" },
];

export interface SettingsPanelProps {
  open: boolean;
  onClose: () => void;
  /** The current side-panel width and its default, for the Display tab. */
  sidebarWidth: number;
  sidebarDefault: number;
  onResetSidebar: () => void;
  /** The run's control token once the operator has entered it in the recording panel; null until then. */
  controlToken: string | null;
  /** The cameras the current session offers, by name (CaptureServer cameras and LiveKit tiles alike). */
  cameraSources?: string[];
  initialTab?: SettingsTab;
  backend?: SettingsBackend;
}

type Loaded<T> = { kind: "loading" } | { kind: "error"; message: string } | { kind: "loaded"; value: T };

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** Loads `load` each time the dialog opens, so a backend restart shows its new values; abandons a load the
 * dialog closed on. */
function useLoaded<T>(open: boolean, load: (signal: AbortSignal) => Promise<T>): [Loaded<T>, (update: (value: T) => T) => void] {
  const [state, setState] = useState<Loaded<T>>({ kind: "loading" });
  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    setState({ kind: "loading" });
    load(controller.signal).then(
      (value) => setState({ kind: "loaded", value }),
      (error: unknown) => {
        if (!controller.signal.aborted) setState({ kind: "error", message: errorMessage(error) });
      },
    );
    return () => controller.abort();
  }, [open, load]);
  const update = (change: (value: T) => T) => setState((current) => (current.kind === "loaded" ? { kind: "loaded", value: change(current.value) } : current));
  return [state, update];
}

/** A dash for an absent value, so the row still shows the setting exists. */
function orDash(value: string | number | null | undefined): string {
  return value === null || value === undefined || value === "" ? "—" : String(value);
}

function onOff(value: boolean): string {
  return value ? "On" : "Off";
}

function list(values: readonly string[]): string {
  return values.length === 0 ? "none" : values.join(", ");
}

function Row({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd className={mono ? "selectable" : undefined}>{value}</dd>
    </div>
  );
}

/** One titled block of a tab. */
function Card({ title, lead, children }: { title: string; lead?: string; children: ReactNode }) {
  return (
    <section className="settings-card" aria-label={title}>
      <h3>{title}</h3>
      {lead !== undefined && <p className="settings-lead">{lead}</p>}
      {children}
    </section>
  );
}

function TokenHint({ controlToken }: { controlToken: string | null }) {
  if (controlToken !== null) return null;
  return <p className="settings-note settings-note--hint">Enter the control token in the recording panel to change these.</p>;
}

/** The pending/failed state of one save-type action, shown next to its button. */
function useAction(): [{ busy: boolean; failure: string | null }, (work: () => Promise<void>) => void] {
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  const run = (work: () => Promise<void>) => {
    setBusy(true);
    setFailure(null);
    work().then(
      () => setBusy(false),
      (error: unknown) => {
        setBusy(false);
        setFailure(errorMessage(error));
      },
    );
  };
  return [{ busy, failure }, run];
}

// --- Station ---------------------------------------------------------------------------------------------------

function StationTab({
  setup,
  controlToken,
  backend,
  cameraSources,
  onOperator,
}: {
  setup: Loaded<StationSetup>;
  controlToken: string | null;
  backend: SettingsBackend;
  cameraSources: string[];
  onOperator: (operator: OperatorSettings) => void;
}) {
  if (setup.kind !== "loaded") return <LoadState state={setup} what="Station setup" />;
  const { operator } = setup.value;
  return (
    <>
      <ReadyLine operator={operator} cameraSources={cameraSources} />
      <Card title="Camera type" lead="Which cameras this station is built with. A label for operators and tools; CaptureServer and the station file open the cameras.">
        <CameraTypeChoice key={operator.camera_type ?? ""} operator={operator} controlToken={controlToken} save={backend.saveOperator} onSaved={onOperator} />
        <TokenHint controlToken={controlToken} />
      </Card>
      <Card title="Camera roles" lead="Which of the session's cameras is the ego view and which is on each wrist. The ego camera gets the main tile; roles are shown on every tile so a swapped wrist is caught before a session, not after.">
        <CameraRoles key={JSON.stringify(operator.camera_roles)} operator={operator} cameraSources={cameraSources} controlToken={controlToken} save={backend.saveOperator} onSaved={onOperator} />
      </Card>
    </>
  );
}

/** The station's readiness at a glance: what is set, what is still to do. */
function ReadyLine({ operator, cameraSources }: { operator: OperatorSettings; cameraSources: string[] }) {
  const assigned = operator.roles.filter((role) => operator.camera_roles[role] !== null);
  const offered = assigned.filter((role) => cameraSources.includes(operator.camera_roles[role] ?? ""));
  const items: Array<[boolean, string]> = [
    [operator.camera_type !== null, operator.camera_type === null ? "Camera type not set" : `Camera type: ${CAMERA_TYPE_LABELS[operator.camera_type]?.split(" — ")[0] ?? operator.camera_type}`],
    [assigned.length === operator.roles.length, `${assigned.length}/${operator.roles.length} roles assigned`],
    [assigned.length > 0 && offered.length === assigned.length, cameraSources.length === 0 ? "No cameras offered by the session yet" : `${offered.length}/${assigned.length} assigned cameras in this session`],
    [operator.task_file !== null, operator.task_file === null ? "No task catalog chosen" : `Task catalog: ${operator.task_file}`],
  ];
  return (
    <p className="settings-ready" aria-label="Station readiness">
      {items.map(([ok, text]) => (
        <span key={text} className={ok ? "settings-ready-item--ok" : "settings-ready-item--todo"}>
          {ok ? "✓" : "○"} {text}
        </span>
      ))}
    </p>
  );
}

/** One select per role over the cameras the session offers; a camera already saved for a role that the session
 * no longer offers stays selectable, so an operator can see (and clear) it. */
function CameraRoles({
  operator,
  cameraSources,
  controlToken,
  save,
  onSaved,
}: {
  operator: OperatorSettings;
  cameraSources: string[];
  controlToken: string | null;
  save: SettingsBackend["saveOperator"];
  onSaved: (operator: OperatorSettings) => void;
}) {
  const [roles, setRoles] = useState<Record<string, string | null>>({ ...operator.camera_roles });
  const [{ busy, failure }, run] = useAction();
  const dirty = operator.roles.some((role) => roles[role] !== operator.camera_roles[role]);
  const options = [...new Set([...cameraSources, ...Object.values(roles).filter((name): name is string => name !== null)])];
  const taken = (role: string) => new Set(operator.roles.filter((other) => other !== role).map((other) => roles[other]));
  const swapWrists = () => setRoles((current) => ({ ...current, wrist_left: current.wrist_right ?? null, wrist_right: current.wrist_left ?? null }));
  return (
    <>
      {options.length === 0 && <p className="settings-note">The session offers no cameras yet; roles can be assigned once CaptureServer lists them.</p>}
      <div className="settings-roles">
        {operator.roles.map((role) => (
          <FragmentRow key={role} role={role}>
            <select
              id={`role-${role}`}
              className="settings-input"
              value={roles[role] ?? ""}
              disabled={busy}
              onChange={(e) => setRoles((current) => ({ ...current, [role]: e.target.value === "" ? null : e.target.value }))}
            >
              <option value="">Not assigned</option>
              {options.map((name) => (
                <option key={name} value={name} disabled={taken(role).has(name)}>
                  {name}
                  {cameraSources.includes(name) ? "" : " (not in this session)"}
                </option>
              ))}
            </select>
          </FragmentRow>
        ))}
      </div>
      <div className="settings-actions">
        <button type="button" className="settings-primary" disabled={!dirty || controlToken === null || busy} onClick={() => controlToken !== null && run(() => save(controlToken, { camera_roles: roles }).then(onSaved))}>
          {busy ? "Saving…" : "Save camera roles"}
        </button>
        <button type="button" className="settings-quiet" disabled={busy || (roles.wrist_left === null && roles.wrist_right === null)} onClick={swapWrists}>
          Swap wrists
        </button>
        {failure !== null && <span className="settings-note tone-warning">Not saved: {failure}</span>}
      </div>
      <TokenHint controlToken={controlToken} />
    </>
  );
}

function FragmentRow({ role, children }: { role: string; children: ReactNode }) {
  return (
    <>
      <label htmlFor={`role-${role}`}>{ROLE_LABELS[role] ?? role}</label>
      {children}
    </>
  );
}

function CameraTypeChoice({
  operator,
  controlToken,
  save,
  onSaved,
}: {
  operator: OperatorSettings;
  controlToken: string | null;
  save: SettingsBackend["saveOperator"];
  onSaved: (operator: OperatorSettings) => void;
}) {
  const [choice, setChoice] = useState<string | null>(operator.camera_type);
  const [{ busy, failure }, run] = useAction();
  const dirty = choice !== operator.camera_type;
  return (
    <fieldset className="settings-fieldset">
      <legend className="visually-hidden">Choose the camera type</legend>
      {operator.camera_types.map((type) => (
        <label key={type} className={`settings-choice${choice === type ? " settings-choice--on" : ""}`}>
          <input type="radio" name="camera-type" value={type} checked={choice === type} onChange={() => setChoice(type)} />
          <span>{CAMERA_TYPE_LABELS[type] ?? type}</span>
        </label>
      ))}
      <label className={`settings-choice${choice === null ? " settings-choice--on" : ""}`}>
        <input type="radio" name="camera-type" value="" checked={choice === null} onChange={() => setChoice(null)} />
        <span>Not set</span>
      </label>
      <div className="settings-actions">
        <button
          type="button"
          className="settings-primary"
          disabled={!dirty || controlToken === null || busy}
          onClick={() => controlToken !== null && run(() => save(controlToken, { camera_type: choice }).then(onSaved))}
        >
          {busy ? "Saving…" : "Save camera type"}
        </button>
        {failure !== null && <span className="settings-note tone-warning">Not saved: {failure}</span>}
      </div>
    </fieldset>
  );
}

// --- Tasks -----------------------------------------------------------------------------------------------------

function TasksTab({
  setup,
  controlToken,
  backend,
  onSetup,
}: {
  setup: Loaded<StationSetup>;
  controlToken: string | null;
  backend: SettingsBackend;
  onSetup: (update: (value: StationSetup) => StationSetup) => void;
}) {
  if (setup.kind !== "loaded") return <LoadState state={setup} what="Task catalogs" />;
  const { operator, taskFiles } = setup.value;
  const canChange = controlToken !== null;
  return (
    <>
      <Card title="Catalogs" lead="The task lists this station can pick from. One is offered in the recording panel's task editor.">
        <CatalogList
          listing={taskFiles}
          chosen={operator.task_file}
          canChange={canChange}
          onChoose={(name) =>
            controlToken !== null &&
            backend.saveOperator(controlToken, { task_file: name, preselected_task: null }).then((saved) => onSetup((value) => ({ ...value, operator: saved })))
          }
          onRemove={(name) =>
            controlToken !== null &&
            backend.removeTaskFile(controlToken, name).then((listing) =>
              onSetup((value) => ({
                ...value,
                taskFiles: listing,
                // A removed catalog cannot stay the picker's source.
                operator: value.operator.task_file === name ? { ...value.operator, task_file: null, preselected_task: null } : value.operator,
              })),
            )
          }
        />
        <TokenHint controlToken={controlToken} />
      </Card>
      <Card title="Add a catalog" lead="A CSV in the studio or Mid Train layout. Give a path on this machine (for example a file in Downloads), or choose the file.">
        <ImportCatalog
          canChange={canChange}
          doImport={(source) => (controlToken === null ? Promise.reject(new Error("no control token")) : backend.importTaskFile(controlToken, source))}
          onImported={(result) => onSetup((value) => ({ ...value, taskFiles: { files: result.files, imported: result.imported } }))}
        />
      </Card>
      {operator.task_file !== null && (
        <Card title="Preselected task" lead="Filled into the task editor when the station's task is still unset, so a standing task is one keypress away.">
          <PreselectedTask
            key={`${operator.task_file}:${operator.preselected_task ?? ""}`}
            fileName={operator.task_file}
            current={operator.preselected_task}
            canChange={canChange}
            loadTaskFile={backend.loadTaskFile}
            save={(task) => (controlToken === null ? Promise.resolve() : backend.saveOperator(controlToken, { preselected_task: task }).then((saved) => onSetup((value) => ({ ...value, operator: saved }))))}
          />
        </Card>
      )}
    </>
  );
}

function CatalogList({
  listing,
  chosen,
  canChange,
  onChoose,
  onRemove,
}: {
  listing: TaskFileListing;
  chosen: string | null;
  canChange: boolean;
  onChoose: (name: string | null) => false | Promise<void>;
  onRemove: (name: string) => false | Promise<void>;
}) {
  const [{ busy, failure }, run] = useAction();
  if (listing.files.length === 0) return <p className="settings-note">No catalogs yet. Add one below.</p>;
  const imported = new Set(listing.imported);
  return (
    <>
      <ul className="settings-list" aria-label="Task catalogs">
        {listing.files.map((name) => (
          <li key={name} className={chosen === name ? "settings-list-item--on" : undefined}>
            <label className="settings-list-main">
              <input
                type="radio"
                name="task-file"
                checked={chosen === name}
                disabled={!canChange || busy}
                onChange={() => run(async () => void (await onChoose(name)))}
                aria-label={`Use ${name} in the task editor`}
              />
              <span className="settings-list-name">{name}</span>
              {imported.has(name) && <span className="settings-badge">imported</span>}
              {chosen === name && <span className="settings-badge settings-badge--on">in task editor</span>}
            </label>
            {imported.has(name) && (
              <button type="button" className="settings-quiet" disabled={!canChange || busy} onClick={() => run(async () => void (await onRemove(name)))}>
                Remove
              </button>
            )}
          </li>
        ))}
      </ul>
      <div className="settings-actions">
        <button type="button" className="settings-quiet" disabled={!canChange || busy || chosen === null} onClick={() => run(async () => void (await onChoose(null)))}>
          Free text only
        </button>
        {failure !== null && <span className="settings-note tone-warning">{failure}</span>}
      </div>
    </>
  );
}

function ImportCatalog({
  canChange,
  doImport,
  onImported,
}: {
  canChange: boolean;
  doImport: (source: TaskFileSource) => Promise<TaskFileListing & { file: TaskFileContents }>;
  onImported: (result: TaskFileListing & { file: TaskFileContents }) => void;
}) {
  const [path, setPath] = useState("");
  const [done, setDone] = useState<string | null>(null);
  const [{ busy, failure }, run] = useAction();
  const fileInput = useRef<HTMLInputElement | null>(null);
  const finish = (result: TaskFileListing & { file: TaskFileContents }) => {
    onImported(result);
    setDone(`Imported ${result.file.name}: ${result.file.tasks.length} task${result.file.tasks.length === 1 ? "" : "s"}.`);
    setPath("");
  };
  return (
    <>
      <div className="settings-import-row">
        <input
          type="text"
          className="settings-input"
          placeholder="/home/operator/Downloads/tasks.csv"
          value={path}
          disabled={!canChange || busy}
          onChange={(e) => setPath(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && path.trim() !== "") run(() => doImport({ path: path.trim() }).then(finish));
          }}
          aria-label="Path to a CSV file on this machine"
        />
        <button type="button" className="settings-primary" disabled={!canChange || busy || path.trim() === ""} onClick={() => run(() => doImport({ path: path.trim() }).then(finish))}>
          {busy ? "Importing…" : "Import path"}
        </button>
        <input
          ref={fileInput}
          type="file"
          accept=".csv,text/csv"
          className="visually-hidden"
          aria-label="Choose a CSV file"
          disabled={!canChange || busy}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file === undefined) return;
            run(() => file.text().then((content) => doImport({ name: file.name, content })).then(finish));
            e.target.value = "";
          }}
        />
        <button type="button" className="settings-quiet" disabled={!canChange || busy} onClick={() => fileInput.current?.click()}>
          Choose file…
        </button>
      </div>
      {done !== null && failure === null && <p className="settings-note">{done}</p>}
      {failure !== null && <p className="settings-note tone-warning">Not imported: {failure}</p>}
      {!canChange && <p className="settings-note settings-note--hint">Enter the control token in the recording panel to import.</p>}
    </>
  );
}

function PreselectedTask({
  fileName,
  current,
  canChange,
  loadTaskFile,
  save,
}: {
  fileName: string;
  current: string | null;
  canChange: boolean;
  loadTaskFile: SettingsBackend["loadTaskFile"];
  save: (task: string | null) => Promise<void>;
}) {
  const [file, setFile] = useState<Loaded<TaskFileContents>>({ kind: "loading" });
  const [choice, setChoice] = useState<string | null>(current);
  const [{ busy, failure }, run] = useAction();
  useEffect(() => {
    const controller = new AbortController();
    loadTaskFile(fileName, controller.signal).then(
      (value) => setFile({ kind: "loaded", value }),
      (error: unknown) => {
        if (!controller.signal.aborted) setFile({ kind: "error", message: errorMessage(error) });
      },
    );
    return () => controller.abort();
  }, [fileName, loadTaskFile]);
  if (file.kind !== "loaded") return <LoadState state={file} what={fileName} />;
  const known = file.value.tasks.some((task) => task.name === current);
  return (
    <>
      <select className="settings-input" aria-label="Preselected task" value={choice ?? ""} disabled={!canChange || busy} onChange={(e) => setChoice(e.target.value === "" ? null : e.target.value)}>
        <option value="">None</option>
        {file.value.tasks.map((task) => (
          <option key={task.name} value={task.name}>
            {task.name} — {task.instruction}
          </option>
        ))}
      </select>
      {current !== null && !known && <p className="settings-note tone-warning">The saved task "{current}" is not in {fileName} any more.</p>}
      <div className="settings-actions">
        <button type="button" className="settings-primary" disabled={!canChange || busy || choice === current} onClick={() => run(() => save(choice))}>
          {busy ? "Saving…" : "Save preselected task"}
        </button>
        {failure !== null && <span className="settings-note tone-warning">Not saved: {failure}</span>}
      </div>
    </>
  );
}

// --- Tools -----------------------------------------------------------------------------------------------------

function ToolsTab({ setup }: { setup: Loaded<StationSetup> }) {
  if (setup.kind !== "loaded") return <LoadState state={setup} what="Tools" />;
  const { tools, operator } = setup.value;
  return (
    <Card title="Operator tools" lead="Calibration and rectification. A tool opens only when this build can launch it; the others say why not.">
      <ul className="settings-list settings-list--tools">
        {tools.map((tool) => {
          const forOtherCameras = tool.applies_to !== null && operator.camera_type !== null && tool.applies_to !== operator.camera_type;
          const reason = !tool.available ? tool.unavailable_reason : forOtherCameras ? `for ${CAMERA_TYPE_LABELS[tool.applies_to ?? ""]?.split(" — ")[0] ?? tool.applies_to} stations` : null;
          return (
            <li key={tool.id}>
              <div className="settings-list-main settings-list-main--stacked">
                <span className="settings-list-name">{tool.name}</span>
                <span className="settings-note" id={`tool-${tool.id}-why`}>
                  {tool.description}
                  {reason !== null && ` — ${reason}.`}
                </span>
              </div>
              <button type="button" className="settings-quiet" disabled={!tool.available || forOtherCameras} aria-describedby={`tool-${tool.id}-why`}>
                Open
              </button>
            </li>
          );
        })}
      </ul>
    </Card>
  );
}

// --- Display ---------------------------------------------------------------------------------------------------

const SHORTCUTS: ReadonlyArray<[string, string]> = [
  ["s", "Record"],
  ["w", "Stop and save"],
  ["a", "Discard"],
  ["t", "Edit task"],
  ["r", "New batch"],
  ["Space", "Voice commands on/off"],
  ["Enter / Esc", "Save / cancel the task editor"],
];

function DisplayTab({ sidebarWidth, sidebarDefault, onResetSidebar }: { sidebarWidth: number; sidebarDefault: number; onResetSidebar: () => void }) {
  return (
    <>
      <Card title="Layout" lead="Remembered in this browser only.">
        <dl className="kv-list">
          <Row label="Side panel width" value={`${sidebarWidth} px${sidebarWidth === sidebarDefault ? " (default)" : ""}`} />
        </dl>
        <div className="settings-actions">
          <button type="button" className="settings-quiet" onClick={onResetSidebar} disabled={sidebarWidth === sidebarDefault}>
            Reset side panel width
          </button>
          <span className="settings-note">Drag the divider between the cameras and the side panel to change it.</span>
        </div>
      </Card>
      <Card title="Keyboard shortcuts" lead="Active while no text field has focus.">
        <dl className="kv-list settings-shortcuts">
          {SHORTCUTS.map(([key, action]) => (
            <div key={key}>
              <dt>{action}</dt>
              <dd>
                <kbd>{key}</kbd>
              </dd>
            </div>
          ))}
        </dl>
      </Card>
    </>
  );
}

// --- Advanced --------------------------------------------------------------------------------------------------

function AdvancedTab({ deployment }: { deployment: Loaded<EffectiveSettings> }) {
  if (deployment.kind !== "loaded") return <LoadState state={deployment} what="Deployment settings" />;
  const { capture, preview, station, recording, visualization, livekit, recordings_browser, task_files, origins } = deployment.value;
  const viewAddress = visualization.host === null ? null : `${visualization.host}:${orDash(visualization.port)}`;
  return (
    <>
      <p className="settings-lead">Set where the backend is launched (environment or station file); a change applies on restart.</p>
      <Card title="Capture" lead="EGOCAP_CAPTURE_SERVER_URL · EGOCAP_CAPTURE_SOURCES · EGOCAP_CAPTURE_PLUGINS">
        <dl className="kv-list">
          <Row label="CaptureServer" value={capture.server_url} mono />
          <Row label="Station label" value={capture.station_label} />
          <Row label="Profile" value={capture.profile_label} />
          <Row label="Tracking sources" value={list(capture.sources)} />
          <Row label="Plugins" value={list(capture.plugins)} />
          <Row label="Cameras requested" value={list(capture.cameras.map((camera) => `${camera.name} (${camera.type})`))} />
          <Row label="Preview rate, cameras" value={`${preview.camera_sample_hz} Hz`} />
          <Row label="Preview rate, tracking" value={`${preview.tracking_sample_hz} Hz`} />
        </dl>
      </Card>
      <Card title="Station file" lead="EGOCAP_STATION_CONFIG — an invalid file blocks recording but not preview">
        <dl className="kv-list">
          <Row label="File" value={station.path === null ? "not configured" : station.path} mono />
          {station.error !== null && <Row label="Status" value={station.error} />}
          <Row label="Station id" value={orDash(station.id)} />
          <Row label="Operator prefill" value={orDash(station.operator_name)} />
          <Row label="Auto-fill operator" value={onOff(station.auto_fill_operator_name)} />
          <Row label="Initial task" value={orDash(station.task_description)} />
          <Row label="Data dir" value={orDash(station.data_dir)} mono />
        </dl>
      </Card>
      <Card title="Recording">
        <dl className="kv-list">
          <Row label="Output root" value={recording.output_root} mono />
          <Row label="LiveKit credentials" value={livekit.credentials_configured ? "Configured" : "Missing — LiveKit streams preview only"} />
          <Row label="Recordings browser" value={onOff(recordings_browser.enabled)} />
          <Row label="Task catalog directory" value={task_files.dir} mono />
          <Row label="Task catalog available" value={task_files.available ? "Yes" : "No — directory missing"} />
        </dl>
      </Card>
      <Card title="3D view" lead="EGOCAP_VISER_* — Viser has no authentication, so it binds to loopback unless told otherwise">
        <dl className="kv-list">
          <Row label="Enabled" value={onOff(visualization.enabled)} />
          {visualization.error !== null && <Row label="Status" value={visualization.error} />}
          <Row label="Bind address" value={orDash(viewAddress)} mono />
          <Row label="Public URL" value={orDash(visualization.public_url)} mono />
          <Row label="Device profile" value={orDash(visualization.device)} />
          <Row label="Head model" value={orDash(visualization.head_model)} mono />
          <Row label="Head model scale" value={orDash(visualization.head_model_scale)} />
        </dl>
      </Card>
      <Card title="Access" lead="EGOCAP_ALLOWED_ORIGINS · EGOCAP_TRUSTED_HOSTNAMES — who may open the live session socket">
        <dl className="kv-list">
          <Row label="Allowed origins" value={list(origins.allowed)} mono />
          <Row
            label="Trusted hostnames"
            value={origins.trusted_hostnames === null ? "derived from each request (development default)" : list(origins.trusted_hostnames)}
            mono={origins.trusted_hostnames !== null}
          />
        </dl>
      </Card>
    </>
  );
}

// --- Shell -----------------------------------------------------------------------------------------------------

function LoadState<T>({ state, what }: { state: Loaded<T>; what: string }) {
  if (state.kind === "loading") return <p className="settings-note">Loading…</p>;
  if (state.kind === "error") return <p className="settings-note tone-warning">{what} unavailable: {state.message}</p>;
  return null;
}

export function SettingsPanel({
  open,
  onClose,
  sidebarWidth,
  sidebarDefault,
  onResetSidebar,
  controlToken,
  cameraSources = [],
  initialTab = "station",
  backend = REAL_SETTINGS_BACKEND,
}: SettingsPanelProps) {
  const [tab, setTab] = useState<SettingsTab>(initialTab);
  const [deployment] = useLoaded(open, backend.loadDeployment);
  const [setup, updateSetup] = useLoaded(open, backend.loadStationSetup);

  // Escape closes, as for any dialog.
  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  const setOperator = (operator: OperatorSettings) => updateSetup((value) => ({ ...value, operator }));

  return (
    <>
      <div className="settings-backdrop" onClick={onClose} aria-hidden="true" />
      <div className="settings-modal" role="dialog" aria-modal="true" aria-label="Settings">
        <nav className="settings-nav" aria-label="Settings sections">
          <h2>Settings</h2>
          {TABS.map((entry) => (
            <button
              key={entry.id}
              type="button"
              className={`settings-tab${tab === entry.id ? " settings-tab--on" : ""}`}
              aria-current={tab === entry.id ? "page" : undefined}
              onClick={() => setTab(entry.id)}
            >
              {entry.label}
            </button>
          ))}
        </nav>
        <div className="settings-content">
          <header className="settings-content-header">
            <h3>{TABS.find((entry) => entry.id === tab)?.label}</h3>
            <button type="button" className="settings-quiet" onClick={onClose}>
              Close
            </button>
          </header>
          <div className="settings-content-body">
            {tab === "station" && <StationTab setup={setup} controlToken={controlToken} backend={backend} cameraSources={cameraSources} onOperator={setOperator} />}
            {tab === "tasks" && <TasksTab setup={setup} controlToken={controlToken} backend={backend} onSetup={updateSetup} />}
            {tab === "tools" && <ToolsTab setup={setup} />}
            {tab === "display" && <DisplayTab sidebarWidth={sidebarWidth} sidebarDefault={sidebarDefault} onResetSidebar={onResetSidebar} />}
            {tab === "advanced" && <AdvancedTab deployment={deployment} />}
          </div>
        </div>
      </div>
    </>
  );
}
