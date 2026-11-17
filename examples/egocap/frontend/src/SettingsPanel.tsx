// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * The Settings dialog: a tabbed modal over the page.
 *
 * Tabs, in the order an operator sets up a station: Station (the stations, the one in use, and each station's
 * cameras; the camera setup itself while there are no stations), Devices (the live checklist of what the station
 * in use needs, whether calibration files are required, and the calibration tools), Tasks (the catalogs the task
 * picker offers, imports, and the preselected task), Display (this browser's choices and the keyboard
 * shortcuts), Advanced (what the deployment is running with, read-only). Station, Tasks and the calibration
 * requirement are saved by the backend for every browser and need the run's control token to change, as
 * recording commands do; the rest of Devices only reports; Display is this browser's only; Advanced values are
 * set where the backend is launched.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { CameraSetupCard } from "./CameraSetupCard";
import { DevicesTab } from "./DevicesTab";
import { Card, LoadState, TokenHint, errorMessage, useAction, useEscapeToClose, useModalFocus, type Loaded } from "./settingsParts";
import { StationsCard } from "./StationsCard";
import { fetchSetup, type SetupStatus } from "./setupApi";
import { addStation, deleteStation, fetchStations, selectStation, updateStation, type StationChange, type StationListing } from "./stationsApi";
import {
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
  /** The camera setup in effect: the station in use supplies it, else the values saved with the operator settings. */
  operator: OperatorSettings;
  tools: ToolInfo[];
  taskFiles: TaskFileListing;
  stations: StationListing;
}

/** The backend calls the dialog makes; tests inject fakes, the page uses the real ones. */
export interface SettingsBackend {
  loadDeployment: (signal: AbortSignal, controlToken?: string | null) => Promise<EffectiveSettings>;
  loadStationSetup: (signal: AbortSignal) => Promise<StationSetup>;
  loadTaskFile: (name: string, signal: AbortSignal) => Promise<TaskFileContents>;
  saveOperator: (controlToken: string, changes: OperatorSettingsChange) => Promise<OperatorSettings>;
  importTaskFile: (controlToken: string, source: TaskFileSource) => Promise<TaskFileListing & { file: TaskFileContents }>;
  removeTaskFile: (controlToken: string, name: string) => Promise<TaskFileListing>;
  addStation: (controlToken: string, fields: StationChange) => Promise<StationListing>;
  updateStation: (controlToken: string, id: string, changes: StationChange) => Promise<StationListing>;
  deleteStation: (controlToken: string, id: string) => Promise<StationListing>;
  selectStation: (controlToken: string, id: string | null) => Promise<StationListing>;
  /** The start-of-day flow, which the Devices tab reads on its own, faster timer. */
  loadSetup: (signal: AbortSignal, controlToken?: string | null) => Promise<SetupStatus>;
}

export const REAL_SETTINGS_BACKEND: SettingsBackend = {
  loadDeployment: fetchSettings,
  loadStationSetup: async (signal) => {
    const [operator, tools, taskFiles, stations] = await Promise.all([
      fetchOperatorSettings(signal),
      fetchTools(signal),
      fetchTaskFiles(signal),
      fetchStations(signal),
    ]);
    return { operator, tools, taskFiles, stations };
  },
  loadTaskFile: fetchTaskFile,
  saveOperator: saveOperatorSettings,
  importTaskFile,
  removeTaskFile,
  addStation,
  updateStation,
  deleteStation,
  selectStation,
  loadSetup: fetchSetup,
};

export type SettingsTab = "station" | "devices" | "tasks" | "display" | "advanced";

const TABS: ReadonlyArray<{ id: SettingsTab; label: string }> = [
  { id: "station", label: "Station" },
  { id: "devices", label: "Devices" },
  { id: "tasks", label: "Tasks" },
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
  /** The cameras the current session offers, by name. */
  cameraSources?: string[];
  /** The start-of-day flow as the page last read it; null while it could not be read. */
  setup?: SetupStatus | null;
  initialTab?: SettingsTab;
  backend?: SettingsBackend;
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
  /** Apply successful local changes only after initial setup has loaded. */
  const update = (change: (value: T) => T) => setState((current) => (current.kind === "loaded" ? { kind: "loaded", value: change(current.value) } : current));
  return [state, update];
}

/** A dash for an absent value, so the row still shows the setting exists. */
function orDash(value: string | number | null | undefined): string {
  return value === null || value === undefined || value === "" ? "—" : String(value);
}

/** Display boolean deployment settings in human-readable form. */
function onOff(value: boolean): string {
  return value ? "On" : "Off";
}

/** Display a list without leaving an empty settings value. */
function list(values: readonly string[]): string {
  return values.length === 0 ? "none" : values.join(", ");
}

/** Render a labelled read-only setting with optional monospace text. */
function Row({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd className={mono ? "selectable" : undefined}>{value}</dd>
    </div>
  );
}

// --- Station ---------------------------------------------------------------------------------------------------

/** The stations and the one in use; each station's cameras are edited in its own row. While there are no
 * stations the camera setup is edited here instead, since no station holds it yet. */
function StationTab({
  stationSetup,
  setup,
  controlToken,
  backend,
  cameraSources,
  onStations,
  onSetup,
}: {
  stationSetup: Loaded<StationSetup>;
  setup: SetupStatus | null;
  controlToken: string | null;
  backend: SettingsBackend;
  cameraSources: string[];
  onStations: (stations: StationListing) => void;
  onSetup: (update: (value: StationSetup) => StationSetup) => void;
}) {
  if (stationSetup.kind !== "loaded") return <LoadState state={stationSetup} what="Station setup" />;
  const { operator, stations } = stationSetup.value;
  return (
    <>
      <StationsCard
        listing={stations}
        controlToken={controlToken}
        backend={backend}
        // The first station takes over the camera setup saved before any station existed.
        inherited={{ camera_type: operator.camera_type, camera_roles: operator.camera_roles }}
        cameraSources={cameraSources}
        headset={setup?.headset ?? null}
        onListing={onStations}
      />
      {stations.stations.length === 0 && (
        <CameraSetupCard
          operator={operator}
          cameraSources={cameraSources}
          controlToken={controlToken}
          save={(changes) => (controlToken === null ? Promise.resolve() : backend.saveOperator(controlToken, changes).then((saved) => onSetup((value) => ({ ...value, operator: saved }))))}
        />
      )}
    </>
  );
}

// --- Tasks -----------------------------------------------------------------------------------------------------

/** Manage the selected catalog, task defaults and browser uploads. */
function TasksTab({
  stationSetup,
  controlToken,
  backend,
  onSetup,
}: {
  stationSetup: Loaded<StationSetup>;
  controlToken: string | null;
  backend: SettingsBackend;
  onSetup: (update: (value: StationSetup) => StationSetup) => void;
}) {
  if (stationSetup.kind !== "loaded") return <LoadState state={stationSetup} what="Task catalogs" />;
  const { operator, taskFiles } = stationSetup.value;
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
      <Card title="Add a catalog" lead="Choose a CSV file from this browser in the studio or Mid Train layout.">
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

/** List catalogs and allow removal only after the selected catalog is released. */
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
              <button type="button" className="settings-quiet" disabled={!canChange || busy || chosen === name} title={chosen === name ? "Choose another catalog or Free text only before removing this catalog." : undefined} onClick={() => run(async () => void (await onRemove(name)))}>
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

/** Upload a browser-selected CSV through the authenticated catalog API. */
function ImportCatalog({
  canChange,
  doImport,
  onImported,
}: {
  canChange: boolean;
  doImport: (source: TaskFileSource) => Promise<TaskFileListing & { file: TaskFileContents }>;
  onImported: (result: TaskFileListing & { file: TaskFileContents }) => void;
}) {
  const [done, setDone] = useState<string | null>(null);
  const [{ busy, failure }, run] = useAction();
  const fileInput = useRef<HTMLInputElement | null>(null);
  /** Refresh the catalog listing after a successful upload. */
  const finish = (result: TaskFileListing & { file: TaskFileContents }) => {
    onImported(result);
    setDone(`Imported ${result.file.name}: ${result.file.tasks.length} task${result.file.tasks.length === 1 ? "" : "s"}.`);
  };
  return (
    <>
      <div className="settings-import-row">
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

/** Choose a default from the tasks in the selected catalog. */
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

/** Explain local layout controls and restore the default sidebar width. */
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

/** Show authenticated deployment details without exposing secrets. */
function AdvancedTab({ deployment }: { deployment: Loaded<EffectiveSettings> }) {
  if (deployment.kind !== "loaded") return <LoadState state={deployment} what="Deployment settings" />;
  const { capture, preview, station, recording, visualization, recordings_browser, task_files, origins } = deployment.value;
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
      <Card title="Station file" lead="EGOCAP_STATION_CONFIG — an invalid file blocks recording and camera requests">
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

/** Present station settings in a modal that owns focus until it closes. */
export function SettingsPanel({
  open,
  onClose,
  sidebarWidth,
  sidebarDefault,
  onResetSidebar,
  controlToken,
  cameraSources = [],
  setup = null,
  initialTab = "station",
  backend = REAL_SETTINGS_BACKEND,
}: SettingsPanelProps) {
  const dialogRef = useModalFocus(open);
  const [tab, setTab] = useState<SettingsTab>(initialTab);
  /** Reload restricted details when the active credential changes. */
  const loadDeployment = useCallback((signal: AbortSignal) => backend.loadDeployment(signal, controlToken), [backend, controlToken]);
  const [deployment] = useLoaded(open && controlToken !== null, loadDeployment);
  const { loadStationSetup } = backend;
  const [stationSetup, applyStationSetup] = useLoaded(open, loadStationSetup);
  // The newest read of the station setup asked for after a change, and whether it is still to answer. A read is
  // a picture of the backend from when it was sent, so only the newest may replace what the dialog shows.
  const reloads = useRef({ latest: 0, pending: false });

  useEscapeToClose(open, onClose);

  if (!open) return null;

  /** Read the station setup again, and show it unless a newer read or change has been made by the time it answers. */
  const reloadStationSetup = () => {
    const mine = ++reloads.current.latest;
    reloads.current.pending = true;
    /** Whether this read is still the newest; if so it is no longer pending. */
    const settled = () => {
      if (reloads.current.latest !== mine) return false;
      reloads.current.pending = false;
      return true;
    };
    loadStationSetup(new AbortController().signal).then(
      (fresh) => {
        if (settled()) applyStationSetup(() => fresh);
      },
      () => void settled(), // the change itself succeeded; the next open reloads everything
    );
  };
  /** Apply a saved change to what the dialog shows. A read still on its way was sent before this change and
   * would put the older value back, so it is replaced by a new one. */
  const updateStationSetup = (update: (value: StationSetup) => StationSetup) => {
    applyStationSetup(update);
    if (reloads.current.pending) reloadStationSetup();
  };
  /** Show a changed registry at once, then read the station setup again: the station in use supplies the camera
   * type and roles the rest of the dialog and the page's tiles go by, so a change of it changes those. */
  const setStations = (stations: StationListing) => {
    applyStationSetup((value) => ({ ...value, stations }));
    reloadStationSetup();
  };
  /** Save whether calibration files are required, and show the saved value. */
  const saveRequireCalibration = (required: boolean): Promise<void> =>
    controlToken === null ? Promise.resolve() : backend.saveOperator(controlToken, { require_calibration: required }).then((saved) => updateStationSetup((value) => ({ ...value, operator: saved })));

  return (
    <>
      <div className="settings-backdrop" onClick={onClose} aria-hidden="true" />
      <div ref={dialogRef} tabIndex={-1} className="settings-modal" role="dialog" aria-modal="true" aria-label="Settings">
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
            {stationSetup.kind === "loaded" && stationSetup.value.operator.warning && <p className="settings-note tone-warning">{stationSetup.value.operator.warning}</p>}
            {tab === "station" && <StationTab stationSetup={stationSetup} setup={setup} controlToken={controlToken} backend={backend} cameraSources={cameraSources} onStations={setStations} onSetup={updateStationSetup} />}
            {tab === "devices" && <DevicesTab setup={setup} controlToken={controlToken} loadSetup={backend.loadSetup} stationSetup={stationSetup} saveRequireCalibration={saveRequireCalibration} />}
            {tab === "tasks" && <TasksTab stationSetup={stationSetup} controlToken={controlToken} backend={backend} onSetup={updateStationSetup} />}
            {tab === "display" && <DisplayTab sidebarWidth={sidebarWidth} sidebarDefault={sidebarDefault} onResetSidebar={onResetSidebar} />}
            {tab === "advanced" && (controlToken === null ? <p className="settings-note">Enter the control token in the recording panel to view deployment settings.</p> : <AdvancedTab deployment={deployment} />)}
          </div>
        </div>
      </div>
    </>
  );
}
