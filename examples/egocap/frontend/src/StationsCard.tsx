// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * The Stations card of the Settings dialog: the stations this site records at, the one in use, and each
 * station's name, camera type and cameras, edited in place.
 *
 * The station in use names every recording and its activity log. It cannot change while a take is being
 * recorded; the backend says so in `locked_reason` and refuses the change either way. A station that is not in
 * use can always be added, edited or deleted.
 */
import { useEffect, useId, useRef, useState } from "react";
import { Card, TokenHint, useAction } from "./settingsParts";
import { CAMERA_TYPE_LABELS, ROLE_LABELS } from "./settingsApi";
import { sentence } from "./sentence";
import type { HeadsetLink } from "./setupApi";
import type { StationChange, StationListing, StationRecord } from "./stationsApi";

/** The registry calls the card makes; the dialog passes the real ones and tests pass fakes. */
export interface StationsBackend {
  addStation: (controlToken: string, fields: StationChange) => Promise<StationListing>;
  updateStation: (controlToken: string, id: string, changes: StationChange) => Promise<StationListing>;
  deleteStation: (controlToken: string, id: string) => Promise<StationListing>;
  selectStation: (controlToken: string, id: string | null) => Promise<StationListing>;
}

/** The longest name the backend accepts, so the field stops where a save would be refused. */
const MAX_NAME_CHARS = 64;

/** Which camera fills each role, in one line, so two stations with different cameras can be told apart at a glance. */
function roleSummary(station: StationRecord, roles: readonly string[]): string {
  return roles.map((role) => `${ROLE_LABELS[role] ?? role}: ${station.camera_roles[role] ?? "—"}`).join(" · ");
}

/** What to tell the operator when the headset still holds an address this host no longer has; null otherwise. */
function headsetWarning(headset: HeadsetLink | null): string | null {
  if (headset === null || !headset.changed || headset.address === null || headset.saved === null) return null;
  return `This host's address changed from ${headset.saved} to ${headset.address}. Enter the new one in the headset.`;
}

export interface StationsCardProps {
  listing: StationListing;
  controlToken: string | null;
  backend: StationsBackend;
  /** The camera setup saved before any station existed; the first station added takes it over. */
  inherited: StationChange;
  /** Whether changes are held by something outside the card, such as a camera setup save the first station would inherit. */
  held?: boolean;
  /** The cameras the current session offers, by name. */
  cameraSources: string[];
  /** Where the headset of the station in use connects, from the start-of-day flow; null while that is not known. */
  headset: HeadsetLink | null;
  /** Called with the backend's listing after every successful change. */
  onListing: (listing: StationListing) => void;
}

/** List the stations, choose the one in use, and add, edit or delete them. */
export function StationsCard({ listing, controlToken, backend, inherited, held = false, cameraSources, headset, onListing }: StationsCardProps) {
  const [{ busy, failure }, run] = useAction();
  // At most one editor is open, so a draft is never left behind in a row the operator has moved on from.
  const [editing, setEditing] = useState<string | null>(null);
  // An editor's save reports to its own row, so it is tracked apart from the card's other changes.
  const [saving, setSaving] = useState(false);
  const locked = listing.locked_reason !== null;
  const idle = controlToken !== null && !busy;
  // Every change answers with a whole listing and the last answer wins, so only one may be in flight.
  const canChange = idle && !saving && !held;
  /** Run one registry change with the token and hand its listing to the dialog; resolves once it has ended either way. */
  const change = (work: (token: string) => Promise<StationListing>): Promise<void> => {
    if (controlToken === null) return Promise.resolve();
    const outcome = work(controlToken).then(onListing);
    run(() => outcome);
    // The card shows a refusal itself, so a caller waiting for the end is not handed it again.
    return outcome.catch(() => undefined);
  };
  /** Add a station; the first one takes over the camera setup already saved and is put in use. Resolves to whether it now exists. */
  const add = (name: string): Promise<boolean> => {
    let exists = false;
    return change(async (token) => {
      const first = listing.stations.length === 0;
      const added = await backend.addStation(token, first ? { ...inherited, name } : { name });
      // From here the station is in the registry, whatever becomes of putting it in use.
      exists = true;
      // During a take the station in use cannot change, so the new station is left for the operator to select.
      if (!first || added.locked_reason !== null) return added;
      try {
        return await backend.selectStation(token, added.stations[added.stations.length - 1].id);
      } catch (error) {
        // The station exists even though it could not be put in use: show its row, then the refusal.
        onListing(added);
        throw error;
      }
    }).then(() => exists);
  };
  /** Save one editor's changes. A refusal is the editor's to show, beside what was typed. */
  const save = (id: string, changes: StationChange): Promise<void> => {
    if (controlToken === null) return Promise.reject(new Error("enter the control token first"));
    setSaving(true);
    return backend
      .updateStation(controlToken, id, changes)
      .then(onListing)
      .finally(() => setSaving(false));
  };

  return (
    <Card title="Stations" lead="The stations this site records at, each with its own cameras. The one in use names every recording and its activity log.">
      {listing.warning !== undefined && <p className="settings-note tone-warning">{listing.warning}</p>}
      {listing.stations.length === 0 ? (
        <p className="settings-note">
          No stations yet. Recordings carry the deployment's label, “{listing.label}”. Add a station to choose between stations here.
        </p>
      ) : (
        <ul className="settings-list settings-list--stations" aria-label="Stations">
          {listing.stations.map((station) => {
            const inUse = listing.selected === station.id;
            return (
              <StationRow
                key={station.id}
                station={station}
                listing={listing}
                inUse={inUse}
                canChange={canChange}
                // The open editor's own save must not disable its Save button under the operator.
                canSave={idle}
                editing={editing === station.id}
                cameraSources={cameraSources}
                // The flow reports the headset of the station in use only.
                headset={inUse ? headset : null}
                onSelect={() => change((token) => backend.selectStation(token, station.id))}
                onEdit={() => setEditing(station.id)}
                // A save that answers after another row's editor was opened must not close that one.
                onCloseEditor={() => setEditing((current) => (current === station.id ? null : current))}
                onSave={(changes) => save(station.id, changes)}
                onDelete={() => change((token) => backend.deleteStation(token, station.id))}
              />
            );
          })}
          <li className={listing.selected === null ? "settings-list-item--on" : undefined}>
            <label className="settings-list-main">
              <input
                type="radio"
                name="station-in-use"
                checked={listing.selected === null}
                disabled={!canChange || locked}
                onChange={() => change((token) => backend.selectStation(token, null))}
                aria-label="Use no station"
              />
              <span className="settings-list-name">No station</span>
              <span className="settings-note">Recording is held until a station is in use.</span>
            </label>
          </li>
        </ul>
      )}
      <AddStation canChange={canChange} onAdd={add} />
      {locked && <p className="settings-note">{sentence(listing.locked_reason ?? "")}.</p>}
      {failure !== null && <p className="settings-note tone-warning">Not saved: {failure}</p>}
      <TokenHint controlToken={controlToken} />
    </Card>
  );
}

/** One station: the radio that puts it in use, what fills its roles, why it cannot record, and its actions. */
function StationRow({
  station,
  listing,
  inUse,
  canChange,
  canSave,
  editing,
  cameraSources,
  headset,
  onSelect,
  onEdit,
  onCloseEditor,
  onSave,
  onDelete,
}: {
  station: StationRecord;
  listing: StationListing;
  inUse: boolean;
  /** Whether the registry can be changed from this row now: nothing else is in flight. */
  canChange: boolean;
  /** Whether the row's open editor can save: as `canChange`, but not held by that editor's own save. */
  canSave: boolean;
  editing: boolean;
  cameraSources: string[];
  headset: HeadsetLink | null;
  onSelect: () => void;
  onEdit: () => void;
  onCloseEditor: () => void;
  onSave: (changes: StationChange) => Promise<void>;
  onDelete: () => void;
}) {
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const locked = listing.locked_reason !== null;
  // The station in use is frozen during a take; every other station stays editable.
  const frozen = inUse && locked;
  const editable = canChange && !frozen;
  const warning = headsetWarning(headset);
  // The row's actions give way to the editor or the delete confirmation and come back as new elements. When
  // that leaves keyboard focus on nothing, it returns to Edit; focus that already moved elsewhere is left alone.
  const editButton = useRef<HTMLButtonElement>(null);
  const replaced = editing || confirmingDelete;
  const wasReplaced = useRef(false);
  useEffect(() => {
    if (wasReplaced.current && !replaced && document.activeElement === document.body) editButton.current?.focus();
    wasReplaced.current = replaced;
  }, [replaced]);
  return (
    <li className={inUse ? "settings-list-item--on" : undefined}>
      <div className="settings-list-main settings-list-main--stacked">
        <label className="settings-station-head">
          <input type="radio" name="station-in-use" checked={inUse} disabled={!canChange || locked} onChange={onSelect} aria-label={`Use ${station.name}`} />
          <span className="settings-list-name">{station.name}</span>
          {inUse && <span className="settings-badge settings-badge--on">in use</span>}
        </label>
        <span className="settings-note">{roleSummary(station, listing.roles)}</span>
        {station.problems.length > 0 && <span className="settings-note tone-warning">Cannot record yet: {station.problems.join("; ")}.</span>}
        {/* The open editor says the same beside the address, so it is not repeated here. */}
        {warning !== null && !editing && <span className="settings-note tone-warning">{warning}</span>}
      </div>
      {!editing && !confirmingDelete && (
        <>
          <button ref={editButton} type="button" className="settings-quiet" disabled={!editable} onClick={onEdit}>
            Edit
          </button>
          <button type="button" className="settings-quiet" disabled={!editable} onClick={() => setConfirmingDelete(true)}>
            Delete
          </button>
        </>
      )}
      {!editing && confirmingDelete && (
        <>
          <button
            type="button"
            className="settings-danger"
            // Takes the focus the Delete button it replaces had, so the confirmation is the next key press.
            autoFocus
            disabled={!editable}
            onClick={() => {
              setConfirmingDelete(false);
              onDelete();
            }}
          >
            Delete {station.name}
          </button>
          <button type="button" className="settings-quiet" onClick={() => setConfirmingDelete(false)}>
            Keep
          </button>
        </>
      )}
      {editing && (
        <StationEditor station={station} listing={listing} cameraSources={cameraSources} headset={headset} canSave={canSave && !frozen} onSave={onSave} onClose={onCloseEditor} />
      )}
    </li>
  );
}

/** A camera setup as saved: the camera type and the camera filling each role. */
export type CameraSetup = Pick<StationRecord, "camera_type" | "camera_roles">;

/** An editor's working copy of a camera setup. */
export interface CameraDraft {
  cameraType: string | null;
  cameraRoles: Record<string, string | null>;
}

/** The editor's working copy of a station. */
interface StationDraft extends CameraDraft {
  name: string;
}

/** A working copy of a saved camera setup. */
export function cameraDraftOf(saved: CameraSetup): CameraDraft {
  return { cameraType: saved.camera_type, cameraRoles: { ...saved.camera_roles } };
}

/** Only the camera fields the draft changes from `original`, the setup the editor was opened on. */
export function cameraChangesOf(original: CameraSetup, roles: readonly string[], draft: CameraDraft): Pick<StationChange, "camera_type" | "camera_roles"> {
  const changes: Pick<StationChange, "camera_type" | "camera_roles"> = {};
  if (draft.cameraType !== original.camera_type) changes.camera_type = draft.cameraType;
  const moved = roles.filter((role) => (draft.cameraRoles[role] ?? null) !== (original.camera_roles[role] ?? null));
  if (moved.length > 0) changes.camera_roles = Object.fromEntries(moved.map((role) => [role, draft.cameraRoles[role] ?? null]));
  return changes;
}

/** Only what the draft changes from `original`, the station as it was when the editor opened. Comparing with
 * the station as it is now would send back, as a change, every field another browser changed in the meantime. */
function changesOf(original: StationRecord, roles: readonly string[], draft: StationDraft): StationChange {
  const changes: StationChange = cameraChangesOf(original, roles, draft);
  if (draft.name.trim() !== original.name) changes.name = draft.name.trim();
  return changes;
}

/** Whether a draft has a wrist camera to swap. */
export function hasWristCamera(draft: CameraDraft): boolean {
  return (draft.cameraRoles.wrist_left ?? null) !== null || (draft.cameraRoles.wrist_right ?? null) !== null;
}

/** The draft with its wrists exchanged, in one step so the assignments stay unique throughout. */
export function withWristsSwapped<T extends CameraDraft>(draft: T): T {
  return { ...draft, cameraRoles: { ...draft.cameraRoles, wrist_left: draft.cameraRoles.wrist_right ?? null, wrist_right: draft.cameraRoles.wrist_left ?? null } };
}

/** The camera type and the camera for each role, as label and control pairs for a `settings-fields` grid. */
export function CameraFields<T extends CameraDraft>({
  idPrefix,
  draft,
  onDraft,
  saved,
  roles,
  cameraTypes,
  cameraSources,
  disabled,
}: {
  /** Makes the field ids unique on the page. */
  idPrefix: string;
  draft: T;
  onDraft: (update: (current: T) => T) => void;
  /** The roles as saved: a saved camera stays selectable when the session no longer offers it, so it can be seen and cleared. */
  saved: Record<string, string | null>;
  roles: readonly string[];
  cameraTypes: readonly string[];
  /** The cameras the current session offers, by name. */
  cameraSources: string[];
  disabled: boolean;
}) {
  const cameras = selectableCameras(cameraSources, saved);
  /** The cameras the other roles hold: one camera cannot fill two roles. */
  const takenBy = (role: string) => new Set(roles.filter((other) => other !== role).map((other) => draft.cameraRoles[other]));
  return (
    <>
      <label htmlFor={`${idPrefix}-type`}>Camera type</label>
      <select
        id={`${idPrefix}-type`}
        className="settings-input"
        value={draft.cameraType ?? ""}
        disabled={disabled}
        onChange={(e) => onDraft((current) => ({ ...current, cameraType: e.target.value === "" ? null : e.target.value }))}
      >
        <option value="">Not set</option>
        {cameraTypes.map((type) => (
          <option key={type} value={type}>
            {CAMERA_TYPE_LABELS[type] ?? type}
          </option>
        ))}
      </select>
      {roles.map((role) => (
        <RoleField
          key={role}
          id={`${idPrefix}-${role}`}
          role={role}
          value={draft.cameraRoles[role] ?? null}
          cameras={cameras}
          offered={cameraSources}
          taken={takenBy(role)}
          disabled={disabled}
          onChange={(camera) => onDraft((current) => ({ ...current, cameraRoles: { ...current.cameraRoles, [role]: camera } }))}
        />
      ))}
    </>
  );
}

/** Every camera a role can be given: the ones the session offers and the ones already saved. */
export function selectableCameras(cameraSources: readonly string[], saved: Record<string, string | null>): string[] {
  return [...new Set([...cameraSources, ...Object.values(saved).filter((name): name is string => name !== null)])];
}

/** What to say where no camera can be chosen yet. */
export const NO_CAMERAS_NOTE = "The session offers no cameras yet; roles can be assigned once CaptureServer lists them.";

/** A station's inline editor: its name, camera type and the camera for each role, saved together. */
function StationEditor({
  station,
  listing,
  cameraSources,
  headset,
  canSave,
  onSave,
  onClose,
}: {
  station: StationRecord;
  listing: StationListing;
  cameraSources: string[];
  /** Where this station's headset connects; null for a station that is not in use. */
  headset: HeadsetLink | null;
  canSave: boolean;
  onSave: (changes: StationChange) => Promise<void>;
  onClose: () => void;
}) {
  const fieldId = useId();
  // The station as the editor opened on it. The listing can bring another browser's change while the editor is
  // open; what this editor changed is judged against what the operator started from, never against that.
  const [original] = useState(station);
  const [draft, setDraft] = useState<StationDraft>({ name: station.name, ...cameraDraftOf(station) });
  const [{ busy, failure }, run] = useAction();
  const { roles } = listing;
  const changes = changesOf(original, roles, draft);
  const dirty = Object.keys(changes).length > 0;
  const named = draft.name.trim() !== "";
  /** Send what changed and close on success; with nothing changed there is nothing to send. */
  const save = () => {
    if (!named || busy) return;
    if (!dirty) onClose();
    else if (canSave) run(() => onSave(changes).then(onClose));
  };
  const warning = headsetWarning(headset);
  return (
    <div
      className="station-editor"
      role="group"
      aria-label={`Edit ${station.name}`}
      onKeyDown={(e) => {
        // Escape belongs to the dialog, which closes on it; here it only closes the editor.
        if (e.key === "Escape") {
          e.stopPropagation();
          onClose();
        }
      }}
    >
      <div className="settings-fields">
        <label htmlFor={`${fieldId}-name`}>Name</label>
        <input
          id={`${fieldId}-name`}
          className="settings-input"
          value={draft.name}
          maxLength={MAX_NAME_CHARS}
          autoFocus
          // Read-only, not disabled, while saving: Enter here starts the save, and a disabled field gives up
          // keyboard focus, after which Escape would close the whole dialog instead of this editor.
          readOnly={busy}
          onChange={(e) => setDraft((current) => ({ ...current, name: e.target.value }))}
          onKeyDown={(e) => {
            if (e.key === "Enter") save();
          }}
        />
        <CameraFields idPrefix={fieldId} draft={draft} onDraft={setDraft} saved={station.camera_roles} roles={roles} cameraTypes={listing.camera_types} cameraSources={cameraSources} disabled={busy} />
      </div>
      {selectableCameras(cameraSources, station.camera_roles).length === 0 && <p className="settings-note">{NO_CAMERAS_NOTE}</p>}
      <div className="settings-actions">
        {/* Not disabled while saving, for the same reason as the name field; `save` ignores a second press. */}
        <button type="button" className="settings-primary" disabled={!canSave || !dirty || !named} aria-busy={busy} onClick={save}>
          {busy ? "Saving…" : "Save"}
        </button>
        <button type="button" className="settings-quiet" disabled={busy || !hasWristCamera(draft)} onClick={() => setDraft(withWristsSwapped)}>
          Swap wrists
        </button>
        <button type="button" className="settings-quiet" onClick={onClose}>
          Cancel
        </button>
        {failure !== null && <span className="settings-note tone-warning">Not saved: {failure}</span>}
      </div>
      {headset !== null && headset.address !== null && <p className="settings-note">Headset connects to {headset.address}</p>}
      {warning !== null && <p className="settings-note tone-warning">{warning}</p>}
    </div>
  );
}

/** The label and select of one camera role. */
function RoleField({
  id,
  role,
  value,
  cameras,
  offered,
  taken,
  disabled,
  onChange,
}: {
  id: string;
  role: string;
  value: string | null;
  /** Every camera that can be chosen. */
  cameras: string[];
  /** The ones the session offers right now; the rest are marked. */
  offered: string[];
  /** The cameras the other roles hold. */
  taken: Set<string | null | undefined>;
  disabled: boolean;
  onChange: (camera: string | null) => void;
}) {
  return (
    <>
      <label htmlFor={id}>{ROLE_LABELS[role] ?? role}</label>
      <select id={id} className="settings-input" value={value ?? ""} disabled={disabled} onChange={(e) => onChange(e.target.value === "" ? null : e.target.value)}>
        <option value="">Not assigned</option>
        {cameras.map((name) => (
          <option key={name} value={name} disabled={taken.has(name)}>
            {name}
            {offered.includes(name) ? "" : " (not in this session)"}
          </option>
        ))}
      </select>
    </>
  );
}

/** The name field and button that add a station. */
function AddStation({ canChange, onAdd }: { canChange: boolean; onAdd: (name: string) => Promise<boolean> }) {
  const [name, setName] = useState("");
  const ready = canChange && name.trim() !== "";
  /** Add the typed name and clear the field for the next one; a refused name stays to be corrected. */
  const submit = () => {
    if (!ready) return;
    onAdd(name.trim()).then((added) => {
      if (added) setName("");
    });
  };
  return (
    <div className="settings-actions">
      <input
        className="settings-input"
        aria-label="New station name"
        placeholder="New station name"
        value={name}
        maxLength={MAX_NAME_CHARS}
        disabled={!canChange}
        onChange={(e) => setName(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") submit();
        }}
      />
      <button type="button" className="settings-primary" disabled={!ready} onClick={submit}>
        Add station
      </button>
    </div>
  );
}
