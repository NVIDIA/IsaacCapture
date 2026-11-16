// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * The Stations card of the Settings dialog: the stations this site records at, and the one in use.
 *
 * The station in use names every recording and its activity log, and its cameras are the ones the cards below
 * edit. It cannot change while a take is being recorded; the backend says so in `locked_reason` and refuses the
 * change either way. A station that is not in use can always be added, renamed or deleted.
 */
import { useState } from "react";
import { Card, TokenHint, useAction } from "./settingsParts";
import { ROLE_LABELS } from "./settingsApi";
import { sentence } from "./sentence";
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

export interface StationsCardProps {
  listing: StationListing;
  controlToken: string | null;
  backend: StationsBackend;
  /** The camera setup saved before any station existed; the first station added takes it over. */
  inherited: StationChange;
  /** Called with the backend's listing after every successful change. */
  onListing: (listing: StationListing) => void;
}

/** List the stations, choose the one in use, and add, rename or delete them. */
export function StationsCard({ listing, controlToken, backend, inherited, onListing }: StationsCardProps) {
  const [{ busy, failure }, run] = useAction();
  const locked = listing.locked_reason !== null;
  const canChange = controlToken !== null && !busy;
  /** Run one registry change with the token and hand its listing to the dialog. */
  const change = (work: (token: string) => Promise<StationListing>) => {
    if (controlToken !== null) run(() => work(controlToken).then(onListing));
  };
  /** Add a station; the first one takes over the camera setup already saved and is put in use. */
  const add = (name: string) =>
    change(async (token) => {
      const first = listing.stations.length === 0;
      const added = await backend.addStation(token, first ? { ...inherited, name } : { name });
      // During a take the station in use cannot change, so the new station is left for the operator to select.
      if (!first || added.locked_reason !== null) return added;
      return backend.selectStation(token, added.stations[added.stations.length - 1].id);
    });

  return (
    <Card title="Stations" lead="The stations this site records at. The one in use names every recording and its activity log, and its cameras are set below.">
      {listing.warning !== undefined && <p className="settings-note tone-warning">{listing.warning}</p>}
      {listing.stations.length === 0 ? (
        <p className="settings-note">
          No stations yet. Recordings carry the deployment's label, “{listing.label}”. Add a station to choose between stations here.
        </p>
      ) : (
        <ul className="settings-list settings-list--stations" aria-label="Stations">
          {listing.stations.map((station) => (
            <StationRow
              key={station.id}
              station={station}
              roles={listing.roles}
              inUse={listing.selected === station.id}
              canChange={canChange}
              locked={locked}
              onSelect={() => change((token) => backend.selectStation(token, station.id))}
              onRename={(name) => change((token) => backend.updateStation(token, station.id, { name }))}
              onDelete={() => change((token) => backend.deleteStation(token, station.id))}
            />
          ))}
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
  roles,
  inUse,
  canChange,
  locked,
  onSelect,
  onRename,
  onDelete,
}: {
  station: StationRecord;
  roles: readonly string[];
  inUse: boolean;
  canChange: boolean;
  /** A take is being recorded: the station in use can be neither replaced nor edited. */
  locked: boolean;
  onSelect: () => void;
  onRename: (name: string) => void;
  onDelete: () => void;
}) {
  const [mode, setMode] = useState<"view" | "rename" | "delete">("view");
  const [draft, setDraft] = useState(station.name);
  // The station in use is frozen during a take; every other station stays editable.
  const editable = canChange && !(inUse && locked);
  const renamed = draft.trim() !== "" && draft.trim() !== station.name;
  /** Save a changed name and leave the editor; an unchanged or empty one just closes it. */
  const commitRename = () => {
    if (renamed) onRename(draft.trim());
    setMode("view");
  };
  return (
    <li className={inUse ? "settings-list-item--on" : undefined}>
      <div className="settings-list-main settings-list-main--stacked">
        <label className="settings-station-head">
          <input type="radio" name="station-in-use" checked={inUse} disabled={!canChange || locked} onChange={onSelect} aria-label={`Use ${station.name}`} />
          {mode === "rename" ? (
            <input
              className="settings-input"
              aria-label={`New name for ${station.name}`}
              value={draft}
              maxLength={MAX_NAME_CHARS}
              autoFocus
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                // Enter saves; Escape belongs to the dialog, which closes on it, so it is stopped here.
                if (e.key === "Enter") commitRename();
                if (e.key === "Escape") {
                  e.stopPropagation();
                  setMode("view");
                }
              }}
            />
          ) : (
            <span className="settings-list-name">{station.name}</span>
          )}
          {inUse && <span className="settings-badge settings-badge--on">in use</span>}
        </label>
        <span className="settings-note">{roleSummary(station, roles)}</span>
        {station.problems.length > 0 && <span className="settings-note tone-warning">Cannot record yet: {station.problems.join("; ")}.</span>}
      </div>
      {mode === "view" && (
        <>
          <button
            type="button"
            className="settings-quiet"
            disabled={!editable}
            onClick={() => {
              setDraft(station.name);
              setMode("rename");
            }}
          >
            Rename
          </button>
          <button type="button" className="settings-quiet" disabled={!editable} onClick={() => setMode("delete")}>
            Delete
          </button>
        </>
      )}
      {mode === "rename" && (
        <>
          <button type="button" className="settings-primary" disabled={!editable || !renamed} onClick={commitRename}>
            Save name
          </button>
          <button type="button" className="settings-quiet" onClick={() => setMode("view")}>
            Cancel
          </button>
        </>
      )}
      {mode === "delete" && (
        <>
          <button
            type="button"
            className="settings-danger"
            disabled={!editable}
            onClick={() => {
              setMode("view");
              onDelete();
            }}
          >
            Delete {station.name}
          </button>
          <button type="button" className="settings-quiet" onClick={() => setMode("view")}>
            Keep
          </button>
        </>
      )}
    </li>
  );
}

/** The name field and button that add a station. */
function AddStation({ canChange, onAdd }: { canChange: boolean; onAdd: (name: string) => void }) {
  const [name, setName] = useState("");
  const ready = canChange && name.trim() !== "";
  /** Add the typed name and clear the field for the next one. */
  const submit = () => {
    if (!ready) return;
    onAdd(name.trim());
    setName("");
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
