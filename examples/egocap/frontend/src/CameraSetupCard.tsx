// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * The Camera setup card of the Settings dialog's Station tab, shown only while the site has no stations: the
 * camera type and the camera filling each role, saved with the operator settings.
 *
 * Once a station exists it holds the camera setup and its own row edits it, so this card goes away. The fields
 * are the station editor's own, so the two cannot drift apart.
 */
import { useId, useState } from "react";
import { Card, TokenHint, useAction } from "./settingsParts";
import type { OperatorSettings, OperatorSettingsChange } from "./settingsApi";
import { CameraFields, NO_CAMERAS_NOTE, cameraChangesOf, cameraDraftOf, hasWristCamera, selectableCameras, withWristsSwapped, type CameraDraft, type CameraSetup } from "./StationsCard";

export interface CameraSetupCardProps {
  /** The saved setup, with the camera types and roles the backend accepts. */
  operator: Pick<OperatorSettings, "camera_type" | "camera_roles" | "camera_types" | "roles">;
  /** The cameras the current session offers, by name. */
  cameraSources: string[];
  controlToken: string | null;
  /** Saves the changed fields; rejects with the backend's reason. */
  save: (changes: OperatorSettingsChange) => Promise<void>;
}

/** Edit the camera type and the camera per role of a site without stations, saving only what changed. */
export function CameraSetupCard({ operator, cameraSources, controlToken, save }: CameraSetupCardProps) {
  const fieldId = useId();
  // The setup the draft started from: what this card changed is judged against it, never against a setup
  // another browser saved meanwhile, which a save from here must not put back.
  const [original, setOriginal] = useState<CameraSetup>(operator);
  const [draft, setDraft] = useState<CameraDraft>(() => cameraDraftOf(operator));
  const [{ busy, failure }, run] = useAction();
  const changes = cameraChangesOf(original, operator.roles, draft);
  // An untouched draft follows the saved setup. The card can appear before the setup has been read again (the
  // last station was just deleted, and what it held is still what the dialog shows), and it must not go on
  // showing that station's cameras as the site's. A draft the operator has changed is theirs and is kept.
  const savedSetup = JSON.stringify([operator.camera_type, operator.roles.map((role) => operator.camera_roles[role] ?? null)]);
  const [followed, setFollowed] = useState(savedSetup);
  if (followed !== savedSetup) {
    setFollowed(savedSetup);
    if (Object.keys(changes).length === 0) {
      setOriginal(operator);
      setDraft(cameraDraftOf(operator));
    }
  }
  /** Send what changed; once saved, the draft is what later changes are judged against. */
  const saveChanges = () => run(() => save(changes).then(() => setOriginal({ camera_type: draft.cameraType, camera_roles: { ...draft.cameraRoles } })));
  const dirty = Object.keys(changes).length > 0;
  const canSave = controlToken !== null && dirty && !busy;
  return (
    <Card title="Camera setup" lead="The camera type and the camera filling each role while this site has no stations. Adding a station moves this setup onto the station, which holds it from then on.">
      <div className="settings-fields">
        <CameraFields
          idPrefix={fieldId}
          draft={draft}
          onDraft={setDraft}
          saved={operator.camera_roles}
          roles={operator.roles}
          cameraTypes={operator.camera_types}
          cameraSources={cameraSources}
          disabled={controlToken === null || busy}
        />
      </div>
      {selectableCameras(cameraSources, operator.camera_roles).length === 0 && <p className="settings-note">{NO_CAMERAS_NOTE}</p>}
      <div className="settings-actions">
        <button type="button" className="settings-primary" disabled={!canSave} onClick={saveChanges}>
          {busy ? "Saving…" : "Save camera setup"}
        </button>
        <button type="button" className="settings-quiet" disabled={controlToken === null || busy || !hasWristCamera(draft)} onClick={() => setDraft(withWristsSwapped)}>
          Swap wrists
        </button>
        {failure !== null && <span className="settings-note tone-warning">Not saved: {failure}</span>}
      </div>
      <TokenHint controlToken={controlToken} />
    </Card>
  );
}
