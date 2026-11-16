// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * The Devices tab of the Settings dialog: the live checklist of what the station in use needs, and the
 * calibration tools.
 *
 * The backend judges every device continuously and holds recording while a required one fails. This tab shows
 * that verdict and never judges a device itself, and nothing here runs a test: the list is read again every
 * couple of seconds while the tab is shown. The dialog is over the page, so the cameras and the 3D view stay
 * live behind it while a device is being fixed.
 */
import { useState } from "react";
import { sentence } from "./sentence";
import { CAMERA_TYPE_LABELS, type OperatorSettings, type ToolInfo } from "./settingsApi";
import { Card, LoadState, type Loaded } from "./settingsParts";
import type { CheckGroup, CheckItem, CheckStatus, SetupStatus } from "./setupApi";
import { useSetup } from "./useSetup";

/** How often the checklist is read while the tab is shown: fast enough to watch a device come back. */
export const DEVICES_REFRESH_MS = 2_000;

/** The mark and the words for each outcome; the mark alone is not text. */
const OUTCOMES: Record<CheckStatus, { mark: string; word: string }> = {
  pass: { mark: "✓", word: "passed" },
  fail: { mark: "✗", word: "failed" },
  skip: { mark: "–", word: "not needed" },
};

/** The groups of the checklist, in the order they are listed. */
const GROUPS: ReadonlyArray<[CheckGroup, string]> = [
  ["tracking", "Tracking"],
  ["cameras", "Cameras"],
  ["system", "System"],
];

/** One device or check: its outcome, its name, and why when it did not simply pass. */
function CheckRow({ item }: { item: CheckItem }) {
  const outcome = OUTCOMES[item.status];
  return (
    <li className={`check-row check-row--${item.status}`}>
      <span className="check-mark" aria-hidden="true">
        {outcome.mark}
      </span>
      <span className="check-label">
        {item.label}
        <span className="visually-hidden"> {outcome.word}</span>
      </span>
      {item.detail !== null && <span className="check-detail">{sentence(item.detail)}</span>}
    </li>
  );
}

/** Whether station setup holds a take right now, in one sentence. */
function Verdict({ setup }: { setup: SetupStatus }) {
  // Without a station in use nothing is held, so the line is neither a pass nor a hold.
  const tone = setup.blocked_reason !== null ? " setup-verdict--held" : setup.enforced ? " setup-verdict--clear" : "";
  return (
    <p className={`setup-verdict${tone}`} role="status">
      {setup.blocked_reason !== null
        ? `Recording is held: ${setup.blocked_reason}.`
        : setup.enforced
          ? "Station setup allows recording."
          : "No station is in use, so these checks advise and do not hold recording."}
    </p>
  );
}

/** The checks by group, leaving out a group the station has nothing in. */
function Checklist({ checks }: { checks: readonly CheckItem[] }) {
  return (
    <Card title="Checklist" lead="Checked continuously: the tracking devices and cameras this station needs, their calibration files, and the room left on the recording disk.">
      {checks.length === 0 && <p className="settings-note">Nothing is being checked yet.</p>}
      {GROUPS.map(([group, title]) => {
        const items = checks.filter((item) => item.group === group);
        if (items.length === 0) return null;
        return (
          <section key={group} aria-label={title}>
            <h4 className="check-group">{title}</h4>
            <ul className="check-list">
              {items.map((item) => (
                <CheckRow key={item.id} item={item} />
              ))}
            </ul>
          </section>
        );
      })}
    </Card>
  );
}

/** What the tools card reads from the dialog's station setup. */
type ToolSetup = Loaded<{ tools: ToolInfo[]; operator: Pick<OperatorSettings, "camera_type"> }>;

/** The calibration tools this build knows about, and why an unavailable one cannot be opened. */
function CalibrationTools({ stationSetup }: { stationSetup: ToolSetup }) {
  if (stationSetup.kind !== "loaded") return <LoadState state={stationSetup} what="Calibration tools" />;
  const { tools, operator } = stationSetup.value;
  return (
    <Card title="Calibration tools" lead="Calibration and rectification. A tool opens only when this build can launch it; the others say why not.">
      <ul className="settings-list">
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

export interface DevicesTabProps {
  /** The flow as the page last read it, shown only until this tab's own, faster read first answers. */
  setup: SetupStatus | null;
  controlToken: string | null;
  loadSetup: (signal: AbortSignal, controlToken?: string | null) => Promise<SetupStatus>;
  /** The dialog's station setup, which lists the tools and the camera type they are judged against. */
  stationSetup: ToolSetup;
}

/** Show whether station setup holds recording, each device and check as it stands now, and the calibration tools. */
export function DevicesTab({ setup, controlToken, loadSetup, stationSetup }: DevicesTabProps) {
  const live = useSetup(null, { controlToken, intervalMs: DEVICES_REFRESH_MS, load: loadSetup });
  // Once the tab has answered, the page's older read must not stand in for a failed one: it would show a
  // device as it was up to ten seconds ago with no sign that the list had stopped following.
  const [answered, setAnswered] = useState(false);
  if (live !== null && !answered) setAnswered(true);
  const shown = live ?? (answered ? null : setup);
  return (
    <>
      {shown === null ? (
        <p className="settings-note tone-warning">The checks are unavailable: the backend did not answer.</p>
      ) : (
        <>
          <Verdict setup={shown} />
          <Checklist checks={shown.checks} />
        </>
      )}
      <CalibrationTools stationSetup={stationSetup} />
    </>
  );
}
