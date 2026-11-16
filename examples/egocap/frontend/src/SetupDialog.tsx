// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * The start-of-day checks dialog: the device connection test, and the calibration and system checks.
 *
 * It opens over the page, so the cameras and the 3D view stay live behind it while devices are being fixed.
 * The test is run on request and its result is the backend's: this dialog shows it and never judges a device
 * itself. Every item is passed, failed with the reason, or marked as not applying to this station.
 */
import { useEffect, useState } from "react";
import { sentence } from "./sentence";
import { Card, errorMessage, useEscapeToClose, useModalFocus } from "./settingsParts";
import { runConnectionTest, type CheckItem, type CheckStatus, type SetupStatus } from "./setupApi";

/** The mark and the words for each outcome; the mark alone is not text. */
const OUTCOMES: Record<CheckStatus, { mark: string; word: string }> = {
  pass: { mark: "✓", word: "passed" },
  fail: { mark: "✗", word: "failed" },
  skip: { mark: "–", word: "not needed" },
};

/** The groups of the connection test, in the order they are listed. */
const GROUPS: ReadonlyArray<[string, string]> = [
  ["tracking", "Tracking"],
  ["cameras", "Cameras"],
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

/** A local clock time for when the test ran. */
function clockTime(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export interface SetupDialogProps {
  open: boolean;
  onClose: () => void;
  /** The flow as the page last read it; null while it could not be read. */
  setup: SetupStatus | null;
  /** The run's control token once entered; without it the checks are shown but the test cannot be run. */
  controlToken: string | null;
  /** Called after the test ran, so the page reads the flow again. */
  onChanged: () => void;
  runTest?: (controlToken: string) => Promise<SetupStatus>;
}

/** Show the connection test and the system checks, and run the test on request. */
export function SetupDialog({ open, onClose, setup, controlToken, onChanged, runTest = runConnectionTest }: SetupDialogProps) {
  const dialogRef = useModalFocus(open);
  useEscapeToClose(open, onClose);
  // The answer to a test just run, shown at once; a newer read from the page replaces it.
  const [answer, setAnswer] = useState<SetupStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  useEffect(() => setAnswer(null), [setup]);

  if (!open) return null;
  const shown = answer ?? setup;
  /** Run the test and show the backend's result. */
  const run = () => {
    if (controlToken === null || busy) return;
    setBusy(true);
    setFailure(null);
    runTest(controlToken).then(
      (result) => {
        setAnswer(result);
        setBusy(false);
        onChanged();
      },
      (error: unknown) => {
        setFailure(errorMessage(error));
        setBusy(false);
      },
    );
  };
  const test = shown?.connection_test ?? null;
  const failed = test === null ? 0 : test.items.filter((item) => item.status === "fail").length;

  return (
    <>
      <div className="settings-backdrop" onClick={onClose} aria-hidden="true" />
      <div ref={dialogRef} tabIndex={-1} className="settings-modal setup-modal" role="dialog" aria-modal="true" aria-label="Start-of-day checks">
        <div className="settings-content">
          <header className="settings-content-header">
            <h3>Start-of-day checks</h3>
            <button type="button" className="settings-quiet" onClick={onClose}>
              Close
            </button>
          </header>
          <div className="settings-content-body">
            {shown === null ? (
              <p className="settings-note tone-warning">The checks are unavailable: the backend did not answer.</p>
            ) : (
              <>
                <p className={`setup-verdict ${shown.blocked_reason === null ? "setup-verdict--clear" : "setup-verdict--held"}`} role="status">
                  {shown.blocked_reason !== null
                    ? `Recording is held: ${shown.blocked_reason}.`
                    : shown.enforced
                      ? "Station setup allows recording."
                      : "No station is in use, so these checks advise and do not hold recording."}
                </p>
                <Card title="Device connection test" lead="Checks that every device this station needs is connected and delivering data right now. Run it at the start of the day and after changing a device.">
                  <div className="settings-actions">
                    <button type="button" className="settings-primary" disabled={controlToken === null || busy} onClick={run}>
                      {busy ? "Testing…" : test === null ? "Run test" : "Run again"}
                    </button>
                    {test !== null && (
                      <span className={`settings-note ${failed === 0 ? "settings-ready-item--ok" : "tone-warning"}`}>
                        {failed === 0 ? `✓ Passed at ${clockTime(test.ran_at)} for ${test.station}` : `${failed} failed at ${clockTime(test.ran_at)} for ${test.station}`}
                      </span>
                    )}
                    {failure !== null && <span className="settings-note tone-warning">Not run: {failure}</span>}
                  </div>
                  {controlToken === null && <p className="settings-note settings-note--hint">Enter the control token in the recording panel to run the test.</p>}
                  {test === null ? (
                    <p className="settings-note">Not run yet for this station in this capture session.</p>
                  ) : (
                    GROUPS.map(([group, title]) => {
                      const items = test.items.filter((item) => item.group === group);
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
                    })
                  )}
                </Card>
                <Card title="Calibration and system" lead="Checked continuously: calibration files for the cameras in use, the glove software, room on the recording disk, and the camera producer.">
                  <ul className="check-list">
                    {shown.system.map((item) => (
                      <CheckRow key={item.id} item={item} />
                    ))}
                  </ul>
                </Card>
              </>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
