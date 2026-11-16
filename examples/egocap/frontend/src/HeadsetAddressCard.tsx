// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * The Headset address card of the Settings dialog: the capture host address the operator enters in the headset.
 *
 * The backend offers this host's addresses, wireless first. Once the operator has entered one in the headset
 * they mark it, and it is saved with the station in use; when the host's address later changes, the card says
 * the headset needs the new one instead of leaving a silent connection failure.
 */
import { useCallback, useEffect, useState } from "react";
import { Card, errorMessage, useAction } from "./settingsParts";
import type { HeadsetAddress, HostAddress, StationChange, StationListing, StationRecord } from "./stationsApi";

/** The calls the card makes; the dialog passes the real ones and tests pass fakes. */
export interface HeadsetAddressBackend {
  loadHeadsetAddress: (controlToken: string, signal: AbortSignal) => Promise<HeadsetAddress>;
  updateStation: (controlToken: string, id: string, changes: StationChange) => Promise<StationListing>;
}

/** What kind of link an address is on, in the operator's words. */
const KIND_LABELS: Record<string, string> = {
  wireless: "wireless",
  wired: "wired",
  other: "other network",
  capture_server: "capture host",
};

/** An address with where it comes from, for the chooser. */
function describe(candidate: HostAddress): string {
  const kind = KIND_LABELS[candidate.kind] ?? candidate.kind;
  return candidate.interface === null ? `${candidate.address} (${kind})` : `${candidate.address} (${candidate.interface}, ${kind})`;
}

type Loaded = { kind: "loading" } | { kind: "error"; message: string } | { kind: "loaded"; value: HeadsetAddress };

export interface HeadsetAddressCardProps {
  controlToken: string | null;
  /** The station in use, which remembers the address; null when no station is in use. */
  station: StationRecord | null;
  backend: HeadsetAddressBackend;
  onListing: (listing: StationListing) => void;
}

/** Show the address to enter in the headset and remember, per station, the one that was entered. */
export function HeadsetAddressCard({ controlToken, station, backend, onListing }: HeadsetAddressCardProps) {
  const [state, setState] = useState<Loaded>({ kind: "loading" });
  const [chosen, setChosen] = useState<string | null>(null);
  // Bumped by "Detect again": the host's addresses change when a network is joined or a cable is plugged in.
  const [generation, setGeneration] = useState(0);
  const [{ busy, failure }, run] = useAction();
  const { loadHeadsetAddress } = backend;
  const saved = station?.headset_address ?? null;

  useEffect(() => {
    if (controlToken === null) return;
    const controller = new AbortController();
    // A refresh keeps showing the last answer instead of flashing back to "Loading".
    setState((current) => (current.kind === "loaded" ? current : { kind: "loading" }));
    loadHeadsetAddress(controlToken, controller.signal).then(
      (value) => {
        setState({ kind: "loaded", value });
        // Keep the operator's choice while it is still offered; otherwise fall back to the best candidate.
        setChosen((current) => (value.candidates.some((c) => c.address === current) ? current : (value.candidates[0]?.address ?? null)));
      },
      (error: unknown) => {
        if (!controller.signal.aborted) setState({ kind: "error", message: errorMessage(error) });
      },
    );
    return () => controller.abort();
    // The saved address is part of the answer, so a change to it reloads.
  }, [controlToken, loadHeadsetAddress, generation, saved]);

  const detectAgain = useCallback(() => setGeneration((n) => n + 1), []);
  const lead = "The headset connects to the capture host over the network. Enter this address in the headset's tracking app, then reconnect it.";

  if (controlToken === null) {
    return (
      <Card title="Headset address" lead={lead}>
        <p className="settings-note settings-note--hint">Enter the control token in the recording panel to see this host's addresses.</p>
      </Card>
    );
  }
  if (state.kind !== "loaded") {
    return (
      <Card title="Headset address" lead={lead}>
        {state.kind === "loading" ? <p className="settings-note">Loading…</p> : <p className="settings-note tone-warning">Addresses unavailable: {state.message}</p>}
      </Card>
    );
  }
  const { candidates, changed } = state.value;
  const stationName = station?.name ?? null;
  return (
    <Card title="Headset address" lead={lead}>
      {candidates.length === 0 ? (
        <p className="settings-note tone-warning">No network address was found for this host. Check its network connection, then detect again.</p>
      ) : (
        <>
          <p className="headset-address" aria-label="Address to enter in the headset">
            {chosen}
          </p>
          {candidates.length > 1 && (
            <select className="settings-input" aria-label="Network to use" value={chosen ?? ""} disabled={busy} onChange={(e) => setChosen(e.target.value)}>
              {candidates.map((candidate) => (
                <option key={candidate.address} value={candidate.address}>
                  {describe(candidate)}
                </option>
              ))}
            </select>
          )}
          {candidates.length === 1 && <p className="settings-note">{describe(candidates[0])}</p>}
        </>
      )}
      {changed && saved !== null && (
        <p className="settings-note tone-warning">
          The address saved for {stationName} ({saved}) is no longer one of this host's. Enter the address above in the headset, then mark it as entered.
        </p>
      )}
      {!changed && saved !== null && chosen === saved && <p className="settings-note settings-ready-item--ok">✓ Entered in the headset for {stationName}.</p>}
      {stationName === null && <p className="settings-note">Put a station in use to remember which address its headset was given.</p>}
      <div className="settings-actions">
        <button
          type="button"
          className="settings-primary"
          disabled={station === null || chosen === null || chosen === saved || busy}
          onClick={() => station !== null && chosen !== null && run(() => backend.updateStation(controlToken, station.id, { headset_address: chosen }).then(onListing))}
        >
          {busy ? "Saving…" : "Mark as entered in the headset"}
        </button>
        <button type="button" className="settings-quiet" disabled={busy} onClick={detectAgain}>
          Detect again
        </button>
        {failure !== null && <span className="settings-note tone-warning">Not saved: {failure}</span>}
      </div>
    </Card>
  );
}
