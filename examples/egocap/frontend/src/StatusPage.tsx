// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * A read-only page for CaptureServer's own status, apart from the recording page. It reads /api/server-status, the
 * backend's relay of CaptureServer's status query, so what it shows is what the server reports and it works with no
 * session and no recording state. Cameras are the server's configuration merged with what a session has observed:
 * configured is not detected, and a camera with no observation says so. It sends no commands; running the check
 * again stays on the recording page, which has the operator and control token.
 */

import { useEffect, useState } from "react";

import {
  CaptureMode,
  DeviceState,
  ProviderState,
  parseServerStatus,
  type ServerCamera,
  type ServerStatus,
} from "./captureStatus";
import { NetworkCheckPanel } from "./NetworkCheckPanel";

const POLL_MS = 2000;
/** What the page last read counts as current only while reads keep succeeding. */
const STALE_AFTER_MS = POLL_MS * 3;
/** A read that has not answered by now is abandoned and counts as a failed read, so a hung request cannot keep the
 * last answer looking current. */
const REQUEST_TIMEOUT_MS = 5000;

function nameOf(table: Record<string, number>, value: number): string {
  const found = Object.entries(table).find(([, v]) => v === value);
  return found === undefined ? `unknown (${value})` : found[0].toLowerCase().replace(/_/g, " ");
}

type Row = { key: string; label: string; state: string; error: string };

function cameraRow(camera: ServerCamera): Row {
  const source =
    camera.transport === "livekit"
      ? `LiveKit room ${camera.room}, track ${camera.track}`
      : camera.device_id !== null
        ? `CaptureServer camera ${camera.device_id}`
        : "CaptureServer camera";
  const size = camera.width !== null && camera.height !== null ? `, ${camera.width}x${camera.height}` : "";
  const rate = camera.fps !== null ? ` @ ${camera.fps} fps` : "";
  // A configured camera is not a detected one: only a session's observation gives it a state, and a LiveKit camera
  // is never opened by the server, so its delivery is the network check's to report.
  const state =
    camera.state !== null
      ? nameOf(DeviceState, camera.state)
      : camera.transport === "livekit"
        ? "offered (delivery: see the network check)"
        : "configured, not observed";
  return {
    key: camera.name,
    label: `${camera.name} (${source}${size}${rate}${camera.selected ? "; in session" : ""})`,
    state,
    error: camera.error ?? "",
  };
}

function Rows({ title, rows, none }: { title: string; rows: Row[]; none: string }) {
  return (
    <section className="status-group" aria-label={title}>
      <h2>{title}</h2>
      {rows.length === 0 ? (
        <p className="status-none">{none}</p>
      ) : (
        <ul>
          {rows.map((row) => (
            <li key={row.key}>
              <span>{row.label}</span> <strong>{row.state}</strong>
              {row.error !== "" && <span className="status-error"> ({row.error})</span>}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

export function StatusPage() {
  const [server, setServer] = useState<ServerStatus | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [lastOkAt, setLastOkAt] = useState<number | null>(null);
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const controller = new AbortController();
    let timer: number | undefined;
    // The page's age of its last read advances on its own clock, not only when a read finishes.
    const clock = window.setInterval(() => setNow(Date.now()), 1000);
    // One read at a time: the next is scheduled only when the last has finished, so a slow server cannot stack
    // requests, and leaving the page aborts whatever is in flight.
    const poll = async () => {
      const request = new AbortController();
      const abandon = window.setTimeout(() => request.abort(), REQUEST_TIMEOUT_MS);
      const stop = () => request.abort();
      controller.signal.addEventListener("abort", stop);
      try {
        const reply = await fetch("/api/server-status", { cache: "no-store", signal: request.signal });
        if (!reply.ok) throw new Error(`HTTP ${reply.status}`);
        const parsed = parseServerStatus(await reply.json());
        setServer(parsed);
        setProblem(null);
        setLastOkAt(Date.now());
      } catch (error) {
        if (controller.signal.aborted) return;
        setProblem(request.signal.aborted ? "the status read timed out" : error instanceof Error ? error.message : "the status could not be read");
      } finally {
        window.clearTimeout(abandon);
        controller.signal.removeEventListener("abort", stop);
      }
      if (controller.signal.aborted) return;
      setNow(Date.now());
      timer = window.setTimeout(() => void poll(), POLL_MS);
    };
    void poll();
    return () => {
      controller.abort();
      window.clearTimeout(timer);
      window.clearInterval(clock);
    };
  }, []);

  const live = lastOkAt !== null && now - lastOkAt <= STALE_AFTER_MS;
  const status = server?.status ?? null;
  return (
    <main className="status-page">
      <h1>CaptureServer status</h1>
      <p>
        <a href="/">Recording page</a>
        {lastOkAt !== null && <span> — read {new Date(lastOkAt).toLocaleTimeString()}</span>}
        {!live && (
          <span className="status-error"> — {problem ?? "waiting for the first status"}; showing the last known state</span>
        )}
      </p>
      {server !== null && (
        <section aria-label="Server and session">
          <p>
            {server.mode === CaptureMode.MOCK ? "mock" : "live"} mode;{" "}
            {server.session_id === null ? "no session running" : `session ${server.session_id}`}
          </p>
        </section>
      )}
      <Rows
        title="Cameras configured on CaptureServer"
        rows={(server?.cameras ?? []).map(cameraRow)}
        none={server === null ? "not read yet" : "no cameras configured"}
      />
      {status !== null ? (
        <>
          <Rows
            title="Providers"
            rows={status.providers.map((x) => ({ key: x.id, label: x.name, state: nameOf(ProviderState, x.status), error: x.error }))}
            none="none reported"
          />
          <Rows
            title="Devices"
            rows={status.devices.map((x) => ({ key: x.id, label: x.description || x.id, state: nameOf(DeviceState, x.status), error: x.error }))}
            none="none reported"
          />
        </>
      ) : (
        server !== null && <p className="status-none">Providers and devices are reported only while a session runs.</p>
      )}
      <NetworkCheckPanel
        check={server?.network_check ?? null}
        live={live}
        runBlockedReason="Run the check again from the recording page"
      />
      {server !== null && server.network_check === null && (
        <p className="status-none">Network check: not configured on this server</p>
      )}
    </main>
  );
}
