// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * The start-of-day flow as a reader needs it: loaded on mount, again whenever `refreshKey` or the control token
 * changes, and on a timer in between. The page bumps the key when a dialog that can change the flow closes and
 * when the recording snapshot's own readiness reason changes, so the rail follows without waiting for the timer.
 * The timer never interrupts a read: the next one is due an interval after the last one answered, so a backend
 * slower than the interval is still heard. A failed load leaves null rather than a flow that may no longer hold,
 * and is read again sooner than the timer would, so one refused read does not leave the page without a flow for long.
 */
import { useEffect, useState } from "react";
import { fetchSetup, type SetupStatus } from "./setupApi";

/** How often the page reads the flow between explicit refreshes: free disk space and calibration files change quietly. */
export const SETUP_REFRESH_MS = 10_000;

/** How soon a failed read is tried again: the backend refuses a read while the station in use is changing. */
export const SETUP_RETRY_MS = 2_000;

export interface SetupOptions {
  /** Sent with each read once the page has it, so the reply carries the headset address. */
  controlToken?: string | null;
  /** How long after one read answers the next is sent, between explicit refreshes. */
  intervalMs?: number;
  /** What the flow is about, such as the station in use: an answer read under another scope is not shown. */
  scope?: string;
  /** Reads the flow; the page's own reader unless a test or a dialog supplies one. */
  load?: (signal: AbortSignal, controlToken?: string | null) => Promise<SetupStatus>;
}

/** An answer, with what it was read for. */
interface Answer {
  setup: SetupStatus | null;
  scope: string;
  controlToken: string | null;
}

/** Keep the flow current, discarding an answer a newer request has overtaken. */
export function useSetup(refreshKey: unknown, { controlToken = null, intervalMs = SETUP_REFRESH_MS, scope = "", load = fetchSetup }: SetupOptions = {}): SetupStatus | null {
  const [answer, setAnswer] = useState<Answer>({ setup: null, scope, controlToken });
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    /** Read the flow once and, when it answers, set the timer for the next read. */
    const read = () => {
      /** Take an answer unless this run of reads was abandoned, and only then wait for the next one. */
      const settled = (next: SetupStatus | null) => {
        if (controller.signal.aborted) return;
        setAnswer({ setup: next, scope, controlToken });
        timer = setTimeout(read, next === null ? Math.min(intervalMs, SETUP_RETRY_MS) : intervalMs);
      };
      load(controller.signal, controlToken).then(settled, () => settled(null));
    };
    read();
    // Only a changed key, token, scope or loader, or leaving the page, abandons a read; the timer never does.
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [refreshKey, load, controlToken, intervalMs, scope]);
  // Another station's flow is not this one's, so it is not shown while the new one is read.
  if (answer.setup === null || answer.scope !== scope) return null;
  // The headset address was given for a credential; without that credential it is no longer shown.
  return answer.controlToken === controlToken ? answer.setup : { ...answer.setup, headset: null };
}
