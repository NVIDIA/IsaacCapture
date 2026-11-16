// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * The start-of-day flow as the page needs it: loaded on mount, again whenever `refreshKey` changes, and on a
 * slow timer in between. The page bumps the key when a dialog that can change the flow closes and when the
 * recording snapshot's own readiness reason changes, so the rail follows without waiting for the timer. A failed
 * load leaves null: the page then shows no rail, which is what it did before the flow existed.
 */
import { useEffect, useReducer, useState } from "react";
import { fetchSetup, type SetupStatus } from "./setupApi";

/** How often the flow is read between explicit refreshes: free disk space and calibration files change quietly. */
export const SETUP_REFRESH_MS = 10_000;

/** Keep the flow current, discarding an answer a newer request has overtaken. */
export function useSetup(refreshKey: unknown, load: (signal: AbortSignal) => Promise<SetupStatus> = fetchSetup): SetupStatus | null {
  const [setup, setSetup] = useState<SetupStatus | null>(null);
  const [timer, bumpTimer] = useReducer((n: number) => n + 1, 0);
  useEffect(() => {
    const controller = new AbortController();
    load(controller.signal).then(
      (loaded) => {
        if (!controller.signal.aborted) setSetup(loaded);
      },
      () => {
        if (!controller.signal.aborted) setSetup(null);
      },
    );
    return () => controller.abort();
  }, [refreshKey, timer, load]);
  useEffect(() => {
    const id = setInterval(bumpTimer, SETUP_REFRESH_MS);
    return () => clearInterval(id);
  }, []);
  return setup;
}
