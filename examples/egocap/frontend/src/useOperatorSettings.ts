// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * The operator settings as the page itself needs them (today: which camera fills which role, for the tiles).
 * Loaded on mount and again whenever `refreshKey` changes -- the page passes a counter it bumps when the Settings
 * dialog closes, so a change saved there shows on the tiles at once. A failed load leaves null: the page then
 * falls back to its name-based layout, which is what it did before roles existed.
 */
import { useEffect, useState } from "react";
import { fetchOperatorSettings, type OperatorSettings } from "./settingsApi";

/** Refresh persisted camera roles while discarding requests from an earlier refresh. */
export function useOperatorSettings(refreshKey: number, load: (signal: AbortSignal) => Promise<OperatorSettings> = fetchOperatorSettings): OperatorSettings | null {
  const [settings, setSettings] = useState<OperatorSettings | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    load(controller.signal).then(
      (loaded) => {
        if (!controller.signal.aborted) setSettings(loaded);
      },
      () => {
        if (!controller.signal.aborted) setSettings(null);
      },
    );
    return () => controller.abort();
  }, [refreshKey, load]);
  return settings;
}

/** The role a camera fills, by its session name, or null when unassigned. */
export function roleOfCamera(roles: Record<string, string | null> | null | undefined, name: string): string | null {
  if (!roles) return null;
  for (const [role, camera] of Object.entries(roles)) if (camera === name) return role;
  return null;
}
