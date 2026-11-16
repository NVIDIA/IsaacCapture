// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** Building blocks shared by the Settings dialog's tabs and the cards that live in them. */
import { useState, type ReactNode } from "react";

/** Turn unknown request failures into a concise operator-facing explanation. */
export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** One titled block of a tab. */
export function Card({ title, lead, children }: { title: string; lead?: string; children: ReactNode }) {
  return (
    <section className="settings-card" aria-label={title}>
      <h3>{title}</h3>
      {lead !== undefined && <p className="settings-lead">{lead}</p>}
      {children}
    </section>
  );
}

/** Explain why station changes require the recording control token. */
export function TokenHint({ controlToken }: { controlToken: string | null }) {
  if (controlToken !== null) return null;
  return <p className="settings-note settings-note--hint">Enter the control token in the recording panel to change these.</p>;
}

/** The pending/failed state of one save-type action, shown next to its button. */
export function useAction(): [{ busy: boolean; failure: string | null }, (work: () => Promise<void>) => void] {
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  /** Keep request failures visible and prevent duplicate submissions. */
  const run = (work: () => Promise<void>) => {
    setBusy(true);
    setFailure(null);
    work().then(
      () => setBusy(false),
      (error: unknown) => {
        setBusy(false);
        setFailure(errorMessage(error));
      },
    );
  };
  return [{ busy, failure }, run];
}
