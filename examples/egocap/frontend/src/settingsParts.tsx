// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/** Building blocks shared by the dialogs: cards, load and save state, and modal focus. */
import { useEffect, useRef, useState, type ReactNode } from "react";

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

/** A request's progress, so a reader shows neither stale controls nor a blank. */
export type Loaded<T> = { kind: "loading" } | { kind: "error"; message: string } | { kind: "loaded"; value: T };

/** Explain an incomplete request without presenting stale controls. */
export function LoadState<T>({ state, what }: { state: Loaded<T>; what: string }) {
  if (state.kind === "loading") return <p className="settings-note">Loading…</p>;
  if (state.kind === "error") return <p className="settings-note tone-warning">{what} unavailable: {state.message}</p>;
  return null;
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

/** Whether a collapsed `<details>` hides the control. The browser renders nothing inside one except its own
 * summary, so focus sent to anything else there goes nowhere and Tab would stop moving. */
function hiddenByClosedDetails(element: HTMLElement): boolean {
  for (let details = element.closest("details"); details !== null; details = details.parentElement?.closest("details") ?? null) {
    const ownSummary = Array.from(details.children).find((child) => child.tagName === "SUMMARY") === element;
    if (!details.open && !ownSummary) return true;
  }
  return false;
}

/** Own keyboard focus for the dialog lifetime and return it to the invoking control. */
export function useModalFocus(open: boolean) {
  const dialogRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open || !dialogRef.current) return;
    const dialog = dialogRef.current;
    const previous = document.activeElement;
    /** Recompute controls because loading, tab changes, saves and disclosures can change their availability. */
    const controls = () => Array.from(dialog.querySelectorAll<HTMLElement>(
      'button, input, select, textarea, summary, a[href], [tabindex]',
    )).filter((element) => element.tabIndex >= 0 && !element.matches(":disabled") && !element.closest("[hidden], [inert]") && !hiddenByClosedDetails(element));
    /** Programmatic focus and pointer interaction must also respect the modal boundary. */
    const containFocus = (event: FocusEvent) => {
      if (!dialog.contains(event.target as Node)) dialog.focus();
    };
    /** Cycle only the current dialog controls, including wraparound in either direction. */
    const onTab = (event: KeyboardEvent) => {
      if (event.key !== "Tab" || event.altKey || event.ctrlKey || event.metaKey) return;
      const available = controls();
      const current = available.indexOf(document.activeElement as HTMLElement);
      const next = event.shiftKey
        ? (current <= 0 ? available.length - 1 : current - 1)
        : (current + 1) % available.length;
      event.preventDefault();
      (available[next] ?? dialog).focus();
    };
    document.addEventListener("focusin", containFocus);
    document.addEventListener("keydown", onTab);
    dialog.focus();
    return () => {
      // Release containment before restoring focus; rerenders must not recapture the invoking control.
      document.removeEventListener("focusin", containFocus);
      document.removeEventListener("keydown", onTab);
      if (previous instanceof HTMLElement && previous.isConnected) previous.focus();
    };
  }, [open]);
  return dialogRef;
}

/** Close a dialog on Escape from any focused control, as every dialog does. */
export function useEscapeToClose(open: boolean, onClose: () => void) {
  useEffect(() => {
    if (!open) return;
    /** Escape releases the dialog wherever focus is. */
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);
}
