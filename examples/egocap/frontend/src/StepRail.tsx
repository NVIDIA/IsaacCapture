// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * The start-of-day progress rail under the header: station, devices, record, wrap-up.
 *
 * Each step shows whether it is done, needs attention, is still to do or does not apply to this station, with a
 * short reason. The first step that is neither done nor skipped is the current one. Station, Devices and Wrap up
 * open the place where they are dealt with; the page decides what that is. Record opens nothing, because
 * recording is the page itself. The rail only reports what the backend said: it never marks a step done itself.
 */
import { sentence } from "./sentence";
import type { SetupStep, StepId, StepStatus } from "./setupApi";

/** The steps that open something. */
export type OpenableStep = Exclude<StepId, "record">;

/** The mark inside a step's circle: its number until something more specific can be said. */
function mark(status: StepStatus, position: number): string {
  if (status === "done") return "✓";
  if (status === "attention") return "!";
  if (status === "skipped") return "–";
  return String(position);
}

/** How a step's state is announced, since the mark alone is not text. */
const STATUS_WORDS: Record<StepStatus, string> = {
  done: "done",
  todo: "to do",
  attention: "needs attention",
  skipped: "not needed",
};

/** The step the operator should deal with next: the first that is neither done nor skipped; null when none is left. */
export function currentStep(steps: readonly SetupStep[]): StepId | null {
  return steps.find((step) => step.status === "todo" || step.status === "attention")?.id ?? null;
}

export interface StepRailProps {
  /** The flow's steps in order, or null while they could not be read (the rail then renders nothing). */
  steps: readonly SetupStep[] | null;
  /** Open the place a step is dealt with. */
  onOpen: (step: OpenableStep) => void;
}

/** Show the flow's steps in order and let each one that is dealt with elsewhere open that place. */
export function StepRail({ steps, onOpen }: StepRailProps) {
  if (steps === null || steps.length === 0) return null;
  const current = currentStep(steps);
  return (
    <nav className="step-rail" aria-label="Start-of-day steps">
      <ol className="step-rail-steps">
        {steps.map((step, index) => {
          const isCurrent = step.id === current;
          const detail = step.detail === null ? null : sentence(step.detail);
          const look = `step step--${step.status}${isCurrent ? " step--current" : ""}`;
          const stepMark = (
            <span className="step-mark" aria-hidden="true">
              {mark(step.status, index + 1)}
            </span>
          );
          const { id } = step;
          return (
            <li key={id} className="step-rail-item">
              {id === "record" ? (
                // Not a control: recording is the page itself, so this step has nothing to open.
                <span className={`${look} step--static`} aria-current={isCurrent ? "step" : undefined} title={detail ?? undefined}>
                  {stepMark}
                  <span className="step-text">
                    <span className="step-label">
                      {step.label}
                      <span className="visually-hidden">: {STATUS_WORDS[step.status]}</span>
                    </span>
                    {detail !== null && <span className="step-detail">{detail}</span>}
                  </span>
                </span>
              ) : (
                <button
                  type="button"
                  className={look}
                  aria-current={isCurrent ? "step" : undefined}
                  aria-label={`Step ${index + 1}, ${step.label}: ${STATUS_WORDS[step.status]}${detail === null ? "" : `. ${detail}`}`}
                  title={detail ?? undefined}
                  onClick={() => onOpen(id)}
                >
                  {stepMark}
                  <span className="step-text">
                    <span className="step-label">{step.label}</span>
                    {detail !== null && <span className="step-detail">{detail}</span>}
                  </span>
                </button>
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
