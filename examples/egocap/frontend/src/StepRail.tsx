// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * The start-of-day progress rail under the header: station, devices, calibration, record, wrap-up.
 *
 * Each step shows whether it is done, needs attention, is still to do or does not apply to this station, with a
 * short reason. The first step that is neither done nor skipped is the current one. A step opens the place where
 * it is dealt with; the page decides what that is. The rail only reports what the backend said: it never marks a
 * step done itself.
 */
import { sentence } from "./sentence";
import type { SetupStep, StepId, StepStatus } from "./setupApi";

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
  onOpen: (step: StepId) => void;
}

/** Show the flow's steps in order and let each open the place it is dealt with. */
export function StepRail({ steps, onOpen }: StepRailProps) {
  if (steps === null || steps.length === 0) return null;
  const current = currentStep(steps);
  return (
    <nav className="step-rail" aria-label="Start-of-day steps">
      <ol className="step-rail-steps">
        {steps.map((step, index) => {
          const isCurrent = step.id === current;
          const detail = step.detail === null ? null : sentence(step.detail);
          return (
            <li key={step.id} className="step-rail-item">
              <button
                type="button"
                className={`step step--${step.status}${isCurrent ? " step--current" : ""}`}
                aria-current={isCurrent ? "step" : undefined}
                aria-label={`Step ${index + 1}, ${step.label}: ${STATUS_WORDS[step.status]}${detail === null ? "" : `. ${detail}`}`}
                title={detail ?? undefined}
                onClick={() => onOpen(step.id)}
              >
                <span className="step-mark" aria-hidden="true">
                  {mark(step.status, index + 1)}
                </span>
                <span className="step-text">
                  <span className="step-label">{step.label}</span>
                  {detail !== null && <span className="step-detail">{detail}</span>}
                </span>
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
